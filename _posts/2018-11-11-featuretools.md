---
layout: post
title: "Automated Feature Engineering with Featuretools"
date: 2018-11-11
description: "Running deep feature synthesis for automated feature engineering, using the Featuretools package for Python."
tags: [feature engineering, featuretools, python]
published: true
---

[Featuretools](https://www.featuretools.com/) is a fantastic python package for automated feature engineering.  It can automatically generate features from secondary datasets which can then be used in machine learning models.  In this post we'll see how automated feature engineering with Featuretools works, and how to run it on complex multi-table datsets!

**Outline**
- [Automated Feature Engineering](#automated-feature-engineering)
- [Deep Feature Synthesis](#deep-feature-synthesis)
- [Using Featuretools](#using-featuretools)
- [Predictions from Generated Features](#predictions-from-generated-features)
- [Running out of Memory](#running-out-of-memory)


<a class="anchor" id="automated-feature-engineering"></a>
## Automated Feature Engineering

What do I mean by "automated feature engineering" and how is it useful?  When building predictive models, we need to have training examples which have some set of features.  For most machine learning algorithms (though of course not all of them), this training set needs to take the form of a table or matrix, where each row corresponds to a single training example or observation, and each column corresponds to a different feature.  For example, suppose we're trying to predict how likely loan applicants are to successfully repay their loans.  In this case, our data table will have a row for each applicant, and a column for each "feature" of the applicants, such as their income, their current level of credit, their age, etc.

Unfortunately, in most applications the data isn't quite as simple as just one table.  We'll likely have additional data stored in other tables!  To continue with the loan repayment prediction example, we could have a separate table which stores the monthly balances of applicants on their other loans, and another separate table with the credit card accounts for each applicant, and yet another table with the credit card activity for each of those accounts, and so on. 

![Data table tree](/assets/img/featuretools/DataframeTree.svg)

In order to build a predictive model, we need to "engineer" features from data in those secondary tables.  These engineered features can then be added to our main data table, which we can then use to train the predictive model.  For example, we could compute the number of credit card accounts for each applicant, and add that as a feature to our primary data table; we could compute the balance across each applicant's credit cards, and add that to the primary data table; we could also compute the balance to available credit ratio and add that as a feature; etc.

With complicated (read: real-life) datasets, the number of features that we could engineer becomes very large, and the task of manually engineering all these features becomes extremely time-intensive.  The [Featuretoools](https://www.featuretools.com/) package automates this process by automatically generating features for our primary data table from information in secondary data sources.


<a class="anchor" id="deep-feature-synthesis"></a>
## Deep Feature Synthesis

Featuretools uses a process they call "deep feature synthesis" to generate top-level features from secondary datasets.  For each secondary table, the child table is merged with the parent table on the column which joins them (usually an ID or something).  Raw features can be transformed according to *transform primitives* like `month` (which transforms a datetime into a month, or `cum_sum` which transforms a value to the cumulative sum of elements in that aggregation bin.  Then, features are built from "aggregation primitives", such as mean, sum, max, etc, which aggregate potentially multiple entries in the child table to a single feature in the parent dataset.  This feature generation process is repeated recursively until we have a single table (the primary table) with features generated from child and sub-child (etc) tables.


<a class="anchor" id="using-featuretools"></a>
## Using Featuretools

To show how Featuretools works, we'll be using it on the [Home Credit Group Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) dataset.  This dataset contains information about individuals applying for loans with [Home Credit Group](http://www.homecredit.net/), a consumer lender specializing in loans to individuals with little credit history.  Home Credit Group hopes to be able to predict how likely an applicant is to default on their loan, in order to decide whether a given loan plan is good for a specific applicant (or whether to suggest a different payment schedule).  

The dataset contains multiple tables which relate to one another in some way.  Below is a diagram which shows each data table, the information it contains, and how each table is related to each other table.

![Table relationships](/assets/img/featuretools/home_credit.png)

The primary tables (`application_train.csv` and `application_test.csv`) have information on each of the loan applications, where each row corresponds to a single application.  The train table has information about whether that applicant ended up defaulting on their loan, while the test table does not (because those are the applications we'll be testing our predictive model on).  The other tables contain information about other loans (either at other institutions, in the `bureau.csv` and `bureau_balance.csv` tables, or previous loans with Home Credit, in `previous_applications.csv`, `POS_CASH_balance.csv`, `instalments_payments.csv`, and `credit_card_balance.csv`).

What are the relationships between these tables?  The value in the `SK_ID_CURR` column of the `application_*.csv` and `bureau.csv` tables identify the applicant.  That is, to combine the two tables into a single table, we could merge on `SK_ID_CURR`.   Similarly, the `SK_ID_BUREAU` column in `bureau.csv` and `bureau_balance.csv` identifies the applicant, though in this case there can be multiple entries in `bureau_balance.csv` for a single applicant.  The text in the line connecting the tables in the diagram above shows what column two tables can be merged on.

We could manually go through all these databases and construct features based on them, but this would entail not just a lot of manual work, but a *lot* of design decisions.  For example, should we construct a feature which corresponds to the maximum amount of credit the applicant has ever carried?  Or the average amount of credit?  Or the monthly median credit?  Should we construct a feature for how many payments the applicant has made, or how regular their payments are, or *when* they make their payments, etc, etc, etc? 

Featuretools allows us to define our datasets, the relationships between our datasets, and automatically extracts features from child datasets into parent datasets using deep feature synthesis.  We'll use Featuretools to generate features from the data in the secondary tables in the Home Credit Group dataset, and keep features which are informative. 

First let's load the packages we need.


```python
# Load packages
import numpy as np
import pandas as pd
import featuretools as ft
from featuretools import selection
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
```

We'll use pandas to load the data.


```python
# Load applications data
train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')
bureau = pd.read_csv('bureau.csv')
bureau_balance = pd.read_csv('bureau_balance.csv')
cash_balance = pd.read_csv('POS_CASH_balance.csv')
card_balance = pd.read_csv('credit_card_balance.csv')
prev_app = pd.read_csv('previous_application.csv')
payments = pd.read_csv('installments_payments.csv')
```

To ensure that featuretools creates the same features for the test set as for the training set, we'll merge the two tables, but add a column which indicates whether each row is a test or training 


```python
# Merge application data
train['Test'] = False
test['Test'] = True
test['TARGET'] = np.nan
app = train.append(test, ignore_index=True, sort=False)
```

Now we can take a look at the data in the main table.


```python
app.sample(10)
```

<div class="scroll_box">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>...</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>118496</th>
      <td>237410</td>
      <td>1.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>247500.0</td>
      <td>12375.0</td>
      <td>247500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.018209</td>
      <td>-16608</td>
      <td>-898</td>
      <td>-26.0</td>
      <td>-119</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Low-skill Laborers</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>THURSDAY</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0548</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>block of flats</td>
      <td>0.0467</td>
      <td>Panel</td>
      <td>No</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>-1007.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>254313</th>
      <td>394281</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>238500.0</td>
      <td>526491.0</td>
      <td>22306.5</td>
      <td>454500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.072508</td>
      <td>-15175</td>
      <td>-1127</td>
      <td>-9211.0</td>
      <td>-3148</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Private service staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>THURSDAY</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.1456</td>
      <td>0.0000</td>
      <td>0.0002</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.3877</td>
      <td>Panel</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1426.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48875</th>
      <td>156605</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>67500.0</td>
      <td>306000.0</td>
      <td>13608.0</td>
      <td>306000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-11140</td>
      <td>-1712</td>
      <td>-821.0</td>
      <td>-845</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0653</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0556</td>
      <td>Panel</td>
      <td>No</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1000.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>52138</th>
      <td>160377</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.0</td>
      <td>296280.0</td>
      <td>19930.5</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.030755</td>
      <td>-9932</td>
      <td>-1871</td>
      <td>-2032.0</td>
      <td>-469</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0791</td>
      <td>0.2717</td>
      <td>0.0000</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0752</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>297167</th>
      <td>444284</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>45000.0</td>
      <td>152820.0</td>
      <td>9949.5</td>
      <td>135000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Separated</td>
      <td>House / apartment</td>
      <td>0.009549</td>
      <td>-16817</td>
      <td>-2620</td>
      <td>-2603.0</td>
      <td>-157</td>
      <td>15.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Drivers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1714.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>146029</th>
      <td>269323</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>0</td>
      <td>202500.0</td>
      <td>521280.0</td>
      <td>28278.0</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Lower secondary</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.019689</td>
      <td>-8448</td>
      <td>-1125</td>
      <td>-3276.0</td>
      <td>-1115</td>
      <td>8.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Drivers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>-536.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>236319</th>
      <td>373722</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>224802.0</td>
      <td>1633473.0</td>
      <td>43087.5</td>
      <td>1363500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Incomplete higher</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.022800</td>
      <td>-8995</td>
      <td>-508</td>
      <td>-24.0</td>
      <td>-1673</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.3271</td>
      <td>0.0039</td>
      <td>0.0087</td>
      <td>org spec account</td>
      <td>block of flats</td>
      <td>0.2932</td>
      <td>Panel</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>327884</th>
      <td>247902</td>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>746280.0</td>
      <td>59094.0</td>
      <td>675000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-16345</td>
      <td>-401</td>
      <td>-168.0</td>
      <td>-5308</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>108336</th>
      <td>225672</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>360000.0</td>
      <td>2250000.0</td>
      <td>59485.5</td>
      <td>2250000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Higher education</td>
      <td>Separated</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-22804</td>
      <td>365243</td>
      <td>-4600.0</td>
      <td>-1567</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.4250</td>
      <td>NaN</td>
      <td>0.4709</td>
      <td>NaN</td>
      <td>block of flats</td>
      <td>0.1515</td>
      <td>Monolithic</td>
      <td>No</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>-649.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>160040</th>
      <td>285530</td>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>2</td>
      <td>135000.0</td>
      <td>178290.0</td>
      <td>14215.5</td>
      <td>157500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Incomplete higher</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.035792</td>
      <td>-13106</td>
      <td>-105</td>
      <td>-1435.0</td>
      <td>-5560</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Managers</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1344.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The first step in using Featuretools is to define the "entities", each of which is one data file or table, and the columns along which they are indexed.  We'll first create an `EntitySet`, which is, obviously, a set of entities or tables.


```python
# Create an entity set
es = ft.EntitySet(id='applications')
```

Now we can add tables to this entity set.  We'll define the datatype of each column (variable) in the table using a dictionary, and pass that to Featuretools' `entity_from_dataframe` function.


```python
# Add dataframe to entityset
es = es.entity_from_dataframe(entity_id='applications',
                              dataframe=app,
                              index='SK_ID_CURR')
```

We can view our entity set so far,


```python
es
```

    Entityset: applications
      Entities:
        applications [Rows: 356255, Columns: 123]
      Relationships:
        No relationships



And the datatypes of each column in the `applications` dataframe.


```python
es['applications']
```

<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>Entity: applications
      Variables:
        SK_ID_CURR (dtype: index)
        TARGET (dtype: numeric)
        NAME_CONTRACT_TYPE (dtype: categorical)
        CODE_GENDER (dtype: categorical)
        FLAG_OWN_CAR (dtype: categorical)
        FLAG_OWN_REALTY (dtype: categorical)
        CNT_CHILDREN (dtype: numeric)
        AMT_INCOME_TOTAL (dtype: numeric)
        AMT_CREDIT (dtype: numeric)
        AMT_ANNUITY (dtype: numeric)
        AMT_GOODS_PRICE (dtype: numeric)
        NAME_TYPE_SUITE (dtype: categorical)
        NAME_INCOME_TYPE (dtype: categorical)
        NAME_EDUCATION_TYPE (dtype: categorical)
        NAME_FAMILY_STATUS (dtype: categorical)
        NAME_HOUSING_TYPE (dtype: categorical)
        REGION_POPULATION_RELATIVE (dtype: numeric)
        DAYS_BIRTH (dtype: numeric)
        DAYS_EMPLOYED (dtype: numeric)
        DAYS_REGISTRATION (dtype: numeric)
        DAYS_ID_PUBLISH (dtype: numeric)
        OWN_CAR_AGE (dtype: numeric)
        FLAG_MOBIL (dtype: numeric)
        FLAG_EMP_PHONE (dtype: numeric)
        FLAG_WORK_PHONE (dtype: numeric)
        FLAG_CONT_MOBILE (dtype: numeric)
        FLAG_PHONE (dtype: numeric)
        FLAG_EMAIL (dtype: numeric)
        OCCUPATION_TYPE (dtype: categorical)
        CNT_FAM_MEMBERS (dtype: numeric)
        REGION_RATING_CLIENT (dtype: numeric)
        REGION_RATING_CLIENT_W_CITY (dtype: numeric)
        WEEKDAY_APPR_PROCESS_START (dtype: categorical)
        HOUR_APPR_PROCESS_START (dtype: numeric)
        REG_REGION_NOT_LIVE_REGION (dtype: numeric)
        REG_REGION_NOT_WORK_REGION (dtype: numeric)
        LIVE_REGION_NOT_WORK_REGION (dtype: numeric)
        REG_CITY_NOT_LIVE_CITY (dtype: numeric)
        REG_CITY_NOT_WORK_CITY (dtype: numeric)
        LIVE_CITY_NOT_WORK_CITY (dtype: numeric)
        ORGANIZATION_TYPE (dtype: categorical)
        EXT_SOURCE_1 (dtype: numeric)
        EXT_SOURCE_2 (dtype: numeric)
        EXT_SOURCE_3 (dtype: numeric)
        APARTMENTS_AVG (dtype: numeric)
        BASEMENTAREA_AVG (dtype: numeric)
        YEARS_BEGINEXPLUATATION_AVG (dtype: numeric)
        YEARS_BUILD_AVG (dtype: numeric)
        COMMONAREA_AVG (dtype: numeric)
        ELEVATORS_AVG (dtype: numeric)
        ENTRANCES_AVG (dtype: numeric)
        FLOORSMAX_AVG (dtype: numeric)
        FLOORSMIN_AVG (dtype: numeric)
        LANDAREA_AVG (dtype: numeric)
        LIVINGAPARTMENTS_AVG (dtype: numeric)
        LIVINGAREA_AVG (dtype: numeric)
        NONLIVINGAPARTMENTS_AVG (dtype: numeric)
        NONLIVINGAREA_AVG (dtype: numeric)
        APARTMENTS_MODE (dtype: numeric)
        BASEMENTAREA_MODE (dtype: numeric)
        YEARS_BEGINEXPLUATATION_MODE (dtype: numeric)
        YEARS_BUILD_MODE (dtype: numeric)
        COMMONAREA_MODE (dtype: numeric)
        ELEVATORS_MODE (dtype: numeric)
        ENTRANCES_MODE (dtype: numeric)
        FLOORSMAX_MODE (dtype: numeric)
        FLOORSMIN_MODE (dtype: numeric)
        LANDAREA_MODE (dtype: numeric)
        LIVINGAPARTMENTS_MODE (dtype: numeric)
        LIVINGAREA_MODE (dtype: numeric)
        NONLIVINGAPARTMENTS_MODE (dtype: numeric)
        NONLIVINGAREA_MODE (dtype: numeric)
        APARTMENTS_MEDI (dtype: numeric)
        BASEMENTAREA_MEDI (dtype: numeric)
        YEARS_BEGINEXPLUATATION_MEDI (dtype: numeric)
        YEARS_BUILD_MEDI (dtype: numeric)
        COMMONAREA_MEDI (dtype: numeric)
        ELEVATORS_MEDI (dtype: numeric)
        ENTRANCES_MEDI (dtype: numeric)
        FLOORSMAX_MEDI (dtype: numeric)
        FLOORSMIN_MEDI (dtype: numeric)
        LANDAREA_MEDI (dtype: numeric)
        LIVINGAPARTMENTS_MEDI (dtype: numeric)
        LIVINGAREA_MEDI (dtype: numeric)
        NONLIVINGAPARTMENTS_MEDI (dtype: numeric)
        NONLIVINGAREA_MEDI (dtype: numeric)
        FONDKAPREMONT_MODE (dtype: categorical)
        HOUSETYPE_MODE (dtype: categorical)
        TOTALAREA_MODE (dtype: numeric)
        WALLSMATERIAL_MODE (dtype: categorical)
        EMERGENCYSTATE_MODE (dtype: categorical)
        OBS_30_CNT_SOCIAL_CIRCLE (dtype: numeric)
        DEF_30_CNT_SOCIAL_CIRCLE (dtype: numeric)
        OBS_60_CNT_SOCIAL_CIRCLE (dtype: numeric)
        DEF_60_CNT_SOCIAL_CIRCLE (dtype: numeric)
        DAYS_LAST_PHONE_CHANGE (dtype: numeric)
        FLAG_DOCUMENT_2 (dtype: numeric)
        FLAG_DOCUMENT_3 (dtype: numeric)
        FLAG_DOCUMENT_4 (dtype: numeric)
        FLAG_DOCUMENT_5 (dtype: numeric)
        FLAG_DOCUMENT_6 (dtype: numeric)
        FLAG_DOCUMENT_7 (dtype: numeric)
        FLAG_DOCUMENT_8 (dtype: numeric)
        FLAG_DOCUMENT_9 (dtype: numeric)
        FLAG_DOCUMENT_10 (dtype: numeric)
        FLAG_DOCUMENT_11 (dtype: numeric)
        FLAG_DOCUMENT_12 (dtype: numeric)
        FLAG_DOCUMENT_13 (dtype: numeric)
        FLAG_DOCUMENT_14 (dtype: numeric)
        FLAG_DOCUMENT_15 (dtype: numeric)
        FLAG_DOCUMENT_16 (dtype: numeric)
        FLAG_DOCUMENT_17 (dtype: numeric)
        FLAG_DOCUMENT_18 (dtype: numeric)
        FLAG_DOCUMENT_19 (dtype: numeric)
        FLAG_DOCUMENT_20 (dtype: numeric)
        FLAG_DOCUMENT_21 (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_HOUR (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_DAY (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_WEEK (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_MON (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_QRT (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_YEAR (dtype: numeric)
        Test (dtype: boolean)
      Shape:
        (Rows: 356255, Columns: 123)
  </code>
  </pre>
  </div>
</div>
    
<br />


Unfortunately it looks like some of the data types are incorrect!  Many of the `FLAG_*` columns should be boolean, not numeric.  Featuretools automatically infers the datatype of each column from the datatype in the pandas dataframe which was input to Featuretools.  To correct this problem, we can either change the datatype in the pandas dataframe, or we can manually set the datatype in Featuretools.  Here we'll do the same operation as before (add the applications dataframe as an entity), but this time we'll manually set the datatype of certain columns.  In addition to the boolean datatype, there are also Index, Datetime, Numeric, Categorical, Ordinal, Text, LatLong, and other [Featuretools datatypes](https://docs.featuretools.com/api_reference.html#variable-types).


```python
# Featuretools datatypes
BOOL = ft.variable_types.Boolean

# Manually define datatypes in app dataframe
variable_types = {
    'FLAG_MOBIL': BOOL,
    'FLAG_EMP_PHONE': BOOL,
    'FLAG_WORK_PHONE': BOOL,
    'FLAG_CONT_MOBILE': BOOL,
    'FLAG_PHONE': BOOL,
    'FLAG_EMAIL': BOOL,
    'REG_REGION_NOT_LIVE_REGION': BOOL,
    'REG_REGION_NOT_WORK_REGION': BOOL,
    'LIVE_REGION_NOT_WORK_REGION': BOOL,
    'REG_CITY_NOT_LIVE_CITY': BOOL,
    'REG_CITY_NOT_WORK_CITY': BOOL,
    'LIVE_CITY_NOT_WORK_CITY': BOOL,
    'FLAG_DOCUMENT_2': BOOL,
    'FLAG_DOCUMENT_3': BOOL,
    'FLAG_DOCUMENT_4': BOOL,
    'FLAG_DOCUMENT_5': BOOL,
    'FLAG_DOCUMENT_6': BOOL,
    'FLAG_DOCUMENT_7': BOOL,
    'FLAG_DOCUMENT_8': BOOL,
    'FLAG_DOCUMENT_9': BOOL,
    'FLAG_DOCUMENT_10': BOOL,
    'FLAG_DOCUMENT_11': BOOL,
    'FLAG_DOCUMENT_12': BOOL,
    'FLAG_DOCUMENT_13': BOOL,
    'FLAG_DOCUMENT_14': BOOL,
    'FLAG_DOCUMENT_15': BOOL,
    'FLAG_DOCUMENT_16': BOOL,
    'FLAG_DOCUMENT_17': BOOL,
    'FLAG_DOCUMENT_18': BOOL,
    'FLAG_DOCUMENT_19': BOOL,
    'FLAG_DOCUMENT_20': BOOL,
    'FLAG_DOCUMENT_21': BOOL,
}

# Add dataframe to entityset, using manual datatypes
es = es.entity_from_dataframe(entity_id='applications',
                              dataframe=app,
                              index='SK_ID_CURR',
                              variable_types=variable_types)
```

And now when we view the column datatypes in the applications entity, they have the correct `boolean` type.


```python
es['applications']
```

<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>Entity: applications
      Variables:
        SK_ID_CURR (dtype: index)
        TARGET (dtype: numeric)
        NAME_CONTRACT_TYPE (dtype: categorical)
        CODE_GENDER (dtype: categorical)
        FLAG_OWN_CAR (dtype: categorical)
        FLAG_OWN_REALTY (dtype: categorical)
        CNT_CHILDREN (dtype: numeric)
        AMT_INCOME_TOTAL (dtype: numeric)
        AMT_CREDIT (dtype: numeric)
        AMT_ANNUITY (dtype: numeric)
        AMT_GOODS_PRICE (dtype: numeric)
        NAME_TYPE_SUITE (dtype: categorical)
        NAME_INCOME_TYPE (dtype: categorical)
        NAME_EDUCATION_TYPE (dtype: categorical)
        NAME_FAMILY_STATUS (dtype: categorical)
        NAME_HOUSING_TYPE (dtype: categorical)
        REGION_POPULATION_RELATIVE (dtype: numeric)
        DAYS_BIRTH (dtype: numeric)
        DAYS_EMPLOYED (dtype: numeric)
        DAYS_REGISTRATION (dtype: numeric)
        DAYS_ID_PUBLISH (dtype: numeric)
        OWN_CAR_AGE (dtype: numeric)
        OCCUPATION_TYPE (dtype: categorical)
        CNT_FAM_MEMBERS (dtype: numeric)
        REGION_RATING_CLIENT (dtype: numeric)
        REGION_RATING_CLIENT_W_CITY (dtype: numeric)
        WEEKDAY_APPR_PROCESS_START (dtype: categorical)
        HOUR_APPR_PROCESS_START (dtype: numeric)
        ORGANIZATION_TYPE (dtype: categorical)
        EXT_SOURCE_1 (dtype: numeric)
        EXT_SOURCE_2 (dtype: numeric)
        EXT_SOURCE_3 (dtype: numeric)
        APARTMENTS_AVG (dtype: numeric)
        BASEMENTAREA_AVG (dtype: numeric)
        YEARS_BEGINEXPLUATATION_AVG (dtype: numeric)
        YEARS_BUILD_AVG (dtype: numeric)
        COMMONAREA_AVG (dtype: numeric)
        ELEVATORS_AVG (dtype: numeric)
        ENTRANCES_AVG (dtype: numeric)
        FLOORSMAX_AVG (dtype: numeric)
        FLOORSMIN_AVG (dtype: numeric)
        LANDAREA_AVG (dtype: numeric)
        LIVINGAPARTMENTS_AVG (dtype: numeric)
        LIVINGAREA_AVG (dtype: numeric)
        NONLIVINGAPARTMENTS_AVG (dtype: numeric)
        NONLIVINGAREA_AVG (dtype: numeric)
        APARTMENTS_MODE (dtype: numeric)
        BASEMENTAREA_MODE (dtype: numeric)
        YEARS_BEGINEXPLUATATION_MODE (dtype: numeric)
        YEARS_BUILD_MODE (dtype: numeric)
        COMMONAREA_MODE (dtype: numeric)
        ELEVATORS_MODE (dtype: numeric)
        ENTRANCES_MODE (dtype: numeric)
        FLOORSMAX_MODE (dtype: numeric)
        FLOORSMIN_MODE (dtype: numeric)
        LANDAREA_MODE (dtype: numeric)
        LIVINGAPARTMENTS_MODE (dtype: numeric)
        LIVINGAREA_MODE (dtype: numeric)
        NONLIVINGAPARTMENTS_MODE (dtype: numeric)
        NONLIVINGAREA_MODE (dtype: numeric)
        APARTMENTS_MEDI (dtype: numeric)
        BASEMENTAREA_MEDI (dtype: numeric)
        YEARS_BEGINEXPLUATATION_MEDI (dtype: numeric)
        YEARS_BUILD_MEDI (dtype: numeric)
        COMMONAREA_MEDI (dtype: numeric)
        ELEVATORS_MEDI (dtype: numeric)
        ENTRANCES_MEDI (dtype: numeric)
        FLOORSMAX_MEDI (dtype: numeric)
        FLOORSMIN_MEDI (dtype: numeric)
        LANDAREA_MEDI (dtype: numeric)
        LIVINGAPARTMENTS_MEDI (dtype: numeric)
        LIVINGAREA_MEDI (dtype: numeric)
        NONLIVINGAPARTMENTS_MEDI (dtype: numeric)
        NONLIVINGAREA_MEDI (dtype: numeric)
        FONDKAPREMONT_MODE (dtype: categorical)
        HOUSETYPE_MODE (dtype: categorical)
        TOTALAREA_MODE (dtype: numeric)
        WALLSMATERIAL_MODE (dtype: categorical)
        EMERGENCYSTATE_MODE (dtype: categorical)
        OBS_30_CNT_SOCIAL_CIRCLE (dtype: numeric)
        DEF_30_CNT_SOCIAL_CIRCLE (dtype: numeric)
        OBS_60_CNT_SOCIAL_CIRCLE (dtype: numeric)
        DEF_60_CNT_SOCIAL_CIRCLE (dtype: numeric)
        DAYS_LAST_PHONE_CHANGE (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_HOUR (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_DAY (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_WEEK (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_MON (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_QRT (dtype: numeric)
        AMT_REQ_CREDIT_BUREAU_YEAR (dtype: numeric)
        Test (dtype: boolean)
        FLAG_MOBIL (dtype: boolean)
        FLAG_EMP_PHONE (dtype: boolean)
        FLAG_WORK_PHONE (dtype: boolean)
        FLAG_CONT_MOBILE (dtype: boolean)
        FLAG_PHONE (dtype: boolean)
        FLAG_EMAIL (dtype: boolean)
        REG_REGION_NOT_LIVE_REGION (dtype: boolean)
        REG_REGION_NOT_WORK_REGION (dtype: boolean)
        LIVE_REGION_NOT_WORK_REGION (dtype: boolean)
        REG_CITY_NOT_LIVE_CITY (dtype: boolean)
        REG_CITY_NOT_WORK_CITY (dtype: boolean)
        LIVE_CITY_NOT_WORK_CITY (dtype: boolean)
        FLAG_DOCUMENT_2 (dtype: boolean)
        FLAG_DOCUMENT_3 (dtype: boolean)
        FLAG_DOCUMENT_4 (dtype: boolean)
        FLAG_DOCUMENT_5 (dtype: boolean)
        FLAG_DOCUMENT_6 (dtype: boolean)
        FLAG_DOCUMENT_7 (dtype: boolean)
        FLAG_DOCUMENT_8 (dtype: boolean)
        FLAG_DOCUMENT_9 (dtype: boolean)
        FLAG_DOCUMENT_10 (dtype: boolean)
        FLAG_DOCUMENT_11 (dtype: boolean)
        FLAG_DOCUMENT_12 (dtype: boolean)
        FLAG_DOCUMENT_13 (dtype: boolean)
        FLAG_DOCUMENT_14 (dtype: boolean)
        FLAG_DOCUMENT_15 (dtype: boolean)
        FLAG_DOCUMENT_16 (dtype: boolean)
        FLAG_DOCUMENT_17 (dtype: boolean)
        FLAG_DOCUMENT_18 (dtype: boolean)
        FLAG_DOCUMENT_19 (dtype: boolean)
        FLAG_DOCUMENT_20 (dtype: boolean)
        FLAG_DOCUMENT_21 (dtype: boolean)
      Shape:
        (Rows: 356255, Columns: 123)
  </code>
  </pre>
  </div>
</div>
    
<br />


Now we'll add the remaining data tables to the entityset.  We'll use `index='New'` to indicate that there is no index column in the dataframe (which uniquely identifies each row), and a new index should be created.


```python
# Featuretools datatypes
BOOL = ft.variable_types.Boolean
ID = ft.variable_types.Id

# Add bureau dataframe to entityset
es = es.entity_from_dataframe(
    entity_id='bureau',
    dataframe=bureau,
    index='SK_ID_BUREAU',
    variable_types={'SK_ID_CURR': ID})

# Add bureau_balance dataframe to entityset
es = es.entity_from_dataframe(
    entity_id='bureau_balance',
    dataframe=bureau_balance,
    index='New',
    variable_types={'SK_ID_BUREAU': ID})

# Add cash_balance dataframe to entityset
es = es.entity_from_dataframe(
    entity_id='cash_balance',
    dataframe=cash_balance,
    index='New',
    variable_types={'SK_ID_PREV': ID,
                    'SK_ID_CURR': ID})

# Add card_balance dataframe to entityset
es = es.entity_from_dataframe(
    entity_id='card_balance',
    dataframe=card_balance,
    index='New',
    variable_types={'SK_ID_PREV': ID,
                    'SK_ID_CURR': ID})
                              
# Add prev_app dataframe to entityset
es = es.entity_from_dataframe(
    entity_id='prev_app',
    dataframe=prev_app,
    index='SK_ID_PREV',
    variable_types={'SK_ID_CURR': ID,
                    'NFLAG_LAST_APPL_IN_DAY': BOOL})

# Add payments dataframe to entityset
es = es.entity_from_dataframe(
    entity_id='payments',
    dataframe=payments,
    index='New',
    variable_types={'SK_ID_PREV': ID,
                    'SK_ID_CURR': ID})
```
    
Now when we view the entity set, we can see it contains all the dataframes.

```python
es
```

    Entityset: applications
      Entities:
        applications [Rows: 356255, Columns: 123]
        bureau [Rows: 1716428, Columns: 17]
        bureau_balance [Rows: 27299925, Columns: 4]
        cash_balance [Rows: 10001358, Columns: 9]
        card_balance [Rows: 3840312, Columns: 24]
        prev_app [Rows: 1670214, Columns: 37]
        payments [Rows: 13605401, Columns: 9]
      Relationships:
        No relationships



The next step is to define the relationships between entities.  That is, what columns in a given entity map to which column in some other entity.


```python
# Define relationships between dataframes
relationships = [
    # parent_entity   parent_variable  child_entity      child_variable
    ('applications', 'SK_ID_CURR',   'bureau',         'SK_ID_CURR'),
    ('bureau',       'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU'),
    ('applications', 'SK_ID_CURR',   'prev_app',       'SK_ID_CURR'),
    ('applications', 'SK_ID_CURR',   'cash_balance',   'SK_ID_CURR'),
    ('applications', 'SK_ID_CURR',   'payments',       'SK_ID_CURR'),
    ('applications', 'SK_ID_CURR',   'card_balance',   'SK_ID_CURR')
]

# Create the relationships
for pe, pv, ce, cv in relationships:
    es = es.add_relationship(ft.Relationship(es[pe][pv], es[ce][cv]))
```
    

Now when we view our entityset, we can see the relationships between tables and columns that we've created.


```python
es
```

    Entityset: applications
      Entities:
        applications [Rows: 356255, Columns: 123]
        bureau [Rows: 1716428, Columns: 17]
        bureau_balance [Rows: 27299925, Columns: 4]
        cash_balance [Rows: 10001358, Columns: 9]
        card_balance [Rows: 3840312, Columns: 24]
        prev_app [Rows: 1670214, Columns: 37]
        payments [Rows: 13605401, Columns: 9]
      Relationships:
        bureau.SK_ID_CURR -> applications.SK_ID_CURR
        bureau_balance.SK_ID_BUREAU -> bureau.SK_ID_BUREAU
        prev_app.SK_ID_CURR -> applications.SK_ID_CURR
        cash_balance.SK_ID_CURR -> applications.SK_ID_CURR
        payments.SK_ID_CURR -> applications.SK_ID_CURR
        card_balance.SK_ID_CURR -> applications.SK_ID_CURR


Next we can define which "feature primitives" we want to use to construct features.  First let's look at a list of all the feature primitives available in Featuretools:


```python
pd.options.display.max_rows = 100
ft.list_primitives()
```

<div class="scroll_box">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>last</td>
      <td>aggregation</td>
      <td>Returns the last value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max</td>
      <td>aggregation</td>
      <td>Finds the maximum non-null value of a numeric ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mode</td>
      <td>aggregation</td>
      <td>Finds the most common element in a categorical...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>std</td>
      <td>aggregation</td>
      <td>Finds the standard deviation of a numeric feat...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>min</td>
      <td>aggregation</td>
      <td>Finds the minimum non-null value of a numeric ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mean</td>
      <td>aggregation</td>
      <td>Computes the average value of a numeric feature.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>skew</td>
      <td>aggregation</td>
      <td>Computes the skewness of a data set.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>trend</td>
      <td>aggregation</td>
      <td>Calculates the slope of the linear trend of va...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sum</td>
      <td>aggregation</td>
      <td>Counts the number of elements of a numeric or ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>median</td>
      <td>aggregation</td>
      <td>Finds the median value of any feature with wel...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>time_since_last</td>
      <td>aggregation</td>
      <td>Time since last related instance.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>num_true</td>
      <td>aggregation</td>
      <td>Finds the number of 'True' values in a boolean.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>avg_time_between</td>
      <td>aggregation</td>
      <td>Computes the average time between consecutive ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>percent_true</td>
      <td>aggregation</td>
      <td>Finds the percent of 'True' values in a boolea...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>all</td>
      <td>aggregation</td>
      <td>Test if all values are 'True'.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>n_most_common</td>
      <td>aggregation</td>
      <td>Finds the N most common elements in a categori...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>count</td>
      <td>aggregation</td>
      <td>Counts the number of non null values.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>any</td>
      <td>aggregation</td>
      <td>Test if any value is 'True'.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>num_unique</td>
      <td>aggregation</td>
      <td>Returns the number of unique categorical varia...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>month</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the month.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>negate</td>
      <td>transform</td>
      <td>Creates a transform feature that negates a fea...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>latitude</td>
      <td>transform</td>
      <td>Returns the first value of the tuple base feat...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>cum_min</td>
      <td>transform</td>
      <td>Calculates the min of previous values of an in...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>divide</td>
      <td>transform</td>
      <td>Creates a transform feature that divides two f...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>week</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the week.</td>
    </tr>
    <tr>
      <th>25</th>
      <td>not</td>
      <td>transform</td>
      <td>For each value of the base feature, negates th...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>cum_count</td>
      <td>transform</td>
      <td>Calculates the number of previous values of an...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>hours</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>minute</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the minute.</td>
    </tr>
    <tr>
      <th>29</th>
      <td>subtract</td>
      <td>transform</td>
      <td>Creates a transform feature that subtracts two...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>cum_sum</td>
      <td>transform</td>
      <td>Calculates the sum of previous values of an in...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>days</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>numwords</td>
      <td>transform</td>
      <td>Returns the words in a given string by countin...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>and</td>
      <td>transform</td>
      <td>For two boolean values, determine if both valu...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>weekday</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>time_since_previous</td>
      <td>transform</td>
      <td>Compute the time since the previous instance.</td>
    </tr>
    <tr>
      <th>36</th>
      <td>years</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>days_since</td>
      <td>transform</td>
      <td>For each value of the base feature, compute th...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>is_null</td>
      <td>transform</td>
      <td>For each value of base feature, return 'True' ...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>months</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>longitude</td>
      <td>transform</td>
      <td>Returns the second value on the tuple base fea...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>mod</td>
      <td>transform</td>
      <td>Creates a transform feature that divides two f...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>weeks</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>percentile</td>
      <td>transform</td>
      <td>For each value of the base feature, determines...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>cum_max</td>
      <td>transform</td>
      <td>Calculates the max of previous values of an in...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>second</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the second.</td>
    </tr>
    <tr>
      <th>46</th>
      <td>minutes</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>multiply</td>
      <td>transform</td>
      <td>Creates a transform feature that multplies two...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>diff</td>
      <td>transform</td>
      <td>Compute the difference between the value of a ...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>cum_mean</td>
      <td>transform</td>
      <td>Calculates the mean of previous values of an i...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>year</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the year.</td>
    </tr>
    <tr>
      <th>51</th>
      <td>hour</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the hour.</td>
    </tr>
    <tr>
      <th>52</th>
      <td>add</td>
      <td>transform</td>
      <td>Creates a transform feature that adds two feat...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>or</td>
      <td>transform</td>
      <td>For two boolean values, determine if one value...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>seconds</td>
      <td>transform</td>
      <td>Transform a Timedelta feature into the number ...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>day</td>
      <td>transform</td>
      <td>Transform a Datetime feature into the day.</td>
    </tr>
    <tr>
      <th>56</th>
      <td>characters</td>
      <td>transform</td>
      <td>Return the characters in a given string.</td>
    </tr>
    <tr>
      <th>57</th>
      <td>weekend</td>
      <td>transform</td>
      <td>Transform Datetime feature into the boolean of...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>isin</td>
      <td>transform</td>
      <td>For each value of the base feature, checks whe...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>absolute</td>
      <td>transform</td>
      <td>Absolute value of base feature.</td>
    </tr>
    <tr>
      <th>60</th>
      <td>haversine</td>
      <td>transform</td>
      <td>Calculate the approximate haversine distance i...</td>
    </tr>
    <tr>
      <th>61</th>
      <td>time_since</td>
      <td>transform</td>
      <td>Calculates time since the cutoff time.</td>
    </tr>
  </tbody>
</table>
</div>



We'll use a simple set of feature primitives: just the mean, count, cumulative sum, and number of unique elements for entries in the secondary data files.  However, you could use whichever combinations of feature primitives you think will be needed for your problem.  You can also simply not pass a list of primitives to use in order to use them all!


```python
# Define which primitives to use
agg_primitives =  ['count', 'mean', 'num_unique']
trans_primitives = ['cum_sum']
```

Finally, we can run deep feature synthesis on our entities given their relationships and a list of feature primitives.  This'll take a while!


```python
# Run deep feature synthesis
dfs_feat, dfs_defs = ft.dfs(entityset=es,
                            target_entity='applications',
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives, 
                            verbose = True,
                            max_depth=2, n_jobs=2)
```

    Built 218 features
    

If we take a look at the dataframe which was returned by Featuretools, we can see that a bunch of features were appended which correspond to our selected feature primitive functions applied to data in the secondary data files which correspond to each row in the main application dataset.

```python
dfs_feat
```

<div class="scroll_box">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>ORGANIZATION_TYPE</th>
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>APARTMENTS_AVG</th>
      <th>BASEMENTAREA_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BUILD_AVG</th>
      <th>COMMONAREA_AVG</th>
      <th>ELEVATORS_AVG</th>
      <th>ENTRANCES_AVG</th>
      <th>FLOORSMAX_AVG</th>
      <th>FLOORSMIN_AVG</th>
      <th>...</th>
      <th>MEAN(prev_app.DAYS_TERMINATION)</th>
      <th>MEAN(prev_app.NFLAG_INSURED_ON_APPROVAL)</th>
      <th>NUM_UNIQUE(prev_app.NAME_CONTRACT_TYPE)</th>
      <th>NUM_UNIQUE(prev_app.WEEKDAY_APPR_PROCESS_START)</th>
      <th>NUM_UNIQUE(prev_app.FLAG_LAST_APPL_PER_CONTRACT)</th>
      <th>NUM_UNIQUE(prev_app.NAME_CASH_LOAN_PURPOSE)</th>
      <th>NUM_UNIQUE(prev_app.NAME_CONTRACT_STATUS)</th>
      <th>NUM_UNIQUE(prev_app.NAME_PAYMENT_TYPE)</th>
      <th>NUM_UNIQUE(prev_app.CODE_REJECT_REASON)</th>
      <th>NUM_UNIQUE(prev_app.NAME_TYPE_SUITE)</th>
      <th>NUM_UNIQUE(prev_app.NAME_CLIENT_TYPE)</th>
      <th>NUM_UNIQUE(prev_app.NAME_GOODS_CATEGORY)</th>
      <th>NUM_UNIQUE(prev_app.NAME_PORTFOLIO)</th>
      <th>NUM_UNIQUE(prev_app.NAME_PRODUCT_TYPE)</th>
      <th>NUM_UNIQUE(prev_app.CHANNEL_TYPE)</th>
      <th>NUM_UNIQUE(prev_app.NAME_SELLER_INDUSTRY)</th>
      <th>NUM_UNIQUE(prev_app.NAME_YIELD_GROUP)</th>
      <th>NUM_UNIQUE(prev_app.PRODUCT_COMBINATION)</th>
      <th>COUNT(payments)</th>
      <th>MEAN(payments.NUM_INSTALMENT_VERSION)</th>
      <th>MEAN(payments.NUM_INSTALMENT_NUMBER)</th>
      <th>MEAN(payments.DAYS_INSTALMENT)</th>
      <th>MEAN(payments.DAYS_ENTRY_PAYMENT)</th>
      <th>MEAN(payments.AMT_INSTALMENT)</th>
      <th>MEAN(payments.AMT_PAYMENT)</th>
      <th>NUM_UNIQUE(payments.SK_ID_PREV)</th>
      <th>COUNT(cash_balance)</th>
      <th>MEAN(cash_balance.MONTHS_BALANCE)</th>
      <th>MEAN(cash_balance.CNT_INSTALMENT)</th>
      <th>MEAN(cash_balance.CNT_INSTALMENT_FUTURE)</th>
      <th>MEAN(cash_balance.SK_DPD)</th>
      <th>MEAN(cash_balance.SK_DPD_DEF)</th>
      <th>NUM_UNIQUE(cash_balance.NAME_CONTRACT_STATUS)</th>
      <th>NUM_UNIQUE(cash_balance.SK_ID_PREV)</th>
      <th>COUNT(bureau_balance)</th>
      <th>MEAN(bureau_balance.MONTHS_BALANCE)</th>
      <th>NUM_UNIQUE(bureau_balance.STATUS)</th>
      <th>MEAN(bureau.COUNT(bureau_balance))</th>
      <th>MEAN(bureau.MEAN(bureau_balance.MONTHS_BALANCE))</th>
      <th>MEAN(bureau.NUM_UNIQUE(bureau_balance.STATUS))</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>568800.0</td>
      <td>20560.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018850</td>
      <td>-19241</td>
      <td>-2329</td>
      <td>-5170.0</td>
      <td>-812</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>18</td>
      <td>Kindergarten</td>
      <td>0.752614</td>
      <td>0.789654</td>
      <td>0.159520</td>
      <td>0.0660</td>
      <td>0.0590</td>
      <td>0.9732</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1379</td>
      <td>0.1250</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>1.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.018801</td>
      <td>-9461</td>
      <td>-637</td>
      <td>-3648.0</td>
      <td>-2120</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>Business Entity Type 3</td>
      <td>0.083037</td>
      <td>0.262949</td>
      <td>0.139376</td>
      <td>0.0247</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6192</td>
      <td>0.0143</td>
      <td>0.000</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.000</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>Family</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003541</td>
      <td>-16765</td>
      <td>-1188</td>
      <td>-1186.0</td>
      <td>-291</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>School</td>
      <td>0.311267</td>
      <td>0.622246</td>
      <td>NaN</td>
      <td>0.0959</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7960</td>
      <td>0.0605</td>
      <td>0.080</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.000</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-19046</td>
      <td>-225</td>
      <td>-4260.0</td>
      <td>-2531</td>
      <td>26.0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.555912</td>
      <td>0.729567</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>99000.000</td>
      <td>222768.0</td>
      <td>17370.0</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.035792</td>
      <td>-18064</td>
      <td>-4469</td>
      <td>-9118.0</td>
      <td>-1623</td>
      <td>NaN</td>
      <td>Low-skill Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>9</td>
      <td>Self-employed</td>
      <td>0.564990</td>
      <td>0.291656</td>
      <td>0.432962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100006</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008019</td>
      <td>-19005</td>
      <td>-3039</td>
      <td>-9833.0</td>
      <td>-2437</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.650442</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100007</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.000</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-19932</td>
      <td>-3038</td>
      <td>-4311.0</td>
      <td>-3458</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>Religion</td>
      <td>NaN</td>
      <td>0.322738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100008</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>99000.000</td>
      <td>490495.5</td>
      <td>27517.5</td>
      <td>454500.0</td>
      <td>Spouse, partner</td>
      <td>State servant</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.035792</td>
      <td>-16941</td>
      <td>-1588</td>
      <td>-4970.0</td>
      <td>-477</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>16</td>
      <td>Other</td>
      <td>NaN</td>
      <td>0.354225</td>
      <td>0.621226</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100009</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>1</td>
      <td>171000.000</td>
      <td>1560726.0</td>
      <td>41301.0</td>
      <td>1395000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.035792</td>
      <td>-13778</td>
      <td>-3130</td>
      <td>-1213.0</td>
      <td>-619</td>
      <td>17.0</td>
      <td>Accountants</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>SUNDAY</td>
      <td>16</td>
      <td>Business Entity Type 3</td>
      <td>0.774761</td>
      <td>0.724000</td>
      <td>0.492060</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>-84.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>-214.0</td>
      <td>-227.0</td>
      <td>8821.260</td>
      <td>8821.260</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100010</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>360000.000</td>
      <td>1530000.0</td>
      <td>42075.0</td>
      <td>1530000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003122</td>
      <td>-18850</td>
      <td>-449</td>
      <td>-4597.0</td>
      <td>-2379</td>
      <td>8.0</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>MONDAY</td>
      <td>16</td>
      <td>Other</td>
      <td>NaN</td>
      <td>0.714279</td>
      <td>0.540654</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100011</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>112500.000</td>
      <td>1019610.0</td>
      <td>33826.5</td>
      <td>913500.0</td>
      <td>Children</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018634</td>
      <td>-20099</td>
      <td>365243</td>
      <td>-7427.0</td>
      <td>-3514</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>14</td>
      <td>XNA</td>
      <td>0.587334</td>
      <td>0.205747</td>
      <td>0.751724</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>-2147.0</td>
      <td>-1189.0</td>
      <td>14588.550</td>
      <td>449.685</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100012</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>405000.0</td>
      <td>20250.0</td>
      <td>405000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.019689</td>
      <td>-14469</td>
      <td>-2019</td>
      <td>-14437.0</td>
      <td>-3992</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>8</td>
      <td>Electricity</td>
      <td>NaN</td>
      <td>0.746644</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>-387.0</td>
      <td>-428.0</td>
      <td>5242.860</td>
      <td>5242.860</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100013</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>663264.0</td>
      <td>69777.0</td>
      <td>630000.0</td>
      <td>NaN</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.019101</td>
      <td>-20038</td>
      <td>-4458</td>
      <td>-2175.0</td>
      <td>-3503</td>
      <td>5.0</td>
      <td>Drivers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>14</td>
      <td>Transport: type 3</td>
      <td>NaN</td>
      <td>0.699787</td>
      <td>0.610991</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100014</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>112500.000</td>
      <td>652500.0</td>
      <td>21177.0</td>
      <td>652500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.022800</td>
      <td>-10197</td>
      <td>-679</td>
      <td>-4427.0</td>
      <td>-738</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>Medicine</td>
      <td>0.319760</td>
      <td>0.651862</td>
      <td>0.363945</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100015</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>38419.155</td>
      <td>148365.0</td>
      <td>10678.5</td>
      <td>135000.0</td>
      <td>Children</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.015221</td>
      <td>-20417</td>
      <td>365243</td>
      <td>-5246.0</td>
      <td>-2512</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>7</td>
      <td>XNA</td>
      <td>0.722044</td>
      <td>0.555183</td>
      <td>0.652897</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100016</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.000</td>
      <td>80865.0</td>
      <td>5881.5</td>
      <td>67500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.031329</td>
      <td>-13439</td>
      <td>-2717</td>
      <td>-311.0</td>
      <td>-3227</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>10</td>
      <td>Business Entity Type 2</td>
      <td>0.464831</td>
      <td>0.715042</td>
      <td>0.176653</td>
      <td>0.0825</td>
      <td>NaN</td>
      <td>0.9811</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100017</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>225000.000</td>
      <td>918468.0</td>
      <td>28966.5</td>
      <td>697500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.016612</td>
      <td>-14086</td>
      <td>-3028</td>
      <td>-643.0</td>
      <td>-4911</td>
      <td>23.0</td>
      <td>Drivers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>13</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.566907</td>
      <td>0.770087</td>
      <td>0.1474</td>
      <td>0.0973</td>
      <td>0.9806</td>
      <td>0.7348</td>
      <td>0.0582</td>
      <td>0.160</td>
      <td>0.1379</td>
      <td>0.3333</td>
      <td>0.3750</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100018</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>189000.000</td>
      <td>773680.5</td>
      <td>32778.0</td>
      <td>679500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.010006</td>
      <td>-14583</td>
      <td>-203</td>
      <td>-615.0</td>
      <td>-2056</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>Transport: type 2</td>
      <td>0.721940</td>
      <td>0.642656</td>
      <td>NaN</td>
      <td>0.3495</td>
      <td>0.1335</td>
      <td>0.9985</td>
      <td>0.9796</td>
      <td>0.1143</td>
      <td>0.400</td>
      <td>0.1724</td>
      <td>0.6667</td>
      <td>0.7083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100019</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>157500.000</td>
      <td>299772.0</td>
      <td>20160.0</td>
      <td>247500.0</td>
      <td>Family</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>Rented apartment</td>
      <td>0.020713</td>
      <td>-8728</td>
      <td>-1157</td>
      <td>-3494.0</td>
      <td>-1368</td>
      <td>17.0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>3</td>
      <td>3</td>
      <td>SATURDAY</td>
      <td>6</td>
      <td>Business Entity Type 2</td>
      <td>0.115634</td>
      <td>0.346634</td>
      <td>0.678568</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100020</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>108000.000</td>
      <td>509602.5</td>
      <td>26149.5</td>
      <td>387000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018634</td>
      <td>-12931</td>
      <td>-1317</td>
      <td>-6392.0</td>
      <td>-3866</td>
      <td>NaN</td>
      <td>Drivers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>12</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.236378</td>
      <td>0.062103</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100021</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>81000.000</td>
      <td>270000.0</td>
      <td>13500.0</td>
      <td>270000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.010966</td>
      <td>-9776</td>
      <td>-191</td>
      <td>-4143.0</td>
      <td>-2427</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>10</td>
      <td>Construction</td>
      <td>NaN</td>
      <td>0.683513</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100022</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>112500.000</td>
      <td>157500.0</td>
      <td>7875.0</td>
      <td>157500.0</td>
      <td>Other_A</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.046220</td>
      <td>-17718</td>
      <td>-7804</td>
      <td>-8751.0</td>
      <td>-1259</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>FRIDAY</td>
      <td>13</td>
      <td>Housing</td>
      <td>NaN</td>
      <td>0.706428</td>
      <td>0.556727</td>
      <td>0.0278</td>
      <td>0.0617</td>
      <td>0.9881</td>
      <td>0.8368</td>
      <td>0.0018</td>
      <td>0.000</td>
      <td>0.1034</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100023</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>90000.000</td>
      <td>544491.0</td>
      <td>17563.5</td>
      <td>454500.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.015221</td>
      <td>-11348</td>
      <td>-2038</td>
      <td>-1021.0</td>
      <td>-3964</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>12</td>
      <td>Kindergarten</td>
      <td>NaN</td>
      <td>0.586617</td>
      <td>0.477649</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100024</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>427500.0</td>
      <td>21375.0</td>
      <td>427500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.015221</td>
      <td>-18252</td>
      <td>-4286</td>
      <td>-298.0</td>
      <td>-1800</td>
      <td>7.0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>13</td>
      <td>Self-employed</td>
      <td>0.565655</td>
      <td>0.113375</td>
      <td>NaN</td>
      <td>0.0722</td>
      <td>0.0801</td>
      <td>0.9781</td>
      <td>0.7008</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100025</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>1</td>
      <td>202500.000</td>
      <td>1132573.5</td>
      <td>37561.5</td>
      <td>927000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-14815</td>
      <td>-1652</td>
      <td>-2299.0</td>
      <td>-2299</td>
      <td>14.0</td>
      <td>Sales staff</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>Trade: type 7</td>
      <td>0.437709</td>
      <td>0.233767</td>
      <td>0.542445</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100026</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>1</td>
      <td>450000.000</td>
      <td>497520.0</td>
      <td>32521.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>Rented apartment</td>
      <td>0.020713</td>
      <td>-11146</td>
      <td>-4306</td>
      <td>-114.0</td>
      <td>-2518</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>3.0</td>
      <td>3</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>6</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.457143</td>
      <td>0.358951</td>
      <td>0.0907</td>
      <td>0.0795</td>
      <td>0.9786</td>
      <td>0.7076</td>
      <td>0.0120</td>
      <td>0.000</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100027</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>83250.000</td>
      <td>239850.0</td>
      <td>23850.0</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.006296</td>
      <td>-24827</td>
      <td>365243</td>
      <td>-9012.0</td>
      <td>-3684</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>FRIDAY</td>
      <td>12</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.624305</td>
      <td>0.669057</td>
      <td>0.1443</td>
      <td>0.0848</td>
      <td>0.9876</td>
      <td>0.8300</td>
      <td>0.1064</td>
      <td>0.140</td>
      <td>0.1207</td>
      <td>0.3750</td>
      <td>0.4167</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100028</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>315000.000</td>
      <td>1575000.0</td>
      <td>49018.5</td>
      <td>1575000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-13976</td>
      <td>-1866</td>
      <td>-2000.0</td>
      <td>-4208</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>11</td>
      <td>Business Entity Type 3</td>
      <td>0.525734</td>
      <td>0.509677</td>
      <td>0.612704</td>
      <td>0.3052</td>
      <td>0.1974</td>
      <td>0.9970</td>
      <td>0.9592</td>
      <td>0.1165</td>
      <td>0.320</td>
      <td>0.2759</td>
      <td>0.3750</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-47.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100029</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>2</td>
      <td>135000.000</td>
      <td>247500.0</td>
      <td>12703.5</td>
      <td>247500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-11286</td>
      <td>-746</td>
      <td>-108.0</td>
      <td>-3729</td>
      <td>7.0</td>
      <td>Drivers</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>14</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.786179</td>
      <td>0.565608</td>
      <td>0.1433</td>
      <td>0.1455</td>
      <td>0.9861</td>
      <td>0.8096</td>
      <td>0.0212</td>
      <td>0.000</td>
      <td>0.3103</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100030</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>90000.000</td>
      <td>225000.0</td>
      <td>11074.5</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-19334</td>
      <td>-3494</td>
      <td>-2419.0</td>
      <td>-2893</td>
      <td>NaN</td>
      <td>Cleaning staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>8</td>
      <td>Business Entity Type 3</td>
      <td>0.561948</td>
      <td>0.651406</td>
      <td>0.461482</td>
      <td>0.0722</td>
      <td>0.0147</td>
      <td>0.9781</td>
      <td>0.7008</td>
      <td>0.0010</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100031</th>
      <td>1.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>112500.000</td>
      <td>979992.0</td>
      <td>27076.5</td>
      <td>702000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.018029</td>
      <td>-18724</td>
      <td>-2628</td>
      <td>-6573.0</td>
      <td>-1827</td>
      <td>NaN</td>
      <td>Cooking staff</td>
      <td>1.0</td>
      <td>3</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.548477</td>
      <td>0.190706</td>
      <td>0.0165</td>
      <td>0.0089</td>
      <td>0.9732</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.0690</td>
      <td>0.0417</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100032</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>112500.000</td>
      <td>327024.0</td>
      <td>23827.5</td>
      <td>270000.0</td>
      <td>Family</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.019101</td>
      <td>-15948</td>
      <td>-1234</td>
      <td>-5782.0</td>
      <td>-3153</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>10</td>
      <td>Industry: type 11</td>
      <td>NaN</td>
      <td>0.541124</td>
      <td>0.659406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100033</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>270000.000</td>
      <td>790830.0</td>
      <td>57676.5</td>
      <td>675000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.046220</td>
      <td>-9994</td>
      <td>-1796</td>
      <td>-4668.0</td>
      <td>-2661</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>WEDNESDAY</td>
      <td>11</td>
      <td>Military</td>
      <td>0.600396</td>
      <td>0.685011</td>
      <td>0.524496</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100034</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>90000.000</td>
      <td>180000.0</td>
      <td>9000.0</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Single / not married</td>
      <td>With parents</td>
      <td>0.030755</td>
      <td>-10341</td>
      <td>-1010</td>
      <td>-4799.0</td>
      <td>-3015</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>16</td>
      <td>Business Entity Type 3</td>
      <td>0.297914</td>
      <td>0.502779</td>
      <td>NaN</td>
      <td>0.1505</td>
      <td>0.0838</td>
      <td>0.9831</td>
      <td>0.7688</td>
      <td>0.0188</td>
      <td>0.160</td>
      <td>0.1379</td>
      <td>0.3333</td>
      <td>0.3750</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100035</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>292500.000</td>
      <td>665892.0</td>
      <td>24592.5</td>
      <td>477000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-15280</td>
      <td>-2668</td>
      <td>-5266.0</td>
      <td>-3787</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>13</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.479987</td>
      <td>0.410103</td>
      <td>0.0124</td>
      <td>NaN</td>
      <td>0.9697</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.0690</td>
      <td>0.0417</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>-1067.0</td>
      <td>-1070.0</td>
      <td>18201.645</td>
      <td>18201.645</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100036</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>112500.000</td>
      <td>512064.0</td>
      <td>25033.5</td>
      <td>360000.0</td>
      <td>Family</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008575</td>
      <td>-11144</td>
      <td>-1104</td>
      <td>-7846.0</td>
      <td>-2904</td>
      <td>NaN</td>
      <td>Private service staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>12</td>
      <td>Services</td>
      <td>0.274422</td>
      <td>0.627300</td>
      <td>NaN</td>
      <td>0.3670</td>
      <td>0.3751</td>
      <td>0.9901</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.280</td>
      <td>0.4828</td>
      <td>0.3750</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100037</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>90000.000</td>
      <td>199008.0</td>
      <td>20893.5</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-12974</td>
      <td>-4404</td>
      <td>-7123.0</td>
      <td>-4464</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>11</td>
      <td>Business Entity Type 2</td>
      <td>NaN</td>
      <td>0.559467</td>
      <td>0.798137</td>
      <td>0.0928</td>
      <td>NaN</td>
      <td>0.9801</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100038</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>180000.000</td>
      <td>625500.0</td>
      <td>32067.0</td>
      <td>625500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-13040</td>
      <td>-2191</td>
      <td>-4000.0</td>
      <td>-4262</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>5</td>
      <td>Business Entity Type 3</td>
      <td>0.202145</td>
      <td>0.425687</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100039</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>360000.000</td>
      <td>733315.5</td>
      <td>39069.0</td>
      <td>679500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.015221</td>
      <td>-11694</td>
      <td>-2060</td>
      <td>-3557.0</td>
      <td>-3557</td>
      <td>3.0</td>
      <td>Drivers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>10</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.321745</td>
      <td>0.411849</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100040</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>1125000.0</td>
      <td>32895.0</td>
      <td>1125000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.019689</td>
      <td>-15997</td>
      <td>-4585</td>
      <td>-5735.0</td>
      <td>-4067</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>14</td>
      <td>Security Ministries</td>
      <td>NaN</td>
      <td>0.172498</td>
      <td>NaN</td>
      <td>0.0825</td>
      <td>0.0804</td>
      <td>0.9762</td>
      <td>0.6736</td>
      <td>0.0056</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100041</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>112500.000</td>
      <td>450000.0</td>
      <td>44509.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.008575</td>
      <td>-12158</td>
      <td>-1275</td>
      <td>-6265.0</td>
      <td>-2009</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>19</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.663158</td>
      <td>0.678568</td>
      <td>0.0948</td>
      <td>0.0792</td>
      <td>0.9861</td>
      <td>0.8096</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.1724</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100042</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>270000.000</td>
      <td>959688.0</td>
      <td>34600.5</td>
      <td>810000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-18604</td>
      <td>-12009</td>
      <td>-6116.0</td>
      <td>-2027</td>
      <td>10.0</td>
      <td>Drivers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>15</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.628904</td>
      <td>0.392774</td>
      <td>0.2412</td>
      <td>0.0084</td>
      <td>0.9821</td>
      <td>0.7552</td>
      <td>0.0452</td>
      <td>0.160</td>
      <td>0.1379</td>
      <td>0.3333</td>
      <td>0.3750</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-2482.0</td>
      <td>-2495.0</td>
      <td>4500.000</td>
      <td>4342.500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100043</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>198000.000</td>
      <td>641173.5</td>
      <td>23157.0</td>
      <td>553500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018850</td>
      <td>-17199</td>
      <td>-768</td>
      <td>-63.0</td>
      <td>-735</td>
      <td>NaN</td>
      <td>Private service staff</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>13</td>
      <td>Other</td>
      <td>0.842763</td>
      <td>0.681699</td>
      <td>0.754406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>-2167.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100044</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.000</td>
      <td>454500.0</td>
      <td>15151.5</td>
      <td>454500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.030755</td>
      <td>-21077</td>
      <td>-1288</td>
      <td>-5474.0</td>
      <td>-4270</td>
      <td>NaN</td>
      <td>Drivers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>10</td>
      <td>Transport: type 4</td>
      <td>0.804586</td>
      <td>0.719799</td>
      <td>0.722393</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100045</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>99000.000</td>
      <td>247275.0</td>
      <td>17338.5</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.006207</td>
      <td>-23920</td>
      <td>365243</td>
      <td>-9817.0</td>
      <td>-4969</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>11</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.650765</td>
      <td>0.751724</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.9851</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.0345</td>
      <td>0.3333</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100046</th>
      <td>0.0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.000</td>
      <td>540000.0</td>
      <td>27000.0</td>
      <td>540000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-16126</td>
      <td>-1761</td>
      <td>-8236.0</td>
      <td>-4292</td>
      <td>3.0</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>8</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.738053</td>
      <td>0.605836</td>
      <td>0.0814</td>
      <td>0.0994</td>
      <td>0.9831</td>
      <td>0.7688</td>
      <td>0.0142</td>
      <td>0.000</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100047</th>
      <td>1.0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>1193580.0</td>
      <td>35028.0</td>
      <td>855000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.025164</td>
      <td>-17482</td>
      <td>-1262</td>
      <td>-1182.0</td>
      <td>-1029</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>9</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.306841</td>
      <td>0.320163</td>
      <td>0.1309</td>
      <td>0.1250</td>
      <td>0.9960</td>
      <td>0.9456</td>
      <td>0.0822</td>
      <td>0.160</td>
      <td>0.1379</td>
      <td>0.2500</td>
      <td>0.2917</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-40.0</td>
      <td>24.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100048</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>604152.0</td>
      <td>29196.0</td>
      <td>540000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.009175</td>
      <td>-16971</td>
      <td>-475</td>
      <td>-3148.0</td>
      <td>-513</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>12</td>
      <td>Industry: type 1</td>
      <td>NaN</td>
      <td>0.037315</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100049</th>
      <td>1.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>135000.000</td>
      <td>288873.0</td>
      <td>16258.5</td>
      <td>238500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.007305</td>
      <td>-13384</td>
      <td>-3597</td>
      <td>-45.0</td>
      <td>-4409</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>Self-employed</td>
      <td>0.468208</td>
      <td>0.674203</td>
      <td>0.399676</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100050</th>
      <td>0.0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>108000.000</td>
      <td>746280.0</td>
      <td>42970.5</td>
      <td>675000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Higher education</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.010966</td>
      <td>-23548</td>
      <td>365243</td>
      <td>-5745.0</td>
      <td>-4576</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>9</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.766138</td>
      <td>0.684828</td>
      <td>0.2186</td>
      <td>0.1232</td>
      <td>0.9851</td>
      <td>0.7960</td>
      <td>0.0528</td>
      <td>0.240</td>
      <td>0.2069</td>
      <td>0.3333</td>
      <td>0.3750</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172294</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>261000.000</td>
      <td>398934.0</td>
      <td>17032.5</td>
      <td>333000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Lower secondary</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.018029</td>
      <td>-20714</td>
      <td>365243</td>
      <td>-2967.0</td>
      <td>-3723</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>3</td>
      <td>3</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>XNA</td>
      <td>0.805874</td>
      <td>0.546836</td>
      <td>0.515495</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172300</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>180000.000</td>
      <td>610762.5</td>
      <td>59629.5</td>
      <td>562500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.030755</td>
      <td>-16409</td>
      <td>-4277</td>
      <td>-3959.0</td>
      <td>-5370</td>
      <td>NaN</td>
      <td>Accountants</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>11</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.722656</td>
      <td>0.754406</td>
      <td>0.1237</td>
      <td>0.0514</td>
      <td>0.9940</td>
      <td>0.9184</td>
      <td>0.0588</td>
      <td>0.120</td>
      <td>0.1034</td>
      <td>0.3750</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172301</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>225000.000</td>
      <td>601470.0</td>
      <td>32629.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-9966</td>
      <td>-1204</td>
      <td>-2765.0</td>
      <td>-2647</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>15</td>
      <td>Business Entity Type 3</td>
      <td>0.099045</td>
      <td>0.561449</td>
      <td>0.538863</td>
      <td>0.0711</td>
      <td>0.1161</td>
      <td>0.9911</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.2069</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172310</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>225000.000</td>
      <td>891000.0</td>
      <td>45621.0</td>
      <td>891000.0</td>
      <td>Family</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.015221</td>
      <td>-18612</td>
      <td>-11709</td>
      <td>-7420.0</td>
      <td>-1941</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>12</td>
      <td>Electricity</td>
      <td>NaN</td>
      <td>0.661887</td>
      <td>0.472253</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172351</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>175500.000</td>
      <td>450000.0</td>
      <td>23107.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Higher education</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.072508</td>
      <td>-23735</td>
      <td>365243</td>
      <td>-8920.0</td>
      <td>-4584</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>FRIDAY</td>
      <td>17</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.737283</td>
      <td>0.706205</td>
      <td>0.1229</td>
      <td>0.1063</td>
      <td>0.9771</td>
      <td>0.6872</td>
      <td>0.0000</td>
      <td>0.048</td>
      <td>0.1724</td>
      <td>0.2000</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172353</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>522814.5</td>
      <td>51061.5</td>
      <td>481500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.019101</td>
      <td>-18196</td>
      <td>-204</td>
      <td>-5891.0</td>
      <td>-1747</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>15</td>
      <td>Trade: type 7</td>
      <td>0.587458</td>
      <td>0.597460</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172358</th>
      <td>NaN</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>2</td>
      <td>135000.000</td>
      <td>180000.0</td>
      <td>9000.0</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.022625</td>
      <td>-12875</td>
      <td>-2904</td>
      <td>-2605.0</td>
      <td>-4418</td>
      <td>7.0</td>
      <td>High skill tech staff</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>University</td>
      <td>0.822206</td>
      <td>0.750721</td>
      <td>0.619528</td>
      <td>0.0722</td>
      <td>0.0840</td>
      <td>0.9811</td>
      <td>0.7416</td>
      <td>0.0077</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172374</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>90000.000</td>
      <td>1483650.0</td>
      <td>62991.0</td>
      <td>1350000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.002134</td>
      <td>-11940</td>
      <td>-3259</td>
      <td>-533.0</td>
      <td>-2640</td>
      <td>NaN</td>
      <td>Medicine staff</td>
      <td>4.0</td>
      <td>3</td>
      <td>3</td>
      <td>WEDNESDAY</td>
      <td>9</td>
      <td>Medicine</td>
      <td>NaN</td>
      <td>0.651738</td>
      <td>0.379100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172380</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>1</td>
      <td>180000.000</td>
      <td>405000.0</td>
      <td>43749.0</td>
      <td>405000.0</td>
      <td>Family</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.011703</td>
      <td>-10896</td>
      <td>-316</td>
      <td>-720.0</td>
      <td>-2268</td>
      <td>14.0</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.691089</td>
      <td>0.275000</td>
      <td>0.1196</td>
      <td>0.1268</td>
      <td>0.9781</td>
      <td>0.7008</td>
      <td>0.0541</td>
      <td>0.000</td>
      <td>0.2759</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172384</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>157500.000</td>
      <td>599544.0</td>
      <td>24070.5</td>
      <td>495000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.010643</td>
      <td>-19940</td>
      <td>-2955</td>
      <td>-1183.0</td>
      <td>-3449</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>20</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.552811</td>
      <td>0.664248</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172400</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>67765.5</td>
      <td>7074.0</td>
      <td>58500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.031329</td>
      <td>-11811</td>
      <td>-3683</td>
      <td>-7871.0</td>
      <td>-4276</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>Services</td>
      <td>0.762212</td>
      <td>0.377727</td>
      <td>NaN</td>
      <td>0.1124</td>
      <td>0.0528</td>
      <td>0.9811</td>
      <td>0.7416</td>
      <td>0.0379</td>
      <td>0.040</td>
      <td>0.0345</td>
      <td>0.5417</td>
      <td>0.5833</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172403</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>525735.0</td>
      <td>39438.0</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>With parents</td>
      <td>0.010032</td>
      <td>-9549</td>
      <td>-1387</td>
      <td>-4239.0</td>
      <td>-2219</td>
      <td>NaN</td>
      <td>Medicine staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>4</td>
      <td>Medicine</td>
      <td>0.667826</td>
      <td>0.249708</td>
      <td>NaN</td>
      <td>0.0619</td>
      <td>0.0350</td>
      <td>0.9742</td>
      <td>0.6464</td>
      <td>0.0216</td>
      <td>0.000</td>
      <td>0.1034</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172408</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>225000.000</td>
      <td>817560.0</td>
      <td>30951.0</td>
      <td>675000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.007120</td>
      <td>-21327</td>
      <td>365243</td>
      <td>-9992.0</td>
      <td>-4871</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>SUNDAY</td>
      <td>13</td>
      <td>XNA</td>
      <td>0.421896</td>
      <td>0.283795</td>
      <td>0.360613</td>
      <td>0.2010</td>
      <td>0.0352</td>
      <td>0.9851</td>
      <td>0.7960</td>
      <td>NaN</td>
      <td>0.080</td>
      <td>0.0345</td>
      <td>0.3333</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172411</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>351792.0</td>
      <td>18090.0</td>
      <td>252000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.011657</td>
      <td>-21864</td>
      <td>365243</td>
      <td>-1813.0</td>
      <td>-1449</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>THURSDAY</td>
      <td>12</td>
      <td>XNA</td>
      <td>0.777975</td>
      <td>0.700182</td>
      <td>0.827703</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172420</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>157500.000</td>
      <td>472500.0</td>
      <td>33016.5</td>
      <td>472500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008866</td>
      <td>-20067</td>
      <td>-12735</td>
      <td>-10143.0</td>
      <td>-3531</td>
      <td>4.0</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>15</td>
      <td>Construction</td>
      <td>NaN</td>
      <td>0.479163</td>
      <td>0.567379</td>
      <td>0.1031</td>
      <td>0.0804</td>
      <td>0.9742</td>
      <td>0.6464</td>
      <td>0.0128</td>
      <td>0.000</td>
      <td>0.1724</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172426</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>58500.000</td>
      <td>232438.5</td>
      <td>21316.5</td>
      <td>211500.0</td>
      <td>Spouse, partner</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.002134</td>
      <td>-14622</td>
      <td>-1676</td>
      <td>-825.0</td>
      <td>-4874</td>
      <td>NaN</td>
      <td>Low-skill Laborers</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>THURSDAY</td>
      <td>10</td>
      <td>Medicine</td>
      <td>NaN</td>
      <td>0.158119</td>
      <td>0.133429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172433</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>180000.000</td>
      <td>269982.0</td>
      <td>28350.0</td>
      <td>238500.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.032561</td>
      <td>-15726</td>
      <td>-1151</td>
      <td>-8206.0</td>
      <td>-4038</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>WEDNESDAY</td>
      <td>18</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.641339</td>
      <td>0.542445</td>
      <td>0.1856</td>
      <td>0.2315</td>
      <td>0.9816</td>
      <td>0.7484</td>
      <td>0.0000</td>
      <td>0.240</td>
      <td>0.2069</td>
      <td>0.2500</td>
      <td>0.2917</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172434</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>166500.000</td>
      <td>388948.5</td>
      <td>19885.5</td>
      <td>324666.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.030755</td>
      <td>-11007</td>
      <td>-3549</td>
      <td>-4378.0</td>
      <td>-3677</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>10</td>
      <td>Self-employed</td>
      <td>0.473368</td>
      <td>0.527331</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172442</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>157500.000</td>
      <td>93829.5</td>
      <td>9981.0</td>
      <td>81000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Incomplete higher</td>
      <td>Separated</td>
      <td>House / apartment</td>
      <td>0.002506</td>
      <td>-18137</td>
      <td>365243</td>
      <td>-9793.0</td>
      <td>-1679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>6</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.372770</td>
      <td>0.762336</td>
      <td>0.1227</td>
      <td>NaN</td>
      <td>0.9826</td>
      <td>0.7620</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.2759</td>
      <td>0.1667</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172443</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.000</td>
      <td>509400.0</td>
      <td>32683.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.019101</td>
      <td>-23786</td>
      <td>365243</td>
      <td>-5163.0</td>
      <td>-4646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>12</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.425645</td>
      <td>0.075966</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172452</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>728460.0</td>
      <td>66942.0</td>
      <td>675000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018850</td>
      <td>-9863</td>
      <td>-1262</td>
      <td>-4638.0</td>
      <td>-2471</td>
      <td>7.0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>15</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.595465</td>
      <td>0.189595</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172454</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>90000.000</td>
      <td>183784.5</td>
      <td>12919.5</td>
      <td>148500.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.035792</td>
      <td>-22654</td>
      <td>365243</td>
      <td>-7409.0</td>
      <td>-4253</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.384018</td>
      <td>0.349055</td>
      <td>0.1098</td>
      <td>0.0495</td>
      <td>0.9722</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.2414</td>
      <td>0.1458</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172456</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>162000.000</td>
      <td>171000.0</td>
      <td>13509.0</td>
      <td>171000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.015221</td>
      <td>-12195</td>
      <td>-909</td>
      <td>-9649.0</td>
      <td>-4078</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>18</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.257882</td>
      <td>0.236611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172460</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>202500.000</td>
      <td>90000.0</td>
      <td>10678.5</td>
      <td>90000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-11180</td>
      <td>-3287</td>
      <td>-4975.0</td>
      <td>-3153</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>10</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.208509</td>
      <td>0.324891</td>
      <td>0.5526</td>
      <td>0.3435</td>
      <td>0.9796</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.600</td>
      <td>0.5172</td>
      <td>0.3333</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172470</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>112500.000</td>
      <td>259794.0</td>
      <td>25825.5</td>
      <td>229500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>Municipal apartment</td>
      <td>0.009175</td>
      <td>-18258</td>
      <td>-5502</td>
      <td>-4749.0</td>
      <td>-451</td>
      <td>NaN</td>
      <td>High skill tech staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>13</td>
      <td>University</td>
      <td>NaN</td>
      <td>0.684308</td>
      <td>0.411849</td>
      <td>0.1309</td>
      <td>0.0728</td>
      <td>0.9901</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.040</td>
      <td>0.0517</td>
      <td>0.2708</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172479</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>112500.000</td>
      <td>450000.0</td>
      <td>35554.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Incomplete higher</td>
      <td>Single / not married</td>
      <td>With parents</td>
      <td>0.010147</td>
      <td>-10626</td>
      <td>-480</td>
      <td>-9931.0</td>
      <td>-2839</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>14</td>
      <td>Business Entity Type 3</td>
      <td>0.161121</td>
      <td>0.707199</td>
      <td>0.659406</td>
      <td>0.1227</td>
      <td>0.1204</td>
      <td>0.9781</td>
      <td>0.7008</td>
      <td>0.0165</td>
      <td>0.000</td>
      <td>0.2759</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172481</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>2</td>
      <td>112500.000</td>
      <td>218016.0</td>
      <td>12645.0</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>With parents</td>
      <td>0.007120</td>
      <td>-13232</td>
      <td>-3363</td>
      <td>-6719.0</td>
      <td>-4445</td>
      <td>9.0</td>
      <td>Laborers</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>10</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.627137</td>
      <td>0.735221</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172483</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>90000.000</td>
      <td>450000.0</td>
      <td>43969.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.026392</td>
      <td>-17781</td>
      <td>-2948</td>
      <td>-6735.0</td>
      <td>-1316</td>
      <td>12.0</td>
      <td>Sales staff</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.661441</td>
      <td>0.776410</td>
      <td>0.0660</td>
      <td>0.0764</td>
      <td>0.9781</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172484</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>90000.000</td>
      <td>225000.0</td>
      <td>14377.5</td>
      <td>225000.0</td>
      <td>NaN</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.022625</td>
      <td>-13972</td>
      <td>-3544</td>
      <td>-8126.0</td>
      <td>-1249</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>Industry: type 3</td>
      <td>NaN</td>
      <td>0.263772</td>
      <td>0.101459</td>
      <td>0.0629</td>
      <td>0.0833</td>
      <td>0.9821</td>
      <td>0.7552</td>
      <td>0.0077</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172485</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>67500.000</td>
      <td>140746.5</td>
      <td>16830.0</td>
      <td>121500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.020713</td>
      <td>-12692</td>
      <td>-1475</td>
      <td>-4691.0</td>
      <td>-4731</td>
      <td>NaN</td>
      <td>Laborers</td>
      <td>4.0</td>
      <td>3</td>
      <td>3</td>
      <td>FRIDAY</td>
      <td>7</td>
      <td>Industry: type 3</td>
      <td>0.585623</td>
      <td>0.440246</td>
      <td>0.634706</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172486</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>1</td>
      <td>81000.000</td>
      <td>189621.0</td>
      <td>12802.5</td>
      <td>144000.0</td>
      <td>Spouse, partner</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.008474</td>
      <td>-17928</td>
      <td>-674</td>
      <td>-396.0</td>
      <td>-1451</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>17</td>
      <td>Construction</td>
      <td>0.484702</td>
      <td>0.490737</td>
      <td>0.535276</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172493</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>1</td>
      <td>225000.000</td>
      <td>1096020.0</td>
      <td>56092.5</td>
      <td>900000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>Office apartment</td>
      <td>0.007120</td>
      <td>-15654</td>
      <td>-1434</td>
      <td>-717.0</td>
      <td>-4530</td>
      <td>8.0</td>
      <td>Managers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>10</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.658448</td>
      <td>0.590233</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172494</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>157500.000</td>
      <td>180000.0</td>
      <td>9319.5</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018209</td>
      <td>-14674</td>
      <td>-122</td>
      <td>-1337.0</td>
      <td>-1482</td>
      <td>NaN</td>
      <td>HR staff</td>
      <td>3.0</td>
      <td>3</td>
      <td>3</td>
      <td>WEDNESDAY</td>
      <td>13</td>
      <td>Security Ministries</td>
      <td>0.502997</td>
      <td>0.203988</td>
      <td>0.627991</td>
      <td>0.1557</td>
      <td>0.2413</td>
      <td>0.9980</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.160</td>
      <td>0.0690</td>
      <td>0.3333</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172497</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>1</td>
      <td>270000.000</td>
      <td>1350000.0</td>
      <td>55845.0</td>
      <td>1350000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018634</td>
      <td>-10324</td>
      <td>-1731</td>
      <td>-113.0</td>
      <td>-2724</td>
      <td>NaN</td>
      <td>Realty agents</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>14</td>
      <td>Self-employed</td>
      <td>0.261527</td>
      <td>0.184967</td>
      <td>NaN</td>
      <td>0.1887</td>
      <td>0.1228</td>
      <td>0.9990</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.160</td>
      <td>0.1379</td>
      <td>0.3750</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172504</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>2</td>
      <td>135000.000</td>
      <td>360000.0</td>
      <td>26194.5</td>
      <td>360000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.018850</td>
      <td>-10327</td>
      <td>-2962</td>
      <td>-5122.0</td>
      <td>-2093</td>
      <td>8.0</td>
      <td>Sales staff</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>13</td>
      <td>Trade: type 2</td>
      <td>NaN</td>
      <td>0.703404</td>
      <td>0.588488</td>
      <td>0.0577</td>
      <td>NaN</td>
      <td>0.9806</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172518</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>67500.000</td>
      <td>1066320.0</td>
      <td>38430.0</td>
      <td>900000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003122</td>
      <td>-9793</td>
      <td>-2830</td>
      <td>-4003.0</td>
      <td>-2412</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3</td>
      <td>3</td>
      <td>MONDAY</td>
      <td>13</td>
      <td>School</td>
      <td>NaN</td>
      <td>0.143042</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172523</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>3</td>
      <td>81000.000</td>
      <td>225000.0</td>
      <td>16951.5</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.007020</td>
      <td>-10823</td>
      <td>-127</td>
      <td>-3400.0</td>
      <td>-3407</td>
      <td>NaN</td>
      <td>Core staff</td>
      <td>5.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>14</td>
      <td>Bank</td>
      <td>NaN</td>
      <td>0.273941</td>
      <td>0.472253</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172527</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>202500.000</td>
      <td>276277.5</td>
      <td>15988.5</td>
      <td>238500.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Incomplete higher</td>
      <td>Married</td>
      <td>With parents</td>
      <td>0.020246</td>
      <td>-12430</td>
      <td>-1271</td>
      <td>-9610.0</td>
      <td>-3222</td>
      <td>NaN</td>
      <td>Sales staff</td>
      <td>3.0</td>
      <td>3</td>
      <td>3</td>
      <td>THURSDAY</td>
      <td>10</td>
      <td>Business Entity Type 3</td>
      <td>0.522961</td>
      <td>0.540662</td>
      <td>0.441836</td>
      <td>0.0876</td>
      <td>0.0808</td>
      <td>0.9801</td>
      <td>0.7280</td>
      <td>0.0056</td>
      <td>0.040</td>
      <td>0.1466</td>
      <td>0.2188</td>
      <td>0.2604</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172533</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>81000.000</td>
      <td>525735.0</td>
      <td>40815.0</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003122</td>
      <td>-21854</td>
      <td>365243</td>
      <td>-7846.0</td>
      <td>-4346</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.569552</td>
      <td>0.838725</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172536</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>225000.000</td>
      <td>260640.0</td>
      <td>26838.0</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.011703</td>
      <td>-14516</td>
      <td>-2356</td>
      <td>-8600.0</td>
      <td>-6214</td>
      <td>1.0</td>
      <td>Drivers</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>15</td>
      <td>Construction</td>
      <td>NaN</td>
      <td>0.641531</td>
      <td>0.438281</td>
      <td>0.1113</td>
      <td>0.0000</td>
      <td>0.9866</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.120</td>
      <td>0.1034</td>
      <td>0.3333</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172540</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>550980.0</td>
      <td>43659.0</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.046220</td>
      <td>-9039</td>
      <td>-2106</td>
      <td>-431.0</td>
      <td>-1707</td>
      <td>10.0</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>18</td>
      <td>Trade: type 3</td>
      <td>0.272675</td>
      <td>0.592707</td>
      <td>0.404878</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172541</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>N</td>
      <td>2</td>
      <td>90000.000</td>
      <td>180000.0</td>
      <td>13225.5</td>
      <td>180000.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.007020</td>
      <td>-10563</td>
      <td>-504</td>
      <td>-2177.0</td>
      <td>-2379</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>13</td>
      <td>Police</td>
      <td>NaN</td>
      <td>0.134089</td>
      <td>0.644679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172543</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>112500.000</td>
      <td>260640.0</td>
      <td>26838.0</td>
      <td>225000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.010147</td>
      <td>-15367</td>
      <td>-2089</td>
      <td>-7787.0</td>
      <td>-4388</td>
      <td>NaN</td>
      <td>Drivers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>FRIDAY</td>
      <td>14</td>
      <td>Self-employed</td>
      <td>NaN</td>
      <td>0.611195</td>
      <td>0.712155</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.9796</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>-310.0</td>
      <td>-312.0</td>
      <td>15353.505</td>
      <td>15353.505</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-53.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172545</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>112500.000</td>
      <td>431280.0</td>
      <td>20875.5</td>
      <td>360000.0</td>
      <td>Family</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.003069</td>
      <td>-10346</td>
      <td>-1170</td>
      <td>-4424.0</td>
      <td>-3002</td>
      <td>NaN</td>
      <td>Drivers</td>
      <td>1.0</td>
      <td>3</td>
      <td>3</td>
      <td>FRIDAY</td>
      <td>10</td>
      <td>Trade: type 7</td>
      <td>NaN</td>
      <td>0.496317</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172547</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.000</td>
      <td>459000.0</td>
      <td>17433.0</td>
      <td>459000.0</td>
      <td>Unaccompanied</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.019689</td>
      <td>-23356</td>
      <td>365243</td>
      <td>-5146.0</td>
      <td>-5146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>SATURDAY</td>
      <td>11</td>
      <td>XNA</td>
      <td>NaN</td>
      <td>0.737084</td>
      <td>0.282248</td>
      <td>0.0804</td>
      <td>0.0567</td>
      <td>0.9757</td>
      <td>0.6668</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.1379</td>
      <td>0.1667</td>
      <td>0.0417</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172551</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.000</td>
      <td>454500.0</td>
      <td>29173.5</td>
      <td>454500.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-18338</td>
      <td>-11264</td>
      <td>-11220.0</td>
      <td>-1892</td>
      <td>NaN</td>
      <td>Medicine staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>13</td>
      <td>Medicine</td>
      <td>NaN</td>
      <td>0.536456</td>
      <td>0.675413</td>
      <td>0.0979</td>
      <td>0.0552</td>
      <td>0.9717</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.0690</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172556</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>180000.000</td>
      <td>500490.0</td>
      <td>52555.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>Municipal apartment</td>
      <td>0.046220</td>
      <td>-15040</td>
      <td>-2125</td>
      <td>-8982.0</td>
      <td>-4474</td>
      <td>18.0</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>FRIDAY</td>
      <td>19</td>
      <td>Other</td>
      <td>0.438157</td>
      <td>0.744736</td>
      <td>0.352340</td>
      <td>0.0577</td>
      <td>NaN</td>
      <td>0.9752</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1034</td>
      <td>0.1667</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172562</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.000</td>
      <td>523152.0</td>
      <td>37336.5</td>
      <td>463500.0</td>
      <td>Unaccompanied</td>
      <td>State servant</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>0.032561</td>
      <td>-17225</td>
      <td>-114</td>
      <td>-1397.0</td>
      <td>-771</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>TUESDAY</td>
      <td>12</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.760022</td>
      <td>NaN</td>
      <td>0.1381</td>
      <td>0.1397</td>
      <td>0.9752</td>
      <td>0.6600</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.3103</td>
      <td>0.1667</td>
      <td>0.2083</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172570</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>382500.000</td>
      <td>967500.0</td>
      <td>31338.0</td>
      <td>967500.0</td>
      <td>Family</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>Municipal apartment</td>
      <td>0.072508</td>
      <td>-14852</td>
      <td>-839</td>
      <td>-3520.0</td>
      <td>-1753</td>
      <td>NaN</td>
      <td>IT staff</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>THURSDAY</td>
      <td>16</td>
      <td>Government</td>
      <td>0.687417</td>
      <td>0.623376</td>
      <td>0.243186</td>
      <td>0.2603</td>
      <td>0.1686</td>
      <td>0.9791</td>
      <td>0.7144</td>
      <td>0.0382</td>
      <td>0.280</td>
      <td>0.2414</td>
      <td>0.3333</td>
      <td>0.3750</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>172574</th>
      <td>NaN</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>1</td>
      <td>112500.000</td>
      <td>539100.0</td>
      <td>22837.5</td>
      <td>450000.0</td>
      <td>Unaccompanied</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>With parents</td>
      <td>0.010276</td>
      <td>-11056</td>
      <td>-142</td>
      <td>-5187.0</td>
      <td>-3709</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
      <td>2</td>
      <td>TUESDAY</td>
      <td>13</td>
      <td>Business Entity Type 3</td>
      <td>0.382110</td>
      <td>0.737846</td>
      <td>0.418854</td>
      <td>0.0247</td>
      <td>NaN</td>
      <td>0.9697</td>
      <td>0.5784</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.0862</td>
      <td>0.0625</td>
      <td>0.1250</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now that we've generated a bunch of features, we should make sure to remove ones which don't carry any information.  Featuretools includes a function to remove features which are entirely NULLs or only have one class, etc. 


```python
# Remove low information features
dfs_feat = selection.remove_low_information_features(dfs_feat)
```    

In some cases it might also be a good idea to do further feature selection at this point, by, say, removing features which have low mutual information with the target variable (loan default).

<a class="anchor" id="predictions-from-generated-features"></a>
## Predictions from Generated Features

Now that we've generated features using Featuretools, we can use those generated features in a predictive model.  First, we would have to perform feature encoding on our generated features.  See my [other post](https://brendanhasz.github.io/2018/10/11/loan-risk-prediction.html) on encoding features of this dataset.  Then, we have to split our features back into training and test datasets, and remove the indicator columns.


```python
# Split data back into test + train
train = dfs_feat.loc[~app['Test'], :].copy()
test = dfs_feat.loc[app['Test'], :].copy()

# Ensure all data is stored as floats
train = train.astype(np.float32)
test = test.astype(np.float32)

# Target labels
train_y = train['TARGET']

# Remove test/train indicator column and target column
train.drop(columns=['Test', 'TARGET'], inplace=True)
test.drop(columns=['Test', 'TARGET'], inplace=True)
```

Then we can run a predictive model, such as LightGBM, on the generated features to predict how likely applicants are to default on their loans.

```python
# Classification pipeline w/ LightGBM
lgbm_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', CalibratedClassifierCV(
                        base_estimator=LGBMClassifier(),
                        method='isotonic'))
])

# Fit to training data
lgbm_fit = lgbm_pipeline.fit(train, train_y)

# Predict loan default probabilities of test data
test_pred = lgbm_fit.predict_proba(test)

# Save predictions to file
df_out = pd.DataFrame()
df_out['SK_ID_CURR'] = test.index
df_out['TARGET'] = test_pred[:,1]
df_out.to_csv('test_predictions.csv', index=False)
```

<a class="anchor" id="running-out-of-memory"></a>
## Running out of Memory

The downside of Featuretools is that is isn't generating features all that intelligently - it simply generates features by applying all the feature primitives to all the features in secondary datasets recursively.  This means that the number of features which are generated can be *huge*!  When dealing with large datasets, this means that the feature generation process might take up more memory than is available on a personal computer.  If you run out of memory, you can always [run featuretools on an Amazon Web Services EC2 instance](https://brendanhasz.github.io/2018/08/30/aws.html) which has enough memory, such as the `r5` class of instances.
