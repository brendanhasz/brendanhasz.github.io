---
layout: post
title: "Automated Feature Engineering with Featuretools"
date: 2018-11-11
description: "Running deep feature synthesis for automated feature engineering, using the Featuretools package for Python."
img_url: /assets/img/featuretools/DataframeTree.svg
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
  ('applications', 'SK_ID_CURR',    'bureau',         'SK_ID_CURR'),
  ('bureau',       'SK_ID_BUREAU',  'bureau_balance', 'SK_ID_BUREAU'),
  ('applications', 'SK_ID_CURR',    'prev_app',       'SK_ID_CURR'),
  ('applications', 'SK_ID_CURR',    'cash_balance',   'SK_ID_CURR'),
  ('applications', 'SK_ID_CURR',    'payments',       'SK_ID_CURR'),
  ('applications', 'SK_ID_CURR',    'card_balance',   'SK_ID_CURR')
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
dfs_feat.head()
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
