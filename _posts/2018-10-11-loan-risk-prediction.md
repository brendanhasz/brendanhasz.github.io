---
layout: post
title: "Home Credit Group Loan Risk Prediction"
date: 2018-10-11
description: "Prediction of loan default using python, scikit-learn, and XGBoost."
github_url: https://github.com/brendanhasz/home-credit-group
kaggle_url: https://www.kaggle.com/brendanhasz/home-credit-group-loan-risk-prediction
img_url: /assets/img/loan-risk-prediction/output_44_1.svg
tags: [python, prediction]
comments: true
---

[Home Credit Group](http://www.homecredit.net/) is a financial institution which specializes in consumer lending, especially to people with little credit history.  In order to determine what a reasonable principal is for applicants, and a repayment schedule which will help their clients sucessfully repay their loans, Home Credit Group wants to use data about the applicant to predict how likely they are to be able to repay their loan.  Home Credit Group recently hosted a [kaggle competition](https://www.kaggle.com/c/home-credit-default-risk) to predict loan repayment probability from (anonymized) applicant information.  In this post we'll use that data to try and predict loan repayment ability.

## Outline

* [Data Loading and Cleaning](#data-loading-and-cleaning)
* [Manual Feature Engineering](#manual-feature-engineering)
* [Feature Encoding](#feature-encoding)
* [Baseline Predictions](#baseline-predictions)
* [Calibration](#calibration)
* [Resampling](#resampling)
* [Final Predictions and Feature Importance](#feature-importance)

First let's load the packages we'll use.


```python
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve, roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
from xgboost import plot_importance
from hashlib import sha256
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# Plot settings
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
sns.set()
```

<a class="anchor" id="data-loading-and-cleaning"></a>
## Data Loading and Cleaning

Let's load both the training and test data.


```python
# Load applications data
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
```

And now we can take a look at the data we're working with.  


```python
train.head()
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
      <th>LIVINGAPARTMENTS_MEDI</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
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
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0205</td>
      <td>0.0193</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
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
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0787</td>
      <td>0.0558</td>
      <td>0.0039</td>
      <td>0.01</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
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
      <td>MONDAY</td>
      <td>9</td>
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
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
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
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
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
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
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
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print info about each column in the train dataset
for col in train:
    print(col)
    Nnan = train[col].isnull().sum()
    print('Number empty: ', Nnan)
    print('Percent empty: ', 100*Nnan/train.shape[0])
    print(train[col].describe())
    if train[col].dtype==object:
        print('Categories and Count:')
        print(train[col].value_counts().to_string(header=None))
    print()
```

<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>SK_ID_CURR
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean     278180.518577
    std      102790.175348
    min      100002.000000
    25%      189145.500000
    50%      278202.000000
    75%      367142.500000
    max      456255.000000
    Name: SK_ID_CURR, dtype: float64
    
    TARGET
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.080729
    std           0.272419
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: TARGET, dtype: float64
    
    NAME_CONTRACT_TYPE
    Number empty:  0
    Percent empty:  0.0
    count         307511
    unique             2
    top       Cash loans
    freq          278232
    Name: NAME_CONTRACT_TYPE, dtype: object
    Categories and Count:
    Cash loans         278232
    Revolving loans     29279
    
    CODE_GENDER
    Number empty:  0
    Percent empty:  0.0
    count     307511
    unique         3
    top            F
    freq      202448
    Name: CODE_GENDER, dtype: object
    Categories and Count:
    F      202448
    M      105059
    XNA         4
    
    FLAG_OWN_CAR
    Number empty:  0
    Percent empty:  0.0
    count     307511
    unique         2
    top            N
    freq      202924
    Name: FLAG_OWN_CAR, dtype: object
    Categories and Count:
    N    202924
    Y    104587
    
    FLAG_OWN_REALTY
    Number empty:  0
    Percent empty:  0.0
    count     307511
    unique         2
    top            Y
    freq      213312
    Name: FLAG_OWN_REALTY, dtype: object
    Categories and Count:
    Y    213312
    N     94199
    
    CNT_CHILDREN
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.417052
    std           0.722121
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           1.000000
    max          19.000000
    Name: CNT_CHILDREN, dtype: float64
    
    AMT_INCOME_TOTAL
    Number empty:  0
    Percent empty:  0.0
    count    3.075110e+05
    mean     1.687979e+05
    std      2.371231e+05
    min      2.565000e+04
    25%      1.125000e+05
    50%      1.471500e+05
    75%      2.025000e+05
    max      1.170000e+08
    Name: AMT_INCOME_TOTAL, dtype: float64
    
    AMT_CREDIT
    Number empty:  0
    Percent empty:  0.0
    count    3.075110e+05
    mean     5.990260e+05
    std      4.024908e+05
    min      4.500000e+04
    25%      2.700000e+05
    50%      5.135310e+05
    75%      8.086500e+05
    max      4.050000e+06
    Name: AMT_CREDIT, dtype: float64
    
    AMT_ANNUITY
    Number empty:  12
    Percent empty:  0.0039022994299390914
    count    307499.000000
    mean      27108.573909
    std       14493.737315
    min        1615.500000
    25%       16524.000000
    50%       24903.000000
    75%       34596.000000
    max      258025.500000
    Name: AMT_ANNUITY, dtype: float64
    
    AMT_GOODS_PRICE
    Number empty:  278
    Percent empty:  0.09040327012692229
    count    3.072330e+05
    mean     5.383962e+05
    std      3.694465e+05
    min      4.050000e+04
    25%      2.385000e+05
    50%      4.500000e+05
    75%      6.795000e+05
    max      4.050000e+06
    Name: AMT_GOODS_PRICE, dtype: float64
    
    NAME_TYPE_SUITE
    Number empty:  1292
    Percent empty:  0.42014757195677555
    count            306219
    unique                7
    top       Unaccompanied
    freq             248526
    Name: NAME_TYPE_SUITE, dtype: object
    Categories and Count:
    Unaccompanied      248526
    Family              40149
    Spouse, partner     11370
    Children             3267
    Other_B              1770
    Other_A               866
    Group of people       271
    
    NAME_INCOME_TYPE
    Number empty:  0
    Percent empty:  0.0
    count      307511
    unique          8
    top       Working
    freq       158774
    Name: NAME_INCOME_TYPE, dtype: object
    Categories and Count:
    Working                 158774
    Commercial associate     71617
    Pensioner                55362
    State servant            21703
    Unemployed                  22
    Student                     18
    Businessman                 10
    Maternity leave              5
    
    NAME_EDUCATION_TYPE
    Number empty:  0
    Percent empty:  0.0
    count                            307511
    unique                                5
    top       Secondary / secondary special
    freq                             218391
    Name: NAME_EDUCATION_TYPE, dtype: object
    Categories and Count:
    Secondary / secondary special    218391
    Higher education                  74863
    Incomplete higher                 10277
    Lower secondary                    3816
    Academic degree                     164
    
    NAME_FAMILY_STATUS
    Number empty:  0
    Percent empty:  0.0
    count      307511
    unique          6
    top       Married
    freq       196432
    Name: NAME_FAMILY_STATUS, dtype: object
    Categories and Count:
    Married                 196432
    Single / not married     45444
    Civil marriage           29775
    Separated                19770
    Widow                    16088
    Unknown                      2
    
    NAME_HOUSING_TYPE
    Number empty:  0
    Percent empty:  0.0
    count                307511
    unique                    6
    top       House / apartment
    freq                 272868
    Name: NAME_HOUSING_TYPE, dtype: object
    Categories and Count:
    House / apartment      272868
    With parents            14840
    Municipal apartment     11183
    Rented apartment         4881
    Office apartment         2617
    Co-op apartment          1122
    
    REGION_POPULATION_RELATIVE
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.020868
    std           0.013831
    min           0.000290
    25%           0.010006
    50%           0.018850
    75%           0.028663
    max           0.072508
    Name: REGION_POPULATION_RELATIVE, dtype: float64
    
    DAYS_BIRTH
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean     -16036.995067
    std        4363.988632
    min      -25229.000000
    25%      -19682.000000
    50%      -15750.000000
    75%      -12413.000000
    max       -7489.000000
    Name: DAYS_BIRTH, dtype: float64
    
    DAYS_EMPLOYED
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean      63815.045904
    std      141275.766519
    min      -17912.000000
    25%       -2760.000000
    50%       -1213.000000
    75%        -289.000000
    max      365243.000000
    Name: DAYS_EMPLOYED, dtype: float64
    
    DAYS_REGISTRATION
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean      -4986.120328
    std        3522.886321
    min      -24672.000000
    25%       -7479.500000
    50%       -4504.000000
    75%       -2010.000000
    max           0.000000
    Name: DAYS_REGISTRATION, dtype: float64
    
    DAYS_ID_PUBLISH
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean      -2994.202373
    std        1509.450419
    min       -7197.000000
    25%       -4299.000000
    50%       -3254.000000
    75%       -1720.000000
    max           0.000000
    Name: DAYS_ID_PUBLISH, dtype: float64
    
    OWN_CAR_AGE
    Number empty:  202929
    Percent empty:  65.9908100848425
    count    104582.000000
    mean         12.061091
    std          11.944812
    min           0.000000
    25%           5.000000
    50%           9.000000
    75%          15.000000
    max          91.000000
    Name: OWN_CAR_AGE, dtype: float64
    
    FLAG_MOBIL
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.999997
    std           0.001803
    min           0.000000
    25%           1.000000
    50%           1.000000
    75%           1.000000
    max           1.000000
    Name: FLAG_MOBIL, dtype: float64
    
    FLAG_EMP_PHONE
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.819889
    std           0.384280
    min           0.000000
    25%           1.000000
    50%           1.000000
    75%           1.000000
    max           1.000000
    Name: FLAG_EMP_PHONE, dtype: float64
    
    FLAG_WORK_PHONE
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.199368
    std           0.399526
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_WORK_PHONE, dtype: float64
    
    FLAG_CONT_MOBILE
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.998133
    std           0.043164
    min           0.000000
    25%           1.000000
    50%           1.000000
    75%           1.000000
    max           1.000000
    Name: FLAG_CONT_MOBILE, dtype: float64
    
    FLAG_PHONE
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.281066
    std           0.449521
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           1.000000
    max           1.000000
    Name: FLAG_PHONE, dtype: float64
    
    FLAG_EMAIL
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.056720
    std           0.231307
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_EMAIL, dtype: float64
    
    OCCUPATION_TYPE
    Number empty:  96391
    Percent empty:  31.345545362604916
    count       211120
    unique          18
    top       Laborers
    freq         55186
    Name: OCCUPATION_TYPE, dtype: object
    Categories and Count:
    Laborers                 55186
    Sales staff              32102
    Core staff               27570
    Managers                 21371
    Drivers                  18603
    High skill tech staff    11380
    Accountants               9813
    Medicine staff            8537
    Security staff            6721
    Cooking staff             5946
    Cleaning staff            4653
    Private service staff     2652
    Low-skill Laborers        2093
    Waiters/barmen staff      1348
    Secretaries               1305
    Realty agents              751
    HR staff                   563
    IT staff                   526
    
    CNT_FAM_MEMBERS
    Number empty:  2
    Percent empty:  0.000650383238323182
    count    307509.000000
    mean          2.152665
    std           0.910682
    min           1.000000
    25%           2.000000
    50%           2.000000
    75%           3.000000
    max          20.000000
    Name: CNT_FAM_MEMBERS, dtype: float64
    
    REGION_RATING_CLIENT
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          2.052463
    std           0.509034
    min           1.000000
    25%           2.000000
    50%           2.000000
    75%           2.000000
    max           3.000000
    Name: REGION_RATING_CLIENT, dtype: float64
    
    REGION_RATING_CLIENT_W_CITY
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          2.031521
    std           0.502737
    min           1.000000
    25%           2.000000
    50%           2.000000
    75%           2.000000
    max           3.000000
    Name: REGION_RATING_CLIENT_W_CITY, dtype: float64
    
    WEEKDAY_APPR_PROCESS_START
    Number empty:  0
    Percent empty:  0.0
    count      307511
    unique          7
    top       TUESDAY
    freq        53901
    Name: WEEKDAY_APPR_PROCESS_START, dtype: object
    Categories and Count:
    TUESDAY      53901
    WEDNESDAY    51934
    MONDAY       50714
    THURSDAY     50591
    FRIDAY       50338
    SATURDAY     33852
    SUNDAY       16181
    
    HOUR_APPR_PROCESS_START
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean         12.063419
    std           3.265832
    min           0.000000
    25%          10.000000
    50%          12.000000
    75%          14.000000
    max          23.000000
    Name: HOUR_APPR_PROCESS_START, dtype: float64
    
    REG_REGION_NOT_LIVE_REGION
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.015144
    std           0.122126
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: REG_REGION_NOT_LIVE_REGION, dtype: float64
    
    REG_REGION_NOT_WORK_REGION
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.050769
    std           0.219526
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: REG_REGION_NOT_WORK_REGION, dtype: float64
    
    LIVE_REGION_NOT_WORK_REGION
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.040659
    std           0.197499
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: LIVE_REGION_NOT_WORK_REGION, dtype: float64
    
    REG_CITY_NOT_LIVE_CITY
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.078173
    std           0.268444
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: REG_CITY_NOT_LIVE_CITY, dtype: float64
    
    REG_CITY_NOT_WORK_CITY
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.230454
    std           0.421124
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: REG_CITY_NOT_WORK_CITY, dtype: float64
    
    LIVE_CITY_NOT_WORK_CITY
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.179555
    std           0.383817
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: LIVE_CITY_NOT_WORK_CITY, dtype: float64
    
    ORGANIZATION_TYPE
    Number empty:  0
    Percent empty:  0.0
    count                     307511
    unique                        58
    top       Business Entity Type 3
    freq                       67992
    Name: ORGANIZATION_TYPE, dtype: object
    Categories and Count:
    Business Entity Type 3    67992
    XNA                       55374
    Self-employed             38412
    Other                     16683
    Medicine                  11193
    Business Entity Type 2    10553
    Government                10404
    School                     8893
    Trade: type 7              7831
    Kindergarten               6880
    Construction               6721
    Business Entity Type 1     5984
    Transport: type 4          5398
    Trade: type 3              3492
    Industry: type 9           3368
    Industry: type 3           3278
    Security                   3247
    Housing                    2958
    Industry: type 11          2704
    Military                   2634
    Bank                       2507
    Agriculture                2454
    Police                     2341
    Transport: type 2          2204
    Postal                     2157
    Security Ministries        1974
    Trade: type 2              1900
    Restaurant                 1811
    Services                   1575
    University                 1327
    Industry: type 7           1307
    Transport: type 3          1187
    Industry: type 1           1039
    Hotel                       966
    Electricity                 950
    Industry: type 4            877
    Trade: type 6               631
    Industry: type 5            599
    Insurance                   597
    Telecom                     577
    Emergency                   560
    Industry: type 2            458
    Advertising                 429
    Realtor                     396
    Culture                     379
    Industry: type 12           369
    Trade: type 1               348
    Mobile                      317
    Legal Services              305
    Cleaning                    260
    Transport: type 1           201
    Industry: type 6            112
    Industry: type 10           109
    Religion                     85
    Industry: type 13            67
    Trade: type 4                64
    Trade: type 5                49
    Industry: type 8             24
    
    EXT_SOURCE_1
    Number empty:  173378
    Percent empty:  56.38107254699832
    count    134133.000000
    mean          0.502130
    std           0.211062
    min           0.014568
    25%           0.334007
    50%           0.505998
    75%           0.675053
    max           0.962693
    Name: EXT_SOURCE_1, dtype: float64
    
    EXT_SOURCE_2
    Number empty:  660
    Percent empty:  0.21462646864665003
    count    3.068510e+05
    mean     5.143927e-01
    std      1.910602e-01
    min      8.173617e-08
    25%      3.924574e-01
    50%      5.659614e-01
    75%      6.636171e-01
    max      8.549997e-01
    Name: EXT_SOURCE_2, dtype: float64
    
    EXT_SOURCE_3
    Number empty:  60965
    Percent empty:  19.825307062186393
    count    246546.000000
    mean          0.510853
    std           0.194844
    min           0.000527
    25%           0.370650
    50%           0.535276
    75%           0.669057
    max           0.896010
    Name: EXT_SOURCE_3, dtype: float64
    
    APARTMENTS_AVG
    Number empty:  156061
    Percent empty:  50.749729277977046
    count    151450.00000
    mean          0.11744
    std           0.10824
    min           0.00000
    25%           0.05770
    50%           0.08760
    75%           0.14850
    max           1.00000
    Name: APARTMENTS_AVG, dtype: float64
    
    BASEMENTAREA_AVG
    Number empty:  179943
    Percent empty:  58.515955526794166
    count    127568.000000
    mean          0.088442
    std           0.082438
    min           0.000000
    25%           0.044200
    50%           0.076300
    75%           0.112200
    max           1.000000
    Name: BASEMENTAREA_AVG, dtype: float64
    
    YEARS_BEGINEXPLUATATION_AVG
    Number empty:  150007
    Percent empty:  48.781019215572776
    count    157504.000000
    mean          0.977735
    std           0.059223
    min           0.000000
    25%           0.976700
    50%           0.981600
    75%           0.986600
    max           1.000000
    Name: YEARS_BEGINEXPLUATATION_AVG, dtype: float64
    
    YEARS_BUILD_AVG
    Number empty:  204488
    Percent empty:  66.49778381911541
    count    103023.000000
    mean          0.752471
    std           0.113280
    min           0.000000
    25%           0.687200
    50%           0.755200
    75%           0.823200
    max           1.000000
    Name: YEARS_BUILD_AVG, dtype: float64
    
    COMMONAREA_AVG
    Number empty:  214865
    Percent empty:  69.87229725115525
    count    92646.000000
    mean         0.044621
    std          0.076036
    min          0.000000
    25%          0.007800
    50%          0.021100
    75%          0.051500
    max          1.000000
    Name: COMMONAREA_AVG, dtype: float64
    
    ELEVATORS_AVG
    Number empty:  163891
    Percent empty:  53.29597965601231
    count    143620.000000
    mean          0.078942
    std           0.134576
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.120000
    max           1.000000
    Name: ELEVATORS_AVG, dtype: float64
    
    ENTRANCES_AVG
    Number empty:  154828
    Percent empty:  50.34876801155081
    count    152683.000000
    mean          0.149725
    std           0.100049
    min           0.000000
    25%           0.069000
    50%           0.137900
    75%           0.206900
    max           1.000000
    Name: ENTRANCES_AVG, dtype: float64
    
    FLOORSMAX_AVG
    Number empty:  153020
    Percent empty:  49.76082156410665
    count    154491.000000
    mean          0.226282
    std           0.144641
    min           0.000000
    25%           0.166700
    50%           0.166700
    75%           0.333300
    max           1.000000
    Name: FLOORSMAX_AVG, dtype: float64
    
    FLOORSMIN_AVG
    Number empty:  208642
    Percent empty:  67.84862980511267
    count    98869.000000
    mean         0.231894
    std          0.161380
    min          0.000000
    25%          0.083300
    50%          0.208300
    75%          0.375000
    max          1.000000
    Name: FLOORSMIN_AVG, dtype: float64
    
    LANDAREA_AVG
    Number empty:  182590
    Percent empty:  59.376737742714894
    count    124921.000000
    mean          0.066333
    std           0.081184
    min           0.000000
    25%           0.018700
    50%           0.048100
    75%           0.085600
    max           1.000000
    Name: LANDAREA_AVG, dtype: float64
    
    LIVINGAPARTMENTS_AVG
    Number empty:  210199
    Percent empty:  68.35495315614726
    count    97312.000000
    mean         0.100775
    std          0.092576
    min          0.000000
    25%          0.050400
    50%          0.075600
    75%          0.121000
    max          1.000000
    Name: LIVINGAPARTMENTS_AVG, dtype: float64
    
    LIVINGAREA_AVG
    Number empty:  154350
    Percent empty:  50.193326417591564
    count    153161.000000
    mean          0.107399
    std           0.110565
    min           0.000000
    25%           0.045300
    50%           0.074500
    75%           0.129900
    max           1.000000
    Name: LIVINGAREA_AVG, dtype: float64
    
    NONLIVINGAPARTMENTS_AVG
    Number empty:  213514
    Percent empty:  69.43296337366793
    count    93997.000000
    mean         0.008809
    std          0.047732
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.003900
    max          1.000000
    Name: NONLIVINGAPARTMENTS_AVG, dtype: float64
    
    NONLIVINGAREA_AVG
    Number empty:  169682
    Percent empty:  55.17916432257708
    count    137829.000000
    mean          0.028358
    std           0.069523
    min           0.000000
    25%           0.000000
    50%           0.003600
    75%           0.027700
    max           1.000000
    Name: NONLIVINGAREA_AVG, dtype: float64
    
    APARTMENTS_MODE
    Number empty:  156061
    Percent empty:  50.749729277977046
    count    151450.000000
    mean          0.114231
    std           0.107936
    min           0.000000
    25%           0.052500
    50%           0.084000
    75%           0.143900
    max           1.000000
    Name: APARTMENTS_MODE, dtype: float64
    
    BASEMENTAREA_MODE
    Number empty:  179943
    Percent empty:  58.515955526794166
    count    127568.000000
    mean          0.087543
    std           0.084307
    min           0.000000
    25%           0.040700
    50%           0.074600
    75%           0.112400
    max           1.000000
    Name: BASEMENTAREA_MODE, dtype: float64
    
    YEARS_BEGINEXPLUATATION_MODE
    Number empty:  150007
    Percent empty:  48.781019215572776
    count    157504.000000
    mean          0.977065
    std           0.064575
    min           0.000000
    25%           0.976700
    50%           0.981600
    75%           0.986600
    max           1.000000
    Name: YEARS_BEGINEXPLUATATION_MODE, dtype: float64
    
    YEARS_BUILD_MODE
    Number empty:  204488
    Percent empty:  66.49778381911541
    count    103023.000000
    mean          0.759637
    std           0.110111
    min           0.000000
    25%           0.699400
    50%           0.764800
    75%           0.823600
    max           1.000000
    Name: YEARS_BUILD_MODE, dtype: float64
    
    COMMONAREA_MODE
    Number empty:  214865
    Percent empty:  69.87229725115525
    count    92646.000000
    mean         0.042553
    std          0.074445
    min          0.000000
    25%          0.007200
    50%          0.019000
    75%          0.049000
    max          1.000000
    Name: COMMONAREA_MODE, dtype: float64
    
    ELEVATORS_MODE
    Number empty:  163891
    Percent empty:  53.29597965601231
    count    143620.000000
    mean          0.074490
    std           0.132256
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.120800
    max           1.000000
    Name: ELEVATORS_MODE, dtype: float64
    
    ENTRANCES_MODE
    Number empty:  154828
    Percent empty:  50.34876801155081
    count    152683.000000
    mean          0.145193
    std           0.100977
    min           0.000000
    25%           0.069000
    50%           0.137900
    75%           0.206900
    max           1.000000
    Name: ENTRANCES_MODE, dtype: float64
    
    FLOORSMAX_MODE
    Number empty:  153020
    Percent empty:  49.76082156410665
    count    154491.000000
    mean          0.222315
    std           0.143709
    min           0.000000
    25%           0.166700
    50%           0.166700
    75%           0.333300
    max           1.000000
    Name: FLOORSMAX_MODE, dtype: float64
    
    FLOORSMIN_MODE
    Number empty:  208642
    Percent empty:  67.84862980511267
    count    98869.000000
    mean         0.228058
    std          0.161160
    min          0.000000
    25%          0.083300
    50%          0.208300
    75%          0.375000
    max          1.000000
    Name: FLOORSMIN_MODE, dtype: float64
    
    LANDAREA_MODE
    Number empty:  182590
    Percent empty:  59.376737742714894
    count    124921.000000
    mean          0.064958
    std           0.081750
    min           0.000000
    25%           0.016600
    50%           0.045800
    75%           0.084100
    max           1.000000
    Name: LANDAREA_MODE, dtype: float64
    
    LIVINGAPARTMENTS_MODE
    Number empty:  210199
    Percent empty:  68.35495315614726
    count    97312.000000
    mean         0.105645
    std          0.097880
    min          0.000000
    25%          0.054200
    50%          0.077100
    75%          0.131300
    max          1.000000
    Name: LIVINGAPARTMENTS_MODE, dtype: float64
    
    LIVINGAREA_MODE
    Number empty:  154350
    Percent empty:  50.193326417591564
    count    153161.000000
    mean          0.105975
    std           0.111845
    min           0.000000
    25%           0.042700
    50%           0.073100
    75%           0.125200
    max           1.000000
    Name: LIVINGAREA_MODE, dtype: float64
    
    NONLIVINGAPARTMENTS_MODE
    Number empty:  213514
    Percent empty:  69.43296337366793
    count    93997.000000
    mean         0.008076
    std          0.046276
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.003900
    max          1.000000
    Name: NONLIVINGAPARTMENTS_MODE, dtype: float64
    
    NONLIVINGAREA_MODE
    Number empty:  169682
    Percent empty:  55.17916432257708
    count    137829.000000
    mean          0.027022
    std           0.070254
    min           0.000000
    25%           0.000000
    50%           0.001100
    75%           0.023100
    max           1.000000
    Name: NONLIVINGAREA_MODE, dtype: float64
    
    APARTMENTS_MEDI
    Number empty:  156061
    Percent empty:  50.749729277977046
    count    151450.000000
    mean          0.117850
    std           0.109076
    min           0.000000
    25%           0.058300
    50%           0.086400
    75%           0.148900
    max           1.000000
    Name: APARTMENTS_MEDI, dtype: float64
    
    BASEMENTAREA_MEDI
    Number empty:  179943
    Percent empty:  58.515955526794166
    count    127568.000000
    mean          0.087955
    std           0.082179
    min           0.000000
    25%           0.043700
    50%           0.075800
    75%           0.111600
    max           1.000000
    Name: BASEMENTAREA_MEDI, dtype: float64
    
    YEARS_BEGINEXPLUATATION_MEDI
    Number empty:  150007
    Percent empty:  48.781019215572776
    count    157504.000000
    mean          0.977752
    std           0.059897
    min           0.000000
    25%           0.976700
    50%           0.981600
    75%           0.986600
    max           1.000000
    Name: YEARS_BEGINEXPLUATATION_MEDI, dtype: float64
    
    YEARS_BUILD_MEDI
    Number empty:  204488
    Percent empty:  66.49778381911541
    count    103023.000000
    mean          0.755746
    std           0.112066
    min           0.000000
    25%           0.691400
    50%           0.758500
    75%           0.825600
    max           1.000000
    Name: YEARS_BUILD_MEDI, dtype: float64
    
    COMMONAREA_MEDI
    Number empty:  214865
    Percent empty:  69.87229725115525
    count    92646.000000
    mean         0.044595
    std          0.076144
    min          0.000000
    25%          0.007900
    50%          0.020800
    75%          0.051300
    max          1.000000
    Name: COMMONAREA_MEDI, dtype: float64
    
    ELEVATORS_MEDI
    Number empty:  163891
    Percent empty:  53.29597965601231
    count    143620.000000
    mean          0.078078
    std           0.134467
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.120000
    max           1.000000
    Name: ELEVATORS_MEDI, dtype: float64
    
    ENTRANCES_MEDI
    Number empty:  154828
    Percent empty:  50.34876801155081
    count    152683.000000
    mean          0.149213
    std           0.100368
    min           0.000000
    25%           0.069000
    50%           0.137900
    75%           0.206900
    max           1.000000
    Name: ENTRANCES_MEDI, dtype: float64
    
    FLOORSMAX_MEDI
    Number empty:  153020
    Percent empty:  49.76082156410665
    count    154491.000000
    mean          0.225897
    std           0.145067
    min           0.000000
    25%           0.166700
    50%           0.166700
    75%           0.333300
    max           1.000000
    Name: FLOORSMAX_MEDI, dtype: float64
    
    FLOORSMIN_MEDI
    Number empty:  208642
    Percent empty:  67.84862980511267
    count    98869.000000
    mean         0.231625
    std          0.161934
    min          0.000000
    25%          0.083300
    50%          0.208300
    75%          0.375000
    max          1.000000
    Name: FLOORSMIN_MEDI, dtype: float64
    
    LANDAREA_MEDI
    Number empty:  182590
    Percent empty:  59.376737742714894
    count    124921.000000
    mean          0.067169
    std           0.082167
    min           0.000000
    25%           0.018700
    50%           0.048700
    75%           0.086800
    max           1.000000
    Name: LANDAREA_MEDI, dtype: float64
    
    LIVINGAPARTMENTS_MEDI
    Number empty:  210199
    Percent empty:  68.35495315614726
    count    97312.000000
    mean         0.101954
    std          0.093642
    min          0.000000
    25%          0.051300
    50%          0.076100
    75%          0.123100
    max          1.000000
    Name: LIVINGAPARTMENTS_MEDI, dtype: float64
    
    LIVINGAREA_MEDI
    Number empty:  154350
    Percent empty:  50.193326417591564
    count    153161.000000
    mean          0.108607
    std           0.112260
    min           0.000000
    25%           0.045700
    50%           0.074900
    75%           0.130300
    max           1.000000
    Name: LIVINGAREA_MEDI, dtype: float64
    
    NONLIVINGAPARTMENTS_MEDI
    Number empty:  213514
    Percent empty:  69.43296337366793
    count    93997.000000
    mean         0.008651
    std          0.047415
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.003900
    max          1.000000
    Name: NONLIVINGAPARTMENTS_MEDI, dtype: float64
    
    NONLIVINGAREA_MEDI
    Number empty:  169682
    Percent empty:  55.17916432257708
    count    137829.000000
    mean          0.028236
    std           0.070166
    min           0.000000
    25%           0.000000
    50%           0.003100
    75%           0.026600
    max           1.000000
    Name: NONLIVINGAREA_MEDI, dtype: float64
    
    FONDKAPREMONT_MODE
    Number empty:  210295
    Percent empty:  68.38617155158677
    count                97216
    unique                   4
    top       reg oper account
    freq                 73830
    Name: FONDKAPREMONT_MODE, dtype: object
    Categories and Count:
    reg oper account         73830
    reg oper spec account    12080
    not specified             5687
    org spec account          5619
    
    HOUSETYPE_MODE
    Number empty:  154297
    Percent empty:  50.176091261776
    count             153214
    unique                 3
    top       block of flats
    freq              150503
    Name: HOUSETYPE_MODE, dtype: object
    Categories and Count:
    block of flats      150503
    specific housing      1499
    terraced house        1212
    
    TOTALAREA_MODE
    Number empty:  148431
    Percent empty:  48.26851722377411
    count    159080.000000
    mean          0.102547
    std           0.107462
    min           0.000000
    25%           0.041200
    50%           0.068800
    75%           0.127600
    max           1.000000
    Name: TOTALAREA_MODE, dtype: float64
    
    WALLSMATERIAL_MODE
    Number empty:  156341
    Percent empty:  50.8407829313423
    count     151170
    unique         7
    top        Panel
    freq       66040
    Name: WALLSMATERIAL_MODE, dtype: object
    Categories and Count:
    Panel           66040
    Stone, brick    64815
    Block            9253
    Wooden           5362
    Mixed            2296
    Monolithic       1779
    Others           1625
    
    EMERGENCYSTATE_MODE
    Number empty:  145755
    Percent empty:  47.39830445089769
    count     161756
    unique         2
    top           No
    freq      159428
    Name: EMERGENCYSTATE_MODE, dtype: object
    Categories and Count:
    No     159428
    Yes      2328
    
    OBS_30_CNT_SOCIAL_CIRCLE
    Number empty:  1021
    Percent empty:  0.3320206431639844
    count    306490.000000
    mean          1.422245
    std           2.400989
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           2.000000
    max         348.000000
    Name: OBS_30_CNT_SOCIAL_CIRCLE, dtype: float64
    
    DEF_30_CNT_SOCIAL_CIRCLE
    Number empty:  1021
    Percent empty:  0.3320206431639844
    count    306490.000000
    mean          0.143421
    std           0.446698
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max          34.000000
    Name: DEF_30_CNT_SOCIAL_CIRCLE, dtype: float64
    
    OBS_60_CNT_SOCIAL_CIRCLE
    Number empty:  1021
    Percent empty:  0.3320206431639844
    count    306490.000000
    mean          1.405292
    std           2.379803
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           2.000000
    max         344.000000
    Name: OBS_60_CNT_SOCIAL_CIRCLE, dtype: float64
    
    DEF_60_CNT_SOCIAL_CIRCLE
    Number empty:  1021
    Percent empty:  0.3320206431639844
    count    306490.000000
    mean          0.100049
    std           0.362291
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max          24.000000
    Name: DEF_60_CNT_SOCIAL_CIRCLE, dtype: float64
    
    DAYS_LAST_PHONE_CHANGE
    Number empty:  1
    Percent empty:  0.000325191619161591
    count    307510.000000
    mean       -962.858788
    std         826.808487
    min       -4292.000000
    25%       -1570.000000
    50%        -757.000000
    75%        -274.000000
    max           0.000000
    Name: DAYS_LAST_PHONE_CHANGE, dtype: float64
    
    FLAG_DOCUMENT_2
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000042
    std           0.006502
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_2, dtype: float64
    
    FLAG_DOCUMENT_3
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.710023
    std           0.453752
    min           0.000000
    25%           0.000000
    50%           1.000000
    75%           1.000000
    max           1.000000
    Name: FLAG_DOCUMENT_3, dtype: float64
    
    FLAG_DOCUMENT_4
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000081
    std           0.009016
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_4, dtype: float64
    
    FLAG_DOCUMENT_5
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.015115
    std           0.122010
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_5, dtype: float64
    
    FLAG_DOCUMENT_6
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.088055
    std           0.283376
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_6, dtype: float64
    
    FLAG_DOCUMENT_7
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000192
    std           0.013850
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_7, dtype: float64
    
    FLAG_DOCUMENT_8
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.081376
    std           0.273412
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_8, dtype: float64
    
    FLAG_DOCUMENT_9
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.003896
    std           0.062295
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_9, dtype: float64
    
    FLAG_DOCUMENT_10
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000023
    std           0.004771
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_10, dtype: float64
    
    FLAG_DOCUMENT_11
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.003912
    std           0.062424
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_11, dtype: float64
    
    FLAG_DOCUMENT_12
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000007
    std           0.002550
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_12, dtype: float64
    
    FLAG_DOCUMENT_13
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.003525
    std           0.059268
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_13, dtype: float64
    
    FLAG_DOCUMENT_14
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.002936
    std           0.054110
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_14, dtype: float64
    
    FLAG_DOCUMENT_15
    Number empty:  0
    Percent empty:  0.0
    count    307511.00000
    mean          0.00121
    std           0.03476
    min           0.00000
    25%           0.00000
    50%           0.00000
    75%           0.00000
    max           1.00000
    Name: FLAG_DOCUMENT_15, dtype: float64
    
    FLAG_DOCUMENT_16
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.009928
    std           0.099144
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_16, dtype: float64
    
    FLAG_DOCUMENT_17
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000267
    std           0.016327
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_17, dtype: float64
    
    FLAG_DOCUMENT_18
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.008130
    std           0.089798
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_18, dtype: float64
    
    FLAG_DOCUMENT_19
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000595
    std           0.024387
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_19, dtype: float64
    
    FLAG_DOCUMENT_20
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000507
    std           0.022518
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_20, dtype: float64
    
    FLAG_DOCUMENT_21
    Number empty:  0
    Percent empty:  0.0
    count    307511.000000
    mean          0.000335
    std           0.018299
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           1.000000
    Name: FLAG_DOCUMENT_21, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_HOUR
    Number empty:  41519
    Percent empty:  13.501630835970095
    count    265992.000000
    mean          0.006402
    std           0.083849
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           4.000000
    Name: AMT_REQ_CREDIT_BUREAU_HOUR, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_DAY
    Number empty:  41519
    Percent empty:  13.501630835970095
    count    265992.000000
    mean          0.007000
    std           0.110757
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           9.000000
    Name: AMT_REQ_CREDIT_BUREAU_DAY, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_WEEK
    Number empty:  41519
    Percent empty:  13.501630835970095
    count    265992.000000
    mean          0.034362
    std           0.204685
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max           8.000000
    Name: AMT_REQ_CREDIT_BUREAU_WEEK, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_MON
    Number empty:  41519
    Percent empty:  13.501630835970095
    count    265992.000000
    mean          0.267395
    std           0.916002
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max          27.000000
    Name: AMT_REQ_CREDIT_BUREAU_MON, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_QRT
    Number empty:  41519
    Percent empty:  13.501630835970095
    count    265992.000000
    mean          0.265474
    std           0.794056
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           0.000000
    max         261.000000
    Name: AMT_REQ_CREDIT_BUREAU_QRT, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_YEAR
    Number empty:  41519
    Percent empty:  13.501630835970095
    count    265992.000000
    mean          1.899974
    std           1.869295
    min           0.000000
    25%           0.000000
    50%           1.000000
    75%           3.000000
    max          25.000000
    Name: AMT_REQ_CREDIT_BUREAU_YEAR, dtype: float64
    
  </code>
  </pre>
  </div>
</div>
    
<br />

```python
# Print info about each column in the test dataset
for col in test:
    print(col)
    Nnan = test[col].isnull().sum()
    print('Number empty: ', Nnan)
    print('Percent empty: ', 100*Nnan/test.shape[0])
    print(test[col].describe())
    if test[col].dtype==object:
        print('Categories and Count:')
        print(test[col].value_counts().to_string(header=None))
    print()
```

<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>SK_ID_CURR
    Number empty:  0
    Percent empty:  0.0
    count     48744.000000
    mean     277796.676350
    std      103169.547296
    min      100001.000000
    25%      188557.750000
    50%      277549.000000
    75%      367555.500000
    max      456250.000000
    Name: SK_ID_CURR, dtype: float64
    
    NAME_CONTRACT_TYPE
    Number empty:  0
    Percent empty:  0.0
    count          48744
    unique             2
    top       Cash loans
    freq           48305
    Name: NAME_CONTRACT_TYPE, dtype: object
    Categories and Count:
    Cash loans         48305
    Revolving loans      439
    
    CODE_GENDER
    Number empty:  0
    Percent empty:  0.0
    count     48744
    unique        2
    top           F
    freq      32678
    Name: CODE_GENDER, dtype: object
    Categories and Count:
    F    32678
    M    16066
    
    FLAG_OWN_CAR
    Number empty:  0
    Percent empty:  0.0
    count     48744
    unique        2
    top           N
    freq      32311
    Name: FLAG_OWN_CAR, dtype: object
    Categories and Count:
    N    32311
    Y    16433
    
    FLAG_OWN_REALTY
    Number empty:  0
    Percent empty:  0.0
    count     48744
    unique        2
    top           Y
    freq      33658
    Name: FLAG_OWN_REALTY, dtype: object
    Categories and Count:
    Y    33658
    N    15086
    
    CNT_CHILDREN
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.397054
    std          0.709047
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          1.000000
    max         20.000000
    Name: CNT_CHILDREN, dtype: float64
    
    AMT_INCOME_TOTAL
    Number empty:  0
    Percent empty:  0.0
    count    4.874400e+04
    mean     1.784318e+05
    std      1.015226e+05
    min      2.694150e+04
    25%      1.125000e+05
    50%      1.575000e+05
    75%      2.250000e+05
    max      4.410000e+06
    Name: AMT_INCOME_TOTAL, dtype: float64
    
    AMT_CREDIT
    Number empty:  0
    Percent empty:  0.0
    count    4.874400e+04
    mean     5.167404e+05
    std      3.653970e+05
    min      4.500000e+04
    25%      2.606400e+05
    50%      4.500000e+05
    75%      6.750000e+05
    max      2.245500e+06
    Name: AMT_CREDIT, dtype: float64
    
    AMT_ANNUITY
    Number empty:  24
    Percent empty:  0.049236829148202856
    count     48720.000000
    mean      29426.240209
    std       16016.368315
    min        2295.000000
    25%       17973.000000
    50%       26199.000000
    75%       37390.500000
    max      180576.000000
    Name: AMT_ANNUITY, dtype: float64
    
    AMT_GOODS_PRICE
    Number empty:  0
    Percent empty:  0.0
    count    4.874400e+04
    mean     4.626188e+05
    std      3.367102e+05
    min      4.500000e+04
    25%      2.250000e+05
    50%      3.960000e+05
    75%      6.300000e+05
    max      2.245500e+06
    Name: AMT_GOODS_PRICE, dtype: float64
    
    NAME_TYPE_SUITE
    Number empty:  911
    Percent empty:  1.8689479730838667
    count             47833
    unique                7
    top       Unaccompanied
    freq              39727
    Name: NAME_TYPE_SUITE, dtype: object
    Categories and Count:
    Unaccompanied      39727
    Family              5881
    Spouse, partner     1448
    Children             408
    Other_B              211
    Other_A              109
    Group of people       49
    
    NAME_INCOME_TYPE
    Number empty:  0
    Percent empty:  0.0
    count       48744
    unique          7
    top       Working
    freq        24533
    Name: NAME_INCOME_TYPE, dtype: object
    Categories and Count:
    Working                 24533
    Commercial associate    11402
    Pensioner                9273
    State servant            3532
    Student                     2
    Businessman                 1
    Unemployed                  1
    
    NAME_EDUCATION_TYPE
    Number empty:  0
    Percent empty:  0.0
    count                             48744
    unique                                5
    top       Secondary / secondary special
    freq                              33988
    Name: NAME_EDUCATION_TYPE, dtype: object
    Categories and Count:
    Secondary / secondary special    33988
    Higher education                 12516
    Incomplete higher                 1724
    Lower secondary                    475
    Academic degree                     41
    
    NAME_FAMILY_STATUS
    Number empty:  0
    Percent empty:  0.0
    count       48744
    unique          5
    top       Married
    freq        32283
    Name: NAME_FAMILY_STATUS, dtype: object
    Categories and Count:
    Married                 32283
    Single / not married     7036
    Civil marriage           4261
    Separated                2955
    Widow                    2209
    
    NAME_HOUSING_TYPE
    Number empty:  0
    Percent empty:  0.0
    count                 48744
    unique                    6
    top       House / apartment
    freq                  43645
    Name: NAME_HOUSING_TYPE, dtype: object
    Categories and Count:
    House / apartment      43645
    With parents            2234
    Municipal apartment     1617
    Rented apartment         718
    Office apartment         407
    Co-op apartment          123
    
    REGION_POPULATION_RELATIVE
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.021226
    std          0.014428
    min          0.000253
    25%          0.010006
    50%          0.018850
    75%          0.028663
    max          0.072508
    Name: REGION_POPULATION_RELATIVE, dtype: float64
    
    DAYS_BIRTH
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean    -16068.084605
    std       4325.900393
    min     -25195.000000
    25%     -19637.000000
    50%     -15785.000000
    75%     -12496.000000
    max      -7338.000000
    Name: DAYS_BIRTH, dtype: float64
    
    DAYS_EMPLOYED
    Number empty:  0
    Percent empty:  0.0
    count     48744.000000
    mean      67485.366322
    std      144348.507136
    min      -17463.000000
    25%       -2910.000000
    50%       -1293.000000
    75%        -296.000000
    max      365243.000000
    Name: DAYS_EMPLOYED, dtype: float64
    
    DAYS_REGISTRATION
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean     -4967.652716
    std       3552.612035
    min     -23722.000000
    25%      -7459.250000
    50%      -4490.000000
    75%      -1901.000000
    max          0.000000
    Name: DAYS_REGISTRATION, dtype: float64
    
    DAYS_ID_PUBLISH
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean     -3051.712949
    std       1569.276709
    min      -6348.000000
    25%      -4448.000000
    50%      -3234.000000
    75%      -1706.000000
    max          0.000000
    Name: DAYS_ID_PUBLISH, dtype: float64
    
    OWN_CAR_AGE
    Number empty:  32312
    Percent empty:  66.28918430986378
    count    16432.000000
    mean        11.786027
    std         11.462889
    min          0.000000
    25%          4.000000
    50%          9.000000
    75%         15.000000
    max         74.000000
    Name: OWN_CAR_AGE, dtype: float64
    
    FLAG_MOBIL
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.999979
    std          0.004529
    min          0.000000
    25%          1.000000
    50%          1.000000
    75%          1.000000
    max          1.000000
    Name: FLAG_MOBIL, dtype: float64
    
    FLAG_EMP_PHONE
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.809720
    std          0.392526
    min          0.000000
    25%          1.000000
    50%          1.000000
    75%          1.000000
    max          1.000000
    Name: FLAG_EMP_PHONE, dtype: float64
    
    FLAG_WORK_PHONE
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.204702
    std          0.403488
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_WORK_PHONE, dtype: float64
    
    FLAG_CONT_MOBILE
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.998400
    std          0.039971
    min          0.000000
    25%          1.000000
    50%          1.000000
    75%          1.000000
    max          1.000000
    Name: FLAG_CONT_MOBILE, dtype: float64
    
    FLAG_PHONE
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.263130
    std          0.440337
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          1.000000
    max          1.000000
    Name: FLAG_PHONE, dtype: float64
    
    FLAG_EMAIL
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.162646
    std          0.369046
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_EMAIL, dtype: float64
    
    OCCUPATION_TYPE
    Number empty:  15605
    Percent empty:  32.014196619071065
    count        33139
    unique          18
    top       Laborers
    freq          8655
    Name: OCCUPATION_TYPE, dtype: object
    Categories and Count:
    Laborers                 8655
    Sales staff              5072
    Core staff               4361
    Managers                 3574
    Drivers                  2773
    High skill tech staff    1854
    Accountants              1628
    Medicine staff           1316
    Security staff            915
    Cooking staff             894
    Cleaning staff            656
    Private service staff     455
    Low-skill Laborers        272
    Secretaries               213
    Waiters/barmen staff      178
    Realty agents             138
    HR staff                  104
    IT staff                   81
    
    CNT_FAM_MEMBERS
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         2.146767
    std          0.890423
    min          1.000000
    25%          2.000000
    50%          2.000000
    75%          3.000000
    max         21.000000
    Name: CNT_FAM_MEMBERS, dtype: float64
    
    REGION_RATING_CLIENT
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         2.038159
    std          0.522694
    min          1.000000
    25%          2.000000
    50%          2.000000
    75%          2.000000
    max          3.000000
    Name: REGION_RATING_CLIENT, dtype: float64
    
    REGION_RATING_CLIENT_W_CITY
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         2.012596
    std          0.515804
    min         -1.000000
    25%          2.000000
    50%          2.000000
    75%          2.000000
    max          3.000000
    Name: REGION_RATING_CLIENT_W_CITY, dtype: float64
    
    WEEKDAY_APPR_PROCESS_START
    Number empty:  0
    Percent empty:  0.0
    count       48744
    unique          7
    top       TUESDAY
    freq         9751
    Name: WEEKDAY_APPR_PROCESS_START, dtype: object
    Categories and Count:
    TUESDAY      9751
    WEDNESDAY    8457
    THURSDAY     8418
    MONDAY       8406
    FRIDAY       7250
    SATURDAY     4603
    SUNDAY       1859
    
    HOUR_APPR_PROCESS_START
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean        12.007365
    std          3.278172
    min          0.000000
    25%         10.000000
    50%         12.000000
    75%         14.000000
    max         23.000000
    Name: HOUR_APPR_PROCESS_START, dtype: float64
    
    REG_REGION_NOT_LIVE_REGION
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.018833
    std          0.135937
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: REG_REGION_NOT_LIVE_REGION, dtype: float64
    
    REG_REGION_NOT_WORK_REGION
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.055166
    std          0.228306
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: REG_REGION_NOT_WORK_REGION, dtype: float64
    
    LIVE_REGION_NOT_WORK_REGION
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.042036
    std          0.200673
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: LIVE_REGION_NOT_WORK_REGION, dtype: float64
    
    REG_CITY_NOT_LIVE_CITY
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.077466
    std          0.267332
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: REG_CITY_NOT_LIVE_CITY, dtype: float64
    
    REG_CITY_NOT_WORK_CITY
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.224664
    std          0.417365
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: REG_CITY_NOT_WORK_CITY, dtype: float64
    
    LIVE_CITY_NOT_WORK_CITY
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.174216
    std          0.379299
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: LIVE_CITY_NOT_WORK_CITY, dtype: float64
    
    ORGANIZATION_TYPE
    Number empty:  0
    Percent empty:  0.0
    count                      48744
    unique                        58
    top       Business Entity Type 3
    freq                       10840
    Name: ORGANIZATION_TYPE, dtype: object
    Categories and Count:
    Business Entity Type 3    10840
    XNA                        9274
    Self-employed              5920
    Other                      2707
    Medicine                   1716
    Government                 1508
    Business Entity Type 2     1479
    Trade: type 7              1303
    School                     1287
    Construction               1039
    Kindergarten               1038
    Business Entity Type 1      887
    Transport: type 4           884
    Trade: type 3               578
    Military                    530
    Industry: type 9            499
    Industry: type 3            489
    Security                    472
    Transport: type 2           448
    Police                      441
    Housing                     435
    Industry: type 11           416
    Bank                        374
    Security Ministries         341
    Services                    302
    Postal                      294
    Agriculture                 292
    Restaurant                  284
    Trade: type 2               242
    University                  221
    Industry: type 7            217
    Industry: type 1            178
    Transport: type 3           174
    Industry: type 4            167
    Electricity                 156
    Hotel                       134
    Trade: type 6               122
    Industry: type 5             97
    Telecom                      95
    Emergency                    91
    Insurance                    80
    Industry: type 2             77
    Industry: type 12            77
    Realtor                      72
    Advertising                  71
    Trade: type 1                64
    Culture                      61
    Legal Services               53
    Mobile                       45
    Cleaning                     43
    Transport: type 1            35
    Industry: type 6             27
    Industry: type 10            24
    Trade: type 4                14
    Religion                     12
    Trade: type 5                 9
    Industry: type 13             6
    Industry: type 8              3
    
    EXT_SOURCE_1
    Number empty:  20532
    Percent empty:  42.12210733628754
    count    28212.000000
    mean         0.501180
    std          0.205142
    min          0.013458
    25%          0.343695
    50%          0.506771
    75%          0.665956
    max          0.939145
    Name: EXT_SOURCE_1, dtype: float64
    
    EXT_SOURCE_2
    Number empty:  8
    Percent empty:  0.016412276382734285
    count    48736.000000
    mean         0.518021
    std          0.181278
    min          0.000008
    25%          0.408066
    50%          0.558758
    75%          0.658497
    max          0.855000
    Name: EXT_SOURCE_2, dtype: float64
    
    EXT_SOURCE_3
    Number empty:  8668
    Percent empty:  17.782701460692596
    count    40076.000000
    mean         0.500106
    std          0.189498
    min          0.000527
    25%          0.363945
    50%          0.519097
    75%          0.652897
    max          0.882530
    Name: EXT_SOURCE_3, dtype: float64
    
    APARTMENTS_AVG
    Number empty:  23887
    Percent empty:  49.00500574429673
    count    24857.000000
    mean         0.122388
    std          0.113112
    min          0.000000
    25%          0.061900
    50%          0.092800
    75%          0.148500
    max          1.000000
    Name: APARTMENTS_AVG, dtype: float64
    
    BASEMENTAREA_AVG
    Number empty:  27641
    Percent empty:  56.7064664368948
    count    21103.000000
    mean         0.090065
    std          0.081536
    min          0.000000
    25%          0.046700
    50%          0.078100
    75%          0.113400
    max          1.000000
    Name: BASEMENTAREA_AVG, dtype: float64
    
    YEARS_BEGINEXPLUATATION_AVG
    Number empty:  22856
    Percent empty:  46.88987362547185
    count    25888.000000
    mean         0.978828
    std          0.049318
    min          0.000000
    25%          0.976700
    50%          0.981600
    75%          0.986600
    max          1.000000
    Name: YEARS_BEGINEXPLUATATION_AVG, dtype: float64
    
    YEARS_BUILD_AVG
    Number empty:  31818
    Percent empty:  65.27572624322994
    count    16926.000000
    mean         0.751137
    std          0.113188
    min          0.000000
    25%          0.687200
    50%          0.755200
    75%          0.816400
    max          1.000000
    Name: YEARS_BUILD_AVG, dtype: float64
    
    COMMONAREA_AVG
    Number empty:  33495
    Percent empty:  68.71614967996061
    count    15249.000000
    mean         0.047624
    std          0.082868
    min          0.000000
    25%          0.008100
    50%          0.022700
    75%          0.053900
    max          1.000000
    Name: COMMONAREA_AVG, dtype: float64
    
    ELEVATORS_AVG
    Number empty:  25189
    Percent empty:  51.67610372558674
    count    23555.000000
    mean         0.085168
    std          0.139164
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.160000
    max          1.000000
    Name: ELEVATORS_AVG, dtype: float64
    
    ENTRANCES_AVG
    Number empty:  23579
    Percent empty:  48.373133103561464
    count    25165.000000
    mean         0.151777
    std          0.100669
    min          0.000000
    25%          0.074500
    50%          0.137900
    75%          0.206900
    max          1.000000
    Name: ENTRANCES_AVG, dtype: float64
    
    FLOORSMAX_AVG
    Number empty:  23321
    Percent empty:  47.84383719021828
    count    25423.000000
    mean         0.233706
    std          0.147361
    min          0.000000
    25%          0.166700
    50%          0.166700
    75%          0.333300
    max          1.000000
    Name: FLOORSMAX_AVG, dtype: float64
    
    FLOORSMIN_AVG
    Number empty:  32466
    Percent empty:  66.60512063023141
    count    16278.000000
    mean         0.238423
    std          0.164976
    min          0.000000
    25%          0.104200
    50%          0.208300
    75%          0.375000
    max          1.000000
    Name: FLOORSMIN_AVG, dtype: float64
    
    LANDAREA_AVG
    Number empty:  28254
    Percent empty:  57.96405711472181
    count    20490.000000
    mean         0.067192
    std          0.081909
    min          0.000000
    25%          0.019000
    50%          0.048300
    75%          0.086800
    max          1.000000
    Name: LANDAREA_AVG, dtype: float64
    
    LIVINGAPARTMENTS_AVG
    Number empty:  32780
    Percent empty:  67.24930247825374
    count    15964.000000
    mean         0.105885
    std          0.098284
    min          0.000000
    25%          0.050400
    50%          0.075600
    75%          0.126900
    max          1.000000
    Name: LIVINGAPARTMENTS_AVG, dtype: float64
    
    LIVINGAREA_AVG
    Number empty:  23552
    Percent empty:  48.317741670769735
    count    25192.000000
    mean         0.112286
    std          0.114860
    min          0.000000
    25%          0.048575
    50%          0.077000
    75%          0.137600
    max          1.000000
    Name: LIVINGAREA_AVG, dtype: float64
    
    NONLIVINGAPARTMENTS_AVG
    Number empty:  33347
    Percent empty:  68.41252256688003
    count    15397.000000
    mean         0.009231
    std          0.048749
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.005100
    max          1.000000
    Name: NONLIVINGAPARTMENTS_AVG, dtype: float64
    
    NONLIVINGAREA_AVG
    Number empty:  26084
    Percent empty:  53.512227145905136
    count    22660.000000
    mean         0.029387
    std          0.072007
    min          0.000000
    25%          0.000000
    50%          0.003800
    75%          0.029000
    max          1.000000
    Name: NONLIVINGAREA_AVG, dtype: float64
    
    APARTMENTS_MODE
    Number empty:  23887
    Percent empty:  49.00500574429673
    count    24857.000000
    mean         0.119078
    std          0.113465
    min          0.000000
    25%          0.058800
    50%          0.085100
    75%          0.150200
    max          1.000000
    Name: APARTMENTS_MODE, dtype: float64
    
    BASEMENTAREA_MODE
    Number empty:  27641
    Percent empty:  56.7064664368948
    count    21103.000000
    mean         0.088998
    std          0.082655
    min          0.000000
    25%          0.042500
    50%          0.077000
    75%          0.113550
    max          1.000000
    Name: BASEMENTAREA_MODE, dtype: float64
    
    YEARS_BEGINEXPLUATATION_MODE
    Number empty:  22856
    Percent empty:  46.88987362547185
    count    25888.000000
    mean         0.978292
    std          0.053782
    min          0.000000
    25%          0.976200
    50%          0.981600
    75%          0.986600
    max          1.000000
    Name: YEARS_BEGINEXPLUATATION_MODE, dtype: float64
    
    YEARS_BUILD_MODE
    Number empty:  31818
    Percent empty:  65.27572624322994
    count    16926.000000
    mean         0.758327
    std          0.110117
    min          0.000000
    25%          0.692900
    50%          0.758300
    75%          0.823600
    max          1.000000
    Name: YEARS_BUILD_MODE, dtype: float64
    
    COMMONAREA_MODE
    Number empty:  33495
    Percent empty:  68.71614967996061
    count    15249.000000
    mean         0.045223
    std          0.081169
    min          0.000000
    25%          0.007600
    50%          0.020300
    75%          0.051700
    max          1.000000
    Name: COMMONAREA_MODE, dtype: float64
    
    ELEVATORS_MODE
    Number empty:  25189
    Percent empty:  51.67610372558674
    count    23555.000000
    mean         0.080570
    std          0.137509
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.120800
    max          1.000000
    Name: ELEVATORS_MODE, dtype: float64
    
    ENTRANCES_MODE
    Number empty:  23579
    Percent empty:  48.373133103561464
    count    25165.000000
    mean         0.147161
    std          0.101748
    min          0.000000
    25%          0.069000
    50%          0.137900
    75%          0.206900
    max          1.000000
    Name: ENTRANCES_MODE, dtype: float64
    
    FLOORSMAX_MODE
    Number empty:  23321
    Percent empty:  47.84383719021828
    count    25423.000000
    mean         0.229390
    std          0.146485
    min          0.000000
    25%          0.166700
    50%          0.166700
    75%          0.333300
    max          1.000000
    Name: FLOORSMAX_MODE, dtype: float64
    
    FLOORSMIN_MODE
    Number empty:  32466
    Percent empty:  66.60512063023141
    count    16278.000000
    mean         0.233854
    std          0.165034
    min          0.000000
    25%          0.083300
    50%          0.208300
    75%          0.375000
    max          1.000000
    Name: FLOORSMIN_MODE, dtype: float64
    
    LANDAREA_MODE
    Number empty:  28254
    Percent empty:  57.96405711472181
    count    20490.000000
    mean         0.065914
    std          0.082880
    min          0.000000
    25%          0.016525
    50%          0.046200
    75%          0.085600
    max          1.000000
    Name: LANDAREA_MODE, dtype: float64
    
    LIVINGAPARTMENTS_MODE
    Number empty:  32780
    Percent empty:  67.24930247825374
    count    15964.000000
    mean         0.110874
    std          0.103980
    min          0.000000
    25%          0.055100
    50%          0.081700
    75%          0.132200
    max          1.000000
    Name: LIVINGAPARTMENTS_MODE, dtype: float64
    
    LIVINGAREA_MODE
    Number empty:  23552
    Percent empty:  48.317741670769735
    count    25192.000000
    mean         0.110687
    std          0.116699
    min          0.000000
    25%          0.045600
    50%          0.075100
    75%          0.130600
    max          1.000000
    Name: LIVINGAREA_MODE, dtype: float64
    
    NONLIVINGAPARTMENTS_MODE
    Number empty:  33347
    Percent empty:  68.41252256688003
    count    15397.000000
    mean         0.008358
    std          0.046657
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.003900
    max          1.000000
    Name: NONLIVINGAPARTMENTS_MODE, dtype: float64
    
    NONLIVINGAREA_MODE
    Number empty:  26084
    Percent empty:  53.512227145905136
    count    22660.000000
    mean         0.028161
    std          0.073504
    min          0.000000
    25%          0.000000
    50%          0.001200
    75%          0.024500
    max          1.000000
    Name: NONLIVINGAREA_MODE, dtype: float64
    
    APARTMENTS_MEDI
    Number empty:  23887
    Percent empty:  49.00500574429673
    count    24857.000000
    mean         0.122809
    std          0.114184
    min          0.000000
    25%          0.062500
    50%          0.092600
    75%          0.149900
    max          1.000000
    Name: APARTMENTS_MEDI, dtype: float64
    
    BASEMENTAREA_MEDI
    Number empty:  27641
    Percent empty:  56.7064664368948
    count    21103.000000
    mean         0.089529
    std          0.081022
    min          0.000000
    25%          0.046150
    50%          0.077800
    75%          0.113000
    max          1.000000
    Name: BASEMENTAREA_MEDI, dtype: float64
    
    YEARS_BEGINEXPLUATATION_MEDI
    Number empty:  22856
    Percent empty:  46.88987362547185
    count    25888.000000
    mean         0.978822
    std          0.049663
    min          0.000000
    25%          0.976700
    50%          0.981600
    75%          0.986600
    max          1.000000
    Name: YEARS_BEGINEXPLUATATION_MEDI, dtype: float64
    
    YEARS_BUILD_MEDI
    Number empty:  31818
    Percent empty:  65.27572624322994
    count    16926.000000
    mean         0.754344
    std          0.111998
    min          0.000000
    25%          0.691400
    50%          0.758500
    75%          0.818900
    max          1.000000
    Name: YEARS_BUILD_MEDI, dtype: float64
    
    COMMONAREA_MEDI
    Number empty:  33495
    Percent empty:  68.71614967996061
    count    15249.000000
    mean         0.047420
    std          0.082892
    min          0.000000
    25%          0.008000
    50%          0.022300
    75%          0.053800
    max          1.000000
    Name: COMMONAREA_MEDI, dtype: float64
    
    ELEVATORS_MEDI
    Number empty:  25189
    Percent empty:  51.67610372558674
    count    23555.000000
    mean         0.084128
    std          0.139014
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.160000
    max          1.000000
    Name: ELEVATORS_MEDI, dtype: float64
    
    ENTRANCES_MEDI
    Number empty:  23579
    Percent empty:  48.373133103561464
    count    25165.000000
    mean         0.151200
    std          0.100931
    min          0.000000
    25%          0.069000
    50%          0.137900
    75%          0.206900
    max          1.000000
    Name: ENTRANCES_MEDI, dtype: float64
    
    FLOORSMAX_MEDI
    Number empty:  23321
    Percent empty:  47.84383719021828
    count    25423.000000
    mean         0.233154
    std          0.147629
    min          0.000000
    25%          0.166700
    50%          0.166700
    75%          0.333300
    max          1.000000
    Name: FLOORSMAX_MEDI, dtype: float64
    
    FLOORSMIN_MEDI
    Number empty:  32466
    Percent empty:  66.60512063023141
    count    16278.000000
    mean         0.237846
    std          0.165241
    min          0.000000
    25%          0.083300
    50%          0.208300
    75%          0.375000
    max          1.000000
    Name: FLOORSMIN_MEDI, dtype: float64
    
    LANDAREA_MEDI
    Number empty:  28254
    Percent empty:  57.96405711472181
    count    20490.000000
    mean         0.068069
    std          0.082869
    min          0.000000
    25%          0.019000
    50%          0.048800
    75%          0.088000
    max          1.000000
    Name: LANDAREA_MEDI, dtype: float64
    
    LIVINGAPARTMENTS_MEDI
    Number empty:  32780
    Percent empty:  67.24930247825374
    count    15964.000000
    mean         0.107063
    std          0.099737
    min          0.000000
    25%          0.051300
    50%          0.077000
    75%          0.126600
    max          1.000000
    Name: LIVINGAPARTMENTS_MEDI, dtype: float64
    
    LIVINGAREA_MEDI
    Number empty:  23552
    Percent empty:  48.317741670769735
    count    25192.000000
    mean         0.113368
    std          0.116503
    min          0.000000
    25%          0.049000
    50%          0.077600
    75%          0.137425
    max          1.000000
    Name: LIVINGAREA_MEDI, dtype: float64
    
    NONLIVINGAPARTMENTS_MEDI
    Number empty:  33347
    Percent empty:  68.41252256688003
    count    15397.000000
    mean         0.008979
    std          0.048148
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.003900
    max          1.000000
    Name: NONLIVINGAPARTMENTS_MEDI, dtype: float64
    
    NONLIVINGAREA_MEDI
    Number empty:  26084
    Percent empty:  53.512227145905136
    count    22660.000000
    mean         0.029296
    std          0.072998
    min          0.000000
    25%          0.000000
    50%          0.003100
    75%          0.028025
    max          1.000000
    Name: NONLIVINGAREA_MEDI, dtype: float64
    
    FONDKAPREMONT_MODE
    Number empty:  32797
    Percent empty:  67.28417856556705
    count                15947
    unique                   4
    top       reg oper account
    freq                 12124
    Name: FONDKAPREMONT_MODE, dtype: object
    Categories and Count:
    reg oper account         12124
    reg oper spec account     1990
    org spec account           920
    not specified              913
    
    HOUSETYPE_MODE
    Number empty:  23619
    Percent empty:  48.45519448547513
    count              25125
    unique                 3
    top       block of flats
    freq               24659
    Name: HOUSETYPE_MODE, dtype: object
    Categories and Count:
    block of flats      24659
    specific housing      262
    terraced house        204
    
    TOTALAREA_MODE
    Number empty:  22624
    Percent empty:  46.41391761037256
    count    26120.000000
    mean         0.107129
    std          0.111420
    min          0.000000
    25%          0.043200
    50%          0.070700
    75%          0.135700
    max          1.000000
    Name: TOTALAREA_MODE, dtype: float64
    
    WALLSMATERIAL_MODE
    Number empty:  23893
    Percent empty:  49.017314951583785
    count     24851
    unique        7
    top       Panel
    freq      11269
    Name: WALLSMATERIAL_MODE, dtype: object
    Categories and Count:
    Panel           11269
    Stone, brick    10434
    Block            1428
    Wooden            794
    Mixed             353
    Monolithic        289
    Others            284
    
    EMERGENCYSTATE_MODE
    Number empty:  22209
    Percent empty:  45.56253077301822
    count     26535
    unique        2
    top          No
    freq      26179
    Name: EMERGENCYSTATE_MODE, dtype: object
    Categories and Count:
    No     26179
    Yes      356
    
    OBS_30_CNT_SOCIAL_CIRCLE
    Number empty:  29
    Percent empty:  0.05949450188741178
    count    48715.000000
    mean         1.447644
    std          3.608053
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          2.000000
    max        354.000000
    Name: OBS_30_CNT_SOCIAL_CIRCLE, dtype: float64
    
    DEF_30_CNT_SOCIAL_CIRCLE
    Number empty:  29
    Percent empty:  0.05949450188741178
    count    48715.000000
    mean         0.143652
    std          0.514413
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max         34.000000
    Name: DEF_30_CNT_SOCIAL_CIRCLE, dtype: float64
    
    OBS_60_CNT_SOCIAL_CIRCLE
    Number empty:  29
    Percent empty:  0.05949450188741178
    count    48715.000000
    mean         1.435738
    std          3.580125
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          2.000000
    max        351.000000
    Name: OBS_60_CNT_SOCIAL_CIRCLE, dtype: float64
    
    DEF_60_CNT_SOCIAL_CIRCLE
    Number empty:  29
    Percent empty:  0.05949450188741178
    count    48715.000000
    mean         0.101139
    std          0.403791
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max         24.000000
    Name: DEF_60_CNT_SOCIAL_CIRCLE, dtype: float64
    
    DAYS_LAST_PHONE_CHANGE
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean     -1077.766228
    std        878.920740
    min      -4361.000000
    25%      -1766.250000
    50%       -863.000000
    75%       -363.000000
    max          0.000000
    Name: DAYS_LAST_PHONE_CHANGE, dtype: float64
    
    FLAG_DOCUMENT_2
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_2, dtype: float64
    
    FLAG_DOCUMENT_3
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.786620
    std          0.409698
    min          0.000000
    25%          1.000000
    50%          1.000000
    75%          1.000000
    max          1.000000
    Name: FLAG_DOCUMENT_3, dtype: float64
    
    FLAG_DOCUMENT_4
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.000103
    std          0.010128
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_4, dtype: float64
    
    FLAG_DOCUMENT_5
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.014751
    std          0.120554
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_5, dtype: float64
    
    FLAG_DOCUMENT_6
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.087477
    std          0.282536
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_6, dtype: float64
    
    FLAG_DOCUMENT_7
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.000041
    std          0.006405
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_7, dtype: float64
    
    FLAG_DOCUMENT_8
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.088462
    std          0.283969
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_8, dtype: float64
    
    FLAG_DOCUMENT_9
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.004493
    std          0.066879
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_9, dtype: float64
    
    FLAG_DOCUMENT_10
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_10, dtype: float64
    
    FLAG_DOCUMENT_11
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.001169
    std          0.034176
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_11, dtype: float64
    
    FLAG_DOCUMENT_12
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_12, dtype: float64
    
    FLAG_DOCUMENT_13
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_13, dtype: float64
    
    FLAG_DOCUMENT_14
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_14, dtype: float64
    
    FLAG_DOCUMENT_15
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_15, dtype: float64
    
    FLAG_DOCUMENT_16
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_16, dtype: float64
    
    FLAG_DOCUMENT_17
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_17, dtype: float64
    
    FLAG_DOCUMENT_18
    Number empty:  0
    Percent empty:  0.0
    count    48744.000000
    mean         0.001559
    std          0.039456
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          1.000000
    Name: FLAG_DOCUMENT_18, dtype: float64
    
    FLAG_DOCUMENT_19
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_19, dtype: float64
    
    FLAG_DOCUMENT_20
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_20, dtype: float64
    
    FLAG_DOCUMENT_21
    Number empty:  0
    Percent empty:  0.0
    count    48744.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: FLAG_DOCUMENT_21, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_HOUR
    Number empty:  6049
    Percent empty:  12.409732479894961
    count    42695.000000
    mean         0.002108
    std          0.046373
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          2.000000
    Name: AMT_REQ_CREDIT_BUREAU_HOUR, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_DAY
    Number empty:  6049
    Percent empty:  12.409732479894961
    count    42695.000000
    mean         0.001803
    std          0.046132
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          2.000000
    Name: AMT_REQ_CREDIT_BUREAU_DAY, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_WEEK
    Number empty:  6049
    Percent empty:  12.409732479894961
    count    42695.000000
    mean         0.002787
    std          0.054037
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          2.000000
    Name: AMT_REQ_CREDIT_BUREAU_WEEK, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_MON
    Number empty:  6049
    Percent empty:  12.409732479894961
    count    42695.000000
    mean         0.009299
    std          0.110924
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          6.000000
    Name: AMT_REQ_CREDIT_BUREAU_MON, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_QRT
    Number empty:  6049
    Percent empty:  12.409732479894961
    count    42695.000000
    mean         0.546902
    std          0.693305
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          1.000000
    max          7.000000
    Name: AMT_REQ_CREDIT_BUREAU_QRT, dtype: float64
    
    AMT_REQ_CREDIT_BUREAU_YEAR
    Number empty:  6049
    Percent empty:  12.409732479894961
    count    42695.000000
    mean         1.983769
    std          1.838873
    min          0.000000
    25%          0.000000
    50%          2.000000
    75%          3.000000
    max         17.000000
    Name: AMT_REQ_CREDIT_BUREAU_YEAR, dtype: float64
    
  </code>
  </pre>
  </div>
</div>    

<br />

The column containing the values we are trying to predict, `TARGET`, doesn't contain any missing values.  The value of `TARGET` is 0 when the loan was repayed sucessfully, and 1 when there were problems repaying the loan.  Many more loans were succesfully repayed than not, which means that the dataset is imbalanced in terms of our dependent variable, which is something we'll have to watch out for when we build a predictive model later:


```python
# Show target distribution
train['TARGET'].value_counts()
```

    0    282686
    1     24825
    Name: TARGET, dtype: int64



There's a lot of categorical columns - let's check that, for each column, all the categories we see in the training set we also see in the test set, and vice-versa.


```python
for col in test:
    if test[col].dtype==object:
        print(col)
        print('Num Unique in Train:', train[col].nunique())
        print('Num Unique in Test: ', test[col].nunique())
        print('Unique in Train:', sorted([str(e) for e in train[col].unique().tolist()]))
        print('Unique in Test: ', sorted([str(e) for e in test[col].unique().tolist()]))
        print()
```

<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>NAME_CONTRACT_TYPE
    Num Unique in Train: 2
    Num Unique in Test:  2
    Unique in Train: ['Cash loans', 'Revolving loans']
    Unique in Test:  ['Cash loans', 'Revolving loans']
    
    CODE_GENDER
    Num Unique in Train: 3
    Num Unique in Test:  2
    Unique in Train: ['F', 'M', 'XNA']
    Unique in Test:  ['F', 'M']
    
    FLAG_OWN_CAR
    Num Unique in Train: 2
    Num Unique in Test:  2
    Unique in Train: ['N', 'Y']
    Unique in Test:  ['N', 'Y']
    
    FLAG_OWN_REALTY
    Num Unique in Train: 2
    Num Unique in Test:  2
    Unique in Train: ['N', 'Y']
    Unique in Test:  ['N', 'Y']
    
    NAME_TYPE_SUITE
    Num Unique in Train: 7
    Num Unique in Test:  7
    Unique in Train: ['Children', 'Family', 'Group of people', 'Other_A', 'Other_B', 'Spouse, partner', 'Unaccompanied', 'nan']
    Unique in Test:  ['Children', 'Family', 'Group of people', 'Other_A', 'Other_B', 'Spouse, partner', 'Unaccompanied', 'nan']
    
    NAME_INCOME_TYPE
    Num Unique in Train: 8
    Num Unique in Test:  7
    Unique in Train: ['Businessman', 'Commercial associate', 'Maternity leave', 'Pensioner', 'State servant', 'Student', 'Unemployed', 'Working']
    Unique in Test:  ['Businessman', 'Commercial associate', 'Pensioner', 'State servant', 'Student', 'Unemployed', 'Working']
    
    NAME_EDUCATION_TYPE
    Num Unique in Train: 5
    Num Unique in Test:  5
    Unique in Train: ['Academic degree', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Secondary / secondary special']
    Unique in Test:  ['Academic degree', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Secondary / secondary special']
    
    NAME_FAMILY_STATUS
    Num Unique in Train: 6
    Num Unique in Test:  5
    Unique in Train: ['Civil marriage', 'Married', 'Separated', 'Single / not married', 'Unknown', 'Widow']
    Unique in Test:  ['Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow']
    
    NAME_HOUSING_TYPE
    Num Unique in Train: 6
    Num Unique in Test:  6
    Unique in Train: ['Co-op apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents']
    Unique in Test:  ['Co-op apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents']
    
    OCCUPATION_TYPE
    Num Unique in Train: 18
    Num Unique in Test:  18
    Unique in Train: ['Accountants', 'Cleaning staff', 'Cooking staff', 'Core staff', 'Drivers', 'HR staff', 'High skill tech staff', 'IT staff', 'Laborers', 'Low-skill Laborers', 'Managers', 'Medicine staff', 'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries', 'Security staff', 'Waiters/barmen staff', 'nan']
    Unique in Test:  ['Accountants', 'Cleaning staff', 'Cooking staff', 'Core staff', 'Drivers', 'HR staff', 'High skill tech staff', 'IT staff', 'Laborers', 'Low-skill Laborers', 'Managers', 'Medicine staff', 'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries', 'Security staff', 'Waiters/barmen staff', 'nan']
    
    WEEKDAY_APPR_PROCESS_START
    Num Unique in Train: 7
    Num Unique in Test:  7
    Unique in Train: ['FRIDAY', 'MONDAY', 'SATURDAY', 'SUNDAY', 'THURSDAY', 'TUESDAY', 'WEDNESDAY']
    Unique in Test:  ['FRIDAY', 'MONDAY', 'SATURDAY', 'SUNDAY', 'THURSDAY', 'TUESDAY', 'WEDNESDAY']
    
    ORGANIZATION_TYPE
    Num Unique in Train: 58
    Num Unique in Test:  58
    Unique in Train: ['Advertising', 'Agriculture', 'Bank', 'Business Entity Type 1', 'Business Entity Type 2', 'Business Entity Type 3', 'Cleaning', 'Construction', 'Culture', 'Electricity', 'Emergency', 'Government', 'Hotel', 'Housing', 'Industry: type 1', 'Industry: type 10', 'Industry: type 11', 'Industry: type 12', 'Industry: type 13', 'Industry: type 2', 'Industry: type 3', 'Industry: type 4', 'Industry: type 5', 'Industry: type 6', 'Industry: type 7', 'Industry: type 8', 'Industry: type 9', 'Insurance', 'Kindergarten', 'Legal Services', 'Medicine', 'Military', 'Mobile', 'Other', 'Police', 'Postal', 'Realtor', 'Religion', 'Restaurant', 'School', 'Security', 'Security Ministries', 'Self-employed', 'Services', 'Telecom', 'Trade: type 1', 'Trade: type 2', 'Trade: type 3', 'Trade: type 4', 'Trade: type 5', 'Trade: type 6', 'Trade: type 7', 'Transport: type 1', 'Transport: type 2', 'Transport: type 3', 'Transport: type 4', 'University', 'XNA']
    Unique in Test:  ['Advertising', 'Agriculture', 'Bank', 'Business Entity Type 1', 'Business Entity Type 2', 'Business Entity Type 3', 'Cleaning', 'Construction', 'Culture', 'Electricity', 'Emergency', 'Government', 'Hotel', 'Housing', 'Industry: type 1', 'Industry: type 10', 'Industry: type 11', 'Industry: type 12', 'Industry: type 13', 'Industry: type 2', 'Industry: type 3', 'Industry: type 4', 'Industry: type 5', 'Industry: type 6', 'Industry: type 7', 'Industry: type 8', 'Industry: type 9', 'Insurance', 'Kindergarten', 'Legal Services', 'Medicine', 'Military', 'Mobile', 'Other', 'Police', 'Postal', 'Realtor', 'Religion', 'Restaurant', 'School', 'Security', 'Security Ministries', 'Self-employed', 'Services', 'Telecom', 'Trade: type 1', 'Trade: type 2', 'Trade: type 3', 'Trade: type 4', 'Trade: type 5', 'Trade: type 6', 'Trade: type 7', 'Transport: type 1', 'Transport: type 2', 'Transport: type 3', 'Transport: type 4', 'University', 'XNA']
    
    FONDKAPREMONT_MODE
    Num Unique in Train: 4
    Num Unique in Test:  4
    Unique in Train: ['nan', 'not specified', 'org spec account', 'reg oper account', 'reg oper spec account']
    Unique in Test:  ['nan', 'not specified', 'org spec account', 'reg oper account', 'reg oper spec account']
    
    HOUSETYPE_MODE
    Num Unique in Train: 3
    Num Unique in Test:  3
    Unique in Train: ['block of flats', 'nan', 'specific housing', 'terraced house']
    Unique in Test:  ['block of flats', 'nan', 'specific housing', 'terraced house']
    
    WALLSMATERIAL_MODE
    Num Unique in Train: 7
    Num Unique in Test:  7
    Unique in Train: ['Block', 'Mixed', 'Monolithic', 'Others', 'Panel', 'Stone, brick', 'Wooden', 'nan']
    Unique in Test:  ['Block', 'Mixed', 'Monolithic', 'Others', 'Panel', 'Stone, brick', 'Wooden', 'nan']
    
    EMERGENCYSTATE_MODE
    Num Unique in Train: 2
    Num Unique in Test:  2
    Unique in Train: ['No', 'Yes', 'nan']
    Unique in Test:  ['No', 'Yes', 'nan']
    
  </code>
  </pre>
  </div>
</div> 

We'll merge the test and training dataset, and create a column which indicates whether a sample is in the test or train dataset.  That way, we can perform operations (label encoding, one-hot encoding, etc) to all the data together instead of doing it once to the training data and once to the test data.


```python
# Merge test and train into all application data
train_o = train.copy()
train['Test'] = False
test['Test'] = True
test['TARGET'] = np.nan
app = train.append(test, ignore_index=True)
```

The gender column contains whether the loan applicant was male or female.  The training datset contains 4 values which weren't empty but were labelled `XNA`.  Normally we would want to create a new column to represent when the gender value is null.  However,  since the test dataset has only `M` and `F` entries, and because there are only 4 entries with a gender of `XNA` in the training set, we'll remove those entries from the training set.


```python
# Remove entries with gender = XNA
app = app[app['CODE_GENDER'] != 'XNA']
```

The `NAME_INCOME_TYPE` column also contained entries for applicants who were on Maternity leave, but no such applicants were in the test set.  There were only 5 such applicants in the training set, so we'll remove these from the training set.


```python
# Remove entries with income type = maternity leave
app = app[app['NAME_INCOME_TYPE'] != 'Maternity leave']
```

Similarly, in the `NAME_FAMILY_STATUS` column, there were 2 entries in the training set with values of `Unknown`, and no entries with that value in the test set.  So, we'll remove those too.


```python
# Remove entries with unknown family status
app = app[app['NAME_FAMILY_STATUS'] != 'Unknown']
```

There were some funky values in the `DAYS_EMPLOYED` column:


```python
app['DAYS_EMPLOYED'].hist()
plt.xlabel('DAYS_EMPLOYED')
plt.ylabel('Count')
plt.show()
```


![svg](/assets/img/loan-risk-prediction/output_21_0.svg)


350,000 days?  That's like 1,000 years!  Looks like all the reasonable values represent the number of days between when the applicant was employed and the date of the loan application.  The unreasonable values are all exactly 365,243, so we'll set those to `NaN`.


```python
# Show distribution of reasonable values
app.loc[app['DAYS_EMPLOYED']<200000, 'DAYS_EMPLOYED'].hist()
plt.xlabel('DAYS_EMPLOYED (which are less than 200,000)')
plt.ylabel('Count')
plt.show()
```


![svg](/assets/img/loan-risk-prediction/output_23_0.svg)



```python
# Show all unique outlier values
app.loc[app['DAYS_EMPLOYED']>200000, 'DAYS_EMPLOYED'].unique()
```




    array([365243])




```python
# Set unreasonable values to nan
app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
```

<a class="anchor" id="manual-feature-engineering"></a>
## Manual Feature Engineering

We'll add some features which may be informative as to how likely an applicant is to repay their loan:

- The proportion of the applicant's life they have been employed.  If a 23-year-old has only been employed for 4 years, this is fine.  If a 50-year-old has only ever been employed for 4 years, they may have trouble repaying their loan.
- The ratio of credit to income.  More income than credit will likely help an applicant be able to repay their loan.
- The ratio of income to annuity.
- The ratio of income to annuity scaled by age.
- The ratio of credit to annuity.  If an applicant has a high level of credit relative to their annuity, they may have trouble repaying their loan.
- The ratio of credit to annuity, scaled by age.  If a young person doesn't have much annuity this doesn't really mean they're less likely to repay their loan.


```python
app['PROPORTION_LIFE_EMPLOYED'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
app['INCOME_TO_CREDIT_RATIO'] = app['AMT_INCOME_TOTAL'] / app['AMT_CREDIT'] 
app['INCOME_TO_ANNUITY_RATIO'] = app['AMT_INCOME_TOTAL'] / app['AMT_ANNUITY']
app['INCOME_TO_ANNUITY_RATIO_BY_AGE'] = app['INCOME_TO_ANNUITY_RATIO'] * app['DAYS_BIRTH']
app['CREDIT_TO_ANNUITY_RATIO'] = app['AMT_CREDIT'] / app['AMT_ANNUITY']
app['CREDIT_TO_ANNUITY_RATIO_BY_AGE'] = app['CREDIT_TO_ANNUITY_RATIO'] * app['DAYS_BIRTH']
app['INCOME_TO_FAMILYSIZE_RATIO'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']
```

<a class="anchor" id="feature-encoding"></a>
## Feature Encoding

Some columns are non-numerical and will have to be encoded to numeric types so that our predictive algorithm can handle them.  We'll encode cyclical variables (like day of the week) into 2 dimensions, encode features with only two possible classes by assigning them 0 or 1, and one-hot encode categorical features with more than two classes.

The column `WEEKDAY_APPR_PROCESS_START` contains categorical information corresponding to the day of the week.  We could encode these categories as the values 1-7, but this would imply that Sunday and Monday are more similar than, say Tuesday and Sunday.  We could also one-hot encode the column into 7 new columns, but that would create 7 additional dimensions.  Seeing as the week is cyclical, we'll encode this information into two dimensions by encoding them using polar coordinates.  That is, we'll represent the days of the week as a circle.  That way, we can encode the days of the week independently, but only add two dimensions.


```python
# Create map from categories to polar projection
DOW_map = {
    'MONDAY':    0,
    'TUESDAY':   1,
    'WEDNESDAY': 2,
    'THURSDAY':  3,
    'FRIDAY':    4,
    'SATURDAY':  5,
    'SUNDAY':    6,
}
DOW_map1 = {k: np.cos(2*np.pi*v/7.0) for k, v in DOW_map.items()}
DOW_map2 = {k: np.sin(2*np.pi*v/7.0) for k, v in DOW_map.items()}

# Show encoding of days of week -> circle
days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
tt = np.linspace(0, 2*np.pi, 200)
xx = np.cos(tt)
yy = np.sin(tt)
plt.plot(xx,yy)
plt.gca().axis('equal')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.title('2D Projection of days of the week')
for day in days:
    plt.text(DOW_map1[day], DOW_map2[day], day, ha='center')
plt.show()
```


![svg](/assets/img/loan-risk-prediction/output_29_0.svg)



```python
# WEEKDAY_APPR_PROCESS_START to polar coords
col = 'WEEKDAY_APPR_PROCESS_START'
app[col+'_1'] = app[col].map(DOW_map1)
app[col+'_2'] = app[col].map(DOW_map2)
app.drop(columns=col, inplace=True)
```

For the housing-related features (e.g. `LIVINGAPARTMENTS_MODE`, `BASEMENTAREA_AVG`, etc) there are combinations of some PREFIX (e.g. `LIVINGAPARTMENTS`,  `BASEMENTAREA`, etc) and some POSTFIX (e.g. `MODE`, `MEDI`, `AVG`, etc) into a variable `PREFIX_POSTFIX`.  However, if one value for a given PREFIX is empty, the other values for that PREFIX will also be empty.  

For each column which has some empty values, we want to add an indicator column which is 1 if the value in the corresponding column is empty, and 0 otherwise.  However, if we do this with the housing-related features, we'll end up with a bunch of duplicate columns!  This is because the same samples have null values across all the POSTFIX columns for a given PREFIX.   The same problem crops up with the CREDIT_BUREAU-related features. To handle this problem, after creating the null indicator columns, we'll check for duplicate columns and merge them.

So, first we'll add columns to indicate where there are empty values in each other column.


```python
# Add indicator columns for empty values
for col in app:
    if col!='Test' and col!='TARGET':
        app_null = app[col].isnull()
        if app_null.sum()>0:
            app[col+'_ISNULL'] = app_null
```

Then we can label encode categorical features with only 2 possible values (that is, turn the labels into either 0 or 1).


```python
# Label encoder
le = LabelEncoder()

# Label encode binary fearures in training set
for col in app: 
    if col!='Test' and col!='TARGET' and app[col].dtype==object and app[col].nunique()==2:
        if col+'_ISNULL' in app.columns: #missing values here?
            app.loc[app[col+'_ISNULL'], col] = 'NaN'
        app[col] = le.fit_transform(app[col])
        if col+'_ISNULL' in app.columns: #re-remove missing vals
            app.loc[app[col+'_ISNULL'], col] = np.nan
```

Then we'll one-hot encode the categorical features which have more than 2 possible values.


```python
# Get categorical features to encode
cat_features = []
for col in app: 
    if col!='Test' and col!='TARGET' and app[col].dtype==object and app[col].nunique()>2:
        cat_features.append(col)

# One-hot encode categorical features in train set
app = pd.get_dummies(app, columns=cat_features)
```

And finally we'll remove duplicate columns.  We'll hash the columns and check if the hashes match before checking if all the values actually match, because it's a lot faster than comparing \\( O(N^2) \\) columns elementwise.


```python
# Hash columns
hashes = dict()
for col in app:
    hashes[col] = sha256(app[col].values).hexdigest()
    
# Get list of duplicate column lists
Ncol = app.shape[1] #number of columns
dup_list = []
dup_labels = -np.ones(Ncol)
for i1 in range(Ncol):
    if dup_labels[i1]<0: #if not already merged,
        col1 = app.columns[i1]
        t_dup = [] #list of duplicates matching col1
        for i2 in range(i1+1, Ncol):
            col2 = app.columns[i2]
            if ( dup_labels[i2]<0 #not already merged
                 and hashes[col1]==hashes[col2] #hashes match
                 and app[col1].equals(app[col2])): #cols are equal
                #then this is actually a duplicate
                t_dup.append(col2)
                dup_labels[i2] = i1
        if len(t_dup)>0: #duplicates of col1 were found!
            t_dup.append(col1)
            dup_list.append(t_dup)
        
# Merge duplicate columns
for iM in range(len(dup_list)):
    new_name = 'Merged'+str(iM)
    app[new_name] = app[dup_list[iM][0]].copy()
    app.drop(columns=dup_list[iM], inplace=True)
    print('Merged', dup_list[iM], 'into', new_name)
```

    Merged ['INCOME_TO_ANNUITY_RATIO_ISNULL', 'INCOME_TO_ANNUITY_RATIO_BY_AGE_ISNULL', 'CREDIT_TO_ANNUITY_RATIO_ISNULL', 'CREDIT_TO_ANNUITY_RATIO_BY_AGE_ISNULL', 'AMT_ANNUITY_ISNULL'] into Merged0
    Merged ['AMT_REQ_CREDIT_BUREAU_HOUR_ISNULL', 'AMT_REQ_CREDIT_BUREAU_MON_ISNULL', 'AMT_REQ_CREDIT_BUREAU_QRT_ISNULL', 'AMT_REQ_CREDIT_BUREAU_WEEK_ISNULL', 'AMT_REQ_CREDIT_BUREAU_YEAR_ISNULL', 'AMT_REQ_CREDIT_BUREAU_DAY_ISNULL'] into Merged1
    Merged ['APARTMENTS_MEDI_ISNULL', 'APARTMENTS_MODE_ISNULL', 'APARTMENTS_AVG_ISNULL'] into Merged2
    Merged ['BASEMENTAREA_MEDI_ISNULL', 'BASEMENTAREA_MODE_ISNULL', 'BASEMENTAREA_AVG_ISNULL'] into Merged3
    Merged ['COMMONAREA_MEDI_ISNULL', 'COMMONAREA_MODE_ISNULL', 'COMMONAREA_AVG_ISNULL'] into Merged4
    Merged ['PROPORTION_LIFE_EMPLOYED_ISNULL', 'DAYS_EMPLOYED_ISNULL'] into Merged5
    Merged ['DEF_60_CNT_SOCIAL_CIRCLE_ISNULL', 'OBS_30_CNT_SOCIAL_CIRCLE_ISNULL', 'OBS_60_CNT_SOCIAL_CIRCLE_ISNULL', 'DEF_30_CNT_SOCIAL_CIRCLE_ISNULL'] into Merged6
    Merged ['ELEVATORS_MEDI_ISNULL', 'ELEVATORS_MODE_ISNULL', 'ELEVATORS_AVG_ISNULL'] into Merged7
    Merged ['ENTRANCES_MEDI_ISNULL', 'ENTRANCES_MODE_ISNULL', 'ENTRANCES_AVG_ISNULL'] into Merged8
    Merged ['FLOORSMAX_MEDI_ISNULL', 'FLOORSMAX_MODE_ISNULL', 'FLOORSMAX_AVG_ISNULL'] into Merged9
    Merged ['FLOORSMIN_MEDI_ISNULL', 'FLOORSMIN_MODE_ISNULL', 'FLOORSMIN_AVG_ISNULL'] into Merged10
    Merged ['LANDAREA_MEDI_ISNULL', 'LANDAREA_MODE_ISNULL', 'LANDAREA_AVG_ISNULL'] into Merged11
    Merged ['LIVINGAPARTMENTS_MEDI_ISNULL', 'LIVINGAPARTMENTS_MODE_ISNULL', 'LIVINGAPARTMENTS_AVG_ISNULL'] into Merged12
    Merged ['LIVINGAREA_MEDI_ISNULL', 'LIVINGAREA_MODE_ISNULL', 'LIVINGAREA_AVG_ISNULL'] into Merged13
    Merged ['NONLIVINGAPARTMENTS_MEDI_ISNULL', 'NONLIVINGAPARTMENTS_MODE_ISNULL', 'NONLIVINGAPARTMENTS_AVG_ISNULL'] into Merged14
    Merged ['NONLIVINGAREA_MEDI_ISNULL', 'NONLIVINGAREA_MODE_ISNULL', 'NONLIVINGAREA_AVG_ISNULL'] into Merged15
    Merged ['YEARS_BEGINEXPLUATATION_MEDI_ISNULL', 'YEARS_BEGINEXPLUATATION_MODE_ISNULL', 'YEARS_BEGINEXPLUATATION_AVG_ISNULL'] into Merged16
    Merged ['YEARS_BUILD_MEDI_ISNULL', 'YEARS_BUILD_MODE_ISNULL', 'YEARS_BUILD_AVG_ISNULL'] into Merged17
    

<a class="anchor" id="baseline-predictions"></a>
## Baseline Predictions

As a baseline, let's use XGBoost with all the default parameters to predict the probabilities of applicants having trouble repaying their loans.


```python
# Split data back into test + train
train = app.loc[~app['Test'], :]
test = app.loc[app['Test'], :]

# Make SK_ID_CURR the index
train.set_index('SK_ID_CURR', inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

# Ensure all data is stored as floats
train = train.astype(np.float32)
test = test.astype(np.float32)

# Target labels
train_y = train['TARGET']

# Remove test/train indicator column and target column
train.drop(columns=['Test', 'TARGET'], inplace=True)
test.drop(columns=['Test', 'TARGET'], inplace=True)

# Classification pipeline
xgb_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier())
])

# Cross-validated AUROC
auroc_scorer = make_scorer(roc_auc_score, needs_proba=True)
scores = cross_val_score(xgb_pipeline, train, train_y, 
                         cv=3, scoring=auroc_scorer)
print('Mean AUROC:', scores.mean())

# Fit to training data
xgb_fit = xgb_pipeline.fit(train, train_y)

# Predict default probabilities of test data
test_pred = xgb_fit.predict_proba(test)

# Save predictions to file
df_out = pd.DataFrame()
df_out['SK_ID_CURR'] = test.index
df_out['TARGET'] = test_pred[:,1]
df_out.to_csv('xgboost_baseline.csv', index=False)
```

    Mean AUROC: 0.7550236128842096
    

<a class="anchor" id="calibration"></a>
## Calibration

One problem with the tree-based model is that the predicted probabilities tend to be overconfident.  That is, when the actual probability of class=1 is closer to 0.5, the model predicts probabilities closer to 0 or 1 than 0.5.  We can measure the extent of this overconfidence (or underconfidence) of our classifier by looking at its calibration curve.  The calibration curve plots the probability predicted by our model against the actual probability of samples in that bin.  A model which is perfectly calibrated should show a calibration curve which lies on the identity (y=x) line.


```python
# Predict probabilities for the training data
train_pred = cross_val_predict(xgb_pipeline, 
                               train, 
                               y=train_y,
                               method='predict_proba')
train_pred = train_pred[:,1] #only want p(default)

# Show calibration curve
fraction_of_positives, mean_predicted_value = \
    calibration_curve(train_y, train_pred, n_bins=10)
plt.figure()
plt.plot([0, 1], [0, 1], 'k:', 
         label='Perfectly Calibrated')
plt.plot(mean_predicted_value, 
         fraction_of_positives, 's-',
         label='XGBoost Predictions')
plt.legend()
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration curve for baseline XGBoost model')
plt.show()
```

![svg](/assets/img/loan-risk-prediction/output_42_1.svg)


The model is pretty well calibrated as is, exept for at higher predicted probabilities.  We can better calibrate our model by adjusting predicted probabilities to more accurately reflect the probability of loan default.  

There are two commonly-used methods for model calibration:

1. Sigmoid calibration (aka Platt's scaling, which transforms the model's predictions using a sigmoid so they more accurately reflect the actual probabilities)
1. Isotonic calibration (which calibrates the model's predictions using a method based on isotonic regression)

We'll try both methods, and see if either betters the calibration of our model.


```python
# Classification pipeline w/ isotonic calibration
calib_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', CalibratedClassifierCV(
                        base_estimator=XGBClassifier(),
                        method='isotonic'))
])

# Classification pipeline w/ sigmoid calibration
sig_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', CalibratedClassifierCV(
                        base_estimator=XGBClassifier(),
                        method='sigmoid'))
])

# Predict probabilities w/ isotonic calibration
calib_pred = cross_val_predict(calib_pipeline, 
                               train, 
                               y=train_y,
                               method='predict_proba')
calib_pred = calib_pred[:,1] #only want p(default)

# Predict probabilities w/ sigmoid calibration
sig_pred = cross_val_predict(sig_pipeline, 
                             train, 
                             y=train_y,
                             method='predict_proba')
sig_pred = sig_pred[:,1] #only want p(default)

# Show calibration curve
fop_calib, mpv_calib = \
    calibration_curve(train_y, calib_pred, n_bins=10)
fop_sig, mpv_sig = \
    calibration_curve(train_y, sig_pred, n_bins=10)
plt.figure()
plt.plot([0, 1], [0, 1], 'k:', 
         label='Perfectly Calibrated')
plt.plot(mean_predicted_value, 
         fraction_of_positives, 's-',
         label='XGBoost Predictions')
plt.plot(mpv_calib, fop_calib, 's-',
         label='Calibrated Predictions - isotonic')
plt.plot(mpv_sig, fop_sig, 's-',
         label='Calibrated Predictions - sigmoid')
plt.legend()
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration curve for Calibrated XGBoost model')
plt.show()

# Cross-validated AUROC for isotonic
print('Mean AUROC with isotonic calibration:', 
      roc_auc_score(train_y, calib_pred))

# Cross-validated AUROC for sigmoid
print('Mean AUROC with sigmoid calibration:',
      roc_auc_score(train_y, sig_pred))
```

![svg](/assets/img/loan-risk-prediction/output_44_1.svg)


    Mean AUROC with isotonic calibration: 0.7557712988933571
    Mean AUROC with sigmoid calibration: 0.755912952365782
    

Sigmoid calibration didn't appear to work very well in this case...  Isotonic calibration didn't work perfectly either, however it did appear to improve the model's discrimination a small bit (the model without calibration has slightly poorer discrimination in that it is more likely to predict probabilities which are close to 0.5).  Isotonic calibration is usually only recommended if one has \\( >>1000 \\) datapoints, which we do (the training set contains around 300,000 datapoins), so we'll go ahead and use isotonic calibration.  Now we can output our predictions after calibrating.


```python
# Fit to the training data
calib_fit = calib_pipeline.fit(train, train_y)

# Predict default probabilities of the test data
test_pred = calib_fit.predict_proba(test)

# Save predictions to file
df_out = pd.DataFrame()
df_out['SK_ID_CURR'] = test.index
df_out['TARGET'] = test_pred[:,1]
df_out.to_csv('xgboost_calibrated.csv', index=False)
```

<a class="anchor" id="resampling"></a>
## Resampling

The target class is very imbalanced: many more people successfully repaid their loans than had trouble repaying.


```python
# Show distribution of target variable
sns.countplot(x='TARGET', data=app)
plt.title('Number of applicants who had trouble repaying')
plt.show()
```


![svg](/assets/img/loan-risk-prediction/output_48_0.svg)


We'll use the [imbalanced-learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html) package to re-sample our dataset such that the classes are balanced.  There are several different common methods we could use for re-sampling: 

1. Random over-sampling (randomly repeat minority class examples in the training data)
1. Random under-sampling (randomly drop majority class examples from the training data)
1. Synthetic minority oversampling technique (SMOTE, generate additional synthetic training examples which are similar to the minority class)

We'll try all three techniques, and see if any of the techniques give better predictive performance in terms of the AUROC.


```python
# A sampler that doesn't re-sample!
class DummySampler(object):
    def sample(self, X, y):
        return X, y
    def fit(self, X, y):
        return self
    def fit_sample(self, X, y):
        return self.sample(X, y)
    
# List of samplers to test
samplers = [
    ['Oversampling', RandomOverSampler()], 
    ['Undersampling', RandomUnderSampler()], 
    ['SMOTE', SMOTE()],
    ['No resampling', DummySampler()]
]

# Preprocessing pipeline
pre_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median'))
])

# Classifier
classifier = CalibratedClassifierCV(
                        base_estimator=XGBClassifier(),
                        method='isotonic')

# Compute AUROC and plot ROC for each type of sampler
plt.figure()
auroc_scorer = make_scorer(roc_auc_score, needs_proba=True)
cv = StratifiedKFold(n_splits=3)
for name, sampler in samplers:
    
    # Make the sampling and classification pipeline
    pipeline = make_pipeline(sampler, calib_pipeline)

    # Cross-validated predictions on training set
    probas = np.zeros(train.shape[0]) # to store predicted probabilities
    for tr, te in cv.split(train, train_y):
        test_pre = pre_pipeline.fit_transform(train.iloc[te])  #preprocess test fold
        train_pre = pre_pipeline.fit_transform(train.iloc[tr]) #preprocess training fold
        train_s, train_y_s = sampler.fit_sample(train_pre, train_y.iloc[tr]) #resample train fold
        probas_ = classifier.fit(train_s, train_y_s).predict_proba(test_pre) #predict test fold
        probas[te] = probas_[:,1]
    
    # Print AUROC value
    print(name, 'AUROC:', roc_auc_score(train_y, probas))
    
    # Plot ROC curve for this sampler
    fpr, tpr, threshs = roc_curve(train_y, probas)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], label='Chance')
plt.legend()
plt.show()
```

    Oversampling AUROC: 0.7554566031903476
    Undersampling AUROC: 0.754677367433835
    SMOTE AUROC: 0.692312181508109
    No resampling AUROC: 0.7544329991005927
    

![svg](/assets/img/loan-risk-prediction/output_50_8.svg)


Unfortunately it looks like none of the sampling techniques actually helped improve the AUROC score!  The SMOTE resampling technique did even more poorly than simply under- or over-sampling.  This is probably because SMOTE generates samples by interpolating between training samples in feature-space, but most of our features are binary.  So, interpolation isn't really adding diversity to the training data, it's just adding noise and making it more difficult for our classification algorithm to decide where to put a threshold in that dimension.

<a class="anchor" id="feature-importance"></a>
## Feature Importance

For our final predictions, we'll use the isotonic calibrated model with no resampling (which we've already used to make predictions on the test data, back in the [calibration](#calibration) section), since resampling didn't appear to help increase the preformance of our model.  We can view how important each feature was to the model's predictions by using XGBoost's `plot_importance` function.


```python
# Fit XGBoost model on the training data
train_pre = pre_pipeline.fit_transform(train) #preprocess training data
model = XGBClassifier()
model.fit(train, train_y)

# Show feature importances
plt.figure(figsize=(6, 15))
plot_importance(model, height=0.5, ax=plt.gca())
plt.show()
```


![svg](/assets/img/loan-risk-prediction/output_53_0.svg)


The three most important factors by far were the three "external sources."  Presumably these were credit scores or some other similar reliability measure from sources outside Home Credit.  The credit-to-annuity ration was also very important, and other factors such as employment length, age, gender - *gender*?


```python
# Show default probability by gender
plt.figure()
sns.barplot(x='CODE_GENDER', y="TARGET", data=train_o)
plt.show()
```

![svg](/assets/img/loan-risk-prediction/output_55_1.svg)


Indeed female applicants only default on their loans around 7% of the time, while male applicants default around 10% of the time.

## Conclusion

Now that we've built a working predictive model, it would normally be time to put it into production.  However, there are a few things we should consider before doing so.  Firstly, because this model could have a direct effect on large number of individuals' financial lives, we need to ensure our model is being equitable, and isn't [discriminating by proxy](https://digitalcommons.law.umaryland.edu/fac_pubs/285/) against certain groups based on race, gender, ethnicity, etc.  Also, because we used applicants' personal information to train the model, we should ensure that those applicants have been informed their information would be used for such a purpose, that they have given consent, and that we have minimized the personally identifiable information present in the dataset.  Still other ethical issues exist which we would want to address before putting our model into production.  There are tools to help us ensure our model and data practices more ethical, such as checklists like [Deon](http://deon.drivendata.org/) and toolkits like the [Ethics and Algorithms Toolkit](https://ethicstoolkit.ai/).

Another thing to prepare for when considering putting a predictive model into deployment is the possibility of covariate shift or concept drift.  We would want to have a monitoring system in place for a deployed model which could alert us when our data inputs appear to be changing over time, or when our model is no longer fitting the data as well as it used to (or, generally, when things are changing unexpectedly).

Finally, remember that the point of building a predictive model to estimate how likely applicants are to pay back their loans is not just for Home Credit Group to use that information to accept or reject applicants.  Rather, they want to be able to predict which principal and payment plan would be the best option for each applicant.  An even more useful model would be one which predicted loan repayment probability given not only the applicant information, but also information about the proposed principal and payment schedule.  This way, Home Credit Group could use the model as a tool to decide not only whether to accept or reject applicants, but to determine the specifics of a loan which would be best for each of their applicants. 
