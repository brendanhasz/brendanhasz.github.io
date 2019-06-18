---
layout: post
title: "Customer Loyalty Prediction 2: Feature Engineering and Feature Selection"
date: 2019-04-04
description: "Engineering features, performing aggregations with transaction information, and using mutual information and permutation-based feature importance to select features."
img_url: /assets/img/loyalty-prediction-2-features/mutual_info.svg
github_url: https://github.com/brendanhasz/loyalty-prediction
kaggle_url: https://www.kaggle.com/brendanhasz/elo-feature-engineering-and-feature-selection
tags: [python, feature engineering, feature selection]
comments: true
---


[Elo](https://elo.com.br/) is a Brazillian debit and credit card brand.  They offer credit and prepaid transactions, and have paired up with merchants in order offer promotions to cardholders.  In order to offer more relevant and personalized promotions, in a [recent Kaggle competition](https://www.kaggle.com/c/elo-merchant-category-recommendation), Elo challenged Kagglers to predict customer loyalty based on transaction history.  Presumably they plan to use a loyalty-predicting model in order to determine what promotions to offer to customers based on how certain offers are predicted to affect card owners' card loyalty.

In a [previous post](https://brendanhasz.github.io/2019/03/20/loyalty-prediction-1-eda.html), we loaded and cleaned the data, and performed some exploratory data analysis.  In this post, we'll engineer new features about the card accounts, compute aggregate statistics about transactions made with each card, and then select which features to use in a predictive model.


**Outline**

- [Feature Engineering](#feature-engineering)
- [Feature Aggregations](#feature-aggregations)
- [Feature Selection](#feature-selection)
  - [Mutual Information](#mutual-information)
  - [Permutation-based Feature Importance](#permutation-based-feature-importance)
- [Conclusion](#conclusion)



Let's first load the packages we'll use:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor

# Plot settings
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
sns.set()

# Encoders and permutation importance
!pip install git+http://github.com/brendanhasz/dsutils.git
from dsutils.encoding import one_hot_encode
from dsutils.encoding import TargetEncoderCV
from dsutils.printing import print_table
from dsutils.evaluation import permutation_importance_cv
from dsutils.evaluation import plot_permutation_importance
```



## Feature Engineering

At this point we have four tables:

* `hist_trans` - contains details about historical transactions
* `new_trans` - contains details about newer transactions
* `merchants` - contains information about the merchants
* `cards` - contains information about the card accounts

Eventually, we'll want one large table where each row corresponds to a card account, and each column corresponds to a feature of that account.  There's not a whole lot of information in the `cards` table as is, so we'll have to engineer features of each card using the information in the other three tables.  The idea is that information about how often an individual is making transactions, when, for how much, with what merchants,  with what *types* of merchants - and so on - will be informative as to how likely that individual is to have a high loyalty score.

Each transaction in `hist_trans` and `new_trans` has a `card_id`, which we can use to find all the transactions from a given card account in the `cards` dataset (there is also a `card_id` column in that table).  Similarly, each transaction also has a corresponding `merchant_id`, which we can use to look up the information about the corresponding merchant in the `merchants` dataset (that table also has a `merchant_id` column).

The first thing to do is merge the transactions datasets with the merchants dataset on the `merchant_id` column.  That is, for each transaction, we need to look up the information about the merchant which participated in that transaction in the merchants dataset, and append it to the transactions data.

The pandas merge function makes this pretty easy:


```python
# Merge transactions with merchants data
hist_trans = pd.merge(hist_trans, merchants, on='merchant_id')
new_trans = pd.merge(new_trans, merchants, on='merchant_id')
```


Next, we'll want to encode some of the simpler categorical columns.  The `category_2` and `category_3` columns have only 5 and 3 unique values (and `NaN`s), so we'll simply one-hot encode them.


```python
# One-hot encode category 2 and 3
cat_cols = ['category_2', 'category_3']
hist_trans = one_hot_encode(hist_trans,
                            cols=cat_cols)
new_trans = one_hot_encode(new_trans,
                           cols=cat_cols)
```


Then, we can create some time-based features about the purchases.  For example: the hour of the day the purchase was made, the day of the week, the week of the year, the month, whether the purchase was made on a weekend, the time of the purchase relative to when the card owner was first active, etc.


```python
# Time-based features for purchases
ref_date = np.datetime64('2017-09-01')
one_hour = np.timedelta64(1, 'h')
for df in [hist_trans, new_trans]:
    tpd = df['purchase_date']
    df['purchase_hour'] = tpd.dt.hour.astype('uint8')
    df['purchase_day'] = tpd.dt.dayofweek.astype('uint8')
    df['purchase_week'] = tpd.dt.weekofyear.astype('uint8')
    df['purchase_dayofyear'] = tpd.dt.dayofyear.astype('uint16')
    df['purchase_month'] = tpd.dt.month.astype('uint8')
    df['purchase_weekend'] = (df['purchase_day'] >=5 ).astype('uint8')
    df['purchase_time'] = ((tpd - ref_date) / one_hour).astype('float32')
    df['ref_date'] = ((tpd - pd.to_timedelta(df['month_lag'], 'M')
                          - ref_date ) / one_hour).astype('float32')

    # Time sime first active
    tsfa = pd.merge(df[['card_id']], 
                    cards[['first_active_month']].copy().reset_index(),
                    on='card_id', how='left')
    df['time_since_first_active'] = ((tpd - tsfa['first_active_month'])
                                     / one_hour).astype('float32')
    
    # Clean up
    del tsfa
    del df['purchase_date']
```

Finally, we need to convert the `first_active_month` column (a datetime) to a month.  This way at this point all of our data will be in a numerical format.


```python
cards['first_active_month'] = (12*(cards['first_active_month'].dt.year-2011) + 
                               cards['first_active_month'].dt.month).astype('float32')
```

## Feature Aggregations

Now we can engineer features for each card account by applying aggregation functions on the transaction data corresponding to each card.  First, we need to group the transactions by the card account which was used to make them:


```python
# Group transactions by card id
hist_trans = hist_trans.groupby('card_id', sort=False)
new_trans = new_trans.groupby('card_id', sort=False)
```

We'll also need to define some custom aggregation functions which will allow us to better extract information from the transactions.  The first is a function which computes the entropy given some categorical data.  A feature corresponding to entropy could be informative - for example, it could be that card accounts with high entropy over the merchants they use their card with are more likely to be more loyal card users than those who only use their card with a single merchant (and therefore have low entropy).


```python
def entropy(series):
    """Categorical entropy"""
    probs = series.value_counts().values.astype('float32')
    probs = probs / np.sum(probs)
    probs[probs==0] = np.nan
    return -np.nansum(probs * np.log2(probs))
```

Another aggregation function which could be useful is one which computes the mean difference between *consecutive* items in a series.  For example, given a column with the purchase date, this function would compute the mean time between purchases.  This could conceivably be a good predictor of how likely an individual is to be a loyal card user: individuals who use their cards regularly and frequently are probably more likely to be loyal.


```python
def mean_diff(series):
    """Mean difference between consecutive items in a series"""
    ss = series.sort_values()
    return (ss - ss.shift()).mean()
```

The period of a sequence of transactions could also be a useful feature.  That is, the difference between the minimum and maximum value.  For example, customers who have been making purchases over a long period of time (the difference between the date of their first and last purchases is large) may be more likely to be loyal card users.


```python
def period(series):
    """Period of a series (max-min)"""
    return series.max() - series.min()
```

Finally, we'll create a function to compute the mode (just because pandas' default mode function doesn't handle `NaN`s well, or cases where there are two equally most common elements).


```python
def mode(series):
    """Most common element in a series"""
    tmode = series.mode()
    if len(tmode) == 0:
        return np.nan
    else:
        return tmode[0]
```

Now we can actually compute the aggregations.  We'll define a list of aggregation functions to perform for each datatype:


```python
# Aggregations to perform for each predictor type
binary_aggs = ['sum', 'mean', 'nunique']
categorical_aggs = ['nunique', entropy, mode]
continuous_aggs = ['min', 'max', 'sum', 'mean', 'std', 'skew', mean_diff, period]
```

And then, using those lists, we'll define a dictionary containing which aggregation functions to apply on which columns of the transactions data.  I've occasionally added a mean() aggregation function when it seems like the variable could be ordinal (but we're not sure because of the anonymous feature names!).


```python
# Aggregations to perform on each column
aggs = {
    'authorized_flag':             binary_aggs,
    'city_id':                     categorical_aggs,
    'category_1':                  binary_aggs,
    'installments':                continuous_aggs,
    'category_3_nan':              binary_aggs,
    'category_3_0.0':              binary_aggs,
    'category_3_1.0':              binary_aggs,
    'category_3_2.0':              binary_aggs,
    'category_2_nan':              binary_aggs,
    'category_2_1.0':              binary_aggs,
    'category_2_2.0':              binary_aggs,
    'category_2_3.0':              binary_aggs,
    'category_2_4.0':              binary_aggs,
    'category_2_5.0':              binary_aggs,
    'merchant_category_id':        categorical_aggs,
    'merchant_id':                 categorical_aggs,
    'month_lag':                   continuous_aggs,
    'purchase_amount':             continuous_aggs,
    'purchase_time':               continuous_aggs + ['count'],
    'purchase_hour':               categorical_aggs + ['mean'],
    'purchase_day':                categorical_aggs + ['mean'],
    'purchase_week':               categorical_aggs + continuous_aggs,
    'purchase_month':              categorical_aggs + continuous_aggs,
    'purchase_weekend':            binary_aggs,
    'ref_date':                    continuous_aggs,
    'time_since_first_active':     continuous_aggs,
    'state_id':                    categorical_aggs,
    'subsector_id':                categorical_aggs,
    'merchant_group_id':           categorical_aggs,
    'numerical_1':                 continuous_aggs,
    'numerical_2':                 continuous_aggs,
    'most_recent_sales_range':     categorical_aggs + ['mean'], #ordinal?
    'most_recent_purchases_range': categorical_aggs + ['mean'], #orindal?
    'avg_sales_lag3':              continuous_aggs,
    'avg_purchases_lag3':          continuous_aggs,
    'active_months_lag3':          continuous_aggs,
    'avg_sales_lag6':              continuous_aggs,
    'avg_purchases_lag6':          continuous_aggs,
    'active_months_lag6':          continuous_aggs,
    'avg_sales_lag12':             continuous_aggs,
    'avg_purchases_lag12':         continuous_aggs,
    'active_months_lag12':         continuous_aggs,
    'category_4':                  binary_aggs,
}
```

Ok, phew, *now* we can actually compute the aggregations.  This'll take a while!


```python
# Perform each aggregation
for col, funcs in aggs.items():
    for func in funcs:
        
        # Get name of aggregation function
        if isinstance(func, str):
            func_str = func
        else:
            func_str = func.__name__
            
        # Name for new column
        new_col = col + '_' + func_str
            
        # Compute this aggregation
        cards['hist_'+new_col] = hist_trans[col].agg(func).astype('float32')
        cards['new_'+new_col] = new_trans[col].agg(func).astype('float32')
```

After blindly doing a bunch of aggregations on a dataset, it's usually a good idea to check for non-informative columns.  That is, columns which are all `NaN`, only contain one unique value, etc.  Let's check for those and remove them from the dataset.


```python
def remove_noninformative(df):
    """Remove non-informative columns (all nan, or all same value)"""
    for col in df:
        if df[col].isnull().all():
            print('Removing '+col+' (all NaN)')
            del df[col]
        elif df[col].nunique()<2:
            print('Removing '+col+' (only 1 unique value)')
            del df[col]

remove_noninformative(cards)
```

    Removing new_authorized_flag_mean (only 1 unique value)
    Removing new_authorized_flag_nunique (only 1 unique value)
    Removing hist_category_3_nan_sum (only 1 unique value)
    Removing new_category_3_nan_sum (only 1 unique value)
    Removing hist_category_3_nan_mean (only 1 unique value)
    Removing new_category_3_nan_mean (only 1 unique value)
    Removing hist_category_3_nan_nunique (only 1 unique value)
    Removing new_category_3_nan_nunique (only 1 unique value)
    Removing hist_category_2_nan_sum (only 1 unique value)
    Removing new_category_2_nan_sum (only 1 unique value)
    Removing hist_category_2_nan_mean (only 1 unique value)
    Removing new_category_2_nan_mean (only 1 unique value)
    Removing hist_category_2_nan_nunique (only 1 unique value)
    Removing new_category_2_nan_nunique (only 1 unique value)
    Removing hist_active_months_lag3_max (only 1 unique value)
    Removing hist_active_months_lag6_max (only 1 unique value)
    

Now we have one, *giant* table, where each row corresponds to a card account for which we want to predict the loyalty, and each column corresponds to a feature of that account.  Unfortunately, what with all the aggregations we performed, we now have well over 400 features!


```python
cards.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 325540 entries, C_ID_92a2005557 to C_ID_87e7979a5f
    Columns: 459 entries, first_active_month to new_category_4_nunique
    dtypes: float32(456), uint8(3)
    memory usage: 569.7+ MB
    

Finally, we'll split the data into test and training data, as well as X and y data.


```python
# Test data
test = cards['target'].isnull()
X_test = cards[test].copy()
del X_test['target']

# Training data
y_train = cards.loc[~test, 'target'].copy()
X_train = cards[~test].copy()
del X_train['target']

# Clean up 
del cards
```



## Feature Selection

Machine learning models don't generally perform well when they're given a huge number of features, many of which are not very informative.  The more superfluous features we give our model to train on, the more likely it is to overfit!  To prune out features which could confuse our predictive model, we'll perform some feature selection. 

Ideally, we'd fit our model a bunch of different times, using every possible different combination of features, and use the set of features which gives the best cross-validated results. But, there are a few problems with that approach.  First, that would lead to overfitting to the training data.  Second, and perhaps even more importantly, it would take forever.

There are a bunch of different ways to perform feature selection in a less exhaustive, but more expedient manner.  Forward selection, backward selection, selecting features based on their correlation with the target variable, and "embedded" methods such as Lasso regressions (where the model itself performs feature selection during training) are all options.  However, here we'll use two different methods: the mutual information between each feature and the target variable, and the permutation-based feature importance of each feature.


### Mutual Information

To get some idea of how well each feature corresponds to the target (the loyalty score), we can compute the [mutual information](https://en.wikipedia.org/wiki/Mutual_information) between each feature and the target. Let's make a function to compute the mutual information between two vectors.


```python
def mutual_information(xi, yi, res=20):
    """Compute the mutual information between two vectors"""
    ix = ~(np.isnan(xi) | np.isinf(xi) | np.isnan(yi) | np.isinf(yi))
    x = xi[ix]
    y = yi[ix]
    N, xe, ye = np.histogram2d(x, y, res)
    Nx, _ = np.histogram(x, xe)
    Ny, _ = np.histogram(y, ye)
    N = N / len(x) #normalize
    Nx = Nx / len(x)
    Ny = Ny / len(y)
    Ni = np.outer(Nx, Ny)
    Ni[Ni == 0] = np.nan
    N[N == 0] = np.nan
    return np.nansum(N * np.log(N / Ni))
```

The mutual information represents the amount of information that can be gained about one variable by knowing the value of some other vairable.  Obviously this is very relevant to the task of feature selection: we want to choose features which knowing the value of will give us as much information as possible about the target variable.

Practically speaking, the nice thing about using mutual information instead of, say, the correlation coefficient, is that it is sensitive to nonlinear relationships.  We'll be using nonlinear predictive models (like gradient boosted decision trees), and so we don't want to limit the features we select to be only ones which have a linear relationship to the target variable.  Notice how the sin-like relationship in the middle plot below has a high mutual information, but not a great correlation coefficient.


```python
# Show mutual information vs correlation
x = 5*np.random.randn(1000)
y = [x + np.random.randn(1000),
     2*np.sin(x) + np.random.randn(1000),
     x + 10*np.random.randn(1000)]
plt.figure(figsize=(10, 4))
for i in range(3):    
    plt.subplot(1, 3, i+1)
    plt.plot(x, y[i], '.')
    rho, _ = spearmanr(x, y[i])
    plt.title('Mutual info: %0.3f\nCorr coeff: %0.3f'
              % (mutual_information(x, y[i]), rho))
    plt.gca().tick_params(labelbottom=False, labelleft=False)
```


![svg](/assets/img/loyalty-prediction-2-features/mutual_info.svg)


We'll use the mutual information of the quantile-transformed aggregation scores (just so outliers don't mess up the mutual information calculation).  So, we'll need a function to perform the [quantile transform](https://en.wikipedia.org/wiki/Quantile_normalization), and one to compute the mutual information after applying the quantile transform:


```python
def quantile_transform(v, res=101):
    """Quantile-transform a vector to lie between 0 and 1"""
    x = np.linspace(0, 100, res)
    prcs = np.nanpercentile(v, x)
    return np.interp(v, prcs, x/100.0)
    
    
def q_mut_info(x, y):
    """Mutual information between quantile-transformed vectors"""
    return mutual_information(quantile_transform(x),
                              quantile_transform(y))
```

Now we can compute the mutual information between each feature and the loyalty score.


```python
# Compute the mutual information
cols = []
mis = []
for col in X_train:
    mi = q_mut_info(X_train[col], y_train)
    cols.append(col)
    mis.append(mi)
    
# Print mut info of each feature
print_table(['Column', 'Mutual_Information'],
            [cols, mis])
```


<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>Column                                    Mutual_Information
    new_purchase_amount_sum                   0.054757
    new_purchase_amount_period                0.054365
    new_active_months_lag12_sum               0.053549
    new_merchant_id_nunique                   0.053511
    new_avg_sales_lag3_sum                    0.053497
    new_avg_purchases_lag3_sum                0.053303
    new_merchant_id_entropy                   0.05321
    new_authorized_flag_sum                   0.053192
    new_purchase_time_count                   0.053192
    new_active_months_lag3_sum                0.05312
    new_active_months_lag6_sum                0.05306
    new_merchant_category_id_nunique          0.053004
    new_avg_sales_lag6_sum                    0.052795
    new_purchase_week_nunique                 0.052637
    new_avg_purchases_lag6_sum                0.052512
    new_avg_sales_lag12_sum                   0.051847
    new_avg_purchases_lag12_sum               0.051539
    new_subsector_id_nunique                  0.050857
    new_merchant_category_id_entropy          0.050746
    new_purchase_hour_nunique                 0.050432
    new_ref_date_sum                          0.050098
    new_purchase_week_entropy                 0.049811
    new_merchant_group_id_nunique             0.048877
    new_purchase_time_sum                     0.048794
    new_purchase_hour_entropy                 0.048575
    new_month_lag_sum                         0.048569
    new_subsector_id_entropy                  0.047022
    new_purchase_day_nunique                  0.046599
    new_purchase_month_sum                    0.046455
    new_purchase_time_period                  0.046102
    new_time_since_first_active_period        0.046101
    new_merchant_group_id_entropy             0.045913
    new_purchase_amount_max                   0.045449
    new_purchase_week_period                  0.044815
    new_purchase_week_sum                     0.04389
    new_purchase_day_entropy                  0.043767
    new_most_recent_purchases_range_nunique   0.042917
    new_most_recent_sales_range_nunique       0.042466
    new_ref_date_period                       0.040804
    new_most_recent_purchases_range_entropy   0.040498
    new_most_recent_sales_range_entropy       0.039465
    new_purchase_month_mean                   0.039101
    new_time_since_first_active_sum           0.037902
    new_avg_sales_lag3_period                 0.037719
    new_avg_purchases_lag3_period             0.037481
    new_month_lag_mean                        0.037309
    new_avg_sales_lag6_period                 0.03696
    new_avg_purchases_lag6_period             0.03626
    new_purchase_month_entropy                0.03607
    new_avg_purchases_lag12_period            0.035874
    new_avg_sales_lag12_period                0.035528
    new_category_3_0.0_sum                    0.033882
    new_month_lag_period                      0.032239
    new_purchase_month_nunique                0.032239
    new_purchase_month_period                 0.032239
    new_purchase_week_mean                    0.030506
    new_numerical_1_period                    0.030269
    new_ref_date_min                          0.030071
    new_purchase_time_min                     0.030042
    new_numerical_2_period                    0.030026
    new_purchase_amount_std                   0.028056
    new_installments_sum                      0.028037
    new_month_lag_std                         0.027715
    new_purchase_time_mean                    0.027591
    hist_purchase_time_max                    0.027389
    new_purchase_weekend_mean                 0.02721
    new_month_lag_mean_diff                   0.027155
    new_ref_date_mean                         0.026726
    new_purchase_week_min                     0.02663
    new_purchase_month_mean_diff              0.026444
    new_purchase_amount_mean                  0.026406
    new_purchase_month_std                    0.026338
    new_city_id_nunique                       0.026197
    new_ref_date_max                          0.026059
    new_purchase_weekend_sum                  0.025851
    new_purchase_week_max                     0.02575
    new_purchase_week_std                     0.025149
    new_numerical_2_sum                       0.024955
    new_purchase_time_max                     0.02463
    new_city_id_entropy                       0.02454
    hist_month_lag_mean_diff                  0.024
    hist_purchase_time_mean_diff              0.023314
    hist_time_since_first_active_mean_diff    0.023314
    hist_ref_date_max                         0.022919
    new_category_2_1.0_sum                    0.022894
    new_purchase_weekend_nunique              0.022806
    new_purchase_month_max                    0.022756
    new_avg_purchases_lag12_min               0.022074
    new_numerical_1_sum                       0.022061
    new_purchase_time_std                     0.021825
    new_time_since_first_active_std           0.021825
    new_avg_sales_lag3_max                    0.021822
    new_avg_purchases_lag6_min                0.021538
    new_purchase_day_mean                     0.021385
    new_avg_purchases_lag3_max                0.021209
    new_avg_sales_lag12_min                   0.0212
    new_category_3_1.0_sum                    0.021029
    new_purchase_month_min                    0.020932
    new_avg_sales_lag6_min                    0.020785
    new_avg_purchases_lag3_min                0.020696
    hist_ref_date_sum                         0.020629
    new_avg_sales_lag3_min                    0.020611
    new_category_3_1.0_mean                   0.020608
    new_most_recent_purchases_range_mean      0.020578
    new_most_recent_sales_range_mean          0.020412
    new_category_4_sum                        0.020386
    new_avg_sales_lag6_max                    0.019944
    hist_purchase_time_sum                    0.019924
    new_purchase_week_mean_diff               0.019582
    hist_ref_date_min                         0.019142
    hist_ref_date_mean                        0.018984
    hist_purchase_week_period                 0.018924
    new_avg_purchases_lag6_max                0.018923
    new_merchant_id_mode                      0.018294
    new_installments_max                      0.018253
    new_numerical_2_max                       0.018197
    new_avg_sales_lag12_max                   0.018155
    hist_authorized_flag_sum                  0.017949
    new_numerical_1_max                       0.017822
    new_avg_purchases_lag12_max               0.017567
    new_installments_period                   0.017396
    hist_avg_sales_lag3_sum                   0.017185
    hist_active_months_lag12_sum              0.01716
    hist_active_months_lag6_sum               0.017157
    hist_purchase_time_count                  0.017145
    hist_active_months_lag3_sum               0.017143
    hist_avg_purchases_lag3_sum               0.017119
    new_category_4_mean                       0.017022
    hist_avg_sales_lag6_sum                   0.016936
    hist_avg_purchases_lag6_sum               0.016819
    new_purchase_month_skew                   0.016749
    hist_avg_sales_lag12_sum                  0.016663
    new_state_id_entropy                      0.016658
    new_purchase_time_mean_diff               0.016612
    new_time_since_first_active_mean_diff     0.016611
    hist_ref_date_mean_diff                   0.016578
    hist_avg_purchases_lag12_sum              0.016576
    hist_merchant_id_nunique                  0.016176
    new_purchase_week_mode                    0.0161
    new_month_lag_skew                        0.015899
    hist_category_3_0.0_sum                   0.015847
    hist_purchase_month_sum                   0.015736
    hist_purchase_week_sum                    0.015632
    new_numerical_2_skew                      0.015542
    new_category_4_nunique                    0.015535
    new_ref_date_std                          0.015378
    new_category_3_2.0_mean                   0.015284
    hist_merchant_category_id_nunique         0.015193
    hist_purchase_amount_sum                  0.015158
    new_numerical_1_skew                      0.015077
    hist_purchase_month_period                0.015014
    hist_merchant_group_id_nunique            0.014956
    new_installments_mean                     0.014832
    new_month_lag_min                         0.014805
    hist_purchase_month_mean_diff             0.014752
    new_numerical_2_mean                      0.014721
    new_purchase_hour_mean                    0.014616
    new_purchase_amount_skew                  0.014573
    hist_purchase_week_mean_diff              0.014502
    hist_avg_sales_lag12_mean_diff            0.014314
    hist_avg_sales_lag6_mean_diff             0.013997
    hist_purchase_month_std                   0.013978
    hist_purchase_week_entropy                0.01396
    new_merchant_group_id_mode                0.01394
    hist_avg_purchases_lag12_mean_diff        0.013919
    new_ref_date_mean_diff                    0.013887
    new_numerical_1_min                       0.013825
    hist_purchase_weekend_sum                 0.013781
    hist_avg_purchases_lag6_mean_diff         0.013741
    hist_purchase_week_std                    0.013535
    new_state_id_nunique                      0.013426
    hist_avg_sales_lag3_mean_diff             0.013339
    hist_category_3_1.0_mean                  0.013327
    hist_purchase_week_max                    0.013283
    hist_subsector_id_nunique                 0.01318
    new_purchase_month_mode                   0.01317
    hist_purchase_week_nunique                0.013119
    new_purchase_amount_mean_diff             0.013111
    new_active_months_lag12_mean              0.012946
    new_numerical_1_mean                      0.0129
    hist_category_3_1.0_sum                   0.012663
    hist_avg_purchases_lag3_mean_diff         0.01259
    hist_installments_mean                    0.012296
    new_installments_std                      0.012075
    hist_purchase_hour_nunique                0.011996
    new_purchase_week_skew                    0.011787
    new_category_3_1.0_nunique                0.011641
    hist_merchant_id_entropy                  0.011539
    new_active_months_lag12_mean_diff         0.011509
    hist_purchase_time_mean                   0.011486
    new_category_3_2.0_sum                    0.011464
    new_numerical_2_min                       0.011085
    new_category_3_2.0_nunique                0.010784
    new_installments_mean_diff                0.010686
    hist_purchase_month_entropy               0.010576
    hist_category_2_1.0_sum                   0.010354
    new_month_lag_max                         0.010285
    hist_purchase_hour_entropy                0.010221
    hist_installments_sum                     0.010082
    hist_ref_date_period                      0.01004
    new_avg_sales_lag3_std                    0.010001
    hist_active_months_lag12_mean_diff        0.0097951
    new_avg_purchases_lag3_std                0.0097771
    hist_purchase_month_max                   0.0095819
    hist_purchase_week_min                    0.0094769
    hist_category_3_0.0_mean                  0.0092955
    new_category_2_1.0_mean                   0.0092706
    new_category_1_mean                       0.0092675
    hist_purchase_time_period                 0.0092008
    hist_time_since_first_active_period       0.0092008
    new_installments_min                      0.0091653
    new_purchase_time_skew                    0.0090182
    new_time_since_first_active_skew          0.0090131
    new_active_months_lag12_skew              0.0089038
    new_avg_purchases_lag12_std               0.0087701
    new_avg_purchases_lag6_std                0.0087281
    hist_month_lag_sum                        0.0087162
    new_avg_sales_lag6_std                    0.0086593
    new_active_months_lag12_period            0.0085932
    hist_avg_sales_lag6_min                   0.0085211
    hist_merchant_group_id_entropy            0.0084834
    hist_avg_sales_lag12_period               0.0084003
    hist_month_lag_skew                       0.008345
    hist_avg_purchases_lag12_min              0.008329
    hist_avg_purchases_lag6_min               0.0082943
    hist_purchase_month_nunique               0.0082444
    hist_avg_sales_lag12_min                  0.0082438
    new_avg_sales_lag12_std                   0.0082004
    new_category_2_1.0_nunique                0.0081688
    hist_avg_sales_lag6_period                0.0081279
    new_avg_sales_lag12_mean_diff             0.0080849
    hist_avg_sales_lag3_period                0.0080384
    hist_month_lag_max                        0.0080244
    new_numerical_2_std                       0.0079489
    new_avg_sales_lag6_mean_diff              0.0079073
    hist_avg_purchases_lag12_period           0.0079065
    hist_time_since_first_active_sum          0.0078141
    hist_category_4_sum                       0.0077914
    hist_city_id_nunique                      0.007767
    hist_merchant_category_id_entropy         0.0077301
    new_avg_purchases_lag6_mean_diff          0.0077202
    hist_purchase_day_entropy                 0.007709
    hist_purchase_time_skew                   0.0077077
    hist_time_since_first_active_skew         0.0077077
    hist_avg_purchases_lag3_period            0.0076734
    hist_numerical_1_sum                      0.0076627
    new_numerical_2_mean_diff                 0.0076452
    hist_avg_purchases_lag6_period            0.0076331
    hist_month_lag_period                     0.0076249
    hist_purchase_time_min                    0.0075276
    hist_category_1_mean                      0.007517
    hist_numerical_2_sum                      0.0075015
    new_avg_purchases_lag12_mean_diff         0.0074972
    new_active_months_lag12_std               0.0074804
    new_numerical_1_std                       0.0074678
    new_ref_date_skew                         0.0072579
    hist_most_recent_purchases_range_nunique  0.0071494
    hist_most_recent_sales_range_nunique      0.0071452
    new_avg_sales_lag3_mean_diff              0.0071183
    hist_month_lag_min                        0.0071006
    hist_installments_mean_diff               0.0070159
    hist_authorized_flag_mean                 0.0069817
    hist_month_lag_mean                       0.0068269
    hist_purchase_month_mode                  0.0068044
    new_active_months_lag12_min               0.0067621
    hist_purchase_week_mode                   0.006748
    hist_purchase_month_min                   0.0065987
    hist_installments_min                     0.006581
    new_category_2_3.0_sum                    0.0064858
    hist_installments_std                     0.0064594
    hist_subsector_id_entropy                 0.0064114
    new_avg_purchases_lag3_mean_diff          0.0063822
    hist_avg_purchases_lag12_max              0.0063011
    hist_installments_skew                    0.0062587
    hist_avg_sales_lag12_max                  0.006192
    hist_numerical_2_skew                     0.0061363
    first_active_month                        0.0061322
    new_avg_sales_lag12_mean                  0.0061219
    hist_numerical_1_period                   0.006049
    hist_category_3_2.0_mean                  0.0059891
    hist_numerical_1_max                      0.005986
    new_numerical_1_mean_diff                 0.0059282
    hist_installments_max                     0.0058908
    hist_avg_sales_lag3_min                   0.0058696
    new_avg_sales_lag6_mean                   0.0058559
    hist_purchase_amount_mean_diff            0.0058548
    new_purchase_amount_min                   0.0058373
    hist_numerical_2_period                   0.0057976
    hist_numerical_2_max                      0.0057563
    hist_numerical_1_skew                     0.0056678
    hist_ref_date_std                         0.0056007
    hist_avg_purchases_lag3_min               0.0055813
    hist_avg_sales_lag6_max                   0.0055765
    new_category_2_5.0_mean                   0.0055583
    hist_active_months_lag12_mean             0.0054629
    new_avg_purchases_lag6_mean               0.0054613
    hist_purchase_day_nunique                 0.0054522
    hist_avg_purchases_lag6_max               0.0054462
    hist_active_months_lag12_skew             0.005362
    new_installments_skew                     0.0053338
    new_avg_purchases_lag12_mean              0.005224
    new_merchant_category_id_mode             0.0051252
    new_category_2_3.0_mean                   0.0050581
    new_category_2_5.0_sum                    0.0050188
    new_avg_purchases_lag3_skew               0.0050157
    hist_most_recent_sales_range_entropy      0.0050028
    new_time_since_first_active_min           0.0049678
    hist_month_lag_std                        0.0049616
    hist_avg_purchases_lag3_max               0.0049593
    hist_avg_sales_lag3_max                   0.0048811
    new_avg_sales_lag3_mean                   0.0048604
    hist_active_months_lag12_std              0.0047854
    hist_numerical_1_std                      0.0047417
    hist_most_recent_purchases_range_entropy  0.0047181
    hist_purchase_time_std                    0.0046897
    hist_time_since_first_active_std          0.0046897
    hist_purchase_amount_skew                 0.0046872
    hist_numerical_2_mean_diff                0.0046632
    hist_category_2_1.0_mean                  0.0046537
    hist_installments_period                  0.0045897
    hist_numerical_2_std                      0.0045355
    hist_numerical_1_mean_diff                0.004512
    hist_avg_sales_lag3_std                   0.0044769
    hist_ref_date_skew                        0.0044716
    hist_purchase_amount_std                  0.0044703
    hist_avg_sales_lag12_mean                 0.0044191
    hist_avg_purchases_lag12_mean             0.0044072
    hist_purchase_month_mean                  0.0044025
    hist_purchase_week_mean                   0.0043976
    new_avg_purchases_lag3_mean               0.004365
    hist_avg_purchases_lag3_std               0.0043399
    hist_purchase_amount_mean                 0.0042991
    new_subsector_id_mode                     0.0042553
    new_avg_purchases_lag12_skew              0.00424
    hist_purchase_amount_period               0.0042385
    hist_category_3_2.0_sum                   0.0042169
    hist_subsector_id_mode                    0.0041857
    hist_avg_purchases_lag6_mean              0.0041039
    hist_active_months_lag12_min              0.0040995
    hist_avg_sales_lag6_mean                  0.0040983
    new_avg_purchases_lag6_skew               0.0040965
    hist_numerical_1_mean                     0.0040761
    hist_time_since_first_active_max          0.0040418
    hist_category_4_mean                      0.0040307
    hist_purchase_amount_max                  0.0040161
    new_category_2_4.0_sum                    0.0039832
    hist_purchase_week_skew                   0.003942
    hist_most_recent_sales_range_mean         0.0039258
    new_avg_sales_lag6_skew                   0.0039179
    new_avg_sales_lag12_skew                  0.0038884
    hist_avg_purchases_lag3_skew              0.0038856
    hist_purchase_month_skew                  0.0038579
    hist_avg_sales_lag6_std                   0.0038449
    new_avg_sales_lag3_skew                   0.0038093
    hist_active_months_lag12_period           0.0037645
    hist_numerical_2_mean                     0.0037435
    hist_avg_purchases_lag6_std               0.0037286
    new_category_1_nunique                    0.0037167
    hist_category_1_sum                       0.0036509
    hist_avg_purchases_lag3_mean              0.0035984
    new_category_3_0.0_mean                   0.0035965
    new_time_since_first_active_mean          0.0035515
    hist_most_recent_purchases_range_mean     0.0035373
    hist_avg_purchases_lag12_std              0.0034711
    hist_purchase_hour_mean                   0.0034215
    hist_avg_sales_lag3_mean                  0.003387
    hist_avg_sales_lag12_std                  0.0033672
    hist_merchant_category_id_mode            0.0033652
    new_category_2_5.0_nunique                0.0033255
    new_category_2_4.0_mean                   0.0032615
    hist_state_id_nunique                     0.0032521
    hist_state_id_mode                        0.0031744
    hist_city_id_mode                         0.0031466
    hist_avg_sales_lag3_skew                  0.0030995
    hist_category_2_5.0_sum                   0.0030599
    hist_purchase_amount_min                  0.0030229
    hist_category_2_5.0_mean                  0.0029206
    new_time_since_first_active_max           0.002807
    new_active_months_lag6_mean_diff          0.0027727
    hist_category_3_1.0_nunique               0.0027499
    hist_avg_purchases_lag6_skew              0.0026864
    feature_1                                 0.0026344
    hist_avg_sales_lag6_skew                  0.0025855
    hist_category_2_3.0_sum                   0.0025425
    hist_category_4_nunique                   0.0025078
    hist_category_3_2.0_nunique               0.0024532
    new_active_months_lag6_std                0.002429
    new_category_2_3.0_nunique                0.0024156
    new_purchase_hour_mode                    0.002373
    hist_time_since_first_active_min          0.0023379
    new_category_1_sum                        0.0023012
    hist_time_since_first_active_mean         0.0022897
    hist_avg_sales_lag12_skew                 0.0022693
    hist_city_id_entropy                      0.0022269
    new_category_2_2.0_mean                   0.002184
    hist_category_2_3.0_mean                  0.0021569
    new_most_recent_purchases_range_mode      0.0021251
    hist_purchase_day_mean                    0.0021149
    hist_merchant_id_mode                     0.002114
    hist_avg_purchases_lag12_skew             0.0021017
    new_state_id_mode                         0.0020227
    hist_category_2_1.0_nunique               0.0019786
    new_city_id_mode                          0.0019017
    new_most_recent_sales_range_mode          0.0018793
    hist_state_id_entropy                     0.0018788
    hist_active_months_lag6_mean_diff         0.001866
    hist_category_2_4.0_sum                   0.0018313
    hist_merchant_group_id_mode               0.001791
    new_active_months_lag6_period             0.0017701
    hist_purchase_weekend_mean                0.0017441
    hist_active_months_lag6_std               0.0017381
    new_category_2_2.0_sum                    0.001606
    hist_category_2_4.0_mean                  0.0014574
    new_category_2_4.0_nunique                0.001447
    hist_category_2_2.0_mean                  0.0014226
    new_active_months_lag12_max               0.0013094
    new_active_months_lag6_min                0.0013084
    new_active_months_lag6_mean               0.0013084
    hist_category_2_5.0_nunique               0.0012898
    hist_category_1_nunique                   0.0012347
    feature_2                                 0.0012139
    hist_purchase_hour_mode                   0.0011777
    hist_purchase_day_mode                    0.0011775
    feature_3                                 0.0011729
    hist_active_months_lag3_mean_diff         0.0011687
    hist_category_2_2.0_sum                   0.0011176
    new_purchase_day_mode                     0.0011036
    hist_active_months_lag3_std               0.001044
    hist_authorized_flag_nunique              0.00092792
    new_active_months_lag3_std                0.00086299
    new_active_months_lag6_skew               0.00080047
    new_category_2_2.0_nunique                0.00077771
    hist_most_recent_purchases_range_mode     0.00077108
    hist_most_recent_sales_range_mode         0.00068015
    hist_purchase_weekend_nunique             0.00063084
    hist_active_months_lag6_period            0.00061179
    hist_numerical_1_min                      0.00058408
    new_active_months_lag3_mean_diff          0.00048793
    hist_category_3_0.0_nunique               0.00047484
    hist_active_months_lag6_min               0.00043478
    hist_active_months_lag6_mean              0.00043478
    hist_active_months_lag6_skew              0.00043441
    hist_numerical_2_min                      0.00029645
    new_category_3_0.0_nunique                0.00027655
    hist_category_2_3.0_nunique               0.00024557
    hist_active_months_lag3_period            0.00024185
    hist_active_months_lag12_max              0.00023522
    new_active_months_lag6_max                0.00023134
    hist_active_months_lag3_skew              0.00019978
    hist_active_months_lag3_min               0.00019977
    hist_active_months_lag3_mean              0.00019977
    new_active_months_lag3_period             0.00014555
    hist_category_2_4.0_nunique               0.00012009
    hist_category_2_2.0_nunique               0.00010438
    new_active_months_lag3_skew               8.4217e-05
    new_active_months_lag3_min                8.4195e-05
    new_active_months_lag3_mean               8.4195e-05
    new_active_months_lag3_max                3.6816e-05          
    
  </code>
  </pre>
  </div>
</div>
    
<br />


Let's only bother keeping the features with the top 200 mutual information scores.


```python
# Create DataFrame with scores
mi_df = pd.DataFrame()
mi_df['Column'] = cols
mi_df['mut_info'] = mis

# Sort by mutual information
mi_df = mi_df.sort_values('mut_info', ascending=False)
top200 = mi_df.iloc[:200,:]
top200 = top200['Column'].tolist()

# Keep only top 200 columns
X_train = X_train[top200]
X_test = X_test[top200]
```


### Permutation-based Feature Importance

A different way to select features is to try and train a model using *all* the features, and then determine how heavily the model's performance depends on the features.  But, we'll need to use a model which can handle a lot of features without overfitting too badly (i.e., an unregularized linear regression wouldn't be a good idea here).  So, we'll use a gradient boosted decision tree, specifically [CatBoost](http://catboost.ai/).  

Let's create a data processing and prediction pipeline.  First, we'll target encode the categorical columns (basically just set each category to the mean target value for samples having that category - see my [previous post](https://brendanhasz.github.io/2019/03/04/target-encoding.html) on target encoding).  Then, we can normalize the data and impute missing data (we'll just fill in missing data with the median of the column).  Finally, we can use CatBoost to predict the loyalty scores from the features we've engineered.


```python
# Regression pipeline
cat_cols = [c for c in X_train if 'mode' in c] 
reg_pipeline = Pipeline([
    ('target_encoder', TargetEncoderCV(cols=cat_cols)),
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', CatBoostRegressor(loss_function='RMSE', 
                                    verbose=False))
])
```

We can measure how heavily the model depends on various features by using permutation-based feature importance.  Basically, we train the model on all the data, and then measure its error after shuffling each row.  When the model's error increases a lot after shuffling a row, that means that the feature which was shuffled was important for the model's predictions.

The advantage of permutation-based feature importance is that it gives a super-clear view and a single score as to how important each feature is.  The downside is that this score is intrinsically linked to the model.  Whereas computing the mutual information between the features and the target only depends on the data, permutation-based feature importance scores depend on the data, the model being used, and the interaction between the two.  If your model can't fit the data very well, your permutation scores will be garbage!

Luckily CatBoost nearly always does a pretty good job of prediction, even in the face of lots of features!  So, let's compute the permutation-based feature importance for each feature (the complete code is [on my GitHub](https://github.com/brendanhasz/dsutils/blob/master/src/dsutils/evaluation.py#L126)).


```python
# Compute the cross-validated feature importance
imp_df = permutation_importance_cv(
    X_train, y_train, reg_pipeline, 'rmse')
```


Then, we can plot the importance scores for each feature.  These scores are just the difference between the model's error with no shuffled features and the error with the feature of interest shuffled.  So, larger scores correspond to features which the model needs to have a low error.


```python
# Plot the feature importances
plt.figure(figsize=(8, 100))
plot_permutation_importance(imp_df)
plt.show()
```

![svg](/assets/img/loyalty-prediction-2-features/importances.svg)


Finally, we'll want to save the features so that we can use them to train a model to predict the loyalty scores.  Let's save the top 100 most important features to a [feather](https://github.com/wesm/feather) file, so that we can quickly load them back in when we do the modeling.  First though, we need to figure out which features *are* the ones with the best importance scores.

```python
# Get top 100 most important features
df = pd.melt(imp_df, var_name='Feature', value_name='Importance')
dfg = (df.groupby(['Feature'])['Importance']
       .aggregate(np.mean)
       .reset_index()
       .sort_values('Importance', ascending=False))
top100 = dfg['Feature'][:100].tolist()
```

Then, we can save those features (and the corresponding target variable!) to a feather file.

```python
# Save file w/ 100 most important features
cards = pd.concat([X_train[top100], X_test[top100]])
cards['target'] = y_train
cards.reset_index(inplace=True)
cards.to_feather('card_features_top100.feather')
```


## Conclusion

Now that we've engineered features for each card account, the next thing to do is create models to predict the target value from those features.  In a future post, we'll try different modeling techniques to see which gives the best predictions.
