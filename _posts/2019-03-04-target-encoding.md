---
layout: post
title: "Representing Categorical Data with Target Encoding"
date: 2019-03-04
description: "Representing categorical variables with high cardinality using target encoding, and mitigating overfitting often seen with target encoding by using cross-fold and leave-one-out schemes."
img_url: /assets/img/target-encoding/cross-fold.gif
github_url: https://github.com/brendanhasz/target-encoding
tags: [python, prediction]
comments: true
---

Most machine learning algorithms require the input data to be a numeric matrix, where each row is a sample and each column is a feature.  This makes sense for continuous features, where a larger number obviously corresponds to a larger value (features such as voltage, purchase amount, or number of clicks).  How to represent categorical features is less obvious.  Categorical features (such as state, merchant ID, domain name, or phone number) don't have an intrinsic ordering, and so most of the time we can't just represent them with random numbers.  Who's to say that Colorado is "greater than" Minnesota?  Or DHL "less than" FedEx?  To represent categorical data, we need to find a way to encode the categories numerically.

There are quite a few ways to encode categorical data.  We can simply assign each category an integer randomly (called label encoding).  Alternatively, we can create a new feature for each possible category, and set the feature to be 1 for each sample having that category, and otherwise set it to be 0 (called one-hot encoding).  If we're using neural networks, we could let our network learn the embeddings of categories in a high-dimensional space (called entity embedding, or in neural NLP models often just "embedding").

However, these methods all have drawbacks.  Label encoding doesn't work well at all with non-ordinal categorical features.  One-hot encoding leads to a humongous number of added features when your data contains a large number of categories.  Entity embedding can only be used with neural network models (or at least with models which are trained using stochastic gradient descent).

A different encoding method which we'll try in this post is called target encoding (also known as "mean encoding", and really should probably be called "mean target encoding").  With target encoding, each category is replaced with the mean target value for samples having that category.  The "target value" is the y-variable, or the value our model is trying to predict.  This allows us to encode an arbitrary number of categories without increasing the dimensionality of our data!  

Of course, there are drawbacks to target encoding as well.  Target encoding introduces noise into the encoding of the categorical variables (noise which comes from the noise in the target variable itself).  Also, naively applying target encoding can allow data leakage, leading to overfitting and poor predictive performance.  To fix that problem, we'll have to construct target encoders which prevent data leakage.  And even with those leak-proof target encoders, there are situations where one would be better off using one-hot or other encoding methods.  One-hot can be better in situations with few categories, or with data where there are strong interaction effects.

In this post we'll evaluate different encoding schemes, build a cross-fold target encoder to mitigate the drawbacks of the naive target encoder, and determine how the performance of predictive models change based on the type of category encoding used, the number of categories in the dataset, and the presence of interaction effects.

**Outline**

- [Data](#data)
- [Baseline](#baseline)
- [Label Encoding](#label-encoding)
- [One-hot Encoding](#one-hot-encoding)
- [Target Encoding](#target-encoding)
- [Cross-Fold Target Encoding](#cross-fold-target-encoding)
- [Leave-one-out Target Encoding](#leave-one-out-target-encoding)
- [Effect of the Learning Algorithm](#effect-of-the-learning-algorithm)
- [Dependence on the Number of Categories](#dependence-on-the-number-of-categories)
- [Effect of Category Imbalance](#effect-of-category-imbalance)
- [Effect of Interactions](#effect-of-interactions)
- [Suggestions](#suggestions)


First let's import the packages we'll be using.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor

np.random.seed(12345)

%matplotlib inline
%config InlineBackend.figure_format = 'svg'
```

## Data

To evaluate the effectiveness of different encoding algorithms, we'll want to be able to generate data with different numbers of samples, features, and categories.  Let's make a function to generate categorical datasets, which allows us to set these different aspects of the data.  The categories have a direct effect on the target variable which we'll try to predict.


```python
def make_categorical_regression(n_samples=100,
                                n_features=10,
                                n_informative=10,
                                n_categories=10,
                                imbalance=0.0,
                                noise=1.0,
                                n_cont_features=0,
                                cont_weight=0.1,
                                interactions=0.0):
    """Generate a regression problem with categorical features.
  
    Parameters
    ----------
    n_samples : int > 0
        Number of samples to generate
        Default = 100
    n_features : int > 0
        Number of categorical features to generate
        Default = 10
    n_informative : int >= 0
        Number of features to carry information about the target.
        Default = 10
    n_categories : int > 0
        Number of categories per feature.  Default = 10
    imbalance : float > 0
        How much imbalance there is in the number of occurrences of
        each category.  Larger values yield a higher concentration
        of samples in only a few categories.  An imbalance of 0 
        yields the same number of samples in each category.
        Default = 0.0
    noise : float > 0
        Noise to add to target.  Default = 1.0
    n_cont_features : int >= 0
        Number of continuous (non-categorical) features.
        Default = 0
    cont_weight : float > 0
        Weight of the continuous variables' effect.
        Default = 0.1
    interactions : float >= 0 and <= 1
        Proportion of the variance due to interaction effects.
        Note that this only adds interaction effects between the 
        categorical features, not the continuous features.
        Default = 0.0
        
    Returns
    -------
    X : pandas DataFrame
        Features.  Of shape (n_samples, n_features+n_cont_features)
    y : pandas Series of shape (n_samples,)
        Target variable.
    """
    
    
    def beta_binomial(n, a, b):
        """Beta-binomial probability mass function.
        
        Parameters
        ----------
        n : int
            Number of trials
        a : float > 0
            Alpha parameter
        b : float > 0
            Beta parameter
            
        Returns
        -------
        ndarray of size (n,)
            Probability mass function.
        """
        from scipy.special import beta
        from scipy.misc import comb
        k = np.arange(n+1)
        return comb(n, k)*beta(k+a, n-k+b)/beta(a, b)


    # Check inputs
    if not isinstance(n_samples, int):
        raise TypeError('n_samples must be an int')
    if n_samples < 1:
        raise ValueError('n_samples must be one or greater')
    if not isinstance(n_features, int):
        raise TypeError('n_features must be an int')
    if n_features < 1:
        raise ValueError('n_features must be one or greater')
    if not isinstance(n_informative, int):
        raise TypeError('n_informative must be an int')
    if n_informative < 0:
        raise ValueError('n_informative must be non-negative')
    if not isinstance(n_categories, int):
        raise TypeError('n_categories must be an int')
    if n_categories < 1:
        raise ValueError('n_categories must be one or greater')
    if not isinstance(imbalance, float):
        raise TypeError('imbalance must be a float')
    if imbalance < 0:
        raise ValueError('imbalance must be non-negative')
    if not isinstance(noise, float):
        raise TypeError('noise must be a float')
    if noise < 0:
        raise ValueError('noise must be positive')
    if not isinstance(n_cont_features, int):
        raise TypeError('n_cont_features must be an int')
    if n_cont_features < 0:
        raise ValueError('n_cont_features must be non-negative')
    if not isinstance(cont_weight, float):
        raise TypeError('cont_weight must be a float')
    if cont_weight < 0:
        raise ValueError('cont_weight must be non-negative')
    if not isinstance(interactions, float):
        raise TypeError('interactions must be a float')
    if interactions < 0:
        raise ValueError('interactions must be non-negative')
        
    # Generate random categorical data (using category probs drawn
    # from a beta-binomial dist w/ alpha=1, beta=imbalance+1)
    cat_probs = beta_binomial(n_categories-1, 1.0, imbalance+1)
    categories = np.empty((n_samples, n_features), dtype='uint64')
    for iC in range(n_features):
        categories[:,iC] = np.random.choice(np.arange(n_categories),
                                            size=n_samples,
                                            p=cat_probs)
        
    # Generate random values for each category
    cat_vals = np.random.randn(n_categories, n_features)
    
    # Set non-informative columns' effect to 0
    cat_vals[:,:(n_features-n_informative)] = 0
    
    # Compute target variable from categories and their values
    y = np.zeros(n_samples)
    for iC in range(n_features):
        y += (1.0-interactions) * cat_vals[categories[:,iC], iC]
      
    # Add interaction effects
    if interactions > 0:
        for iC1 in range(n_informative):
            for iC2 in range(iC1+1, n_informative):
                int_vals = np.random.randn(n_categories,
                                           n_categories)
                y += interactions * int_vals[categories[:,iC1],
                                             categories[:,iC2]]
    
    # Add noise
    y += noise*np.random.randn(n_samples)
    
    # Generate dataframe from categories
    cat_strs = [''.join([chr(ord(c)+49) for c in str(n)]) 
                for n in range(n_categories)]
    X = pd.DataFrame()
    for iC in range(n_features):
        col_str = 'categorical_'+str(iC)
        X[col_str] = [cat_strs[i] for i in categories[:,iC]]
        
    # Add continuous features
    for iC in range(n_cont_features):
        col_str = 'continuous_'+str(iC)
        X[col_str] = cont_weight*np.random.randn(n_samples)
        y += np.random.randn()*X[col_str]
                    
    # Generate series from target
    y = pd.Series(data=y, index=X.index)
    
    # Return features and target
    return X, y
```

Now, we can easily generate data to test our encoders on:


```python
# Generate categorical data and target
X, y = make_categorical_regression(n_samples=2000,
                                   n_features=10,
                                   n_categories=100,
                                   n_informative=1,
                                   imbalance=2.0)

# Split into test and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5)
```


The ten features in the dataset we generated are all categorical:


```python
X_train.sample(10)
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
      <th>categorical_0</th>
      <th>categorical_1</th>
      <th>categorical_2</th>
      <th>categorical_3</th>
      <th>categorical_4</th>
      <th>categorical_5</th>
      <th>categorical_6</th>
      <th>categorical_7</th>
      <th>categorical_8</th>
      <th>categorical_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>792</th>
      <td>cf</td>
      <td>c</td>
      <td>d</td>
      <td>a</td>
      <td>ed</td>
      <td>ca</td>
      <td>dj</td>
      <td>g</td>
      <td>b</td>
      <td>bj</td>
    </tr>
    <tr>
      <th>276</th>
      <td>di</td>
      <td>b</td>
      <td>bd</td>
      <td>fg</td>
      <td>d</td>
      <td>j</td>
      <td>e</td>
      <td>a</td>
      <td>hc</td>
      <td>h</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>ei</td>
      <td>di</td>
      <td>cj</td>
      <td>he</td>
      <td>hb</td>
      <td>gh</td>
      <td>b</td>
      <td>bh</td>
      <td>df</td>
      <td>c</td>
    </tr>
    <tr>
      <th>1372</th>
      <td>ca</td>
      <td>c</td>
      <td>be</td>
      <td>ce</td>
      <td>cg</td>
      <td>bf</td>
      <td>de</td>
      <td>fe</td>
      <td>ba</td>
      <td>fd</td>
    </tr>
    <tr>
      <th>1860</th>
      <td>db</td>
      <td>dh</td>
      <td>ba</td>
      <td>bh</td>
      <td>di</td>
      <td>bh</td>
      <td>db</td>
      <td>bi</td>
      <td>gf</td>
      <td>bi</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>h</td>
      <td>ce</td>
      <td>ea</td>
      <td>i</td>
      <td>eb</td>
      <td>g</td>
      <td>da</td>
      <td>da</td>
      <td>fc</td>
      <td>e</td>
    </tr>
    <tr>
      <th>328</th>
      <td>j</td>
      <td>db</td>
      <td>df</td>
      <td>fa</td>
      <td>fe</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>da</td>
      <td>bg</td>
    </tr>
    <tr>
      <th>1708</th>
      <td>cd</td>
      <td>ci</td>
      <td>f</td>
      <td>be</td>
      <td>e</td>
      <td>fb</td>
      <td>dc</td>
      <td>bi</td>
      <td>ec</td>
      <td>da</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>ei</td>
      <td>cj</td>
      <td>ch</td>
      <td>bc</td>
      <td>bb</td>
      <td>f</td>
      <td>ch</td>
      <td>bi</td>
      <td>c</td>
      <td>he</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>bi</td>
      <td>bh</td>
      <td>bf</td>
      <td>ba</td>
      <td>dc</td>
      <td>da</td>
      <td>g</td>
      <td>cc</td>
      <td>bi</td>
      <td>ee</td>
    </tr>
  </tbody>
</table>
</div>



Using the pandas package, these are stored as the "object" datatype:


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1000 entries, 523 to 583
    Data columns (total 10 columns):
    categorical_0    1000 non-null object
    categorical_1    1000 non-null object
    categorical_2    1000 non-null object
    categorical_3    1000 non-null object
    categorical_4    1000 non-null object
    categorical_5    1000 non-null object
    categorical_6    1000 non-null object
    categorical_7    1000 non-null object
    categorical_8    1000 non-null object
    categorical_9    1000 non-null object
    dtypes: object(10)
    memory usage: 85.9+ KB
    

While all the features are categorical, the target variable is continuous:


```python
y_train.hist(bins=20)
plt.xlabel('Target value')
plt.ylabel('Number of samples')
plt.show()
```


![svg](/assets/img/target-encoding/output_11_1.svg)


Now the question is: which encoding scheme best allows us to glean the most information from the categorical features, leading to the best predictions of the target variable?


## Baseline

For comparison, how well would we do if we just predicted the mean target value for all samples?  We'll use the mean absolute error (MAE) as our performance metric.


```python
mean_absolute_error(y_train, 
                    np.full(y_train.shape[0], y_train.mean()))
```

    1.139564825988808


So our predictive models should definitely be shooting for a mean absolute error of less than that!  But, we added random noise with a standard deviation of 1, so even if our model is *perfect*, the best MAE we can expect is:


```python
mean_absolute_error(np.random.randn(10000), 
                    np.zeros(10000))
```

    0.7995403442995148


## Label Encoding

The simplest categorical encoding method is label encoding, where each category is simply replaced with a unique integer.  However, there is no intrinsic relationship between the categories and the numbers being used to replace them.  In the diagram below, category A is replaced with 0, and B with 1 - but there is no reason to think that category A is somehow greater than category B.


<iframe src="/assets/img/target-encoding/LabelEncoding.html" style="border:none;overflow:hidden;" width="599" height="480"></iframe>


We'll create a [scikit-learn](https://scikit-learn.org/stable/index.html)-compatible transformer class with which to label encode our data.  Note that we could instead just use [scikit-learn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) - although their version is a little wasteful in that it doesn't choose a data type efficiently.


```python
class LabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder.
    
    Replaces categorical column(s) with integer labels
    for each unique category in original column.

    """
    
    def __init__(self, cols=None):
        """Label encoder.
        
        Parameters
        ----------
        cols : list of str
            Columns to label encode.  Default is to label 
            encode all categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit label encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to label encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [c for c in X if str(X[c].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Create the map from objects to integers for each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            self.maps[col] = dict(zip(
                X[col].values, 
                X[col].astype('category').cat.codes.values
            ))
                        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the label encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to label encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
          
            # Map the column
            Xo[col] = Xo[col].map(tmap)
            
            # Convert to appropriate datatype
            max_val = max(tmap.values())
            if Xo[col].isnull().any(): #nulls, so use float!
                if max_val < 8388608:
                    dtype = 'float32'
                else:
                    dtype = 'float64'
            else:
                if max_val < 256:
                    dtype = 'uint8'
                elif max_val < 65536:
                    dtype = 'uint16'
                elif max_val < 4294967296:
                    dtype = 'uint32'
                else:
                    dtype = 'uint64'
            Xo[col] = Xo[col].astype(dtype)
            
        # Return encoded dataframe
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via label encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to label encode
        y : pandas Series, shape = [n_samples]
            Target values

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
```

Now we can convert the categories to integers:


```python
# Label encode the categorical data
le = LabelEncoder()
X_label_encoded = le.fit_transform(X_train, y_train)
X_label_encoded.sample(10)
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
      <th>categorical_0</th>
      <th>categorical_1</th>
      <th>categorical_2</th>
      <th>categorical_3</th>
      <th>categorical_4</th>
      <th>categorical_5</th>
      <th>categorical_6</th>
      <th>categorical_7</th>
      <th>categorical_8</th>
      <th>categorical_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>884</th>
      <td>13</td>
      <td>40</td>
      <td>83</td>
      <td>5</td>
      <td>82</td>
      <td>32</td>
      <td>44</td>
      <td>6</td>
      <td>23</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>15</td>
      <td>66</td>
      <td>34</td>
      <td>70</td>
      <td>56</td>
      <td>36</td>
      <td>69</td>
      <td>3</td>
      <td>86</td>
      <td>48</td>
    </tr>
    <tr>
      <th>853</th>
      <td>56</td>
      <td>6</td>
      <td>12</td>
      <td>30</td>
      <td>27</td>
      <td>25</td>
      <td>56</td>
      <td>10</td>
      <td>54</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1667</th>
      <td>17</td>
      <td>6</td>
      <td>77</td>
      <td>8</td>
      <td>22</td>
      <td>65</td>
      <td>22</td>
      <td>33</td>
      <td>5</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>56</td>
      <td>5</td>
      <td>8</td>
      <td>34</td>
      <td>0</td>
      <td>31</td>
      <td>48</td>
      <td>80</td>
      <td>50</td>
      <td>56</td>
    </tr>
    <tr>
      <th>362</th>
      <td>0</td>
      <td>15</td>
      <td>47</td>
      <td>50</td>
      <td>42</td>
      <td>7</td>
      <td>9</td>
      <td>3</td>
      <td>81</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>21</td>
      <td>19</td>
      <td>22</td>
      <td>56</td>
      <td>56</td>
      <td>13</td>
      <td>1</td>
      <td>43</td>
      <td>13</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1784</th>
      <td>35</td>
      <td>66</td>
      <td>37</td>
      <td>66</td>
      <td>4</td>
      <td>1</td>
      <td>76</td>
      <td>8</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>127</th>
      <td>82</td>
      <td>42</td>
      <td>11</td>
      <td>63</td>
      <td>12</td>
      <td>39</td>
      <td>58</td>
      <td>76</td>
      <td>59</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>2</td>
      <td>66</td>
      <td>35</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>45</td>
      <td>24</td>
      <td>23</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



But again, these integers aren't related to the categories in any meaningful way - aside from the fact that each unique integer corresponds to a unique category.

We can create a processing pipeline that label-encodes the data, and then uses a Bayesian ridge regression to predict the target variable, and compute the cross-validated mean absolute error of that model.


```python
# Regression model
model_le = Pipeline([
    ('label-encoder', LabelEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
mae_scorer = make_scorer(mean_absolute_error)
scores = cross_val_score(model_le, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('Cross-validated MAE: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    Cross-validated MAE: 1.132 +/- 0.022
    

That's not much better than just predicting the mean!

The error is similarly poor on validation data.


```python
# MAE on test data
model_le.fit(X_train, y_train)
y_pred = model_le.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print('Validation MAE: %0.3f' % test_mae)
```

    Validation MAE: 1.176
    

## One-hot Encoding

One-hot encoding, sometimes called "dummy coding", encodes the categorical information a little more intelligently.  Instead of assigning random integers to categories, a new feature is created for each category.  For each sample, the new feature is 1 if the sample's category matches the new feature, otherwise the value is 0.  This allows us to encode the categorical information numerically, without loss of information, but ends up adding a lot of columns when the original categorical feature has many unique categories.


<iframe src="/assets/img/target-encoding/OneHotEncoding.html" style="border:none;overflow:hidden;" width="599" height="480"></iframe>


Like before, we'll create an sklearn transformer class to perform one-hot encoding.  And again we could have used sklearn's built-in [OneHotEncoder class](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).


```python
class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoder.
    
    Replaces categorical column(s) with binary columns 
    for each unique value in original column.

    """
    
    def __init__(self, cols=None, reduce_df=False):
        """One-hot encoder.
        
        Parameters
        ----------
        cols : list of str
            Columns to one-hot encode.  Default is to one-hot 
            encode all categorical columns in the DataFrame.
        reduce_df : bool
            Whether to use reduced degrees of freedom for encoding
            (that is, add N-1 one-hot columns for a column with N 
            categories). E.g. for a column with categories A, B, 
            and C: When reduce_df is True, A=[1, 0], B=[0, 1],
            and C=[0, 0].  When reduce_df is False, A=[1, 0, 0], 
            B=[0, 1, 0], and C=[0, 0, 1]
            Default = False
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.reduce_df = reduce_df
        
        
    def fit(self, X, y):
        """Fit one-hot encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [c for c in X 
                         if str(X[c].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Store each unique value
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            self.maps[col] = []
            uniques = X[col].unique()
            for unique in uniques:
                self.maps[col].append(unique)
            if self.reduce_df:
                del self.maps[col][-1]
        
        # Return fit object
        return self

        
    def transform(self, X, y=None):
        """Perform the one-hot encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to one-hot encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, vals in self.maps.items():
            for val in vals:
                new_col = col+'_'+str(val)
                Xo[new_col] = (Xo[col]==val).astype('uint8')
            del Xo[col]
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via one-hot encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to one-hot encode
        y : pandas Series, shape = [n_samples]
            Target values

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
```

Now, instead of replacing categories with integer labels, we've create a new column for each category in each original column.  The value in a given column is 1 when the original category matches, otherwise the value is 0.  The values in the dataframe below are mostly 0s because the data we generated has so many categories.


```python
# One-hot-encode the categorical data
ohe = OneHotEncoder()
X_one_hot = ohe.fit_transform(X_train, y_train)
X_one_hot.sample(10)
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
      <th>categorical_0_ec</th>
      <th>categorical_0_ba</th>
      <th>categorical_0_bg</th>
      <th>categorical_0_b</th>
      <th>categorical_0_h</th>
      <th>categorical_0_j</th>
      <th>categorical_0_ge</th>
      <th>categorical_0_cg</th>
      <th>categorical_0_fh</th>
      <th>categorical_0_dc</th>
      <th>...</th>
      <th>categorical_9_ga</th>
      <th>categorical_9_eb</th>
      <th>categorical_9_gg</th>
      <th>categorical_9_hj</th>
      <th>categorical_9_gi</th>
      <th>categorical_9_he</th>
      <th>categorical_9_ff</th>
      <th>categorical_9_hb</th>
      <th>categorical_9_gd</th>
      <th>categorical_9_fi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>294</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>900</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>39</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>443</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>1268</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>280</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>382</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>624</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>430</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>836</th>
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
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>10 rows × 853 columns</p>
</div>



Note that although we've now encoded the categorical data in a meaningful way, our data matrix is huge!


```python
# Compare sizes
print('Original size:', X_train.shape)
print('One-hot encoded size:', X_one_hot.shape)
```

    Original size: (1000, 10)
    One-hot encoded size: (1000, 853)
    

We can fit the same model with the one-hot encoded data as we fit to the label-encoded data, and compute the cross-validated error.


```python
# Regression model
model_oh = Pipeline([
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
scores = cross_val_score(model_oh, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('Cross-validated MAE: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    Cross-validated MAE: 1.039 +/- 0.028
    

Unlike with label encoding, when using one-hot encoding our predictions are definitely better than just guessing the mean - but not by a whole lot!  Performance on the validation dataset is about the same:


```python
# MAE on test data
model_oh.fit(X_train, y_train)
y_pred = model_oh.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print('Validation MAE: %0.3f' % test_mae)
```

    Validation MAE: 1.029
    

## Target Encoding

The problem with one-hot encoding is that it greatly increases the dimensionality of the training data (by adding a new feature for each unique category in the original dataset).  This often leads to poorer model performance due to the curse of dimensionality - i.e., all else being equal, it is harder for machine learning algorithms to learn from data which has more dimensions.

Target encoding allows us to retain actual useful information about the categories (like one-hot encoding, but unlike label encoding), while keeping the dimensionality of our data the same as the unencoded data (like label encoding, but unlike one-hot encoding).  To target encode data, for each feature, we simply replace each category with the mean target value for samples which have that category.


<iframe src="/assets/img/target-encoding/TargetEncoding.html" style="border:none;overflow:hidden;" width="599" height="480"></iframe>


Let's create a transformer class which performs this target encoding.


```python
class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.

    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target 
            encode all categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X 
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
```

Now, instead of creating a bazillion columns (like with one-hot encoding), we can simply replace each category with the mean target value for that category.  This allows us to represent the categorical information in the same dimensionality, while retaining some information about the categories.  By target-encoding the features matrix, we get a matrix of the same size, but filled with continuous values instead of categories:


```python
# Target encode the categorical data
te = TargetEncoder()
X_target_encoded = te.fit_transform(X_train, y_train)
X_target_encoded.sample(10)
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
      <th>categorical_0</th>
      <th>categorical_1</th>
      <th>categorical_2</th>
      <th>categorical_3</th>
      <th>categorical_4</th>
      <th>categorical_5</th>
      <th>categorical_6</th>
      <th>categorical_7</th>
      <th>categorical_8</th>
      <th>categorical_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>711</th>
      <td>-0.030636</td>
      <td>0.192812</td>
      <td>0.269273</td>
      <td>0.131628</td>
      <td>0.319769</td>
      <td>0.190861</td>
      <td>0.142159</td>
      <td>0.393587</td>
      <td>-0.766454</td>
      <td>-0.312227</td>
    </tr>
    <tr>
      <th>475</th>
      <td>0.365423</td>
      <td>-0.605778</td>
      <td>-0.258930</td>
      <td>0.038321</td>
      <td>-0.283131</td>
      <td>0.046135</td>
      <td>0.316052</td>
      <td>-0.120822</td>
      <td>-0.425927</td>
      <td>-0.163454</td>
    </tr>
    <tr>
      <th>273</th>
      <td>0.110462</td>
      <td>1.093313</td>
      <td>0.309760</td>
      <td>0.474308</td>
      <td>0.090909</td>
      <td>0.003120</td>
      <td>1.558923</td>
      <td>0.244971</td>
      <td>-0.387846</td>
      <td>-0.327537</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.848020</td>
      <td>0.300673</td>
      <td>0.125095</td>
      <td>-0.650361</td>
      <td>-0.252932</td>
      <td>0.293856</td>
      <td>-0.197504</td>
      <td>0.050085</td>
      <td>-0.587633</td>
      <td>-0.413439</td>
    </tr>
    <tr>
      <th>275</th>
      <td>0.126068</td>
      <td>0.180776</td>
      <td>-0.143977</td>
      <td>-0.131238</td>
      <td>0.090909</td>
      <td>-0.760367</td>
      <td>0.326620</td>
      <td>-0.037488</td>
      <td>-0.121713</td>
      <td>-0.244310</td>
    </tr>
    <tr>
      <th>336</th>
      <td>0.543412</td>
      <td>-0.045947</td>
      <td>0.180144</td>
      <td>0.279675</td>
      <td>-0.532591</td>
      <td>0.338287</td>
      <td>0.071977</td>
      <td>0.113531</td>
      <td>0.527567</td>
      <td>1.290724</td>
    </tr>
    <tr>
      <th>1462</th>
      <td>-0.065061</td>
      <td>-0.605778</td>
      <td>0.172445</td>
      <td>-0.268622</td>
      <td>-0.283131</td>
      <td>-0.270112</td>
      <td>-0.197504</td>
      <td>0.068821</td>
      <td>0.371461</td>
      <td>0.966579</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>0.186988</td>
      <td>0.564015</td>
      <td>0.135396</td>
      <td>0.474308</td>
      <td>0.338006</td>
      <td>0.479294</td>
      <td>0.063384</td>
      <td>0.342170</td>
      <td>-0.054090</td>
      <td>-0.163454</td>
    </tr>
    <tr>
      <th>1611</th>
      <td>0.350326</td>
      <td>-0.188197</td>
      <td>-0.537682</td>
      <td>-0.391143</td>
      <td>0.212399</td>
      <td>-1.811086</td>
      <td>0.204642</td>
      <td>-0.622682</td>
      <td>-0.425927</td>
      <td>-0.489371</td>
    </tr>
    <tr>
      <th>166</th>
      <td>0.480637</td>
      <td>0.975462</td>
      <td>0.135396</td>
      <td>-0.131238</td>
      <td>0.323870</td>
      <td>0.279502</td>
      <td>-0.533179</td>
      <td>0.050085</td>
      <td>0.016449</td>
      <td>1.410449</td>
    </tr>
  </tbody>
</table>
</div>



Note that the size of our target-encoded matrix is the same size as the original (unlike the huge one-hot transformed matrix):


```python
# Compare sizes
print('Original size:', X_train.shape)
print('Target encoded size:', X_target_encoded.shape)
```

    Original size: (1000, 10)
    Target encoded size: (1000, 10)
    

Also, each column has exactly as many unique continuous values as it did categories.  This is because we've simply replaced the category with the mean target value for that category.


```python
# Compare category counts
print('Original:')
print(X_train.nunique())
print('\nTarget encoded:')
print(X_target_encoded.nunique())
```

    Original:
    categorical_0    84
    categorical_1    81
    categorical_2    85
    categorical_3    88
    categorical_4    84
    categorical_5    86
    categorical_6    88
    categorical_7    88
    categorical_8    90
    categorical_9    79
    dtype: int64
     
    Target encoded:
    categorical_0    84
    categorical_1    81
    categorical_2    85
    categorical_3    88
    categorical_4    84
    categorical_5    86
    categorical_6    88
    categorical_7    88
    categorical_8    90
    categorical_9    79
    dtype: int64
    

If we fit the same model as before, but now after target-encoding the categories, the error of our model is far lower!


```python
# Regression model
model_te = Pipeline([
    ('encoder', TargetEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
scores = cross_val_score(model_te, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('Cross-validated MAE: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    Cross-validated MAE: 0.940 +/- 0.030
    

The performance on the test data is about the same, but slightly better, because we've given it more samples on which to train.



```python
# MAE on test data
model_te.fit(X_train, y_train)
y_pred = model_te.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print('Validation MAE: %0.3f' % test_mae)
```

    Validation MAE: 0.933
    

While the error is lower using target encoding than with one-hot encoding, in naively target-encoding our categories, we've introduced a data leak from the target variable for one sample into the features for that same sample!  

In the diagram above, notice how the i-th sample's target value is used in the computation of the mean target value for the i-th sample's category, and then the i-th sample's category is replaced with that mean.  Leaking the target variable into our predictors like that causes our learning algorithm to over-depend on the target-encoded features, which results in the algorithm overfitting on the data.  Although we gain predictive power by keeping the dimensionality of our training data reasonable, we loose a lot of that gain by allowing our model to overfit to the target-encoded columns!


## Cross-Fold Target Encoding

To clamp down on the data leakage, we need to ensure that we're not using the using the target value from a given sample to compute its target-encoded values.  However, we can still use *other* samples in the training data to compute the mean target values for *this* sample's category.  

There are a few different ways we can do this.  We could compute the per-category target means in a cross-fold fashion, or by leaving the current sample out (leave-one-out).

First we'll try cross-fold target encoding, where we'll split the data up into \\( N \\) folds, and compute the means for each category in the \\( i \\)-th fold using data in all the other folds.  The diagram below illustrates an example using 2 folds.


<iframe src="/assets/img/target-encoding/TargetEncodingCV.html" style="border:none;overflow:hidden;" width="599" height="480"></iframe>


Let's create a transformer class to perform the cross-fold target encoding.  There are a few things we need to watch out for now which we didn't have to worry about with the naive target encoder.  First, we may end up with NaNs (empty values) even when there were categories in the original dataframe.  This will happen for a category that appears in one fold, but when there are no examples of that category in the other folds.  Also, we can't perform cross-fold encoding on our test data, because we don't have any target values for which to compute the category means!  So, we have to use the category means from the training data in that case.


```python
class TargetEncoderCV(TargetEncoder):
    """Cross-fold target encoder.
    """
    
    def __init__(self, n_splits=3, shuffle=True, cols=None):
        """Cross-fold target encoding for categorical features.
        
        Parameters
        ----------
        n_splits : int
            Number of cross-fold splits. Default = 3.
        shuffle : bool
            Whether to shuffle the data when splitting into folds.
        cols : list of str
            Columns to target encode.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.cols = cols
        

    def fit(self, X, y):
        """Fit cross-fold target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        self._target_encoder = TargetEncoder(cols=self.cols)
        self._target_encoder.fit(X, y)
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses cross-fold target encoding for the training fold,
        and uses normal target encoding for the test fold.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """

        # Use target encoding from fit() if this is test data
        if y is None:
            return self._target_encoder.transform(X)

        # Compute means for each fold
        self._train_ix = []
        self._test_ix = []
        self._fit_tes = []
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        for train_ix, test_ix in kf.split(X):
            self._train_ix.append(train_ix)
            self._test_ix.append(test_ix)
            te = TargetEncoder(cols=self.cols)
            if isinstance(X, pd.DataFrame):
                self._fit_tes.append(te.fit(X.iloc[train_ix,:],
                                            y.iloc[train_ix]))
            elif isinstance(X, np.ndarray):
                self._fit_tes.append(te.fit(X[train_ix,:],
                                            y[train_ix]))
            else:
                raise TypeError('X must be DataFrame or ndarray')

        # Apply means across folds
        Xo = X.copy()
        for ix in range(len(self._test_ix)):
            test_ix = self._test_ix[ix]
            if isinstance(X, pd.DataFrame):
                Xo.iloc[test_ix,:] = \
                    self._fit_tes[ix].transform(X.iloc[test_ix,:])
            elif isinstance(X, np.ndarray):
                Xo[test_ix,:] = \
                    self._fit_tes[ix].transform(X[test_ix,:])
            else:
                raise TypeError('X must be DataFrame or ndarray')
        return Xo

            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
```

With this encoder, we can convert the categories into continuous values, just like we did with the naive target encoding.


```python
# Cross-fold Target encode the categorical data
te = TargetEncoderCV()
X_target_encoded_cv = te.fit_transform(X_train, y_train)
X_target_encoded_cv.sample(10)
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
      <th>categorical_0</th>
      <th>categorical_1</th>
      <th>categorical_2</th>
      <th>categorical_3</th>
      <th>categorical_4</th>
      <th>categorical_5</th>
      <th>categorical_6</th>
      <th>categorical_7</th>
      <th>categorical_8</th>
      <th>categorical_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>236</th>
      <td>0.233017</td>
      <td>0.266851</td>
      <td>0.620411</td>
      <td>-0.0917691</td>
      <td>0.0238002</td>
      <td>0.205387</td>
      <td>-0.182844</td>
      <td>0.209843</td>
      <td>-0.201101</td>
      <td>1.95302</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.0419825</td>
      <td>-0.133423</td>
      <td>-0.152355</td>
      <td>0.768532</td>
      <td>-0.105238</td>
      <td>-0.0010254</td>
      <td>-0.0768051</td>
      <td>-0.164632</td>
      <td>-0.108223</td>
      <td>-1.16817</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.335987</td>
      <td>0.686906</td>
      <td>0.00831367</td>
      <td>0.780618</td>
      <td>0.253249</td>
      <td>-0.588782</td>
      <td>-0.0104415</td>
      <td>-0.139042</td>
      <td>0.258339</td>
      <td>2.08561</td>
    </tr>
    <tr>
      <th>870</th>
      <td>0.251286</td>
      <td>-0.221214</td>
      <td>-0.21522</td>
      <td>-0.528595</td>
      <td>-0.320334</td>
      <td>0.484078</td>
      <td>0.593479</td>
      <td>0.131563</td>
      <td>0.152882</td>
      <td>-0.333682</td>
    </tr>
    <tr>
      <th>357</th>
      <td>-0.105762</td>
      <td>-0.108175</td>
      <td>-0.2422</td>
      <td>-1.08681</td>
      <td>-0.0790136</td>
      <td>-0.367782</td>
      <td>0.287205</td>
      <td>0.542695</td>
      <td>0.064133</td>
      <td>-0.670692</td>
    </tr>
    <tr>
      <th>1372</th>
      <td>0.169969</td>
      <td>0.366924</td>
      <td>0.399639</td>
      <td>-0.0954622</td>
      <td>0.0220233</td>
      <td>-0.588782</td>
      <td>-0.529951</td>
      <td>0.233605</td>
      <td>-0.260713</td>
      <td>-0.130225</td>
    </tr>
    <tr>
      <th>620</th>
      <td>0.372039</td>
      <td>0.110516</td>
      <td>-0.259249</td>
      <td>-0.0814691</td>
      <td>0.294292</td>
      <td>0.705151</td>
      <td>0.300228</td>
      <td>0.227451</td>
      <td>0.185972</td>
      <td>1.53523</td>
    </tr>
    <tr>
      <th>1147</th>
      <td>-0.294882</td>
      <td>0.477974</td>
      <td>0.531971</td>
      <td>0.210054</td>
      <td>-0.171589</td>
      <td>-0.106227</td>
      <td>0.0837924</td>
      <td>-0.201896</td>
      <td>-0.595051</td>
      <td>0.659421</td>
    </tr>
    <tr>
      <th>1650</th>
      <td>-0.882803</td>
      <td>0.647945</td>
      <td>0.177125</td>
      <td>-0.190479</td>
      <td>0.644579</td>
      <td>0.208487</td>
      <td>0.657135</td>
      <td>0.227451</td>
      <td>-0.701029</td>
      <td>-0.00746989</td>
    </tr>
    <tr>
      <th>68</th>
      <td>NaN</td>
      <td>0.831874</td>
      <td>-0.113836</td>
      <td>-0.190479</td>
      <td>-0.475449</td>
      <td>-1.90497</td>
      <td>-0.991536</td>
      <td>0.649424</td>
      <td>-0.326514</td>
      <td>-0.2205</td>
    </tr>
  </tbody>
</table>
</div>



Like with normal target encoding, our transformed matrix is the same shape as the original:


```python
# Compare sizes
print('Original size:', X_train.shape)
print('Target encoded size:', X_target_encoded_cv.shape)
```

    Original size: (1000, 10)
    Target encoded size: (1000, 10)
    

However, now we have more unique continuous values in each column than we did categories, because we've target-encoded the categories separately for each fold (since we used 3 folds, there are about 3 times as many unique values).


```python
# Compare category counts
print('Original:')
print(X_train.nunique())
print('\nTarget encoded:')
print(X_target_encoded_cv.nunique())
```

    Original:
    categorical_0    84
    categorical_1    81
    categorical_2    85
    categorical_3    88
    categorical_4    84
    categorical_5    86
    categorical_6    88
    categorical_7    88
    categorical_8    90
    categorical_9    79
    dtype: int64
     
    Target encoded:
    categorical_0    214
    categorical_1    203
    categorical_2    201
    categorical_3    203
    categorical_4    208
    categorical_5    207
    categorical_6    207
    categorical_7    205
    categorical_8    213
    categorical_9    200
    dtype: int64
    

We can fit the same model as before, but now using cross-fold target encoding.


```python
# Regression model
model_te_cv = Pipeline([
    ('encoder', TargetEncoderCV()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
scores = cross_val_score(model_te_cv, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('Cross-validated MAE: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    Cross-validated MAE: 0.835 +/- 0.044
    

Now our model's error is very low - pretty close to the lower bound of around 0.8!  And the cross-validated performance matches the performance on the validation data.


```python
# MAE on test data
model_te_cv.fit(X_train, y_train)
y_pred = model_te_cv.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print('Validation MAE: %0.3f' % test_mae)
```

    Validation MAE: 0.839
    

## Leave-one-out Target Encoding

We could also prevent the target data leakage by using a leave-one-out scheme.  With this method, we compute the per-category means as with the naive target encoder, but we don't include the current sample in that computation.


<iframe src="/assets/img/target-encoding/TargetEncodingLOO.html" style="border:none;overflow:hidden;" width="599" height="480"></iframe>


This may seem like it will take much longer than the cross-fold method, but it actually ends up being faster, because we can compute the mean without the effect of each sample in an efficient way.  Normally the mean is computed with:

$$
v = \frac{1}{N_C} \sum_{j \in C} y_j
$$

where \\( v \\) is the target-encoded value for all samples having category \\( C \\), \\( N_C \\) is the number of samples having category \\( C \\), and \\( j \in C \\) indicates all the samples which have category \\( C \\).

With leave-one-out target encoding, we can first compute the count of samples having category \\( C \\) ( \\( N_C \\) ), and then separately compute the sum of the target values of those categories:

$$
S_C = \sum_{j \in C} y_j
$$

Then, the mean target value for samples having category \\( C \\), excluding the effect of sample \\( i \\), can be computed with

$$
v_i = \frac{S_C - y_i}{N_C-1}
$$

Let's build a transformer class which performs the leave-one-out target encoding using that trick.


```python
class TargetEncoderLOO(TargetEncoder):
    """Leave-one-out target encoder.
    """
    
    def __init__(self, n_splits=3, shuffle=True, cols=None):
        """Leave-one-out target encoding for categorical features.
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.
        """
        self.cols = cols
        

    def fit(self, X, y):
        """Fit leave-one-out target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to target encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.sum_count = dict()
        for col in self.cols:
            self.sum_count[col] = dict()
            uniques = X[col].unique()
            for unique in uniques:
                ix = X[col]==unique
                self.sum_count[col][unique] = \
                    (y[ix].sum(),ix.sum())
            
        # Return the fit object
        return self

    
    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses leave-one-out target encoding for the training fold,
        and uses normal target encoding for the test fold.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Use normal target encoding if this is test data
        if y is None:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    vals[X[col]==cat] = sum_count[0]/sum_count[1]
                Xo[col] = vals

        # LOO target encode each column
        else:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col]==cat
                    vals[ix] = (sum_count[0]-y[ix])/(sum_count[1]-1)
                Xo[col] = vals
            
        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)
```

Using the leave-one-out target encoder, we can target-encode the data like before:


```python
# Cross-fold Target encode the categorical data
te = TargetEncoderLOO()
X_target_encoded_loo = te.fit_transform(X_train, y_train)
X_target_encoded_loo.sample(10)
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
      <th>categorical_0</th>
      <th>categorical_1</th>
      <th>categorical_2</th>
      <th>categorical_3</th>
      <th>categorical_4</th>
      <th>categorical_5</th>
      <th>categorical_6</th>
      <th>categorical_7</th>
      <th>categorical_8</th>
      <th>categorical_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>915</th>
      <td>-0.012727</td>
      <td>0.314121</td>
      <td>-0.233903</td>
      <td>0.175900</td>
      <td>0.132491</td>
      <td>-1.140074</td>
      <td>0.069677</td>
      <td>0.256593</td>
      <td>-0.430652</td>
      <td>-0.103827</td>
    </tr>
    <tr>
      <th>1647</th>
      <td>0.215025</td>
      <td>0.394832</td>
      <td>-0.449700</td>
      <td>0.600029</td>
      <td>-0.201291</td>
      <td>0.260293</td>
      <td>0.191454</td>
      <td>-2.622372</td>
      <td>0.165401</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>82</th>
      <td>-0.025525</td>
      <td>-0.097241</td>
      <td>0.306800</td>
      <td>0.302967</td>
      <td>0.211843</td>
      <td>-0.075201</td>
      <td>0.487799</td>
      <td>-0.062378</td>
      <td>0.137296</td>
      <td>-0.608199</td>
    </tr>
    <tr>
      <th>328</th>
      <td>-0.022190</td>
      <td>0.183942</td>
      <td>0.237844</td>
      <td>-0.622328</td>
      <td>-0.305189</td>
      <td>0.444699</td>
      <td>0.145842</td>
      <td>0.451886</td>
      <td>0.397333</td>
      <td>0.897512</td>
    </tr>
    <tr>
      <th>369</th>
      <td>0.612622</td>
      <td>-0.219190</td>
      <td>0.430788</td>
      <td>0.595137</td>
      <td>-0.299720</td>
      <td>-0.210595</td>
      <td>-0.073953</td>
      <td>-0.153809</td>
      <td>0.342509</td>
      <td>0.827628</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>0.158465</td>
      <td>-0.104834</td>
      <td>0.117011</td>
      <td>1.055839</td>
      <td>0.071813</td>
      <td>-0.166491</td>
      <td>-0.598097</td>
      <td>0.394334</td>
      <td>-0.909585</td>
      <td>-0.650951</td>
    </tr>
    <tr>
      <th>800</th>
      <td>0.287101</td>
      <td>-0.072489</td>
      <td>0.402740</td>
      <td>-0.044492</td>
      <td>0.157021</td>
      <td>0.542464</td>
      <td>0.601372</td>
      <td>0.215462</td>
      <td>0.184972</td>
      <td>-2.275490</td>
    </tr>
    <tr>
      <th>735</th>
      <td>0.050074</td>
      <td>0.252479</td>
      <td>-0.164370</td>
      <td>-0.143356</td>
      <td>0.126941</td>
      <td>0.417575</td>
      <td>0.044603</td>
      <td>-0.167643</td>
      <td>NaN</td>
      <td>-0.427960</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>0.572234</td>
      <td>0.578534</td>
      <td>-0.286133</td>
      <td>0.142657</td>
      <td>-0.537536</td>
      <td>-0.144131</td>
      <td>1.009878</td>
      <td>0.019303</td>
      <td>-0.524030</td>
      <td>0.880222</td>
    </tr>
    <tr>
      <th>1355</th>
      <td>-0.188658</td>
      <td>-0.107702</td>
      <td>0.052277</td>
      <td>-0.121903</td>
      <td>0.130991</td>
      <td>-0.264774</td>
      <td>0.082149</td>
      <td>0.031939</td>
      <td>0.136631</td>
      <td>0.335117</td>
    </tr>
  </tbody>
</table>
</div>



The transformed matrix is stil the same size as the original:


```python
# Compare sizes
print('Original size:', X_train.shape)
print('Target encoded size:', X_target_encoded_loo.shape)
```

    Original size: (1000, 10)
    Target encoded size: (1000, 10)
    

But now there are nearly as many unique values in each column as there are samples:


```python
# Compare category counts
print('Original:')
print(X_train.nunique())
print('\nLeave-one-out target encoded:')
print(X_target_encoded_loo.nunique())
```

    Original:
    categorical_0    84
    categorical_1    81
    categorical_2    85
    categorical_3    88
    categorical_4    84
    categorical_5    86
    categorical_6    88
    categorical_7    88
    categorical_8    90
    categorical_9    79
    dtype: int64
     
    Leave-one-out target encoded:
    categorical_0    993
    categorical_1    994
    categorical_2    992
    categorical_3    987
    categorical_4    990
    categorical_5    990
    categorical_6    990
    categorical_7    991
    categorical_8    992
    categorical_9    996
    dtype: int64
    

Also, there are less empty values in the leave-one-out target encoded dataframe than there were in the cross-fold target encoded dataframe.  This is because with leave-one-out target encoding, a value will only be null if it is the only category of that type (or if the original feature value was null).


```python
# Compare null counts
print('Original null count:')
print(X_train.isnull().sum())
print('\nCross-fold target encoded null count:')
print(X_target_encoded_cv.isnull().sum())
print('\nLeave-one-out target encoded null count:')
print(X_target_encoded_loo.isnull().sum())
```

    Original null count:
    categorical_0    0
    categorical_1    0
    categorical_2    0
    categorical_3    0
    categorical_4    0
    categorical_5    0
    categorical_6    0
    categorical_7    0
    categorical_8    0
    categorical_9    0
    dtype: int64
     
    Cross-fold target encoded null count:
    categorical_0     9
    categorical_1    12
    categorical_2    22
    categorical_3    21
    categorical_4    12
    categorical_5    15
    categorical_6    19
    categorical_7    20
    categorical_8    23
    categorical_9    15
    dtype: int64
     
    Leave-one-out target encoded null count:
    categorical_0     7
    categorical_1     6
    categorical_2     8
    categorical_3    13
    categorical_4    10
    categorical_5    10
    categorical_6    10
    categorical_7     9
    categorical_8     8
    categorical_9     4
    dtype: int64
    

But more importantly, how well can our model predict the target variable when trained on the leave-one-out target encoded data?


```python
# Regression model
model_te_loo = Pipeline([
    ('encoder', TargetEncoderLOO()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
scores = cross_val_score(model_te_loo, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('Cross-validated MAE: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    Cross-validated MAE: 0.833 +/- 0.038
    


```python
# MAE on test data
model_te_loo.fit(X_train, y_train)
y_pred = model_te_loo.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print('Validation MAE: %0.3f' % test_mae)
```

    Validation MAE: 0.838
    

The leave-one-out target encoder performs *slightly* better than the cross-fold target encoder, because we've given it more samples with which to compute the per-category means (\\( N-1 \\), instead of \\( N-N/K \\), where K is the number of folds).  While the increase in performance was very small, the leave-one-out target encoder is faster, due to the effecient way we computed the leave-one-out means (instead of having to compute means for each fold).


```python
%%time
Xo = TargetEncoderCV().fit_transform(X_train, y_train)
```

    CPU times: user 6.73 s, sys: 118 ms, total: 6.85 s
    Wall time: 6.76 s
    


```python
%%time
Xo = TargetEncoderLOO().fit_transform(X_train, y_train)
```

    CPU times: user 4.25 s, sys: 25 ms, total: 4.27 s
    Wall time: 4.27 s
    

## Effect of the Learning Algorithm

The increase in predictive performance one gets from target encoding depends on the machine learning algorithm which is using it.  As we've seen, target encoding is great for linear models (throughout this post we were using a Bayesian ridge regression, a variant on a linear regression which optimizes the regularization parameter).  However, target encoding doesn't help as much for tree-based boosting algorithms like XGBoost, CatBoost, or LightGBM, which tend to handle categorical data pretty well as-is.

Fitting the Bayesian ridge regression to the data, we see a huge increase in performance after target encoding (relative to one-hot encoding).


```python
# Bayesian ridge w/ one-hot encoding
model_brr = Pipeline([
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
scores = cross_val_score(model_brr, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ Bayesian Ridge + one-hot encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    MAE w/ Bayesian Ridge + one-hot encoding: 1.039 +/- 0.028
    


```python
# Bayesian ridge w/ target-encoding
model_brr = Pipeline([
    ('encoder', TargetEncoderLOO()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Cross-validated MAE
scores = cross_val_score(model_brr, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ Bayesian Ridge + target encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    MAE w/ Bayesian Ridge + target encoding: 0.833 +/- 0.038
    

However, using XGBoost, there is only a modest perfomance increase (if any at all).


```python
# Regression model
model_xgb = Pipeline([
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', XGBRegressor())
])

# Cross-validated MAE
scores = cross_val_score(model_xgb, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ XGBoost + one-hot encoding: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    MAE w/ XGBoost + one-hot encoding: 0.869 +/- 0.040
    


```python
# Regression model
model_xgb = Pipeline([
    ('encoder', TargetEncoderLOO()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', XGBRegressor())
])

# Cross-validated MAE
scores = cross_val_score(model_xgb, X_train, y_train, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ XGBoost + target encoding: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    MAE w/ XGBoost + target encoding: 0.864 +/- 0.052
    

## Dependence on the Number of Categories

There is also an effect of the number of categories on the performance of a model trained on target-encoded data.  Target encoding works well with categorical data that contains a large number of categories.  However, if you have data with only a few categories, you're probably better off using one-hot encoding.

For example, let's generate two datasets: one which has a large number of categories in each column, and another which has only a few categories in each column.


```python
# Categorical data w/ many categories
X_many, y_many = make_categorical_regression(
    n_samples=1000, 
    n_features=10, 
    n_categories=100,
    n_informative=1,
    imbalance=2.0)

# Categorical data w/ few categories
X_few, y_few = make_categorical_regression(
    n_samples=1000, 
    n_features=10, 
    n_categories=5,
    n_informative=1,
    imbalance=2.0)
```
    

Then we'll construct two separate models: one which uses target-encoding, and another which uses one-hot encoding.


```python
# Regression model w/ target encoding
model_te = Pipeline([
    ('encoder', TargetEncoderLOO()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])

# Regression model w/ one-hot encoding
model_oh = Pipeline([
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', BayesianRidge())
])
```

On the dataset with many categories per column, target-encoding outperforms one-hot encoding by a good margin.


```python
print('Many categories:')

# Target encoding w/ many categories
scores = cross_val_score(model_te, X_many, y_many, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ target encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))

# One-hot encoding w/ many categories
scores = cross_val_score(model_oh, X_many, y_many, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ one-hot encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    Many categories:
    MAE w/ target encoding: 0.820 +/- 0.029
    MAE w/ one-hot encoding: 1.049 +/- 0.045
    

On the other hand, with the dataset containing only a few categories per column, the performance of the one-hot encoded model is nearly indistinguishable from the performance of the model which uses target encoding.


```python
print('Few categories:')

# Target encoding w/ few categories
scores = cross_val_score(model_te, X_few, y_few, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ target encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))

# One-hot encoding w/ few categories
scores = cross_val_score(model_oh, X_few, y_few, 
                         cv=3, scoring=mae_scorer)
print('MAE w/ one-hot encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    Few categories:
    MAE w/ target encoding: 0.815 +/- 0.030
    MAE w/ one-hot encoding: 0.830 +/- 0.025
    

## Effect of Category Imbalance

I would have expected target encoding to perform better than one-hot encoding when the categories were extremely unbalanced (most samples have one of only a few categories), and one-hot encoding to outperform target encoding in the case of balanced categories (categories appear about the same number of times thoughout the dataset).  However, it appears that category imbalance effects both one-hot and target encoding similarly.  

Let's generate two datasets, one of which has balanced categories, and another which has highly imbalanced categories in each column.


```python
# Categorical data w/ many categories
X_bal, y_bal = make_categorical_regression(
    n_samples=1000, 
    n_features=10, 
    n_categories=100,
    n_informative=1,
    imbalance=0.0)

# Categorical data w/ few categories
X_imbal, y_imbal = make_categorical_regression(
    n_samples=1000, 
    n_features=10, 
    n_categories=100,
    n_informative=1,
    imbalance=2.0)
```
    

Fitting the models from the previous section (one of which uses target encoding and the other uses one-hot encoding), we see that how imbalanced the data is doesn't have a huge effect on the perfomance of the model which uses target encoding.


```python
print('Target encoding:')

# Target encoding w/ imbalanced categories
scores = cross_val_score(model_te, X_imbal, y_imbal, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ imbalanced categories: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))

# Target encoding w/ balanced categories
scores = cross_val_score(model_te, X_bal, y_bal, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ balanced categories: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    Target encoding:
    MAE w/ imbalanced categories: 0.873 +/- 0.054
    MAE w/ balanced categories: 0.845 +/- 0.041
    

Nor does it appear to have a big effect on the performance of the model which uses one-hot encoding.


```python
print('One-hot encoding:')

# One-hot encoding w/ imbalanced categories
scores = cross_val_score(model_oh, X_imbal, y_imbal, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ imbalanced categories: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))

# One-hot encoding w/ balanced categories
scores = cross_val_score(model_oh, X_bal, y_bal, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ balanced categories: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))
```

    One-hot encoding:
    MAE w/ imbalanced categories: 1.030 +/- 0.024
    MAE w/ balanced categories: 0.993 +/- 0.029
    

I've tried various combinations of predictive models, levels of imbalance, and numbers of categories, and the level of imbalance doesn't seem to have a very systematic effect.  I suspect this is because for both target encoding and one-hot encoding, with balanced categories we have more information about all categories on average (because examples with each category are more evenly distributed).  On the other hand, we have *less* information about the most common categories - because those categories are no more "common" than any other in a balanced dataset.  Therefore, the level of uncertainty for those categories ends up actually being higher for balanced datasets.  Those two effects appear to cancel out, and the predictive performance of our models don't change.

## Effect of Interactions

So far, target encoding has performed as well or better than other types of encoding.  However, there's one situation where target encoding doesn't do so well: in the face of strong interaction effects.

An interaction effect is when the effect of one feature on the target variable depends on the value of a second feature.  For example, suppose we have one categorical feature with categories A and B, and a second categorical feature with categories C and D.  With no interaction effect, the effect of the first and second feature would be additive, and the effect of A and B on the target variable is independent of C and D.  An example of this is the money spent as a function of items purchased.  If a customer purchases both items 1 and 2, they will be charged the same as if they had purchased either item independently:


```python
plt.bar(np.arange(4), [0, 2, 3, 5])
plt.ylabel('Cost')
plt.xticks(np.arange(4), 
           ['No purchases', 
            'Purchased only item 1', 
            'Purchased only item 2', 
            'Purchased both 1 + 2'])
```


![svg](/assets/img/target-encoding/output_98_1.svg)


On the other hand, if there is an interaction effect, the effect on the target variable will not be simply the sum of the two features' effects.  For example, just adding sugar *or* stirring coffee may not have a huge effect on the sweetness of the coffee.  But if one adds sugar *and* stirs, there is a large effect on the sweetness of the coffee.


```python
plt.bar(np.arange(4), [1, 1, 3, 10])
plt.ylabel('Coffee sweetness')
plt.xticks(np.arange(4), 
           ['Nothing', 
            'Stir', 
            'Sugar', 
            'Sugar + stir'])
```


![svg](/assets/img/target-encoding/output_100_1.svg)


Target encoding simply fills in each category with the mean target value for samples having that category.  Because target encoding does this for each column individually, it's fundamentally unable to  handle interactions between columns!  That said, one-hot encoding doesn't intrinsically handle interaction effects either - it depends on the learning algorithm being used.  Linear models (like the Bayesian ridge regression we've been using) can't pull out interaction effects unless we explicitly encode them (by adding a column for each possible interaction).  Nonlinear learning algorithms, such as decision tree-based models, SVMs, and neural networks, are able to detect interaction effects in the data as-is.

To see how well interaction effects are captured by models trained on target-encoded or one-hot-encoded data, we'll create two categorical datasets: one which has no interaction effects, and one whose variance is completely explained by interaction effects (and noise).


```python
# Categorical data w/ no interaction effects
X_no_int, y_no_int = make_categorical_regression(
    n_samples=1000, 
    n_features=10,
    n_categories=100,
    n_informative=2,
    interactions=0.0)

# Categorical data w/ interaction effects
X_inter, y_inter = make_categorical_regression(
    n_samples=1000, 
    n_features=10,
    n_categories=100,
    n_informative=2,
    interactions=1.0)
```


To capture interaction effects, we'll have to use a model which can handle interactions, such as a tree-based method like XGBoost (a linear regression can't capture interactions unless they are explicitly encoded).


```python
# Regression model w/ target encoding
model_te = Pipeline([
    ('encoder', TargetEncoderLOO()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', XGBRegressor())
])

# Regression model w/ one-hot encoding
model_oh = Pipeline([
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', XGBRegressor())
])
```

As we've seen before, without interaction effects the target encoder performs better than the one-hot encoder.


```python
print('No interaction effects:')

# Target encoding w/ no interaction effects
scores = cross_val_score(model_te, X_no_int, y_no_int, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ target encoding: %0.3f +/- %0.3f'
      % (scores.mean(), scores.std()))

# One-hot encoding w/ no interaction effects
scores = cross_val_score(model_oh, X_no_int, y_no_int, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ one-hot encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    No interaction effects:
    MAE w/ target encoding: 1.013 +/- 0.033
    MAE w/ one-hot encoding: 1.155 +/- 0.029
    

However, when most of the variance can be explained by interaction effects, the model trained on one-hot encoded data performs better (or at least it's unlikely that the target-encoded model has better performance).


```python
print('With interaction effects:')

# Target encoding w/ interaction effects
scores = cross_val_score(model_te, X_inter, y_inter, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ target encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))

# One-hot encoding w/ interaction effects
scores = cross_val_score(model_oh, X_inter, y_inter, 
                         cv=5, scoring=mae_scorer)
print('MAE w/ one-hot encoding: %0.3f +/- %0.3f' 
      % (scores.mean(), scores.std()))
```

    With interaction effects:
    MAE w/ target encoding: 1.222 +/- 0.035
    MAE w/ one-hot encoding: 1.189 +/- 0.009
    

## Suggestions

Target encoding categorical variables is a great way to represent categorical data in a numerical format that machine learning algorithms can handle, without jacking up the dimensionality of your training data.  However, make sure to use cross-fold or leave-one-out target encoding to prevent data leakage!  Also keep in mind the number of categories, what machine learning algorithm you're using, and whether you suspect there may be strong interaction effects in your data.  With only a few categories, or in the presence of interaction effects, you're probably better off just using one-hot encoding and a boosting algorithm like XGBoost/CatBoost/LightGBM.  On the other hand, if your data contains many columns with many categories, it might be best to use target encoding!
