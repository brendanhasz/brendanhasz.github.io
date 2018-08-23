---
layout: post
title: "Automated Feature Engineering on AWS using Featuretools and Dask"
date: 2018-08-23
description: "Running deep feature synthesis for automated feature engineering, using Featuretools and Dask on AWS EC2."
tags: [feature engineering, featuretools, dask, AWS]
published: false
---

[Featuretools](https://www.featuretools.com/) is a fantastic python package for automated feature engineering.  It can automatically generate features from secondary datasets which can then be used in machine learning models.  Unfortunately, with larger datasets Featuretools will quickly run out of memory on most personal computers.  In this post we'll see how automated feature engineering with Featuretools works, how to run it at scale using [Dask](http://dask.pydata.org/en/latest/), and how to set up an Amazon Web Services EC2 cluster to run it on!

**Outline**

- [Featuretools and Automated Feature Engineering](#featuretools)
- [Dask](#dask)
- [Setting up an AWS EC2 Cluster](#aws)



<a id='featuretools'></a>

## Featuretools and Automated Feature Engineering

What do I mean by "automated feature engineering" and how is it useful?  When building predictive models, we need to have training examples which have some set of features.  For most machine learning algorithms (though of course not all of them), this training set needs to take the form of a table or matrix, where each row corresponds to a single training example or observation, and each column corresponds to a different feature.  For example, suppose we're trying to predict how likely loan applicants are to successfully repay their loans.  In this case, our data table will have a row for each applicant, and a column for each "feature" of the applicants, such as their income, their current level of credit, their age, etc.

Unfortunately, in most applications the data isn't quite as simple as just one table.  We'll likely have additional data stored in other tables!  To continue with the loan repayment prediction example, we could have a separate table which stores the monthly balances of applicants on their other loans, and another separate table with the credit card accounts for each applicant, and yet another table with the credit card activity for each of those accounts, and so on.  

In order to build a predictive model, we need to "engineer" features from data in those secondary tables.  These engineered features can then be added to our main data table, which we can then use to train the predictive model.  For example, we could compute the number of credit card accounts for each applicant, and add that as a feature to our primary data table; we could compute the balance across each applicant's credit cards, and add that to the primary data table; we could also compute the balance to available credit ratio and add that as a feature; etc.

With complicated (read: real-life) datasets, the number of features that we could engineer becomes very large, and the task of manually engineering all these features becomes extremely time-intensive.  The [Featuretoools](https://www.featuretools.com/) package automates this process by automatically generating features for our primary data table from information in secondary data sources.

**TODO**: explain deep feature synthesis, feature primitives, etc

**TODO**: show how to use it after loading data in w/ pandas, and then run example model on it (e.g. lightGBM)

The downside of Featuretools is that is isn't generating features intelligently - it simply generates features by applying all the feature primitives to all the features in secondary datasets recursively.  This means that the number of features which are generated can be *huge*!  When dealing with large datasets, it's easy to run out of memory when attempting to run deep feature synthesis.

<a id='featuretools'></a>

## Dask

**TODO**: talk about what dask is and go over the code to get featuretools to work on dask

Of course, maybe you don't have a cluster handy.  I don't!  TODO: transition to aws



<a id='featuretools'></a>

## Setting up an AWS EC2 Cluster

**TODO**: talk about how to set up AWS, start an instance, connect to your instance, upload files, run the stuff, download files, and terminate your instance.