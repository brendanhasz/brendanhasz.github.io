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
- [Setting up an AWS EC2 Instance](#aws)



<a id='featuretools'></a>

## Featuretools and Automated Feature Engineering

What do I mean by "automated feature engineering" and how is it useful?  When building predictive models, we need to have training examples which have some set of features.  For most machine learning algorithms (though of course not all of them), this training set needs to take the form of a table or matrix, where each row corresponds to a single training example or observation, and each column corresponds to a different feature.  For example, suppose we're trying to predict how likely loan applicants are to successfully repay their loans.  In this case, our data table will have a row for each applicant, and a column for each "feature" of the applicants, such as their income, their current level of credit, their age, etc.

Unfortunately, in most applications the data isn't quite as simple as just one table.  We'll likely have additional data stored in other tables!  To continue with the loan repayment prediction example, we could have a separate table which stores the monthly balances of applicants on their other loans, and another separate table with the credit card accounts for each applicant, and yet another table with the credit card activity for each of those accounts, and so on.  

**TODO**: svg of table connections etc

In order to build a predictive model, we need to "engineer" features from data in those secondary tables.  These engineered features can then be added to our main data table, which we can then use to train the predictive model.  For example, we could compute the number of credit card accounts for each applicant, and add that as a feature to our primary data table; we could compute the balance across each applicant's credit cards, and add that to the primary data table; we could also compute the balance to available credit ratio and add that as a feature; etc.

With complicated (read: real-life) datasets, the number of features that we could engineer becomes very large, and the task of manually engineering all these features becomes extremely time-intensive.  The [Featuretoools](https://www.featuretools.com/) package automates this process by automatically generating features for our primary data table from information in secondary data sources.

**TODO**: explain deep feature synthesis, feature primitives, etc

**TODO**: show how to use it after loading data in w/ pandas, and then run example model on it (e.g. lightGBM)

The downside of Featuretools is that is isn't generating features intelligently - it simply generates features by applying all the feature primitives to all the features in secondary datasets recursively.  This means that the number of features which are generated can be *huge*!  When dealing with large datasets, this will take both a lot of time and a lot of memory.  To alleviate this problem, we can use [Dask](http://dask.pydata.org/en/latest/) to parallelize the computation of the new features.

<a id='featuretools'></a>

## Using Dask with Featuretools

**TODO**: talk about what dask is and go over the code to get featuretools to work on dask

Of course, maybe you don't have a cluster handy.  I don't!  Thankfully we can rent server time on Amazon Web Service's "Elastic Compute Cloud" (EC2).



<a id='featuretools'></a>

## Setting up an AWS EC2 Instance

Amazon Web Service's Elastic Compute Cloud (AWS EC2) is Amazon's cloud computing platform, which allows users to rent server time to run their own applications.  EC2 provides a selection of different hardware configurations (or [instance types](https://aws.amazon.com/ec2/instance-types/)), so customers can select the hardware configuration which best suits their needs.  For example, a scientific modelling application might use an compute-optimized EC2 instance; an in-memory database might use a memory-optimized EC2 instance; an application training deep neural networks might use an EC2 instance which has GPUs available to it; and a Hadoop application might use a storage-optimized EC2 instance.  This allows customers to rent a hardware environment which suits their needs without having to purchase said hardware up front.  We're going to set up an EC2 instance which has a (relatively) large amount of RAM, so that we can run deep feature synthesis on our dataset.

In this section we'll see how to set everything up for a new AWS account, launch an instance, upload our data, run our automated feature engineering code on it, download the results, and shut down our instance.  Amazon has guides which cover most of the information in this section (except for the automated feature engineering), if you'd prefer to get it straight from the source:

- [Setting Up with Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html)
- [Getting Started with Amazon EC2 Linux Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
- [How to Deploy Docker Containers on AWS](https://aws.amazon.com/getting-started/tutorials/deploy-docker-containers/)

### 1) Create an AWS Account

To run an EC2 instance, you'll first need an Amazon Web Services account.  You can sign up at the [AWS website](https://aws.amazon.com/) by clicking on the "Create an AWS Account" button and following the instructions.

### 2) Create an IAM User

After creating an account, you should create an Identity and Access Management (IAM) user through which you can administrate your rented instances.

1. Sign in to the [IAM console](https://console.aws.amazon.com/iam/) using your AWS account information.
2. In the panel on the left, choose "Users", and then select the "Add user" button.
3. For the User name, enter "Administrator" (or whatever you want, really).
4. Select the "AWS Management Console access" checkbox.
5. Select the "Custom password" radio button and enter the password you'd like to use for this IAM user.
6. Unselect the "User must create a new password at next sign-in" checkbox (unless you're making the account for someone else, in which case this will enable them to set their own password).
7. Then select "Next: Permissions".
8. Select "Add user to group", and choose "Create group".
9. For the Group name, enter "Administrators" (or, again, whatever you want).
10. In the table of policies, select the check box next to the AdministratorAccess policy (with the description "Provides full access to AWS services").
11. Select "Create group".
12. Select the checkbox next to your newly-created group.
13. Select "Next: Review", and then "Create user".
14. You can now log out of the AWS console, and log back in as the IAM Administrator user you just created.   Go to `https://<your_aws_account_id>.signin.aws.amazon.com/console/`, where `<your_aws_account_id>` is your AWS account number without the hyphens.
15. Sign in using the IAM user username (`Administrator`), and the password you set for that user.

### 3) Create a Key Pair 

To secure your connection to your instance, you'll want to create a cryptographic key pair.  This will allow you to connect to your instance using the private key via SSH.

1. Sign in to the [AWS EC2 console](https://console.aws.amazon.com/ec2/) using your IAM Administrator user information.
2. In the panel on the left, in the "NETWORK & SECURITY" section, select "Key Pairs".
3. In the upper-right, select the region (e.g. "US East (Ohio)" or "US West (Oregon)") for which you want to generate the key pair.  It's probably fine to leave it as is.  However, if you want to run an instance in different regions, you'll need to generate a new key pair for each region - key pairs are region-specific.
4. Click the blue "Create Key Pair" button.
5. Enter a name for your new key pair.  Amazon recommends using `<username>-key-pair-<region>`, where `<username>` is your IAM username (or some other name that you'll remember), and `<region>` is the AWS region you'll be connecting to (the region shown in the upper-right of the page).

The private key will be downloaded by your browser.  Open up a terminal and set user permissions for the file:

```bash
chmod 400 key-pair-name.pem
```

If you're in Windows, you can either use PuTTY (see the "To prepare to connect to a Linux instance from Windows using PuTTY" section of [this page](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html)), or personally I'd reccomend using [cmder](http://cmder.net/), which comes with an SSH client and bash emulator which allows you to run that `chmod` command directly.

Save the private key file in a safe place, because you won't be able to download it again!  However, you can always create a new key pair when you launch an instance.

### 4) Create a Security Group

You'll also want to create what's called a "security group", which sets limits on what kind of network traffic your instance can receive. 

1. From the upper-right of the [EC2 console](https://console.aws.amazon.com/ec2/), select the region for the security group you're about to create (e.g. "US East (Ohio)").  This should be the same region for which you created the key pair (and will probably already be set to the one you want).
2. In the panel on the left, under "NETWORK & SECURITY", select "Security Groups".
3. Click the blue "Create Security Group" button.
4. For the Security group name, enter a name for your new security group.  Amazon recommends `<name>_SG_<region>`, where `<name>` is some name you'll remember, and `<region>` is the AWS region for which you're creating the security group (because, like key pairs, security groups are region-specific).
5. Enter a description of your security group (e.g. "My default SG for Ohio")
6. In the "Inbound tab", create these three rules:

- Type=HTTP, Source=Anywhere
- Type=HTTPS, Source=Anywhere
- Type=SSH, Source="My IP"

These rules ensure that your instance will accept HTTP requests from any IP, but only SSH connections from the IP address of the computer you're currently using, which helps keep your instance secure.  You can of course change this to accept whatever IP addresses or range of addresses from which you want to connect to your instance.

Finally, click the blue "Create" button to create the security group.

### 5) Launch an Instance

**TODO**: say how to launch a free one (what instance that is), and how to launch the one we'll usee in this tut

### 6) Docker?

TODO: talk about how to build/load a docker image. Using ECS or command line, haven't decided which yet...

### 7) Upload Data to an AWS Instance

TODO: either scp or aws cli...

### 8) Connect to an AWS Instance and Run Code

TODO: ssh and run the featuretools code

### 9) Download Data from an AWS Instance

TODO: again either scp or aws cli

### 10) Shut Down an AWS Instance

TODO