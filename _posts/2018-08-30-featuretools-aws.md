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

[TOC]

## Featuretools and Automated Feature Engineering

What do I mean by "automated feature engineering" and how is it useful?  When building predictive models, we need to have training examples which have some set of features.  For most machine learning algorithms (though of course not all of them), this training set needs to take the form of a table or matrix, where each row corresponds to a single training example or observation, and each column corresponds to a different feature.  For example, suppose we're trying to predict how likely loan applicants are to successfully repay their loans.  In this case, our data table will have a row for each applicant, and a column for each "feature" of the applicants, such as their income, their current level of credit, their age, etc.

Unfortunately, in most applications the data isn't quite as simple as just one table.  We'll likely have additional data stored in other tables!  To continue with the loan repayment prediction example, we could have a separate table which stores the monthly balances of applicants on their other loans, and another separate table with the credit card accounts for each applicant, and yet another table with the credit card activity for each of those accounts, and so on.  

![Data table tree](C:\Users\brendan\Documents\Code\brendanhasz.github.io\_posts\DataframeTree.png)

In order to build a predictive model, we need to "engineer" features from data in those secondary tables.  These engineered features can then be added to our main data table, which we can then use to train the predictive model.  For example, we could compute the number of credit card accounts for each applicant, and add that as a feature to our primary data table; we could compute the balance across each applicant's credit cards, and add that to the primary data table; we could also compute the balance to available credit ratio and add that as a feature; etc.

With complicated (read: real-life) datasets, the number of features that we could engineer becomes very large, and the task of manually engineering all these features becomes extremely time-intensive.  The [Featuretoools](https://www.featuretools.com/) package automates this process by automatically generating features for our primary data table from information in secondary data sources.

**TODO**: explain deep feature synthesis, feature primitives, etc

**TODO**: show how to use it after loading data in w/ pandas, and then run example model on it (e.g. lightGBM)

The downside of Featuretools is that is isn't generating features intelligently - it simply generates features by applying all the feature primitives to all the features in secondary datasets recursively.  This means that the number of features which are generated can be *huge*!  When dealing with large datasets, this will take both a lot of time and a lot of memory.  To alleviate this problem, we can use [Dask](http://dask.pydata.org/en/latest/) to parallelize the computation of the new features.

<a id='dask'></a>

## Using Dask with Featuretools

**TODO**: talk about what dask is and go over the code to get featuretools to work on dask

Of course, maybe you don't have a cluster handy.  I don't!  Thankfully we can rent server time on Amazon Web Service's "Elastic Compute Cloud" (EC2).



<a id='aws'></a>

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
5. Enter a name for your new key pair.  I'll refer to the name you chose below as `<key-pair-name>`. Amazon recommends using `<username>-key-pair-<region>`, where `<username>` is your IAM username (or some other name that you'll remember), and `<region>` is the AWS region you'll be connecting to (the region shown in the upper-right of the page).

The private key will be downloaded by your browser.  Open up a terminal and set user permissions for the file:

```bash
chmod 400 <key-pair-name>.pem
```

If you're in Windows, you can either use PuTTY (see the "To prepare to connect to a Linux instance from Windows using PuTTY" section of [this page](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html)), or personally I'd recommend using [cmder](http://cmder.net/), which comes with an SSH client and bash emulator which allows you to run that `chmod` command directly.

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

Now that we have things set up, we can launch our AWS virtual machine (called an "instance"). From the [EC2 console](https://console.aws.amazon.com/ec2/), click the blue "Launch Instance" button.

Select the Amazon Machine Image (AMI) you want to use.  If you want to ensure you're using only AMIs you won't be charged for, select the "Free tier only" checkbox on the left.  We'll use the "Amazon Linux 2" AMI.

Select the Instance Type you want and click the blue "Review and Launch" button.  For testing things out, you'll probably want to select the `t2.micro` instance type (which is free).   However, for running Featuretools you'll want to select an instance type which has enough memory.  Here's a list of [AWS instance types](https://aws.amazon.com/ec2/instance-types/) and their [pricing](https://aws.amazon.com/ec2/pricing/on-demand/).  To run Featuretools on the full dataset, we'll need far more memory than is available to a `t2.micro` instance, and so we'll use the `m5.4xlarge` instance type.

Note that the `m5.4xlarge` instance type is *not* free!  So don't launch an instance of that type just yet.  I'd suggest trying everything in this post out using the `t2.micro` instance first (which, again, is free), and then only once you know what you're doing going back and starting a paid instance.

Now we'll set our instance to belong to the security group we created earlier.  Under "Security Groups", click the "Edit security groups" link.

Select the "Select an existing security group" radio button, and select the box next to the security group you created earlier.  

If you'll want to connect to your instance from a different IP address than you were using when you set up the security group, you'll have to either edit your security group, or create a new security group which allows connections from your new IP. 

To create a new security group,

1. Select the "Create a new security group" radio button.
2. Give your new security group a name, if you want.
3. Enter the same three rules as above (Type=HTTP,Source=Anywhere; Type=HTTPS,Source=Anywhere; Type=SSH,Source=My IP) 

Or, to edit an existing security group (you only have to do this if you're connecting from a different IP than you were when you set up the security group)

1. Go back to the EC2 console and select "Security groups" from the panel on the left.
2. Select the box next to the security group you want to change.
3. Click the "Actions" button and select "Edit inbound rules".
4. Change the "Source" of the SSH rule to be "My IP".
5. Click the blue "Save" button.

After you've selected a security group (or created a new one), click the blue "Review and Launch" button.

Click the blue "Launch" button.

Here you can either select to use the key pair you created earlier, or create a new one (for example if you don't have access to the key pair you created earlier).  To use the key pair you created earlier, select "Choose an existing key pair" from the drop-down, and then select the name of the key-pair you want to use from the second drop-down.  But if you want to create a new one, select "Create a new key pair" from the drop-down, then give it a name, and select "Download Key Pair", and repeat the `chmod` command from earlier on the downloaded file.

Finally, click the acknowledgement checkbox and click the blue "Launch Instances" button.  From here you can click the blue "View Instances" button (or, from the EC2 console, go to "Instances" in the panel on the left).  From here you can see info about your instance which should now be up and running!

### 6) Connect to an AWS Instance

To connect to your AWS instance via SSH, run (in a terminal):

```bash
ssh -i <key-pair-name>.pem ec2-user@<publicDNS>
```

where `<publicDNS>` is the public DNS of your instance.  You can see the public DNS of your instance by going to the Instances page (in the EC2 console, in the panel on the left under "INSTANCES", click on "Instances"), and selecting the instance you want.  In the lower right of the page, in the "Description" tab, there will be a domain after "Public DNS (IPv4)", for example `ec2-99-999-99-99.us-east-2.compute.amazonaws.com`.  Use this as the `<publicDNS>` when logging in via SSH.

Again, if you're in windows, you'll have to use [Putty](https://www.putty.org/), [cmder](http://cmder.net/), or some other SSH client for windows.

### 7) Install Docker, Build, and Run a Container

**TODO**: intro about why we want to use a docker container (reproducibility and speed!)

This info available here: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-basics.html

```bash
sudo yum update -y
```

Then to install docker:

```bash
sudo yum install -y docker
```

Then start the docker service:

```bash
sudo service docker start
```

Finally, set the user permissions so you don't have to type `sudo` before every command...

```bash
sudo usermod -a -G docker ec2-user
```

Log out and log back in again to your instance (type `exit` and then log back in with `ssh -i key-pair-name.pem ec2-user@<publicIP>`).

Make sure docker has started up successfully by running the following command, which shows information about local or running docker images:

```bash
docker info
```

Now you can either create new docker image from scratch, or run one that's already been created.  To run a docker container from [Docker Hub](https://hub.docker.com/) (the docker equivalent of GitHub), run:

```bash
docker run -it <owner>/<image>
```

Where `<owner>` is the owner on Dockerhub of the image you want to run, and `<image>` is the image's name.  For example, to use [Kaggle's docker image for Python](https://hub.docker.com/r/kaggle/python/), run (though note that this won't work on a `t2.micro` instance because it doesn't have enough memory!):

```bash
docker run -it kaggle/python
```

You can also create your own docker image.  To run automated feature engineering with featuretools and dask, we'll  only need pandas, dask, and featuretools (and their dependencies).  So, we can build a relatively simple Dockerfile:

````dockerfile
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 python3-pip

RUN pip3 install numpy \
    pandas \
    featuretools \
    "dask[complete]"
````

Save the above code to a file called `Dockerfile`.

To build the docker image from that Dockerfile (if it's the only file named `Dockerfile` in your current directory), run

```bash
docker build -t <image-name> .
```

where `<image-name>` is the name you want to give your docker image (for example, `featuretools-dask`).  

If you have multiple docker files with different file names in the same directory, you can build a specific one by piping that file into the build command:

```bash
docker build -t <image-name> - < <dockerfile-name>
```

where `<dockerfile-name>` is the filename of the dockerfile you want to build.

**TODO**: how to push the image -> docker hub

**TODO**: `docker images` to see all local images

Then to run that image that you built, run:

```bash
docker run -it <image-name>
```

Or, if you haven't built the image on this machine but you've pushed it to Docker Hub, run:

```bash
docker run -it <your-dockerhub-username>/<image-name>
```

Now you've been dropped into a bash shell running in your Docker container!

**TODO**: though maybe should do via ECS...

### 8) Upload Data to an AWS Instance

**TODO**: either scp or aws cli..., then copy to the docker container w/ `docker cp` (https://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container)

### 9) Run Automated Feature Engineering

TODO: run the featuretools code

### 10) Download Data from an AWS Instance

TODO: again either scp or aws cli

### 11) Shut Down an AWS Instance

TODO