* Note: I wrote an iPython script for this problem. In case you cant run it, there is a plain python script containing
the same code. For that I used: ipython nbconvert --to script customer_satisfaction.ipynb

# Summary:

1. Understanding the Problem and Generating Hypothesis
2. Data Loading and Exploration
3. Feature Engineering and Attempts to find the “magic feature”
4. Ensemble and Stacking Techniques
5. Final Results

I found this tutorial very useful:
http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/


# 1. Understanding the Problem and Generating Hypothesis:

Santander wants to predict whether a customer is satisfied with the bank. Which parameters can influence this factor?
By doing some research, I found:

http://www.mybanktracker.com/news/2013/08/23/10-most-common-customer-banking-complaints/

1. Excessive/hidden fees

2. Bad customer service: complains

3. Checks/funds bouncing: ??

4. Most expensive debits charged first: ??

6. Mortgage/loan issues: Good reasons to refinance a mortgage include a desire to reduce monthly payments,
trim the amount you will pay over the duration of the loan, shorten the term of the loan,
getting out of an adjustable rate mortgage (ARM), or obtaining cash from equity.

7. Huge errors/mistakes: complains

8. Bad branch experiences: complains

9. Difficult for small businesses: small business work better with smaller banks (at least in the States)

10. Failing to honor their promises: banks which dont give cash-back rewards to customers after promising it.


In summary, it is logical to think customers with excessive or hidden fees and registered complains are more likely
to be dissatisfied. This fact could be an important hint to find the 'magic feature'.

The idea is to find variables related to these reasons. Which ones could be found in a data set?
Lets explore the data set first,  then for each of the previous points, lets try to match them to the variables given.



# 2. Data Loading and Exploration

Key points and summary of the data set:

The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.

- MISSING VALUES: By using describe, we can see there are no missing values for any variable, as the count is the same for all variables.


* See: What techniques can be used to perform exploratory data analysis on high dimensional data?
https://www.quora.com/What-techniques-can-be-used-to-perform-exploratory-data-analysis-on-high-dimensional-data


* To use pylab: install matplotlib (from conda or pycharm).
Add the import matplotlib.pyplot as plt and the magic command %matplotlib inline

- EXTREME VALUES/OUTLIERS: histogram and boxplot:

df[''].hist(bins=??)

df.boxplot(column='')




* RESULTS:

Gaussian Naive bayes.
 · Cross-validation score = 0.906176006314
 · AUC score = 0.649838
 · Leaderboard position = 2984


Bagging (sequencial training to reduce bias) with kneighbors classifier. Its all zeros...
 · Cross-validation score = 0.959517232307
 · AUC score = 0.500000
 · Leaderboard position =


Bagging (sequencial training to reduce bias) with gaussian naive bayes classifier.
 · Cross-validation score = 0.729643514865
 · AUC score = 0.685573
 · Leaderboard position = 2942 (43 positions up!!)


Random forest classifier. Its all zeros...
 · Cross-validation score =
 · AUC score = 0.500000
 · Leaderboard position =


 My next step is to figure out why some classifiers get all zeros. There are two possble reasons:

 1. It might be because of some feature I have to get rid of.

 2. I think this problem happens because of some variables with negative values. Some classifiers are based on
 underlying mathematical operations which fail with negative values: I remember shifting negative values to zero when
 working with gaussian processes, given the kernel matrix would fail when dealing with these during the training phase.

 I tried AdaBoost with Gaussian Naive Bayes, but now I get all zeros. Given this classifier got me a good score before,
 the problem doesnt seem to relate to the mathematical implementation of the classifier clashing with negative values.
 It might be something simpler, and probably related to the problem 1. I will have to work on feature engineering.


 Now, I will do feature selection with k=5 and I will explore some of those variables graphically
 (scatterplot, histograms, etc).

I used scatterplots from seaborn, but these are generated too slowly. I considered using rpy2 to connect to R and use
its plotting tools. I will try it once this competition is over.

So now I will do some feature engineering:

- What could define if a customer is happy or not? I already answered this question.
To recap, I will choose the most useful ones:

1. Excessive/hidden fees: search for variables related to fees. They might be of the form of percentages.

2. Bad customer service: complains: search for complains variables, such as textual complains, or boolean ones.

3. Checks/funds bouncing: search for a checks/funds variable which bounces, ie, identify this variable and check if it
has a big variance. If it does, check why. If it is a founds variable, it shouldn't decrease overtime, just increase.
So if it has big fluctuations overtime which does not seem related to the customer adding or getting money from it,
that might be a bouncing fund. If so, create a variable bouncing_fund, which if true, it will be more likely to make the
customer unhappy.

4. Most expensive debits charged first: ??

7. Huge errors/mistakes: complains: same as 2.

8. Bad branch experiences: complains: same as 2.


* Another important factor would be the demographics. Younger generations tend to be less conformist, so they are very
likely to be unhappy clients if there is something wrong. There is no variable named "age", so it would be important
to find it. Describe could give us this information, for example, by looking at the mean. var15 has a mean of 33.21. It
sounds like the average age of clients, so lets explore this variable and other variables with similar mean values.


- I will also have a look at the column names again, as I may find more useful variables.


* Interesting update!
By combining the boolean variables with some variables chosen by the feature selector, including the following three:

saldo_var30
var15
var36

i.e., 12 variables, plus a support vector machine which balances the unbalanced classes, and using a subset of only 150
training instances of the training set with class proportion 0.0533 -> 5.33% of TARGET = 1 examples.
I moved up 185 positions. Instead of 0.68~ auc score, now I get 0.733829 auc score. This is a great improvement,
given I just use a very small amount of the training set: 150/76020 = 0.0019731649 - > 0.197%, and a small proportion
of features: 12/307 = 0.03908794 -> 3.9%.

This could be a good stating point for new tests, which could consist on refining the training subset, adding
other interesting features or playing around with the available ones.

- From the new feature analysis based on these features, I test the SVC with similar variables, without improvements.

- Balancing the training set: I didn't see much improvement.


COMPETITION HAS FINISHED


At this point, I want to try a few extra things:

1. Run the equivalent code with PySpark (see pyspark_version.ipynb)

2. Run the equivalent code with Scala-Spark

--


1. To use PySpark,

    - I installed the preconfigured version of Spark for hadoop 2.6 in C:\spark

    - I downloaded winutils.exe and put it into C:\hadoop\bin. This is required to fix a bug for spark on windows.
    Explanation here: http://nishutayaltech.blogspot.co.uk/2015/04/how-to-run-apache-spark-on-windows7-in.html

    - I created a new ipython notebook, starting with the following lines:

# PySpark configuration
import sys
sys.path.append("C:\spark\python")
sys.path.append("C:\spark\python\lib\pyspark.zip")
sys.path.append("C:\spark\python\lib\py4j-0.9-src.zip")

import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\hadoop"

from pyspark import SparkContext
from pyspark import SparkConf

sc = SparkContext("local", "test")


    - I wrote spark.jars.packages 				com.databricks:spark-csv_2.11:1.4.0 in spark\conf\spark-defaults.conf


I finished my pyspark version, where I compared both mllib and ml packages. Next task is to repeat this code
with Scala Spark and compare it to pyspark. The folder 5.ScalaSpark_Customer_Satisfaction contains this task 
and its corresponding README.md file.

When using Saddle or Framian, use File > Project Structure> Libraries > add from Maven.

