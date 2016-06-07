------------
Description:
------------

This project consists on the solution to a Kaggle competition: 

Telstra Network Disruptions: Predict service faults on Australia's largest telecommunications network

I found this competition quite late, when there were just a few days left before the final submission. I decided to start it anyways, as I considered it very useful to learn how to apply some data tyding operations, given the dataset was split between several tables, in which some variables were spread along the rows instead the columns. Python does not offer a clear set of data tyding operations like those found in R's tidyr package. I haven't spent more time on this competition, as I would like to start a new one whose deadline has not been reached yet. That is why you won't find the data science side of this project. For a data science focused project, check my 4th folder, the Santander Customer Satisfaction prediction. It is coming soon.


I heavily relied in Pandas for these data preprocessing steps, reading the csv directly, transforming them to pandas dataframes, applying the corresponding operations and generating numpy arrays to feed the scikit-learn models. I particularly found the get_dummies() function quite useful as it takes all the values of a column an spreads them along new columns, which contain 1 or 0 if the variable takes that value or not for the given observation. Note this functions is used after applying groupby(). The problem I found is this get_dummies() function returns a floating point dataframe, which can bring memory limitations when working with bigger datasets. This is an issue I would like to solve by contributing to the pandas libary by updating this method.


Below you can find some notes I wrote during the project. However, further details can be found in the code comments.


* UPDATE 1: After I finished my solution and uploaded it to Kaggle, I had a look at two other solutions I found:

http://www.analyticsvidhya.com/blog/2016/03/complete-solution-top-11-telstra-network-disruptions-kaggle-competition/

http://gereleth.github.io/Telstra-Network-Disruptions-Writeup/

They both mention a magic feature hidden in the data set. Turns out that ids and locations are sorted by time, so a new feature (a time index), can be added to the dataset to significantly improve the results. Following the second link, I found a very useful ipython notebook explaining how to find this magic feature:

https://github.com/gereleth/kaggle-telstra/blob/master/Discovering%20the%20magic%20feature.ipynb




------
STEPS:
------

1. Create an auxiliar function which shows, for every csv file, the column name and all its possible values

2. clean the columns. E. g. 'event_type 2' (string) -> 2 (integer)

3. Merge two datasets. E. g. train and event_type.

The resulting merged dataframe has some rows like the following:
          id       location  fault_severity     event_type
8      14804   		 120               0  	 		34
9      14804   		 120               0  	 		11
10     14804   		 120               0  	 		36
11     14804   		 120               0  	 		20

By applying tidying operations, this repetition could be removed by moving event_type from rows to columns, so that the final dataframe looks like:

          id       location  fault_severity  event_type 11  event_type 20  event_type 34  event_type 36 ..  event_type X  ..

8      14804   		120               0  		1			1				1			  1  			..  0



Note: when doing a right outter join, being event_type the 'right' table, there are some rows like the following:

7      	6741      1066               1          13
28      7187       812               0          11
29     16780       343               0          34
...      ...       ...             ...         ...
31140   7605       NaN             NaN          11
31141   5146       NaN             NaN          11
31142   7784       NaN             NaN          11
31143   7784       NaN             NaN          14


which means event_type table has id which does not correspond to any row of the train table. This data is missing -> should it be ommited? -> Yes, but only for the training set, because we need the fault_severity variable. 

NOTE!
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  merged_df.iloc[i]['event_type_bit_'+str(x)] = i_event_type_bin[x]



 