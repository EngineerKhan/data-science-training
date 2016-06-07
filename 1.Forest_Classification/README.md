!!: Read my personal notes inside curly parentheses {}, they include useful clarifications. 

----------------------------------------------------------------
A SOFTWARE ENGINEERING APPROACH TO SOLVE DATA SCIENCE PROBLEMS
----------------------------------------------------------------

I initally wrote this guide for myself, but I thought it could be useful to share it because of three main reasons:

- Experienced people can share their ideas to guide me and correct my mistakes.

- Other beginners have an starting point to approach kaggle competitions.

- Demonstrate and discuss why Data Science should be approached from a software engineering perspective (see step 5.2. below)


Steps to approach the problem:

1. [General view of the problem] Define the problem as if you were to explain it to someone else. {You will understand it much better. Take notes of what you search for and the new concepts you learnt. It is important to spend as much time as neccesary at this stage.}

2. [Scientific view of the problem] Define all the Machine Learning (ML) approaches/models available to this problem. {Same reason as before.}
It is worth to check out this ML map: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

	- For each approach/model:

	[Engineering view of the problem]
	3. define the processing steps to be performed {see 9.x below, they will help you on defining the ML steps}. Common steps are: data visualization, data pre-processing, feature selection

	4. define the software arquitecture of the system and choose the programming language/tools/libraries/technologies to be used {modularize the system as much as you can}

	5.1. define the minimum system. An example would be a single script where all the steps are performed with direct function calls. For small systems, this may not be possible, as the system may be irreducible.

	5.2. VERY IMPORTANT!! Define the steps to go from the minimum system to the full architecture. {You will see a big part of the scientific work of a data scientist lies here as software engineering work}.

	6. define tests for each component


	{Software design tips: for those with good knowledge of object-oriented programming and software design patterns, I suggest that, instead of creating modules as functions, implement them as classes. For example, if you want to implement a data visualization module whose code can slighly vary depending on the visualization to be done, create an abstract class DataVisualization and extend it with different subclasses for different visualization methods. Use creational patterns such as builder and factory method. As behavioral patterns, you may find useful the strategy pattern.}


	[Engineering work] {use git with github for opensource solutions or bitbucket for private projects}

	7. Implement the minimum system and test each of its components. Do steps 9.5 and 9.6. {Yes, you will do a pre-data science cycle here. It will help you when comparing how well data visualization, data preprocessing and feature selection help}.

	8. Follow 5.2. to implement the full architecture.


	{Note that most of steps between 3 and 8 will be performed only once, as the infrastructure will not differ that much between ML models. This is just an engineer's suggestion for you to organize yourself. Also, you may find the unit testing thing like killing a flea with a sledgehammer, and you would be right, but what if you want to reuse these modules and make them bigger later on? This guide pretends you stablish your own library of modules for future use.}

	[Scientific work]
	9. then go for the real fun and do the Machine Learning. {The idea is that you create your own working and tested data infrastructure, so that it can be used in future projects with minimum modifications. Following Machine Learning models will not probably need further modifications in the architecture, thus you will reuse most of the modules and only modify the machine learning module.} 

		9.1. Visualize your data with your data visualization module.

		9.2. Detect outliers.

		9.3. Use your data pre-processing module if required (to scale/perform normalization when dealing with raw data).

		9.4. Perform feature selection with the corresponding module.

		9.5. Tune the parameters of your model and make it learn from the data, using the corresponding module for your model.

		9.6. Test your model. Using cross-validation is higly recommended.

		9.7. Try out ensemble models. You will need an specific module for this task.

		{The more you know a model, the better parameter tunning you can do. I recommend the following book for improving your theoretical understanding of both models and the previous machine learning steps, without going into very deep maths (by deep maths I mean what you will find in Murphy's, Bishop's or Barber's books) : Principles of Data Mining: http://cognet.mit.edu/library/books/view?isbn=026208290X}




-----------------------
1. PROBLEM DEFINITION:
-----------------------

This is a multi-class classification problem, where each data point may belong to one and only one of the seven classes. Predicting several labels for a single data point is called multi-label classification, which is not the scope of this problem. {I consider it is very important to identify what kind of problem we are dealing with at this stage, it will help on defining the ML approaches/models later on.}

There are 15120 observations (~15k) in the training set, containing the features and the cover type (the label). The test set contains only the features of 565892 observations (~566k). The problem requires to predict the cover type for the training set.

Remember the data given is raw, henceforth not scaled. What are the implications? {This are questions to yourself, go and google it}. The answers to these questions can be found in the last section, new concepts (*1). Reading this answer, it seems important to scale the features, so there is no governing features over others. The methods are: rescaling, standarization and scaling to unit length. {What about the log-scale?} It is very likely that scikit-learn contains functions to scale the data, so it is worth checking.

When working with simple problems like this, there are no more things to consider at this stage. More complex problems may involve gathering more data, creating new features, doing more data cleaning, etc.



--------------------------------------
2. MACHINE LEARNING APPROACHES/MODELS
--------------------------------------

Following this ML map: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html and the problem definition, we know this is a multi-class classification problem, where we have a training set of ~15k samples and it is not text data, the recommendation is to start by using a Linear SVC, then a KNeighbors classifier if not working, then SVC ensemble classifiers if not working. Nice! Scikit-learn people recommended us what to do. But it would be nicer to know why they recommend this classsifiers and no others such as naive bayes, SGD classifiers, kernel approximations or logistic regression. Moreover, we are dealing with multiple labels, not just 0 or 1. So why not using a classification neural net with several output neurons? Well, it turns out this is useful for a multi-class problem, but probably not for this one. Anyways, it is good to get used to ask yourself this kind of questions {you are a data scientist, aren't you? Scientists discover new stuff by asking questions.}. And what about bayesian networks? How to create a multi-label classifier from binary classifiers?

The previous lines lead to several questions that need to be solved before doing any coding. I think that by having a clear idea of what ML approaches we want to try out, it is easier to define the software architecture and the functionality of our modules. For example, we may find out we want to use ensemble models, so we need a specific module to deal with this. Of course this process will not be lineal in most of the cases, as we may consider trying out a new model (during the scientific work) we did not think of before, impliying the definition of a new module. It still helps to reduce this iterative process and focus on the science once we have built most of the code. 

So let's solve our questions by reading a short summary of the named models and their advantages/disadvantages using the scikit-learn documentation:
http://scikit-learn.org/stable/user_guide.html

- Logistic regression: in this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

- Linear SVC (Support vector classification): 
The advantages of support vector machines are:
Effective in high dimensional spaces.
Still effective in cases where number of dimensions is greater than the number of samples.
Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
The disadvantages of support vector machines include:
If the number of features is much greater than the number of samples, the method is likely to give poor performances.
SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).


- KNeigbors: it is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

The k-neighbors classification in KNeighborsClassifier is the more commonly used of the two techniques. The optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct.

In cases where the data is not uniformly sampled, radius-based neighbors classification in RadiusNeighborsClassifier can be a better choice. The user specifies a fixed radius r, such that points in sparser neighborhoods use fewer nearest neighbors for the classification. For high-dimensional parameter spaces, this method becomes less effective due to the so-called “curse of dimensionality”.

See: 1.4.4.4. Choice of Nearest Neighbors Algorithm


- SVC ensemble classifiers: 
· An ensemble is a technique for combining many weak learners in an attempt to produce a strong learner. 
· The term ensemble is usually reserved for methods that generate multiple hypotheses using the same base learner. 
· The broader term of multiple classifier systems also covers hybridization of hypotheses that are not induced by the same base learner.
· Evaluating the prediction of an ensemble typically requires more computation than evaluating the prediction of a single model, so ensembles may be thought of as a way to compensate for poor learning algorithms by performing a lot of extra computation. Fast algorithms such as decision trees are commonly used with ensembles (for example Random Forest), although slower algorithms can benefit from ensemble techniques as well.
· Ensembles can be shown to have more flexibility in the functions they can represent. This flexibility can, in theory, enable them to over-fit the training data more than a single model would, but in practice, some ensemble techniques (especially bagging) tend to reduce problems related to over-fitting of the training data.


· Types: averaging (eg: bagging{bootstrap aggregating}, random forest) and boosting (AdaBoost, grandient tree boosting). Bagging methods work best with strong and complex models (e.g., fully developed decision trees), in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees).

	- Averaging: the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
	
	- Boosting: base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.


- SGD (stochastic gradient descent) classifiers: SVM, logistic regression, etc... with SGD training. Very good when the training set is very large (<100k samples)

- Perceptron: simple, large scale learning algorithm. Updates its model only on mistakes. Not regularized. Does not require a learning rate. slightly faster to train than SGD with the hinge loss and that the resulting models are sparser.

- Kernel approximations: 
· Standard kernelized SVMs do not scale well to large datasets, but using an approximate kernel map it is possible to use much more efficient linear SVMs. In particular, the combination of kernel map approximations with SGDClassifier can make non-linear learning on large datasets possible.

· E.g.: The radial basis function kernel constructs an approximate mapping for the radial basis function kernel. This transformation can be used to explicitly model a kernel map, prior to applying a linear algorithm, for example a linear SVM.

· The mapping relies on a Monte Carlo approximation to the kernel values. The fit function performs the Monte Carlo sampling, whereas the transform method performs the mapping of the data. Because of the inherent randomness of the process, results may vary between different calls to the fit function.


- Naive Bayes: it involves the bayes rule, then the chain rule, and finally the independance assumption. The feature probabilities can be multinomial, bernoulli (the most common as examples), gaussian, etc..

- Bayesian networks: Not available.

- Neural Nets: Not available (for supervised learning)


{NOTE: HAVE A LOOK AT THE PIPELINE TOOL and grid search!!}

{we will study more models later on}



--------------------
3. PROCESSING STEPS
--------------------

- Data loading

- Data visualization

- Outlier detection

- Data preprocessing (standarization {mean removal and variance scaling}, normalization, binarization, categorical features encoding, label preprocessing, feature extraction)

- Feature selection

- Learn from data with the chosen models (basic ones)

- Test the models (use cross-validation)

- Learn from data with the chosen models (ensemble ones: averaging or boosting, depending on the models)

- Test the models (use cross-validation, precision/recall, roc curves)

* For learning purposes, test the models against unprocessed and pre-processed data, to see how visual analysis and pre-processing can improve the results.

* I would do this work with weka first, before going further with scitkit-learn. It will help you to understand the data science basics quickly.

* Four tutorials from the University of Edinburgh can be found here: http://www.inf.ed.ac.uk/teaching/courses/dme/labs.html {useful even if you go through scikit-learn directly}:



-------------------------
4. SOFTWARE ARCHITECTURE
-------------------------

- Data loading module

- Data visualization module

- Outlier detection module (might be just a user action within the data visualization module, not a module itself)

- Data preprocessing module

- Feature selection module

- Learning module

- Testing module

- Ensemble module for averaging/boosting



-------------------------
5.1. MINIMUM SYSTEM
-------------------------

- Data loading function

- Learning function

- Testing function


--------------------------------------------------------------
5.2. STEPS TO GO FROM MINIMUM SYSTEM TO THE FULL ARCHITECTURE
--------------------------------------------------------------

- For each of the initial functions, modularize

- Add a data visualization (and outlier detection) module and test it

- Add a data preprocessing module and test it. Take notes of the results and compare them to what you got with no preprocessing.

- Add a feature selection module and test it. Take notes of the results and compare them to what you got with no feature selection (and no preprocessing, and whatever combination you want to compare).

- Add and ensemble module and test it. Take notes of the results and compare them to what you got with no ensemble models.




{Here you see that the scientific work of trying out data preprocessing steps, different feature selection methods and ensemble models are both an engineering (you incrementaly build software by adding components) and a scientific work (you try out experiments with your data and write down the results)}.


-------------------------------
6. TESTS FOR MODULES/FUNCTIONS
-------------------------------

- Data loading module: check whether the first rows (the first datapoints) of the loaded data are ok

- Data visualization module: visualize only the first datapoints and check whether they correspond to what is expected.

- Outlier detection module: just by visualizing, nothing to test here really

- Data preprocessing module: with scaling, do it by hand (?) and check whether it is correct for a single datapoint.

- Feature selection module: check whether the selected features correspond to what we expect by having a look at a scatterplot (in the visualization module)

- Learning module: just by running the function. Nothing to test really.

- Testing module: just by running the function.

- Ensemble module for averaging/boosting: This is dark stuff. I don't have a clue about testing this.


--------------------------------
7. IMPLEMENT THE MINIMUM SYSTEM
--------------------------------

Take notes of the results you got when testing the learners with this minimum system, you will need it later on.

- Implement the data loading function:
	· Import csv, use open to open the file, csv.reader to read it and .next() to iterate over the file.
	· Save data sets as required by the learners (have a look at logistic regression). There is an available example in the scikit learn documentation for linear regression which can help us:

			"LinearRegression will take in its fit method arrays X, y and will store the coefficients w of the linear model in its coef_ member:
		>>>
		>>> from sklearn import linear_model
		>>> clf = linear_model.LinearRegression()
		>>> clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
		LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
		>>> clf.coef_
		array([ 0.5,  0.5])"

	Given that we want to use logistic regression, let's have a look at the corresponding documentation and try to implement the code:

		class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

	Based on this information, we can write the following code:

		from sklearn import linear_model
		clf = linear_model.LogisticRegression()
		clf.fit (...)

	and we know the format of the features is a list of lists, where each internal list is a list of the features. The format of the labels is just a list of these labels, so let's implement a loop which can build these two structures simultaneously.


	The learner works! I have split the labeled dataset, so I have a training set and a test set. I get around 25% accuracy with SVM, whereas Logistic regression outperforms it with 58%.

	An improved way of testing is using cross-validation, which has replaced the previous split function I did.

	The following step is to prepare a CSV for the results with logistic regression, which gave me a score of ~66. I also wrote some code for Naive Bayes, Decision Trees and Random Forest. Once done, I would spend some time in improving the code, by creating functions for repetitive code (writing them into mylib folder) and modularizing the sections of the code.

	
	After that, add the visualization module: Scatter plot has been tested but it is too heavy for ipython. It is better to use Weka for this functionality till I find another efficient way of plottinmg scatterplots.


	TABLE:

	
	Results when applying LDA:


		Score for Logistic Regression:  0.649305555556
		Score for Support Vector Classification:  0.648478835979
		Score for Naive Bayes:  0.655092592593
		Score for Nearest Neighbors:  0.782076719577
		Score for Decision Tree:  0.727513227513
		Score for Random Forest:  0.768683862434


	Results when applying feature selection with LDA:

		n_components=5

		Score for Logistic Regression:  0.641038359788
		Score for Support Vector Classification:  0.636243386243
		Score for Naive Bayes:  0.634259259259
		Score for Nearest Neighbors:  0.771825396825
		Score for Decision Tree:  0.73578042328
		Score for Random Forest:  0.760747354497


		n_components=4


		Score for Logistic Regression:  0.625826719577
		Score for Support Vector Classification:  0.621362433862
		Score for Naive Bayes:  0.616567460317
		Score for Nearest Neighbors:  0.747023809524
		Score for Decision Tree:  0.696759259259
		Score for Random Forest:  0.748511904762


		n_components=3

		Score for Logistic Regression:  0.584325396825
		Score for Support Vector Classification:  0.574404761905
		Score for Naive Bayes:  0.602017195767
		Score for Nearest Neighbors:  0.708664021164
		Score for Decision Tree:  0.65162037037
		Score for Random Forest:  0.703042328042

		


	Results without LDA:

		Score for Logistic Regression:  0.668154761905
		Score for Support Vector Classification:  0.448412698413
		Score for Naive Bayes:  0.457010582011
		Score for Nearest Neighbors:  0.774636243386
		Score for Decision Tree:  0.761574074074
		Score for Random Forest:  0.807043650794


	NOTE: LDA only improves SVC and Naive Bayes. Why??


(*2) Read notes about feature selection in the New concepts section.

* Check out http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html


	Comparation with results applying recursive feature elimination (RFE) and recursive feature selection with cross-validation (RFECV):


	+Score for Logistic Regression:  0.668154761905
	Using recursive feature elimination (RFE):
	Score for Logistic Regression with the 5 most relevant features (RFE):  0.40625
	Score for Logistic Regression with the 10 most relevant features (RFE):  0.497023809524
	Score for Logistic Regression with the 15 most relevant features (RFE):  0.535548941799
	Score for Logistic Regression with the 20 most relevant features (RFE):  0.55291005291
	Score for Logistic Regression with the 25 most relevant features (RFE):  0.57671957672
	+Score for Logistic Regression with the 30 most relevant features (RFE):  0.588458994709
	Score for Logistic Regression with the 35 most relevant features (RFE):  0.587797619048
	+Score for Logistic Regression with the  most relevant features (RFECV):  0.664186507937

	+Score for Support Vector Classification:  0.391203703704
	Using recursive feature elimination (RFE):
	Score for Support Vector Classification with the 5 most relevant features (RFE):  0.394841269841
	Score for Support Vector Classification with the 10 most relevant features (RFE):  0.454034391534
	Score for Support Vector Classification with the 15 most relevant features (RFE):  0.52380952381
	Score for Support Vector Classification with the 20 most relevant features (RFE):  0.567956349206
	Score for Support Vector Classification with the 25 most relevant features (RFE):  0.574074074074
	+Score for Support Vector Classification with the 30 most relevant features (RFE):  0.582341269841
	Score for Support Vector Classification with the 35 most relevant features (RFE):  0.315972222222
	+Score for Support Vector Classification with the  most relevant features (RFECV):  0.578538359788


	NAIVE BAYES CAN'T USE THIS METHOD

	NEAREST NEIGHBOR CAN'T USE THIS METHOD

	DECISION TREE CAN'T USE THIS METHOD

	RANDOM FOREST CAN'T USE THIS METHOD


	* PICK the best results for each classifier, print the pruned features and observe why these have been excluded (doing visualization, for example). Answer why.

	* Answer why some classifiers improve with feature selection while others get worse results.

	* Answer why some classifiers can't use the feature selection method.


	Answers:
	--------

	Score for Logistic Regression:  0.668154761905
	Using recursive feature elimination (RFE):
	Score for Logistic Regression with the 30 most relevant features (RFE):  0.588458994709
	[10 11 12 13 14 15 16 17 18 19 23 24 25 26 27 29 30 31 33 34 35 36 39 43 44
	 48 50 51 52 53]
	Score for Logistic Regression with the  most relevant features (RFECV):  0.664186507937
	[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25
	 26 27 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51
	 52 53]


	Score for Support Vector Classification:  0.320601851852
	Using recursive feature elimination (RFE):
	Score for Support Vector Classification with the 30 most relevant features (RFE):  0.585482804233
	[ 2 10 11 12 13 14 15 16 17 18 19 23 24 25 26 27 30 31 33 35 36 37 42 43 44
	 45 46 51 52 53]

	 


ALGORITHMS AND MODELS DETAILED:
-------------------------------

To best perform in a Kaggle competition, there are several factors to be taken into account when choosing models and algorithms: 

1. The task the algorithm is used to address (e.g. classification, clustering, etc.)

2. The structure of the model or pattern we are fitting to the data (e.g. a linear regression model)

3. The score function used to judge the quality of the fitted models or patterns (e.g. accuracy, BIC, etc.)

4. The search or optimization method used to search over parameters and/or structures (e.g. steepest descent, MCMC, etc.)

5. The data management technique used for storing, indexing, and retrieving data (critical when data too large to reside in
memory)


We have only worked on choosing the structure of the model and tried out some dimensionallity reduction techniques for testing the minimum system. Now we have to get our hands on fine tunning the models by studying and comparing the results obtained by tunning the parameters of its structure, using different score functions and trying different optimization methods. We will start with the simplest classification model: logistic regression.


Logistic Regression:
--------------------

(penalty, solver and multi_class)

· Parameter tunning:

	- penalty: when regularization is added, it becomes ridge regression (L?) or lasso (L1). This basically prevents overfitting by penalizing models with extreme parameter values. There is no difference when using this parameter (l1 or l2), as the parameter values don't seem to be extreme (Use a visualization tool to check this, or another tool). See the new concepts section for more details (*3).

		* conclusions: find a way to plot and visualize parameter values. find a function which returns a score saying how extreme the values of a list of parameters are.

	- solver: ‘newton-cg’, ‘lbfgs’, ‘liblinear’: Algorithm to use in the optimization problem, i.e., the search or optimization method used to search over parameters and/or structures. Description of solvers (an online course in optimization would be useful):

		· ‘liblinear’: http://en.wikipedia.org/wiki/Coordinate_descent

		· ‘newton-cg’: http://en.wikipedia.org/wiki/Conjugate_gradient_method

		· ‘lbfgs’: http://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm


	IMPORTANT NOTE! Setting multi_class to “multinomial” with the “lbfgs” or “newton-cg” solver in LogisticRegression learns a true multinomial logistic regression model, which means that its probability estimates should be better calibrated than the default “one-vs-rest” setting. L-BFGS and newton-cg cannot optimize L1-penalized models, though, so the “multinomial” setting does not learn sparse models.



	* Notes on penalty-solver parameters: 

		- Lasso consists of a linear model trained with l_1 prior as regularizer.
		The implementation in the class Lasso uses coordinate descent as the algorithm to fit the coefficients.

		FOR SPARSE MODELS: L1 penalization yields sparse predicting weights. "liblinear" is the only solver available for this penalty term.

		FOR NON-SPARSE MODELS: The lbfgs and newton-cg solvers only support L2 penalization and are found to converge faster for some high dimensional data.

		* Compare:

			default : (penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0)

			(penalty='l1', solver='liblinear', multi_class='ovr'), 
			(penalty='l2', solver='lbfgs', multi_class='multinomial'), 
			(penalty='l2', solver='newton-cg', multi_class='multinomial')
			

	QUESTIONS: Using scikit-learn, why would you use bfgs optimization which is non-linear for a linear classifier as logistic regression? I am confused. Does the optimization method finds the optimum of the chosen score function? if so, which one? I can't choose it when defining the estimator. does the linearity or non-linearity of the score function depend on the model (whether it is linear or non-linear)?

	*NOTE: I had a problem related to the scikit-learn current version. Logistic Regression couldn't find a "solver" parameter, as it was using scikit-learn 0.15, even if anaconda had the 0.16 version. This is because before installing anaconda, I already had scikit learn 0.15 installed, so it was always refering to this instalation instead. I deleted it with "pip uninstall scikit-learn", and problem solved.

	- RESULTS FOR THE COMPARISON:

	Score for Logistic Regression, no parameters:  0.657738095238
	Score for Logistic Regression, penalty='l1', solver='liblinear', multi_class='ovr':  0.668485449735
	Score for Logistic Regression, penalty='l2', solver='lbfgs', multi_class='multinomial':  0.479332010582
	Score for Logistic Regression, penalty='l2', solver='newton-cg', multi_class='multinomial':  0.701058201058

	/home/cesarpumar/anaconda/lib/python2.7/site-packages/sklearn/utils/optimize.py:157: UserWarning: newton-cg failed to converge. Increase the number of iterations.
  	warnings.warn("newton-cg failed to converge. Increase the "

  	· newton-cg solver increases the score in ~5%, even when it fails to converge. Let's increase the number of iterations, as suggested. The default setting is 100, so I will try with 200:

  	(penalty='l2', solver='newton-cg', multi_class='multinomial', max_iter=200)

  	The improvement is minimum: 
  	Score for Logistic Regression, penalty='l2', solver='newton-cg', multi_class='multinomial':  0.702876984127


  	In conclusion, training the model for several classes (multinomial) rather than a one vs all for each class, improves the score in ~5% as long as newton-cg is used instead of lbfgs (score 0.701058201058 vs 0.479332010582). It would be interesting to understand why there is such a big difference between both optimization methods: http://scicomp.stackexchange.com/questions/507/bfgs-vs-conjugate-gradient-method
  	although it may be difficult it I don't understand the data. It would be recommended to take a statistics course/read a stats book to understand data much better(http://work.thaslwanter.at/Stats/html/index.html), then the performance of the optimization methods for this particular problem would be easier to understand too. Have a look at the MLPR slides (classification and optimization).


  	NOTE: READ THE DATA MINING BOOK AGAIN TO UNDERSTAND THE RELATIONSHIP BETWEEN MODEL, MLE, SCORE FUNCTION AND OPTIMIZATION/SEARCH METHOD



-----------------------------------
8. IMPLEMENT THE FULL ARCHITECTURE
-----------------------------------

Follow the steps defined in 5.2. and take notes of your testing results.


-------------------
9. DO MORE SCIENCE
-------------------

Try out new combinations of parameters and methods. Repeat previous experiments if needed. Take notes of your testing results. At this point there should not be that much engineering to do.



----------------
* NEW CONCEPTS:
----------------


(*1). Raw data, scaling data:
-----------------------------

Feature scaling is a method used to standardize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

Motivation:
Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. For example, the majority of classifiers calculate the distance between two points by the distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

Methods: 
- Rescaling: the simplest method is rescaling the range of features to scale the range in [0, 1] or [−1, 1]. Selecting the target range depends on the nature of the data.
- Standarization: the general method of calculation is to determine the distribution mean and standard deviation for each feature. Next we subtract the mean from each feature. Then we divide the values (mean is already subtracted) of each feature by its standard deviation.
- Scaling to unit length: scaling the components of a feature vector such that the complete vector has length one. This usually means dividing each component by the Euclidean length of the vector.


Application:
In stochastic gradient descent, feature scaling can sometimes improve the convergence speed of the algorithm. In support vector machines, it can reduce the time to find support vectors. Note that feature scaling changes the SVM result.



(*2). Feature selection for classification:
-------------------------------------------

There are three types of features in classification problems:

	- Flat features: features are assumed to be independent.

		· Filter models: filter models evaluate features without utilizing
			any classification algorithms. A typical filter algorithm consists of two steps. In the first step, it ranks features based on certain criteria. Feature evaluation could be either univariate or multivariate. In the univariate scheme, each feature is ranked independently of the feature space, while the multivariate scheme evaluates features in an batch way. Therefore, the multivariate scheme is naturally capable of handling redundant features. In the second step, the features with highest rankings are chosen to induce classification models.
			In the past decade, a number of performance criteria have been proposed for filter-based feature selection such as Fisher score, methods based on mutual information and ReliefF and its variants.

				- Fisher score: Fisher Score evaluates features individually; therefore, it cannot handle feature redundancy. Recently, Gu et al. proposed a generalized Fisher score to jointly select features,
				which aims to find an subset of features that maximize the lower bound of traditional Fisher
				score and solve the following problem.

				- Mutual Information: Due to its computational efficiency and simple interpretation, information gain is one of the most popular feature selection methods. It is used to measure the dependence between features and labels and calculates the information gain between the i-th feature f i and the class labels C. It uses entropy (in information theory, it is the average amount of information contained in each message received).

					· In information gain, a feature is relevant if it has a high information gain. Features are selected in a univariate way, therefore, information gain cannot handle redundant features.

					· a fast filter method FCBF based on mutual information was proposed to identify relevant features as well as redundancy among relevant features and measure feature-class and feature-feature correlation.

					· Minimum-Redundancy-Maximum-Relevance (mRmR) is also a mutual information based method and it selects features according to the maximal statistical dependency criterion [52]. Due to the difficulty in directly implementing the maximal dependency condition, mRmR is an approximation to maximizing the dependency between the joint distribution of the selected features and the classification variable.

				- Relief: the authors related the relevance evaluation criterion of ReliefF to the hypothesis of margin maximization, which explains why the algorithm provide superior performance in many applications.


		· Wrapped models: Filter models select features independent of any specific classifiers. wrapper models utilize a specific classifier to evaluate the quality of selected features, and offer a simple and powerful way to address the problem of feature selection, regardless of the chosen learning machine.

		Wrapper models obtain better predictive accuracy estimates than filter models.
		However, wrapper models are very computationally expensive compared to filter models. It produces better performance for the predefined classifier since we aim to select features that maximize the quality therefore the selected subset of features is inevitably biased to the predefined classifier.



		· Embedded models: Embedded Models embedding feature selection with classifier construction, have the advantages of (1) wrapper models - they include the interaction with the classification model and (2) filter models - they are far less computationally intensive than wrapper methods [40, 57, 46]. There are three types of embedded methods. The first are pruning methods that first utilizing all features to train a model and then attempt to eliminate some features by setting the corresponding coefficients to 0, while maintaining model performance such as recursive feature elimination using support vector machine (SVM) [19]. The second are models with a build-in mechanism for feature selection such as ID3 [55] and C4.5 [54]. The third are regularization models with objective functions that minimize fitting errors and in the mean time force the coefficients to be small or to be exact zero. Features with coefficients that are close to 0 are then eliminated [46]. Due to good performance, regularization models attract increasing attention. We will review some representative methods below based on a survey paper of embedded models based on regularization [46].


	- Streamming features: In this scenario, the candidate features are generated dynamically and the size of features is unknown. For example, the famous microblogging website Twitter produces more than 250 millions tweets per day and many new words (features) are generated such as abbreviations. When performing feature selection for tweets, it is not practical to wait until all features have been generated, thus it could be more preferable to streaming feature selection.

	A general framework for streaming feature selection is presented in Figure 9. A typical streaming feature selection will perform the following steps,
		• Step 1: Generating a new feature;
		• Step 2: Determining whether adding the newly generated feature to the set of currently selected features;
		• Step 3: Determining whether removing features from the set of currently selected features;
		• Step 4: Repeat Step 1 to Step 3.	

	Different algorithms may have different implementations for Step 2 and Step 3:

		· Grafting Algorithm
		· Alpha-investing Algorithm
		· The only streamming feature algorithm


	- Structured features: The models introduced in the last section assume that features are independent and totally overlook the feature structures [73]. However, for many real-world applications, the features exhibit certain intrinsic structures, e.g., spatial or temporal smoothness [65, 79], disjoint/overlapping groups [29], trees [33], and graphs [25]. Incorporating knowledge about the structures of features may significantly improve the classification performance and help identify the important features.

		· Features with group structure
		· Features with Tree Structure
		· Features with Graph Structure



		* READ ABOUT REGULARIZATION (L1, LASSO, etc)


		· Group structure: Lasso (does not consider the group structure and selects a subset of features among all groups), group Lasso (can perform group selection and select a subset of groups. Once the group is selected, all features in this group are selected), sparse group Lasso (can select groups and features in the selected groups at the same time). Groups for overlapping group Lasso regularization may overlap, while groups in group Lasso are disjoint.

		· Tree structure: In many applications, features can naturally be represented using certain tree structures. For example, the image pixels of the face image can be represented as a tree, where each parent node contains a series of child nodes that enjoy spatial locality; genes/proteins may form certain hierarchical tree structures [42]. Tree-guided group Lasso regularization is proposed for features represented as an index tree [33, 42, 30]. In the index tree, each leaf node represents a feature and each internal node denotes the group of the features that correspond to the leaf nodes of the subtree rooted at the given internal node. Each internal node in the tree is associated with a weight that represents the height of the subtree, or how tightly the features in the group for that internal node are correlated. 

		· Graph structure: features form an undirected graph, where the nodes represent the features, and the edges imply the relationships between features. Several recent studies have shown that the estimation accuracy can be improved using dependency information encoded as a graph. If nodes i and j are connected by an edge in E, then the i-th feature and the j-th feature are more likely to be selected together, and they should have similar weights. For features with graph structure, a subset of highly connected features in the graph is likely to be selected or not selected as a whole.




(*3). Regularization:
---------------------

In statistics and machine learning, regularization methods are used for model selection, in particular to prevent overfitting by penalizing models with extreme parameter values. The most common variants in machine learning are L₁ and L₂ regularization, which can be added to learning algorithms that minimize a loss function E(X, Y) by instead minimizing E(X, Y) + α‖w‖, where w is the model's weight vector, ‖·‖ is either the L₁ norm or the squared L₂ norm, and α is a free parameter that needs to be tuned empirically (typically by cross-validation; see hyperparameter optimization). This method applies to many models. When applied in linear regression, the resulting models are termed ridge regression or lasso, but regularization is also employed in (binary and multiclass) logistic regression, neural nets, support vector machines, conditional random fields and some matrix decomposition methods. L₂ regularization may also be called "weight decay", in particular in the setting of neural nets.

L₁ regularization is often preferred because it produces sparse models and thus performs feature selection within the learning algorithm, but since the L₁ norm is not differentiable, it may require changes to learning algorithms, in particular gradient-based learners.[1][2]