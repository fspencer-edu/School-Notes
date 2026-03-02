
# PART 1 - The Fundamentals of Machine Learning

## The Machine Learning Landscape

- Supervision training type
	- Supervised
	- Unsupervised
	- Semi-supervised
	- Self-supervised
	- Reinforcement
- Learning
	- Online
	- Batch/offline
- Model type
	- Instance based
	- Model based

- Performance metrics
	- Fitness/utility function
	- Cost function

- Models
	- Linear regression
	- K-nearest neighbour (KNN)
	- Decision tree regressor
	- Random forest regressor

- Training data results
	- Underfitting
	- Overfitting

- Testing
	- Hold out method
	- Cross validation
- Validating
	- Training error
	- Generalization error

## End-to-End Machine Learning Project

- Models
	- Multiple regression
	- Univariate regression
	- Multivariate regression

Performance Measure
- Root mean squared error (RMSE)
	- Prediction error is $\hat{y} - y$

![[Pasted image 20260301164634.png]]

- Mean absolute error (MAE)

![[Pasted image 20260301164904.png]]

- Standard correlation coefficient (Pearson's)
	- `corr()`


- Measures
	- Standard deviation
	- Percentiles

- Training
	- Random samples
	- Stratified sampling

- Data cleaning
	- Imputation
		- `fillna()`
		- `SimpleImputer`
	- Remove redundant attributes
		- `dropna(), drop()`

- Scikit-Learn objects
	- Consistency
		- Estimators
		- Transformers
		- Predictors
	- Inspection
	- Non-proliferation of classes
	- Composition
	- Sensible defaults

- Categorical data types
	- `OrdinalEncoder`
	- `OneHotEncoder`
		- Sparse matrix
	- `get_dummies(df)`
		- Converts each categorical feature into a one-hot representation, with one binary feature per category

- Replace category with a learnable, low-dimensional vector called embedding
	- Representation learning

- Feature scaling
	- Min-max scaling (normalization)
		- `MinMaxScaler(feature_range=(-1, 1))`
	- Standardization
		- `StandardScaler()`

- Distribution
	- Bucketizing
- Similarity
	- Radial basis function (RBF)

- Transformations
	- `inverse_transform()`
	- `TransformedTargetRegressor`
	- `FunctionTransfomer`
	- `ColumnTransformer`

- Evaluation on training set
	- k-fold cross validation
	- Bootstrap

- Hyperparameters
	- `GridSearchCV`
	- `RandomizedSearchCV`

- Feature importance
	- `feature_importances_`

## Classification

- Binary classifier
	- Stochastic radient descent (SGD)

- Measuring accuracy
	- Cross validation
		- Stratified k folds
	- Dummy classifier
	- Confusion matrix (CM)
		- Column = predicted classes
		- Row = actual classes

```c
[a, b]
```


## Training Models

## Support Vector Machines

## Decision Trees

## Ensemble Learning and Random Forests

## Dimensionality Reduction

## Unsupervised Learning

# Part 2 - Neural Networks and Deep Learning

## Introduction to Artificial Neural Networks with Keras

## Training Deep Neural Networks

## Custom Models and Training with TensorFlow

## Loading and PReprocessing

## Deep Computer Vision Using Convolutional Neural Networks

## Processing Sequences Using RNNs and CNNs

## Natural Language Processing with RNNs and Attention

## Autoencoders, GANs, and Diffusion Models

## Reinforcement Learning

## Training and Deploying TensorFlow Models at Scale