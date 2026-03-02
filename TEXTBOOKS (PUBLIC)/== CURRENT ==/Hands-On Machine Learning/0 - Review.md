
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
[a, b
 c, d]
 
[TN, FP,
 FN, TP]
```
- Precision

![[Pasted image 20260301212827.png]]

- Recall/sensitivity or true positive rate (TPR)

![[Pasted image 20260301212841.png]]

- $F_1$ score/harmonic mean of precision and recall

![[Pasted image 20260301213020.png]]

- Precision/recall trade-off
	- Decision function
	- Decision threshold

- ROC curve (Receiver Operating Characteristic)
	- FPR vs. TPR
	- AUC (area under the curve)

- Binary classifiers
	- Stochastic gradient descent
	- Single value decomposition
- Multinomial classifiers
	- Logistic regression
	- Random forest
	- Gaussian Naives Bayes

- Training strategies
	- One vs all
	- One vs one

- Error analysis
	- Weights
	- Data augmentation

- Multilabel classification
	- Binary tags

- Support and confidence
	- `ClassifierChain`

- Multioutput classification

## Training Models

- Models
	- Closed
	- Iterative

- Linear regression model prediction

![[Pasted image 20260301214211.png]]

- Linear regression model prediction (vectorized form)

![[Pasted image 20260301214250.png]]

- Components
	- Predicted value
	- Feature value
	- Parameters
	- Bias/intercept term


- MSE cost function for a linear regression model

![[Pasted image 20260301214402.png]]

- Normal equation

![[Pasted image 20260301214420.png]]

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b @ y

>>> theta_best
array([[4.21509616],
       [2.77011339]])
```

function => $y = 4 +3x_1$ + Gaussian noise

predicted => $\theta_0 = 4.215, \theta_1 = 2.770$

- Singular value decomposition (SVD)
	- Pseudoinverse = $X^+ = V \sum^+U^T$
	- Moore-Penrose inverse

- Gradient descent
	- Random initialization
	- Learning rate
	- Feature scaling

- Batch gradient descent
	- Partial derivative

![[Pasted image 20260301215136.png]]

- Gradient vector of the cost function

![[Pasted image 20260301215157.png]]

- Gradient descent step

![[Pasted image 20260301215213.png]]
$\eta$ - learning rate
$\epsilon$ = tolerance

- Stochastic gradient descent
	- Learning schedule

- Mini-batch gradient descent

![[Pasted image 20260301215557.png]]

- Polynomial regression
	- Quadratic

- 



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