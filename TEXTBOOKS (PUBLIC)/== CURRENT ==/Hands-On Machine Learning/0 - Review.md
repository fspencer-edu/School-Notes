
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

<img src="/images/Pasted image 20260301164634.png" alt="image" width="500">

- Mean absolute error (MAE)

<img src="/images/Pasted image 20260301164904.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260301212827.png" alt="image" width="500">

- Recall/sensitivity or true positive rate (TPR)

<img src="/images/Pasted image 20260301212841.png" alt="image" width="500">

- $F_1$ score/harmonic mean of precision and recall

<img src="/images/Pasted image 20260301213020.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260301214211.png" alt="image" width="500">

- Linear regression model prediction (vectorized form)

<img src="/images/Pasted image 20260301214250.png" alt="image" width="500">

- Components
	- Predicted value
	- Feature value
	- Parameters
	- Bias/intercept term


- MSE cost function for a linear regression model

<img src="/images/Pasted image 20260301214402.png" alt="image" width="500">

- Normal equation

<img src="/images/Pasted image 20260301214420.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260301215136.png" alt="image" width="500">

- Gradient vector of the cost function

<img src="/images/Pasted image 20260301215157.png" alt="image" width="500">

- Gradient descent step

<img src="/images/Pasted image 20260301215213.png" alt="image" width="500">
$\eta$ - learning rate
$\epsilon$ = tolerance

- Stochastic gradient descent
	- Learning schedule

- Mini-batch gradient descent

<img src="/images/Pasted image 20260301215557.png" alt="image" width="500">

- Polynomial regression
	- Quadratic

- Learning curves

- 3 generalization errors
	- Bias
	- Variance
	- Irreducible error


- Regularized linear models
	- Ridge regression (Tikhonov regularization)

- Ridge regression cost function

<img src="/images/Pasted image 20260301215936.png" alt="image" width="500">

- Ridge regression closed-form solution

<img src="/images/Pasted image 20260301220041.png" alt="image" width="500">


<img src="/images/Pasted image 20260301220056.png" alt="image" width="500">

- Lasso regression
	- Least absolute shrinkage and selection operator regression
	- Uses the $\ell_1$ norm of the weight vector instead of the square of the $\ell_2$ norm 

<img src="/images/Pasted image 20260301220210.png" alt="image" width="500">

<img src="/images/Pasted image 20260301220227.png" alt="image" width="500">


- Lasso regression sub-gradient vector

<img src="/images/Pasted image 20260301220336.png" alt="image" width="500">

- Elastic net cost function
	- Between lasso and ridge

<img src="/images/Pasted image 20260301220359.png" alt="image" width="500">

- Early stopping
	- Deep copy hyperparameter and learned parameters

- Logistic regression model estimated probability (vectorized form)

<img src="/images/Pasted image 20260301220520.png" alt="image" width="500">

- Logistic function

<img src="/images/Pasted image 20260301220545.png" alt="image" width="500">

- Logistic regression model prediction using a 50% threshold probability

<img src="/images/Pasted image 20260301220615.png" alt="image" width="500">

- Cost function of a single training instance

<img src="/images/Pasted image 20260301220640.png" alt="image" width="500">


- Logistic regression cost function (log loss)

<img src="/images/Pasted image 20260301220651.png" alt="image" width="500">

- Logistic cost function partial derivatives

<img src="/images/Pasted image 20260301220722.png" alt="image" width="500">

- Decision boundary

- Softmax regression
	- Multi-nominal logistic regression
	- Used for mutually exclusive classes

- Softmax score for class k

<img src="/images/Pasted image 20260301220901.png" alt="image" width="500">

- Softmax function

<img src="/images/Pasted image 20260301220918.png" alt="image" width="500">

- Softmax regression classifier prediction

<img src="/images/Pasted image 20260301220947.png" alt="image" width="500">

- Cross entropy cost function

<img src="/images/Pasted image 20260301221025.png" alt="image" width="500">

- Cross entropy gradient vector for class k

<img src="/images/Pasted image 20260301221047.png" alt="image" width="500">

<img src="/images/Pasted image 20260301221057.png" alt="image" width="500">

## Support Vector Machines

- SVM
	- Linear
	- Non-linear
	- Regression
	- Novelty detection

- Margin classification
	- Soft
	- Hard

- Polynomial kernel
- Similarity features
- Gaussian RBF kernel
- String kernels

- Hard margin linear SVM classifier objective

<img src="/images/Pasted image 20260301225936.png" alt="image" width="500">

- Soft margin linear SVM classifier objective

<img src="/images/Pasted image 20260301230008.png" alt="image" width="500">

- Hard and soft margin problems are both convex quadratic optimization problems with linear constraints
	- Quadratic programming (QP) problems
- Hinge loss or squared hinge loss

- Dual form of the linear SVM objective

<img src="/images/Pasted image 20260301230248.png" alt="image" width="500">

- Dual solution to primal solution

<img src="/images/Pasted image 20260301230355.png" alt="image" width="500">

- Second-degree polynomial mapping

<img src="/images/Pasted image 20260301230437.png" alt="image" width="500">

- Kernel trick for a second-degree polynomial mapping

<img src="/images/Pasted image 20260301230455.png" alt="image" width="500">

- Common kernels

<img src="/images/Pasted image 20260301230523.png" alt="image" width="500">

- Making predictions with a kernelized SVM

<img src="/images/Pasted image 20260301230556.png" alt="image" width="500">


- Using the kernel trick to compute the bias term

<img src="/images/Pasted image 20260301230615.png" alt="image" width="500">



## Decision Trees

- Tree components
	- Root node
	- Internal/split node
	- Leaf node

- Gini impurity

<img src="/images/Pasted image 20260301230834.png" alt="image" width="500">

- Classification and Regression Tree (CART)
	- Greedy algorithm

- CART cost function for classification

<img src="/images/Pasted image 20260301231015.png" alt="image" width="500">

- Complexity
	- NP-complete
	- $O(exp(m))$

- Entropy

<img src="/images/Pasted image 20260301231143.png" alt="image" width="500">

- $\chi^2$ test
	- Estimate the probability that the improvement is purely the result of chance (null hypothesis)


- Decision tree regressor

- CART cost function for regression

<img src="/images/Pasted image 20260301231408.png" alt="image" width="500">

- Axis orientation
	- Solve with PCA

## Ensemble Learning and Random Forests

- Ensemble learning
	- `VotingClassifier`
	- Stacking

- Sampling
	- Bagging
	- Pasting

- Random patches method
	- Sampling both training instances and features
- Random subspaces method
	- Keeping all training instances, but sampling features

- Boosting
	- AdaBoost
		- SAMME (Stagewise Additive MOdeling using a Multiclass Exponential loss function)
	- Gradient boosting
		- Histogram -based gradient boosting (HGB)

- Weighted error rate of the jth predicator

<img src="/images/Pasted image 20260302084433.png" alt="image" width="500">

- Predictor weight

<img src="/images/Pasted image 20260302084447.png" alt="image" width="500">

- Weight update rule

<img src="/images/Pasted image 20260302084500.png" alt="image" width="500">



## Dimensionality Reduction

- Dimensionality reduction
	- Projection
		- Random projection
			- `GaussianRandomProjection`
			- `SparseRandomProjection`
	- Manifold learning
		- Locally linear embedding (LLE)
	- Principal component analysis (PCA)
		- Explained variance ratio
		- Reconstruction error
		- Randomized PCA
		- Incremental PCA (IPCA)


- Other dimensionality reduction techniques
	- Multidimensional scaling (MDS)
		- `sklearn.manifold.MDS`
	- Isomap
		- `sklearn.maniford.Isomap`
	- t-distributed stochastic neighbour embedding (t-SNE)
		- `sklearn.manifold.TSNE`
	- Linear discriminant analysis (LDA)
		- `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

- Principal component matrix

<img src="/images/Pasted image 20260302085008.png" alt="image" width="500">

- Projecting the training set down to d dimensions

<img src="/images/Pasted image 20260302085049.png" alt="image" width="500">

- PCA inverse transformation, back to the original number of dimensions

<img src="/images/Pasted image 20260302085207.png" alt="image" width="500">

- LLE step 1: Linearly modeling local relationships

<img src="/images/Pasted image 20260302085439.png" alt="image" width="500">

- LLE step 2: Reducing dimensionality while preserving relationships

<img src="/images/Pasted image 20260302085503.png" alt="image" width="500">


## Unsupervised Learning

- Clustering
	- k-means
		- Centroid initialization
		- Accelerated k-means
		- Mini-batch k-means
		- Silhouette score
	- Density estimation
		-  Density based spatial clustering of applications with noise (DBSCAN)
			- Hierarchical DBSCAN (HDDBSCAN)
		- Probability density function (PDF)
	- Anomaly detection
		- Fast-MCD (minimum covariance determinant)
		- Isolation forest
		- Local outlier factor (LOF)
		- One-class SVM
		- OCA and other dimensionality reduction with inverse transformation

- Other clustering algorithms
	- Agglomerative clustering
	- BIRCH (balanced iterative reducing and clustering using hierarchies)
	- Mean shift
	- Affinity propagation
	- Spectral clustering

- Uncertainty sampling

- Gaussian mixture model (GMM)

- Bayesian information criterion (BIC) and Akaike information criterion (AIC)

<img src="/images/Pasted image 20260302090453.png" alt="image" width="500">


# Part 2 - Neural Networks and Deep Learning

## Introduction to Artificial Neural Networks with Keras

- Artificial neural networks (ANNs)
- Perceptron
	- Linear threshold unit (LTU)
	- Heaviside step function

- Common step functions used in perception (threshold=0)

<img src="/images/Pasted image 20260302090924.png" alt="image" width="500">

- Layer types
	- Input layer
	- Dense layer
	- Output layer

- Computing the outputs of a fully connected layer

<img src="/images/Pasted image 20260302091001.png" alt="image" width="500">

- Components
	- Output matrix, $\hat{Y}$
	- Activation function, $\theta$
	- Matrix of input features, $X$
	- Weight matrix, $W$
	- Bias, $b$

- Perception learning rule (weight update)

<img src="/images/Pasted image 20260302091127.png" alt="image" width="500">

- Multilayer perceptrons
	- Feedforward neural network (FNN)
	- Deep neural network (DNN)
	- Reverse-mode automatic differentiation (forward, and backward)
	- Backpropagation
		- Chain rule

- Activation functions
	- Sigmoid
	- Hyperbolic tangent
		- $tanh(z) = 2\sigma(2z)-1$
	- Rectified linear unit function
		- $ReLU(z) = max(0, z)$
	- Softmax

- Loss function
	- x-entropy

- Model optimization
	- Callbacks
	- Neural network hyperparameters
		- Parameter efficiency
		- Learning rate
		- Optimizer
		- Batch size
		- Activation function
		- Number of iterations
	- Transfer learning

- Keras

```python
# Wide and deep model

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 0 - load/prepare data

# 1 - train.valid/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
	X_train_full, y_train_full, test_size=0.2, random_state=42
)

# 2 - splice wide/deep inputs
# wide: 5 features
# deep: 6 features
X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_train[:, :5], X_train[:, 2:]
X_test_wide, X_test_deep = X_train[:, :5], X_train[:, 2:]

X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

# 3 - build the wide and deep model
input_wide = tf.keras.layers.Input(shape=[5], name"wide_input")
input_deep = tf.keras.layers.Input(shape=[5], name"deep_input")

norm_layer_wide = tf.keras.layers.Normalization(name="wide_norm")
norm_layer_deep = tf.keras.layers.Normalization(name="deeo_norm")

norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_wide(input_deep)

hidden1 = tf.keras.layers.Dense(30, activation="relu", name="hidden1")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu", name="hidden2")(hidden1)

concat = tf.keras.layers.Concatenate(name="concat")([norm_wide, hidden2])
output = tf.keras.layers.Dense(1, name="output")(concat)

model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=output, name="wide_deep_model")

# 4 - adapt normalization layers
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)

# 5 - compile
optimizer = tf.keras.optimizer.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.summary()

# 6 - train
history = model.fit(
	(X_train_wide, X_train_deep),
	y_train,
	epochs=20,
	validation_data((X_valid_wide, X_valid_deep), y_valid),
	verbose=1
)

# 7 - evaluate
mse_test, rmse_test = model.evaluate((X_test_wide, X_test_deep), y_test, verbose=0)
print(f"Test MSE: {mse_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")

# 8 - Predict
y_pred = model.precict((X_new_wide, X_new_deep), verbose=0)
print("Predications for 3 samples:\n", y_pred.reshape(-1)) 
```

## Training Deep Neural Networks

- Gradient problems
	- Vanishing
	- Exploding

- Glorot initialization (sigmoid activation function)

<img src="/images/Pasted image 20260302093518.png" alt="image" width="500">

- Leaky ReLU
- Randomized leaky ReLU (RReLU)
- Parametric leaky ReLU (PReLU)
- Exponential linear unit (ELU)
- Scaled ELU (SELU)
- Gaussian Error Linear Unit (GELU)
- Sigmoid linear unit (SiLU)
- Swish
- Mish

- ELU activation function

<img src="/images/Pasted image 20260302093629.png" alt="image" width="500">

- GELU activation function

<img src="/images/Pasted image 20260302093730.png" alt="image" width="500">

<img src="/images/Pasted image 20260302093825.png" alt="image" width="500">

- Batch normalization algorithm

<img src="/images/Pasted image 20260302093903.png" alt="image" width="500">

- Gradient clipping
- Transfer learning

- Unsupervised model
	- Autoencoder
	- Generative adversarial network (GAN)
- Greedy layer-wise pretraining

- Optimization algorithms
	- Momentum
	- Nesterov accelerated gradient
	- AdaGrad
	- RMSProp
	- Adam

- Momentum algorithm

<img src="/images/Pasted image 20260302094223.png" alt="image" width="500">

- Nesterov accelerated gradient (NAG) algorithm

<img src="/images/Pasted image 20260302094243.png" alt="image" width="500">

<img src="/images/Pasted image 20260302094300.png" alt="image" width="500">

- AdaGrad algorithm

<img src="/images/Pasted image 20260302094315.png" alt="image" width="500">


<img src="/images/Pasted image 20260302094344.png" alt="image" width="500">


- RMSProp algorithm

<img src="/images/Pasted image 20260302094331.png" alt="image" width="500">

- Adaptive moment estimation (Adam)
	- AdaMax
	- Nadam
	- AdamW

<img src="/images/Pasted image 20260302094426.png" alt="image" width="500">


- Learning rate scheduling
	- Exponential
	- Piecewise constant
	- Performance
	- Power
	- 1cycle

<img src="/images/Pasted image 20260302094509.png" alt="image" width="500">

- Avoiding overfitting through regularization
	- $\ell_1$ and $\ell_2$ Regularization

- Dropout regularization
	- Monte Carlo (MC) dropout
- Max-Norm regularization

- Default DNN configuration (hyperparameter => default value)
	- Kernel initializer => He initialization
	- Activation function => ReLU is shallow, Swish if deep
	- Normalization => none if shallow, batch norm if deep
	- Regularization => early stopping, weight decay if needed
	- Optimizer => Nesterov accelerated gradients or AdamW
	- Learning rate schedule => Performance or 1 cycle


- DNN configuration for self-normalization net
	- Kernel initialization => LeCun
	- Activation function => SELU
	- Normalization => None
	- Regularization => Alpha dropout
	- Optimizer => Nesterov accelerated gradients
	- Learning rate => Performance or 1 cycle

## Custom Models and Training with TensorFlow

<img src="/images/Pasted image 20260302095208.png" alt="image" width="500">

- TensorFlow data structures
	- Sparse tensors
	- Tensor arrays
	- Ragged tansors
	- String tensors
	- Sets
	- Queues

## Loading and Preprocessing

- TFRecord format
	- CRC checksum
	- Protobufs
- Chaining transformations
- Shuffling data
- Interleaving lines
- Prefetching
- Discretization
- Category encoding layer
	- Multi-hot encoding
- Hasing collision
- Embedding
- Representation learning

## Deep Computer Vision Using Convolutional Neural Networks

## Processing Sequences Using RNNs and CNNs

## Natural Language Processing with RNNs and Attention

## Autoencoders, GANs, and Diffusion Models

## Reinforcement Learning

## Training and Deploying TensorFlow Models at Scale