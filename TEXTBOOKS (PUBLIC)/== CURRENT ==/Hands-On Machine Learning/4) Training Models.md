
**2 Ways to Train Linear Regression Model**
- Closed form
	- Directly computes the model parameters that best fir the model to the training set
- Iterative optimization approach called gradient descent (GD)
	- Gradually tweaks the model parameters to minimize the cost function over the training set
		- Batch GD
		- Mini batch GD
		- Stochastic GD

**Polynomial Regression**
- A more complex model that gan fit non-linear datasets
- More prone to overfitting the training data

- Logistic regression
- Softmax regression

# Linear Regression

Life Satisfaction Regression Model

$life_s = \theta_1 + \theta_1 \times GDP_{perCaptia}$

- $\theta$ are the models parameters
- A linear model makes a prediction by computing a weighted sum of the input features, plus a constant called the bias terms (intercept term)

**Linear Regression Model Prediction**

$\hat{y} = \theta_1 + \theta_1x_1 + \theta_2x_2 + ...+$

$\hat{y}$ = predicted value
$n$ = number of features
$x_i$ = feature value
$\theta_j$ = model parameter

**Linear Regression Model Prediction (Vectorized Form)**
$\hat{y} = h_{\theta} = \theta \cdot x$

$h_{\theta}$ = hypothesis function
$x$ = instance feature vector
$\theta \cdot x$ = dot product

- Vectors are represented as column vectors
- Measure how well the model fits the training data
- To train a linear regression model, find the value of $\theta$ that minimizes the RMSE
- Easier to minimize the mean square error (MSE) that RMSE

- Learning algorithms will optimize a different loss function during training than the performance measure on the final model
- A good performance metric is as close as possible to the final business objective
- Classifiers are often trained using a cost function, but evaluated using precision/recall

**MSE cost function for a linear regression model**

<img src="/images/Pasted image 20260202083046.png" alt="image" width="500">

- Model is parametrized by the vector $\theta$

## The Normal Equation

- To find $\theta$ that minimizes the MSE, there is a closed-form solution, normal equation

**Normal Equation**

$\hat{\theta} = (X^TX)^{-1}y$

$\hat{\theta}$ = value of $\theta$ that minimizes the cost function
$y$ = vector of target values

```python
import numpy as np

np.random.seed(42)
n = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
```

<img src="/images/Pasted image 20260202083441.png" alt="image" width="500">

- Compute $\hat{\theta}$ using the Normal equation

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.t @ y
```
- @ for matrix multiplication
- The function used to generate the data is $y = 4 + 3x_1 +$ Gaussian noise
- The ideal $\theta_0 = 4$ and $\theta_1 = 3$
```python
theta_best
array([[4.21509616],
       [2.77011339]])
```

- Noise made it difficult to recover the exact parameters

```python
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feture(X_new)
y_predict = X_new_b @ theta_best
y_predict
array([[4.21509616],
       [9.75532293]])
       
import matplotlib.pyplot as plt
plt.plot(X_new, y_predict, "r-", labal="Predictions")
plt.show()
```

<img src="/images/Pasted image 20260202084014.png" alt="image" width="500">

- Perform linear regression
- Scikit separates the bias terms from the feature weights

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
lin_reg.predict(X_new)
array([[4.21509616],
       [9.75532293]])
theta_best_svd, residuals, rank, s = np,linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
array([[4.21509616],
       [2.77011339]])
```
- Compute $\hat{\theta} = X^+y$
	- $X^+$ is the pseudoinverse of X (Moore-Penrose inverse)

```python
np.linalg.pinv(X_b) @ y
array([[4.21509616],
       [2.77011339]])
```
- Pseudoinverse is computed using a standard matrix factorization technique called singular value decomposition (SVD) that can decompose the training set matrix into the matrix multiplication of 3 matrices

$X^+ = V\sum^+U^T$

## Computational Complexity

- The normal equation computes a $(n+1) \times (n+1)$ matrix
- Computational complexity of inverting a matrix is typically about $O(n^{2.4})$ to $O(n^3)$
- If the features are doubled, then multiply the computation time by the complexity
- The SVD approach class is about $O(n^2)$

- After training linear regression model, computational complexity is linear with regards to both number of instances and features

# Gradient Descent

- Gradient Descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems
- Tweak parameters iteratively in order to minimize a cost function
- Measures the local gradient of the error function with regards to the parameter vector, and goes in the direction of descending gradient
- Minimum is at zero
- Fill $\theta$ with random values (random initialization)
- Converge at minimum

<img src="/images/Pasted image 20260202085125.png" alt="image" width="500">

- Size of step is determined by learning rate hyperparameter
- If the random initialization starts on the left, then it will converge to a local minimum, not the global minimum

<img src="/images/Pasted image 20260202085244.png" alt="image" width="500">

- MSE cost function for a linear regression model happens to be a convex function
- MSE function is convex, therefore no local minima, just one global minimum
- Continuous function with a slope the never change abruptly
- Gradient descent is guaranteed to approach arbitrarily close the global minimum
- Cost function can be elongated depending on feature scales

<img src="/images/Pasted image 20260202085624.png" alt="image" width="500">

- Training a model means searching for a combination of model parameters that minimizes a cost function
- More parameters a mode has, the mode dimensions the space has, and the harder the search is

## Batch Gradient Descent

- Calculates how much the cost function will change if you change $\theta_j$ a little
	- Partial derivative

**Partial Derivatives of the Cost Function**

<img src="/images/Pasted image 20260202092256.png" alt="image" width="500">

- The gradient vector, $\Delta_{\theta}MSE(\theta)$, contains all partial derivatives of the cost function

**Gradient Vector of the Cost Function**

<img src="/images/Pasted image 20260202092403.png" alt="image" width="500">

- Batch gradient descent uses the whole batch of training data at every step
- Slow
- Scales will with the number of features
- Faster on a linear regression model with many features compared to normal equation or SVD decomposition

 - Subtract $\Delta_{\theta}MSE(\theta)$ from $\theta$ to get downhill vector

**Gradient descent step**

<img src="/images/Pasted image 20260202092633.png" alt="image" width="500">

```python
eta = 0.1 # learning rate
n_epochs = 1000
m = len(X_b)

np.random.seed(42)
theta = np.random.randn(2, 1)
for epcoh in range(n_epcochs):
	gradients = 2 / m * X_b.T @ (X_b @ theta - y)
	theta = theta - eta * gradients
theta
array([[4.21509616],
       [2.77011339]])
```

- Each iteration over the training set is called an epoch

<img src="/images/Pasted image 20260202092906.png" alt="image" width="500">

- To find a good learning rate, use a grid search
- Set a very large number of epochs but to interrupt the algorithm when the gradient vector becomes tiny
- Norm becomes smaller than the tolerance, $\epsilon$, and reached minimum
- A convex slope will converge $O(1/\epsilon)$ iterations
- Uses entire training set to compute the gradient at each step


## Stochastic Gradient Descent

- Stochastic gradient descent picks a random instance in the training set at every step and computes the gradients based only on that single instance
- Algorithm is faster, but less regular than batch GD
- Cost function will bounce up and down, decreasing only on overage
- Final value is good but not optimal

<img src="/images/Pasted image 20260202093510.png" alt="image" width="500">

- SGD has a better change for irregular cost functions
- Gradually reduce the learning rate
	- Simulated annealing
		- An algorithm inspired by the processing in metallurgy of annealing, where molten metal is slowly cooled down
- The function that determines the learning rate at each iteration is called the learning schedule

```python
n_epochs = 50
t0, t1 = 5, 50 # learning sch

def learning_schedule(t):
	return t0/ (t + t1)
	
np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
	for iteration in range(m):
	random_index = np.random.randint(m)
	xi = X_b[random_index : random_index + 1]
	yi = y[random_index : random_index + 1]
	gradients = 2 * xi.T @ (xi @ theta - yi)
	eta = learning_schedule(epoch * m + iteration)
	theta = theta - eta * gradients
theta
array([[4.21076011],
       [2.74856079]])
```

<img src="/images/Pasted image 20260202094008.png" alt="image" width="500">

- Some instance may be picked several times per epoch
- Another approach is to shuffle the training set, then go through instance by instance
- Training instances must be independent and identically distributed (IID) to ensure that the parameters get pulled towards the global optimum, on average
- Shuffle the instances, or SGD will optimize for one label

```python
from sklearn.linear_model import SGDRegression

sgd_reg = SGDRegression(max_iter=1000, tol=1e-5, pentalty=None, eta0=0.01,
		n_iter_no_change=100, random_state_42)
sgd_reg.fit(X, y.ravel())
```


## Mini-Batch Gradient Descent

- At each step, instead of computing the gradients based on the full training set (GD) or on one instance (SGD), mini-batch computes the gradients on small random sets of instances
- Performance boost from hardware optimization of matrix operations, GPUs
- Walking closer to min that SGD, harder to escape the local minima

<img src="/images/Pasted image 20260202094649.png" alt="image" width="500">

<img src="/images/Pasted image 20260202094720.png" alt="image" width="500">

# Polynomial Regression

- Add powers of each feature as new features, then train a linear model on this extended set of features
- Quadratic equation, $y = ax^2 + bx + c$

```python
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randm(m, 1)
```

<img src="/images/Pasted image 20260202094910.png" alt="image" width="500">

```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_Transform(X)
X[0]
array([-0.75275929])
X_poly[0]
array([-0.75275929,  0.56664654])

lin_reg = LinearRegression()
lin_reg.git(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
(array([1.78134581]), array([[0.93366893, 0.56456263]]))
```

<img src="/images/Pasted image 20260202095115.png" alt="image" width="500">

<img src="/images/Pasted image 20260202095131.png" alt="image" width="500">


# Learning Curves

- If you perform high-degree polynomial regression, the fitted training data is better than plain linear regression

<img src="/images/Pasted image 20260202095333.png" alt="image" width="500">

- Model that will generalize best is the quadratic model
- Use cross validation to estimate a model's generalization performance
- Learning curves
	- Plots a model's training error and validation error as a function of the training iteration


```python
from sklearn.model_selection import learning_curve
train_size, train_score, valid_scores = learning_curve(
	LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
	scoring="neg_root_mean_squared_error"
)

train_erorrs = -train_scores.mean(axis=1)
valid_erorrs = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=2, label="valid")
plt.show()
```
<img src="/images/Pasted image 20260202095812.png" alt="image" width="500">

- The model is underfitting
- Error ends up at a plateau

```python
from sklearn.pipeline import make_pipeline

polynormial_regression = make_pipeline(
	PolynomialFeatures(degree=10, include_bias=False),
	LinearRegrssion()
)

train_size, train_scores, valid_scores = learing_curve(
	polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
	scoring="neg_root_mean_sqaured_error"
)
```

<img src="/images/Pasted image 20260202100244.png" alt="image" width="500">

- Error on the training data is much lower
- There is a gap between the cruves
	- Model perform better on training than validation data
	- Overfitting model
- Use a larger training set, to get curves to meet

**Bias/Variance Trade-Off**
- Model's generalization error can be expressed as the sum of 3 very different errors

Bias
- Due to wrong assumptions

Variance
- Model's excessive sensitivity to small variations in the training data

Irreducible error
- Noisiness of data
- Clean up data

- Increasing a model's complexity will increase its variance and reduce its bias
- Reducing complexity increases its bias and reduces its variance

# Regularized Linear Models

- A good way to reduce overfitting it to regularize the model
	- Fewer degrees of freedom
	- Constraining weights of the models
	- Reducing the number of polynomial degrees
- Ridge regression
- Lasso regression
- Elastic net regression

## Ridge Regression

- Ridge regression also called Tikhonov regularization
- Regularized version of linear regression
- A regularization term equal to $\alpha / m \sum^n_{i=1}\theta^2_i$ is added to the MSE
- Regularization term is only added to the cost function during training
- Use unregularized MSE to evaluate model's performance
- $\alpha$ controls the regularization
	- If = 0, then ridge regression is linear
	- If larger, then all weights end up very close to zero, and result is a flat line through data's mean

**Ridge regression cost function**

<img src="/images/Pasted image 20260202101100.png" alt="image" width="500">

- $\theta_0$, bias term is not regularized
- Scale data before performing ridge regression

<img src="/images/Pasted image 20260202101247.png" alt="image" width="500">

- Perform ridge regression by computing closed-form equation or GD

**Ridge Regression Closed-Form Solution**

<img src="/images/Pasted image 20260202101343.png" alt="image" width="500">

```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
array([[1.55325833]])

# SGD
sgd_reg = SGDRegressor(penalty="12", alpha=0.1/m, tol=None,
			max_iter=1000, eta=0.01, random_state=42)
sgd_reg.fit(X, y.rave1())
sgd_reg.predict([[1.5]])
array([1.55302613])
```

## Lasso Regression

- Least absolute shrinkage and selection operator regression (lasso)
- Another regularized version of linear regression
- Adds a regularization term to the cost function
- Uses $\ell_1$ norm of the weight vector instance of the square of $\ell_2$ norm

**Lasso regression cost function**

<img src="/images/Pasted image 20260202101758.png" alt="image" width="500">

<img src="/images/Pasted image 20260202101808.png" alt="image" width="500">

- Tends to eliminate the weights of the least important features


<img src="/images/Pasted image 20260202101854.png" alt="image" width="500">

- The lasso cost function is not differentiable at $\theta_i$, but GD works for sub-gradient vector

**Lasso Regression Subgradient Vector**

<img src="/images/Pasted image 20260202102153.png" alt="image" width="500">

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lassor_reg.predict([[1.5]])
array([1.53788174])
```

## Elastic Net Regression

- Middle ground between ridge and lasso regression
- Regularization term is a weighted sum of both ridge and lasso
- Control the mix ratio $r$
	- When $r=0$
		- Elastic net is equivalent to ridge
	- When $r=1$
		- Equivalent to lasso

**Elastic Net Cost Function**

<img src="/images/Pasted image 20260202102422.png" alt="image" width="500">

- Always preferable to have a little bit of regularization

```python
from sklearn.linear_model import ElasticNet
elastic_new = ElasticNew(alpha=0.1, 11_ratio=0.5)
elatic_net.fit(X, y)
elastic_net.predict([[1.5]])
array([1.54333232])
```

## Early Stopping

- A different way to regularize iterative learning algorithms is to stop training as soon as validation error reaches a min
- With stochastic and mini-batch gradient descent, the curves are not as smooth
- Stop only after the validation error has been above the min, then role back the model parameters to the point of min
<img src="/images/Pasted image 20260202102605.png" alt="image" width="500">

```python
from copy import deepcopy
from sklearn.metrics imprt root_mean_squared_error
from sklearn.preprocessing import StandardScaler

X_train, y_train, X_valid, y_valid = [...]
preprocessing = make_peipline(PolynomialFeature(degree=90, include_bias=False),
		StandardScaler())
		
x_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.tranform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
	sgd_reg.partial_fit(X_train_prep, y_train)
	y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

- Adds the polynomial features and scales all the input features for training and validation set
- Measures the RMSE on the validation set at each epoch
- If lower than the previous RMSE, the saves a copy of model in the `best_model`

# Logistic Regression

- Some regression algorithms can be used for classification
- Logistic regression also called logit regression is used to estimate the probability that an instance belongs to a class
	- Positive and negative class

## Estimating Probabilities

- Computes a weighted sum of the input features
- Outputs the logistic of this result

**Logistic regression model estimated probability (vectorized form)**

<img src="/images/Pasted image 20260202103820.png" alt="image" width="500">

- $\alpha(\cdot)$ is the sigmoid function that outputs a number between 0 and 1

**Logistic Function**

<img src="/images/Pasted image 20260202103906.png" alt="image" width="500">

<img src="/images/Pasted image 20260202103916.png" alt="image" width="500">

**Logistic Regression model prediction using a 50% threshold probability**

<img src="/images/Pasted image 20260202104300.png" alt="image" width="500">

- The score $t$ is often called the logit
	- Logit function is the inverse of the logistic function
	- Log-odds
	- Log of the ratio between the estimated probability for the positive and negative class


## Training and Cost Function

- The objective of training is to set the parameter vector $\theta$ so that the model estimates high probabilities for positive instances (y = 1) and negative instances (y = 1)

**Cost Function of a single training instance**

<img src="/images/Pasted image 20260202104605.png" alt="image" width="500">

**Logistic regression cost function (log loss)**

<img src="/images/Pasted image 20260202104654.png" alt="image" width="500">

- Log loss can be show mathematically using Bayesian inference
- There is no known closed-form equation to compute the value of $\theta$ that minimizes this cost function
- Cost function is convex, so GD is guaranteed to find the global minimum


**Logistic cost function partial derivatives**

<img src="/images/Pasted image 20260202104851.png" alt="image" width="500">

- For each instance it computes the prediction error and multiples its by the $j$ th feature value
- Computes the average over all training instances
- Use gradient vector in batch GD

## Decision Boundaries

- Use iris dataset to illustrate logistic regression

```python
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
list(iris)
['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names',
 'filename', 'data_module']
iris.target.head(3)
0    0
1    0
2    0
Name: target, dtype: int64
iris.target_names
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

# split data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.targer_names[iris_target] == 'virginica'
X_train, X_test, y_train = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Model's estimated probabilities
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg,predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2,
	label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
plt.show()
```

<img src="/images/Pasted image 20260202105537.png" alt="image" width="500">

- The petal width of Iris virginica flowers (triangles) ranges from 1.4 to 2.5 cm, while other iris flowers (squares) have a smaller petal width, 0.1 to 1.8
- There is a decision boundary at around 1.6 cm where both probabilities are equal to 50%

```python
decision_boundary
1.6516516516516517
log_reg.predict([[1.7], [1.5]])
array([ True, False])
```

<img src="/images/Pasted image 20260202105809.png" alt="image" width="500">


## Softmax Regression

- The logistic regression model can be generalized to support classes directly, without having to train and combine binary classifiers
	- Softmax or multinomial logistic regression
- When given an instance x, the softmax regression model first computes a score $s_k(x)$ for each class k, then estimates the probability of each class by applying the softmax function (normalized exponential)

**Softmax score for class k**

<img src="/images/Pasted image 20260202110031.png" alt="image" width="500">

- Each class has its own dedicated parameter vector
- All vectors are stored as rows in a parameter matrix, $\Theta$
- Function computes the exponential of every score, then normalizes them

**Softmax Function**

<img src="/images/Pasted image 20260202110153.png" alt="image" width="500">

$K$ = number of classes
$s(x)$ = a vector containing the scores of each class
$\sigma(s(x))_k$ = estimated probability that the instance x belongs to class k

**Softmax Regression Classifier Prediction**

<img src="/images/Pasted image 20260202110317.png" alt="image" width="500">

- argmax operator returns the value of a variable that maximizes a function

- The objective is to have a model that estimates a high probability for the target class
- Minimizing the cost function, called the cross entropy, should lead to this objective because it penalizes the model when it estimates a low probability for a target class
- Cross entropy is used to measure how well a set of estimated class probabilities matches the target classes

**Cross entropy cost function**

<img src="/images/Pasted image 20260202110510.png" alt="image" width="500">

$y_k$ = target probability

- Cross entropy originates from information theory
- Measures the average number of bits you send per option
- If assumption is perfect, cross entropy will be equal to the entropy of the weather
- If wrong, cross entropy will be greater by an amount called the Kullback–Leibler (KL) divergence

**Cross entropy gradient vector for class k**

<img src="/images/Pasted image 20260202110755.png" alt="image" width="500">


```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax.fit(X_train, y_train)

softmax_reg.predict([[5, 2]])
array([2])
softmax_reg.predict_proba([[5, 2]]).round(2)
array([[0.  , 0.04, 0.96]])
```

<img src="/images/Pasted image 20260202111016.png" alt="image" width="500">

- Decision boundaries between two classes are linear
- 