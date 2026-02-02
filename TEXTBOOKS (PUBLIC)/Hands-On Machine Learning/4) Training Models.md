
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

![[Pasted image 20260202083046.png]]

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

![[Pasted image 20260202083441.png]]

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

![[Pasted image 20260202084014.png]]

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

![[Pasted image 20260202085125.png]]

- Size of step is determined by learning rate hyperparameter
- If the random initialization starts on the left, then it will converge to a local minimum, not the global minimum

![[Pasted image 20260202085244.png]]

- MSE cost function for a linear regression model happens to be a convex function
- MSE function is convex, therefore no local minima, just one global minimum
- Continuous function with a slope the never change abruptly
- Gradient descent is guaranteed to approach arbitrarily close the global minimum
- Cost function can be elongated depending on feature scales

![[Pasted image 20260202085624.png]]

- Training a model means searching for a combination of model parameters that minimizes a cost function
- More parameters a mode has, the mode dimensions the space has, and the harder the search is

## Batch Gradient Descent

- Calculates how much the cost function will change if you change $\theta_j$ a little
	- Partial derivative

**Partial Derivatives of the Cost Function**

![[Pasted image 20260202092256.png]]

- The gradient vector, $\Delta_{\theta}MSE(\theta)$, contains all partial derivatives of the cost function

**Gradient Vector of the Cost Function**

![[Pasted image 20260202092403.png]]

- Batch gradient descent uses the whole batch of training data at every step
- Slow
- Scales will with the number of features
- Faster on a linear regression model with many features compared to normal equation or SVD decomposition

 - Subtract $\Delta_{\theta}MSE(\theta)$ from $\theta$ to get downhill vector

**Gradient descent step**

![[Pasted image 20260202092633.png]]

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

![[Pasted image 20260202092906.png]]

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

![[Pasted image 20260202093510.png]]

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

![[Pasted image 20260202094008.png]]

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

![[Pasted image 20260202094649.png]]

![[Pasted image 20260202094720.png]]

# Polynomial Regression

- Add powers of each feature as new features, then train a linear model on this ex

# Learning Curves

# Regularized Linear Models

## Ridge Regression
## Lasso Regression
## Elastic Net Regression
## Early Stopping


# Logistic Regression

## Estimating Probabilities
## Training and Cost Function
## Decision Boundaries
## Softmax Regression