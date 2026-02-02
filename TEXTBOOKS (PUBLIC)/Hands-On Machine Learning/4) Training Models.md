
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

## Batch Gradient Descent
## Stochastic Gradient Descent
## Mini-Batch Gradient Descent


# Polynomial Regression

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