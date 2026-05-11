- Multiple independent variables

<img src="/images/Pasted image 20260429163036.png" alt="image" width="500">

- Dummy variables
	- Equal 1 for users, and 0 for users without
	- Binary variables
	- Represent categorical data

# The Model

$y_i = \alpha + \beta_1x_{i1} + ... + \beta_k x_{ik} + \epsilon_i$

- Multiple regression
	- Parameters = $\beta$

```python
from scratch.linear_algebra import dot, Vector

def predict(x: Vector, beta: Vector) -> float:
    """assumes that the first element of x is 1"""
    return dot(x, beta)
```

# Further Assumptions of the Least Squares Model

- Columns of x are linearly independent
	- Cannot write one as a weighted sum of the others
- Columns of x are all uncorrelated with the errors, $\epsilon$

# Fitting the Model

- Minimize the SSE
- Find an exact solution, through gradient descent
- Take an vector of arbitrary length as parameters

```python
from typing import List

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2

x = [1, 2, 3]
y = 30
beta = [4, 4, 4]  # so prediction = 4 + 8 + 12 = 24

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

assert sqerror_gradient(x, y, beta) == [-12, -24, -36]
```

# Interpreting the Model

- The coefficients represent all-else-being-equal estimates of the impact of each factor
- Does not show the interactions among the variables

# Goodness of Fit

- R-squared
- Standard errors of the coefficients
	- Measure how certain the estimated of each $\beta_i$ is

# Digression: The Bootstrap

- Resampling technique that helps in estimating the uncertainty of a statistical model
- Bootstrap new datasets by choosing n data points with replacement from data

# Standard Errors of Regression Coefficients

- Estimate the standard error of the regression coefficients
- Repeatedly take bootstrap samples of data and estimate beta based on the sample
- If the coefficient corresponding to one of the independent variables does not vary, then there is a high confidence that the estimate is right

# Regularization

- Regularization
	- Add to the error term a penalty that gets larger as beta gets larger
	- Minimize the combined error and penalty
- Ridge regression
	- Add a penalty proportional to the sum of the squares of the $\beta_i$
	- use in gradient descent

```python
# alpha is a *hyperparameter* controlling how harsh the penalty is.
# Sometimes it's called "lambda" but that already means something in Python.
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector,
                        y: float,
                        beta: Vector,
                        alpha: float) -> float:
    """estimate error plus ridge penalty on beta"""
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)
    
from scratch.linear_algebra import add

def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    """gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: Vector,
                           y: float,
                           beta: Vector,
                           alpha: float) -> Vector:
    """
    the gradient corresponding to the ith squared error term
    including the ridge penalty
    """
    return add(sqerror_gradient(x, y, beta),
               ridge_penalty_gradient(beta, alpha))
```

- Lasso regression

```python
def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])
```

- Ridge
	- Shrank the coefficients
- Lasso
	- Force coefficients to 0
	- Used for sparse models
	- Does not work on gradient descent

# For Further Exploration