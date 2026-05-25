
- `correlation`
	- Measure the strength of the linear relationship between two variables

# The Model

$y_i = \beta x_i + \alpha + \epsilon_i$

$\alpha, \beta$ = constants

```python
def predict(alpha: float, beta: float, x_i: float) -> float:
	return beta * x_i + alpha
	
def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha
    when the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i
```

- Predict the values of the constants be computing the error for each pair
- Sum of squared errors (SSE)

```python
from scratch.linear_algebra import Vector

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))
```

- Least squares
	- Makes the SSE as small of possible

```python
from typing import Tuple
from scratch.linear_algebra import Vector
from scratch.statistics import correlation, standard_deviation, mean

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
```

<img src="/images/Pasted image 20260429162540.png" alt="image" width="500">

- Coefficient of determination (R-squared)
	- Measure the fraction of the total variation in the dependent variable that is captured by the model

```python
from scratch.statistics import de_mean

def total_sum_of_squares(y: Vector) -> float:
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330
```

- The higher the R-value the better

# Using Gradient Descent

```python
import random
import tqdm
from scratch.gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]  # choose random value to start

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess

        # Partial derivative of loss with respect to alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i in zip(num_friends_good,
                                         daily_minutes_good))

        # Partial derivative of loss with respect to beta
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i in zip(num_friends_good,
                                         daily_minutes_good))

        # Compute loss to stick in the tqdm description
        loss = sum_of_sqerrors(alpha, beta,
                               num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.3f}")

        # Finally, update the guess
        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

# We should get pretty much the same results:
alpha, beta = guess
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905
```

# Maximum Likelihood Estimation

- Maximum likelihood estimation (MLE)

$p(v_1, ..., v_n | \theta)$

$L(\theta | v_1, ..., v_n)$

- The parameter, $\theta$, can be found as the likelihood of $\theta$ given the sample

- The most likely value is one that maximizes the likelihood function
	- Makes the observed data the most probable
- Regression errors are normally distributed with mean 0 and a standard deviation, $\sigma$

<img src="/images/Pasted image 20260429163013.png" alt="image" width="500">

# For Further Exploration
