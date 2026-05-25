# The Problem

# The Logistic Functions

- Logistic function

```python
def logistic(x: float) -> float:
    return 1.0 / (1 + math.exp(-x))
```

<img src="/images/Pasted image 20260429164458.png" alt="image" width="500">

- As input get large and positive, it gets closer to 1
- As inputs gets large and negative, gets closer to 0

- Derivative, $log'$

```python
def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)
```

- Log likelihood
	- Any $\beta$ that maximizes the log likelihood also maximizes the likelihood, and vice versa
- Gradient descent minimizes
- Use negative log likelihood to max the likelihood
- The overall points are independent
- The overall likelihood is the product of individual likelihoods

<img src="/images/Pasted image 20260429164704.png" alt="image" width="500">

```python
import math
from scratch.linear_algebra import Vector, dot

def _negative_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    """The negative log likelihood for one data point"""
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))
        
from typing import List

def negative_log_likelihood(xs: List[Vector],
                            ys: List[float],
                            beta: Vector) -> float:
    return sum(_negative_log_likelihood(x, y, beta)
               for x, y in zip(xs, ys))
               
# gradients
from scratch.linear_algebra import vector_sum

def _negative_log_partial_j(x: Vector, y: float, beta: Vector, j: int) -> float:
    """
    The jth partial derivative for one data point.
    Here i is the index of the data point.
    """
    return -(y - logistic(dot(x, beta))) * x[j]

def _negative_log_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    """
    The gradient for one data point.
    """
    return [_negative_log_partial_j(x, y, beta, j)
            for j in range(len(beta))]

def negative_log_gradient(xs: List[Vector],
                          ys: List[float],
                          beta: Vector) -> Vector:
    return vector_sum([_negative_log_gradient(x, y, beta)
                       for x, y in zip(xs, ys)])
```

# Applying the Model

# Goodness of Fit

# Support Vector Machines

- Hyperplane
	- Boundary that splits the parameter space into 2 half-spaces corresponding to the predicted values
- SVM
	- Best separate the classes in the training data
	- Finds the hyperplane the maximizes the distance to the nearest point in each class

<img src="/images/Pasted image 20260429165001.png" alt="image" width="500">

- Transform data into a higher-dimensional space then perform SVM
- Kernel trick
	- Rather than mapping the points to a higher dimensional space
	- Use kernel function to compute dot products in HDS to find a hyperplane

<img src="/images/Pasted image 20260429165103.png" alt="image" width="500">

# For Further Exploration

- LIBSVM