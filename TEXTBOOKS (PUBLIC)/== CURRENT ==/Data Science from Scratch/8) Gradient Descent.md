
- Best model
	- Minimize error of its predictions
	- Maximize likelihood of the data
	- Optimization problem

# The Idea Behind Gradient Descent

```python
from scratch.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)
```
- Gradient
	- Partial derivaties
	- Gives the input direction in which the function most quickly increases

<img src="/images/Pasted image 20260428162235.png" alt="image" width="500">

# Estimating the Gradient

- If `f` is a function of one variable, its derivative at a point `x` measure how `f(x)` changes when there is a small change to `x`
- Limit of the difference of quotients

```python
from typing import Callable

def difference_quotient(f: Callable[[float], float],
		x: float,
		h: float) -> float:
	return (f(x+h) - f(x)) / h
```

- The derivative is the slope of the tangent line at $(x, f(x))$
- Difference quotient is the slop of the not-quite tangent line that runs through $(x + h,  f(x +h))$
- As $h$ gets smaller, the tangent line gets closer to the the real tangent line

<img src="/images/Pasted image 20260428162529.png" alt="image" width="500">

```python
def square(x: float) -> float:
	return x * x
	
def derivative(x: float) -> float:
	return 2 * x
```

- When `f` is a function of many variable, it has multiple partial derivatives
- Calculate its ith partial derivative by treating it as a function of ith variables

```python
def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h
    
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]
```

# Using the Gradient

```python
import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]
    
# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)    # compute the gradient at v
    v = gradient_step(v, grad, -0.01)    # take a negative gradient step
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001    # v should be close to 0
```

# Choosing the Right Step Size

- Using a fixed step size
- Gradually shrinking the step side
- At each step, choose the step size the minimizes the value of the objective function

# Using Gradient Descent to Fit Models

- Use gradient descent to fit parameterized models to data
- Loss function
	- Measure how well the model fits the data

```python
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual).
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad
```

- Gradient descent “decides” each step by looking at the slope of the loss function and moving in the direction that decreases it the fastest
- Start with a random value for `theta`
- Compute the mean of the gradients
- Adjust `theta` in that direction
- Repeat

```python
from scratch.linear_algebra import vector_mean

# Start with random values for slope and intercept
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

for epoch in range(5000):
    # Compute the mean of the gradients
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # Take a step in that direction
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5
```
# Minibatch and Stochastic Gradient Descent

- The previous approach requires to evaluation of gradients on the entire dataset before taking a gradient step and updating parameters
- Minibatch gradient descent
	- Compute the gradient and take the step based on a minibatch from a larger dataset

```python
from typing import TypeVar, List, Iterator

T = TypeVar('T')  # this allows us to type "generic" functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
```

- `TypeVar(T)`
	- Generic function

```python
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
```

- Stochastic gradient descent
	- Take gradient steps based on one training example at a time
	- Find optimal parameters in a small number of epochs

```python
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
```

# For Further Exploration