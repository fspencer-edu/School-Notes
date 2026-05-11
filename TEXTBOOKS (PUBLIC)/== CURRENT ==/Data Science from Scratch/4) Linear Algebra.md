
- Vector spaces

# Vectors

 - Vector
	 - Objects that can be added together to form new vectors
	 - Multiple by scalars
	- Points in a finite-dimensional space

```python
from typing import List

Vector = List[float]

height_weight_age = [70, 170, 40]

grades = [95, 80, 75, 62]
```

- Vectors add component wise
- `zip` vectors using list comprehension

```python
def add(v: Vector, w: Vector) -> Vector:
	"""Subtracts corresponding elements"""
	assert len(v) == len(w), "vecs must be same length"
	
	return [v_i + w_i for v_i, w_i in zip(v, w)]
	
assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
```

- Dot product
	- Sum of component wise products

```python
def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3

import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function

assert magnitude([3, 4]) == 5
```

# Matrices

- Matrix
	- 2D collection of numbers
	- Lists of lists
- `A[i][j]`
	- Zero-indexed

```python
Matrix = List[List[float]]

A = [[1, 2, 3],  # A has 2 rows and 3 columns
     [4, 5, 6]]

B = [[1, 2],     # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]
```

- Identity matrix
	- 1s on the diagonal and 0s elsewhere
- $n \times k$ matrix
	- Linear function that maps k-dimensional vectors to n-dimensional vectors
- Binary relationships
- Edge of a network as a collection or pairs

# For Further Exploration

