
- Lower level TensorFlow API
	- Custom loss functions
	- Custom metrics
	- Layers
	- Models
	- Initializers
	- Regularizers
	- Weight constraints

# A Quick Tour of TensorFlow

**TensorFlow**
- GPU support
- Supports distributed computing
- Just-in-time (JIT) compiler
- Computation graphs
- Reverse-mode autodiff


- TF operations is implemented using highly efficient C++ code
- Many operations have implementations called kernels
- Each kernel is dedicated to a specific device type
	- CPU, GPU, TPU
- GPUs can increase speed computations by splitting them into smaller chunks and running in parallel across threads
- TPU are even faster
	- ASIC chips for deep learning operations
<img src="/images/Pasted image 20260204105607.png" alt="image" width="500">

- Most code will use high-level APIs
- TF execution engine will take care of running the operations

<img src="/images/Pasted image 20260204105615.png" alt="image" width="500">

- TensorBoard
	- Visualization
- TensorFlow Extended (TFX)
	- Productionize TF projects
	- Data validation
	- Preprocessing
	- Model analysis
	- Serving
- TensorFlow Hub
	- Easy way to download and reuse pretrained NN


# Using TensorFlow like NumPy

- TF API uses tensors
- A tensor is similar to a NumPy `ndarray`
- Multidimensional array, that also hold a scalar

## Tensors and Operations

```python
# tensor with 3 rows and 3 col
import tensorflow as tf
t = tf.constant([[1., 2., 3., 4., 5., 6.]])
t
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>
       
t.shape
TensorShape([2, 3])
t.dtype
tf.float32

# indexing
t[:, 1:]
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[2., 3.],
       [5., 6.]], dtype=float32)>
>>> t[..., 1, tf.newaxis]
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[2.],
       [5.]], dtype=float32)>
       
# tensor operations
>>> t + 10
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[11., 12., 13.],
       [14., 15., 16.]], dtype=float32)>
>>> tf.square(t)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[ 1.,  4.,  9.],
       [16., 25., 36.]], dtype=float32)>
>>> t @ tf.transpose(t)
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[14., 32.],
       [32., 77.]], dtype=float32)>
```

- Many functions and class have aliases

```python
# scalar values
tf.constant(42)
<tf.Tensor: shape=(), dtype=int32, numpy=42>
```

- Keras API has a lower level API, `tf.keras.backend`
- In TF, a new tensor is creates with its own copy of the transposed data

## Tensors and NumPy

- Create a NumPy array, and vice versa with TF

```python
>>> import numpy as np
>>> a = np.array([2., 4., 5.])
>>> tf.constant(a)
<tf.Tensor: id=111, shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>
>>> t.numpy()  # or np.array(t)
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
>>> tf.square(a)
<tf.Tensor: id=116, shape=(3,), dtype=float64, numpy=array([4., 16., 25.])>
>>> np.square(t)
array([[ 1.,  4.,  9.],
       [16., 25., 36.]], dtype=float32)
```


- NumPy uses 64-bit precision, TF uses 32-bit

## Types Conversions

- Type conversion can hurt performance
- TF does not perform any time conversions automatically
- Raises an exception if execute an operation on tensors with incompatible types
- 

```python
>>> tf.constant(2.) + tf.constant(40)
[...] InvalidArgumentError: [...] expected to be a float tensor [...]
>>> tf.constant(2.) + tf.constant(40., dtype=tf.float64)
[...] InvalidArgumentError: [...] expected to be a float tensor [...]

# cast
>>> t2 = tf.constant(40., dtype=tf.float64)
>>> tf.constant(2.0) + tf.cast(t2, tf.float32)
<tf.Tensor: id=136, shape=(), dtype=float32, numpy=42.0>
```



## Variables
## Other Data Structures

# Customizing Models and Training Algorithms

## Custom Loss Functions
## Saving and Loading Models That Contain Custom Components
## Custom Activation Functions, Initializers, Regularizers, and Constraints
## Custom Metrics
## Custom Models

<img src="/images/Pasted image 20260204105646.png" alt="image" width="500">

## Losses and Metrics Based on Model Internals
## Computing Gradients Using Autodiff
## Custom Training Loops

# TensorFlow Functions and Graphs

<img src="/images/Pasted image 20260204105705.png" alt="image" width="500">

## AutoGraph and Tracing
## TF Function Rules
