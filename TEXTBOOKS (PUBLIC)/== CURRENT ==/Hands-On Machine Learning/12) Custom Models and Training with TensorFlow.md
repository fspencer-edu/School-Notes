
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

- `tf.Tensor` values are immutable
- Cannot use regular tensors to implement weights in a NN
- Other parameters need to change over time

```python
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>
       
v.assign(2 * v)           # v now equals [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42)        # v now equals [[2., 42., 6.], [8., 10., 12.]]
v[:, 2].assign([0., 1.])  # v now equals [[2., 42., 0.], [8., 10., 1.]]
v.scatter_nd_update(      # v now equals [[100., 42., 0.], [8., 10., 200.]]
    indices=[[0, 0], [1, 2]], updates=[100., 200.])
    
# direct assignment will not work
>>> v[1] = [7., 8., 9.]
[...] TypeError: 'ResourceVariable' object does not support item assignment

```

- `tf.Variable` acts like a tensor, but can be modified in place using `assign()`
- Keras provides an `add_weight()` method that will modify TF
- Model parameters are updated directly by the optimizers

## Other Data Structures

- TF support other data structures

- Sparse tensors, `tf.SparseTensor`
	- Represents tensors containing mostly zeros
- Tensor arrays, `tf.TensorArray`
	- Lists of all tensors
	- Fixed length by default
	- Same shape and data type
- Ragged tensors, `tf.RaggedTensor`
	- Represents a list of tensors, of same rank and data type
- String tensors
	- `tf.string`
	- Represent byte strings
	- Encoded to UTF-8
- Sets
	- Regular tensors
- Queues
	- Store tensors across multiple steps
	- FIFO
	- Priority queues
	- Shuffle
	- Batch
	- Padding


# Customizing Models and Training Algorithms

## Custom Loss Functions

- Use the Huber loss instead of MSE
- Huber loss is available in Keras
- Create a function that takes the lebels and the mode's predictions as arguments
- Use TF operations to compute a tensor containing all the losses

```python
def huber_fn(y_true, y_pred):
	error = y_true - y_pred
	is_small_error = tf.abs(error) < 1
	squared_loss = tf.sqaure(error) / 2
	linear_loss = tf.abs(error) -0.5
	return tf.where(is_small_error, squared_loss, linear_loss)
	
mode.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])
```

## Saving and Loading Models That Contain Custom Components

- Saving a model containing a custom loss function work
- But when loading it, provide a dictionary that maps the function names to actual functions

```python
model = tf.keras.models.load_model("my_model_with_a_custom_loss",
				custom_object={"huber_fn": huber_fn})
```

- Function that creates a configured loss function

```python
def create_huber(threshold=1.0):
	def huber_fn(y_true, y_pred):
		error = y_true - y_pred
		is_small_error = tf.abs(error) < threshold
		squared_loss = tf.squared(error) /2
		linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
		return tf.where(is_small_error, squared_loss, linear_loss)
	return huber_fn
	
model.compile(loss=create_huber(2.0), optimizer="nadam")
```

- When the model is saved, `threshold` will not be saved

```python
nodel = tf.keras.model.load_model(
	"my_model_with_a_custom_loss_threshold_2",
	custom_objects={"huber_fn":create_huber(2.0)}
)

# subclass with loss class
class HuberLoss(tf.keras.losses.Loss):
	def __init__(self, threshold=1.0, **kwargs):
		self.threshold = threshold
		super().__init__(**kwards)
		
	def call(self, y_true, y_pred):
		error = y_true - y_pred
		is_small_error = tf.abs(error) < self.threshold
		squared_loss = tf.abs(error) < self.threshold
		linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
		return.where(is_small_error, squared_loss, linear)loss

	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "threshold": self.threshold}
```

- Constructor `**kwargs` accepts and passes to parent
- Loss will by the sum of instance losses, weighted by the sample weights
- `call()` takes the labels and predictions, computes all the instance losses, and returns them
- `get_config()` returns a dictionary mapping each hyperparameter name to its value

```python
model.compile(loss=HuberLoss(2.), optimizer="nadam")

# threshold will be saved with model
model = tf.keras.models.load_model("my_model_with_a_custom_loss_class",
				custom_objects={"HuberLoss": HuberLoss})
```

## Custom Activation Functions, Initializers, Regularizers, and Constraints

```python
def my_softplus(z):
	return tf.math.log(1.0 + tf.exp(z))
	
def my_glorot_initializer(shape, dtype=tf.float32):
	stddev = tf.sqrt(2. / (shape[0] + shape[1]))
	return tf.random.normal(shape, stddev=stddev, dtype=dtype)
	
def my_11_regularizer(weights):
	return tf.reduce_sum(tf.abs(0.01 * weights))
	
def my_positive_weights(weights):
	return tf.where(wegithts < 0., tf.zeros_like(weights), weights)
```

- Arguments depend on the type of custom function
- Custom functions are used normally

```python
layer = tf.keras.layers.Dense(1, activation=my_softplus,
				kernel_initializer=my_glorot_initializer,
				kernel_regularizer=my_11_regularizer,
				kernel_constraint_my_positive_weights)
```

- Use subclass to save hyperparameters with model

```python
class MyL1Regularizer(tf.keras.regularizers.Regularizer):
	def __init__(self, factor):
		self.factor = factor
		
	def __call__(self, weights):
		return tf.reduce_sum(tf.abs(self.factor * weights))
		
	def get_config(self):
		return {"factor": self.factor}
```

## Custom Metrics

- Losses and metrics are conceptually not the same
- Losses
	- Used by GD to train a model, differentiable
	- Gradients should not be zero
- Metrics
	- Used to evaluate a model
	- More easily interpretable
	- Non-differentiable or have zero gradients
- In most cases, defining a custom metric function is the same as a custom loss function

```python
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
```

- For each batch training, Keras will compute this metric and keep track of its mean since the beginning of the epoch
- Keep track of the the true positives and false positives to compute precision

```python
precision = tf.keras.metrics.Precision()
precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
<tf.Tensor: shape=(), dtype=float32, numpy=0.8>
precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
<tf.Tensor: shape=(), dtype=float32, numpy=0.5>
```

- Streaming metric (stateful metric)
	- Gradually updates precision
- `result()` gets the current value of the metric


```python
precision.result()
<tf.Tensor: shape=(), dtype=float32, numpy=0.5>
precision.variables
[<tf.Variable 'true_positives:0' [...], numpy=array([4.], dtype=float32)>,
 <tf.Variable 'false_positives:0' [...], numpy=array([4.], dtype=float32)>]
precision.reset_states() # both variables get reset to 0.0
```

- Create a subclass of metrics for custom streaming metrics

```python
Class HuberMetric(tf.keras.metrics.Metric):
	def __init__(self, threshold=1.0, **kwargs):
		super().__init__(**kwargs)
		self.threshold = threshold
		self.huber_fn = create_huber(threshold)
		self.total = self.add_weight("total", initializer="zeros")
		self.count = self.add_weight("count", initializer="zeros")
		
	def update_state(self, y_true, y_pred, sample_weight=None):
		sample_netrics = self.huber_fn(y_rue, y_pred)
		self.total.assign_add(tf.reduce_sum(sample_metrics))
		self.count(assign_add(tf.cast(tf.size(y_true), tf.float32)))
		
	def result(self):
		return self.total / self.count
		
	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "threshold": self.threshold}
```

- Constructor ass weights to create the variables needed to keep track of metric's state over multiple matches
- Update called then you use an instance
- Result computes and returns the final result

- Keras takes care of variable persistence
- Keras automatically call for each batch, and keep track of the mean during each epoch


## Custom Layers

- To create a custom layer with no weights, wrap it in `tf.keras.layers.lambda`

```python
exponential_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))
```
- To build a custom stateful layer, create a subclass of `tf.keras.layers.Layer`

```python
class MyDense(tf.keras.layers.Layer):
	def __init__(self, units, activation=None, **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.activation = tf.keras.activations.get(activation)
		
	def build(self, batch_input_shape):
		self.kernel = self.add_weight(
			name="kerne", shape=[batch_input_shape[-1], self.units],
			initializer="glorot_normal")
		self.bias = self.add_weight(
			name="bias", shape=[self.units], initializer="zeros")
			
	def call(self, X):
		return self.activation(X @ self.kernel + self.bias)
		
	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "units":self.units,
				"activation":tf.keras.activations.serialize(self.activation)}
```

- Constructors takes all the hyperparameters as arguments
- `build()` creates the layer's variables by calling `add_weight()`
- `call()` method perform the desired operations
	- Compute matrix multiplication
	- Add bias vector
	- Apply activation function
- `get_config()` returns values

- Toy layer takes two inputs and return 3 outputs

```python
class MyMultiLayer(tf.keras.layers.Layer):
	def call(self, X):
		X1, X2 = X
		return X1 + X2, X1 * X2, X1 / X2
		
# create a layer that add Gaussian noise during training, not during testing
class MyGaussianNoise(tf.keras.layers.Layer):
	def __init-_(self, stddev, **kwarfs):
		super()__init__(**kwargs)
		self.stddev = stddev
		
	def call(self, X, training=False):
		if training:
			noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
			return X + noise
		else:
			return X
```

## Custom Models

- To create a custom model, subclass `tf.keras.Model`, create layers and variables int he constructor, and implement the `call()`
- Inputs go through a first dense layer, then residual block (composed of two dense layers and an addition operation), then through the same residual block 3 more times, then second residual block, and a final dense output layer
- Create model with loops and skip connections
- Create a `ResidualBlock` layer

<img src="/images/Pasted image 20260204105646.png" alt="image" width="500">

```python
class ResidualBlock(tf.keras.layers.Layer):
	def __init__(self, n_layers, n_neurons, **kwargs):
		super().__init__(**kwargs)
		self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu",
							kernal_initializer="he_normal"),
					for _ in range(n_layers)]
					
	def call(self, inputs):
		Z = inputs
		for layer in self.hidden:
			Z = layer(Z)
		return inputs + Z
```

- `hiddn` attribute contain trackable objects, so their variables are automatically addded to layer's list of variables

```python
class ResidualRegressor(tf.keras.Model):
	def __init__(self, output_dim, **kwargs):
		super().__init__(**kwargs)
		self.hidden1 = tf.keras.layers.Dense(30, activation="relu",
							kernel_initializer="he_normal")
		self.block1 = ResidualBlock(2, 30)
		self.block2 = ResidualBlock(2, 30)
		self.out = tf.keras.layers.Dense(output_dim)
		
	def call(self, inputs):
		Z = self.hidden1(inputs)
		for _ in range(1 + 3)L
			Z = self.block1(Z)
		Z = self.block2(Z)
		return self.out(Z)
```

- Create the layers in the constructor and use them in the `call()` method
- `Model` has extra functionalities compared to `Layer` class
- Build almost any model
	- Sequential API
	- Functional API
	- Subclassing API

## Losses and Metrics Based on Model Internals

- Custom losses and metrics are based on the labels and predictions
- To define a custom loss based on model internals, compute it based on any part of the model
- Pass the result to `add_loss()`
- Build a custom regression MLP model composed of a stack of 5 hidden layers, plus an output layer
- Loss associated with output is called the reconstruction loss
	- Mean squared different between reconstruction and the inputs
- Loss improves generalization (regularization loss)
- Add custom metrics using `add_metric()`

```python
class ReconstructingRegressor(tf.keras.Model):
	def __init__(self, output_dim, **kwargs):
		super()__init__(**kwargs)
		self.hidden = [tf.keras.layers.Dense(30, activation="relu",
					kernel _initializer="he_normal")
					for _ in range(5)]
					
		self.out = tf.keras.layers.Dense(output_dim)
		self.reconstruction_mean = tf.keras.metrics.Mean(
			name="reconstruction_error")
			
		def build(self, batch_input_shape):
			n_inputs = batch_input_shape[-1]
			self.recontruct = tf.keras.layers.Dense(n_inputs)
			
		def call(self, inputs, training=False):
			Z = inputs
			for layer in self.hidden:
				Z = layer(Z)
			reconstruction = self.reconstruct(Z)
			recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
			self.add_loss(0.05 * recon_loss)
			if training:
				result = self.reconstruction_mean(recon_loss)
				self.add_netric(result)
			return self.out(Z)
```

- Constructor creates the DNN with 5 dense hidden layers and one dense output layer
- Create a `Mean` streaming metric to keep track of the reconstruction error during training
- `build()` creates an extra dense layer that will be used to reconstruct the inputs of the model
- `call()` method processes the inputs through all 5 hidden layers, passes the result through reconstruction layer
	- Computes the reconstruction loss, and add it to the model's list of losses
	- Scale down the reconstruction loss my multiplying it by 0.05
- During training, the `call()` method updates the reconstruction metric and adds it to the model
- `call()` method passes the output of the hidden layers to the output layer and returns its output

```python
Epoch 1/5
363/363 [========] - 1s 820us/step - loss: 0.7640 - reconstruction_error: 1.2728
Epoch 2/5
363/363 [========] - 0s 809us/step - loss: 0.4584 - reconstruction_error: 0.6340
[...]
```

- Total loss and reconstruction loss will go down
- For some architectures such as GAN, customize the training loop

## Computing Gradients Using Autodiff

```python
def f(w1, w2):
	return 3 * w2 ** 2 + 2 ** w1 * w2
```

- Analytically find the partial derivative of this function
- Compute an approximation of each partial derivative by measuring how much the function's output changes when is is tweaks

```python
w1, w2 = 5, 3
eps = 1e-6
(f(w1 - eps, w2) - f(w1, w2)) / eps
36.000003007075065
(f(w1, w2 + eps) - f(w1, w2)) / eps
10.000000003174137

# reverse-mode autodiff
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
	z = f(w1, w2)
	
gradients = tape.gradient(z, [w1, w2])
```

- `tf.GradientTape`
	- Automatically record every operation that involves a variable, and finally we ask this tape to compute the gradients of the result `z` with regards to both variables

```python
>>> gradients
[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
```

- To save memory only put the strict min inside the gradient tape block
- Tape is automatically erased immediately after `gradiet()` is called

```python
with tf.GradientTape() as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # returns tensor 36.0
dz_dw2 = tape.gradient(z, w2)  # raises a RuntimeError!

# make tape persistent and delete it each time to free resources
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # returns tensor 36.0
dz_dw2 = tape.gradient(z, w2)  # returns tensor 10.0, works fine now!
del tape

# tape will only track operations involving variables
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])  # returns [None, None

# force the tape to watch any tensors, to record every operation
with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])  # returns [tensor 36., tensor 10.]
```

- 

## Custom Training Loops

# TensorFlow Functions and Graphs

<img src="/images/Pasted image 20260204105705.png" alt="image" width="500">

## AutoGraph and Tracing
## TF Function Rules
