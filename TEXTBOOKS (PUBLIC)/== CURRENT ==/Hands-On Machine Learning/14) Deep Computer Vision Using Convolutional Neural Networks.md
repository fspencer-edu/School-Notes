- Convolutional neural networks (CNNs) emerge from the study of the brain's visual cortex, and how they have been used in computer image recognition



# The Architecture of the Visual Cortex

- Visual cortex
	- Many neuron have a small local receptive field
		- React only to visual stimuli located in a limited region of the visual field
	- Neurons react only to images of horizontal lines, while others react only to lines with different orientations
	- Some neurons have larger receptive fields, and they react to more complex patterns that are combinations of the lower-level fields
	- Higher-level neurons are based on neighbouring lower-level neurons
- Neocognitron
	- Evolved to convolutional neural networks
	- LeNet-5 architecture
- New building blocks
	- Convolutional layers
	- Pooing layers

- For a huge number of parameters, it increases for every layer
- CNN uses partially connected layers and weight sharing

<img src="/images/Pasted image 20260204105856.png" alt="image" width="500">

# Convolutional Layers

- Convolutional layer
	- Neurons in the first layer are not connected to every single pixel in the input image
	- Connect to pixels in their receptive fields
- Network to concentrate on small low-level features, then assemble into larger higher-level feature

<img src="/images/Pasted image 20260204105908.png" alt="image" width="500">
- Each layer is represented in 2D, not flattened
- For a layers to have the same height and width, add zero padding
- Connect a large input layer to a smaller layer by spacing the receptive fields
	- Stride
		- Horizontal or vertical step size



<img src="/images/Pasted image 20260204105934.png" alt="image" width="500">
## Filters
- A neuron's weight can be represented as a small image the size of the receptive field
- Filters (convolution kernels)
- A layer full of neurons using the same filter outputs a feature map
	- Feature map 1
		- Ignore everything, except for vertical line
	- Feature map 2
		- Ignore everything, except for horizontal line
<img src="/images/Pasted image 20260204105951.png" alt="image" width="500">
## Stacking Multiple Feature Maps

- A convolutional layer has multiple filters and outputs one feature map per filter
- More accurately represented in 3D
- A convolutional layer simultaneously applied multiple trainable filters to its inputs, making it capable of detecting multiple features anywhere in its inputs

<img src="/images/Pasted image 20260204110003.png" alt="image" width="500">
- Input images are composed of multiple sublayers
	- One per colour channel
		- RGB
- Computing the output of a neuron in a convolutional layer

![[Pasted image 20260302113740.png]]

$z_{i, j, k}$ = output of the neuron
$x_{i', j', k'}$ = output of the neuron located in layer l-1
$w_{u, v, k', k}$ = connection weight between any neuron in feature map k of layer l

## Implementing Convolutional Layers with Keras

- Load and preprocess a couple of sample images

```python
from sklearn.datasets import load_sample_images
import tensorflow as tf

images = load_sample_images()["images"]
images = tf.keraas.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1/255)(images)

images.shape
TensorShape([2, 70, 120, 3])
```

- 4D tensor
	- 2 sample images
	- Each images is 70 x 120
	- 3 colour channels (RGB)
- Create a 2D convolutional layer and feed these images
	- Removes zero padding, subtracts 6 from image dimension
	- 32 output feature maps
		- Intensity of RGB at each location

```python
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
fmaps.shape
TensorShape([2, 64, 114, 32])
```
- 2D refers to the number of spatial dimensions
- To set feature maps to the same size, `padding="same"`

```python
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding="same")
fmaps = conv_layer(images)
fmaps.shape
TensorShape([2, 70, 120, 32])
```
- If the stride is greater than 1, then the output size will not be equal to the input size

<img src="/images/Pasted image 20260204110019.png" alt="image" width="500">

- Layer holds all the layer's weights, including the kernels and biases
- kernels are initialized randomly, while biases are set to zero

7 x 7 kernel size
3 input channels
32 different filters


```python
kernels, biases = conv_layer.get_weights()
kernel.shape
(7, 7, 3, 32)
biases.shape
(32,)
```
- `kernels` => `[kernel_height, kernel_width, input_channels, output_channel)`
- `biases` => `[output_channels]`

- All the neurons in the output feature maps share the same weights
- Feed images of any size to list layer, as long as they are at least as large as the kernels
- A convolutional layer performs a linear operations, so stacked multiple convolutional layers without any activation functions should be equivalent to a single convolutional layer

- Layer hyperparamers
	- `filter`
	- `kernel_size`
	- `padding`
	- `strides`
	- `activation`
	- `kernel_initializer`

## Memory Requirements

- CNNs require a huge amount of RAM
- Reverse pass of backpropagation requires all the intermediate values computer during the forward pass
- During inference
	- Only need as much RAN as required by two consecutive layers
- During training
	- The forward pass needs to be preserved for the reverse pass
	- RAM is the total amount of RAM for all layers

- If training crashes from an out-of-memory error, reduce the mini-batch size
- Try reducing dimensionality using stride, removing layers, using 16-bit floats, or distributing the CNN across multiple devices
# Pooling Layers

- The goal of pooling layers is to subsample (shrink) the input image in order to reduce the computational load, memory usage, and number of parameters
- Each neuron in a pooling layer is connected to the outputs of a limited number of neurons in the previous layer
	- Size, stride, padding type
- A pooling neuron has no weights
	- Aggregates the inputs using a max or mean function

- Max pooling layer
	- 2 x 2 pooling kernel, with a stride of 2 and no padding
	- Max value of the pooling kernel is 5, and is propagated to the next layer
	- Compresses the images

<img src="/images/Pasted image 20260204110030.png" alt="image" width="500">

- Introduces invariance to small translations
	- Bright pixels have a lower value than dark pixels
	- All the images are same image, but shifted by one and two pixels to the right
	- Max pooling layer for images A and B are the same
	- Max pooling produces a small amount of rotational invariance and scale invariance
- Goal is equivariance
	- A small change to the inputs should lead to a corresponding small change in the output

<img src="/images/Pasted image 20260204110039.png" alt="image" width="500">

# Implementing Pooling Layers with Keras

- Pooling layers
	- `MaxPool2D`
	- `AvgPool2D`

```python
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)
```

- Max and average pooling can be performed along the depth dimension instead of spatial dimension
	- CNN can lear to be invariance to various feature

```python
class DepthPool(tf.keras.layers.Layer):
	def __init__(self, pool_size=2, **kwargs):
		super().__init__(**kwargs)
		self.pool_size = pool_size
		
	def call(self, inputs):
		shape = tf.shape(inputs)
		groups = shape[-1]
		new_shape = tf.concat([shape[:-1], [groups, self.pool_size]], axis=0)
		return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)
```

- Reshaped its inputs to split the channels into groups of the pool size, then computes the max of each group

<img src="/images/Pasted image 20260204110051.png" alt="image" width="500">

- Global average pooling
	- Compute the mean of each entire feature map
	- Averaging pooling layer using a pooling kernel with the same spatial dimensions as the inputs
	- Used before the output layer

```python
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()

global_avg_pool = tf.keras.layers.Lambda(
	lambda X: tf.reduce_mean(X, axis=[1, 2])
)

global_avg_pools(images)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[0.64338624, 0.5971759 , 0.5824972 ],
       [0.76306933, 0.26011038, 0.10849128]], dtype=float32)>
```

- Get the mean intensity of red, greed, and blue for each image

# CNN Architectures

- CNN architecture
	- Stacks a few convolutional layers
		- Each on followed by a ReLU layer
	- Pooling layer
	- Another few convolutional layers (+ReLU)
	- Another pooling layer
- As the image gets smaller, it obtained more feature maps
- At the top, a regular feed-forward neural network is added
- Final layer outputs the prediction (a softmax layer)

- A common mistake is to used convolutional kernels that are too large
	- Large kernels are used in the beginning layers
<img src="/images/Pasted image 20260204110103.png" alt="image" width="500">
```python
from functools import partial

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
				activation="relu", kernel_initializer="he_normal")
				
model = tf.keras.Sequential([
	DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
	tf.keras.layers.MaxPool2D(),
	DefaultConv2D(filters=128),
	DefaultConv2D(filters=128),
	tf.keras.layers.MaxPool2D(),
	DefaultConv2D(filters=256),
	DefaultConv2D(filters=256),
	tf.keras.layers.MaxPool2D(),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(units=128, activation="relu",
					kernel_initializer="he_normal"),
	tf.keras.layers.Droppout(0.5),
	tf.keras.layers.Dense(units=64, activation="relu",
					kernel_initializer="he_normal"),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(units=10, activation="softmax")
])
```

- Set a default convolutional layer with `functools.partial()`
- Create the `Sequential` model
	- 64 large filters (7 x 7)
	- Default stride of 1
	- Single colour channel
- Add a max pooling layer that uses the default pool size of 2
- Repeat the same structure twice
	- Two convolutional layers followed by a max pooling layer
- The number of tilers double as we climb up the CNN
	- Number of low-level feature are low
- The output layer has 10 units, and uses the softmax activation function
	- Flatten the inputs before the first dense layer, since is uses 1D array for features for each instance
	- Add 2 dropout layers, to reduce overfitting

- The compiled model should reach over 92% accuracy on test set
- Variants of the architecture have been developed

## LaNet-5

- Used for handwritten digit recognition (MNIST)

![[Pasted image 20260302123012.png]]

## AlexNet

![[Pasted image 20260302123053.png]]

- To reduce overfitting, the 2 regularization techniques are used
	- Dropout with a 50% dropout rate during training to the outputs of layers F9 and F10
	- Data augmentation by randomly shifted the training images by various offsets
		- Flipping them horizontally
		- Change the light conditions

- Data augmentation increases the size of training, increased variants of each training instance
- Adding white noise does not help
- 

<img src="/images/Pasted image 20260204110117.png" alt="image" width="500">


## GoogLeNet

<img src="/images/Pasted image 20260204110127.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110140.png" alt="image" width="500">

## VGGNet
## ResNet


<img src="/images/Pasted image 20260204110150.png" alt="image" width="500">
## Xception

<img src="/images/Pasted image 20260204110225.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110236.png" alt="image" width="500">


<img src="/images/Pasted image 20260204110248.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110257.png" alt="image" width="500">


## SENet


<img src="/images/Pasted image 20260204110307.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110317.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110327.png" alt="image" width="500">

## Other Noteworthy Architectures

## Choosing the Right CNN Architecture


# Implementing a ResNet-34 CNN Using Keras

# Using Pretrained Models from Keras

# Pretrained Models for Transfer Learning

# Classification and Localization

<img src="/images/Pasted image 20260204110343.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110354.png" alt="image" width="500">
# Object Detection

## Fully Convolutional Networks
## Your Only Look Once

<img src="/images/Pasted image 20260204110405.png" alt="image" width="500">

# Object Tracking

# Semantic Segmentation