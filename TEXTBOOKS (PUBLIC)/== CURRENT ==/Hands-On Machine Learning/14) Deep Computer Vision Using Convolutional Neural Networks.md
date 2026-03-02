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

<img src="/images/Pasted image 20260302113740.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260302123012.png" alt="image" width="500">

## AlexNet

<img src="/images/Pasted image 20260302123053.png" alt="image" width="500">

- To reduce overfitting, the 2 regularization techniques are used
	- Dropout with a 50% dropout rate during training to the outputs of layers F9 and F10
	- Data augmentation by randomly shifted the training images by various offsets
		- Flipping them horizontally
		- Change the light conditions

- Data augmentation increases the size of training, increased variants of each training instance
- Adding white noise does not help
- Synthetic minority oversampling technique (SMOTE)

<img src="/images/Pasted image 20260204110117.png" alt="image" width="500">

- Local response normalization (LRN)
	- AlexNet also uses a competitive normalization step immediately after the ReLU step of layers C1 and C2
	- The most strongly activate neurons inhibit other neurons located at the same position in the neighbouring feature maps

- Local response normalization (LRN)

<img src="/images/Pasted image 20260302123506.png" alt="image" width="500">

$b_i$ = normalized output of the neuron located in feature map i
$a_i$ = activation of that neuron after the ReLU step, but before normalization
$k, \alpha, \beta, r$ = hyperparameters (bias, depth radius)
$f_n$ = number of feature maps

- A $r=2$ shows that a neuron has a strong activation, and will inhibit the activation of the neurons located in the feature maps immediately above of below

- A variance of AlexNet is ZF net
	- Tweaked hyperparameters
		- Number of feature maps
		- Kernel size
		- Stride

## GoogLeNet

- Uses sub-networks called inception modules
	- Use parameters more efficiently than previous architectures
- 3 x 3 + 1(S)
	- 3 x 3 kernel, stride 1, and same padding
- The input signal is fed to the 4 different layers in parallel
- All convolutional layers use the ReLU activation function
- Top convolutional layer used different kernel sizes, allowing them to capture patterns at different scaled
- Concatenate all the output along the depth dimension in the final concatenation layer

<img src="/images/Pasted image 20260204110127.png" alt="image" width="500">
- 1 x 1 kernels
	- Capture patterns along the depth dimension (across channels)
	- Output fewer feature maps than their input
		- Serve as bottleneck layers, to reduce dimensionality
	- Each pair of convolutional layers, act like a single powerful convolutional layer, to capture more complex patterns


- The number of feature maps output by each convolutional layer and each pooling layer is shown before the kernel size
- 9 inception modules

- First 2 layers divide the image's height and width by 4
- Local response normalization layer ensures that the previous layers learn a wide variety of features
- Two convolutional layers follow
	- Bottleneck layer
- Local response normalization layer ensures that the previous layers capture a wide variety of patterns
- Max pooling layer reduces the image height and width by 2
- CNNs backbone
	- A tall stack of 9 interception modules, interleaves with a couple of max pooling layers to reduce dimensionality and speed up the net
- Global average pooling layer outputs the mean of each feature map
	- Drops any remaining spatial information
- Dropout for regularization
- Softmax activation function to output estimated class probabilities

<img src="/images/Pasted image 20260204110140.png" alt="image" width="500">

## VGGNet

- 2 or 3 convolutional layers and a pooling layer, repeated
## ResNet

- Residual Network (ResNet)
- Skip connections (shortcut connections)
	- The signal feeding into a layer is also added to the output of a layer located higher up the stack
- When training a neural network, the goal is to make it model a target function $h(x)$
- Residual learning

<img src="/images/Pasted image 20260204110150.png" alt="image" width="500">

- When a regular neural network is initialized, its weights are close to zero, so the output is close to zero
- With skip connections, the resulting network outputs a copy of its inputs
	- Initially models the identity function
	- If the target function is close to the identity function, this speed up training
- With many skip connections, the network can start making progress even if several layers have not started learning yet
- The signal can easily make its way across the entire network
- The deep residual network can be seen as a stack of residual units (RUs)

- ResNet architecture
	- Starts and ends like GoogLeNet (without a dropout layer)
	- Between is a deep stack of residual units
	- Each residual unit is composed of two convolutional layers (no pooling layers), with batch normalization (BN), and RELU activation


<img src="/images/Pasted image 20260204110225.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110236.png" alt="image" width="500">
- The number of feature maps is doubled every few residual units, at the same time as their height and width are halved
- Inputs cannot be added directly to the outputs of the residual unit because they do not have the same shape
- Inputs are passed through a 1 x 1 convolutional layer with stride 2 and the right number of output feature maps

<img src="/images/Pasted image 20260204110248.png" alt="image" width="500">
- Different variations of this architecture with different number of layers

## Xception

- Variant of GoogLeNet architecture
	- Extreme Inception
- Merges GoogLeNet and ResNet, but it replaced the inception modules with a special type of layer called a depthwise separable convolution layer
- Regular convolutional layer filters
	- Spatial patterns
	- Cross channel patterns
- Separable convolutional layer
	- Separates spatial and cross channel patterns
- Applied a single spatial filter to each input feature map
- Exclusively look for cross channel patterns with a 1 x 1 filter

- Xception starts with 2 regular convolutional layers, the result uses only separable convolutions, plus a few max pooling layers, and a final layer
- An inception module contains convolutional layers with 1 x 1 filters
	- Only cross channel patterns
- Convolutional layers that sit on top of inception modules are regular convolutional layers that look for both spatial and cross channel patterns
- Separable convolutional layers use fewer parameters, less memory, and fewer computations than regular convolutional layers
	- Perform better
	- `SeperableConv2D`
	- `DepthwiseConv2D`


<img src="/images/Pasted image 20260204110257.png" alt="image" width="500">

## SENet

- Squeeze and Excitation Network (SENet)
- Extends using inception networks and REsNets, and boosts their performance
- Adds a small neural network, called an SE block, to every inception model or residual unit in the original architecture

<img src="/images/Pasted image 20260204110307.png" alt="image" width="500">
- SE block analyses the output of the unit it is attached to, focusing exclusively on the depth dimension
- Learns which features are most active together
- Uses this to recalibrate the feature maps
	- Boost connected features, and reduces irrelevant feature maps

<img src="/images/Pasted image 20260204110317.png" alt="image" width="500">
- SE block components
	- A global average pooling layer
	- A hidden dense layer using the ReLU activation function
	- Dense output layer using sigmoid activation function

<img src="/images/Pasted image 20260204110327.png" alt="image" width="500">

- The global average pooling computes the mean activation for each feature map
	- 256 -> 256
- SE layer, reduces feature maps to 16 dimensions
	- Embedding of the distribution of feature responses
	- Bottleneck step forces the SE block to learn a general representation of the feature combinations
- Output layer takes the embedding and outputs a recalibration vector containing one number per feature map, `[0, 1]`
	- Feature maps are multiplied by this recalibration vector, so irrelevant features get scaled down, while relevant are left along

## Other Noteworthy Architectures

- ResNetXt
- DenseNet
- MobileNet
- CSPNet (Cross Stage Partial Network)
- EfficientNet
	Compound scaling

## Choosing the Right CNN Architecture

- Large models are more accurate, but are also more computationally expensive


<img src="/images/Pasted image 20260302131446.png" alt="image" width="500">


# Implementing a ResNet-34 CNN Using Keras

- Most CNN architectures are implemented naturally using Keras
- Create all the layers
	- Main layers
	- Skip layers

```python
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
			padding="same", kernel_initializer="he_normal",
			use_bias=False)
			
class RedisualUnit(tf.keras.layers.Layer):
	def __init__(self, filters, strides=1, activation="relu", **kwargs):
		super().__init__(**kwargs)
		self.activation = tf.keras.activations.get(activation)
		self.main_layers = [
			DefaultConv2D(filters, strides=strides),
			tf.keras.layers.BatchNormalization(),
			self.activation,
			DefaultConv2D(filters),
			tf.keras.layers.BatchNormalization()
		]
		self.skip_layers = []
		if strides > 1:
			self.skip_layers = [
				DeffaultConv2D(filters, kernel_size=1, strides=strides).
				tf.keras.layers.BatchNormalization()
			]
			
	def call(self, inputs):
		Z = inputs
		for layer in self.main_layers:
			Z = layer(Z)
		skip_Z = inputs
		for layer in self.skip_layers:
			skip_Z = layer(skip_Z)
		return self.activation(Z + skip_Z)
```
- ResNet-34 model
	- Treat each residual unit a single layer

```python
model = tf.keras.Sequential([
	DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Activation("relu"),
	tf.keras.layers.MaxPool2D(pool_size=3, strides=2 padding="same")
])

prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
	strides = 1 if filters == prev_filters else 2
	model.add(ResidualUnit(filters, strides=strides))
	prev_filters = filters
	
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

- The first 3 RUs have 64 filters, then the next 4 RUs have 128
- At each iteration, set the stride to 1 when the number of filters is the same as the previous RU, or else set to 2
- Add RU, and update `prev_filters`

# Using Pretrained Models from Keras

- Pretrained networks are available with a single line of code
	- `tf.keras.applications`

```python
# load ResNet-50 odel
model = tf.keras.applications.ResNet50(weight="imagenet")

# resize images
images = load_sample_images()["images"]
images_resized = tf.keras.layers.Resizing(height=244, width=244,
			crop_to_aspect_ratio=True)(images)
			
# models rpovide a preprocessing function
inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)

Y_proba = model.predict(inputs)
Y_proba.shape
(2, 1000)

# return top K predictions
top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
	print(f"Image #{image_index}")
	for class_id, name, y_proba in top_K[image_index]:
		print(f"{class_id} - [name:12s]{y_prob:.2%}")
		
Image #0
  n03877845 - palace       54.69%
  n03781244 - monastery    24.72%
  n02825657 - bell_cote    18.55%
Image #1
  n04522168 - vase         32.66%
  n11939491 - daisy        17.81%
  n03530642 - honeycomb    12.06%
```
- The correct classes are palace and dahlia
	- Correct for first image but not second


# Pretrained Models for Transfer Learning

- To build an image classifier, reuse the lower layers of a pretrained model
- Train a model to classify pictures of flowers, with reused of Xception model

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label'].names
n_classes = info.features["label"].num_classes
```

- Only a train dataset, no test or validation set
- Split the training set, taking 10%, 15%, and 75% (test, validate, and training)

```python
test_set_raw, valid_set_raw, traib_set_raw = tfds.load(
	"tf_flowers",
	split["train[:10%]", "train[10%:25%]", "train[25%:]"],
	as_supervised=True
)
```

- Ensure they are all the same size, then batch

```python
batch_size = 32
preprocess = tf.keras.Sequential([
	tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
	tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])

train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
```

- Each batch contains 32 images (224 x 244) pixels from -1 to 1
- Add data augmentation to flip images, rotate, and tweak constrast

```python
data_augmentation = tf.keras.Sequential([
	tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
	tf.keras.layers.RandomRotation(factor=0.05, seed=42),
	tf.keras.layers.RandomConstrast(factor=0.2, seed=42)
])
```

- Load an Xception model, pretrained on ImageNet
- Exclude the top of the entwork
	- Excludes the global average pooling layer, and the dense output layer

```python
base_model = tf.keras.applications.xception.Xception(weight="imagenet",
			include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, output=output)

# Freeze the weights of the pretrained layers
for layer in base_model.layers:
	layer.trainable = False
	
# compile and start training
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
			metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=3)

# unfreeze some of the base model's top layers, then continue training
for layer in base_model.layers[56:]:
	layer.trainable =True
	
# recompile model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
				metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)
```

# Classification and Localization

- Localizing an object in a picture can be expressed as a regression task
	- Predict a bounding box around the object
	- Predict horizontal and vertical coordinates of the object's centre, with height and width
- Add a second dense output layer with 4 units
- Train using MSE loss

```python
base_model = tf.keras.applications.xception.Xception(weight="imagenet",
				include_top=False)
avg = tf.keras.layers.GlobalAveragePoolin2D()(base_model.output)
class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = tf.keras.layers.Dense(4)(avg)
model = tf.keras.Model(inputs=base_model.input,
			outputs=[class_output, loc_output])
optimizer = tf.keras.optimizer.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
		loss_weights=[0.8, 0.2],
		optimizer=optimizer, metrics=["accuracy"]
)
```

- Annotate images with bounding boxes
	- VGG image Annotator
	- LabelImg
	- OpenLabeler
	- ImgLab
	- LabelBox
	- Supervisely
- Create a dataset whose items will be batches or preprocessed images along with their class labels and their bounding boxes
	- `(images, (class_labels, bounding_boxes))`
- Intersection over union (IoU)
	- Area of overlap between predicted bounding box and the target bounding box, divided by the area of their unions

<img src="/images/Pasted image 20260204110343.png" alt="image" width="500">


# Object Detection

- Object detection
	- Classifying and localizing multiple objects in an image
- Take a CNN that was trained to classify and locate a single object roughly centred in the image, slide the CNN across the image and make predictions at each step
- CNN was trained to predict the objectness score
	- Estimated probability that the image does contain an object centred near the middle

<img src="/images/Pasted image 20260204110354.png" alt="image" width="500">
- Non-max suppression
	- Get rid of all the bounding boxes for which the objectness score is below some threshold
	- Find the remaining bounding box with the highest objectness score, and remove remaining ones
	- Repeat step 2 until there are not more bounding boxes to get rid of

## Fully Convolutional Networks

- Semantic segmentation
	- Task of classifying every pixel in an image according to the class of the object it belongs to
- To convert a dense layer to a convolutional layer, the number of filters in the convolutional layer must be equal to the number of units in the dense layer, the filter size must be equal to the size of the input feature maps, and use `"valid"` padding
- FCN contains only convolutional layers, it can be trained and executed on images of any size

- CNN for flower classification and localization, outputs 10 numbers
	- 0 to 4 are sent through the softmax activation function, and this gives the class probabilities
	- Output 5 is sent through the sigmoid activation function, and gives the objectness score
	- 6 and 7 represent the bounding box's centre coodtndates
		- Normalized
	- 8 and 9 represent the bounding box's height and width

- FCN will process the whole image only once, and will output a grid where each cell contains 10 numbers (5 class probabilities, 1 objectness score, and 4 bounding box coordinates)
- You only look once (YOLO)

<img src="/images/Pasted image 20260204110405.png" alt="image" width="500">

## Your Only Look Once

- YOLO is fast and accurate object detection architecture
- Run in real time on a video
- For each grid cell, YOLO only considers objects whose bounding box centre lies within that cell
	- Bounding box coordinates are relative
- Outputs two bounding boxes for each grid cell, which allows the model to handle cases where two objects are close to each other
- Outputs a class probability distribution for each grid cell
	- Probability map
	- Anchor priors

- Mean average precision (maP)
- Average precision (AP)

- Object detection
	- YOLOv5
	- SSD
	- Faster R-CNN
		- Region proposal network (RPN)
	- EfficentDET

# Object Tracking

- Object tracking is the task of identifying an image
	- Moving
	- Changing sizes
	- Changing appearances
		- Light conditions
		- Backgrounds
- DeepSORT
	- Kalman filters
		- Estimates the more likely current position of an object given prior detections, assuming that objects tend to move at a constant speed
		- Uses deep learning model to measure the resemblance between new detections and existing tracked objects
	- Hungarian algorithm
		- Maps new detections to existing track objects

# Semantic Segmentation

- Each pixel is classified according to the class of the object it belongs to
- When images go through a regular CNN, they gradually lose their spatial resolution
- Regular CNN may end up known that there's a person somewhere
- Takes a pretrained CNN and turns it into an FCN
	- CNN applies an overall stride of 32 to the input image
	- Add upsampling layer that multiples the resolution by 32
		- Bilinear interpolation
		- Transposed convolutional layer
			- Stretching the image by inserting empty rows and columns, then performing a regular convolution
- In a transposed convolutional layer, the stride defined how much the input will be stretched, not the size of the filter step

![[Pasted image 20260302150406.png]]

- Convolutional layers
	- `tf.keras.layers.Conv1D`
	- `tf.keras.layers.Conv32`
	- `dilation_rate`
		- a-trous convolutional layer

- Transposed convolutional layers are imprecise
- Add skip connections from lower layers
	- This recovered some of the spatial resolution that was lost in the earlier pooling layers
	- Output of the original CNN goes through
		- Upsample x 2
		- Add the output of a lower layer
		- Upsample x 2
		- Add the output of an even lower layer
		- Upsample x 8
- Super resolution
	- Scale up beyond the size of the original image

![[Pasted image 20260302150649.png]]

- Instance segmentation is similar to semantic segmentation
- Mask R-CNN
	- Extends the Faster R-CNN model by producing a pixel mask for each bounding box
- Adversarial learning
	- Make the network more resistant to images designed to fool it
- Explainability
	- Understanding why the network makes a specific classification
- Image generation
- Single-shot learning
	- A system that can recognize an object after it has seen it once
