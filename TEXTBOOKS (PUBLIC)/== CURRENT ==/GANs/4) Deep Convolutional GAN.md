- Deep convolutional GAN (DCGAN)
	- Uses CNNs instead of feed-forward network layers

# Convolutional Neural Networks

## Convolutional filters

- Feed-forward NN
	- Neurons are flat, fully connected layers
- CNN
	- Arranged in 3D
	- Sliding one or more filters over the input layer
	- Small receptive field, but extends the entire depth
- Each filter outputs a single activation value
	- Dot produce between the input value and filter entries

## Parameter sharing

- Filter parameters are shared by all the input values
- Efficiently learn visual features and shapes regardless of where they are located
- Parameter sharing reduces the number of trainable parameters
	- Scale up

## ConvNets visualized

![[Pasted image 20260307213623.png]]

- The input volume is usually 3D, and there are several stacked filters
- Each filter produces a single values per step

![[Pasted image 20260307213721.png]]

# Brief History of the DCGAN

- Alex Radford, Luke Metz, and Soumith Chintala (1026)
	- https://arxiv.org/pdf/1511.06434
- ConvNets help reduce instability and gradient saturation
- LAPGAN
	- Deep generative image model using a Laplacian pyramid of adversarial networks

# Batch Normalization

- Helps stablize the training process by normalizing inputs at each layer

## Understanding normalization

- Normalization is the scaling of data so that is has zero mean and unit variance

$\hat{x} = x( - \micro)/\sigma$

- Comparisons between features are easier, and makes the training process less sensitive
- Forward and backpropagation causes covariate shift

## Computing batch normalization

$\micro_B$ = mean of mini batch B
$\sigma_B^2$ = variance of the mini batch B
$\epsilon$ = numerical stability (avoid zero division)
$\hat{x}$ = normalized value

$\hat{x} = (x - \micro_b)/\sqrt{\sigma^2 + \epsilon}$

$y = \gamma \hat{x} + \beta$

- $\gamma, \beta$ are trainable parameters
- `keras.layers.BatchNormalization`
- Limits the amount by which updating parameters in the previous layer can affect the distribution of inputs received by the current layer

# Generating handwritten digits with DCGAN

- Use DCGAN architecture for the MNIST dataset

![[Pasted image 20260307214854.png]]


## Importing modules and specifying model input dimensions

```python
# import
import matplotlib.pyplot as plt
import numpy as np

import keras.datasets import mnist
from keras.layers import (
	Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizer import Adam

# input dim
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100
```

## Implementing the generator

- ConvNets have traditionally been used for image classification takes
	- h x w x c
- Instead of taking an image and processing it into a vector (class label)
- Transposed convolution
	- Take a vector and up-size it to an image
	- Used to increase the width and height, while reducing depth

![[Pasted image 20260307215329.png]]

- After each transposed convolutional layer, apply batch normalization and the leaky ReLU activation function
- The final layer, uses tanh and removes batch normalization

1. Noise vector is reshaped => 7 x 7 x 256 tensor
2. Transposed => 14 x 14 x 128
3. Batch normalization and leaky ReLU
4. Transposed => 14 x 14 x 64 tensor
	1. Stride parameter to 1
5. Batch normalization and leaky ReLU
6. Transpose => 28 x 28 x 1
7. Tanh

```python
# DCGAN generator
def build_generator(z_dim):
	model = Sequential()
	model.add(Dense(256 * 7 * 7, input_dim=z_dim))
	model.add(Reshape((7, 7, 256)))
	
	model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(Conv2DTranspose(1, kernel_size=3, stides=2, padding='smae'))
	model.add(Activiation('tanh'))
	return model
```


## Implementing the discriminator

- Takes an image and outputs a prediction vector
- Binary classification, real or fake

![[Pasted image 20260307220009.png]]

1. Convolutional layer 28 x 28 x 1 => 14 x 14 x 32 tensor
2. Leaky ReLU activation function
3. Convolutional layer => 7 x 7 x 64
4. Batch normalization and leaky ReLU
5. Convolutional layer => 3 x 3 x 128
6. Batch normalization and leaky ReLU
7. Sigmoid to produce probability

```python
# DCGAN discriminator
def build_disciminator(img_shape):
	model = Sequential()
	model.add(
		Conv2D(
			32,
			kernel_size=3,
			strides=2,
			input_shape=img_shape,
			padding='same'
		)
	)
	model.add(LeakyReLU(alpha=0.01))
	model.add(
		Conv2D(
			64,
			kernel_size=3,
			strides=2,
			input_shape=img_shape,
			padding='same'
		)
	)
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.01)
	model.add(
		Conv2D(
			128,
			kernel_size=3,
			strides=2,
			input_shape=img_shape,
			padding='same'
		)
	)
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.01)
	model.add(Flatten())
	model.add(Dense(1, activiation='sigmoid'))
	return model
```

## Building and running the DCGAN

```python
# build and compile DCGAN
def build_gan(generator, discriminator):
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	return model
	
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
					 optimizer=Adam(),
					 metrics=['accuracy'])
					 
generator = build_generator()
```

## Model output

# Conclusion