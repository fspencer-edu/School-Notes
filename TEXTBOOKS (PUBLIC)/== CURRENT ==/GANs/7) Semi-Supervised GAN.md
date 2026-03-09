
- GAN models
	- Semi-supervised GAN
	- Conditional GAN
	- CycleCAN

# Introducing the Semi-Supervised GAN

- Semi-supervised learning trains model using a small amount of labeled data combined with a large volume of unlabeled data

## What is a Semi-Supervised GAN?

- SGAN is a GAN whose discriminator is a multiclass classifier
- Learns to distinguish between N + classes, where N is the number of classes in the training dataset, which one added for the fake generated samples

![[Pasted image 20260309104332.png]]

- Generator takes in a random noise vector z, and produces fake examples, $x*$
- Discriminator receives 3 types of inputs
	- Fake data
	- Real unlabeled data
	- Real labeled examples (x, y)
- Discriminator outputs a classification
	- To identify fake examples
	- Identify the correct class for real examples

## Architecture

- The SGAN generator is the same
- The discriminator
	- Receives 3 inputs
	- Multi classification
## Training process

- For GAN
	- The discriminator uses the loss for $D(x)$ and $D(x*)$ and backpropagating the total loss to update the trainable parameters to minimize loss
	- The generator is trained by backpropagating the discriminator's loss for $D(x*)$, and seeking to maximize it
- For SGAN
	- Unsupervised loss
		- In addition to $D(x)$ and $D(x*)$ 
	- Supervised loss
		- Computes the loss for the supervised training examples, $D((x, y))$


## Training objective

- In GAN
	- Once the generator is fully trained, the discriminator is usually discarded
- In SGAN
	- The goal of the training process is to make this network into a semi-supervised classifier whose accuracy is as close as possible to a fully supervised classifier

# Implementing a Semi-Supervised GAN

- Implement SGAN model to classify handwritten digits in the MNIST dataset with 100 training examples


## Architecture diagram

- Generator turns random noise into fake examples
- Discriminator receives real images with labels (x, y), real image without labels, and fake images
- Discriminator uses sigmoid function to distinguish real examples from fake
- Uses softmax, to distinguish the real classes


![[Pasted image 20260309105235.png]]


- Softmax
	- Gives probability distribution over a specified number of classes
	- The higher the probability assigned to a given label, the more confident the discriminator is right
- Cross-entropy loss
	- Used as classification error
	- Measures the different between the output probabilities and the target, one-hot-encoded labels
- Sigmoid
	- Fake vs real probability
	- Trains its parameters by backpropagating the binary cross-entropy loss

## Implementation

## Setup

```python
#import
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormaliztion, Concatenate, Dense,
		Dropout, Flatten, Input, Lambda, Reshape)
		
from keras.layers.advanced_activations imoport LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

# input dim
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100
num_classes = 10
```

## The dataset

- Only set a portion of the dataset with labels, and the rest unlabeled

```python
class Dataset:
	def __init__(self, num_labeled):
		self.num_labeled = num_labeled
		(self.x_train, self.y_train), 
		(self.x_test, self.y_test) = mnist.load_data()
		
		def preprocess_imgs(x):
			x = (x.astype(np.float32) - 127.5 / 127.5)
			x = np.expand_dims(x, axis=3)
			return x
			
		def preprocess_labels(y):
			return y.reshape(-1, 1)
			
		self.x_train = preprocess_imgs(self.x_train)
		self.y_train = preprocess_labels(self.y_train)
		
		self.x_test = preprocess_imgs(self.x_test)
		self.y_test = preprocess_imgs(self.y_test)
		
	def batch_labeled(self, batch_size):
		idx = np.random.randint(0, self.num_labeled, batch_size)
		imgs = self.x_train[idx]
		labels = self.y_train[idx]
		return imgs, labels
		
	def batch_unlabeled(self, batch_size):
		idx = np.random.randint(self.num_labeled, self.x_train.shape[0],
					batch_size)
		img = self.x_train[idx]
		return imgs
		
	def training_set(self):
		x_train = self.x_train[range(self.num_labeled)]
		y_train = self.y_train[range(self.num_labeled)]
		return x_train, y_train
		
	def test_set(self):
		return self.x_test, self.y_test
		
# data set
num_labeled = 100
dataset = Dataset(num_labeled)
```

## The Generator

```python
def build_generator(z_dim):
	model = Sequential()
	model.add(Dense(256 * 7 * 7, input_dim=z_dim))
	model.add(Reshape((y, y, 256)))
	
	model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
	
	model.add(BatchNormalization())
	
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
	
	model.add(BatchNormalization())
	
	model.add(LeakyRelU(alpha=0.01))
	model.add(Conv2DTranspose(1, kernel_size=3, strides=3, padding='same'))
	
	model.add(Activation('tanh'))
	
	return model
```

## The Discriminator

- Goals
	- Real vs fake
	- Find class labels for real examples

**The Core Discriminator Network**

- Start by defining the core discriminator network similar to ConvNet
	- Normalization and leaky ReLU
- Add dropout
	- A regularization technique that helps prevent overfitting by randomly dropping neurons and their connections from the NN during training
	- Reduces neuron codependence

```python
def build_discriminator_new(img_shape):
	model = Sequential()
	
	model.add(
		Conv2D(32,
		kernel_size=3,
		strides=2,
		input_shape=img_shape,
		padding='same'))
		
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(
		Conv2D(64,
		kernel_size=3,
		strides=2,
		input_shape=img_shape,
		padding='same'))
		
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(
		Conv2D(128,
		kernel_size=3,
		strides=2,
		input_shape=img_shape,
		padding='same'))
		
		
	model.add(BatchNormalization())
	
	mode.add(LeakyReLU(alpha=0.01))
	
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	
	model.add(Dense(num_classes))
	
	return model
```

- Dropout is added after batch normalization
- The network ends with a fully connected layer with 10 neurons
	- 2 discriminator outputs
		- Multiclass classification (softmax)
		- Binary classification (sigmoid)


**Supervised Discriminator**

```python
def build_discriminator_supervised(discriminator_net):
	model = Sequential()
	model.add(discriminator_net)
	model.add(Activation('softmax'))
	return model
```

**Unsupervised Discriminator**

```python
def build_discriminator_unsupervised(discriminator_net):
	model = Sequential()
	model.add(discriminator_net)
	
	def predict(x):
		prediction = 1.0 - (1.0 /
					K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0)
		return predictiont
		
	model.add(Lambda(predic))
	return model
```

- `predict(x)` is used to transform distribution over real classes into binary real vs fake probability


## Building the model

```python
# classification metrics
def build_gan(generator, discriminator):
	model = Sequential()
	
	model.add(generator)
	model.add(discriminator)
	
	return model
	
discriminator_net = build_discriminator_new(img_shape)
discriminator
```

## Training

# Comparison to a Fully Supervised Classifier

# Conclusion