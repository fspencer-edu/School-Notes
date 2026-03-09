
- GAN models
	- Semi-supervised GAN
	- Conditional GAN
	- CycleCAN

# Introducing the Semi-Supervised GAN

- Semi-supervised learning trains model using a small amount of labeled data combined with a large volume of unlabeled data

## What is a Semi-Supervised GAN?

- SGAN is a GAN whose discriminator is a multiclass classifier
- Learns to distinguish between N + classes, where N is the number of classes in the training dataset, which one added for the fake generated samples

<img src="/images/Pasted image 20260309104332.png" alt="image" width="500">

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


<img src="/images/Pasted image 20260309105235.png" alt="image" width="500">


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

# supervised
discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy',
								 metrics=['accuracy'],
								 optmizer=Adam())

# unsupervised
discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossentropy',
								   optimizer=Adam())
								   
generator = build_generator(z_dim)
discriminator_unsupervised.trainable = False
gan = build_gan(generator, discriminator_unsuperivsed)
gan.compile(loss='binary_crossentropy', optimizer=Adam())
```

## Training

1. Train the discriminator (supervised)
	1. Take a random mini-batch of labeled real examples (x, y)
	2. Compute D((x, y)), update $\theta^{(D)}$ to minimize loss (multi-class)
2. Train the discriminator (unsupervised)
	1. Take a random mini=batch of unlabeled real examples
	2. Compute D(x), and update $\theta^{(D)}$ to minimize loss (binary classifier)
	3. Take a mini-batch of random noise vector z , and generate fake examples, $G(z) = x*$
	4. Compute $D(x*)$, and  update $\theta^{(D)}$ to minimize loss (binary)
3. Train the generator
	1. Take a mini-batch of random noise z, and generate fake examples
	2. Compute $D(x*)$, and update $\theta^{(D)}$ to maximize loss (binary)


```python
supervised_losses =[]
iteration_checkpoints = []

def train(iterations, batch_size, sample_intervals):
	real = np.ones((batch_size, 1))
	fake = np.zeros((batch_size, 1))
	
	for iteration in range(iterations):
		imgs, labels = dataset.batch_labeled(batch_size)
		labels = to_categorical(labels, num_classes=num_classes)
		imgs_unlabeled = dataset.batch_unlabeled(batch_size)
		
		z = np.random.normal(0, 1, (batch_size, z_dim))
		gen_imgs = generator.predict(x)
		
		d_loss_supervised, accuracy =
			discriminator_supervised.train_on_batch(imgs, labels)
			
		d_loss_real =
			discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)
			
		d_loss_fake = discriminator_unsupervised.train_on_batch(gens_imgs, fake)
		
		d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)
		
		z = np.random.normal(0, 1, (batch_size, z_dim))
		
		g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))
		
		if (iteration + 1) % sample_interval == 0:
			supervised_losses.append(d_loss_supervised)
			iteration_checkpoints.append(iteration + 1)
			
			print(
			"%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss" +
                " unsupervised: %.4f] [G loss: %f]"
                % (iteration + 1, d_loss_supervised, 100 * accuracy,
                  (d_loss_unsupervised, g_loss))
```

**Training the model**

- Use a smaller batch size for the 100 labeled examples for training
- Number of iterations is determined by trail and error

```python
iterations = 8000
batch_size = 32
sample_interval = 800

train(iterations, batch_size, sample_interval)
```

**Model training and test accuracy**

```python
x, y = dataset.test_set()
y = to_categorical(y, num_classes=num_classes)

_, accuracy = discriminator_supervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))
```

# Comparison to a Fully Supervised Classifier

```python
mnist_classifier = build_discriminator_supervised(
			build_discriminator_net(img_shape))
			
mnist_classifier.compile(loss='categorical_crossentropy',
						 metrics=['accuracy'],
						 optimizer=Adam())
```

- Train the fully supervised classifier with the same 100 training examples
- The unsupervised and supervised achieves 100% accuracy on the training dataset
	- Unsupervised gets 89% on test dataset
	- Supervised gets 70% on test set

# Conclusion

