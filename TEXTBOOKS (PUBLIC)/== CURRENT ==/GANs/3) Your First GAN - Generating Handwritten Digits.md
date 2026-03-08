
# Foundations of GANs: Adversarial Training

- Generator and discriminator are represented by differentiable functions
	- NN each with its own cost function
- The two networks are trained by backpropagation using the discriminator's loss

<img src="/images/Pasted image 20260306192022.png" alt="image" width="500">

- Generator's goal is to produce examples that capture the data distribution of the training dataset
- Object recognition models learn the patterns in images to discern an image's content

## Cost functions

$J^{(G)}$ = Generator's cost function
$J^{(D)}$ = Discriminator's cost function

- The trainable parameters

Weight
$\theta^{(G)}$
$\theta^{(D)}$

- Cost function of a traditional NN is defined exclusively in terms of its own trainable parameters, $J(\theta)$
- GANs consists of two networks whose cost functions are dependent on both the networks' parameters

Generator's cost function = $J^{G}(\theta^{(G)}\theta^{(D)})$
Discriminator's cost function = $J^{D}(\theta^{(G)}\theta^{(D)})$

- Traditional networks can tune all its parameters during the training process
- In GAN, each network can tune only its own weights and biases

## Training Process

- Training of a traditional NN is an optimization problem

<img src="/images/Pasted image 20260306192703.png" alt="image" width="500">

- GAN training can be described as a game, rather than optimization
- GAN training ends when two network reach Nash equilibrium
	- Generator cost function is minimized 
	- Discriminator cost function is minimized 

<img src="/images/Pasted image 20260306192842.png" alt="image" width="500">

- High-dimensional, non-convex training of GANs
- Training GANs successfully requires trails and error


# The Generator and the Discriminator

$G(z) = x^*$
$D(x)$ => 1
$D(x*)$ => 0


<img src="/images/Pasted image 20260306193055.png" alt="image" width="500">

## Conflicting objectives

- The generator's goal is opposite of the discriminators

$G(x)$ => 0
$G(x*)$ => 1

## Confusion matrix

- Discriminator's classifications can be expressed in terms of a confusion matrix

TP, $D(x) =1$
FN, $D(x) =0$
TN, $D(x*) =0$
FP, $D(x*) =1$

- Discriminator tries to max. TP, and TN
- Minimize FP and FN

# GAN Training Algorithm

1) Train the discriminator
2) Train the generator

- In step 1, the generator's parameters are kept intact while the discriminator is training
- In step 2, the discriminator's parameters are fixes

- Update only the weights and biases of the network being trained
- Each network gets relevant signals about the updates to make, without interferences

# Generating Handwritten Digits

<img src="/images/Pasted image 20260306193550.png" alt="image" width="500">

- Create a GAN that learns to produce realistic handwriting digits

## Importing modules and specifying model input dimensions

```python
# import
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

from keras.datasts import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

# input dim.
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)
z_dim = 100
```

## Implementing the generator

- The generator is a NN with only a single hidden layer (leaky ReLU, small positive gradient)
- Takes z as input, and produces a 28 x 28 x 1 image
- Output layer uses the tanh activation function, `[-1, 1]`

```python
# generator
def build_generator(img_shape, z_dim):
	model = Sequential()
	model.add(Dense(128, input_dim=z_dim))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(28*28*1, activation='tanh'))
	model.add(Reshape(img_shape))
	return model
```

## Implementing the discriminator

- Takes a 28 x 28 x 1 image and outputs a probability indicating whether the input is real or fake
- 2 NN, with 128 hidden units and a leaky ReLU activation function
- Uses sigmoid as the output, `[0,1]`

```python
def build_discriminator(img_shape):
	model = sequential()
	model.add(Flatten(input_shape=img_shape))
	model.add(Dense(128))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(1, activation='sigmoid'))
	return model
```

## Building the model

- Use binary cross-entropy as the loss function to minimize during training
	- Measures the difference between the computed and actual probabilities for predictions with 2 classes 
- Use Adam optimization algorithm
	- Adaptive moment estimation
	- An advanced gradient-descent-based optimizer

```python
# build and compile GAN
def build_gan(generator, discriminator):
	model = sequential()
	model.add(generator)
	model.add(discriminator)
	return model
	
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
					  optimizer=Adam(),
					  metrics=['accuracy'])
					  
generator = build_generator(img_shape, z_dim)
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy',
			optimizer=Adam())
```

- `trainable` is set to false to keep discriminator's parameters fixed during generator training

## Training

- Get a random mini-batch of MNIST images as real examples and generate a mini-batch of fake images from random noise vectors z
- Train the discriminator network on images with noise, and keep the generators parameters fixed
- Generate a mini-batch of fake images and use those to train the generator network, while keeping the discriminator fixed, and repeat
- Rescale the image in the training from -1 to 1

```python
# GAN training loop
losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_intervals):
	(X_train, _), (_, _) = mnist.load_data()
	
	X_train = X_train / 127.5 - 1.0
	X_train = np.expand_dims(X_train, axis=3)
	
	real = np.one((batch_size, 1))
	fake = np.zeros((batch_size, 1))
	
	for iteration in range(iterations):
		idx = np.random.randint(0, X_train.shape[0], batch_size)
		gen_ims = generator.predict(z)
		
		d_loss_real = discriminator.train_on_batch(imgs, real)
		d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
		d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
		
		z = np.random.normal(0, 1, (batch_size, 100))
		gen_imgs = generator.predict(z)
		
		g_loss = gan.train_on_batch(z, real)
		
		if (iteration + 1) % sample_interval == 0:
			
			losses.append((d_loss, g_loss))
			accuracies.append(100.0 * accuracy)
			iteration_checkpoints.append(iteration + 1)
			
			print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
				(iteration + 1, d_loss, 100.0 * accuracy, g_loss))
				
			sample_images(generator)
		
```
## Outputting sample images

- `sample_images()` function, outputs a 4x4 grid of images synthesized by the generator after each iteration

```python
def sample_images(generator, image_grid_rows=4, image_grid_colums=4):
	z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
	gen_imgs = generator.predict(z)
	gen_imgs = 0.5 * gen_imgs + 0.5
	
	fig, axis = plt.subplots(image_grid_rows,
							 image_grid_columns,
							 figsize=(4, 4),
							 sharey=True,
							 sharex=True)
							 
	cnt = 0
	for i in range(image_grid_rows):
		for i in range(image_grid_columns):
			axsp[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
			axs[i, j].axis('off')
			cnt += 1
```

## Running the model

- Each mini-batch must be small enough to fit inside the processing memory (32 to 512)
- Monitor training loss and set the iteration number around the point when the loss plateaus

```python
iterations = 20000
batch_size = 128
sample_interval = 1000

train(iterations, batch_size, sample_intervals)
```

## Inspecting the results

![[Pasted image 20260307212857.png]]

![[Pasted image 20260307212917.png]]


![[Pasted image 20260307212942.png]]
# Conclusion

