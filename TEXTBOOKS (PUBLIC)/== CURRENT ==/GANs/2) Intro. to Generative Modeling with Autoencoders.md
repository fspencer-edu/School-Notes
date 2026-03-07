# Introduction to Generative Modeling

- Latent space
	- Compressed, lower-dimensional representation of data where similar concepts are clustered together
	- Random noise is often referred to as a sample from the latent space
	- Hidden representation of a data point

# How do Autoencoders Function on a High Level?

- Autoencoders
	- An unsupervised AI NN designed to learn efficient, compressed representations of input data and then reconstruct the original input as closely as possible
	- Composed of 2 parts
		- Encoder
		- Decoder
	- Reconstruction loss, $||x - x^*||$
	- Uncovers information-efficient patterns, define them, and use them to increase the information throughput
	- Saves bandwidth
	- Passes through the information bottleneck without sacrificing too much understanding

**Latent Space**
- The latent space is the hidden representation of the data
- Rather than expressing words or images in their uncompressed versions, an autoencoder compresses and clusters them based on its understanding of the data

# What are Autoencoders to GANs?

- Autoencoders end-to-end train the entire network with one loss function
	- Specific cost function to optimize
- GANs have distinct loss functions
	- Do not have an explicit metric as simple as AE

**Cost function**
- A cost/loss/objective function is what is trying to be optimized/minimized
	- Root mean squared error (RMSE)
	- AUC

![[Pasted image 20260306180808.png]]


# What is an Autoencoder Made of?

**Autoencoder**
1) Encoder network
	1) Takes $x$ and reduces the dimension from y to z by using a learned encoder
2) Latent space (z)
	1) During training, try to establish the latent space to have some meaning
	2) Representation of a smaller dimension and acts as an intermediate step
3) Decoder network
	1) Reconstruct the original object into the original dimension
	2) Mirror image of the encoder, from z to $x^*$

![[Pasted image 20260306181123.png]]

**Autoencoder Training**
1) Take images x and feed them through the autoencoder
2) Get $x^*$, reconstruction of the images
3) Measure the reconstruction loss
	1) Distance
	2) This gives an explicit object function to optimize via a version of gradient descent


# Usage of Autoencoders

- Reduces dimensionality
- Comparing closeness in latent space
- Denoising
- Training does not require labeled data
- Use autoencoders to generate new data

**MNIST**
- The Modified National Institute of Standards and Technology (MNIST) database is a database of handwritten digits

# Unsupervised Learning

**Unsupervised Learning**
- A type of ML in which we learn from the data itself without additional labels
- Clustering

**Supervised Learning**
- A type of ML where algorithms are trained using labeled datasets
- Anomaly detection

**Self-Supervised**
- ML that trains models on unlabeled data by automatically generator supervisory signals from the data's own structure

## New Take on an old idea

- An autoencoder is composed of an encoder and a decoder
	- Both have activation function and intermediate layers
	- 2 weight matrices in each network
		- Encoder (2): Input -> Intermediate, Intermediate -> Latent
		- Decoder (2): Latent -> Intermediate, Intermediate -> Output

- A network with one weight matrix, would resemble principal component analysis (PCA)
	- Dimensionality reduction technique
	- Numerically deterministic
- Autoencoders are trained with a stochastic optimizer

## Generation using an autoencoder

![[Pasted image 20260306182330.png]]

## Variational autoencoder

- A variational autoencoder has a latent space represented as a distribution with a learned mean and standard deviation rather that a set of numbers
- Multi-variate Gaussian
- Bayes on Bayesian machine learning
	- Find the right parameters defining a distribution
	- Sample from the latent distribution to get numbers for the decoder

# Code if Life

- Keras is a high-level API for deep learning frameworks
	- TensorFlow
	- Microsoft Cognitive Toolkit (CNTK)
	- Theano


**Pooling**
- A pooling block is an operation on a layer that allows us to pool several inputs into fewer
- Reduces complexity
- The API uses lambda functions to return constructors for another function
- Sample from the latent space, and feed this information through to the decoder

- Create an project that will generate handwritten digit based on the latent space, given an numerical input

```python
# imports
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np

# setting hyperparamters
batch_size = 100
original_dim = 28*28
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5
epslion_std = 1.0

# encoder
x = Input(shape=(original_dim,), name="input")
h = Dense(intermediate_dim, activation='relu', name="encoding")
z_mean = Dense(latent_dim, mean="mean")(h)
z_log_var = Dense(latent_dim, name="log-variance")(h)
z = Lambda(sampling, output_shape(latent_dim,))([z_mean, z_log_var])
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")

# Sampling helper function
def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
	return z_mean + K.exp(z_log_var / 2) * epsilon
```
- We learn the mean $\micro$ and variance $\sigma$
- Where there is one $\omega$ connected through a sampling function
- This allows us to train and sample
- During generation, sample from this distribution according to learned parameters, and feed these values to the decoder

![[Pasted image 20260306183759.png]]

```python
# decoder
input_decoder = Input(shape=(latent_dim,), name="decocer_input")
decoder_h = Dense(intermediate_dim, activation='relu',
	name="decoder_h")(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid',
	name="flat_decoded")(decoder_h)
	decoder = Model(input_deocder, x_decoded, name="decoder")
	
# combining the model
output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()

# Loss function
def vae_loss(x, x_decoded_mean, z_log_var, z_mean,
		original_dim=original_dim):
	xent_loss = oroginal_dim * objectives.binary_crossentropy(
		x, x_decoded_mean)
	kl_loss = - 0.5 * K.sum(
		1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
			axis=-1)
	return xent_loss + kl_loss
	
vae.compile(optimizer='rmsprop', loss=vae_loss)
```

**Binary Cross-Entropy**
- Common loss function for two-class classification

**KL divergence/Relative entropy**
- Measures the difference between distributions
- Kullback-Leiber divergence
- Difference between cross-entropy of two distributions and their own entropy
- Non-overlap of two distributions

- Compiled
	- RMSprop
	- Adam
	- Vanilla. stochastic gradient descent


**Stochastic Gradient Descent**
- An optimization technique that allows to to train complex models by figuring out the contribution of any given weight to an error and updating this weight

- Train the model with train-test split and input normalization

```python
x_train, y_train, x_test, y_test = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_train.astype('float32')/255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
	shiffle=True,
	nb_epoch=nb_epoch,
	batch_size=batch_size,
	validation_data=(x_test, x_test), verbose=1)
```

![[Pasted image 20260306184955.png]]

![[Pasted image 20260306185011.png]]



# Why Did We Try aGAN?

**Bimodal**
- Having 2 peaks, or modes

- The point estimate can be wrong, and live in an area where there is no actual data sampled from the true distribution
- We learned 2D normal distribution in the latent space centred around the origin
- VAE uses the Gaussian as a way to build representation of the data it sees
- VAEs do not scale up as well as GANs
- VAEs live in the directly estimated max. likelihood model familty
- GANs have an implicit and hard-to-analyze understanding of the real data distribution


## Summary