
- Autoencoders are artificial neural networks capable of learning dense representations of the input data without supervision
	- Latent representations or codings
	- Lower dimensionality
- Autoencoders also act as feature detectors
	- Unsupervised pretraining of deep neural networks
- Generative models
	- Autoencoders capable of randomly generating new data similar to the training data
- Generative adversarial networks (GANs)
	- Super resolution
	- Colourization
	- Image editing
	- Photo-realistic cimages
	- Predicting next frames
	- Augmenting a dataset
	- Generating other types of data
	- Identifying the weakness in other models
- Diffusion models
	- Generative AI, that creates high quality media
	- Training a neural network to reverse corrupted data

- Autoencoders
	- Copy inputs to their outputs
	- Codings are byproducts of the autoencoder learning the identity function under some constraint
- GANs
	- Generator
		- Tries to generate data that looks similar to the training data
	- Discriminator
		- Tries to tell real data from fake data
	- Generator and discriminator compete during raining
	- Adversarial learning
		- Training competing neural networks
- Denoising diffusion probabilistic model (DDPM)
	- Trained to remove noise from an image


# Efficient Data Representations

- An autoencoder looks at the inputs, converts them to an efficient latent representation, and then outputs something that looks closed to the inputs
	- Encoder/recognition network
		- Converts the inputs to a latent representation
	- Decoder/generative network
		- Converts the internal representation to the outputs

![[Pasted image 20260304093239.png]]

- An autoencoder has the same architecture as a multilayer perceptron (MLP)
- The outputs are called reconstructions
- Cost function contains a reconstruction loss that penalizes the model when the reconstructions are different from the inputs
- Autoencoder is undercomplete, when input data is lower dimensionality
	- Forced to learn the most important features in the input data

# Performing PCA with an Undercomplete Linear Autoencoder

- Autocoder with linear activations and the cost function is the mean squared error, then it end up performing principal component analysis

```python
# PCA: Project 3D to 2D
import tensorflow as tf
encoder = tf.keras.Sequential([tf.keras.layers.Dense(2)])
decoder = tf.keras.Sequential([tf.keras.layers.Dense(3)])
autoencoder = tf.keras.Sequential([encoder, decoder])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
autoencoder.compile(loss="mse", optimizer=optimizer)
```

- Organized the autoencoder into 2 subcomponents
- Autoencoder's number of output is equal to the number of inputs

```python
# train model on 3D dataset
X_train = [...]
history = autoencoder.fit(X_train, X_train, epochhs=500, verbose=False)
codings = encoder.predict(X_train)
```

- `X_train` is used as inputs and the target
- Autoencoder found the best 2D plane to project the data onto

![[Pasted image 20260304093900.png]]


# Stacked Autoencoders

- Stacked autoencoders/deep autoencoders
	- Multiple hidden layers
	- Symmetrical to the central hidden layer

<img src="/images/Pasted image 20260204110826.png" alt="image" width="500">

## Implementing a Stacked Autoencoder Using Keras

```python
stacked_encoder = tf.keras.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(30, activation="relu"),
])
stacked_decoder = tf.keras.Sequential([
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(28 * 28),
	tf.keras.layers.Reshape([28, 28])
])
stacked_ae = tf.keras.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="mse", optimizer="nadam")
history = stacked_ae.fit(X_train, X_train, epochs=20,
			validation_data(X_train, X_valid))
```

## Visualizing the Reconstructions
## Visualizing the Fashion MNIST Dataset
## Unsupervised Pretraining Using Stacked Autoencoder
## Trying Weights



## Training One Autoencoder at a Time

# Convolutional Autoencoders

# Denoising Autoencoders

# Spares Autoencoders

# Variational Autoencoders

# Generating Fashion MNIST Images

# Generative Adversarial Networks

## The Difficulties of Training GANs
## Deep Convolutional GANs
## Progressive Growing of GANs
## StyleGANs

# Diffusion Models