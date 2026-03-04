
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

- Split the autoencoder into 2 submodels
- Encoder takes a flattened image
- The decoder takes codings of size 30, and processes them
- Train the model using `X_train` as inputs and targets

## Visualizing the Reconstructions

```python
import numpy as np

def plot_reconstructions(model, images=X_valid, n_images=5):
	reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
	fit = plt.figure(figsize=(n_images * 1.5, 3))
	for image_index in range(n_images):
		flt.subplot(2, n_images, 1 + image_index)
		plt.imshow(images[image_index], cmap="binary")
		plt.axis("off")
		plt.subplot(2, n_images, 1 + n_images + image_index)
		plt.imshow(reconstructions[image_index], cmap="binary")
		plt.axis("off")
		
plot_reconstruction(stacked_ae)
plt.show()
```

![[Pasted image 20260304094656.png]]

## Visualizing the Fashion MNIST Dataset

- Use the autoencoder to reduce the dataset's dimensionality
- Use dimensionality reduction algorithm for visualization
	- t-SNE

```python
from sklearn.manifold import TSNE

X_valid_compressed. stacked_encoder.predict(X_valid)
tsne = TSNE(init="pca", learning_rate="auto", random_state=42)
X_valid_2D = tsne.fit_transform(X_valid_compressed)

# plot dataset
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
plt.show()
```

![[Pasted image 20260304094924.png]]

- The t-SNE algorithm identified several cluster that match the classes


## Unsupervised Pretraining Using Stacked Autoencoder

- With limited labeled training data, a solution is to find a neural network that performs a similar task and reuse its lower layers

![[Pasted image 20260304095102.png]]

- Train an autoencoder using all the training data, then reuse its encoder layers to create a new neural network

## Trying Weights

- When an AE is neatly symmetrical, a common technique is to tie the weights of the decoder layer to the weights of the encoder layer
	- Halves the number of weights in the model

```python
class DenseTranspose(tf.keras.layers.Layer):
	def __init__(self, dense, activation=None, **kwargs):
		super().__init__(**kwargs)
		self.dense = dense
		self.activation = tf.keras.activations.get(activation)
		
	def build(self, batch_input_shape):
		self.biases = self.add_weight(name="bias",
					shape=self.dense.input_shape,
					initalizer="zeros")
					
	def call(self, inputs):
		Z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
		return self.activation(Z + self.biases)
```

- The custom layer acts like a dense layer, but uses another dense layer's weight
	- Transposed is equivalent to transposing the second argument
- Build a new stacked autoencoder, with the decoder's dense layers tied to the encoder's

```python
dense_1 = tf.keras.layer.Dense(100, activation="relu")
dense_2 = tf.keras.layer.Dense(30, activation="relu")

tied_encoder = tf.keras.Sequential([
	tf.keras.layers.Flatten(),
	dense_1,
	dense_2
])

tied_decoder = tf.keras.Sequential([
	DenseTranspose(dense_2, activation="relu"),
	DenseTransport(dense_1),
	tf.keras.layers.Reshape([28, 28])
])

tied_ae = tf.keras.Sequential([tied_encoder, tied_decoder])
```

## Training One Autoencoder at a Time

- Rather than training the whole stacked autoencoder, it is possible to train one shallow autoencoder at a time, then stack all of them into a single stacked autoencoder
- Greedy layer-wise training

![[Pasted image 20260304095744.png]]

- During the first phase of training, the first AE learns to reconstruct the inputs
- Then the entire training set uses the first autoencoder, and outputs a new training set
- The second AE uses the new training set
- Stack the hidden layers, then output layers in reverse order

# Convolutional Autoencoders

- To work with images, an autoencoder will need a convolutional autoencoder
- The encoder is a regular CNN composed of convolutional layers and pooling layers
- The decoder must do the reverse
	- Upscale the image, and reduce its depth

```python
conv_encoder = tf.keras.Sequential([
    tf.keras.layers.Reshape([28, 28, 1]),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),  # output: 14 × 14 x 16
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),  # output: 7 × 7 x 32
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),  # output: 3 × 3 x 64
    tf.keras.layers.Conv2D(30, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAvgPool2D()  # output: 30
])
conv_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(3 * 3 * 16),
    tf.keras.layers.Reshape((3, 3, 16)),
    tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu"),
    tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same",
                                    activation="relu"),
    tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same"),
    tf.keras.layers.Reshape([28, 28])
])
conv_ae = tf.keras.Sequential([conv_encoder, conv_decoder])
```

- Also possible to create autoencoder with other architecture types
	- RNNs
- Overcomplete autoencoder
	- Coding layers that are large than the input

# Denoising Autoencoders

- For AE to learn features, add noise to inputs, training it to recover the original
- Stacked denoising autoencoders
	- Gaussian
	- Random
- Regular stacked AE with an additional `Dropout` layer in the encoder's inputs

```python
dropout_encoder = tf.keras.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(30, activation="relu")
])

dropout_decoder = tf.keras.Sequential([
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(28 * 28),
	tf.keras.layers.Reshape([28, 28])
])
dropout_ae = tf.keras.Sequential([dropout_encoder, dropout_decoder])
```

![[Pasted image 20260304100542.png]]

- The AU guesses the details that are not in the input

![[Pasted image 20260304100608.png]]

# Spares Autoencoders

- Another type of constraint is sparsity
	- Adding an appropriate term to the cost function
	- AE is pushed to reduce the number of active neurons in the coding layer
	- Coding layers end up representing a useful feature
- Use sigmoid activation in the coding layer (0 and 1)
- A large coding layer
- Add $\ell_1$ regularization to coding layer's activations

```python
sparse_11_encoder = tf.keras.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(300, activation="sigmoid"),
	tf.keras.layers.ActivityRegularization(11=1e-4)
])
spare_11_decoder = tf.keras.Sequential([
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(28 * 28),
	tf.keras.layers.Reshape([28, 28])
])
sparse_11_ae = tf.keras.Sequential([sparse_11_encoder, sparse_11_decoder])
```

- `ActivityRegularization`
	- Returns its inputs, bus as a side effect is adds a training loss equal to the sum of the absolute values of its inputs
- Penalty will encourage the NN to produce encodings close to 0
- Penalized if it does not reconstruct the inputs correctly
- Measure the actual sparsity of the coding layer at each training iteration
- Compute the average activation of each neuron in the coding layer, over the whole training batch
- Penalize the neurons that are too, and not enough active by adding a sparsity loss to the cost function
- Kullback-Leibler (KL) divergence
	- Stronger gradients than the MSSE


![[Pasted image 20260304101439.png]]


- Kullback-Leibler Divergence

![[Pasted image 20260304101457.png]]

- KL divergence between the target sparsity $p$ and the actual sparsity $q$

![[Pasted image 20260304101553.png]]

 - Sum of the losses and add the result to the cost function
	 - Multiply the sparsity loss by a sparsity weight hyperparameter

```python
kl_divergence = tf.keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
	def __init__(self, weight, target):
		self.weight = weight
		self.target = target
		
	def __call__(self, inputs):
		mean_activities = tf.reduce_mean(inputs, axis=0)
		return self.weight * (
			kl_divergence(self.target, mean_activities) +
			kl_divergence(1. - self.target, 1. - mean_activities))
			
# sparse AE
kld_reg = KLDivergenceRegularizer(weight=5e-3, target=0.1)
sparse_kl_encoder = tf.keras.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(300, activation="sigmoid",
				activity_regularizer=kld_red),
])
sparse_kl_decoder = tf.keras.Sequential([
	tf.keras.layers.Dense(100, activation="relu"),
	tf.keras.layers.Dense(28 * 28),
	tf.keras.layers.Reshape([28, 28])
])
sparse_kl_ae = tf.keras.Sequential([sparse_kl_encoder, sparse_kl_decoder])
```

# Variational Autoencoders

- Variational autoencoders (VAEs)
	- Probabilistic AE
		- Outputs are partly determined by chance
	- Generative AE
		- Generate new instance that look like they were sampled from the training set

- RBM need to wait for the network to stablize into a "thermal equilibrium" before sampling a new instance
- Approximate Bayesian inference
	- Updating a probability distribution based on new data
- Original distribution = prior
- Updated distribution = posterior

- Find a good approximation of the data distribution

![[Pasted image 20260304102451.png]]

- Instead of directly producing a coding for a given input, the encoder produces a mean coding, $\micro$, and standard deviation, $\sigma$
- Coding is then sampled randomly from a Gaussian distribution with mean and std
- The decoder samples the coding normally
- Training instance goes through this AE, encoder produces mean and std, then a coding is sampled randomly, coding id decoded
- During training, the cost function pushes the codings to migrate within the coding space (latent space)
- Cost function
	- Reconstruction loss
		- MSE
	- Latent loss
		- KL divergence between the target distribution and the actual distribution of the codings

- Variational autoencoder's latent loss

![[Pasted image 20260304102824.png]]

- A common tweak to the variational autoencoder's architecture is to make the encoder output, $\gamma = log(\sigma)^2$

- Variational AE's latent loss
![[Pasted image 20260304102930.png]]


```python
class Sampling(tf.keras.layers.Layer):
	def call(self, inputs):
		mean, log_var = inputs
		return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean
# encoder
coding_size = 10
inputs = tf.keras.layers.Input(shape=[28, 28])
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="relu")(Z)
Z = tf.keras.layers.Dense(100, activation="relu")(Z)
codings_mean = tf.keras.layers.Dense(coding_size)(Z) # μ
codings_log_var = tf.keras.layers.Dense(coding_size)(Z) # γ
codings = Sampilng()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
	inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
)
# decoder
decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(100, activation="relu")(decoder_inputs)
x = tf.keras.layers.Dense(150, activation="relu")(x)
x = tf.keras.layers.Dense(28 * 29)(x)
outputs = tf.keras.layers.Reshape([28, 28])(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])

# variational AE
_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

# Add latent and reconstruction loss
latent_loss = -0.5 * tf.reduce_sum(
	1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
	axis=-1)
variational_ae.add_loss(tf.reduce_mean(latent_loss) / 784.)

# compile and fit AE
variational_ae.compile(loss="mse", optimizer="nadam")
history = variational_ae.fit(X_train, X_train, epochs=25, batch_size=128,
			validation_data(X_valid, X_valid))
```

# Generating Fashion MNIST Images


- Generate images that look like fashion items
- Sample random codings from a Gaussian distribution and decode them

```python
codings = tf.random.normal(shape=[3 * 7, codings_size])
images = variational_decoder(codings).numpy()
```

![[Pasted image 20260304103830.png]]

- Variational AE make is possible to perform semantic interpolation
	- Instead of interpolating between two images at the pixel level, interpolate at the codings level

```python
codings = np.zeros([7, codings_size])
codings[:, 3] = np.linspace(-0.8, 0.8, 7)
images = variational_decoder(codings).numpy()
```

![[Pasted image 20260304104001.png]]

# Generative Adversarial Networks

- GAN is composed of 2 NN
	- Generator
		- Takes a random distribution as input, and outputs some data
		- Similar to decoder in a variational AE
	- Discriminator
		- Takes a fake or real image from the generator as input, and determines the category

![[Pasted image 20260304104141.png]]

- Each iteration has 2 phases
	- Train the discriminator
		- Real images are sampled from the training set, and is compete with an equal number of fake images from generator
		- Binary cross-entropy loss
		- Backpropagation optimizes the weights of the discriminator
	- Train the generator
		- Produce another batch of fake images
		- Produce images that the discriminator will believe to be rule
		- Weights of the discriminator are froze during this step, backprop. only affects the generator

- Generator never sees any real images
- The better the discriminator gets, the more information about the real images is contained in these secondhand gradients

- Build the generator and discriminator
	- Generator is similar to an AE decoder
	- Discriminator is a regular binary classifier

```python
codings_size = 30
Dense = tf.keras.layers.Dense
generator = tf.keras.Sequential([
	Dense(100, activation="relu", kernel_initialization="he_normal"),
	Dense(150, activation="relu", kernel_initialization="he_normal"),
	Dense(28 * 28, activation="sigmoid"),
	tf.keras.layers.Reshape([28, 28])
])
discriminator = tf.keras.Sequential([
	tf.keras.layers.Flatten(),
	Dense(150, activation="relu", kernel_initialization="he_normal"),
	Dense(100, activation="relu", kernel_initialization="he_normal"),
	Dense(1, activation="sigmoid")
])
gan = tf.keras.Sequential([generator, discriminator])
```

- Use binary cross-entropy loss
- Generator will only be trained through `gan` model, so we do not need to compile it at all
- Discriminator should not be trained during the second phase

```python
discriminator.compile(loss="binary_crossentropy", optimizer="nadam")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# write a custom training loop
batch_size = 32
daraset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, codings_size, n_epochs):
	generator, discriminator = gan.layers
	for epoch in range(n_epochs):
		for X_batch in dataset:
			# phase 1 - training the disc.
			noise = tf.random.normal(shape=[batch_size, codings_size])
			generated_images = generator(noise)
			X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
			y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
			discriminator.train_on_batch(X_fake_and_real, y1)
			# phase 2 - training the gen.
			noise = tf.random.normal(shape=[batch_size, codings_size])
			y2 = tf.constant([[1.]] * batch_size)
			gan.train_on_batch(noise, y2)
			
train_gan(gan, dataset, batch_size, codings_size, n_epochs=50)
```

- Phase 1
	- Feed Gaussian noise to the generator to produce fake images
	- Concatenate with real images
	- The targets `y1` are set to 0 and 1, fake and real
	- Train the discriminator on this batch
- Phase 2
	- Feed the GAN some Gaussian noise
	- Generator produces fake images to discriminator
	- Improve the generator (want discriminator to fail)
		- Discriminator is not trainable in this part

```python
codings = tf.random.normal(shape=[batch_size, codings_size])
generated_iamges = generator.predict(codings)
```

![[Pasted image 20260304105633.png]]

## The Difficulties of Training GANs

- Nash equilibrium
	- No player would be better off changing their own strategy
- GAN can only reach a single Nash equilibrium
	- Generator produces realistic images, and the discriminator is forced to guess (50% real, 50% fake)
- Mode collapse
	- The generator's outputs gradually become less diverse
	- Generator forgets and produces only one type of class
- Experience relay
	- Storing the images produces by the generator at each iteration in a replay buffer
	- Training the discriminator using real images and fake images from this buffer
- Mini-batch discrimination
	- Measures how similar images are across the batch and provides this statistic to the discriminator

## Deep Convolutional GANs

- Deep convolutional GANs (DCGANs)
	- Replace any pooling layers with strided convolutions and transposed convolutions
	- Use batch normalization
	- Remove fully connected hidden layers for deeper architectures
	- Use ReLU activation in the generator for all layers except the output layer, use tanh instead
	- Use leaky ReLU activation in the discriminator for all layers


```python
coding_size = 100

generator = tf.keras.Sequential([
	tf.keras.layers.Dense(7 * 7 * 128),
	tf.keras.Reshape([7, 7, 128]),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2,
				padding="same", activation="relu"),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2,
				padding="same", activation="tanh"),
])

discriminator = tf.keras.Sequential([
	tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
				activation=tf.keras.layers.LeakyReLU(0.2)),
	tf.keras.layers.Dropout(0.4),
	tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
				activation=tf.keras.layers.LeakyReLU(0.2)),
	tf.keras.layers.Dropout(0.4),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1, activation="sigmoid")
])
gan = tf.keras.Sequential([generator, discriminator])

X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1.
```

- Before training the GAN, rescale the training set to the same range
- Build the dataset then compile and train this model

![[Pasted image 20260304114923.png]]

- Codings that were used to generate the images were averages, and an image was generated based on the resulting mean codings
- Mean computed in the latent space

![[Pasted image 20260304115059.png]]

- Add each image's class as an extra input to both the generator and the discriminator
	- Conditional GAN (CGAN)

## Progressive Growing of GANs

- Generating small images at the beginning of training, then gradually adding convolutional layers to both the generator and the discriminator to produce large images
	- Greedy-layer-wise training of stacked AE
- Extra layer is added at the end of generator and beginning of discriminator
- Fade-in/fade-out technique is used when a new convolutional layer is added to the discriminator

![[Pasted image 20260304115328.png]]


- Other techniques for stable training
	- Mini-batch standard deviation layer
		- Added near the end of the discriminator
		- Computes the std across all channels and all instance in the back
	- Equalized learning rate
		- Initializes all weights using GD with mean 0 and std 1, rather than He
		- Weights are scaled down at runtime by the same factor as in He
	- Pixelwise normalization layer
		- Added after each convolutional layer in the generator
		- Normalizes each activation based on all the activations in the same image and at the same location, but across all channels
		- Avoids explosions in the activations due to excessive competition

- Measure similarity between local image structure of the generated images and the training images, at every scale

## StyleGANs

- SoTA in high-resolution image generation
- Style transfer techniques in the generator to ensure that the generated images have the same local structure as the training images, at every scale

2 Networks
- Mapping network
	- 8 layer MLP that maps the latent representations, $Z$ to a vector $w$
	- Vector goes through multiple affine transformations
	- Maps the codings to multiple style vectors
- Synthesis network
	- Generates the images
	- Constant learned input (after training)
	- Multiple convolutional and upsampling layers
	- Noise is added to the input and to all the output of the convolutional layers (before activation)
	- Each noise layer is followed by an adaptive instance normalization (AdaIN)
		- Standardizes each feature map independently, then uses the stuye vector to determine the scale and offset of each feature map

![[Pasted image 20260304120116.png]]

- GAN is able to use the provided noise to add the right amount of stochasticity to each part of the image
- Each noise input consists of a single feature map full of Gaussian noise, which broadcast to all feature map and scaled using learned per-feature scaling factors ("B" boxes)
- StyleGAN uses mixing regularization where a percentage of the generated images are produces using 2 different codings

$c_1$ and $c_2$ => mapping network => $w_1$ and $w_2$ 

- Synthesis network generates an image based on the styles $w_1$ for the first levels and $w_2$ for the remaining levels

# Diffusion Models

- Tools from thermodynamics to model a diffusion process
- Train a model to learn the reverse process
	- Start from a mixed state, then unmix
- Denoising diffusion probabilistic model (DDPM)
	- Long time compared to GANs or VAE

DDPM
- Start with an image, $x_0$
- At each time step t add a bit of Gaussian noise to the image, with mean 0 and variance $\beta_t$
- Noise is independent for each pixel (isotropic)
- Obtain, $x_t$, until image is hidden by noise
- Last step, $T$, where the image is drowned in noise (forward process)
- Pixel values get rescaled at each step by $\sqrt{1-\beta_t}$

- Probability distribution q of the forward diffusion process

![[Pasted image 20260304121018.png]]

- Shortcut the forward process
- Sample an image $x_t$ given $x_0$ without computing $x_1, x_2, ...$
- Sum of multiple Gaussian distributions is also a Gaussian distribution
	- Noise is added in one shot

- Shortcut for the forward diffusion process

![[Pasted image 20260304121141.png]]

- Create many new images, to perform revers process, $x_t$ to $x_t-1$

![[Pasted image 20260304121224.png]]

- Variance schedule equations for the forward diffusion process

![[Pasted image 20260304121307.png]]

```python
def variance_schedule(T, s=0.008, max_beta=0.999):
	t = np.arrange(T + 1)
	f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
	alpha = np.clip(f[1:] / f[:-1], 1 - max_beta, 1)
	alpha = np.append(1, alpha).astype(np.float32)
	beta = 1 - alpha
	alpha_cumprod = np.cumprod(alpha)
	return alpha, alpha_cumprod, beta
	
T = 4000
alpha, alpha_cumprod, beta = variance_schedule(T)

# create noisy images
def prepare_batch(X):
	X = tf.caast(X[..., tf.newaxis], tf.float32) * 2 - 1
	X_shape = tf.shape(X)
	t = tf.random.uniform([X_shape[0]], minval=1, maxval=T + 1, dtype=tf.int32)
	alpha_cm = tf.gather(alpha_cumprod, t)
	alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * (len(X_shape) - 1))
	noise = tf.random.normal(X_shape)
	return {
		"X_noise:" alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise,
			"time": t,
	}, noise
```

- Add a channel axis
- `t` create a vector containing a random time step for each image, between 1 and T
- Use `tf.gather()` to get the value of `alpha_cumprod` for each of the time steps int he vector
- Reshape `alpha_cm`
- Generate Gaussian noise with mean 0 and variance 1
- Apply the diffusion process to the images
- Model will predict the noise that should be subtracted from the input image to get the original image

```python
# create training dataset and validation
def prepare_dataset(X, batch_size=32, shuffle=False):
	ds = tf.data.Dataset.from_tensor_slices(X)
	if shuffle:
		ds = ds.shuffle(buffer_size=10_000)
	return ds.batch(batch_size).map(prepare_batch).prefetch(1)
	
train_set = prepare_dataset(X_train, batch_size=32, shuffle=True)
valid_set = prepare_dataset(X_valid, batch_size=32)

# build diffusion model
def build_diffusion_model():
	X_noisy = tf.keras.layers.Input(shape=[28, 28, 1], name="X_noisy")
	time_input = tf.keras.layers.Input(shape=[], dtype=tf.int32, name="time")
	[...] # build model
	outputs = [...] # predict the noise
	return tf.keras.Model(inputs=[X_noisy, time_input], outputs=[outputs])
	
model = build_diffision_model()
model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")
history = model.fit(train_set, validation_data=valid_set, epochs=100)
```

- Once the model is trained, you it to generated new image
- Going one step in reverse in the diffusion process

![[Pasted image 20260304122454.png]]

```python
# reverse process, to generate images
def generate(model, batch_size=32):
	X = tf.random.normal([batch_size, 28, 28, 1])
	for t in range(T, 0, -1):
		noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
		X_noise = model({
			"X_noisy": X,
			"time": tf.constant([t] * batch_sze)
		})
		X = (
			1 / alpha[t] ** 5
			* (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
			+ (1 - alpha[t]) ** 0.5 * noise
		)
	return X
	
X_gen = generate(model)
```

![[Pasted image 20260304122731.png]]

- Latent diffusion models
	- Diffusion process takes place in latent space, rather than pixel space
	- AE used to compress each training image into a smaller latent space
- Stable diffusion
