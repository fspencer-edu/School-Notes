- Conditional GAN (CGAN) uses labels to train both the generator and discriminator

# Motivation

# What is Conditional GAN

- CGAN is a generative adversarial network whose generator and discriminator are conditioned during training using additional information
	- Class labels
	- Tags
	- Written descriptions

- Generator learns to produce examples for each label in the training dataset
- Discriminator learns to distinguish fake example-labels pairs from real example-label pairs
- Discriminator in CGAN does not learn to identify which class the samples are from
	- Learn to reject all image-label pairs in which the image if fake, even if label matches

## CGAN Generator

<img src="/images/Pasted image 20260309120323.png" alt="image" width="500">

$z$ = noise vector
$y$ = conditional label
$G(z, y) = x*|y$ => x* conditioned on y

## CGAN Discriminator
- The discriminator receives real examples (x, y) and fake examples (x*| y, y)
- Discriminator outputs a single probability indicating its conviction that the input is a real, matching pair

<img src="/images/Pasted image 20260309120536.png" alt="image" width="500">

## Summary Table
## Architecture diagram

<img src="/images/Pasted image 20260309120549.png" alt="image" width="500">

- For each fake example, the same label y is passed to both the generator and discriminator
- Discriminator is never explictily trained to reject mismatched pairs by being trained on real examples with mismatching labeles
	- Its ability to identify mismatches pairs is a by-product of being trained to accept only real matching pairs
- Discriminator
	- Receives fake labeled examples from generator
	- Real labeled examples

# Implementing a Conditional GAN

- Implement a CGAN model to generate specific hand-written digits
## Implementation

## Setup

```python
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import (
	Activation, BatchNormalization, Concatenate, Dense,
	Embedding, Flatten, Input, Multiply, Reshape
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
```

```python
# input dim
img_rows = 28
img_cols = 28
channels = 1

imgs_shape = (img_rows, img_cols, channels)

z_dim = 100
num_classes = 10
```

## CGAN Generator

- Take label y (from integer 0 to 9) and turn it into a dense vector of size `z_dim`
- Combine the label embedding with the noise vector z into a joint representation with `Multiply`
- Feed the resulting vector as input into the rest of the CGAN generator network to synthesize image


<img src="/images/Pasted image 20260309121118.png" alt="image" width="500">

```python
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
	
	model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
	
	model.add(Activation('tanh'))
	
	return model
	
def build_cgan_generator(z_dim):
	z = Input(shape=(z_dim, ))
	
	label = Input(shape(1, ), dtype='int32')
	
	label_embedding = Embedding(num_classes, z_dim, input_length)
	
	label_embedding = Flatten()(label_embedding)
	
	joined_representation = Multipl()([z, label_embedding])
	
	generator = build_generator(z_dim)
	
	conditioned_img = generator(joined_representation)
	
	return Model([z, label], conditioned_img)
```

## CGAN Discriminator

- Take a label (0-9) and label turn label into a dense vector 28 x 28 x 1 (flattened image)
- Reshape the label embeddings into the image dimensions (28 x 28 x 1)
- Concatenate the reshaped label embedding onto the corresponding image, creating a joint representation
- Feed the image-label joint representation as input into the GGAN discriminator

<img src="/images/Pasted image 20260309121546.png" alt="image" width="500">
- Adjust the model inputs dimensions to (28 x 28 x 2) to reflect the new input shape
- Increase the depth of the first convolutional layer form 32 to 64
- Use sigmoid activation function as the output layer

```python
def build_discriminator(img_shape):
	model = Sequential()
	
	model.add(
		Conv2D(64,
			  kernel_size=3,
			  strides=2,
			  input_shape(img_shape[0], img_shape[1], img_shape[2] + 1,
			  padding='same'))
	
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(
		Conv2D(64,
			  kernel_size=3,
			  strides=2,
			  input_shape=img_shape,
			  padding='same'))
			  
	model.add(BatchNormalization())
	
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(
		Conv2D(128,
			  kernel_size=3,
			  strides=2,
			  input_shape=img_shape,
			  padding='same'))
	
	model.add(BatchNormalization())
	
	model.add(LeakyReLU(alpha=0.01))
	
	model.add(Flatten())
	mode.add(Dense(1, activation='sigmoid'))
	
def build_cgan_discriminator(img_shape):
	img = Input(shape=img_shape)
	
	label = Input(shape=(1, ), dtype='int32')
	
	label_embedding = Embedding(num_classes,
								np.prod(img_shape),
								input_length=1)(label)
								
	label_embedding = Flatten()(label_embedding)
	label_embedding = Reshape(img_shape)(label_embedding)
	concatenated = Concatenate(axis=-1)([img, label_embedding])
	discriminator = build_discriminator(img_shape)
	classification = discriminator(concatenated)
	return Model([img, label], classificaiton)
```

## Building the model

```python
def build_gan(generator, discriminator):
	z = Input(shape=(z_dim, ))
	
	label = Input(shape=(1, ))
	
	img = generator([z, label])
	
	classification = discriminator([img, label])
	
	model = Model([z, label], classification)
	
	return model
	
discriminator = build_cgan_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
					  optimizer=Adam(),
					  metrics=['accuracy'])
					  
generator = build_cgan_generator(z_dim)

discriminator.trainable = False

cgan = build_cgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer=Adam())
```

## Training

- Train the discriminator
	- Take a random mini-batch of real examples and labels
	- Computes D((x, y)) for the mini batch and backpropagate the binary classification loss
	- Take mini-batch of random noise vectors and class labels (z, y) and generate fake examples G(z, y)
	- Computes D((x*|y, y))
- Train the generatir
	- Take a mini-batch of random noise vectors and class labels (Z, y), generate fake examples
	- Computes D(x*|y, y) to max. loss
```python
accuracies = []
losses = []

def train(iterations, batch_size, sample_intervals):
	(X_train, y_train), (_, _) = mnist.load_data()
	
	X_train = X_train / 127.5 - 1.
	X_train = np.expand_dims(X_train, axis=3)
	
	real = np.ones((batch_size, 1))
	
	fake = np.zeros((batch_size, 1))
	
	for iteration in range(iteration):
	
		idx = np.random.randint(0, X_train.shape[0], batch_size)
		imgs, labels = X_train[idx], y_train[idx]
		
		z = np.random.normal(0, 1, (batch_size, z_dim))
		gen_imgs = generator.predict([z, labels])
		
		d_loss_real = discriminator.train_on_batch([imgs, labels], real)
		d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
		
		z = np.random.normal(0, 1, (batch_size, z_dim))
		
		labels = np.random.randint(0, num_classes, batcH_size).reshape(-1, 1)
		
		g_loss = cgan.train_on_batch([z, labels], real)
		
		if (iteration + 1) % sample_interval == 0:
			
			print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %         
                  (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss))
                  
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])
            
	        sample_images()
```

## Outputting sample images

- Instead of a 4 x 4 grid of random digits
- Generate 2 x 5 grid of numbers
	- 1 to 5
	- 6 to 9

```python
def sample_images(image_grid_rows=2, image_grid_columns=5):
	z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
	labels = np.arrange(0, 10).reshape(-1, 1)
	
	gen_imgs = generator.predict([z, labels])
	
	gen_imgs = 0.5 * gen_imgs + 0.5
	
	fig, axs = plt.subplots(image_grid_rows,
							image_grid_columns,
							figsize=(10, 4),
							sharey=True,
							sharex=True)
							
	cnt = 0
	for i in range(image_grid_rows):
		for j in range(image_grid_columns):
			axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
			axs[i, j].axis('off')
			axs[i, j].set_title("Digit: %d" % labels[cnt])
			cnt += 1
			
```

<img src="/images/Pasted image 20260309150358.png" alt="image" width="500">

<img src="/images/Pasted image 20260309150408.png" alt="image" width="500">

## Training the model

```python
iteration = 12000
batch_size = 32
sample_interval = 1000
train(iterations, batch_size, sample_interval)
```
## Inspecting the output: Targeted data generation

<img src="/images/Pasted image 20260309150436.png" alt="image" width="500">


# Conclusion

- pix2pix
	- Uses pairs of images to learn to translate from one domain into another
- Scenarios
	- Colourization tasks