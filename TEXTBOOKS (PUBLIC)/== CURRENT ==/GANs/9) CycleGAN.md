
- CycleGANs
	- ML models designed for image-to-image translation without requiring paired training data

# Image-to-image translation

- Image-to-image translation
	- Mapping an image from one domain to another
- Previously, the latent vector seeding the generation was an uninterpretable vector
- Conditioning on a complete image, of the same dimensionality as the output image, that is then provided to the network as a label

![[Pasted image 20260309150921.png]]

- Mapping
	- Semantic labels to photorealistic images
	- Satellite images to street view
	- Images from day to night
	- Back-and-white to colour
	- Outlines to synthesized fashion items

# Cycle-consistency loss: There and back aGAN

 - Complete the cycle
	 - Translate from one domain to another and back again

$(a)$ = original picture
$(\hat{a})$ = reconstructed picture

- Ideally the original and reconstructed picture is the same
- Measure their loss on a pixel level, with cycle-consistency loss

![[Pasted image 20260309151254.png]]

- Back-translation
	- Measure the cycle-consistency loss by how much the first and third sentences differ
- 2 Generators
	- One translating from A to B => $G_{AB}$ called $G$
	- One translating from B to A => $G_{BA}$ called$ F$
- 2 losses
	- Forward cycle-consistency loss
		- $\hat{a} = F(G(a))=a$
	- Backward cycle-consistency loss
		- $\hat{b} = G(F(a))=b$


# Adversarial loss

- Adversarial loss
	- Every translation with a generator has a corresponding discriminator, $D_A$ and $D_B$
- Make sure that the translation from A to B is real, and also that the translation from our estimated A back to reconstructed B is real
- Mappings
	- A-B-A
	- B-A-B

# Identity loss

- Identity loss
	- Enforce that CycleGAN preserved the overall colour structure (temperature) of the picture
	- Introduce a regularization term that helps us keep the tint of the picture consistent with the original image
- Feed the images already in domain A to the Generator from B to A $(G_{BA})$
- CycleCAN should understand that they are already in the correct contain
	- Penalize unnecessary changes to the image


![[Pasted image 20260309151928.png]]

**Losses**
- Adversarial loss
- Cycle-consistency loss: forward pass
- Cycle-consistency loss: backward pass
- Overall loss
- Identity loss

# Architecture

- CycleGAN builds on the CGAN architecture
	- 2 CGAN joined together
- An input image x and the reconstructed image x* are fed through the latent space z

![[Pasted image 20260309152211.png]]

- In CycleGAN the latent space has equal dimensionality
- CycleGAN needs to find domain B
- The two mappings are two autoencoders
	- $F(G(a))$
	- $G(F(a))$
- Explicit loss function is substituted by the cycle-consistency loss
- The 2 discriminators ensure that both translations look like real images in their respective domains


## CycleGAN Architecture: Building the network

- A-B-A
	- Starts from an image in domain A
- B-A-A
	- Starts from an image in domain B

![[Pasted image 20260309152524.png]]


- Path 1
	- Goes to discriminator
- Path 2
	- Generator translates it to B
	- Evaluated by the discriminator B
	- Translated back to A, measure cyclic loss

- The bottom image is is an off-by-one cycle of the top image


- Generator from A to B
	- Load a real picture from A or translation fro B to A
	- Translate to domain B
	- Create images in domain B
- Generator from B to A
	- Load real picture from B or translation from A to B
	- Translate to domain A
	- Create images in domain A
- Discriminator A
	- Provide a picture in A domain
	- Output probability (update Generator from B to A)
- Discriminator B
	- Provide a picture in B domain
	- Output probability (update Generator from A to B)

## Generator architecture

- U-net architecture

![[Pasted image 20260309153012.png]]


- Use standard convolutional layers to the encoder
- Create skip connections so that information has an easier time propagating through the network
	- Concatenate the entire block to the equivalently colours tensor in the decoder part of the generator
- The decoder uses de-convolutional layers with one final convolutional layer to upscale the image
- Encoder
	- Convolutional layers that reduce the resolutional of the feature map (D0 to D3)
- Decoder
	- De-convolutional layers (transposed convolutions) that upscaled the image (U1 to U4)


- During downsampling we can can focus on classification and undserstanding of large regions
- Higher-resolution skip connections preserves the detail that can then be accurately segmented
- ResNet
	- Fewer parameters
	- Transformer
		- Residual connections in lieu of the encoder-decoder skip connections

## Discriminator architecture

- CycleGANs discriminator is based on the PatchGAN architecture
- Do not get a single float as an output, rather a set of single-channel values, as a set of mini-discriminators, to average
- This helps is scale to higher resolutions


# Object-oriented design of GANs


# CycleGAN


```python
# import
from __future__ import print_function, division
import scipy
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datatime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
```

```python
class CycleGAN():
	def __init__(self):
		self.img_rows = 128
		self.img_cols = 128
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		
		self.dataset_name = 'apple2orange'
		self.data_loader = DataLoader(dataset_name=self.dataset_name,
										img_res(self.img_rows, self.img_cols))
										
		patch = int(self.img_row / 2**4)
		self.disc_patch = (patch, patch, 1)
		
		self.gf = 32
		self.df = 64
		
		self.lambda_cycle = 10.0
		self.lambda_id = 0.9 * self.lambda_cycle
		
		optimizer = Adam(0.0002, 0.5)
		
		# building the networks
		
		self.d_A = self.build_discriminator()
		self.d_B = self.build_discriminator()
		self.d_A.compile(loss='mse',
						 optimizer=optimizer,
						 metrics=['accuracy'])
		self.d_B.compile(loss='mse',
						 optimizer=optimizer,
						 metrics=['accuracy'])
						 
		self.g_AB = self.build_generator()
		self.g_BA = self.build_generator()
		
		img_A = Input(shape=self.img_shape)
		img_B = Input(shape=self.img_shape)
		
		fake_B = self.g_AB(img_A)
		fake_A = self.g_BA(img_B)
		
		reconstr_A = self.g_BA(fake_B)
		reconstr_B = self.g_AB(fake_A)
		
		img_A_id = self.g_BA(img_A)
		img_B_id = self.g_AB(img_B)
		
		self.d_A.trainable = False
		self.d_B.trainable = False
		
		valid_A = self.d_A(fake_A)
		valid_B = self.d_B(fake_B)
		
		self.combined = Model(inputs=[img_A, img_B],
							  outputs=[valid_A, valid_B,
							  reconstr_A, reconstr_B,
							  img_A_id, img_B_id])
		self.combined.compile(loss=['mse', 'mse',
									'mae', 'mae',
									'mae', 'mae'],
									loss_weights=[1, 1,
										self.lambda_cycle,
										self.lambda_id, self.lambda_id],
										optimizer=optimizer)
```

- `lambda_cycle` and `lambda_id`
	- The first hyperparameter controls how strictly the cycle-consistency loss is enforced
		- A higher value will ensure that your original  and reconstructed images are close together
	- The second hyperparameter influences identity loss
	- A lower values leads to unnecessary changes
		- Inverting colours

## Building the network

1. Create two discriminators, $D_A$ and $D_B$
2. Create two generators
	1. Instantiate $G_{AB}$ and $G_{BA}$
	2. Create placeholders for the image input for both directions
	3. Link them both to an image in the other domain
	4. Create placeholders for the reconstructed images back in the original domain
	5. Create the identity loss constraint for both directions
	6. Set discriminators trainable parameter to false
	7. Compile the two generators

- The output from the `combined` model comes in lists of 6
	- The following for A-B-A and B-A-B
		- Validities from discriminator
		- Reconstruction
		- Identities losses

- The first two are squared errors, and the rest are mean absolute errors
- Relative weights are set by `lambda` factors

## Building the generator


- Use skip connections
- U-Net architecture

1. Define the `conv2d()`
	1. Standard 2D convolutional layer
	2. Leaky ReLU activation
	3. Instance normalization
		1. Normalizes each feature map within each channel separately
2. Define `deconv2d()` - transposed conv2d
	1. Upsamples the `input_layer`
	2. Possibly applies dropout
	3. Always applies `InstanceNormalization`
	4. Creates a skip connection between its output layer and layer of corresponding dimensionality from the downsampling
Create generator
3. Take input and assign to `d0`
4. Run through a conv layer `d1`
5. Take `d1` and apply conv2d to get `d2`
6. Take `d2` and apply conv2d to get `d3`
7. Take `d3` and apply conv2d to get `d4`
8. `u1`: upsample `d4` and create a skip connection between `d3` and `u1`
9. `u2`: upsample `u1` and create a skip connection between `d2` and `u2`
10. `u3`: upsample `u2` and create a skip connection between `d1` and `u3`
11. `u4`: use regular upsampling to get a 128 x 128 x 64 image
12. Use a regular 2D conv to remove extra features and output an image


```python
def build_generator(self):
	"""U-Net Generator"""

	def conv2d(layer_input, filters, f_size=4):
		""""Downsampling"""
		d = Conv2D(filters, kernel_size=f_size,
					strides=2, padding='same')(layer_input)
					
		d = LeakyReLU(alpha=0.2)(d)
		d = InstanceNormalization()(d)
		return d
		
	def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
		"""Upsampling"""
		u = UpSampling2D(size=2)(layer_input)
		u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
					 activation='relu')(u)
		if dropout_rate:
			u = Dropout(dropout_rate)(u)
		u = InstanceNormalization()(u)
		u = Concatenate()([u, skip_input])
		return u
		
	d0 = Input(shape=self.img_shape)
	
	d1 = conv2d(d0, self.gf)
	d2 = conv2d(d1, self.gf * 2)
	d3 = conv2d(d2, self.gf * 4)
	d4 = conv2d(d3, self.gf * 8)
	
	u1 = deconv2d(d4, d3, self.gf * 4)
	u2 = deconv2d(u1, d2, self.gf * 2)
	u3 = deconv2d(u2, d1, self.gf)
	
	u4 = UpSampling2D(size=2)(u3)
	output_img = Conv2D(self.channels, kernel_size=4,
				strides=1, padding='same', activation='tanh')(u4)
				
	return Model(d0, output_img)
```

## Building the discriminator

- Use helper functions

1. Take input image and assign to `d1`
2. Take `d1` and assign to `d2`
3. Take `d2` and assign to `d3`
4. Take `d3` and assign to `d4`
5. Take `d4` and flatten by conv2d to 8 x 8 x 1 (average output)

```python
def build_discriminator(self):

	def d_layer(layer_input, filters, f_size=4, normalization=True):
		d = Conv2D(filters, kernel_size=f_size,
				strides=2, padding='same')(layer_input)
		d = LeakyReLU(alpha=0.2)(d)
		if normalization:
			d = InstanceNormalization()(d)
		return d
		
	img = Input(shape=self.img_shape)
	
	d1 = d_layer(img, self.df, normalization=False)
	d2 = d_layer(d1, self.df * 2)
	d3 = d_layer(d2, self.df * 4)
	d4 = d_layer(d3, self.df * 8)
	
	validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
	
	return Model(img, validity)
```


## Training the CycleGAN

- Create the training look

1. Train the discriminator
	1. Take a mini-batch of random images from each domain
	2. Use the $G_{AB}$ to translate $imgs_A$ to domain B and vice versa with $G_{BA}$
	3. $D_A$
		1. Compute $D_A(img_A, 1)$ => losses for real images in A
		2. $D_A(G_{BA}(img_B),0)$ => losses of translated images from B
		3. Add losses, 1 and 0 in $D_A$ serve as labels
	4. $D_B$
		1. Compute $D_B(img_B, 1)$ => losses for real images in A
		2. $D_B(G_{AB}(img_A),0)$ => losses of translated images from B
		3. Add losses, 1 and 0 in $D_B$ serve as labels
	5. Add the losses to get the total discriminator loss
2. Train the generator
	1. Input the images from domain A and B
		1. Output
			1. $D_A(G_{BA}(img_B))$ => validity of A
			2. $D_B(G_{AB}(img_A))$ => validity of B
			3. $G_{BA}(G_{AB}(img_A))$ => reconstructed A
			4. $G_{AB}(G_{BA}(img_B))$ => reconstructed B
			5. $G_{BA}(imgs_A)$ => identity mapping of A
			6. $G_{AB}(imgs_B)$ => identity mapping of B
	2. Update parameters with
		1. MSE for scalars (probabilities)
		2. MAE for images (reconstructed or identity mapped)


```python
def train(self, epochs, batch_size=1, sample_interval=50):

	start_time = datatime.datetime.now()
	
	valid = np.ones((batch_size,) + self.disc_patch)
	fake = np.zeros((batch_size,) + self.disc_patch)
	
	for epoch in range(epochs):
		for batch_i, (imgs_A, imgs_B) in enumerate(
			self.data_loader.load_batch(batch_size)):
			
			fake_B = self.g_AB.predict(imgs_A)
			fake_A = self.g_BA.predict(imgs_B)
			
			dA_loss_read = self.d_A.train_on_batch(imgs_A, valid)
			dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
			dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
			
			dB_loss_read = self.d_B.train_on_batch(imgs_B, valid)
			dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
			dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
			
			d_loss = 0.5 * np.add(dA_loss, dB_loss)
			
			g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
												  [valid, valid,
												  imgs_A, imgs_B,
												  imgs_A, imgs_B])
												  
			if batch_i % sample_interval == 0:
				self.sample_images(epoch, batch_i)
```

## Running the CycleGAN

```python
gan = CycleGAN()
gan.train(epochs=100, batch_size=64, sample_interval=10)
```

![[Pasted image 20260309155645.png]]


# Expansions, augmentations, and applications

## Augmented CycleGAN

- Learning Many-to-Many mappings from unpaired data

![[Pasted image 20260309155740.png]]

## Application


- Cycle Consistent Adversarial Domain Adaptation (CyCADA)

![[Pasted image 20260309155833.png]]