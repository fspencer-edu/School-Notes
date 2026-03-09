
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
- Make sture that the translation from A to B is real, and also that the translation from our estimated 


# Identity loss

# Architecture

## CycleGAN Architecture: Building the network
## Generator architecture
## Discriminator architecture


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


## Building the network


## Building the generator

```python
def build_generator(self):

	def conv2d
```

## Building the discriminator
## Training the CycleGAN

## Running the CycleGAN


# Expansions, augmentations, and applications

## Augmented CycleGAN
## Application
