
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
8. `u1`: upsample 


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
