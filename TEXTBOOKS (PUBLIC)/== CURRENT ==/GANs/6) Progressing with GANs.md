
- Progressive GAN
	- Progressively growing and smoothing fading in higher-resolution layesr
	- Mini-batch standard deviation
	- Equalized learning rate
	- Pixel-wise feature normalization

**2 Examples**
- Progressive GANs
- Pretrained model using TFHub

- ML automation
	- Cloud AutoML
	- Amazon SageMake
	- PyTorch Hub

# Latent space interpolation

- Latent space
	- A compressed, lower-dimensional representation of data within ML model where similar items are mapped close together
- Interpolation
	- The process of estimating unknown values that fall within the range of a known, discrete dataset

# They grow up so fast

- Challenges with GANs
	- Mode collapse and lack of convergence

## Progressive growing and smoothing of higher-resolution layers

- Stochastic gradient descent
- Progressive growing
	- Increasing the resolution of the terrain as we go
- Progressively smooth in and slowly introduce complexity
	- Low-resolution convolutional layers to high-resolution

![[Pasted image 20260309094442.png]]

![[Pasted image 20260309094515.png]]

- When the 16 x 16 resolution has trained enough, a transposed convolution in the Generator is introduced
- Another convolution in the discriminator is added to get 32 x 32

## Example implementation
## Mini-batch standard deviation
## Equalized learning rate

## Pixel-wise feature normalization in the generator

# Summary of key innovations

# TensorFlow Hub and Hands-On

# Practical Applications