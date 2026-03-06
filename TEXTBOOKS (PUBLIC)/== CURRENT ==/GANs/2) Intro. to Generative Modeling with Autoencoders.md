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
- 
- Anomaly detection

**Self-Super**

## New Take on an old idea

## Generation using an autoencoder

## Variational autoencoder

# Code if Life

# Why Did We Try aGAN?

## Summary