- Classification
	- Assigning the correct category to an example
- Regression
	- Estimating a numerical value based on a variety of inputs

# What are Generative Adversarial Networks?

- Generative Adversarial Networks (GANs)
	- 2 simultaneously trained models
		- Generator
		- Discriminator
- Generative
	- Creating new data
- Adversarial
	- Competitive dynamic between the two models
- Neural networks
	- The class of machine learning models most commonly used to represent the generator and discriminator
- Types of NN
	- Feed-forward
	- Convolutional
	- U-net

# How to GANs Work?

- Goal of generator
	- Produce examples that capture the characteristics of the training dataset, that they are indistinguishable from the training data
	- Object recognition model in reverse
	- Learns from feedback from discriminator's classifications
- Goal of discriminator
	- Determine whether a particular example is real of fake

- Object recognition algorithms
	- Learn patterns in images to discern an image's content

# GANs in Action

![[Pasted image 20260306174407.png]]

- Training dataset, $x$
- Random noise vector, $z$
- Generator network, $x^*$
- Discriminator network
- Iterative training/tuning
	- D weights and biases are updated to max. its classification accuracy
	- G weights and biases are updated to max. the probability that the D misclassifies $x^*$ as real

## GAN Training

- 


## Reaching Equilibrium

# Why Study GANs

