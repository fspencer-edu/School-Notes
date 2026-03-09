
- GAN models
	- Semi-supervised GAN
	- Conditional GAN
	- CycleCAN

# Introducing the Semi-Supervised GAN

- Semi-supervised learning trains model using a small amount of labeled data combined with a large volume of unlabeled data

## What is a Semi-Supervised GAN?

- SGAN is a GAN whose discriminator is a multiclass classifier
- Learns to distinguish between N + classes, where N is the number of classes in the training dataset, which one added for the fake generated samples

![[Pasted image 20260309104332.png]]

- Generator takes in a random noise vector z, and produces fake examples, $x*$
- Discriminator receives 3 types of inputs
	- Fake data
	- Real unlabeled data
	- Real labeled examples (x, y)
- Discriminator outputs a classification
	- To identify fake examples
	- Identify the correct class for real examples

## Architecture

- The SGAN generator is the same
- The discriminator
	- Receives 3 inputs
	- Multi classification
## Training process

- 
## Training objective

# Implementing a Semi-Supervised GAN

## Architecture diagram
## Implementation
## Setup
## The dataset

## The Generator
## The Discriminator
## Building the model
## Training

# Comparison to a Fully Supervised Classifier

# Conclusion