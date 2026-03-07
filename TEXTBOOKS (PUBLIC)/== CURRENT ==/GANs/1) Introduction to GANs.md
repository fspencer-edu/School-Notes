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

<img src="/images/Pasted image 20260306174407.png" alt="image" width="500">

- Training dataset, $x$
- Random noise vector, $z$
- Generator network, $x^*$
- Discriminator network
- Iterative training/tuning
	- D weights and biases are updated to max. its classification accuracy
	- G weights and biases are updated to max. the probability that the D misclassifies $x^*$ as real

## GAN Training

**GAN Training Algorithm**
1) Train the discriminator
	1) Take a random real example from the training dataset
	2) Get a new random noise vector
	3) Use generator to synthesis, $x^*$
	4) Use discriminator to classify $x$ and $x^*$
	5) Compute classification errors and backpropagate the total error
		1) Updates the discriminators trainable parameters
2) Train the generator
	1) Get a new random noise vector
		1) Synthesis $x^*$
	2) Use the discriminator to classify $x^*$
	3) Compute the classification error and backpropagate the error to update the generators trainable parameters

**GAN Training Visualized**

<img src="/images/Pasted image 20260306174911.png" alt="image" width="500">


## Reaching Equilibrium

- Zero-sum game
	- Situation in which one player's gains equal the other player's losses
- Nash equilibrium
	- A point at which neither player can improve their situation or payoff by changing their actions

**Nash Equilibrium**
- Generator produces fake examples that are indistinguishable from the real data in the training dataset
- Discriminator can at best randomly guess whether an example is real or fake

- With equilibrium achieved, the GAN has converged
- Nearly impossible to find the Nash equilibrium for GANs
	- Complexities

# Why Study GANs

- Applications
	- Hyper-realistic imagery
	- Image-to-image translation
	- Recommendations
	- Diagnostic accuracy
	- Artificial general intelligence

