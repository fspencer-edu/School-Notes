
# Foundations of GANs: Adversarial Training

- Generator and discriminator are represented by differentiable functions
	- NN each with its own cost function
- The two networks are trained by backpropagation using the discriminator's loss

<img src="/images/Pasted image 20260306192022.png" alt="image" width="500">

- Generator's goal is to produce examples that capture the data distribution of the training dataset
- Object recognition models learn the patterns in images to discern an image's content

## Cost functions

$J^{(G)}$ = Generator's cost function
$J^{(D)}$ = Discriminator's cost function

- The trainable parameters

Weight
$\theta^{(G)}$
$\theta^{(D)}$

- Cost function of a traditional NN is defined exclusively in terms of its own trainable parameters, $J(\theta)$
- GANs consists of two networks whose cost functions are dependent on both the networks' parameters

Generator's cost function = $J^{G}(\theta^{(G)}\theta^{(D)})$
Discriminator's cost function = $J^{D}(\theta^{(G)}\theta^{(D)})$

- Traditional networks can tune all its parameters during the training process
- In GAN, each network can tune only its own weights and biases

## Training Process

- Training of a traditional NN is an optimization problem

<img src="/images/Pasted image 20260306192703.png" alt="image" width="500">

- GAN training can be described as a game, rather than optimization
- GAN training ends when two network reach Nash equilibrium
	- Generator cost function is minimized 
	- Discriminator cost function is minimized 

<img src="/images/Pasted image 20260306192842.png" alt="image" width="500">

- High-dimensional, non-convex training of GANs
- Training GANs successfully requires trails and error


# The Generator and the Discriminator

$G(z) = x^*$
$D(x)$ => 1
$D(x*)$ => 0


<img src="/images/Pasted image 20260306193055.png" alt="image" width="500">

## Conflicting objectives

- The generator's goal is opposite of the discriminators

$G(x)$ => 0
$G(x*)$ => 1

## Confusion matrix

- Discriminator's classifications can be expressed in terms of a confusion matrix

TP, $D(x) =1$
FN, $D(x) =0$
TN, $D(x*) =0$
FP, $D(x*) =1$

- Discriminator tries to max. TP, and TN
- Minimize FP and FN

# GAN Training Algorithm

1) Train the discriminator
2) Train the generator

- In step 1, the generator's parameters are kept intact while the discriminator is training
- In step 2, the discriminator's parameters are fixes

- Update only the weights and biases of the network being trained
- Each network gets relevant signals about the updates to make, without interferences

# Generating Handwritten Digits

<img src="/images/Pasted image 20260306193550.png" alt="image" width="500">

- Create a GAN that learns to produce realistic handwriting digits

## Importing modules and specifying model input dimensions

```python
# import
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

from keras.datasts import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

# input dim.
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)
z_dim = 100
```

## Implementing the generator

- The generator is a NN with only a single hidden layer (leaky ReLU, small positive gradient)
- Takes z as input, and produces a 28 x 28 x 1 image
- Output layer uses the tanh activation function, `[-1, 1]`

```python
# generator
def build_generator(img_shape, )
```

## Implementing the discriminator




## Building the model

## Training

## Outputting sample images

## Running the model

## Inspecting the results


# Conclusion