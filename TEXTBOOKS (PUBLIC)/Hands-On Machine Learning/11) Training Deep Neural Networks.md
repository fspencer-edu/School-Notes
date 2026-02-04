
- Issues with DNN
	- Gradients growing smaller or larger, when flowing backward during training
	- Costly to label enough data
	- Training is slow
	- A model with millions of parameters is at risk of overfitting the training set


# The Vanishing/Exploding Gradients Problems

- Backpropagation algorithm's second phase works by going from the output layer to input layer, propagating the error gradient
- Uses these gradients to update each parameter with a gradient descent step
- GD updates leaves the lower-layers' connection weight unchanged, and training never converges to a good solution
	- Vanishing gradients
- In some case the opposite happens
	- Gradients grow larger until layer

## Glorot and He Initialization
## Better Activation Functions
## Batch Normalization
## Gradient Clipping

# Reusing Pretrained Layers

## Transfer Learning with Keras
## Unsupervised Pretraining
## Pretraining on an Auxiliary Task

# Faster Optimizers

## Momentum
## Nesterov Accelerated Gradient
## AdaGrad
## RMSProp
## Adam
## AdaMax
## Nadam
## AdamW
# Learning Rate Scheduling

# Avoiding Overfitting Through Regularization

## $\ell_1$ and $\ell_2$ Regularization

## Dropout
## Monte Carlo (MC) Dropout
## Max-Norm Regularization

# Summary and Practical Guidelines