- Deep convolutional GAN (DCGAN)
	- Uses CNNs instead of feed-forward network layers

# Convolutional Neural Networks

## Convolutional filters

- Feed-forward NN
	- Neurons are flat, fully connected layers
- CNN
	- Arranged in 3D
	- Sliding one or more filters over the input layer
	- Small receptive field, but extends the entire depth
- Each filter outputs a single activation value
	- Dot produce between the input value and filter entries

## Parameter sharing

- Filter parameters are shared by all the input values
- Efficiently learn visual features and shapes regardless of where they are located
- Parameter sharing reduces the number of trainable parameters
	- Scale up

## ConvNets visualized

- 

# Brief History of the DCGAN

# Batch Normalization

## Understanding normalization

## Computing batch normalization

# Generating handwritten digits with DCGAN

## Importing modules and specifying model input dimensions

## Implementing the generator

## Implementing the discriminator

## Building and running the DCGAN

## Model output

# Conclusion