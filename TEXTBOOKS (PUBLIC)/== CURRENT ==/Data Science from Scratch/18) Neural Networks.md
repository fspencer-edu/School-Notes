
- Artificial neural network (ANN)
	- Predictive model motivated by the way the brain operates

# Perceptions

- Simplest NN
- Approximates a single neuron with n binary inputs
- Computes a weighted sum of its inputs and "fires" if that weighted sum is 0 or greater

```python
from scratch.linear_algebra import Vector, dot

def step_function(x: float) -> float:
	return 1.0 if x >= 0 else 0.0
	
def perceptron_output(weights: Vector, bias: float, x: Vector):
	calculation = dot(weights, x) + biax
	return step_function(calculation)
```
- With specific weights, perceptions can solve a number of simple problems
- Gates
	- AND
	- OR
	- XOR
	- NOT

# Feed-Forward Neural Networks

- Feed-forward NN
	- Consists of discrete layers of neurons, each connected to the next
	- Input layer
	- hidden layers
	- Output layers
- each neuron has a weight corresponding to each of its inputs and a bias
- Add the bias to the end of the weights vector
- Give each neuron a bias input that always equals 1

```python
import math

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))
```
<img src="/images/Pasted image 20260429170509.png" alt="image" width="500">


```python
from typing import List

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]              # Add a constant.
        output = [neuron_output(neuron, input_with_bias)  # Compute the output
                  for neuron in layer]                    # for each neuron.
        outputs.append(output)                            # Add to results.

        # Then the input to the next layer is the output of this one
        input_vector = output

    return outputs
```

 - For a given input, the hidden layer produces a 2D vector consisting of the AND of the two input values and OR of the two output values
 - The result is a network that performs XOR

<img src="/images/Pasted image 20260429170616.png" alt="image" width="500">


# Backpropagation


- Backpropagation
	- Uses gradient descent
	- Used to train nn
	- Adjusts the weights for optimal output

1. Run `feed_forward` on an input vector to produce the outputs of all the numerous in the network
2. Compute the loss of SSE
3. Compute gradient loss as a function of the output neuron's weights
4. Propagate the gradients and errors backward to compute the gradients with respect to the hidden neuron's weights
5. Take a gradient descent step


# Example: Fizz Buzz

# For Further Exploration