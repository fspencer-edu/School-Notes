
- Artificial neural networks (ANNs)
	- Machine learning models inspired by the networks of biological neurons found in our brains
- ANNs are the core of deep learning
- TensorFlow's Keras API
	- Building
	- Training
	- Evaluating
	- Running neural networks

# From Biological to Artificial Neurons

- ANNs were first introduced in 1943 by neurophysiologist Warren McCulloch and Walter Pitts
- Simplified computational model of how biological neurons might work to perform complex computations using propositional logic
- Connectionism (1980s), the study of neural networks
- Support vector machines (1990s)

Renewed ANNs
- Huge amount of data available to train neural networks
- ANNs outperform other ML techniques on large and complex problems
- Increase in computing power
	- Moore's law
	- GPU
	- Cloud platforms
- Training algorithms have been improved
- Theoretical limitations of ANNs have turned out to be benign in practice
	- Stuck in local optima
- Virtuous cycle of funding and progress

## Biological Neurons

- Composed of a cell body containing the nucleus and most of the cell's complex components
- Dendrites
	- Branching extensions
- Axon
	- Long extensions
	- Axon splits off into many branches called telodendria
	- Top of branches are minuscule structures called synaptic terminals (synapses)
- Neurons produce short electrical impulses called action potentials (AP) or signals
- Signals travel along the axions and make the synapses release chemical signals called neurotransmitter
- Fires electrical impulses

![[Pasted image 20260203143253.png]]

- Architecture of biological neural networks (BNNs)
- Neurons are often organized in consecutive layers, especially in the cerebral cortex

![[Pasted image 20260203143400.png]]


## Logical Computations with Neurons

- McCulloch and Pitts proposed a simple model
- Each layer is the artificial neuron
	- One or more binary inputs and one binary output
- Can compute any logical proposition

![[Pasted image 20260203143524.png]]

1) Identity function
	1) If A is activated, then C gets activated
2) Logical AND
	1) C is activated only when both neurons are activated
3) Logical OR
	1) C is activated if either A or B is activated
4) Logical NOT
	1) Neuron C is activated when B is off


## The Perceptron

- The perceptron is the simplest ANN architectures (1957)
- Based on different artificial neuron called a threshold logic unit (TLU) or linear threshold unit (LTU)
- Inputs and outputs are numbers and each input connection is associated with a weight
- TLU first computes a linear function

$z = w_1x_1 + w_2x_2 + ... + b = w^Tx + b$

- Applied a step function

$h_w(x) = step(z)$

- Similar to logistic regression, except it uses a step function instead of logistic
- Model parameters are the input weights $w$ and the bias term, $b$

![[Pasted image 20260203144005.png]]

- Most common step function is the Heaviside step function
- Sign function is also used

**Common Step Functions Used in Perceptron (threshold = 0)**

![[Pasted image 20260203144051.png]]

- A single TLU can be used for simple linear binary classification
- A perceptron is composed of one or more TLUs organized in a single layer
	- Every TLU is connected to every input
	- Fully connected layer, or a dense layer

![[Pasted image 20260203144215.png]]

- Classify instances simultaneously into 3 different binary classes, makes it a multilabel classifier

**Computing the outputs of a fully connected layer**

$\hat{Y} = \phi(XW+b)$

$\hat{Y}$ = output matrix
$X$ = matrix of input features
$W$ = weight matrix
$b$ = bias vector
$\phi$ = activation functions

- In mathematics, the sum of a matrix and a vector is undefined
- In data science, "broadcasting" is used to add vector to a matrix
	- Adds every row in the matrix
- When a biological neuron triggers another neuron, the connection between these two neurons grows stronger
- Connection weights between two neurons tends to increase when they fire simultaneously
	- Hebbian learning
- Perceptrons are trained using a variant of this rule
- Learning rule reinforces connections that help reduce the error
- For every output neuron that produced a wrong prediction, it reinforces the connection weights from the inputs that would have contributed to the correct prediction

**Perception Learning Rule (Weight Update)**

![[Pasted image 20260203144734.png]]

$w_{i,j}$ = connection weight
$x_i$ =input value of current training instance
$\hat{y}_j$ = output of the jth output neuron
${y}_j$ = target output
$\eta$ = learning rate

- The decision boundary of each output neuron is linear
- Incapable of complex patterns
- If training instances are linearly separable, algorithms converge to a solution
	- Perceptron convergence theorem

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)
per_clf = Perception(random_state=42)
per_clf.fit(X, y)
X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)  # predicts True and False for these 2 flowers
```

- Perception learning strongly resemble stochastic gradient descent
- `Perception` is equivalent to `SDGClassifier` with
	- `loss="Perceptron", learning_rate="cosntant", eta0=1, penalty=None`
- Weaknesses of perceptrons
	- Incapable of solving trivial problems
		- XOR classification problems

![[Pasted image 20260203145407.png]]

- Some limitations can be eliminated by stacking multiple perceptrons
- Multilayer perceptron (MLP)
	- Can solve XOR
- Perceptrons do not output a class probability
- Perceptrons do not use any regularization by default, and training stops as soon as there are not more prediction errors on the training set
- Perceptrons may train a bit faster

## The Multiplayer Perceptron and Backpropagation

- An MLP is composed of one input layer, one or more layers of TLU called hidden layers, and one final layer called the output layer
- Layers closed to the input is are the lower layers, and then upper layers

![[Pasted image 20260203145612.png]]


- Signal flow in one direction
- Feedforward neural network (FNN)
- When ANN contains a deep stack of hidden layers, it is called a deep neural network (DNN)
- Research discussed the possibility of using gradient descent to train NN (1960)
- Reverse-mode automatic 

## Regression MLPs
## Classification MLPs

# Implementing MLPs with Keras

## Building an Image Classifier Using the Sequential API
## Building a Regression MLP Using the Sequential API
## Building Complex Models Using the Functional API
## Using the Subclassing API to Build Dynamic Models
## Saving and Restoring a Model
## Using Callbacks

## Using TensorBoard for Visualization

# Fine-Tuning Neural Network Hyperparameters

## Number of Hidden Layers
## Number of Neurons per Hidden Layer
## Learning rate, Batch Size, and Other Hyperparameters

