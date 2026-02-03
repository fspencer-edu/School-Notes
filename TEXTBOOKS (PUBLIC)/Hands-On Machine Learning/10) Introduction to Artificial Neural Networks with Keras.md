
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

<img src="/images/Pasted image 20260203143253.png" alt="image" width="500">

- Architecture of biological neural networks (BNNs)
- Neurons are often organized in consecutive layers, especially in the cerebral cortex

<img src="/images/Pasted image 20260203143400.png" alt="image" width="500">


## Logical Computations with Neurons

- McCulloch and Pitts proposed a simple model
- Each layer is the artificial neuron
	- One or more binary inputs and one binary output
- Can compute any logical proposition

<img src="/images/Pasted image 20260203143524.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260203144005.png" alt="image" width="500">

- Most common step function is the Heaviside step function
- Sign function is also used

**Common Step Functions Used in Perceptron (threshold = 0)**

<img src="/images/Pasted image 20260203144051.png" alt="image" width="500">

- A single TLU can be used for simple linear binary classification
- A perceptron is composed of one or more TLUs organized in a single layer
	- Every TLU is connected to every input
	- Fully connected layer, or a dense layer

<img src="/images/Pasted image 20260203144215.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260203144734.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260203145407.png" alt="image" width="500">

- Some limitations can be eliminated by stacking multiple perceptrons
- Multilayer perceptron (MLP)
	- Can solve XOR
- Perceptrons do not output a class probability
- Perceptrons do not use any regularization by default, and training stops as soon as there are not more prediction errors on the training set
- Perceptrons may train a bit faster

## The Multiplayer Perceptron and Backpropagation

- An MLP is composed of one input layer, one or more layers of TLU called hidden layers, and one final layer called the output layer
- Layers closed to the input is are the lower layers, and then upper layers

<img src="/images/Pasted image 20260203145612.png" alt="image" width="500">


- Signal flow in one direction
- Feedforward neural network (FNN)
- When ANN contains a deep stack of hidden layers, it is called a deep neural network (DNN)
- Research discussed the possibility of using gradient descent to train NN (1960)
- Reverse-mode automatic differentiation (1970)
	- In 2 passes through a network, it is able to compute the gradients of the NN;s error with regard to every single model parameter
- Combination of reverse-mode autodiff and gradient descent is called backpropagation (backprop)
- Reverse-mode autodiff is suited when the function to differentiate has many variables and few outputs
- Backpropagation can be applied to computational graphs

**Backpropagation**
- Handles one mini-batch at a time, and goes through the full training set multiple times
	- Epoch
- Each mini-batch enters the network through the input layer
	- Computes the output of all the neurons in the first hidden layer
	- Result is passed to the next layer, and so on
	- Forward pass
		- All intermediate results are preserved for backward pass
- Algorithm measures the network's output error
- Computes how much each output bias and each connection to the output layer contributed to the error
	- Chain rule
- Algorithm them measure error contributions from connection in layer below, until it reaches input layer
	- Measure the error gradient across all the connection weights and biases
- Algorithm performs a gradient descent step to tweak all the connection weights in the network, using the error gradients

- Initialize the hidden layers' connection weights randomly
- Break the symmetry and allow backpropagation to train a diverse team of neurons

- Replaced the step function in MLP with the logistic function $\alpha(z) = 1 / (1 + exp(-z))$
	- Sigmoid function
- Step function contains only flat segments
	- Ranges from 0 to 1

**Backpropagation on other activation functions**

Hyperbolic tangent function: $tanh(z) = 2\sigma(2z)-1$
- Activation function is S-shaped, continuous, and differentiable
- Output value ranges from -1 to 1
- Each layer's output is more or less centred around at the beginning of training, helps speed of convergence

<img src="/images/Pasted image 20260203151054.png" alt="image" width="500">

The rectified linear unit function: $ReLU(z) = max(0, z)$
- Continuous but not differentiable at $z=0$
- Fast to compute, default
- Does not have a max output value
	- reduce issues during gradient descent


<img src="/images/Pasted image 20260203151039.png" alt="image" width="500">

- If you chain several linear transformations, the result is a linear transformation
- Deep stack of linear layers, is a complex problem
- A large enough DNN with non-linear activations can theoretically approximate any continuous function


<img src="/images/Pasted image 20260203151009.png" alt="image" width="500">



## Regression MLPs

- MLP can be used for regression tasks
- For multivariate regression, you need one output neuron per output dimension
	- Locate the centre of an object in an image
	- Predict 2D coordinates
	- Place a bounding box around the object (4 output neurons)
- `MLPRegression` is used to build an MLP with 3 hidden layers composed of 50 neurons each

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics improt root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
	housing.data, housing.taget, random_state=42)
x_train, X_valud, y_train, y_valid = train_test_split(
	X_train_full, y_train_full, random_state=42)
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = nake_pipeline(StandardScaler(), mlp_reg)
y_pred = pipeline.predict(X_valid)
rmse = root_mean_squared_error(y_valid, y_pred)  # about 0.505
```

- The result is RMSE of ~0.505

- Gradient descent does not converge well when features have large scales
- Code training the model and evaluates its validation error
- Model uses ReLU activation function, and it uses a variant of gradient descent called Adam to minimize the mean squared error
- MLP does not use any activation function from the output layer
- To guarantee that the output will always be positive, use ReLU activation function in the output layer, or the softplus activation function
	- Smooth variant of ReLU: $softplus(z) = log(1 +exp(z))$
		- Close to 0 when z is negative
		- Close to z when z is positive
- Neural net features are limited in Scikit-Learn
- Use mean absolute error (MAE) if there are many outliers
- Huber loss
	- Combination of both
		- Quadratic when the error is smaller than a threshold, linear when error is larger


<img src="/images/Pasted image 20260203152340.png" alt="image" width="500">
## Classification MLPs

- For binary classification, use a single output neuron using the sigmoid activation function
	- Output between 0 and 1
- MLPs can also handle multilabel binary classification tasks
- 2 output neurons, both using the sigmoid activation function
	- Output the probability that the email is spam, second would output the probability that it is urgent
- Model outputs any combination of labels
- Use softmax activation function for the whole output layer
	- Probabilities are between 0 and 1, add up to 1
	- Classes are exclusive
- Cross-entropy loss (x-entropy or log loss) is a good choice for a loss function

![[Pasted image 20260203152724.png]]

- `MLPClassifier` is similar to `MLPRegressor`
	- Minimizes the cross entropy rather than the MSE

![[Pasted image 20260203152812.png]]


# Implementing MLPs with Keras

- Keras is TensorFlow's high-level deep learning API
- Build, train, evaluate, and execute NN
- Keras library was developed by Francois Chollet


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

