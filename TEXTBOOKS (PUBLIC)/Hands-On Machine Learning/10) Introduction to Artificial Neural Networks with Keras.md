
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

<img src="/images/Pasted image 20260203152724.png" alt="image" width="500">

- `MLPClassifier` is similar to `MLPRegressor`
	- Minimizes the cross entropy rather than the MSE

<img src="/images/Pasted image 20260203152812.png" alt="image" width="500">


# Implementing MLPs with Keras

- Keras is TensorFlow's high-level deep learning API
- Build, train, evaluate, and execute NN
- Keras library was developed by Francois Chollet
- Other popular deep learning libraries include
	- PyTorch by Facebook
	- JAX by Google

- Colab runtimes come with recent version of TF and Keras preinstalled


## Building an Image Classifier Using the Sequential API

- Load a dataset
- Fashion MNIST, images represent fashion items rather than handwritten digits

### Using Keras to load the dataset

```python
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000], y_train_full[-5000]
```

- Every image is represented as a 28 x 28 array rather than a 1D array
- Pixel intensities are represented as integers rather than float

```python
X_train.shape
(55000, 28, 28)
X_train.dtype
dtype('uint8')

# Scale intentities down to the 0-1 range
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# list of class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
               
class_names[y_train[0]]
'Ankle boot'
```

<img src="/images/Pasted image 20260203153548.png" alt="image" width="500">

### Creating the model using the sequential API

```python
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28,28]))
model.add(tf.keras.layers.Flattern())
model.add(tf.keras.layesr.Dense(300, activation="relu"))
model.add(tf.keras.layesr.Dense(100, activation="relu"))
model.add(tf.keras.layesr.Dense(10, activation="softmax"))
```
- Set TF random seed to make the results reproducible
- `Sequential` creates a model for NN that are composed of a single stack of layers connected sequentially
- Build the first layer `Input` and add to model
	- Shape of instances
- `Flatten` layer
	- Converts each input image into a 1D array
	- `[32, 28, 28]` -> `[32, 784]`
- `Dense` hidden layer with 300 neurons
	- Use ReLU
	- Manages its own weight matrix, and vector of bias terms
- Second `Dense` hidden layer
- Final `Dense` hidden layer with softmax


- Instead of adding the layers one by one, pass a list of layers when creating a sequential model
- Drop the input layer, and specify input shape

```python
nodel = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape[28, 28]),
	tf.keras.layesr.Dense(300, activation="relu"),
	tf.keras.layesr.Dense(100, activation="relu"),
	tf.keras.layesr.Dense(10, activation="softmax"),
])
```

- `summary()` displays all the models layers
	- Name
	- Output shape
	- Number of parameters

<img src="/images/Pasted image 20260203154241.png" alt="image" width="500">

- `Dense` layers have a lot of parameters
- First hidden layer is 784 x 300 connection weights, plus 300 bias term; 235500 parameters
- Each model must have a unique name
- Keras converts layer's class name into snake case
- Ensures that name is globally unique
- All global state managed by Keras is stored in a Keras session
	- `tf.keras.backend.clear_session()`


```python
model.layers
[<keras.layers.reshaping.flatten.Flatten at 0x17380e9b0>,
 <keras.layers.core.dense.Dense at 0x1776211b0>,
 <keras.layers.core.dense.Dense at 0x177622410>,
 <keras.layers.core.dense.Dense at 0x176e78c40>]
 hidden1 = model.layers[1]
 hidden.name
 'dense'
 model.get_layer('dense') is hidden1
 True
 
 # access models weights and bias
>>> weights, biases = hidden1.get_weights()
>>> weights
array([[ 5.3297073e-02,  2.4198458e-02, -2.1023259e-02, ...,  4.6089381e-02],
       [ 2.2632368e-02,  5.9892908e-03,  1.4587238e-02, ...,  2.4750374e-02],
       ...,
       [-4.4557646e-02, -5.9672445e-02,  6.5973431e-02, ...,  5.1353276e-02],
       [-1.4996272e-02,  1.0063291e-02, -3.2075007e-02, ..., -6.4764827e-02]],
       dtype=float32)
>>> weights.shape
(784, 300)
>>> biases
array([0., 0., 0., 0., 0., 0., 0., 0., 0., ...,  0., 0., 0.], dtype=float32)
>>> biases.shape
(300,)
```

- `Dense` layer initialized the connection weights randomly and bias terms are set to zero
- Kernel is another name for the matrix of connection weights
- Shape of weight matrix depends on the number of inputs

### Compiling the model

- After a model is created, `compile()` method is used to specify the loss function and the optimizer to use

```python
model.compile(loss="sparse_categorical_crossentropy",
	optimizer="sgd",
	metrics=["accuracy"])
```

- `"sparse_categorical_crossentropy"` is used for loss because we have sparse labels
- Use one-hot vectors to represent class 3 => `[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]`
- Use `sigmoid` activation for binary or multilabel binary classification
- Convert sparse labels to one-hot vector labels, use `tf.keras.utils.to_categorical()`
- Optimizer, `sgd`, means that the model will train using stochastic gradient descent
	- Backpropagation
- Tune learning rate
	- `optimizer=tf.keras.optimizers.SGD(learning_rate=__??__)`
	- Default to 0.01
- Measure its accuracy during training and evaluation, `metrics=["accuracy"]`

### Training and evaluating the model

- Call `fit()` to train model

```python
history = model.fit(X_train, y_train, epochs=30,
		validation_data=(X_valid, y_valid))
Epoch 1/30
1719/1719 [==============================] - 3s 2ms/step
  - loss: 0.7239 - sparse_categorical_accuracy: 0.7616
  - val_loss: 0.5026 - val_sparse_categorical_accuracy: 0.8298
Epoch 2/30
1719/1719 [==============================] - 3s 2ms/step
  - loss: 0.4890 - sparse_categorical_accuracy: 0.8306
  - val_loss: 0.4506 - val_sparse_categorical_accuracy: 0.8360
[...]
Epoch 30/30
1719/1719 [==============================] - 6s 3ms/step
  - loss: 0.2246 - sparse_categorical_accuracy: 0.9204
  - val_loss: 0.3021 - val_sparse_categorical_accuracy: 0.8904
```
- Pass input features `X_train` and target classes `y_train`, and number of epochs
- Pass a validation set
- Keras will measure the loss and extra metrics on this set at the end of each epoch
- If performance on the training set is better than on the validation set, model is overfitting
- Shape errors are common
	- Remove with `Flatten` layer
	- `loss="categorical_crossentropy"`

- Neural network is trained
- At each epoch during training, Keras displays the number of mini-batches processed
	- 1,719 batches per epoch
	- Mean training time per sample
	- Loss and accuracy on training and validation set
- Training loss when down
- Validation accuracy reached 89.04% after 30 epochs

- Instead of passing a validation set, use a `validation_split=0.1` to the ratio of the training set to use for validation
- If training is skewed, set `class_weight` when calling `fit()`
	- Give a larger weight to under-represented classes, and lower to over-r classes
	- Give more weights to instances labeled by experts compared to open source
- `fit()` returns a `History` object containing
	- Training parameters (`history.params`)
	- Epoch
	- Dictionary with loss and extra metrics

```python
import matplotlib.pyplot as plt
import pandas as pd
pd.DataFrame(history.history).plot(
	figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], gird=True, xlabel="Epoch",
	style=["r--", "r--.", "b-", "b-*"])
plt.show()
```

<img src="/images/Pasted image 20260203170239.png" alt="image" width="500">

- Both training and validation accuracy increase during training
- Training and validation loss decrease
- Validation curves get further apart overtime, show overfitting
- Validation error is computed at the end of each epoch
- Training error is computer using a running mean during each epoch
	- Therefore, the training curve should be shifted by half an epoch to the left
	- Training set performance ends up beating the validation
- Calling the `fit()` method again, continues training, to reach max accuracy

- Optimize model performance
	- Check learning rate
	- Try another optimizer
	- Tube model hyperparameters
		- Number of layers
		- Number of neurons per layer
		- Types of activation functions
	- Batch size

- Evaluate performance on the test set to estimate the generalization error before deployment
- `evaluate()`

```python
model.evaluate(X_test, y_test)
313/313 [==============================] - 1s 626us/step
  - loss: 0.3213 - sparse_categorical_accuracy: 0.8858
[0.3213411867618561, 0.8858000040054321]
```

- Common to get lower performance on test set than validation set
- Hyperparameters are tuned on the validation set
- Resist temptation to tweak the hyperparameters on the test set, or else your estimate of the generalization error will be too optimistic

### Using the model to make predictions

- Make predictions on the new instances

```python
X_new = X_test[L3]
y_proba = model.predict(X_new)
y_proba.round(2)
array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.02, 0.  , 0.97],
       [0.  , 0.  , 0.99, 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]],
      dtype=float32)
```

- For each instance the model estimates one probability per class, 0-9
- First image estimates that the probability of class 9 (ankle boot) is 96%
- Use `argmax()` to get the highest probability class index for each instance

```python
import numpy as np
y_pred = y_proba.argmax(axis=-1)
y_pred
array([9, 2, 1])
np.array(class_names)[y_pred]
array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')

y_new = y_test[:3]
y_new
array([9, 2, 1], dtype=uint8)
```

<img src="/images/Pasted image 20260203171152.png" alt="image" width="500">


## Building a Regression MLP Using the Sequential API

- Using sequential API is similar to classification
- Output layer has a single neuron and it uses no activation function
- Loss function is the RMSE
- Use `Normalization`, same as Scikit-Learn `StandardScaler`, but fitted to the training data

```python
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequational([
	norm_layer,
	tf.keras.layers.Dense(50, activation="relu"),
	tf.keras.layers.Dense(50, activation="relu"),
	tf.keras.layers.Dense(50, activation="relu"),
	tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizer.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredERror"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
			validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```

- `Normalization` layer learns the feature means and standard deviations in the training data at `adapt()`
	- Parameters are not affected by gradient descent

- `Sequational` model are common, useful to build NN with ore complex topologies, with multiple inputs or outputs

## Building Complex Models Using the Functional API

- A non-sequential NN is a wide and deep neural network (2016)
- Connects all or part of the inputs directly to the output layer
- Makes is possible for the NN to learn both deep patterns and simple rules (short path)
- Regular MLP forces all the data to flow through the full stack of layers
	- Simple patterns in data are distorted by sequence transformations

<img src="/images/Pasted image 20260203171949.png" alt="image" width="500">

```python
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[inputs_], outputs=[output])
```

- At a high level, the first 5 lines create all the layers
- The next 6 lines use the layers like functions to form the input to the output
- `Model` points to the input and the output

- `Normalization` standardize the inputs
- `Dense` layers with 30 neurons, using ReLU activation function
- `Concatenate` layer
- `Input` object specifies the shape and type, default 32-bit floats
- Functional API
	- Pass input object to normalization layer
	- No actual data is being processed
- Use `concat_layer` to concatenate the input and second hidden layer
- Pass `concat` to `output_layer` for final output
- `Model` specifies the inputs and outputs

- To send a subset of features through the wide path a a different subset through the deep path

```python
input_wide = tf.keras.layers.Input(shape=[5]) # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6]) # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden1])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[outputs])
```

<img src="/images/Pasted image 20260203173332.png" alt="image" width="500">


- Each `Dense` layer is created and called on the same line
	- Cannot do this with `Normalization` layer since we need to reference layers to call `adapt()` method before
- Use concat which created a layer and calls it with inputs
- Specified `inputs` when creating the model, since there are two inputs

- Compile the model, but when calling `fit()`, instead of passing a single input matrix `X_train`, pass a pair of matrices `(X_train_wide, X_train_deep)`, one per input

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquredRoot"])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]

X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(X_train_wide, X_train_deep), y_train, epoch=20,
		validation_data=((X_valid_wide, X_valid_deep), y_valid)
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))
```

- Instead of passing a tuple `(X_train_wide, X_train_deep)` use a dictionary
	- `{"input_wide": X_train_wide, "input_deep": X_train_deep}`

- Cases to have multiple outputs
	- Locate and classify the main object in the picture
		- Regression and classification tasks
- Have multiple independent tasks based on the same data
	- Train a single NN with one output per task
	- NN can learn features in the data that are useful across tasks
	- Multitask classification on pictures of faces
- Regularization technique
	- Training constraint to reduce overfitting
	- Add auxiliary output in a neural network architecture to ensure underlying part of the network learn something on its own


<img src="/images/Pasted image 20260203174225.png" alt="image" width="500">

- Add an extra output requires connection to the models's list of outputs

```python
output = tf.keras.layers.Dense(1)(concat)
aux_output = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=[input_wide, input_deep],
			outputs=[outputm aux_output])
```

- Each output needs its own loads function
- Compile the model, to pass a list of losses

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1), optimizer=optimizer,
			metrics=["RootMeanSquaredError"])
```

- Instead of passing `loss=("mse", "mse")`, use dictionary `loss={"ouput":"mse", "aux_output":"mse"}`
- Provide labels for each output
- The main output and auxiliary output should try to predict the same thing
	- Use the same labels

```python
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
	(X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
	validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid))
)
```

- Keras returns the weighted sum of the losses, as well as all individual losses and metrics

```python
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
```

- Set `return_dict=True` for a dictionary instead of a tuple

```python
y_pred_main, y_pred_aux = model.predict((X_new_wide, y_new_deep))

# create dictonary from predict()
y_pred_tuple = model.predict((X_new_wide, X_new_deep))
y_pred = dict(zip(model.output_names, y_pred_tuple))
```

## Using the Subclassing API to Build Dynamic Models

- Sequential and functional API are declarative
	- Start by declaring layers to use then how they should be connected
	- Model is easily saved, cloned, and shared
	- Structure can be displayed and analyzed
	- Framework can infer shapes and check types
	- Errors can be caught early
- Model is a static graph of layers
- Imperative programming style is preferred for
	- Loops
	- Varying shapes
	- Conditional branching
	- Dynamic behaviours

- Use subclass `Model`, to create the layers you need in the constructor, and use them to perform the computations you want in the `call()`
- Creating an instance of the following `WideAndDeepModel` gives an equivalent model to functional API

```python
class WideAndDeepModel(tf.keras.Model):
	def __init__(self, units=30, activation="relu", **kwards):
		super().__init__(**kwargs)
		self.norm_layer_wide = tf.keras.layers.Normalization()
		self.norm_layer_deep = tf.keras.layers.Normalization()
		self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
		self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
		self.main_output = tf.keras.layers.Dense(1)
		self.aux_output = tf.keras.layers.Dense(1)
		
	def call(self, inputs):
		input_wide, input_deep = inputs
		norm_wide = self.norm_layer_wide(input_wide)
		norm_deep = self.norm_layer_deep(input_deep)
		hidden1 = self.hidden1(norm_deep)
		hidden2 = self.hidden2(hidden1)
		concat = tf.keras.layers.concatenate([norm_wide, hidden2])
		output = self.main_output(concat)
		aux_output = self.aux_output(hidden2)
		return output, aux_output
		
model = WideAndDeepModel(30, activation="relu", name="my_cool_omdel")
```

- Separate the creation of the layers in the constructor from their usage in `call()`
- Do not create the `Input` object, use the `input` argument to `call()` method
- Compile the model instance, adapt its normalization layers, fit it, evaluate it, and use it to make predictions
- API can include anything, but
	- Model's architecture is hidden within the `call()`, so Keras cannot inspect it
	- Model cannot be cloned
	- A list of layers is returned with `summary()`, without information
	- Keras cannot check types and shapes ahead of time
- Use sequential or functional API

- Keras models can be used like regular layers, combine to build complex architectures

## Saving and Restoring a Model

```python
# saving model
model.save("my_keras>model", save_format="tf")
```

- Keras saves the model using TF save model format
	- Directory containing files and subdirectories
	- `saved_model.pb` file contains the model's architecture and logic in the form of a serialized computation graph
	- Do not deploy the model's source code for production
- `keras_metadata.pb` file contains extra information
- `/variables` subdirectory contains all the parameter values
	- Weights, biases, normalization statistics, optimizer's parameters
- `/assets`, contains extra files
	- Data samples
	- Feature names
	- Class banes

- `save_format="h5"`, saves the model to a single file using Keras-specific format based on HDF5 format


- Have a script that trains a model and saves it, and one more script (web services) that load the model and use t to evaluate or make predictions

```python
# loading model
model = tf.keras.models.load_model("my_keras_model")
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
```

- `save_weights()` and `load_weights()` to save and load parameter values
	- Saved in an index file, `.index`
- Save checkpoints regularly in case a computer crashes

## Using Callbacks

- `fit()` method accepts a `callbacks` argument that specify a list of objects that Keras will call before and after training, before and after each epoch, and before and after processing each batch
- `ModelCheckpoint` callback saves checkpoints of the model at regular intervals during training

```python
# checkpoint
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints",
			save_weights_only=True)
history = model.fit([...], callbacks=[checkpoint_cb])
```

- Set `save_best_only=True` to save model's best performance
- `EarlyStopping` callback is used to stop when there is no progress

```python
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
					restore_best_weights=True)
history = model.fit([...], callbacks=[checkpoint_cb, early_stopping_cb])
```

- Custom callback

```python
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		ratio = logs["val_loss"] / logs["loss"]
		print(f"Epoch={epoch}, val/train={ratio:.2f}")
```

- Callbacks can be used during evaluation and prediction

## Using TensorBoard for Visualization

- TensorBoard is a interactive visualization tool to view learning curves during training, metrics, analyze training statistics, view image generated by model, visualize complex multi-dimensional data projected, and clusters

```python
pip install -q -U tensorboard-plugin-profile
```

- Modify program so that it outputs the data to a binary logfiles called event files
- Each binary data record is called a summary
- Server will monitor the log directory, and automatically pick up the changes and update

```python
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
	return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")
	
run_logdir = get_run_logdir()  # e.g., my_logs/run_2022_08_01_17_25_59

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,
				profile_batch=(100, 200))
history = model.fit([...], callbacks=[tensorboard_cb])
```

- Code will profile the network between batch 100 and 200 during the first epoch

<img src="/images/Pasted image 20260204082507.png" alt="image" width="500">

- One directory per run, each containing one subdirectory for training logs, and validation logs
	- Both files contain training logs, and profile traces
- Start a TensorBoard server for a directory, which connects to the server and displays the user interface

```python
load_ext tensorboard
tensorboard --logdir=./my_logs
```

- Use SCALARS tab to view the learning curves

<img src="/images/Pasted image 20260204082743.png" alt="image" width="500">

- TensorFlow offers a lower-level API in the `tf.sumary` package
- To create `SummaryWriter` using `create_file_writer()` function
	- Uses writer as a Python context to log scalars, histograms, images, audio, text


```python
test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)

        data = (np.random.randn(100) + 2) * step / 100  # gets larger
        tf.summary.histogram("my_hist", data, buckets=50, step=step)

        images = np.random.rand(2, 32, 32, 3) * step / 1000  # gets brighter
        tf.summary.image("my_images", images, step=step)

        texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
        tf.summary.text("my_text", texts, step=step)

        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)
```

- Share results at `https://tensorboard.dev`
- Run `!tensorboard dev upload --logdir ./my_logs` to upload results


# Fine-Tuning Neural Network Hyperparameters

- There are many hyperparameters to tweak in NN
	- Number of layers
	- Number of neurons
	- Types of activation functions
	- Weight initialization logic
	- Type of optimizer
	- Learning rate
	- Batch size
- Convert model to Scikit-Learn estimator, and then use `GridSearchCV` or `RandomizedSearchCV` to find tune the hyperparameters
- `KerasRegressor` or `KerasClassifier` wrapper
- Use the Keras Tuner library, which is a hyperparameter tuning library for Keras models

```python
pip install -q -U keras-tuner
```

- Write function that builds, compiles, and returns a Keras model
	- Takes `kt.HyperParameters` objects and range of possible values
- Te following function builds and compiles an MLP to classify Fashion MNIST images

```python
import keras_tuner as kt

def build_model(hp):
	n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
	n_neurons = hp.Int("n_neurons", min_vlue=16, max_value=256)
	learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
						sampling="log")
	optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
	if optimizer = "sgd":
		optimizer = tf.keras.optimizers.SGD(learning_rate=_learing_rate)
	else
		optimizer = tf.kears.optimizers.Adam(learning_rate=learning_rate)
		
	model = tf.keras.Sequential()
	model.add(tf.kears.layers.Fatten())
	for _ in range(n_hidden):
		model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
		model.compile(loss="sparse_categorical_crossentropy",
							optimizer=optimizer, metrics=["accuracy"])
	return model
```

- Sets a hyperparameter value from 0 to 8, and returns default value 2
- Learning rate of all scales will be sampled equally
- Second part of the function builds the model using the hyperparameter values
- Creates a sequential with a flatten layer, followed by requested number of hidden layers, using the ReLU activation function, and an output layer with 10 neurons
- Function compiles the model and returns it

- Create a `kt.RandomSearch` tuners, passing the `build_model` to the constructor

```python
random_search_tuner = kt.RandomSearch(
	build_model, objective="val_accuracy", max_trials=5, overwrite=True,
	directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_serach_tuner.search(X_train, y_train, epochs=10,
		validation_data=(X_valid, y_valid))
```

- `RandomSearch` tuner calls `build_model()` once with an empty `HyperParameters` objects, to get specification
- Then runs 5 trials
	- For each trial, use random hyperparameters from respective ranges
	- Trains the model for 10 epochs and saves

```python
top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
top3_params[0].values
{'n_hidden': 7,
 'n_neurons': 100,
 'learning_rate': 0.0012482904754698163,
 'optimizer': 'sgd'}
```

- Each tuner is guided by a oracle
- Find best oracle

```python
best_trial = random_search_tuner.oracle.get_best_trial(num_models=1)[0]
best_trial.summary()
Trial 1 summary
Hyperparameters:
n_hidden: 7
n_neurons: 100
learning_rate: 0.0012482904754698163
optimizer: sgd
Score: 0.8596000075340271

best_trial.metrics.get_last_value("val_accuracy")
0.8596000075340271

# continue training on full set, evaluate, and deploy
best_model.fit(X_train_full, y_train_full, epochs=10)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
```

- Fine-tune pre-processing hyperparameters
	- Batch size
- Use `kt.HyperModel`
	- `build()`
	- `fit()`


- Following code builds the same model as before, with the same hyperparameters, but also uses a Boolean `normalize` to control training

```python
class MyClassificationHyperModel(kt.HyperModel):
	def build(self, hp):
		return build_model(hp)
		
	def fit(self, hp, model, X, y, **kwargs):
		if hp.Boolean("normalize"):
			norm_layer = tf.keras.layers.Normalization()
			X = norm_layer(X)
		return model.fit(X, y, **kwargs)
```

- Pass an instance of this class to tuner

```python
hyperband_tuner = kt.Hyperband(
	MyClassificationHyperModel(), objective="val_accuracy", seed=42,
	max_epochs=10, factor=3, hyperband_iterations=2,
	overwrite=True, directory="my_fashion_mnist", project_name="hyperband"
)
```

- Trains different models for a few epochs, then eliminates the worst, and keeps only 1/factor models
- Repeats until a single model is left

```python
root_logdir = Path(hyperband_tuner.project_dir)/"tensorboard"
tensorbarod_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(X_train, y_train, epochs=10,
	validation_data=(X_valid, y_valid),
	callbacks=[early_stopping_cb, tensorboard_cb])
```

- HPARAMS
	- Contains a summary of all the hyperparameters combinations that were tried and metrics
	- Table view
	- Parallel coordinates view
	- Scatterplot matrix view
- Hyperband is smarter than pure random search in the way it allocates resources
- Keras Tuner also includes `kt.BayesianOptimization` tuner
	- Gradually learns which regions of the hyperparameter space are most promising by fitting a probabilistic models, called Gaussian process
	- Zoom to best hyperparameters
	- Has its own hyperparameters, `alpha` = noise, `beta` = algorithm to explore

```python
bayestian_opt_tuner = kt.BayesianOptimization(
	MyClassificationHyperModel(), objective="val_accuracy", seed=42,
	max_trials=10, alpha=1e-4, beta=2.6,
	overwrite=True, directory="my_fashion_mnist", project_name="bayestion_opt"
	bayesian_opt_tuner.search([...])
)
```

- AutoML
	- Refers to any system that takes care of a large part of the ML workflow

## Number of Hidden Layers

- For complex problems, deep network have a much higher parameter efficiency than shallow one
- A deep neural network automatically takes advantage that data is hierarchical
- Kickstart the training by reusing the lower layers of a network
	- Transfer learning

## Number of Neurons per Hidden Layer

- The number of neurons in the input and output layers is determined by the type of input and output of the task
- Using the same number of neurons in all hidden layers performs just as well in most cases, or even better
	- Only one hyperparameter to tune, instead of per layer
- Build a model with more layers and neurons that needed, then use early stopping and other regularization techniques
	- "Stretch pants"
	- Avoid bottleneck layers

## Learning rate, Batch Size, and Other Hyperparameters

**Other Hyperparameters**

- Learning rate
	- The optimal learning rate will be lower than the point at which the loss starts to climb
	- Reinitialize model and train it using the optimal learning rate
- Optimizer
- Batch size
	- Use largest batch size that can fit in GPU RAM
		- Learning rate warmup
	- Small batch size
- Activation function
	- ReLU for hidden layer
	- Output layer depends
- Number of iterations

- 