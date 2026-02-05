
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
	- Gradients grow larger until layer causes algorithm to diverge
		- Exploding gradients
- Sigmoid activation function and initialization scheme (normal distribution) causes the variance of the output of each layer to be greater than the variance of its inputs

<img src="/images/Pasted image 20260204093152.png" alt="image" width="500">

- When the inputs become large, the function saturates at 0 or 1, with a derivative close to 0
- When backpropagation kicks in it has no gradient to propagate through and is diluted

## Glorot and He Initialization

- Need signal to flow properly in both direction
- Need the variance of the outputs of each layer to be equal to the variance of its inputs
- Fan-in and fan-out of the layers
	- Input and output are equal
- Connection weights of each layer must be initialized randomly

$fan_{avg} = (fan_{in} + fan_{out})$

- Xavier initialization or Glorot initialization

**Glorot Initialization**

<img src="/images/Pasted image 20260204093548.png" alt="image" width="500">

- Using Glorot initialization can seep up training

<img src="/images/Pasted image 20260204093700.png" alt="image" width="500">

- Keras uses Glorot initialization by default with a uniform distribution

```python
import tensorflow as tf

dense = keras.layers.Dense(50, activation="relu", kernel_initializer="he_normal")

he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg",
			distribution="uniform")
dense = tf.keras.layers.Dense(50, activation="sigmoid",
			kernel_initializer=he_avg_init)
```

## Better Activation Functions

- Unstable gradients were cause by a poor choice of activation function
- AF are better in DNN, such as ReLU, because it does not saturate for positive values
	- Dying ReLU
		- During training, some neurons "die", stop outputting other than 0
		- A neuron dies when its weights get tweaked in a way that the input of the ReLU function is negative for all instances in the training set
- Leaky ReLU to solve

### Leaky ReLU

$LeakyReLU_a(z) = max(az, z)$

- $a$ defines how much the function "leaks"
- Slope of the function for z < 0
- Having the slope z < 0, ensure that leaky ReLU never die
- Outperform strick ReLU
- Randomized leaky ReLU (RReLU)
- Parametric leaky ReLU (PReLU)
	- $a$ is authorized to be learning during training
	- Outperforms ReLU on large datasets, but overfitting on smaller

<img src="/images/Pasted image 20260204094434.png" alt="image" width="500">

- Keras has `LeakyReLY, PReLU`

```python
leaky_reul = tf.keras.layers.LeakyReLU(alpha=0.2)
dense = tf.keras.layers.Dense(50, activation=leaky_relu,
			kernal_intializer="he_normal")
```

- ReLU, leaky ReLU, and PReLU are all non smooth functions
- The discontinuity cases GD to bounce around the optimum, and slow down convergence

### ELU and SELU

- Exponential linear unit (ELU)
	- Outperformed all the ReLU variances

**ELU Activation Function**

<img src="/images/Pasted image 20260204094741.png" alt="image" width="500">

- When z < 0, allows unit to have an average output closer to 0, and helps alleviate the vanishing
- Non-zero gradient for z < 0, avoid dead neuron problem
- When $a=1$, function is smooth
- Slower to compute
- Raster convergence rate during training

<img src="/images/Pasted image 20260204094924.png" alt="image" width="500">

- Scaled ELU (SELU)
	- Scaled variant of the ELU
- If all hidden layers use the SELU, then the network will self-normalize
- The output of each layer will tend to preserve a mean of 0, and standard deviation of 1
- Input feature must be standardized
- Hidden layer weights must be initialized using LeCun normal initialization
- Self-normalizing property is only guaranteed with plain MLPs
- Cannot use regularization techniques

### GELU, Switch, and Mish

- GELU
	- Smooth variant of the ReLU activation functions
	- $\phi$ is the standard Gaussian cumulative distribution function (CDF)

**GELU Activation Function**

<img src="/images/Pasted image 20260204095310.png" alt="image" width="500">

- Neither convex or monotonic
- Curvature at every point
	- Easier for GD to find complex patterns
- Outperforms every activation function so far
- More computationally intensive

<img src="/images/Pasted image 20260204095446.png" alt="image" width="500">

- Sigmoid linear unit (SiLU)
	- Also called Switch
- Mish
	- Smooth, non-convex, and non-monotonic variant of ReLU

- Switch is better default for more complex tasks

## Batch Normalization

- Batch normalization (BN) that addresses vanishing/exploding gradients
- Adding an operation in the model just before or after the activation function
- Zero-centres and normalizes each input, the scales and shifts the result using two new parameter vectors per layer
	- Scaling
	- Shifting
- Algorithm needs to estimate each input's mean and standard deviation
- Evaluates the mean and standard deviation of the input over the current mini-bactch

**Batch normalization algorithm**

<img src="/images/Pasted image 20260204095958.png" alt="image" width="500">

<img src="/images/Pasted image 20260204100007.png" alt="image" width="500">

- BN standardizes the inputs, rescales and offsets them
- Estimates the final statistics during training using a moving average
	- $\gamma$ = output scale vector
	- $\beta$ = output offset vector
	- $\micro$ = final input mean vector
	- $\sigma$ = final standard deviation vector

- BN improved all the DNN
- Less sensitive to weight initialization
- Acts as a regularizer, reducing the need for regularization techniques
- Slower predictions
- Fuse the BN with the previous layer after training, avoiding the runtime penalty

### Implementing batch normalization with Keras

- Add BN layer before or after each hidden layer
- Add BN layer as the first layer

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

<img src="/images/Pasted image 20260204100807.png" alt="image" width="500">

- Each BN layer add four parameters per input, 4 x 784 = 3136
- Last parameters are moving average, and not affected by backpropagation
	- "Non-trainable"
- BN params: $(3136 + 1200 + 400) / 2 = 2368$


```python
>>> [(var.name, var.trainable) for var in model.layers[1].variables]
[('batch_normalization/gamma:0', True),
 ('batch_normalization/beta:0', True),
 ('batch_normalization/moving_mean:0', False),
 ('batch_normalization/moving_variance:0', False)]
```

- Add BN layers before activation function, rather than after
- remove the activation functions from the hidden layers and add them as separate layers after the BN layers
- remove the bias term from previous layer
- Drop the first BN layer to avoid sandwiching the first hidden layer between two BN layers

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

- Tweak `momentum`
	- Updates the exponential moving average

<img src="/images/Pasted image 20260204101258.png" alt="image" width="500">

- `axis`
	- Determines which axis should be normalized
	- Default normalize the last axis

- BN is used in convolutional neural networks

## Gradient Clipping

- Another technique to mitigate the exploding gradients is to clip the gradients during backpropagation to never exceed some threshold
	- Gradient clipping
- Used in recurrent NN, where using BN is difficult
`
```python
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile([...], optimizer=optimizer)
```

- Clip every component of the gradient vector to a value between -1.0 and 1.0
- All the partial derivates of the loss will be clipped
- May change the orientation of the gradient vector

# Reusing Pretrained Layers

- Reuse layers, except for the top time
- Transfer learning
- Add a preprocessing step to resize to the expected original model

<img src="/images/Pasted image 20260204101857.png" alt="image" width="500">

- Make the weights of the reused layers non-trainable so the GD won't modify them
- Then train model
- Unfreeze one or two on the top hidden layers to let BP tweak them
- Useful to reduce the learning rate when unfreeze reused layers
	- Avoid wrecking fine-tuned weights
- Drop hidden layers and freeze all remaining hidden layers


## Transfer Learning with Keras

- Load model A and create anew model based on that model's layers
- Reuse all the layers except for the output layer

```python
[...]
model_A = tf.keras.models.load_model("my_model_A")
model_B_on_A = tf.keras.Sequential(model_A.layers[:1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid""))
```

- When `model_B_on_A` is trained, is will affect model A
- To avoid, clone `model_A` before

```python
model_A_clone = tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
```

- Train `model_B_on_A` for task B
- New output layer was initialized randomly and creates large errors
- Freeze the reuse layers, during the first few epochs
- Giving the new layers to learn reasonable weights
- Set every layer's `trainable=False`

```python
for layer in model_B_on_A.layers[:-1]:
	layer.trainable = False
	
optimizer = tf.keras.optimizer.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
			metrics=["accuracy"])
```

- Unfreeze the reused layers and continue training to fine-tune the reused layers for task B
- Reduce the learning rate, to avoid damaging the reused weights

```python
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4
		validation_data=(X_valid_B, y_valid_B))
for layer in model_B_on_A.layers[:-1]:
	layer.trainable = True
	
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
		metrics=["accuracy"])
		
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
			validation_data=(X_valid_B, y_valid_B))
			
>>> model_B_on_A.evaluate(X_test_B, y_test_B)
[0.2546142041683197, 0.9384999871253967]
```

- Transfer learning has reduced the error rate by 25%
- Transfer learning does not work well with small dense networks
- Small networks learn few patterns, and dense network learn very  specific patterns
- Transfer learning works best with deep convolutional NN
- Feature detectors that are more general

## Unsupervised Pretraining

- Cheap to gather unlabeled training examples, but expensive to label them
- Used to train an unsupervised model
	- Autoencoder
	- Generative adversarial network (GAN)
- Reuse the lower layers of the autoencoder of the GAN's discriminator, add the output layer for the task on tope, fine-tune the final network using supervised learning

- Unsupervised pretraining is a good option when you have a complex task to solve, no similar model, and little labeled training data, but lots of unlabeled training data
- Greedy layer-wise pretraining
	- Train an unsupervised model with a single layer
		- RBM (Restricted Boltzmann machine)
	- Freeze the layer and add another one on top
	- Train model again, and repeat
- Now, models train the full unsupervised model in one shat and use autoencoders or GANs rather than RBMs

<img src="/images/Pasted image 20260204103736.png" alt="image" width="500">

## Pretraining on an Auxiliary Task

- If there is little labeled training data, an alternative is to train a first NN on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers for actual task
- Build a system to recognize faces
	- Gather pictures of random people, and train NN
	- Learn feature detectors
- For natural language processing (NLP) application, download text documents and automatically generate labeled data
	- Mask out words and train a model to predict missing words
	- Train model to reach good performance, then reused

- Self-supervised learning is when you automatically generate the labels from the data itself
	- Text-masking
	- Train a model on the results "labeled" dataset using supervised learning techniques


# Faster Optimizers

- 4 ways to speed up training
	- Initialization strategy for connection weights
	- Activation function
	- Batch normalization
	- Reusing pretraining networks
- Faster optimizer
	- Momentum
	- Nesterov accelerated gradient
	- AdaGrad
	- RMSProp
	- Adam

## Momentum

- Momentum optimization
- Regular GD will take small steps when the slope is gentle and large when slope is deep
	- Never pick up speed
	- Slower to reach the min
- Momentum optimization depends on previous gradients
	- Subtracts the local gradient from the momentum vector $m$
- Gradient is used as an acceleration, not a speed
- Algorithm introduces perparameter $\beta$, call momentum, which is set between 0 and 1 (high to low friction)

**Momentum algorithm**

<img src="/images/Pasted image 20260204104549.png" alt="image" width="500">

- When the inputs have different scales, the cost function will look like an elongated bowl
- Help roll past local optima
- Optimizer may overshoot a but, then come back, overshoot again, and oscillate before stabilizing at the min
- Good to have friction in the system

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
```

- Momentum value of 0.9 works well

## Nesterov Accelerated Gradient

- Nesterov accelerated gradient (NAG)
	- Measures the gradient of the cost function not at the local position $\theta$, but ahead in the direction at, $\theta + \beta m$

**Nesterov Accelerated Gradient Algorithm**

<img src="/images/Pasted image 20260204105013.png" alt="image" width="500">


<img src="/images/Pasted image 20260204104755.png" alt="image" width="500">

- Nesterov update ends up closer to the optimum
- Momentum pushes weights across the valley
- NAG reduces oscillations

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
			nesterov=True)
```

## AdaGrad

- GD starts by going down the steepest slope
- AdaGrad algorithm achieves this over shooting by scaling down the gradient vector along the steepest dimensions

**AdaGrad Algorithm**

<img src="/images/Pasted image 20260204105357.png" alt="image" width="500">
- First step accumulates the square of the gradients into the vector $s$
- Gradient vector is scaled down by a factor of $\sqrt{s+ \epsilon}$
- Algorithm decays the learning rate, and faster for steep dimensions
- Adaptive learning

<img src="/images/Pasted image 20260204105510.png" alt="image" width="500">
- Performs well for simple quadratic problems
- Stops too early when training neural networks

## RMSProp

- Accumulates only the gradients from the most recent iterations
- Exponential decay

**RMSProp Algorithm**

<img src="/images/Pasted image 20260205085928.png" alt="image" width="500">

```python
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

## Adam

- Adam (Adaptive moment estimation)
- Combines the momentum optimization and RMSProp
- Keeps track of an exponentially decaying average of past gradients
- Keeps track of an exponentially decaying average of past squared gradients
- Mean is called the first moment
- Variance is called the second moment

**Adam Algorithm**

<img src="/images/Pasted image 20260205090146.png" alt="image" width="500">

$t$ = iteration number
- In step 1, computes exponentially decaying average rather than exponentially decaying sum

```python
optimizer = tf.keras.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

- Adam requires less tuning of the learning rate hyperparameter $\eta$

## AdaMax

- Adam scales down the parameter updates by the square root of $s$
- AdaMax replaced the $\ell_2$ norm with $\ell_{∞}$ 
- More stable, but depends on dataset

## Nadam

- Adam with Nesterov trick
- Converge gaster

## AdamW

- Integrates a regularization techniques called weight decay
- Weight decay reduces the size of the model's weight each training iteration by multiplying them by a decay factor such as 0.99

- Adaptive optimization methods help model converge fast
- Although not always the best case
- All optimization techniques are first-order partial derivatives
- All optimization algorithm discussed produce dense models
	- Parameters will be non-zero
- Use a sparse model for faster and less memory intense models
	- Train model as usual, remove tiny weights and set them to zero
- Apply a string $\ell_1$ regularization during training
	- Pushes optimizer to zero

<img src="/images/Pasted image 20260205091034.png" alt="image" width="500">

# Learning Rate Scheduling

- Find a good learning rate by training the model for a few hundred iterations, exponentially increasing the learning rate from small to large
- Choose the learning rate slightly lower than the one at which the learning curve starts to rise again
- Reinitialize model and train it with that learning rate

- Start with a large learning rate, and reduce it once training stops, to reach a faster optimal constant learning rate

**Learning Schedules**
- Exponential
	- Set learning rate to $\eta(t) = \eta_00.1^{t/s}$
	- Will gradually drop by a factor of 10 every step
- Piecewise
	- use a constant learning rate for number of epochs
	- Then a smaller learning rate for another number of epochs
- Performance
	- Measure the validation error every $N$ steps and reduce the learning rate by a factor of $\lambda$ when the error stops dropping
- Power
	- Set the learning rate to a function of iteration number
	- $t: \eta(t) = \eta_0/(1+t/s)^c$
	- Learning rate drops each step
- 1cycle
	- Starts by increasing the initial learning rate, growing linearly down to $\eta_0$ halfway through training
	- Decreases the learning rate again the second half
	- Super-convergence

<img src="/images/Pasted image 20260204105525.png" alt="image" width="500">
- Both performance scheduling and exponential scheduling performed well
- Favoured exponential scheduling because it was easy to tune

```python
def exponential_decay_fn(epoch):
	return 0.01 * 0.01 ** (epoch/20)
	
def exponential_decay(lr0, s):
	def exponential_decay_fn(epoch):
		return lr0 * 0.01 ** (epoch/2)
	return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.02, s=20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])
```

- `LearningRateScheduler` will update the optimizer's `learning_rate` attribute at the beginning of each epoch
- Schedule function can optionally take the current learning rate as a second argument

```python
def exponential_decay_fn(epoch, lr):
	return lr * 0.01 ** (1/20)
```

- Optimizer and learning rate is saved with model
- Epoch does not get saves, and resets to 0
- Set the `fit()` method's `initial_epoch=epoch`
- Piecewise constant scheduling

```python
def piecewise_constant_fn(epoch):
	if epoch < 5:
		return 0.01
	elif epoch < 15:
		return 0.005
	else:
		return 0.001
		
# multiply 0.5 when best val. does not improve 5 consecutive epochs
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])

# define a lr, then pass to optimizer
# updates at each step, rather than at each epoch
batch_size = 32
n_epochs = 20
n_steps = n_epochs * np.ceil(len(X_train) / batch_size)
scheduled_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)
optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_learning_rate)
```

```python
# power scheduling
lr_schedule = tf.keras.optimizer.schedules.InverseTimeDecay(
	initial_learning_rate=0.01,
	decay_steps=10_000,
	decay_rate=1.0,
	staircase=False
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

# Avoiding Overfitting Through Regularization

- The best regularization technique is early stopping
- Batch normalization to solve the unstable gradients
- $\ell_1$ and $\ell_2$ regularization
- Dropout
- Max-norm regularization

## $\ell_1$ and $\ell_2$ Regularization

- Use $\ell_2$ regularization to constrain a neural network's connection weights, and $\ell_1$ for a sparse model

```python
layer = tf.keras.layers.Dense(100, activation="relu",
		kernel_initizlier="he_normal",
		kernel_regularizer=tf.keras.regularizers.12(0.01))
```

- `12()` returns a regularizer that will be called at each step during training to compute the regularization loss
	- This is added to the final loss
- Apply the same regularizer to all layers in network, as well as same activation function and the same initialization strategy
- `functools.partial()`, thin wrapper for any callable function

```python
from functools import partial

RegularizedDense = partial(tf.keras.laysers.Dense,
							activation="relu",
							kernel_initializer="he_normal",
							kernel_regularizer=tf.keras.regularizer.12(0.01))
model = tf.keras.Sequential([
	tf.keras.layers.Fatten(input_shape[28,28]),
	RegularizedDense(100),
	RegularizedDense(100),
	RegularizedDense(10, activation="softmax")
])
```
## Dropout

- Dropout is one of the most popular regularization techniques for DNN
- 1-2% accuracy boost
- At every training step, every neuron has a probability $p$ of being temporarily dropped out (ignored during training step)
- Dropout rate, $p$, is set 10% and 50^
- After training, neurons don't get dropped
- Neurons train with dropout cannot co-adapt with their neighbouring neurons
- Become less sensitive to changes in the input, and result in a more robust network that generalizes better

<img src="/images/Pasted image 20260204105547.png" alt="image" width="500">

- NN is an averaging ensemble of all the smaller NN
- Apply dropout only to neurons in the top one to three layers (excluding output)
- Multiply each neuron's input connection weights by four during training
	- Divide connection weights by the keep probability $(1-p)$ durning training

```python
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape[28, 28]),
	tf.keras.layers.Dropout(rate=0.2),
	tf.keras.layers.Dense(100, ativation="relu",
					kernel_initialization="he_normal"),
	tf.keras.layers.Dropout(rate=0.2),
	tf.keras.layers.Dense(100, activation="relu",
					kernel_initialization="he_normal"),
		tf.keras.layers.Dropout(rate=0.2)
		tf.keras.layers.Dense(10, activation="softmax")
])
[...]
```

- Using Dropout, may overfit the training set, and have similar training and validation losses
- Evaluate training loss without dropout
- Alpha dropout
	- Variant that preserves the mean and standard deviation of its inputs

## Monte Carlo (MC) Dropout

- Connection between dropout networks and approximation Bayesian inference
- MC dropout
	- Boost the performance of any trained dropout model without having to retrain it of modify it
	- Provides a better measure of model's uncertainty

```python
import numpy as np
y_probas = np.stack([model(X_test, training=True)
				for sample in range(100)])
y_probas = y_probas.mean(axis=0)
```

- `model(X)` returns a tensor rather than a NumPy array
- Ensures the `Dropout` layer remains active, so all prediction swill be different
- Compute average over 100 predictions

```python
>>> model.predict(X_test[:1]).round(3)
array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.024, 0.   , 0.132, 0.   ,
        0.844]], dtype=float32
        
>>> y_proba[0].round(3)
array([0.   , 0.   , 0.   , 0.   , 0.   , 0.067, 0.   , 0.209, 0.001,
       0.723], dtype=float32)
       
>>> y_std = y_probas.std(axis=0)
>>> y_std[0].round(3)
array([0.   , 0.   , 0.   , 0.001, 0.   , 0.096, 0.   , 0.162, 0.001,
       0.183], dtype=float32)
       
>>> y_pred = y_proba.argmax(axis=1)
>>> accuracy = (y_pred == y_test).sum() / len(y_test)
>>> accuracy
0.8717
```

- The model is confident 84.4% that this image belongs to class 9
- MC dropout tends to improve the reliability of the model's probability estimates
- Standard deviation is 0.183

- Replace `Dropout` layer with `MCDropout`

```python
class MCDropout(tf.keras.layers.Dropout):
	def call(self, inputs, ttraining=False):
		return super().call(inputs, training=True)
```

## Max-Norm Regularization

- For each neuron, it constrains the weight $w$ of the incoming connections
- Does not add a regularization loss term to the overall loss function
- Reducing $r$ increases the amount of regularization and helps reduce overfitting

```python
dense = tf.keras.layers.Dense(
	100, activaton="relu", kernel_initializer="he_normal",
	kernel_constraint=tf.keras.constraints.max_norm(1.0))
```

 - After each training iteration, the mode's `fit()` method will call the object returned by `max_norm()`

# Summary and Practical Guidelines

<img src="/images/Pasted image 20260205095516.png" alt="image" width="500">

<img src="/images/Pasted image 20260205095528.png" alt="image" width="500">

- 