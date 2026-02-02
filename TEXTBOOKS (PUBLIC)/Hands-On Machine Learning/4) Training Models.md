
**2 Ways to Train Linear Regression Model**
- Closed form
	- Directly computes the model parameters that best fir the model to the training set
- Iterative optimization approach called gradient descent (GD)
	- Gradually tweaks the model parameters to minimize the cost function over the training set
		- Batch GD
		- Mini batch GD
		- Stochastic GD

**Polynomial Regression**
- A more complex model that gan fit non-linear datasets
- More prone to overfitting the training data

- Logistic regression
- Softmax regression

# Linear Regression

Life Satisfaction Regression Model

$life_s = \theta_1 + \theta_1 \times GDP_{perCaptia}$

- $\theta$ are the models parameters
- A linear model makes a prediction by computing a weighted sum of the input features, plus a constant called the bias terms (intercept term)

**Linear Regression Model Prediction**

$\hat{y} = \theta_1 + \theta_1x_1 + \theta_2x_2 + ...+$

$\hat{y}$ = predicted value
$n$ = number of features
$x_i$ = feature value
$\theta_j$ = model parameter

**Linear Regression Model Prediction (Vectorized Form)**
$\hat{y} = h_{\theta} = \theta \cdot x$

$\hat{y}$ = pr


## The Normal Equation
## Computational Complexity

# Gradient Descent

## Batch Gradient Descent
## Stochastic Gradient Descent
## Mini-Batch Gradient Descent


# Polynomial Regression

# Learning Curves

# Regularized Linear Models

## Ridge Regression
## Lasso Regression
## Elastic Net Regression
## Early Stopping


# Logistic Regression

## Estimating Probabilities
## Training and Cost Function
## Decision Boundaries
## Softmax Regression