
# Working with Real Data

**Data Repo**
- OpenML
- Kaggle
- PapersWIthCode
- UC Irving Machine Learning Repository
- AMazon's AWS datasets
- TensorFlow datasets

**Meta portals**
- DataPortals
- OpenDataMonitor


# Look at the Big Picture

- Block groups are the smallest geographical unit for which the US census Bureau publishes sample data
	- 600 to 3000 people
- Attributes
	- Population
	- Median income
	- Median housing price


## Frame the Problem

- Building a model is not the end goal
- Current situation will give a reference for performance
- Pipeline
	- A sequence of data processing components
	- Components run asynchronously
	- Each component pulls in data, processes it, and outputs it
	- Self-contained
		- Interface between components is the data store
	- If a component breaks down, the downstream compnents can run normally by using the last output from the broken component

- Determine
	- Type of training supervision the model will use
	- Classification taks
	- Batch learning or online learning techniques

- To find analyze district's median housing price
	- Supervised learning task, with labeled instances
	- Regression task, since the model will be asked to predict a value
	- Multiple regression
		- System will use multiple features to make a prediction
	- Univariate regression
		- Predict a single value for each district
	- No continuous flow of data, use batch learning

## Select a Performance Measure

- Select a performance measure
	- Root mean squared error (RMSE)
	- Shows how much error the system makes in its prediction

**Root Mean Squared Error (RMSE)**

$RMSE(X, h) = \sqrt{1/m\sum^m_{i=1}(h(x^{(i)})-y^{(i)})^2}$

$m$ = number of instances in the dataset
$x^{(i)}$ = vector of all the feature values of the instance in the dataset
$y^{(i)}$ = label


$x^{(1)} = \begin{pmatrix} -118.29 \\ 33.91 \\ 1416 \\ 38372 \end{pmatrix}$

$y^{(1)} = 156400$ => median house value of $x^{(1)}$


- $X$ is a matrix containing all the values values of all instances in the data set
	- One row per instance
- $h$ is system's prediction function, called hypothesis, $\hat{y} = h(x^{()})$




![[Pasted image 20260128111426.png]]

## Check the Assumptions

# Get the Data

## Running the Code Examples Using Google Collab
## Saving Your Code Changes and Your Data
## The Power and Danger of Interactivity
## Book Code vs. Notebook Code
## Download the Data
## Take a Quick Look at the Data Structure
## Create a Test Set

# Explore and Visualize the Data to Gain Insight

## Visualizing Geographical Data
## Look for Correlations
## Experiment with Attribute Combinations



# Prepare the Data for Machine Learning Algorithms

## Clean the Data
## Handling Text and Categorical Attributes
## Feature Scaling and Transformation
## Custom Transformers
## Transformation Pipeline


# Select and Train a Model

## Train and Evaluate on the Training Set
## Better Evaluation Using Cross-Validation


# Fine-Tune Your Model

## Grid Search
## Randomized Search
## Ensemble Methods
## Analyzing the Best Models and Their Errors
## Evaluate Your System on the Test Set


# Launch, Monitor, and Maintain Your System