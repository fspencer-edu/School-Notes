
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
- $h$ is system's prediction function, called hypothesis, $\hat{y} = h(x^{(i)}) = 158400$
- Prediction error for district is $\hat{y}^{(1)} - y^{(1)} = 2000$
- $RMSE(X, h)$ is the cost function measured on the set of example using hypothesis, $h$


![[Pasted image 20260128111426.png]]



- In some contents, use mean absolute error (MAE, also called the average absolute deviation)


**Mean Absolute Error (MAE)**

$MAE(X, h) = 1/m \sum^m_{i=1}|h(x^{(i)})-y^{(i)}|$ 

- Both RMSE and MAE are ways to measure the distance between two vector
	- Vector of predictions and the vector of target values
	- Various distance measures, or norms
		- RMSE corresponds to the Euclidean norm, $\ell_2$ or $|| \cdot ||_2$
		- Computing MAE corresponds to $\ell_1$ or Manhattan norm
			- Measures the distance between two points in a city along orthogonal blocks
		- $\ell_k$ norm of a vector is defined as
			- $||v||_k = (|v_1|^k + |v_1|^k ...)^{1/k}$

- The higher the norm index, the more it focuses on large values and neglects small one
- RMSE is more sensitive to outliers than MAE


## Check the Assumptions

- Verify the assumption that have been made so far

# Get the Data

## Running the Code Examples Using Google Colab

- A Jupyter notebook is composed of a list of cells
- Each cell contains either executable code or text
- Colab allocates a new runtime
	- Free virtual machine located on Google's servers that contains Python libraries
- Code runs on runtime, not on machine

## Saving Your Code Changes and Your Data

- Any command starting with a `!` will be treated as a shell command

## The Power and Danger of Interactivity

- Restart runtime, and run the cells again from the beginning to help fix errors

## Book Code vs. Notebook Code

- Code differences
	- Library changes
	- Extra code to beautify the figures

- PEP 8 Python style

## Download the Data

- In typical environments, data will be available in a relational database or other common data store, across multiple files
- Preferable to write a function to download and decompress data

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
	tarball_path = Path("datasets/housing.tgz")
	if not tarball_path.is_file():
		Path("datasets").mkdir(parents=True, exist_ok=True)
		url = "https://github.com/ageron/data/raw/main/housing.tgz"
		urllib.request.urlretrieve(url, tarball_path)
		with tarfile.open(tarball_path) as housing_tarball:
			housing_tarball.extractall(path="datasets")
	return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
```

## Take a Quick Look at the Data Structure

![[Pasted image 20260128113704.png]]


- Row represents one district
- `info()` returns a quick description of data
- All attributes are numerical, except for `ocean_proximity`
	- Its type is `object`, text attribute
- Find categorical attribute

```python
housing["ocean_proximity"].value_counts()
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
```

- `describe()` method shows a summary of the numerical attributes
- Null values are ignored
- `std` shows the standard deviation, which measures how dispersed the values are
- Percentailes
	- Indicates the value below which a given percentage of observations in a group observations fall
- Histogram shows the number of instances that have a given value range

```python
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize(12, 8))
plt.show()
```

![[Pasted image 20260128114048.png]]

- Median income attributes does not look like it is expressed in dollars
	- Data has been scaled and capped at 15 for higher incomes, and 0.5 for lower median incomes
- Housing median age and house value were also capped
- Attributes have different scales
- Many histograms are skewed right
	- Extend father to the right of the median than to the left

## Create a Test Set

- Data snooping bias
	- Estimate the generalization error using the test set, estimate will be too optimistic

```python
import numpy as np
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    
    
train_set, test_set = shuffle_and_splot_data(housing, 0.2)
len(train_set)
16512
len(test_set)
4128
```
- Save the test set on the first run and load it in subsequent runs
- Set the random number generator's seed before called permutation so that is always generates the same shuffled indices
- To have a stable train/test split aver updating the dataset, use each instance's identifier to decide whether or not is should go in the test set
- Compute a hash of each instance's identifier and put that instance in the test if the hash is lower or equal to 20% of the max hash value
	- Test set will remain consistent across multiple runs
	- New sets will contain 20% of new instances

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
	r
```


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