
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


<img src="/images/Pasted image 20260128111426.png" alt="image" width="500">



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

<img src="/images/Pasted image 20260128113704.png" alt="image" width="500">


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

<img src="/images/Pasted image 20260128114048.png" alt="image" width="500">

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
	return crc32(np.int64(identifier)) < test_ratio * 2 **32
	
def split_data_with_id_hash(data, test_ratio, id_column):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
	return data.loc[~in_test_set], data.loc[in_test_set]
```

- Housing dataset does not have an identifier column
- Use row index as the ID

```python
housing_with_id = housing.reset_index()
train_set, test_set = split_data_with_id_hash(housing_with_id, 0,2, "index")
```

- Scikit-learn splits data into multiple subsets in various ways, `train_test_splt()`

```python
from sklearn.model_selection import train_test_split

train_set, test_test = train_test_split(housing, test_size=0.2, random_state=42)
```
- `random_state` parameter allows you to set the random generator seed
- Pass multiple datasets with an identical number of row, and split them on the same indices

- Random sampling methods on small datasets can introduce sampling bias
- Stratified sampling
	- The population is decided into homogeneous subgroups called strata, and the right number of instances are sampled from each stratum to guarantee that the test set is representative of the overall population
- `pd.cut()` is used to create an income category attribute with five categories

```python
housing["income_cat"] = pd.cut(housing["median_income"],
								bins=[0., 1.0, 3.0, 4.5, 6., np.inf],
								labels=[1, 2, 3, 4, 5])
								
								
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income Category")
plt.ylabel("Number of districts")
plt.show()
```

<img src="/images/Pasted image 20260128115752.png" alt="image" width="500">

- Scikit-Learn provides a number of splitter classes in the `sklearn.model_seletion`
- Each splitter has a `split()` method that returns an iterator over different training/test splits on the same data
- Yields test indices

```python
from sklearn.model_selection import StratifiedSuffleSplot

splitter = StratifiedSuffleSplot(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
	strat_train_set_n = housing.iloc[train_index]
	strat_test_set_n = housing.iloc[test_index]
	strat_splits.append([strat_train_set_n, strat_test_set_n])
	
# first split
strat_train_set, strat_test_set = strat_splits[0]

# stratify arg
start_train_set, strat_test_set = train_test_split(
	housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
	
# dropping
for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis=1, inplace=True)
```

# Explore and Visualize the Data to Gain Insight

```python
housing = strat_train_set.copy()
```

## Visualizing Geographical Data

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.show()
```

<img src="/images/Pasted image 20260129122240.png" alt="image" width="500">

- ```python
# setting alpha value, density value
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()
```

<img src="/images/Pasted image 20260129122337.png" alt="image" width="500">

```python
# plot customization
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
			s=housing["population"]/100, label="population",
			c="median_house_value", cmap="jet", colorbar=True,
			legend=True, sharex=False, figsize=(10, 7))
			
plt.show()
```

<img src="/images/Pasted image 20260129122610.png" alt="image" width="500">


- Housing prices are related to the location and the population density
- Clustering algorithm should be used for detecting the main cluster and adding new features that measure the proximity to the cluster centres

## Look for Correlations

- Compute the standard correlation coefficient (Pearson's r)

```python
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_value(ascending=False)
median_house_value    1.000000
median_income         0.688380
total_rooms           0.137455
housing_median_age    0.102175
households            0.071426
total_bedrooms        0.054635
population           -0.020153
longitude            -0.050859
latitude             -0.139584
Name: median_house_value, dtype: float64
```
- Correlation coefficient ranges from -1 to 1
- Alternative is to use `scatter_matrix()`
	- Plots every numerical attribute against every other numerical attribute

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
```

<img src="/images/Pasted image 20260129122950.png" alt="image" width="500">


```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
```

<img src="/images/Pasted image 20260129123057.png" alt="image" width="500">

- Correlation coefficient only measure linear correlation

<img src="/images/Pasted image 20260129123212.png" alt="image" width="500">

## Experiment with Attribute Combinations

# Prepare the Data for Machine Learning Algorithms

- Write a function to prepare data
	- Reproduce transformations on any dataset
	- Build a library of transformation function to reuse
	- Use these functions in live systems to transfer the new data
	- Try various transformations

- Separate the predictors and the labels

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()
```

## Clean the Data

- Fix missing values
	- Remove corresponding values
	- Remove entire attribute
	- Set the missing value to some zero
		- Imputation

```python
housing.dropna(subset=["total_bedrooms"], inplace=True)
housing.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedroons"].fillna(median, inplace=True)
```

- `SimpleImputer`
	- Store the median value of each feature

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)

imputer.statistics_
array([-118.51 , 34.26 , 29. , 2125. , 434. , 1167. , 408. , 3.5385])
housing_num.median().values
array([-118.51 , 34.26 , 29. , 2125. , 434. , 1167. , 408. , 3.5385])

# transform the training set by replacing values
X = imputer.transform(housing_num)
```

- Other imputers
	- `KNNImputer`
		- Replaces each missing value with the mean of the k-nearest neighbour values for that feature
	- `InterativeImputer`
		- Trains a regression model per feature to predict the missing values based on all the other available features


**Scikit-Learn API Principles**
1) Consistency
	1) Estimators
	2) Transformers
	3) Predictors
2) Inspection
3) Nonproliferation of classes
4) Composition
5) Sensible defaults

- Scikit-Learn transformers output NumPy arrays
	- Neither column or index names
	- Wrap in a data frame and recover the column names and index

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

## Handling Text and Categorical Attributes

```python
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(8)
      ocean_proximity
13096        NEAR BAY
14973       <1H OCEAN
3785           INLAND
14689          INLAND
20507      NEAR OCEAN
1286           INLAND
18078       <1H OCEAN
4396         NEAR BAY
```

- Convert categories from text to numbers, using `OrdinalEncoder`

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = original_encoder.fit_transform(housing_cat)

housing_cat_encoded[:8]
array([[3.],
       [0.],
       [1.],
       [1.],
       [4.],
       [1.],
       [0.],
       [3.]])
       
ordinal_encoder.categories_
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
```

- ML algorithms will assume that two nearby values are more similar than two distant values
- Create one binary attribute per category
	- One-hot encoding
	- Only one attribute will be equal to 1, while the other 0
- The new attributes are called dummy attributes

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
<16512x5 sparse matrix of type '<class 'numpy.float64'>'
with 16512 stored elements in Compressed Sparse Row format>
 
housing_cat_1hot.toarray()
array([[0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       ...,
       [0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.]])
```

- Default output, is sparse matrix
	- Efficient representation for matrices that contain zeros
	- Stores only non-zero values and their positions


- `get_dummies()`
	- Converts each categorical feature into a one-hot representation, with one binary feature

```python
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)
   ocean_proximity_INLAND  ocean_proximity_NEAR BAY
0                       1                         0
1                       0                         1
```

- One hot encoder remembers which categories it was trained on
- When dealing with neural networks, replace each category with a learnable, low-dimensional vector called an embedding
	- Representational learning

- Estimator stores the columns names

```python
cat_encoder.feature_names_in_
array(['ocean_proximity'], dtype=object)
```

## Feature Scaling and Transformation

- Feature scaling
	- Min-max scaling
	- Standardization

- Fit the scalers to the training data only

**Min-Max Scaling (Normalization)**
- Simplest
- For each attribute, the values are shifted and rescaled so they end up ranging from 0 to 1
- Subtracting the min value and dividing by the difference between min and max

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
```
**Standardization**
- Subtracts the mean value, then divides the result by the standard deviation
- Does not restrict values to a specific range
- Less affected by outliers

```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

- When a feature distribution has a heavy tail, both min-max and standardization will squash most values into a small range
- Before scaling the feature, transform it to shrink the heavy tail
- Replace the feature with its square root
- If a power law distribution, replace with its logarithm

<img src="/images/Pasted image 20260129130557.png" alt="image" width="500">

- Bucketizing
	- Chopping its distribution into roughly equal-sized buckets, and replacing each feature value with the index of the bucket it belongs to
	- Replace value with its percentile
- Multimodal distribution, bucketize is used to treat the bucket IDs as categories
- Add a feature for each modes
- Radial basis function (RBF)
	- Any function that depends only on the distance between the input value and a fixed point
	- Gaussian RBF

```python
from sklearn.metrics.pairwise import rbf_kernel

age.simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

<img src="/images/Pasted image 20260129131005.png" alt="image" width="500">

- If the target distribution has a heavy tail, you may choose to replace the target with its logarithm
	- Regression model will now predict the log of the value, not median house value
	- Compute the exponential of model's prediction to get median value

```python
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scared_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]]iloc[:5]
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
```

- Simpler option is to use `TransformedTargetRegressor`
	- Construct it, given the regression model and label transformer
	- Fit training set, using the original unscaled labels

```python
from skilearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
				transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
```

## Custom Transformers

- Write your own transformers for tasks
	- Custom transformations
	- Cleanup operations
	- Combining specific attributes

- Write a function that takes a NumPy array as input and outputs the transformed array
- Transform features with heavy tailed distributions by replacing them with their log

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])
```

- Transformation function can take hyperparameters as additional arguments

```python
rbf_transformer = FunctionTransformer(rbf_kernel, 
						kw_args=dict(Y=[[35.]], gamma = 0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])
```

- If you pass an array with two features, it will measure the 2D distance to measure similarity
- To add a feature that will measure the geographic similarity

```python
sf_coords = 37.7, -122
sf_transformer = FunctionTransformer(rbf_kernel,
						kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["lat", "long"]])
```

- Custom transformers are useful to combine features

```python
>>> ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
>>> ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
array([[0.5 ],
       [0.75]])
```

- Write a custom class
- Scikit-learn relies on duck typing, so this class does not have to inherit from any particular base class
- The default implementation will call `fit()` then `transform()`

Custom Transformers, similar to `StandardScaler`
```python
from skilearn.base import BaseEstimator, TransformerMixin
from skilearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
	def __init__(self, with_mean=True):
		self.with_mean = with_mean
		
	def fit(self, X, y=None):
		X = check_array(X)
		self.mean_ = X.mean(axis=0)
		self.scale_ = X.std(axis=0)
		self.n_features_in_ = X.shape[1]
		return self
		
	def transform(self, X):
		check_is_fitted(self)
		X = check_array(X)
		assert serf.n_features_in_ == X.shape[1]
		if self.with_mean:
			X = X - self.mean_
		return X / self.scale_
```


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