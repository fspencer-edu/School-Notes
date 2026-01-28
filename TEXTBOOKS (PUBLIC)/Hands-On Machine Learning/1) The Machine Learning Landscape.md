
- Supervise vs. unsupervised learning
- Online vs. batch learning
- Instance-based vs. model-based learning

- Machine learning is the science of programming computers so they can learn from data
- Each training set is a training instance/sample
- ML system that learns and makes prediction is the model
	- Neural networks
	- Random forests
- Accuracy
	- Performance measure
	- Classification tasks


# Why Use Machine Learning?

Traditional Approach
1. Examine spam
2. Write a detection algorithm for each pattern
3. Test program, and repeat steps 1 and 2

- Machine learning techniques automatically learns which words and phrases are good predictors of space
- The best solution is to write an algorithm that learns by itself
- ML models can be inspected to see what they have learned
- Data mining
	- Analyzing large amounts of data to find hidden patterns

ML Is Used For
- Fine-tuning or long lists of rules
- Complex problems
- Fluctuating environments
- Insights about complex problems and large amounts of data

# Examples of Applications

- Images of products on a production line to classify them
	- Image classification
	- Convolutional neural networks (CNN)
- Detecting tumours in the brain
	- Image segmentation
- Automatically classifying news articles
	- Natural language processing (NLP)
	- Text classification
	- Recurrent neural networks (RNNs) and CNNs
	- Transformers
- Flagging offensive comments
	- Text classification, NLP
- Summarizing long documents
	- Branch of NLP called text summarization
- Creating a chatbot
	- NLP
	- Natural language understanding (NLU)
	- Question-answering modules
- Forecasting revenue
	- Regression (predicting values)
		- Linear
		- Polynomial
		- Vector machine
		- Random forest
		- Artificial neural network
- App react to voide
	- Speech recognition
	- RNNs, CNNs, or transformers
- Detecting credit card fraud
	- Anomaly detection
	- Isolation forests
	- Gaussian mixture models
	- Autoencoders
- Segmenting clients based on purchases
	- Clustering
		- k-means
		- DBSCAN
- Complex, high-dimensional dataset, datagram
	- Data visualization
- Recommending a product
	- Artificial neural network
- Building an intelligent bot for a game
	- Reinforcement learning (RL)

# Types of Machine Learning Systems

**Criteria**
- Supervised during training
	- Supervised
	- Unsupervised
	- Semi-supervised
	- Self-supervised
- Incremental learning
	- Online
	- Batch learning
- Type of data
	- Instance-based
	- Model-based


## Training Supervision

- ML systems can be classified according to the amount and type of supervision they get during training

### Supervised Learning
- Training set includes the desired solution, called labels
- Classification learning task
- Predict a target numerical value, based on features
	- Regression
- Logistic regression is used for classification
	- Outputs a value that corresponds the the probability of belonging to a given class

![[Pasted image 20260128091856.png]]

- Target
	- More common in regression tasks
- Label
	- Common in classification tasks
- Features also called, predictors, or attributes

### Unsupervised Learning

- Training data is unlabeled
- Hierarchical clustering algorithms can be used to subdivide each group into small groups
- Visualization algorithms help one to understand data and patterns
- Dimensionality reduction
	- Simplify the data without losing information
	- Merge several correlated features
	- Feature extraction
- Anomaly detection
	- Detecting unusual transactions
	- System is shown mostly normal instances during training
- Novelty detection
	- Aims to detect new instances that look difference for all instances
	- Clean training set
- Association rule learning
	- Discover relations between attributes


![[Pasted image 20260128092339.png]]


### Semi-Supervised Learning
- Partially labeled data
- Combination of supervised and unsupervised algorithms
- A clustering may be used to group similar instances, then similar unlabeled instance can be labeled with its common
- Google Photos

![[Pasted image 20260128092556.png]]


### Self-supervised Learning

- Generating a full labeled dataset from a fully unlabeled one
- Mask a small part of each image, then train a model to recover the original image
- Model training using self-supervised learning is not the final goal
	- Tweak model to classify, instead of repair images
- Transfer learning
	- Transferring knowledge from one task to another
- Deep learning networks
	- Neural networks composed of may layers of neurons

- Uses generated labels during training


### Reinforcement Learning

- Learning system, called an agent can observer the environment, select and perform actions, and get rewards/penalties in return
- Learn a policy, to get the most reward over time
	- Policy defines action the agents takes


## Batch vs. Online Learning
### Batch Learning
- System is incapable of learning incrementally
- Trained using all the available data
- Offline learning
- Performance tends to decay over time
	- Model rot or data drift
- Regularly retrain the model on up to data data
- Training, evaluating, and launching a machine learning system can be automated
	- Time consuming
	- Computing resources
	- Data storage

### Online Learning

- Train the system incrementally by feeding it data instances sequentially
	- Mini-batches
- Learning steps are fast and cheap
- Adapts to change
- Trains on limited computing resource
- Train models on huge datasets that cannot fit on main memory
	- Out-of-core learning
		- Done offline
- Learning rate
	- Adapt to new data, but forget old data
- System declines with bad data
	- Bug
	- Spamming
- Monitor system and switch learning off, or reach to abnormal data


## Instance-Based vs. Model-Based Learning

- Generalization of data
### Instance-based Learning

- Measure of similarity
	- Similarity measure between two data objects
- System learns by data points, then generalized a new case using similarity measures

![[Pasted image 20260128094514.png]]

### Model-Based Learning

- Generalize from a set of examples to build a model of examples, then use that model to make predictions


![[Pasted image 20260128094603.png]]

- Model selection with specific parameters, that represent a function
- Specify a performance measure
	- Utility/fitness function that measure how good the model is
	- Define a cost function that measure how bad the model is
		- Distance between linear models
- Objective is to minimize distance
- Final training model find the optimal parameter values

```python
import matplotlib.pyplot as plt
import numbpt as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Prepare Data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize data
lifesat.plot(kind='scatter', grid=True,
		x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a Lienar Model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Made a prediction
X_new = [[37_655.2]]
print(model.predict(X_new))
```

- Replacing linear regression with k-nearest neighbours regression

```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
```

- Inference
	- Apply the model to make prediction on new cases

# Main Challenges of Machine Learning

- Bad model
- Bad data

## Insufficient Quantity of Training Data

- ML takes a lot of data
- Different learning algorithms, performed almost identically well on a complex problem of natural language disambiguation once given enough data

![[Pasted image 20260128102518.png]]

## Non-representative Training Data

- Training data must be representative of the new cases to be generalized
- Trained model is unlikely to make accurate predictions
- If sample is too small, it will have sampling noise
- Flawed sample in a sampling bias

## Poor-Quality Data

- Errors, outliers, and noise will make it difficult to detect underlying patterns
- Cleaning data
	- Discard or fix outliers
	- Ignore instances, or fill in missing values, or train with specific features

## Irrelevant Features

- Feature engineering
	- Feature selection
	- Feature extraction
	- Creating new features by gathering new data

## Overfitting the Training Data

- Overgeneralizing or overfitting
	- Model performs well on the training data, but it does not generalize well
- Complex models such as deep neural networks can detect subtle patterns in the data
- Overfitting occurs when the model is too complex relative to the amount and noisiness of the training data
	- Simplify the model by selecting fewer parameters (regularization)
	- Gather more training data
	- Reduce the noise in the training data

- A parameter correlated to the degrees of freedom to adapt to the model to the training data
	- Tweak both the heigh and the slope of the line
- Amount or regularization to apply during learning can be controlled by a hyperparameter
- A hyperparameter is a parameter of a learning algorithm (not of the model)
- 
## Underfitting the Training Data
## Stepping Back
 

# Testing and Validating

## Hyperparameter Tuning and Model Selection
