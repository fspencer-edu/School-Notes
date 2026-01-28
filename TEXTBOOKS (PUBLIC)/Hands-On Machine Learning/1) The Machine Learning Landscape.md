
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

- 


## Batch vs. Online Learning
## Instance-Based vs. Model-Based Learning

# Main Challenges of Machine Learning
## Insufficient Quantity of Training Data
## Non-representative Training Data
## Poor-Quality Data
## Irrelevant Features

## Overfitting the Training Data
## Underfitting the Training Data
## Stepping Back
 

# Testing and Validating

## Hyperparameter Tuning and Model Selection
## 
## 
## 