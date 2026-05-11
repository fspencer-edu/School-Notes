# Decision Trees

- Predictive modeling tool
- Uses a tree structure to represent a number of possible decision paths and an outcome for each path
- Can use a mix of numerical and categorical attributes

<img src="/images/Pasted image 20260429165215.png" alt="image" width="500">

- Classification trees
	- Produce categorical outputs
- Regression trees
	- Produce numerical outputs


# Entropy

- Entropy
	- A measure of impurity, disorder, or uncertainty within a data set
	- Determined how mixed the target classes are

# Entropy of a Partition

- Want a partition to have low entropy if it splits the data into subsets
- A high entropy contains subsets that have high uncertainty

# Creating a Decision Tree

- Decision nodes
- Leaf nodes
	- Represent prediction

- ID3 algorithm
	- If data has the same label, create a leaf node and stop
	- If the list of attributes is empty, create a leaf node that predicts the most common label and stop
	- Partition the data by each of the attributes
	- Partition with the lowest entropy
	- Add a decision node based on attribute
	- Recur on each partitioned subset using the remaining attributes

- Greedy algorithm
	- Chooses the most immediate best option

# Putting It All Together

- Leaf
	- Predicts a single value
- Split
	- Containing an attribute to split on
	- Subtree for specific values of that attribute
	- Default value for unknown values

# Random Forests

- Random forests
	- Build multiple decision trees and combine their outputs
	- Average predictions
- Bootstrap aggregating or bagging
- Ensemble learning
	- Choose best attribute to split on
	- Combine weak learners to produce a strong model

# For Further Exploration