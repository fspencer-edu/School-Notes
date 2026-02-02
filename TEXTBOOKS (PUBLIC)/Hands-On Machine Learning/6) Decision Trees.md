
- Decision trees are versatile machine learning algorithms that can perform both classification and regression tasks
- Fundamental components of random forests

# Training and Visualizing a Decision Tree

```python
from sklearn.datasets import load_iris
from sklearn.tree import. DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

# visualize
from sklearn.tree import export_graphviz

export_graphviz(
	tree_clf,
	out_file="iris_tree.dot",
	feature=[["petal length (cm)", "petal width (cm)"]],
	class_names=iris.target_names,
	rounded=True,
	filled=True
)

# load and display tree
from graphviz import Source
Source.from_file("iris_tree.dot")
```

<img src="/images/Pasted image 20260202123727.png" alt="image" width="500">

# Making Predictions

- Start at the root node
	- Asks whether the flower's petal length is smaller than 2.45 cm
	- If it is, then move to left child node, else right
- Decision trees do not require scaling or centring
- A node's `samples` attribute counts how many training instances it apples to
- `value` attribute tells how many training instance of each class this node applies to
- `gini` attribute measure its Gini impurity
	- =0, "pure", if all instances it applies to belong to the same class

**Gini Impurity**

![[Pasted image 20260202124235.png]]

$p_{i, k}$ = ratio of the class k instances among the training instances

Example - Gini of Green Nose
$1 - (0/54)^2 - (49/54)^2 - (5/54)^2 = 0.168$

- Scikit learn uses the CART algorithm, which produces only binary trees
- ID3, can produce decision trees with nodes that have more than two children

- The thick vertical line represents the decision boundary of the root
- Righthand area is impure, and further divided into two nodes

![[Pasted image 20260202124458.png]]

- Decision trees are intuitive, and their decisions are easy to interpret
- Models are often called white box models
- Random forests and neural networks are considered black box models
- Interpretable ML aims at creating ML systems that can explain their decisions in a way humans can understand

# Estimating Class Probabilities

- A decision tree can also estimate the probability that an instance belongs to a particular class k
- Traverses the tree to find the leaf node for this instance, and returns the ratio of training instances of class k in this node

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)
array([[0.   , 0.907, 0.093]])
tree_clf.predict([[5, 1.5]])
array([1])
```

- The estimated probabilities are identical anywhere else int he bottom-right rectangle

# The CART Training Algorithm

- Scikit-learn uses the Classification and Regression Tree (CART) algorithm to train decision trees
- Algorithm works by splitting the training set into two subsets using a single feature k and a threshold $t_k$

**CART cost function for classification**

![[Pasted image 20260202125300.png]]

- Once CART algorithm has successfully split the training set in two, it splits the subsets using the same logic, then the sub-subsets, and so on, recursively
- Stops when reaches the max depth

- CART algorithm is a greedy algorithm
	- Search for an optimum split at the top level, then repeats
	- Does not check whether or not the split will lead to the lowest possible impurity level
	- Results is good but not optimal
- Finding the optimal tree is known as the NP-complete problem
	- $O(exp(m))$ time

# Computational Complexity

- Decision trees generally are approximately balanced, so traversing the decision tree is $O(log_2(m))$ nodes, where $log_2(m)$ is the binary logarithm of m
- The training algorithm compares all features on all samples at each node
- Training complexity is $O(n \times m log_n(m))$

# Gini Impurity or Entropy?

- `DecisionTreeClassifier` uses the Gini impurity measure
- Also select entropy impurity measure by setting the `criterion` to `entropy`
- Entropy originates in thermodynamics as a measure of molecular disorder
	- Entropy approaches zero when molecules are still and well ordered
	- Entropy is zero, when all messages are identical
	- Used as an impurity measure


**Entropy**

![[Pasted image 20260202125949.png]]

Example - Entropy of Green Node

$-(49/54)log_1(49/54)-(5/54)log_2(5/54) = 0.445$

- Most of the times Gini and entropy do not make a difference
- Gini impurity is slightly faster to compute
	- Isolate the most frequent class in its on branch of the tree
- Entropy produces more balanced tree

# Regularization Hyperparameters

- 

# Regression

# Sensitivity to Axis Orientation

# Decision Trees Have a High Variance