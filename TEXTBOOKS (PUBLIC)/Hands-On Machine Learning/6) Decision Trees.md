
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

# The CART Training Algorithm

# Computational Complexity

# Gini Impurity or Entropy?

# Regularization Hyperparameters

# Regression

# Sensitivity to Axis Orientation

# Decision Trees Have a High Variance