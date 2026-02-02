
- Wisdom of the crowd
	- The aggregate answer is better than an expert's answer
- A group of predictors is called an ensemble
	- Ensemble learning
- Random forest
	- Train a group of decision tree classifiers, each on a different random subset of the training set
	- Vote on the best prediction
- Use ensemble methods near the end of a project
- Netflix Prize competition

# Voting Classifiers

- The aggregate of classifiers will produce a better result
- The class that gets the most votes is the ensemble's prediction
- The majority vote classifier is called a hard voting classifier

<img src="/images/Pasted image 20260202141100.png" alt="image" width="500">

- Even if the classifier is a weak learning, the ensemble can still be a string learning
- Law of large numbers
	- As you keep tossing the coin, the ratio of heads get closer and closer to the probability

<img src="/images/Pasted image 20260202141235.png" alt="image" width="500">

- Ensemble methods work best when the predictor are as independent from one another as possible
- Train using different algorithms
- `VotingClassifier` is used to give a list of name/predictor pairs, and use it like a normal classifier
- Create and train a voting classifier composed of 3 diverse classifiers

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
	estimators=[
		('lr', LogisticRegression(random_state=42)),
		('rf', RandomForestClassifier(random_state=42)),
		('svc', SVC(random_state=42))
	]
)
voting_clf.fit(X_train, y_train)
```

- `VotingClassifier` clones every estimator and fits the clones
- The original estimators are available via `estimators` and fitted clones `estimators_`

```python
for name, clf in voting_clf.named_estimator_.items():
	print(name, "=", clf.score(X_test, y_test))
lr = 0.864
rf = 0.896
svc = 0.896
voting_clf.predcit(X_test[:1])
array([1])
[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]
[array([1]), array([1]), array([0])]
voting_clf.score(X_test, y_test)
0.912
```

- `predict()`, performs hard voting
- Voting classifier predicts class 1 for the first instance, because 2 out of 3 are class 1
- The voting classifier outperforms all the individual classifiers
- If all classifiers are able to estimate class probabilities, then predict the class with the highest class probability, averaged over all the individual classifiers
	- Soft voting
	- Gives more weight to highly confident votes

```python
voting_clf.voting = "soft"
voting_clf.names_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
0.92
```

# Bagging and Pasting

- Another approach to use the same training algorithm for every predictor, but train them on different random subsets of the training set
- Bagging also called bootstrap aggregating
	- Sampling with replacement
- Pasting
	- Sampling performed without replacement

- Bagging and pasting allow training instances to be sampled several times across multiple predictors, only bagging allows training for the same predictor

<img src="/images/Pasted image 20260202142512.png" alt="image" width="500">

- After predictors are trains, the ensemble can make a prediction for a new instance by aggregating the predictions of all predictors
	- Statistical mode for classification or average for regression
- Each individual predictor has a higher bias than if it were trained on the original, but aggregation reduces bias and variance
- Net result has a similar bias, but lower variance than a single predictor on the original training set

- Predictors can be trained in parallel, via CPU cores

## Bagging and Pasting in Scikit-Learn

- Scikit-Learns offers a API
	- `BaggingClassifier`

```python
fron sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimator=500,
			max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

- Trains 500 decision tree classifiers, each one 100 training instances randomly sampled from the training set with replacement
- Use `bootstrap=False` for pasting
- `-1` for using all available CPU cores
- Automatically performs soft voting instead of hard voting if the base classifier can estimate class probabilities

<img src="/images/Pasted image 20260202143051.png" alt="image" width="500">

- Bagging introduces more diversity in the subsets that each predictor is trained on
- Higher bias than pasting

## Out-of-Bag Evaluation

- With bagging, some training instances may be sampled several times for any given predictor
- Samples m training instances with replacement, there m is the size of the training set
- 63% of the training instances are sampled on average for each predictor
- 37% are not sampled, and called out-of-bag (OOB) instances
- Bagging ensemble can be evaluated using OOB instances, without the need for a separate validation set
- Get an automatic OOB evaluation after training with `obb_score=True`

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
				oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_Train)
bag_clf.obb_score_
0.896
```

- Therefore, the `BaggingClassifier` is likely to achieve about 89.6% accuracy on the test set

```python
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
0.92
>>> bag_clf.oob_decision_function_[:3]  # probas for the first 3 instances
array([[0.32352941, 0.67647059],
       [0.3375    , 0.6625    ],
       [1.        , 0.        ]])
```

## Random Patches and Random Subspaces

- `BaggingClassifier` supports sampling the feature as well
- Sampling is controlled by `max_features` and `bootstrap_features`
- Used for high-dimensional inputs, images
- Sampling both training instances and features is called random patches method
- Keeping all training instances, but sampling features is called the random subspace method
- Sampling features result in more predictor diversity, trading a bit more bias for a lower variance

# Random Forests

- A random forest is an ensemble of decision trees, generally trained via the bagging method
- Instead of building a classifier and passing a decision tree classifier, use `RandomForestClassifier`

```python
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifer(n_estimators=500, max_leaf_nodes=16,
			n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

- Trains a random forest classifier with 500 trees, each limited to max 16 leaf nodes, using all available CPU cores
- RF introduces randomness when growing trees
- Instead of searching for the very best feature when splitting a node, it search for the best feature among a random subset of features
- Samples $\sqrt{n}$ features
- Greater tree diversity, trades higher bias for a lower variance

```python
bag_clf = BaggingClassifier(
	DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
	n_estimators=500, n_jobs=-1, random_state=42
)
```

## Extra-Trees

- A forest with extremely random trees is called an extremely randomized trees
	- Extra-trees
- Trades more bias for a lower variance
- Makes extra-trees classifiers much faster to train than regular random forests
- `ExtraTreeClassifier`
- `splitter="random"` when creating `DecisionTreeClassifier`

## Feature Importance

- RF measure the relative importance of each feature
- Measures a feature's importance by looking at how much the tree nodes that use that feature reduce impurity on average
- Weighted average, where each node's weight is equal to the number of training samples that are associated
- Scikit-Learn computes the score automatically for each feature after training, then scales the results so that the sum of all importances is equal to 1

```python
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
rnd_clf = randomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
	print(round(score, 2), name)
0.11 sepal length (cm)
0.02 sepal width (cm)
0.44 petal length (cm)
0.42 petal width (cm)
```

- Therefore, the important features are petal length (44%), width (42%), sepal length (11%) and width (2%)
- Train a random forest classifier on the MNIST dataset, and plot each axis's importance

<img src="/images/Pasted image 20260202145447.png" alt="image" width="500">


# Boosting

- Boosting (hypothesis boosting) refers to any ensemble method that can combine several weak learners into a strong learner
- Train predictors sequentially, each trying to correct its predecessor
	- AdaBoost (adaptive boosting)
	- Gradient boosting

## AdaBoost

- Pay a bit more attention on the training instances that the predecessor underfit
- When training an AdaBoost classifier, the algorithm first trains a base classifier (decision tree) and uses it to make predictions on the training set
- Algorithm then increases the relative weight of misclassified training instances

<img src="/images/Pasted image 20260202145620.png" alt="image" width="500">

- The first classifier get many instances wrong, so their weights get boosted
- Sequential learning technique is similar with GD
	- Adds predictors to the ensemble

<img src="/images/Pasted image 20260202145954.png" alt="image" width="500">

- Training cannot be parallelized since each predictor can only be trained after the previous one
- Each instance weight is initially set to $1/m$


**Weighted error rate of the j-th predictor**

<img src="/images/Pasted image 20260202150123.png" alt="image" width="500">

- The predictor weight $\alpha_j$ is computed, where $\eta$ is the learning rate hyperparameter (defaults to 1)
- The more accurate the predictor is, the higher its weight will b


**Predictor weight**

<img src="/images/Pasted image 20260202150131.png" alt="image" width="500">

- AdaBoost algorithm updates the instance weights


**Weight update rule**

<img src="/images/Pasted image 20260202150146.png" alt="image" width="500">

- Scikit-Learn uses a multiclass version of AdaBoost called SAMME (Stagewise Additive Modelling using a Multiclass Exponential loss function)

```python
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
	DecisionTreeClassifier(max_depth=1), n_estimators=30,
	learning_rate=0.5, random_state=42
)
ada_clf.fit(X_train, y_train)
```

- AdaBoost classifier based on 30 decision stumps
- A decision stump is a decision tree with a max_depth of 1
	- Node with two leaf nodes
- If AdaBoost ensemble is overfitting the training set, reduce the number of estimators or more strongly regularizing the bast estimator

## Gradient Boosting

- Gradient boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor
- Instead of tweaking the instance weights at every iteration, it tries to fit the new predictor to the residual error made by the previous predictor
- Using decision trees as the base predictor
	- Gradient tree boosting or gradient boosted regression trees (GBRT)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)
X = np.random.seed(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100) # y = 3x^2 + Gaussian noise
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# Train the second DTR on the residual errors
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

# Train 3rd regression on the residual errors from 2nd predictor
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_ref3.fit(X, y3)

# Make predictions on a new instance by addding up the predictions of all trees
X_new = np.array(([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
array([0.49484029, 0.04021166, 0.75026781])

# gradient boosting
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimator=3,
			learning_rate=1.0, random_state=42)
gbrt.git(X, y)
```
- The graphs show the predictions of the 3 trees, and the ensemble's predictions
- First row
	- Ensemble has one tree, so predictions are the same
- Second row
	- A new tree is trained on the residual error of the first tree
	- Ensemble's prediction are equal to the sum of the predictions of the first two tress
- Third row
	- Another tress is trained on the residual error of the second tree
	- Ensemble's prediction gradually gets better as trees are add


<img src="/images/Pasted image 20260202151522.png" alt="image" width="500">


- The `learning_rate` scales the contribution of each tree
- Shrinkage
	- Regularization technique
	- If the learning rate is low, then you will need more trees in the ensemble to fit the training set

<img src="/images/Pasted image 20260202151901.png" alt="image" width="500">

- To find the optimal number of trees, perform cross-validation using grid search or randomized search CV
- Or set `n_iter_no_change` to an integer value, then GBR will automatically stop adding more trees during training if it sees that the last 10 trees did not help
- Tolerate having no progress, then stops

```python
gbrt_best = GradientBoostingRegressor(
	max_depth=2, learning_rate=0.05, n_estimators=500,
	n_iter_no_change=10, random_state=42
)

gbrt_best.fit(X, y)
gbrt_best.n_estimators_
92
```

- Stochastic gradient boosting
	- Speeds up training
	- Each tree is trained on 25% of the training instances


## Histogram-Based Gradient Boosting

- Another GBRT implementations for large datasets is histogram-based gradient boosting (HGB)
- Binning the input features, replacing them with integers
- Binning can reduce the number of possible thresholds that the training algorithm needs to evaluate
- Working with integers makes it possible to use faster and more memory-efficient data structures
- Removes the need for sorting the feature when training each tree
- Complexity of $O(b \times m)$
- Binning causes a precision loss
- Two classes
	- `HistGradientBoostingRegressor`
	- `HistGradientBoostingClassifier`
- Early stopping is automatically activated if the number of instances is greater than 10,000
- Subsampling is not supported
- `n_estimators` is renamed to `max_iter`
- The only decision tree hyperparameter than can be tweaked are `max_leaf_nodes, min_samples_leaf, max_depth`
- 2 features
	- Support for categorical features and missing features

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
	make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
					remainder="passthrough"),
	HistGradientBoostingRegressor(categoritcal_features=[0], random_state=42)
)

hgb_reg.fit(housing, housing_labels)
```

- Other gradient boosting implementations
	- XGBoost
	- CatBoost
	- LightGBM

# Stacking

- Stacking also called stacked generalization
- Instead of using trivial functions to aggregate the predictions of all predictors, train a model to perform this aggregation
- Each of the bottom 3 predictors predict a different value, and then a final predictor (blender, or a meta learner) takes these predictions as inputs and makes a final prediction
<img src="/images/Pasted image 20260202153048.png" alt="image" width="500">

- To train the blender, first build the blending training set
- Use `cross_val_predict()` on every predictor to get out-of-sample prediction for each instance in the original training set
- The blending training set will contain one input feature per predictor
- Once blender is trained, the base predictors are retrained one last time on the full original training set

<img src="/images/Pasted image 20260202153059.png" alt="image" width="500">
- Train several different blenders (with different classifiers) to get a whole layer of blenders
- Add blenders on top of each other to produce the final prediction

<img src="/images/Pasted image 20260202153119.png" alt="image" width="500">

- `StackingClassifier`
- `StackingRegressor`

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
	estimators=[
		('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
	],
	final_estimator=RandomForestCl
)
```