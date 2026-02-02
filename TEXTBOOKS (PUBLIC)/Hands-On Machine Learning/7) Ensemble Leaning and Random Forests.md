
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

![[Pasted image 20260202141100.png]]

- Even if the classifier is a weak learning, the ensemble can still be a string learning
- Law of large numbers
	- As you keep tossing the coin, the ratio of heads get closer and closer to the probability

![[Pasted image 20260202141235.png]]

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

![[Pasted image 20260202142512.png]]

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

![[Pasted image 20260202143051.png]]

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

- Therefore, the `BaggingClassifier` is likely to achieve about 89.6% a

## Random Patches and Random Subspaces

# Random Forests

## Extra-Trees
## Feature Importance

# Boosting

## AdaBoost
## Gradient Boosting
## Histogram-Based Gradient Boosting

# Stacking