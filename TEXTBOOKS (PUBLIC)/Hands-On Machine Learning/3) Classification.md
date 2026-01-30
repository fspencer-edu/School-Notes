
- The most common supervised learning tasks are regression and classification
	- Predicting values and classes

# MNIST

- 70,000 small images of digits handwritten

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
```

- Dataset package contains mostly 3 types of functions
	- `fetch_*`
	- `load_*`
	- `make_*`

- Set `as_frame=False` to get the data as NumPy arrays instead of DataFrames

```python
X, y = mnist.data, mnist.target
X
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
>>> X.shape
(70000, 784)
y
array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)
y.shape
(70000,)
```

- 70,000 image, with 784 features
	- Each image is 28 x 28 pixels
	- Each feature represents one pixel intensity, 0 to 255

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
	image = image_data.reshape(28, 28)
	plt.imshow(image, cmap="binary")
	plt.axis("off")
	
some_digit = X[0]
plot_digit(some_digit)
plt.show()

y[0]
'5'
```

<img src="/images/Pasted image 20260130145530.png" alt="image" width="500">

<img src="/images/Pasted image 20260130145552.png" alt="image" width="500">

- Create a test set and set it aside for inspecting
- First 60000 is training set, 10,000 is for the test set

```python
x_train, x_test, y_train_, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

- Learning algorithms are sensitive to the order of the training instances
	- Perform poorly if they get many similar instances in a row

# Training a Binary Classifier

- Identify one digit
- 5 detection is an example of a binary classifier

```python
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```
- Stochastic gradient descent (SGD) classifier
	- Capable of handing large datasets
	- Trains instances independently
	- Online learning

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

sgd_clf.predict([some_digit])
array([True])
```

- The classifier guesses that this image represents a 5

# Performance Measures

- Evaluating a classifier is harder than evaluating a regressor

## Measuring Accuracy Using Cross-Validation

- A good way to evaluate a model is to use cross-validation
- k-fold cross-validation means splitting the training set into k fold, then training the model k times, holding out a different fold each time

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")
array([0.95035, 0.96035, 0.9604 ])
```

- 95% accuracy on all cross-validation folds
- Look at dummy classifier to classify each image in the most frequent class

```python
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier()
dummy_clf.fit(x_train, y_train_5)
print(any(dummy_clf.predict(x_train))) # prints False: no 5s detected
cross_val_score(dummy_clf, x_train, y_train_5, cv=3, scoring="accuracy")
array([0.90965, 0.90965, 0.90965])
```
- Accuracy of 90%
- Only about 10% of the images are 5
- Therefore, it is guessing that everything is a 5
- Accuracy is not the preferred performance measure for classifiers
- A better performance classifier is the confusion matrix (CM)

- Implement custom CV

```python
from sklearn.model_selection import StatifiedKFold
from sklearn.base import clone

skfolds = StatifiedKFold(n_splits=3)

for train_index, test_index in skfolds.split(x_train, y_train_5):
	clone_clf = clone(sgd_clf)
	x_train_folds = x_train[train_index]
	y_train_folds = y_train_5[train_index]
	x_test_fold = x_train[test_index]
	y_test_fold = y_train_5[test_index]
	
	clone_clf.fit(x_train_fold, y_train_fold)
	y_pred = clone_clf.predict(x_test_fold)
	n_correct = sum(y_pred == y_test_fold)
	print(n_correct / len(y_pred)) # prints 0.95035, 0.96035, and 0.9604
```

- A stratified sampling produces folds that contain a representative ratio of each class
- At each iteration the code creates a clone of the classifier, trains that clone on the training folds, and makes predictions on the test fold
- Count the number of correct prediction, and outputs the ratio of correct predictions

## Confusion Matrices

- The idea of the confusion matrix is to count the number of times instances of class A are classified as B, for all A/B pairs
- Compute the CM by obtaining a set of prediction

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
```

- Performs k-fold CV, but instead of returning the evaluation scores, it returns the prediction made on each test fold
- Get a clean prediction for each instance in the training set

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
cm
array([[53892,   687],
       [ 1891,  3530]])
```
- Each row represents an actual class
- Each column represents a predicted class
- First row are non-5 images
	- 53892 true negatives
	- 687 are false positives/type I errors
- Second row considers the image is a 5
	- 1891 are false negatives
	- 3530 are true positives
- A perfect classifier would only have true positives and true negatives
- Accuracy of positive predictions
	- Precision

**Precision**

<img src="/images/Pasted image 20260130151924.png" alt="image" width="500">

- Create a classifier that always makes negative predictions, except for one single positive prediction on the instance it's most confident about
- Recall also called sensitivity or the true positive rate (TPR)
	- Ratio of positive instances that are correctly detected by the classifier

**Recall**

<img src="/images/Pasted image 20260130152108.png" alt="image" width="500">


<img src="/images/Pasted image 20260130152132.png" alt="image" width="500">


## Precision and Recall

```python
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) # == 3530 / (687 + 3530)
0.8370879772350012
recall_score(y_train_5, y_train_pred) # == 3530 / (1891 + 3530)
0.6511713705958311
```
- Combine precision and recall into a single metric called the $F_1$ score
- Harmonic mean of precision and recall
- Regular mean treats all values equally, the harmonic mean gives much more weight to low values

<img src="/images/Pasted image 20260130152535.png" alt="image" width="500">


```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
0.7325171197343846
```

- $F_1$ scores favour classifiers that have similar precision and recall
- Precision/recall trade-off
	- Increasing precision reduces recall, and vice versa

## The Precision/Recall Trade-Off

- For each instance the SGD classifier computes a score based on a decision function
- If the score is greater than a threshold, it assigns the instance to the positive class, otherwise the negative class

<img src="/images/Pasted image 20260130153059.png" alt="image" width="500">

- The decision threshold is positioned at the central arrow
- Calling `predict()` method, set the threshold

```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores
array([2164.22030239])
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
array([True])

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
array([False])
```

- Raising the threshold, decreases recall
- Use CV function to get the scores of all instances in the training set

```python
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3,
				method="decision_function")
				
				
from sklearn.metrices import precision_recall_curve
precisions, recalls, thesholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
[...]  # beautify the figure: add grid, legend, axis, labels, and circles
plt.show()
```

<img src="/images/Pasted image 20260130153832.png" alt="image" width="500">

- Precision may sometimes go down when you raise the threshold
- Whereas recall can only go down
- As the threshold value, precision is near 90%, recall is around 50%
- Plot precision directly against recall

```python
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall Curve")
plt.show()
```
<img src="/images/Pasted image 20260130154043.png" alt="image" width="500">


- Precision starts to fall around 80% recall
- Select a threshold at around 60% recall
- Search for the lowest threshold that gives 90% precision

```python
idx_for_90_precision = (precisions >=90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threhold_for_90_precision
3370.0194991439557

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
precision_score(y_train_5, y_train_pred_90)
0.9000345901072293
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision
0.4799852425751706
```

- 90% precision classifier
- A high precision classifier is not useful if the recall is too low

## The ROC Curve

- The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers
- Similar to precision/recall curve, but instead of plotting, the ROC curve plots the true positive rate, against the false positive rate (FPR)
- FPR, also called fallout, is the ratio of the negative instances that are incorrectly classified as positive
- True-negative rate (TNR), is the ratio of negative instances that are correctly classified as negative
- TNR is also called specificity

# Multiclass Classification

# Error Analysis

# Multilabel Classification

# Multioutput Classification