
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

![[Pasted image 20260130145530.png]]

![[Pasted image 20260130145552.png]]

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


## Confusion Matrices
## Precision and Recall
## The Precision/Recall Trade-Off
## The ROC Curve
# Multiclass Classification

# Error Analysis

# Multilabel Classification

# Multioutput Classification