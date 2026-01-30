
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

- ROC curve plots sensitivity (recall) versus 1 - specificity

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)
```

- Plot FPR against the TPR
- To find the point that corresponds to 90% precision, look for the index of the desired threshold

```python
idx_for_threshold_at_90 = (threshold <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = trp[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random Classifier's ROC Curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.show()
```

![[Pasted image 20260130161559.png]]

- The higher the TPR, the more FPR
- Dotted line represents the ROC curve of a purely random classifier
- A good classifier stays as far aways from that line as possible
- Compare classifiers by measuring the area under the curve (AUC)
- A perfect ROC AUC is equal to 1, whereas a purely random classifier is 0.5

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
0.9604938554008616
```

- ROC curve is similar to precision/recall (PR) curve
- Prefer the PR curve then the positive class is rare or when false positives are more important than false negatives
- Otherwise, use ROC curve

```python
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3,
			method="predict_proba")
y_probas_forest[:2]
array([[0.11, 0.89],
       [0.99, 0.01]])
```

- The model predicts that the first image is positive with 89%, and predicts the second image is negative with a 99%
- These are estimated probabilities, not actual probability

```python
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
		y_train_5, y_scores_forest
)
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.show()
```

![[Pasted image 20260130162819.png]]

- PR curve look much better than the SGD
- $F_1$ score and ROC AUC score are better

```python
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
f1_score(y_train_5, y_train_pred_forest)
0.9242275142688446
roc_auc_score(y_train_5, y_scores_forest)
0.9983436731328145
```

# Multiclass Classification

- Multiclass classifiers (multi-nominal classifiers) can distinguish between more than two classes
	- `LogisticRegression, RandomForestClassifier, GaussianNB`
- Create a system that can classify the digit images into 10 classes (0-9)
- Get a decision score from each classifier for that image, and select the class whose classifier outputs the highest score
- One-versus-the-rest (OvR)
	- Also called one-versus-all (OvA)

- Train a binary classifier for every pair of digits
	- One-versus-one (OvO)
	- $N x (N-1)/2$ classifiers
- Some algorithms scale poorly with the size of the training set
- OvO is preferred because it is faster to train many classifiers on small training sets
- For most binary classification algorithms, OvR is preferred

```python
from sklearn.svm import SVC

svc_clf = SVC(random_state=42)
svm_clf.fit(x_train[:2000], y_train[:2000])

svm_clf.predict([some_digit])
array(['5'], dtype=object)
```
- The `SVC` using the original target classes
- Uses OvO strategy and 45 binary classifiers
- Returns 10 scores per instance, once per class
- Each class get a score equal to the number of won duels plus or minus a small tweak

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)
array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,
         4.82]]
class_id = some_digit_scores.argmax()
class_id
5

svm_clf.classes_
array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
svm_clf.classes_[class_id]
'5'
```

- The index of each class in the `classes_` array matches the class itself
- To force OvO or OvR
- Create an instance and pass a classifier to its constructor

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(x_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])
array(['5'], dtype='<U1')
len(ovr_clf.estimators_)
10

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
array(['3'], dtype='<U1')
```

- The SGD classifier on a multiclass dataset is incorrect

```python
sgd_clf.decision_function([some_digit]).round()
array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
        -16148.,  -4604., -12051.]])
>>> cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
array([0.87365, 0.85835, 0.8689 ])
```
- Classifiers are not very confident about its prediction
	- All scores are negative

- After evaluating the mode, over 85.8% on all test folds
- Improve the accuracy may scaling the inputs

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
array([0.8983, 0.891 , 0.9018])
```

# Error Analysis

- A coloured diagram of the confusion matrix is easier to analyze

![[Pasted image 20260130164604.png]]

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

- Confusion matrix (top left) shows most images are on the main diagonal, classified correctly
- Normalize the confusion matrix by dividing each value by the total number of images in the corresponding class (top right)

```python
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
			normalize="true", values_format=".0%")
plt.show()
```

- Add a zero weight on the correct predictions (bottom left)
- Column for class 8 is bright, which confirms that many images got misclassified as 8s

```python
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
				sample_weight=sample_weight,
				normalize="true", values_format=".0%")
plt.show()
```

- Normalize the confusion matrix by column rather than by row (bottom right)
- CM helps to identify ways to improve the classifier

![[Pasted image 20260130165302.png]]

- Most misclassified images seem obvious
- Data augmentation
	- Force a model to learn to be more tolerant to variations of data


# Multilabel Classification

```python
from numpy as np
from sklearn.neighbors import KNneightborsClassifier

y_train_large = (y_train >= '7')
y_train_odd (y_train.astype('int8') % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)
```
- Creates an array containing two target labels fro each digit image
	- First indicates whether or not the digit is large
	- Second indicates whether or not it is odd
- Code creates a KN classifier instance, which support multilabel classification, and train this model using the multiple target array

```python
knn_clf.predict([some_digit])
array([[False, True]]) # 5 is large-F, Odd-T
```

- Measure $F_1$ score for each individual label, then compute the average score

```python
y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
0.976410265560605
```
- This approach assumes all labels are equally important
- Given each label a weight equal to its support
	- The number of instances with that target label
- To use a classifier that does not natively support multilabel classification, such as `svc`, train one model per label
- Models can be organized in a chain
	- When a model makes a prediction, it uses the input features plus all the predictions of the models that come before it
- Use the true labels for training, feeding each model the appropriate labels depending on their position in the chain
- If `cv` is set, cross-validation will be used to get "clean" prediction from each trained model for every instance in the training set, and will be used to train all the models later in the chain

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(x_train[:2000], y_multilabel[:2000])

chain_clf.predict([some_digit])
array([[0., 1.]])
```

# Multioutput Classification

- A generalization of multilabel classification where each label can be multiclass
- Build a system that removes noise from images
- The line between classification and regression is blurring
	- Predicting pixel intensity is more akin to regression than to classification
- Multioutput systems are not limited to classifications tasks
- Add noise to the MNIST images

```python
np.random.seed(42)
noise = np.random.randint(0, 100, (len(x_train), 784))
x_train_mod = x_train + noise
noise = np.random.randint(0, 100, (len(x_test), 784))
x_test_mode = x_test + noise
y_train_mod = x_train
y_test_mod = x_test
```

![[Pasted image 20260130170906.png]]

```python
knn_clf = KNeighborsClassifier()
nkk_clf.fit(x_train_mod, y_train_mod)
clear_digit = knn_clf.predict([x_test_mod[0]])
plot_digit(clean_digit)
plt.show()
```