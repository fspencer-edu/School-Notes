
- A support vector machine (SVM) is a powerful and versatile machine learning model
- Linear or non-linear classification, regression, and novelty detection
- Used with small to medium non-linear datasets

# Linear SVM Classification

![[Pasted image 20260202112226.png]]

- Left plot
	- Decision boundaries of 3 possible linear classifiers
- Right plot
	- SVM classifier
	- Fitting the widest possible street
	- Large margin classification

- The more training instances "off the street" will not affect the decision boundary
- Determined by the instances located on the edge of the street
	- Support vectors
- SVM are sensitive to the feature scales

![[Pasted image 20260202112511.png]]


## Soft Margin Classification

- Hard margin classification
	- Strictly impose that all instances must be off the street on the correct side
	- Only works if data is linearly separable
	- Sensitive to outliers

![[Pasted image 20260202112619.png]]

- Soft margin classification
	- Keep street at large as possible and limit the margin violations
- Specify hyperparameters, including regularization hyperparameters $C$

![[Pasted image 20260202112824.png]]

- If the SVM model is overfitting, try regularizing by reducing C

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)

svm_clf = make_pipeline(StandardScaler(),
			LinearSVC(C=1, random_state=42))
svm_clf.fit(X, y)

# Use model to make predictions
X_new = [[5.5, 1.7], [5.0, 1.5]]
svm_clf.predict(X_new)
array([ True, False])
svm_clf.decision_function(X_new)
array([ 0.66163411, -0.22036063])
```
- The first plant is classified as an Iris virginica, while the second is not

# Nonlinear SVM Classification

- Many datasets are not close to being linearly separable
- Add more features, such as polynomial features

![[Pasted image 20260202113237.png]]

- Create a pipeline containing a `PolynomialFeatures` transformer
- Use moons dataset, a binary classification in which the data points are shaped as two interleaving crescent moons

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
	PolynomialFeatures(degere=3),
	StandardScaler(),
	LinearSVC(C=10, max_iter=10_000, random_state=42)
)

polynomial_svm_clf.fit(X, y)
```

![[Pasted image 20260202113513.png]]

## Polynomial Kernel

- Adding polynomial features is simple
- At a low polynomial degree, this method cannot deal with complex datasets
- With a high polynomial degree, it creates a high number of features
- With SVM use kernel trick to make it possible to get the same results as if you has added many polynomial features, without adding them
- No combinatorial explosion of number of features

```python
from sklearn.svm import SVC
poly_kernel_svm_clf = make_pipeline(StandardScaler(),
			SVC(kernel="poly", degree=3, coef0=1, C=5))
poly_kernel_svm_clf.fit(X, y)
```

- This code trains an SVM classifier using a third-degree polynomial kernel

![[Pasted image 20260202113823.png]]


## Similarity Features

- Another technique for non-linear problems is to add features computer using a similarity function, which measure how much each instance resembles a particular landmark

![[Pasted image 20260202114019.png]]

- Simplest approach is to create a landmark at the location of each and every instance in the dataset

## Gaussian RBF Kernel

- The similarity features method can be used with machine learning algorithms

```python
rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X, y)
```

![[Pasted image 20260202114248.png]]

- Increasing gamma makes the bell-shape curve narrower
- Some kernels are specialized for specific data structures
	- String kernels are sometimes used when classifying text documents or DNA sequences

- 

## SVM Classes and Computational Complexity

# SVM Regression

# Under the Hood of Linear SVM Classifiers

# The Dual Problem

## Kernelized SVMs