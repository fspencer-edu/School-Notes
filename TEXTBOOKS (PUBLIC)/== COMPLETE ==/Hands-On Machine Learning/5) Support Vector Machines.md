
- A support vector machine (SVM) is a powerful and versatile machine learning model
- Linear or non-linear classification, regression, and novelty detection
- Used with small to medium non-linear datasets

# Linear SVM Classification

<img src="/images/Pasted image 20260202112226.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260202112511.png" alt="image" width="500">


## Soft Margin Classification

- Hard margin classification
	- Strictly impose that all instances must be off the street on the correct side
	- Only works if data is linearly separable
	- Sensitive to outliers

<img src="/images/Pasted image 20260202112619.png" alt="image" width="500">

- Soft margin classification
	- Keep street at large as possible and limit the margin violations
- Specify hyperparameters, including regularization hyperparameters $C$

<img src="/images/Pasted image 20260202112824.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260202113237.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260202113513.png" alt="image" width="500">

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

<img src="/images/Pasted image 20260202113823.png" alt="image" width="500">


## Similarity Features

- Another technique for non-linear problems is to add features computer using a similarity function, which measure how much each instance resembles a particular landmark

<img src="/images/Pasted image 20260202114019.png" alt="image" width="500">

- Simplest approach is to create a landmark at the location of each and every instance in the dataset

## Gaussian RBF Kernel

- The similarity features method can be used with machine learning algorithms

```python
rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X, y)
```

<img src="/images/Pasted image 20260202114248.png" alt="image" width="500">

- Increasing gamma makes the bell-shape curve narrower
- Some kernels are specialized for specific data structures
	- String kernels are sometimes used when classifying text documents or DNA sequences

- Always try the linear kernel first
- Try kernelized SVM, starting with Gaussian RBF kernel 

## SVM Classes and Computational Complexity

- `LinearSVC` scales almost linearly with the number of training instances and the number of features
- Training time complexity is $O(m \times n)$

- `SVC` implements an algorithm that supports the kernel trick
- Training time complexity is between $O(m^2 \times n)$ and $O(m^3 \times n)$
- Best for small or medium-sized non-linear training sets

- `SGDClassifier` also performs large margin classification by default
- For training it uses the stochastic GD
- Uses little memory

<img src="/images/Pasted image 20260202120126.png" alt="image" width="500">


# SVM Regression

- To use SVMs for regression instead of classification, tweak the obective
- Instead of trying to fit the largest possible street between two classes while limiting margin violations, SVM regression tries to fit as many instances as possible on the street while limiting margin violations


<img src="/images/Pasted image 20260202120216.png" alt="image" width="500">


- Reducing $\epsilon$ increase the number of support vectors, which regularizes the model
- If your add more training instances within the margin, it will not affect the model's prediction, $\epsilon$-insensitive

```python
from sklearn.svm import LinearSVR

X, y = [...]
svm_reg = make_pipeline(StandardScaler(),
			LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X, y)
```

- For non-linear regression tasks, use a kernelized SVM model

<img src="/images/Pasted image 20260202120548.png" alt="image" width="500">

```python
from sklearn.svm import SVR

X, y = [...]
svm_poly_reg = make_pipeline(StandardScaler(),
		SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1))
svm_poly_reg.fit(X, y)
```
- `SVR` class is the regression equivalent of the `SVC` class
- SVMs can also be used for novelty detection


# Under the Hood of Linear SVM Classifiers

- A linear SVM classifier predicts the class of a new instance x by first computing the decision function, where $x_0$ is the bias features
- If the result is positive, then the predicted class $\hat{y}$ is the positive class, otherwise, it is the negative class

- Up to now, used the convention of putting all the model parameters in one vector $\theta$
	- Including the bias term, and input feature weights
	- Adding $x_0 = 1$ to all instances
- Another convention is to separate the bias term b ($\theta_0$) and the feature weights vector $w$
	- No bias feature needs to be added to the input feature vectors, and the linear SVMs function is $w^Tx + b = w_1x_1 + ...$

- Find the weight vectors and the bias term to make the street at wide as possible, while limiting the number of margin violations


<img src="/images/Pasted image 20260202120959.png" alt="image" width="500">

- Avoid margin violations
	- Decision function should be greater that 1 for all positive training instances and lower than -1 for negative training instances

**Hard margin linear SVM classifier objective**

<img src="/images/Pasted image 20260202122046.png" alt="image" width="500">


- Optimization algorithms often work much better on differentiable functions
- Slack variables, $\zeta$
	- Measures how much the $i$ instance is allowed to violate the margin
	- Make stack variables as small as possible, and make $1/2w^tw$ as small as possible to increase margin
- `C` is used to define the trade-off between these two objects


**Soft margin linear SVM classifier objective**

<img src="/images/Pasted image 20260202122311.png" alt="image" width="500">

- The hard and soft margin problems are both convex quadratic optimization problems with linear constraints
	- Quadratic programming (QP) problems
- Using a QP solver is one way to train an SVM
- Another way is to use a GD to minimize the hinge loss or the squared hinge loss
- The further away an instance is from the correct side of the margin, the higher the loss
	- Grows linearly for the hinge loss, and quadratically for the squared loss
- If the dataset is clean, it tends to converge faster

<img src="/images/Pasted image 20260202122535.png" alt="image" width="500">


# The Dual Problem

- Given a constrained optimization problem, primal problem, it is possible to express a different but closely related problem, called its dual problem
- Gives a lower bound to the solution, but under some conditions it can have the same solution as the primal problem
- SVM meets these conditions, and is used to solve primal or dual problems
	- Same solution

**Dual form of the linear SVM objective**

<img src="/images/Pasted image 20260202122741.png" alt="image" width="500">

- Once you find the vector $\hat{\alpha}$ that minimizes the equation, use dual solution to find $\hat{w}$ and $\hat{b}$ that minimize the primal problem


**From the dual solution to the primal solution**

<img src="/images/Pasted image 20260202122905.png" alt="image" width="500">


- The dual problem is faster to solve than the primal one when the number of training instances is smaller than the number of features
- Kernel trick is possible on dual, not primal

## Kernelized SVMs

- The transformed vector is 32 instead of 2D
- 

**Second-degree polynomial mapping**

<img src="/images/Pasted image 20260202123000.png" alt="image" width="500">

**Kernel trick for a second-degree polynomial mapping**

<img src="/images/Pasted image 20260202123011.png" alt="image" width="500">

- The dot product of the transformed vector is equal to the square of the dot product of the original vectors
- In ML, a kernel is a function capable of computing the dot product $\phi(a)^T\phi(b)$

**Common kernels**

<img src="/images/Pasted image 20260202123219.png" alt="image" width="500">


**Making predictions with a kernelized SVM**

<img src="/images/Pasted image 20260202123246.png" alt="image" width="500">

**Using the kernel trick to compute the bias term**

<img src="/images/Pasted image 20260202123258.png" alt="image" width="500">

- 