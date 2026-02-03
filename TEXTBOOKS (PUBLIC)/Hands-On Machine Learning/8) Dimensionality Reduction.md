
- Curse of dimensionality
- Reduce the number of features
- Two neighbouring pixels are often highly correlated
	- Merge into a single pixel
- Cause some information loss
- Speeds up training
- Used for data visualization
	- Plot a condensed view of a high-dimensional training set on a graph
- 2 approaches to dimensionality reduction
	- Projection
	- Manifold learning
- Techniques
	- PCA
	- Random projection
	- Locally linear embedding (LLE)

# The Curse of Dimensionality

- Base 4D is difficult to picture

<img src="/images/Pasted image 20260202155447.png" alt="image" width="500">

- High-dimensional datasets are at risk of being very sparse
- Most training instances are likely to be far away from each other
- More dimensions the training set has, the greater the risk of overfitting
- The number of training instances required to reach a given density grows exponentially when the number of dimensions

# Main Approaches for Dimensionality Reduction

## Projection

- In most real-world problems, training instances are not spread out uniformly
- All training instances lie within a lower-dimensional subspace

<img src="/images/Pasted image 20260202155836.png" alt="image" width="500">

- First 2 PC are the projection plane, third PC is the axis orthogonal to that plane

- All training instances lie close to a plane
- If we project every training instance perpendicular onto this subspace, the 2D dataset can be used
- Axes correspond to new features coordinates of the projections on the plane

<img src="/images/Pasted image 20260202155946.png" alt="image" width="500">

## Manifold Learning

- Projection is not always the best approach
- Subspace can twist and turn

<img src="/images/Pasted image 20260202160027.png" alt="image" width="500">

<img src="/images/Pasted image 20260202160039.png" alt="image" width="500">

- The Swiss roll is an example of a 2D manifold
- A 2D manifold is a 2D shape that can be bent and twisted in a higher-dimensional space
- d-dimensional manifold is part of an n-dimensional space that locally resembled a d-dimensional hyperplane
- Manifold learning
	- Relies on the manifold assumption, also called manifold hypothesis
	- Most real-world high dimensional datasets lie close to a much lower-dimensional manifold
- The manifold assumption is often accompanied by another implicit assumption
	- Task will be simpler if expressed in the lower-dimensional space

<img src="/images/Pasted image 20260202160401.png" alt="image" width="500">

- Implicit assumption does not always hold
- If the decision boundary is at $x_1 = 5$, the 2D space causes a more complex manifold

# PCA

- Principal component analysis is the most popular dimensionality reduction algorithm
- Identifies the hyperplane that lies closest to the data, then projects the data onto it

## Preserving the Variance

- Choose the right hyperplane
- A simple 2D dataset is represented, with 3 different axis
- The result of the projection of the dataset onto each of the 3 axis is shown on the right
- The projection is a solid line, and preserves the max variance, while the projection on dotted line preserves the little variance

<img src="/images/Pasted image 20260202160712.png" alt="image" width="500">

- Select the axis that preserves the max variance

## Principal Components

- PCA identifies the axis that accounts for the largest amount of variance in the training set
- The ith axis is the principal component of the data
- The first PC is the axis on $c_1$, second PC is $c_2$
- For each principal component, PCA finds a zero-centred unit vector pointing in the direction of the PC
- Since two opposing unit vectors lie on the same axis, the direction of the unit vector returned by PCA is not stable

- Singular value decomposition (SVD)
	- Decomposes the training set matrix $X$ into the matrix multiplication of $U\sum V^T$

**Principal Component Matrix**

<img src="/images/Pasted image 20260202161254.png" alt="image" width="500">

```python
import numpy as np
X = [...]
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt[0]
c2 = Vt[1]
```

- Obtain all the principal components of the 3D training set, the extracts the two unit vectors that define the first 2 PCs

- PCA assumes the the dataset is centred around the origin

## Projecting Down to d Dimensions

- After identifying all the principal components, reduce the dimensionality of the dataset by projecting it onto the hyperplane defined by the first $d$ PC
- A hyperplane ensures the projection will preserve as much variance as possible
- To project the training set onto the hyperplane and obtain a reduced dataset, compute the matrix multiplication of the training set matrix $X$ by $W_d$

**Projecting the training set down to d dimensions**

$X_{dProj} = XW_d$

```python
# projects the training set onto the plane defined by first 2 PC
W2 = Vt[:2].T
X2D = X_centered @ W2
```

## Using Scikit-Learn

- `PCA` uses SVD to implement

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

- After fitting `PCA` transformer, `components_` holds the transpose of $W_d$
	- Contains one row for each of the $d$ PCs

## Explained Variance Ratio

- Explained variance ratio
	- Indicates the proportion of the dataset's variance that lies along each principal component

```python
pca.explained_variance_ratio_
array([0.7578477 , 0.15186921])
```

- Therefore, 76% of the dataset's variance lies along the first PC, 15% along the second PC
- 9% on the third

## Choosing the Right Number of Dimensions

- Simpler to choose the number of dimensions that add up to sufficiently large portion of variance, ~95%

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000], mnist.targett[60_000:]

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
```

- Loads and splits the MNIST dataset
- Performs PCA without reducing dimensionality, them computes the min number of dimensions required to preserve 95% of training set variance
- Instead of specifying the PCs to preserve, set `n_components` to be a float, indicating ratio of variance to preserve

```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
pca.n_components_
154
```

- Another option is to plot the explained variance as a function of the number of dimensions
- There is an elbow in the curve, where the explained variance stops growing

<img src="/images/Pasted image 20260202161543.png" alt="image" width="500">

- Tune the number of dimensions

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

clf = make_pipeline(PCA(random_state=42),
			RandomForestClassifier(random_state=42))
param_distrib = {
		"pca__n_components": np.arange(10, 80),
		"randomforestclassifier__n_estimators": np.arange(50, 500)
}
rnd_search = RandomizedSearchCV(clf, parm_distrib, n_iter=10,
		random_state=42)
rnd_search.fit(X_train[:1000], y_train[:1000])

print(rnd_search.best_params_)
{'randomforestclassifier__n_estimators': 465, 'pca__n_components': 23}

```

- Reduce the number of dimensions from 784 to 23

## PCA for Compression

- Decompress the reduced dataset back to 784 dimensions by applying the inverse transformation of the PCA projection
- Reconstruction error
	- Mean squared distance between the original and reconstructed data

```python
X_recovered = pca.inverse_transform(X_reduced)
```

<img src="/images/Pasted image 20260202161559.png" alt="image" width="500">
**PCA Inverse Transformation, back to the original number of dimensions**

$X_{recovered} = X_{dProj}W_d^T$


## Randomized PCR

- Scikit-Learn uses a stochastic algorithm called randomized PCA that quickly finds an approximation of the first d principal components
	- Complexity is $O(m t\times d^2) + O(d^3)$
	- Faster than the full SVD approach

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

-  By default the `svd_solver` is set to "auto"

## Incremental PCA

- Incremental PCA (IPCA) splits the training set into mini-batches and feed these in one mini-batch at a time
- Useful for applying PCA online

```python
from sklearn.decomponsition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
	inc_pca.partial_fit(X_batch)
	
X_reduced = inc_pca.transfom(X_train)
```

- Splits the MNIST training set into 100 mini batches and feeds them into the incremental class to reduce the dimensionality
- Alternatively, use `memmap` class, to manipulate large array stored in a binary file on disk as if it were in memory
- Create a memory-mapped file and copy the MNIST training set to it
- Load training data chunk by chunk and save each chunk to the right part of the memmap array

```python
filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mmap[:] = X_train
X_mmap.flush()

X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
batch_size = X_mmap.shape[0]
inc_pca = IncrementalPCA(n_componoents=154, batch_size=batch_size)
inc_pca.fit(X_mmap)
```

- Only the raw binary data is saved to disk
- Specify the data type and shape of the array when loading it
- For high-dimensional datasets, PCA can to slow

# Random Projection

- Random projection algorithm projects the data to a lower-dimensional space using a random linear projection
- Two similar instances will remain similar after projection, and two very different instances will remain very different

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim
m, ε = 5_000, 0.1
d = johnsom_lindenstrauss_min_dim(m, eps=ε)
d
7300
```

- Generate a random matrix P of shape $[d, n]$, where each item os sampled randomly frpm Gaussian distribution with mean 0 and variance 1/d, and use it to project a dataset from n dimensions down to d

```python
n = 20_000
np.random.seed(42)
P = np.random.randn(d, n) / np.sqrt(d)
X = np.random.randn(m, n)
X_reduced = X @ P.T

from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd.proj.fit_transform(X
```

- `SparseRandomProjection`
	- Determines the target dimensionality in the same way, generates a random matrix of the same shape, and performs the projection identically
	- Uses less memory
- The ratio r of non-zero items in the spare random matrix is called its density

```python
components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)
X_recovered = X_reduced @ components_pinv.T
```

- Computing the pseudo-inverse may take a long time if the components matrix is large
- Complexity of `pinv()` is $O(dn^2)$
- Locality sensitive hashing (LSH)
	- Used in search engines to group similar documents
# LLE

- Locally linear embedding (LLE) is a non-linear dimensionality reduction (NLDR) technique
- Manifold techniques that does not rely on projections
- Measures how each training instance linearly relates to its nearest neighbours, then looks for a low-dimensional representation of the training set where these local relationships are best preserved
- Used for unrolling twisted manifolds

```python
from skilearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbours=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)
```

- 

<img src="/images/Pasted image 20260202161616.png" alt="image" width="500">


# Other Dimensionality Reduction Techniques

<img src="/images/Pasted image 20260202161624.png" alt="image" width="500">