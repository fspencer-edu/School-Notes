- Clustering
	- Tools for data analysis, customer segmentation, recommender systems, search engines, image segmentation, semi-supervised learning, dimensionality reduction

- Anomaly detection/outlier detection
	- Use "normal" data to detect abnormal instances

- Density estimation
	- Probability density function (PDF) of the random process that generated the dataset
	- Instances located in a very low-density regions are likely to be anomalies


# Clustering Algorithms: k-means and DBSCAN

- Clustering is an unsupervised task

![[Pasted image 20260203110520.png]]

Application of Clustering
- Customer segmentation
- Data analysis
- Dimensionality reduction
- Feature engineering
- Anomaly detection
- Semi-supervised learning
- Search engines
- Image segmentation

- Centroid
	- Instances centred around a particular point
- Some algorithms are hierarchical or continuous

## k-means

- A technique for pulse-code modulation

![[Pasted image 20260203110753.png]]

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobc([...])
k=5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
```

- An instance's label is the index of the cluster to which the algorithm assigns this instance

```python
y_pred
array([4, 0, 1, ..., 2, 1, 0], dtype=int32)
y_pred is kmeans.labal_
True

kmeans.cluster_centers_
array([[-2.80389616,  1.80117999],
       [ 0.20876306,  2.25551336],
       [-2.79290307,  2.79641063],
       [-1.46679593,  2.28585348],
       [-2.80037642,  1.30082566]])
       
import numpy as np
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)
array([1, 1, 2, 2], dtype=int32)
```

![[Pasted image 20260203111057.png]]

- k-means algorithm does not behave well when the blobs have different diameters
- Instead of assigning each instance to a single cluster, which is called hard clustering
- Give each instance a score per cluster, called soft clustering
- Score can be the distance between the instance and centroid or a similarity score
	- Gaussian radial basis function

```python
kmeans.transform(X_new).round(2)
array([[2.81, 0.33, 2.9 , 1.49, 2.89],
       [5.81, 2.8 , 5.85, 4.48, 5.84],
       [1.21, 3.29, 0.29, 1.69, 1.71],
       [0.73, 3.22, 0.36, 1.55, 1.22]])
```

### The k-means algorithm

- Locate each cluster's centroid by computing the mean of the instances in that cluster
- Place the centroids randomly, label the instances, update the centroid and repeat
- This converges in a finite number of steps

![[Pasted image 20260203111553.png]]

- The computational complexity of the algorithm is generally linear with regards to the numbers of instances m, the number of clusters k, and the number of dimensions n
- Complexity can increase exponentially with the number of instances

- Converging at the right solution depends on the centroid initialization

![[Pasted image 20260203111733.png]]

### Centroid initialization methods

```python
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
```

- Uses a performance metric to keep the best solution
	- Inertia
		- Sum of the squared distances between the instances and their closest centroid
		- Keeps model with lowest inertia
- Score, returns negative inertia

```python
kmeans.inertia_
211.59853725816836
kmeans.score(X)
-211.5985372581684
```

- An improvement to the k-means algorithm, is the k-means++
- Smarter initialization step that tends to select centroids that are distant from one another and less likely to converge to suboptimal solutions

1. Take one centroid, from random dataset
2. Take new centroid, choosing an instance with probability
	1. Ensures instances farther away from already chosen centroids are more likely to be selected
3. Repeat the previous step until all k centroids have been chosen

### Accelerated k-means and mini-batch k-means

- Another improvement was avoiding unnecessary distance calculation
- Triangle inequality
	- Straight line is always the shortest distance between two points
- Keeping track of lower and upper bounds for distances between instances and centroids
- `algorithm="elkan"`

- Another variant of k-means algorithm was the use of mini-matches, moving the centroid slightly at each iteration

```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_cluster=5, random_state=42)
minibatch_kmeans.fit(X)
```

- If the dataset does not fit in memory, the simplest option is to use `memmap`
- Or pass one mini-batch at a time to `partial_fit()`
- The inertia in mini-batch k-mean algorithm is slightly worse

![[Pasted image 20260203112816.png]]

### Finding the optimal number of clusters

- The inertial is not a good performance metric when trying to choose k, because it keeps getting lower as we increase k

![[Pasted image 20260203112930.png]]

- The curve contains an inflexion point called the elbow
- Inertia drops quickly as k increases, but then decreases
- A more precise approach is to use a silhouette score, which is the mean silhouette coefficient over all the instances
- An instances's silhouette coefficient is equal to $(b-1)/max(a,b)$
	- a = mean distance to other instances in the same cluster
	- b = mean nearest-cluster distance
	- -1 to 1
	- 1 means that the instance is well inside its own cluster 
	- -1 means that the instances may have been assigned the wrong cluster

```python
from sklearn.metrics inport silhouette_score
silhouette_score(X, kmeans.labels_)
0.655517642572828
```

![[Pasted image 20260203113246.png]]

- 

## Limits of k-means
## Using Clustering for Image Segmentation
## Using Clustering for Semi-Supervised Learning
## DBSCAN
## Other Clustering Algorithms

# Gaussian Mixtures

## Using Gaussian Mixtures for Anomaly Detection
## Selecting the Number of Clusters
## Bayesian Gaussian Mixture Models
## Other Algorithms for Anomaly and Novelty Detection
