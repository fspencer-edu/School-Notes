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

- A more informative visualization is obtained when every instance's silhouette coefficient is plotted, and sorted by the clusters they are assigned to and by the value of the coefficient
	- Silhouette diagram
- Each diagram contain one knife shape per cluster
	- The shapes height indicates the number of instances in the cluster
	- The width represents the sorted silhouette coefficients of the instances in the cluster
	- Wider is better
- Vertical dash lines represent the mean silhouette score for each number of clusters

![[Pasted image 20260203113624.png]]

- Shows use that k=5, is a good choice to get clusters of similar sizes
## Limits of k-means

- k-means does not behave well when the clusters have varying sizes, different densities, or non-spherical shapes

![[Pasted image 20260203113742.png]]

- Scale input features before running k-means, or the clusters will be stretches


## Using Clustering for Image Segmentation

- Image segmentation is the task of partitioning an image into multiple segments
	- Colour segmentation
	- Semantic segmentation
	- Instance segmentation

- Semantic or instance segmentation is achieved using complex architectures based on convolutional neural networks

- Pillow package is used as a python imaging library

```python
import PIL
image = np.asarray(PIL.Image.open(filepath))
image.shape
(533, 800, 3)
```
- (height, width, colour channels)

```python
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```

- Reshapes the array to a long list of RGB colours, then clusters the colours using k-means with 8 clusters

![[Pasted image 20260203114252.png]]



## Using Clustering for Semi-Supervised Learning

- Semi-supervise learning

```python
# load and split the dateset
from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]

# train a logistic regression model on 50 labeled instances
from sklearn.linear_model import LogisticRegression
n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labaled])

# measure accuracy
log_reg.score(X_test, y_test)
0.7481108312342569
```

- Training the model on a full set, will have a higher accuracy
- Cluster the training set into 50 clusters
- Then for each cluster, find the image closest to the centroid
	- Representative images

```python
k = 50
kmeans = KMeans(n_cluster=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digits_idx]
```

![[Pasted image 20260203114758.png]]

```python
# manually label
y_representative_digits = np.array([1, 3, 6, 0, 7, 9, 2, ..., 5, 1, 9, 9, 3, 7])

# Performance
log_reg = LogisticRegression(max_iter=10_000)
>>> log_reg.fit(X_representative_digits, y_representative_digits)
>>> log_reg.score(X_test, y_test)
0.8488664987405542
```

- Label representative instances rather than random instances
- Label propagation
	- Propagated the labels to all the other instances in the same cluster

```python
y_train_propagated = np.empty(len(X_train), dtype=np.int64)
for i in range(k):
	y_train_propagated[kmeans.labales_ == i] = y_representative_digits[i]
	
# train
log_reg = LogisticRegression()
>>> log_reg.fit(X_train, y_train_propagated)
>>> log_reg.score(X_test, y_test)
0.8942065491183879

# train again, ignoring 1% fathest from their cluster centre

percentile_closest = 99
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
	in_cluster = (kmeans.labels_ == i)
	cluster_dist = X_cluster_dist[in_cluster]
	cutoff_distance = np.percentile(cluster_dist, percentile_closest)
	above_cutoff = (X_cluster_dist > cutoff_distance)
	X_cluster_dist[in_cluster & above_cutoff] = -1
	
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# train
log_reg = LogisticRegression(max_iter=10_000)
>>> log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
>>> log_reg.score(X_test, y_test)
0.9093198992443325

(y_train_partially_propagated == y_train[partially_propagated]).mean()
0.9755555555555555
```

- Propagated labels have a accuracy of 97%

- Scikit-learn classes to propagate labels automatically
	- `LabelSpeading`
	- `LabelPropagation`

- Active learning
	- Human expert interacts with the learning algorithm, providing labels for specific instances when the algorithm requests them
- Uncertainty sampling
	- The model is trained on the labeled instances, and this model is used to make prediction on all the unlabeled instances
	- Instances for the model the model is more uncertain are given to the expert of labeling
	- Iterative this process until the performance improvement stops being worth the labeling effort

- Other active learning strategies
	- Labeling the instances that would results in the largest model change or the largest drop in the model's validation error 


## DBSCAN

- Popular clustering algorithm that illustrates an approach based on local density estimation
- Identifies clusters of arbitrary shapes
- Density-based spatial clustering of applications with noise (DBSCAN)
	- Defines clusters as continuous regions of high density

- For each instance, the algorithm counts how many instances are located within a small distance $\epsilon$
	- $\epsilon$-neighbourhood
- If an instance has at least `min_samples` instances in its $\epsilon$-neighbourhood, then it is considered a core instance
- All instances in the neighbourhood of a core instance belong to the same cluster
- 

## Other Clustering Algorithms

# Gaussian Mixtures

## Using Gaussian Mixtures for Anomaly Detection
## Selecting the Number of Clusters
## Bayesian Gaussian Mixture Models
## Other Algorithms for Anomaly and Novelty Detection
