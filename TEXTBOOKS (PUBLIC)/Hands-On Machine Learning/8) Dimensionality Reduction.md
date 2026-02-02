
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

![[Pasted image 20260202155447.png]]

- High-dimensional datasets are at risk of being very sparse
- Most training instances are likely to be far away from each other
- More dimensions the training set has, the greater the risk of overfitting
- The number of training instances required to reach a given density grows exponentially when the number of dimensions

# Main Approaches for Dimensionality Reduction

## Projection

- In most real-world problems, training instances are not spread out uniformly
- All training instances lie within a lower-dimensional subspace

![[Pasted image 20260202155836.png]]

- All training instances lie close to a plane
- If we project every training instance perpendicular onto this subspace, the 2D dataset can be used
- Axes correspond to new features coordinates of the projections on the plane

![[Pasted image 20260202155946.png]]

## Manifold Learning

- Projection is not always the best approach
- Subspace can twist and turn

![[Pasted image 20260202160027.png]]

![[Pasted image 20260202160039.png]]

- The Swiss roll is an example of a 2D manifold
- A 2D manifold is a 2D shape that can be bent and twisted in a higher-dimensional space
- d-dimensional manifold is part of an n-dimensional space that locally resembled a d-dimensional hyperplane
- Manifold learning
	- Relies on the manifold assumption, also called manifold hypothesis
	- Most real-world high dimensional datasets lie close to a much lower-dimensional manifold
- The manifold assumption is often accompanied by another implicit assumption
	- Task will be simpler if expressed in the lower-dimensional space

![[Pasted image 20260202160401.png]]

- Implicit assumption does not always hold
- If the decision boundary is at $x_1 = 5$, the 2D space causes a more complex manifold


# PCA

- Principal component analysis is the most popular dimensionality reduction algorithm
- Identifies the hyperplane that lies closest to the data, then projects the data onto it

## Preserving the Variance

- Choose the right hyperplane
- A simple 2D dataset is represented, with 3 different axis
- The result of the projection of the dataset onto each of the 3 axis is

## Principal Components
## Projecting Down to d Dimensions
## Using Scikit-Learn
## Explained Variance Ratio
## Choosing the Right Number of Dimensions
## PCA for Compression
## Randomized PCR
## Incremental PCA

# Random Projection

# LLE

# Other Dimensionality Reduction Techniques