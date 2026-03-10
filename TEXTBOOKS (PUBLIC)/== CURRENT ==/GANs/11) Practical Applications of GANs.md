
- GANs can be useful in situations with limited data


# GANs in medicine

- Use of GANs to produce synthetic data to enlarge a training dataset to help improve diagnostic accuracy

## Using GANs to improve diagnostic accuracy

- SGAN can be used if there is a small amount of labeled data, and a large amount of unlabeled data
- In medical applications, there may only be a small set of labeled data
- Data augmentation is used to help increase the dataset size
	- Scaling
	- Translations
	- Rotations
- Enriching a dataset with synthetic examples can help researchers

## Methodology

- Used standard data-augmentation techniques to create a larger dataset
- They used this dataset to train a GAN to create synthetic examples
- Used the augmented dataset from step 1 with GAN produced synthetic examples to train a liver lesion classifier
- DCGAN

![[Pasted image 20260309221220.png]]

- Input size = 64 x 64 x 1
- 5 x 5 convolutional kernels


## Results

- 


# GANs in fashion

## Using GANs to design fashion
## Methodology
## Creating new items matching individual preferences
## Adjusting existing items to better match individual preferences

# Conclusion