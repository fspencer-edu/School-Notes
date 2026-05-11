
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

<img src="/images/Pasted image 20260309221220.png" alt="image" width="500">

- Input size = 64 x 64 x 1
- 5 x 5 convolutional kernels


## Results

<img src="/images/Pasted image 20260309221325.png" alt="image" width="500">


# GANs in fashion

## Using GANs to design fashion

- Using a dataset compiled from users, items, and reviews from Amazon, created models to recommend fashion and create it
- For any person-item pair, it returns a preference score
- The greater the score, the better match the item is
- Created new fashion items matching the taste of an individual
- Personalized alterations to existing items based on individual's preferences

## Methodology

- CGAN
	- Product's category as the conditioning label
	- Dataset consisted of gender based clothing categories
		- Tops
		- Bottoms
		- Shoes
- The generator uses random noise and conditioning information to synthesize an image
- The discriminator outputs a probability that a particular image-category is real or fake

<img src="/images/Pasted image 20260309221929.png" alt="image" width="500">

- Each box represents a layer
	- fc = fully connected layer
	- st = strides

## Creating new items matching individual preferences

- Preference maximization
	- CGAN generator produces a fashion item maximizing an individual's preference
- Constraint maximization
	- The constraint is the size of the latent space, given by the size of the vector z
- Used gradient ascent, to max. a reward function by iteratively moving in the direction fo the steepest increase


<img src="/images/Pasted image 20260309222421.png" alt="image" width="500">

## Adjusting existing items to better match individual preferences

- Vectors that are close tend to produce images that are similar in terms of content and style
- To generate variations of some image A, find the latent vector zA that the generator would use to create the image
- Produce images from neighbouring vectors to generate similar imges

<img src="/images/Pasted image 20260309222555.png" alt="image" width="500">

- Find a vector z that the generator uses to synthesize an image similar to the real image, and use it as a proxy for the hypothetical z that would have produced the real image
- Reconstruction loss
	- A measure of the difference between two images
	- The greater the loss, the more different a given pair of images is
- Move around the latent space to points that generate images similar to the one we want to modify, while also optimizing for the preferences of the given user

<img src="/images/Pasted image 20260309222838.png" alt="image" width="500">


# Conclusion