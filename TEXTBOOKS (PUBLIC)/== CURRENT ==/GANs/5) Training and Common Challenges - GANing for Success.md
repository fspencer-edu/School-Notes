
![[Pasted image 20260308224007.png]]

- All of these generative models are derived from maximum likelihood

# Evaluation

## Evaluation framework

- Approximations of max. likelihood tend to over-generalize the results are varied
- To diminish over-generalization, identify the probability distribution of the output with the distance function
	- KL and JS divergence
- Evaluation metrics
	- Inception score (IS)
	- Frechet inception distance (FID)

## Inception score

- Generator samples look real
- Classifiers are able to identify images to a class with certain confidence
- Generated samples are varies and contain classes that are represented in the original data
	- No interclass mode collapse


**Computing IS**
1. Take Kullback-Leibler (KL) divergence between real and generated distribution
2. Exponentiate the result

- A failure mode in an auxiliary classifier GAN (ACGAN)
	- Generate examples of daisies from the ImageNet dataset

![[Pasted image 20260308225117.png]]

- The Inception classifier is not certain what the image is

## Frechet inception distance

- FID improves on the IS by making it more robust to noise and allowing the detection of intraclass sample omissions
- Accepts the IS baseline
- The GAN detected patterns by memorizing the items, which creates an undesirable outcome
- The FID evaluates the distance of the embedded means, the variances, and covariances of the two distributions (real and generated)
- Attracts from human evaluators and allows statistical reasoning for the realism of an image

# Training Challenges

- Mode collapse
	- Some of the modes are not well represented in the generated samples
	- Network has converged
- Slow convergence
	- Speed of convergence and available compute are constraints
- Over-generalization
	- Modes that should not have support do
	- Hallucinations

**Improve training process**
- Add network depth
- Change setup
	- Min-max
	- Non-saturating design
	- Wasserstein GAN
- Normalizing the inputs
- Penalizing the gradients
- Training the discriminator
- Avoid sparse gradients
- Changing to soft and noisy labels

## Adding network depth

- Stability
- Speed of training
- Quality of samples

## Game setups

- Given the rules and distance metric, change the approach

## Min-Max GAN

- 

## Non-saturating GAN

## When to stop training

## Wasserstein GAN

# Summary of Game Setups

## Normalizations of inputs
## Batch normalization
## Gradient penalties
## Train the discriminator score

## Avoid sparse gradients

## Soft and noisy labels

# Training Hacks