
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

![[Pasted image 20260308230214.png]]

$E$ = expectation
$x$ = true data distribution
$z$ = latent sapce
$D$ = discriminator function
$G$ = generator function

**Discriminator's Loss Function**

![[Pasted image 20260308230328.png]]

 - The discriminator is trying to minimize the likelihood of mistaking a TN or FP

**Generator's Loss Function**

![[Pasted image 20260308230425.png]]

- Distance metrics
	- Jensen-Shannon divergence (JSD) is a symmetric version of KL divergence
	- Wasserstein


## Non-saturating GAN

- Problems with min-max
	- Slow convergence for the discriminator
- NS-GAN
	- The two loss functions are independent, but directionally consistent with the original formulation
	- Generator is trying to minimize the opposite of the second term of the discriminator
		- "Get caught for the samples it generates"

![[Pasted image 20260308230826.png]]


- MM-GAN can easily saturate (get close to 0) which leads to slow convergence
- Weight updates that are backpropogated are either 0 or tiny

![[Pasted image 20260308231058.png]]

- NS-GAN
	- Initial training is faster
		- Generator and discriminator learns faster

## When to stop training

- Stop training when the NS-GAN
	- Is no longer asymptotically consistent with the JSD
	- Has an equilibrium state that theoretically is more elusive

## Wasserstein GAN

- WGAN
	- Improves on the loss function
	- Better results
	- Clear theoretical backing that start form the loss
	- Uses the earth mover's distance as a loss function that correlates with the visual quality of the generated samples

**WGAN-Discriminator**

![[Pasted image 20260308231545.png]]

$f_w$ = discriminator

- Critic tries to estimate the earth's mover's distance, and looks for the max. difference between the real and generated distribution under different parameterizations of the $f_w$ function
- Critic looks at different projections using $f_w$ into shared spaced in order to max. the about of probability mass it has to move

**WGAN-Generator**

![[Pasted image 20260308231725.png]]

- Min. the distance between the expectation of the real and generated distribution
- The image is from the real of generated distribution
- Generated samples are sampled from the $z$ and transformed via $g_{\theta}$ to get $x*$ in the same space, and then evaluated using $f_2$
- Try to minimize the loss function

- No log is used
- More tunable training, since we can set a clipping constant (acts as a learning rate in standard ML)
	- Can end up very sensitive
- Measure the Wasserstein distance to determine when to stop
- Train the WGAN to convergence
	- JS loss and the divergence can be meaningless

- A gaussian distribution plot represents both the real and generated samples
- z-axis represents the probability of that point being samples

![[Pasted image 20260308232335.png]]

![[Pasted image 20260308232422.png]]


# Summary of Game Setups

- DCGAN
- SAGAN
- Adam optimizer instead of vanilla stochastic gradient descent

## Normalizations of inputs

- Normalization between -1 and 1
- Reduces machine resources
- tanh activation function

## Batch normalization

- Accelerates and stabilizes deep NN training by normalizing the inputs to each layer to have a zero mean and unit variance for each mini-batch
## Gradient penalties

- Naive weighed clipping can produce vanishing or exploding gradients
- Restrict gradient norm or the discriminator output with respect to its input
	- Weights should updated less than the ch
## Train the discriminator score

- 

## Avoid sparse gradients

## Soft and noisy labels

# Training Hacks