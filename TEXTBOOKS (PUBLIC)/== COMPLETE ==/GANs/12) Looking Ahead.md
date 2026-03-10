# Ethics

# GAN Innovations

- Practical paper (RGAN)
- Artistic application (BigCAN)

## Relativistic GAN

- RGAN is the addition of an extra term to the generator
- Forcing it to make the generated data seem more real than the real data
- Generator should make data seem more real, and make real data seem comparatively less real

$L_D = E[D(x)] - E[D(G(z))]$

$L_G = E[D(G(z))]$


<img src="/images/Pasted image 20260310121347.png" alt="image" width="500">

## Self-Attention GAN

- Use attention use feature detections with a flexible receptive field to focus on key aspects of a given picture

<img src="/images/Pasted image 20260310123323.png" alt="image" width="500">

- Attention allows us to pick out relevant regions and consider them appropriately

<img src="/images/Pasted image 20260310123601.png" alt="image" width="500">

- SAGAN

## BigGAN

- BigGAN builds on the SAGAN and spectral normalization and has further innovated in five directions
- Truncation trick
	- Controls the trade-off between variety and fidelity

# Further Reading

- Style GAN
	- Conditional GAN from NVIDIA has managed to produce full HD results
- Spectral normalization
	- A complex regularization technique and requires somewhat advanced linear algebra
- SPADE, aka GauGAN
	- Produces photorealistic images based solely on a semantic map of the image


# Looking back and closing thoughts