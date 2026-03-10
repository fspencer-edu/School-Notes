# Context of adversarial examples

- Adversarial models attack with Gaussian noise by injecting carefully crafted, low-magnitude random perturbations into input data (images, 3D models, or text) to induce misclassification

# Lies, damned lies, and distributions


$\hat{y} = f_{\theta}(x)$ = classification

$L = ||y - \hat{y}||$ = loss function

$min_{\theta}||y - \hat{y}||$

- Loss is the difference between the true and predicted label
- SGD-based method takes batches of x
- Takes the derivative of the loss function with respect to the current parameters $\theta_t$ multiplied by the learning rate ($\alpha$), which constitutes the new parameters, $(\theta_{t+1})$

$\theta_{t+1} = \theta - \alpha * \dfrac{\partial L}{\partial \theta}$

- Maximizing the error rather than minimizing it is easier

<img src="/images/Pasted image 20260309214907.png" alt="image" width="500">

- With adversarial examples, we are conditioning on an entire image and trying to produce a domain transferred to a similar image
- The generator can be a stochastic gradient ascent to fool the discriminator
- Fast sign gradient method (FSGM)
	- Start with the gradient update, look at the sign, and them make a step in the opposite direction

<img src="/images/Pasted image 20260309215155.png" alt="image" width="500">


# Use and abuse of training

```python
# imports
import numpy as np
import keras.applications.resnet50 impport ResNet50
from foolbox.criteria import MisClassification, ConfidenceMisclassification
from keras.preprocessing import images as img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import foolbox
import print as pp
Import keras
```

```python
def load_image(img_path: str):
	image = img.load_img(img_path, target_size(224, 244))
	plt.imshow(image)
	x = img.img_to_array(image)
	return x
	
image = load_image('DSC_0897.jpg')
```

```python
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)

fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255),
		preprocessing=proprocessing)
		
to_classify = np.expand_dims(images, axis=0)
preds = kmodel.predict(to_classify)
print('Predicted:', pp.pprint(decode_predictions(preds, top=20)[0]))
label = np.argmax(preds)     

image = image[:, :, ::-1]                                                
attack = foolbox.attacks.FGSM(fmodel, threshold=.9,                      
     criterion=ConfidentMisclassification(.9))                          
adversarial = attack(image, label)                                       

new_preds = kmodel.predict(np.expand_dims(adversarial, axis=0))          
print('Predicted:', pp.pprint(decode_predictions(new_preds, top=20)[0]))
```

<img src="/images/Pasted image 20260309215310.png" alt="image" width="500">

- Adversarial examples generalize beyond deep learning and transfer to different ML techniques


<img src="/images/Pasted image 20260309215423.png" alt="image" width="500">

# Signal and the noise

- Many adversarial examples are easy to construct and can be fooled with Gaussian noise

<img src="/images/Pasted image 20260309215542.png" alt="image" width="500">

<img src="/images/Pasted image 20260309215620.png" alt="image" width="500">

```python
# Guassian noise
fit = plt.figure(figsize=(20, 20))
sigma_list = list(max_vals.sigma)
mu_list = list(max_vals.mu)
conf_list = []

def make_subplots(x, y, z, new_row=False):
	rand_noise = np.random.normal(loc=mu, scale=sigma, size=(224, 224, 3))
	rand_noise = np.clip(rand_noise, 0, 255.)
	noise_preds = kmodel.predict(np.expand_dims(rand_noise, axis=)0)
	prediction, num = decode_predictions(noise_preds, top=20)[1:3]
	num = round(num * 100, 2)
	conf_list.append(num)
	ax = fig.add_subplot(x, y, z)
	ax.annotate(prediciton, xy=(0.1, 0.6),
			xycoords=ax.transAves, fontsize=16, color='yellow')
	ax.annotate(f'{num}%', xy=(0.1, 0.4),
			xycoords=ax.transAves, fontsize=20, color='orange')	
	if new_row:
		ax.annotate(f'$\mu$:{mu}, $\sigma$:{sigma}' ,
                    xy=(-.2, 0.8), xycoords=ax.transAxes,
                    rotation=90, fontsize=16, color='black')
    ax.imshow(rand_noise / 255)                                         
    ax.axis('off')
    
    for i in range(1,101):                                                    
    if (i-1) % 10==0:
        mu = mu_list.pop(0)
        sigma = sigma_list.pop(0)
        make_subplot(10,10, i, new_row=True)
    else:
        make_subplot(10,10, i)

plt.show()
```

# Not all hope is lost

# Adversaries to GANs

# Conclusion

- Get adversarial noise that changes the label of a picture without changing the image perceptibility