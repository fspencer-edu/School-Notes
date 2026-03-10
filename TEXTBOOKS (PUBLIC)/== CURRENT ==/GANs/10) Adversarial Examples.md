# Context of adversarial examples

# Lies, damned lies, and distributions


$\hat{y} = f_{\theta}(x)$ = classification

$L = ||y - \hat{y}||$ = loss function

$min_{\theta}||y - \hat{y}||$

- Loss is the difference betw

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


# Signal and the noise

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

# Adversarial to GANs

# Conclusion