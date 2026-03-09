# Context of adversarial examples

# Lies, damned lies, and distributions

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
attack = foolbox.attacks.FGSM(fmodel, threshold=.9,                      7
     criterion=ConfidentMisclassification(.9))                           7
adversarial = attack(image, label)                                       8

new_preds = kmodel.predict(np.expand_dims(adversarial, axis=0))          9
print('Predicted:', pp.pprint(decode_predictions(new_preds, top=20)[0]))
```


# Signal and the noise

# Not all hope is lost

# Adversarial to GANs

# Conclusion