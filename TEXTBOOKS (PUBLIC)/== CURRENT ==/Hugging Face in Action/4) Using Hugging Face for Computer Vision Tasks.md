## Hugging Face Computer Vision Model

- Computer vision model tasks
	- Object detection
	- Image classification
	- Image segmentation
	- Video classification
	- Depth estimaation
	- Image-to-image
	- Unconditional image generation
	- Zero-shot image classification

## Object Detection

- Object detection
	- Identifying and locating objects of interest
	- Classify the objects in the image and determine their precise positions
	- `facebook/detr-resnet-50`


### Using the model directly

- Use model
	- Transformer pipeline
	- Load model directly

- Packages
	- `transformer`
	- `timm`
		- Deep learning library
		- Reproduce ImageNet training results

```python
from transformers import DetrImageProcessor, DetrForObjectDetection

image_processor = DetrImageProcessor.from_pretrained(
                      "facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
                      "facebook/detr-resnet-50")
```

- DETR (Detection Transformer)

```python
# display image
from PIL import Image, ImageDraw
import requests
import torch

def loadImage(url):
	if url.startswith('http'):
		image = Image.open(requests.get(url, stream=True).raw)
	else:
		image = Image.open(url)
	return image
	
image = loadImage('http://bit.ly/46xv3sL')
display(image)

# prepare the input image (preprocess)
inputs = image_processor(image = image,
						 return_tensors = "pt")
						 
# pass preprocessed tensors to model
outputs = model(**inputs)

# output, for object detection
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
				outputs,
				target_sizes = target_sizes,
				threshold = 0.9)[0]
				
results
```

- `post_process_object_detection()`
	- Output of the model
	- Target size of image
	- Threshold value for filtering output predictions (confidence = 0.9)
- Returns a dictionary containing the objects detected in the image
	- `scores`
		- Confidence of each detected object
	- `labels`
		- Index of the detected object
	- `boxes`
		- Bounding boxes of each detected object


```python
# drawing bounding boxes
import random

draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{(score.item() * 100):.2f}% at {box}"
    )

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = (r, g, b)

    draw.rectangle(box,  #1
                   outline=color,   #1
                   width=2)   #1

    draw.text((box[0], box[1]-10),  #2
              model.config.id2label[label.item()],   #2
              fill='white')   #2

display(image)
```

![[Pasted image 20260515122708.png]]
### Using the transformer pipeline

```python
from transformer import pipeline

detection = pipeline("object-detection", model="facebook/detr-resnet-50")

# pass image to pipeline
results = detection(image)
results

# results = detection('http://bit.ly/46xv3sL')

# visualizing detected objects
import random

draw = ImageDraw.Draw(image)

for object in results:
    box = [i for i in object['box'].values()]
    print(
        f"Detected {object['label']} with confidence "
        f"{(object['score'] * 100):.2f}% at {box}"
    )

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = (r, g, b)

    draw.rectangle(box,  #1
                   outline=color,   #1
                   width=2)   #1

    draw.text((box[0], box[1]-10),  #2
              object['label'],   #2
              fill='white')   #2

display(image)
```

### Binding to a webcam

- OpenCV

```python
!pip install opencv-python

# display webcam images in python
import cv2
stream = cv2.VideoCapture(0)
whie(True):
	(grabbed, grame) = stream.read()
	cv2.imshow("Image", frame)
	key = cv2.waitKey(1)
	if key == ord("q"):
		break
		
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

# Detecting objects
from transformer import pipeline
from PIL import Image
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 255)
stroke = 2
detection = pipeline("object-detection",
                     model="facebook/detr-resnet-50")
                     
stream = cv2.VideoCapture(0)
while(True):
	(grabbed, frame) = stream.read()
	image = Image.fromarray(frame)
	results = detection(image)
	for object in results:
		box = [i for i in object['box'].values()]
		cv2.rectangle(frame,
					  (box[0],box[1]),
					  (box[2],box[3]),
					  color, stroke)
        cv2.putText(frame, f'({object["label"]})',  #9
                    (box[0],box[1]-8),   #9
                    font, 1, color,   #9
                    stroke, cv2.LINE_AA)     #9
    cv2.imshow("Image", frame)  #10
    key = cv2.waitKey(1) & 0xFF  
    if key == ord("q"):  #11
        break

stream.release()  #12
cv2.waitKey(1)   #12
cv2.destroyAllWindows()   #12
cv2.waitKey(1) 
```

- Convert image captured by webcam from a NumPy array to a `PIL` image before sending it to model for object detection

## Image Classification

- Image classification
	- Categorizing, labeling an image in one or mode predefined classes or categories

## Image Segmentation

- Image segmentation
	- Separate an image into multiple segments or regions
- Applications
	- Medical imaging
	- Object detection and recognition
	- Document processing
	- Biometrics

### Using the model programmatically

- Load the model
	- Check how many objects the model can detect

```python
from transformers import pipeline
segmentation = pipeline("image-segmentation",
               model="nvidia/segformer-b0-finetuned-ade-512-512")
segmentation.model.config.id2label

# 150 objects

# image segmentation
from PIL import Image
import requests
url = 'https://bit.ly/46iDeJQ'
results = segmentation(url)
results

# mask element
for result in results:
	print(result['label'])
	display(result['mask'])
	
# apply mask to iri
```

### 
### 

## Video Classification