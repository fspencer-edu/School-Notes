- Hugging Face is an AI community that promotes the building, training, and deployment of open source ML models
- Best known for its Transformers library for developing NLP applications

## Hugging Face Transformers Library

- Transformers library
	- Package that contains open source implementations of the Transformer architecture
		- Text, image, audio
	- API for pretrained models

```python
from transformers import pipeline

classifier = pipeline('text-classification,
				model = 'distilbert-base-uncased-finetuned-sst-2-english',
	            revision = 'af0f99b')
```

- Pipeline
	- High level API that simplified building and using complex NLP workflows

```python
import pandas as pd

text = '''
I though...
'''

result = classifier(text)
pd.DataFrame(result)
```

- The output of the sentiment analysis if a positive score of 0.99

## Hugging Face Models

- All pretrained models are stored in repositories
- COCO (Common Objects in Context)
	- Large scale object detection, segmentation, and captioning dataset

## Hugging Face Gradio Python Library

- Gradio
	- Open source Python library
	- Customizable user interfaces for ML and data science

```python
from skimage.color import rgb2gray

def transform_image(img):
	return rgb2gray(img)
	
import gradio as gr

demo = dr.Interface(fn = transform_image,
					inputs = gr.Image(),
					outputs = "image")
					
demo.launch()
```

## Understanding the Hugging Face Mental Model

<img src="/images/Pasted image 20260514204349.png" alt="image" width="500">

### User Need

### Model Hub Discovery
### Model Card

- Every model has a detailed model card
	- Documentation and gateway
	- Performance benchmarks
	- Training details
	- Information about using the model
### Two Execution Paths

- Hosted inference API
	- HTTP
- Direct download
	- Git Large File Storage (LFS)

### Results Delivered