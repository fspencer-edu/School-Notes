
- Multimodal models
	- Combine multiple types of data
## Fine-Tuning Pretrained Models

- Fine tuning
	- A pretrained model is further trained on a small, domain specific dataset to adapt it for a particular task
	- Adjusting later layers of the model
	- Applying lower learning rate to avoid losing features from pretrained phase
	- NLP
	- Computer vision
	- CNN

### Loading the `yelp_polarity` dataset

- Sentiment analysis of restaurant reviews

```python
from datasets import load_dataset

dataset = load_dataset("yelp_polarity")
print(dataset)

train_dataset = dataset['train']
print(train_dataset[0])
```

### Filtering dataset

- Topic variety
- Dataset size

```python
train_dataset = dataset["train"]
test_dataset = dataset["test"]

restaurant_train_reviews = train_dataset.filter(
	lambda x: "restaurant" in x["text"].lower()
)

restaurant_test_reviews = test_dataset.filter(
    lambda x: "restaurant" in x["text"].lower()
)

number_of_reviews = 5000
subset_train_reviews = restaurant_train_reviews.shuffle( 
    seed = 42).select(range(number_of_reviews))
subset_test_reviews = restaurant_test_reviews.shuffle( 
    seed = 42).select(range(number_of_reviews))

subset_dataset = { 
    "train": subset_train_reviews,
    "test": subset_test_reviews
}

from datasets import DatasetDict
yelp_restaurant_dataset = DatasetDict(subset_dataset)  

print(yelp_restaurant_dataset)
```

### Tokenizing the reduced dataset

- Perform tokenization on the reduced dataset using the `distilbert-base-uncased` model

```python
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def tokenize_function(examples):  
    return tokenizer(examples["text"],
                     padding = "max_length",
                     truncation = True,
                     max_length = 512)

tokenized_datasets = yelp_restaurant_dataset.map( 
                         tokenize_function,
                         batched=True)
tokenized_datasets
```

### Setting up a pretrained model for sequence classification

```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained( 
            model_checkpoint, num_labels = 2)

if torch.backends.mps.is_available():  
    device = torch.device("mps")
else:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
```

- `AutoModelForSequenceClassification`
	- Model loader for sequence classification tasks
	- Load compatible pretrained model architecture
- Sequence classification tasks
	- Assigning a single label or category to an entire sequence of data

**MPS (Metal Performance Shaders) and CUDA**

- MPS
	- Apple's framework for GPU-accelerated computations
- CUDA
	- Nvidia's parallele computing platform

### Configuring and initializing a trainer for fine-tuning a pretrained model

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(  #1
    output_dir = "./results",  #2
    eval_strategy = "epoch",  #3
    save_strategy = "epoch",  #4
    learning_rate = 2e-5,  #5
    per_device_train_batch_size = 16,  #6
    per_device_eval_batch_size = 16,  #7
    num_train_epochs = 3,  #8
    weight_decay = 0.01,  #9
    logging_dir = "./logs",  #10
    logging_steps = 10,  #11
    save_steps = 500,  #12
    load_best_model_at_end = True,  #13
)

trainer = Trainer(  #14
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
)

trainer.train() 
```

- When the model is trained, save it to disk

```python
model.save_pretrained("./results/final_model")  #1
tokenizer.save_pretrained("./results/final_tokenizer")

# evaluate model
eval_results = trainer.evaluate()  #1
print(f"Evaluation results: {eval_results}")
```

- Results
	- Evaluation loss
	- Evalution runtime
	- Evaluation samples per second
	- Evaluation steps per second
	- Epoch

### Using the fine-tuned model

- Performa sentiment analysis on a new restaurant review

```python
from transformers import AutoTokenizer, \
                         AutoModelForSequenceClassification
import torch

new_model = AutoModelForSequenceClassification.from_pretrained(  #1
                "./results/final_model")
new_tokenizer = AutoTokenizer.from_pretrained(  #1
                "./results/final_tokenizer")

new_model.to(device)  #2

sentence = '''
I had an amazing experien
'''

inputs = new_tokenizer(sentence,  #3
                       return_tensors = "pt",
                       padding = True,
                       truncation = True,
                       max_length = 512)

inputs = {key: value.to(device) for key, value in inputs.items()}  #4

new_model.eval()  #5

with torch.no_grad():
    outputs = new_model(**inputs)  #6
logits = outputs.logits  #7
probabilities = torch.nn.functional.softmax(logits, dim=-1)  #8
predicted_class = torch.argmax(probabilities, dim=-1).item()  #9

if predicted_class == 1:  #10
    print(f"Sentiment: Positive (Confidence: \
          {probabilities[0][1].item():.2f})")
else:
    print(f"Sentiment: Negative (Confidence: \
           {probabilities[0][0].item():.2f})")
```

### Fine tuning models for multiclass text classification

- `yelp_review_full`
	- Labels corresponding to a star rating system from 1 to 5

```python
# restaurant specific
from datasets import DatasetDict

train_dataset = dataset["train"]  #1
test_dataset = dataset["test"]   #1

restaurant_train_reviews = train_dataset.filter(  #2
    lambda x: "restaurant" in x["text"].lower()
)

restaurant_test_reviews = test_dataset.filter(
    lambda x: "restaurant" in x["text"].lower()
)

number_of_reviews = 5000  #3
subset_train_reviews = restaurant_train_reviews.shuffle(
    seed=42).select(range(number_of_reviews))
subset_test_reviews = restaurant_test_reviews.shuffle(
    seed=42).select(range(number_of_reviews))

subset_dataset = {  #4
    "train": subset_train_reviews,
    "test": subset_test_reviews
}

yelp_restaurant_dataset = DatasetDict(subset_dataset)  #5

print(yelp_restaurant_dataset)

# load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(  #1
            model_checkpoint,
            num_labels = 5)
            
model.save_pretrained("./results/final_model_multiclass")  #1
tokenizer.save_pretrained("./results/final_tokenizer_multiclass")   #1
eval_results = trainer.evaluate()  #2
print(eval_results)
```

- Perform multiclass sentiment analysis

```python
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch

new_reviews = [
    "The food was amazing and the service was excellent!",
    "The restaurant was dirty and the food was cold.",
    "Decent experience, but nothing special."
]

if torch.backends.mps.is_available():  #1
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_tokenizer = AutoTokenizer.from_pretrained(  #2
    "./results/final_tokenizer_multiclass")

inputs = new_tokenizer(new_reviews,  #3
                       padding = "max_length",
                       truncation = True,
                       return_tensors = "pt")
                       
inputs = {key: value.to(device) for key, value in inputs.items()}  #4

new_model = AutoModelForSequenceClassification.from_pretrained(  #5
    "./results/final_model_multiclass")
new_model.to(device)

new_model.eval()

with torch.no_grad():  #6
    outputs = new_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

star_ratings = predictions + 1   #7
for review, rating in zip(new_reviews, star_ratings):
    print(f"Review: {review}\nPredicted Star Rating: \
          {rating.item()}\n")
```



## Working with Multimodal Models

- Single modal model
	- ML model designed to work with data from a single modality
		- NLP
		- CNN
- Multimodal model
	- ML model designed to process and integrate data from multiple modalities
		- Image captioning
		- Visual question answering
		- Speech-to-tech
### Single modal models

- `facebook/detr-resnet-50`
	- DETR (Detection Transformer)

```python
from PIL import Image, ImageDraw
import requests

url = 'https://images.unsplash.com/' + \
      'photo-1563460716037-460a3ad24ba9' 
      
if url.startswith('http'):
    image = Image.open(requests.get(url, stream=True).raw)
else: 
    image = Image.open(url) 
image

# model to detect dog/cat in pics
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

image_processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = image_processor(images = image,  #1
                         return_tensors = "pt")

model.eval()

with torch.no_grad():
    outputs = model(**inputs)  #2
target_sizes = torch.tensor([image.size[::-1]])  #3

results = image_processor.post_process_object_detection(  #4
              outputs,
              target_sizes = target_sizes,
              threshold = 0.9)[0]
print(results)

# plotting bounding boxes
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"],
                             results["labels"],
                             results["boxes"]): 
    print(  #1
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{(score.item() * 100):.2f}% at {box}"
    )
    box = [round(i, 2) for i in box.tolist()]  #2
    draw.rectangle(box,
                   outline = 'green',
                   width = 10) 
    draw.text((box[0], box[1]-10),  #3
              model.config.id2label[label.item()],
              fill = 'green')
display(image)
```

<img src="/images/Pasted image 20260516111317.png" alt="image" width="500">

### Multimodal Models

- CLIP (Contrastive Language-Image Pretraining)
	- Process images and text
	- Developer by OpenAI
	- Aligns text descriptions with corresponding images
- Neural netoworks
	- Text encoder to process text
	- An image encoder (ResNet and Vision Transformer) to process images

```python
import torch
import requests
from PIL import image
url = 'https://images.unsplash.com/' + \
      'photo-1491604612772-6853927639ef'
      
image = Image.open(requests.get(url, stream=True).raw)
display(image)

# process visual and textural data
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  #1
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = ["cat", "dog", "tiger", "train"]             
inputs = processor(text = labels,  #2
                   images = image,
                   return_tensors = "pt",
                   padding=True)

model.eval()

with torch.no_grad():
    outputs = model(**inputs)  #3

logits_per_image = outputs.logits_per_image  #4

probs = logits_per_image.softmax(dim=1)  #5
most_likely_index = torch.argmax(probs, dim=1).item()  #6
most_likely_object = labels[most_likely_index]

print(f"The most likely object is: {most_likely_object}")
```
