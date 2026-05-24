
## What are Hugging Face Datasets?

### Getting the list of datasets available

```python
!pip install huggingface_hub
!pip install datasets

from huggingface_hub import list_datasets

datasets = list_datasets()
```

- Do not use `list()` function to convert the list of datasets
- Use `next()` to iterate through the list of datasets

```python
dataset = next(datasets)
print(dataset)

# print id of dataset
for i in range(5):
	dataset = next(datasets)
	print(dataset.id)
```

### Validating the availability of a dataset

```python
import requests

token = 'Hugging_Face_Token' 
dataset_id = 'fka/awesome-chatgpt-prompts' 

headers = {"Authorization": f"Bearer {token}"}
API_URL = 
  f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_id}"

def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()

data = query()
data

$ curl -X GET "https://datasets-server.huggingface.co/is-
valid?dataset=fka/awesome-chatgpt-prompts"
```

### Downloading a dataset

```python
from datasets impot load_dataset
dataset_id = 'standfordnlp/imbd'

dataset = load_dataset(dataset_id)
print(dataset)
```

- The result `DatasetDict` object has splits
	- Train
	- Test
	- Unsupervised

```json
{
  "splits":[
    {
      "dataset":"stanfordnlp/imdb",
      "config":"plain_text",
      "split":"train"
    },
    {
      "dataset":"stanfordnlp/imdb",
      "config":"plain_text",
      "split":"test"
    },
    {
      "dataset":"stanfordnlp/imdb",
      "config":"plain_text",
      "split":"unsupervised"
    }
  ],
  "pending":[],
  "failed":[]
}
```

- Download the dataset from Hugging face
	- `~/.cache/huggingface/datasets`

<img src="/images/Pasted image 20260515221419.png" alt="image" width="500">

- Download a particular split

```python
dataset = load_dataset(dataset_id,
					   split='train')
					   
print(dataset)

dataset[0]
```

### Shuffling a dataset

- Use `shuffle()` to randomize the order of the data

```python
dataset_id = 'stanfordnlp/imdb'
dataset = load_dataset(dataset_id)
shuffled_dataset = dataset.shuffle(seed=42)
```
### Streaming a dataset

```python
from datasets import load_dataset

dataset_id = 'stanfordnlp/imdb'
dataset = load_dataset(dataset_id,
					   streaming=True)
print(dataset)
```
- Streaming returns an `IterableDatasetDict` object
- Enumerate through, to retrieve rows one at a time

```python
for i, example in enumerate(dataset["train"]):
	if i < 5:
		print(example)
	else:
		break
```

### Getting the Parquet files of a dataset

- Parquet
	- Columnar storage file designed for efficient data storage and processing
	- Querying and analyzing large datasets
	- Schema based

```json
{
  "parquet_files":[
    {
      "dataset":"stanfordnlp/imdb",
      "config":"plain_text",
      "split":"test",
      "url":"https://huggingface.co/datasets/stanfordnlp/
             imdb/resolve/refs%2Fconvert%2Fparquet/
             plain_text/test/0000.parquet",
      "filename":"0000.parquet",
      "size":20470363
    },
    {
      "dataset":"stanfordnlp/imdb",
      "config":"plain_text",
      "split":"train",
      "url":"https://huggingface.co/datasets/stanfordnlp/
             imdb/resolve/refs%2Fconvert%2Fparquet/
             plain_text/train/0000.parquet",
      "filename":"0000.parquet",
      "size":20979968
    },
    {
      "dataset":"stanfordnlp/imdb",
      "config":"plain_text",
      "split":"unsupervised",
      "url":"https://huggingface.co/datasets/stanfordnlp/
             imdb/resolve/refs%2Fconvert%2Fparquet/
             plain_text/unsupervised/0000.parquet",
      "filename":"0000.parquet",
      "size":41996509
    }
  ],
  "pending":[],
  "failed":[],
  "partial":false
}
```

## Tokenization in NLP

- Tokenization
	- NLP process
	- Breaks text into manageable units or tokens
		- Text preprocessing
		- Representation for ML models
		- Efficiency and memory optimization
		- Foundation for further NLP tasks
			- NER
			- Speech tagging
			- Machine translation
			- Summarization

### Types of tokenization methods

- Work level
- Subword level
- Character level

- Subword or byte-pair encoding (BPE)
	- BERT
	- GPT
	- Handles out-of-vocabulary (OOV) words
	- Preserved more information

### Tokenizing datasets

- HF datasets are compatible with build-in tokenizers and data loaders

```python
from transformers import AutoTokenizer

dataset = load_dataset(dataset_id)
tokenizer = AutoTokenizer.from_pretrained('bert-based-uncased')
tokenized_dataset = dataset.map(
	lambda examples:
		tokenizer(examples['text'],
				  truncaction = True,
				  padding = 'max_length')
		batched = True)
```

```python
# tokenized dataset
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids',
                   'attention_mask'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids',
                   'attention_mask'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids',
                   'attention_mask'],
        num_rows: 50000
    })
})
```

- Each number in the `input_ids` represent the ID of corresponding token

```python
# convert tokens back
tokens = tokenizer.convert_ids_to_tokens(
             tokenized_dataset['train'][0]['input_ids'])
print(tokens)

   ['[CLS]', 'i', 'rented', 'i', 'am', 'curious', '-',
```
- `[CLS]`
	- Start of string
- `##`
	- String continuation
- `[PAD]`
	- Padding in tokenized sequences

- `token_type_ids`
	- Used to differentiate among multiple segments of a single input
		- Next-sentence prediction
		- Question answering
	- Help model determine which tokens belong to which segments
- `attention_mask`
	- Inform the model which tokens should be attended to
		- 1 (attended to) or 0 (padding)

<img src="/images/Pasted image 20260515222840.png" alt="image" width="500">

## Visualizing Datasets

### Using the twitter-financial-news-topic dataset

- Dataset is an English language dataset containing an annotated corpus of finance-related tweets

```python
from datasets import load_dataset

dataset = load_dataset('zeroshot/twitter-financial-news-topic')
train_data = dataset['train']

print(train_data[0])
print(train_data[-1])

# map topics for labels
topics = {                                   
    "LABEL_0": "Analyst Update",
    "LABEL_1": "Fed | Central Banks",
    "LABEL_2": "Company | Product News",
    "LABEL_3": "Treasuries | Corporate Debt",
    
mapped_labels = [topics[f"LABEL_{label}"]    
                 for label in train_data['label']]
                 
# plot dataset
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
bins = np.arange(len(topics) + 1) - 0.5
plt.hist(mapped_labels,
		 bins = binsm
		 edgecolor = 'black',
		 color = 'skyblue,
		 alpha = 0.7)
		 
plt.xticks(np.arange(len(topics)),
		   list(topics.values()),
		   rotation = 90,
		   ha = 'center')
		   
plt.title("Topic Distribution - Twitter Financial News")
plt.xlabel("Topics")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.show()
```

<img src="/images/Pasted image 20260515223306.png" alt="image" width="500">

### Using the CIFAR-10 dataset

- ML model for computer vision
- 60,000 labeled 32x32 colour images divided into 10 classes
- CNN

```python
# download
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

dataset = load_dataset('uoft-cs/cifar10')

print(dataset)

# display grid of images
labels = {  #1
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def show_images(images, labels, labels_dict):
	plt.figure(figsize=(5, 5))
	for i in range(25):
		plt.subplot(5, 5, i + 1)
		plt.imshow(images[i])
		plt.title(labels_dict[labels[i]])
		plt.axis('off')
	plt.tight_layout()
	plt.show()
	
train_samples = dataset['train'].shuffle(seed=42).select(range(25)) 


images = [sample['img'] for sample in train_samples]
class_labels = [sample['label'] for sample in train_samples]


show_images(images, class_labels, labels)
```

