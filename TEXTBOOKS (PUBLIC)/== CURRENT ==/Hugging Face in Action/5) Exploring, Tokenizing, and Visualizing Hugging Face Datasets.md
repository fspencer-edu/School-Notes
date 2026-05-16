
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

![[Pasted image 20260515221419.png]]

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
- 

### 
### 

## Visualizing Datasets