
- When training TF models on large datasets, your may prefer to use TF loading and preprocessing API, called `tf.data`
- Reads multiple files in parallel using multithreading and queuing, shuffling and batching samples, and more
- Loads and preprocesses next batch across multiple CPU cores, while GPU or TPU are busy training the current batch of data
- Read files
	- CSV
	- Binary with fixed-size records and TFRecord format
- TFRecord is a flexible and efficient binary format usually containing protocol buffers
- Support for reading from SQL databases
- Google BigQuery Service


# The tf.data API

- `tf.data` API revolves around `tf.dataset.Dataset`
- Sequence of data items

```python
import tensorflow as tf
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
<TensorSliceDataset shapes: (), types: tf.int32>
```

- `from_tensor_slices()` takes a tensor and create a dataset whose elements are all the slices of `X` with the first dimension, and 10 items

```python
for item in dataset
	print(item)
	...
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
[...]
tf.Tensor(9, shape=(), dtype=int32)
```

- The `tf.data` API is a streaming API, efficiently iterate through a dataset's items, but not designed for indexing or slicing
- Dataset
	- Tuples
	- Dictionaries
	- Nested tuples and dictionaries
- Dataset will only slice the tensors it contains, while preserving the tuple/dictionary structure

```python
X_nested = {"a": ([1, 2, 3], [4, 5, 6]), "b": [7, 8, 9]
dataset = tf.data.Dataset.from_tensor_slices(X_nested)
for item in dataset:
	print(item)
{'a': (<tf.Tensor: [...]=1>, <tf.Tensor: [...]=4>), 'b': <tf.Tensor: [...]=7>}
{'a': (<tf.Tensor: [...]=2>, <tf.Tensor: [...]=5>), 'b': <tf.Tensor: [...]=8>}
{'a': (<tf.Tensor: [...]=3>, <tf.Tensor: [...]=6>), 'b': <tf.Tensor: [...]=9>}
```

## Chaining Transformations

- Apply transformation to dataset by calling methods
- Each method returns a new dataset

```python
dataset = td.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
for item in dataset:
	print(item)
tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
tf.Tensor([8 9], shape=(2,), dtype=int32)
```

- Use `batch()` on new dataset, to create new dataset

<img src="/images/Pasted image 20260204105728.png" alt="image" width="500">
- Dataset methods do not modify datasets
- Transform items by calling the `map()` method

```python
dataset = dataset.map(lambda: x: x 8)
```


## Shuffling the Data
## Interleaving Lines from Multiple Files
## Preprocessing the Data
## Putting Everything Together

<img src="/images/Pasted image 20260204105738.png" alt="image" width="500">

## Prefetching

<img src="/images/Pasted image 20260204105749.png" alt="image" width="500">

## Using the Dataset with Keras

# The TFRecord Format

## Compressed TFRecord Files
## A Brief Introduction to Protocol Buffers
## TensorFlow Protobufs
## Loading and Parsing Examples
## Handling Lists of Lists Using the SequenceExample Protobuf

# Keras Preprocessing Layers

## The Normalization Layer

<img src="/images/Pasted image 20260204105806.png" alt="image" width="500">

<img src="/images/Pasted image 20260204105817.png" alt="image" width="500">


## The Discretization Layer
## The CategoryEncoding Layer
## The StringLookup Layer
## The Hashing Layer
## Encoding Categorical Features Using Embeddings

<img src="/images/Pasted image 20260204105827.png" alt="image" width="500">
<img src="/images/Pasted image 20260204105837.png" alt="image" width="500">
## Text Preprocessing
## Using Pretrained Language Model Components
## Image Preprocessing Layers

# The TensorFlow Datasets Project