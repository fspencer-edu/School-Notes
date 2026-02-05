
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
dataset = dataset.map(lambda: x: x * 2)
for item in dataset:
	print(item)
tf.Tensor([ 0  2  4  6  8 10 12], shape=(7,), dtype=int32)
tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
[...]
```

- Spawn multiple threads with `num_parallel_call` to speed up processing
- `filter()` method is used to change dataset

```python
dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)
for item in dataset:
	print(item)
tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int32)
tf.Tensor([ 2  4  6  8 10 12 14], shape=(7,), dtype=int32)

for item in dataset.take(2)
	print(item)
tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int32)
```

## Shuffling the Data


- GD works best when the instances in the training set are independent and identically distributed (IID)
- Shuffle instances
	- Creates a new dataset that will start by filling up a buffer with the first items of the source dataset
	- Pull randomly, and replace with a new item form dataset
	- Specify buffer size
- Provide a random seed for a random order every time the program is ran

```python
dataset = tf.data.Dataset.range(10).repeat(2)
dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
for item in dataset:
	print(item)
tf.Tensor([3 0 1 6 2 5 7], shape=(7,), dtype=int64)
tf.Tensor([8 4 1 9 4 2 3], shape=(7,), dtype=int64)
tf.Tensor([7 5 0 8 9 6], shape=(6,), dtype=int64)
```

- Creates and displays a dataset containing the integers 0-9, repeated twice, shuffled using a buffer 4, and a random seed 42, batched with a size of 7
- For large datasets, shuffling-buffer may not work
- Shuffle the source data itself
- To shuffle the instances more, split the source data into multiple files, then read in a random order during training
- Instances in the same file will end up close
- Pick multiple files randomly and read simultaneously, interleaving records

## Interleaving Lines from Multiple Files

- Load dataset, shuffle it, and split into training and validation set
- Test set
- Split each set into many CSV files

```python
MedInc,HouseAge,AveRooms,AveBedrms,Popul…,AveOccup,Lat…,Long…,MedianHouseValue
3.5214,15.0,3.050,1.107,1447.0,1.606,37.63,-122.43,1.442
5.3275,5.0,6.490,0.991,3464.0,3.443,33.69,-117.39,1.687
3.1,29.0,7.542,1.592,1328.0,2.251,38.44,-122.98,1.621
[...]

>>> train_filepaths
['datasets/housing/my_train_00.csv', 'datasets/housing/my_train_01.csv', ...]

# Create dataset with these filepaths
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

# interleave from 5 files
n_readers = 5
dataset = filepath_dataset.interleave(
	lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
	cycle_length=n_readers)
```

- Result is 7 datasets
	- Filepath dataset
	- Interleave dataset
	- 5 datasets created internally from interleaving
- Have files of identical length
- `interleave()` does not use parallelism
	- Add `num_parallel_calls` to the number of threads

```python
for lin in dataset.take(5):
	print(line)
tf.Tensor(b'4.5909,16.0,[...],33.63,-117.71,2.418', shape=(), dtype=string)
tf.Tensor(b'2.4792,24.0,[...],34.18,-118.38,2.0', shape=(), dtype=string)
tf.Tensor(b'4.2708,45.0,[...],37.48,-122.19,2.67', shape=(), dtype=string)
tf.Tensor(b'2.1856,41.0,[...],32.76,-117.12,1.205', shape=(), dtype=string)
tf.Tensor(b'4.1812,52.0,[...],33.73,-118.31,3.215', shape=(), dtype=string)
```

## Preprocessing the Data

- Parse the strings and scale the data

```python
X_mean, X_std = [...]
n_inputs = 8

def parse_csv_line(line):
	defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
	fields = tf.io.decode_csv(line, record_defaults=defs)
	return tf.stack(fields[:-1]), tf.stack(fields[-1:])
	
def preprocess(line):
	x, y = parse_csv_line(line)
	return (x - X_mean) / x_std, y
```

- Code assumes that the mean and standard deviation is precomputed of each feature in training set
- Parse function takes one CSV line, parses the value, number of columns, and types
- `td.io.decode.csv()` returns a list of scalar tensors
- Use `tf.stack()` on all tensors except for last one to create a 1D array
- `preprocess()` called the parse function, scales the input feature by subtracting the feature means and dividing by the feature standard deviation, and return a tuple containing the scaled features and the target

```python
preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
(<tf.Tensor: shape=(8,), dtype=float32, numpy=
 array([ 0.16579159,  1.216324  , -0.05204564, -0.39215982, -0.5277444 ,
        -0.2633488 ,  0.8543046 , -1.3072058 ], dtype=float32)>,
 <tf.Tensor: shape=(1,), dtype=float32, numpy=array([2.782], dtype=float32)>)
```

- Preprocessing function can convert an instance from a byte string to a scaled tensor with labels
- Use `map()` to apply the function to each sample in the dataset


## Putting Everything Together

- Put everything into a helper function
- Create and return a dataset that will efficiently load data from multiple CSV files, preprocess, shuffle t, and batch it

```python
def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None,
			n_parse_threads=5, shuffle_buffer_size=10_000, seed=42,
			batch_size=32):
	dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
	dataset = dataset.interleave(
		lambda filepath: tf.Data.TextLineDataset(filepath).skip(1),
		cycle_length=n_readers, num_parallel_calls_n_read_threads)
	dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
	dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
	return dataset.batch(batch_size).prefetch(1)
```


<img src="/images/Pasted image 20260204105738.png" alt="image" width="500">

## Prefetching

- `prefetch(1)` creates a dataset that will do its best to always be one batch ahead
- While the training algorithm works on one batch, the dataset will be working in parallel on getting the next one ready
- Ensure loading and preprocessing are multithreaded
- Make preparing one batch of data shorter than running a training step on the GPU
	- GPU will almost be 100% utilized, and training will be faster

<img src="/images/Pasted image 20260204105749.png" alt="image" width="500">
GPU
- RAM
- Memory bandwidth

- Speed up training by using the dataset's `cache()`
- Use after loading and preprocessing the data, but before shuffling, repeating, batching, and prefetching
- Each instance will only be read and preprocessed one

**Other Methods**
- `concatenate()`
- `zip()`
- `window()`
- `reduce()`
- `shard()`
- `flat_map()`
- `apply()`
- `unbatch()`
- `padded_batch()`


## Using the Dataset with Keras

- Training set will be shuffled at each epoch

```python
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)
```

- Build and train a Keras model using these datasets
- Pass `train_set` into `fit()`, and `validation=valid_set` 
	- Take care of repeating the training dataset once per epoch, using a different random order

```python
model = tf.keras.Sequental([...])
model.compile(loss="mse", optimizer="sgd")
model.fit(train_set, validation_data=valid_set, epoch=5)

# Evaluate and predict
test_mse = model.evaluate(test_set)
new_set = test_set.take(3)
y_pred = model.predict(new_set)

# Custom training loop
n_epochs = 5
for epoch in range(n_epochs):
	for X_batch, y_batch in train_set:
		[...] # perform one gradient descent step
		
		
```


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