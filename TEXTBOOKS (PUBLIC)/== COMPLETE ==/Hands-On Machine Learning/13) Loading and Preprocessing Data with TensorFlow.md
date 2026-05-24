
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
		
# Create a TF function that ttrains the model for a whole epoch
@tf.function
def train_one_epoch(model, optimizer, loss_fn, train_set):
	for X_batch, y_batch in train_set:
		with tf.GradientTape() as tape:
			y_pred = model(X_batch)
			main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
			loss = tf.add_n([main_loss] + model.losses)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.appy_gradients(zip(gradients, model.trainable_variables))
		
optimizer = tf.keras.optimizer.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
for epoch in range(n_epochs):
	print("\rEpoch {}/{}".format(epoch + 1, n_epoch), end="")
	train_one_epoch(model, optimizer, loss_fn, train_set)
```

# The TFRecord Format

- TFRecord format is TF preferred format for storing large amount of data
- Simple binary format containing a sequence of binary of varying sizes
	- Length
	- CRC checksum for length
	- Data
	- CRC checksum for data

```python
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
	f.write(b"This is the first record")
	f.write(b"This is the second record")
	
# Read one or more files
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
	print(item)
tf.Tensor(b'This is the first record', shape=(), dtype=string)
tf.Tensor(b'And this is the second record', shape=(), dtype=string)
```

- By default will read one by one


## Compressed TFRecord Files

- Compress TFRecords

```python
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
	f.write(b"Compress, compress, compress!")
	
# read compressed file
dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
				compression_type="GZIP")
```

## A Brief Introduction to Protocol Buffers

- TFRecords files contain serialized protocol buffers (protobufs)
- Portable extensible, and efficient binary format developed by Google (2001)

```python
syntax = "proto3";
message Person {
	string name = 1;
	int32 id = 2;
	repeated string email = 3;
}
```

- Using version 3 format, specifies `Person` objects, that has a name, id, and zero or more `email` fields
- Compile `.proto` file using `protoc`, to generate access classes in Python

```python
from person_pb2 impor Person
person = Person(name="A1", id=123, email["a@b.com"])
print(person)
name: "Al"
id: 123
email: "a@b.com"
person.name
'A1'
person.name = "Fiona"
person.email[0]
'a@b.com'
person.email.append("c@d.com"
serialized = person.SerializeToString()
b'\n\x05Fiona\x10{\x1a\x07a@b.com\x1a\x07c@d.com'
person2 = Person()
person2.ParseFromString(serialized)
27
person == person2
True
```

- Create an `Person` instance
- Save the serialized object to a TFRecord file, then load, and parse it
- `ParseFromString()` is not a TF operation
- Use wrapper `tf.py_function()` operation or `tf.io.decode_proto()`
- Generally use the predefined protobuf for which TF provides dedicated parsing operations

## TensorFlow Protobufs

- Main protobuf is used in a `TFRecord` file is the `Example`
	- One instance in the dataset
	- List of named features

```python
syntax = "proto3";
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };
```

- TF developers may decide to add more fields to it

```python
from tensorflow.train import ByteList, FloatList, Int64List
from tensorflow.train impot Feature, Features, Example

person_example = Example(
	features=Features(
		feature={
			"name": Feature(bytes_list=BytesList(value=[b"Fiona"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
		}
	)
)

# wrap code in smaller helper function
with tf.io.TFRecordWriter("my_contacts.tfrecord") a f:
	for _ in range(5):
		f.write(person_example.SerializeToString())
```
- Write more than 5 `Example`
- Create a conversion script that reads from current format, creates an `Example` protobuf for each instance, serialized them, and saves to several TFRecords (shuffling)

## Loading and Parsing Examples

- Code defines a description dictionary, then creates a `TFRecordDataset` and applied a custom preprocessing function to it to parse each serialized protobuf

```python
feature_description = {
	"name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

def parse(serialized_example):
	return tf.io.parse_single_example(serialized_exmple, feature_description)
	
dataset = tf.data.TFRecordDataset(["my_contacts.tfrecrod"]).map(parse)
for parsed_example in dataset:
	print(parse_example)
```

- Convert a sparse tensor to a dense tensor, `tf.sparse.to_dense()`

```python
>>> tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
>>> parsed_example["emails"].values
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>

# Parse by batch
def parse(serialized_examples):
	return tf.io.parse_example(serialized_examples, feature_description)
	
dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(2).map(parse)
for parsed_examples in dataset:
	print(parsed_examples)
```
## Handling Lists of Lists Using the SequenceExample Protobuf

```python
message FeatureList { repeated Feature feature = 1 };
message FeatureLists { maps<string, FeatureList> feature_list = 1; };
message SequenceExample {
	Features context = 1;
	FeatureLists feature_lists = 2;
};
```
- If the feature list contain sequences of varying sizes, convert them to ragged tensors

```python
parsed_content, parsed_feature_lists = tf.io.parse_signle_sequence_example(
	serizlized_sequence_example, context_feature_descriptions,
	sequence_feature_descriptions
)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
```

# Keras Preprocessing Layers

- Preparing data
	- Normalization the numerical features
	- Encoding the categorical features
	- Cropping and resizing images

## The Normalization Layer

- Specify the mean and variance of each feature when creaming the layer
- Pass the training set to the layer's `adapt()`method before fitting the model

```python
norm_layer = tf.keras.layers.Normalization()
model = tf.keras.models.Sequential([
	norm_layer,
	tf.keras.layers.Dense(1)
])

# train model
model.compile(loss="mse", optimizer=tf.keras.optimizer.SGD(leaning_rate=2e-3))
norm_layer.adapt(X_train)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5)
```

- Deploy the model to production without normalizing again

<img src="/images/Pasted image 20260204105806.png" alt="image" width="500">
- Including the preprocessing layer directly i the model is easy, but slows down training
	- Happens once per epoch
- Normalize the entire training set once before training instead

```python
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)
X_train_scaled = norm_layer(X_train)
X_valid_scaled = norm_layer(X_valid)

# train model on the scaled data
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
model.comple(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
model.fit(X_train_Scaled, y_train, epoch=5,
	validation_data=(X_valid_scaled, y_valid))
```

- Create a new model that wraps both the adapted `Normalization` layer and the trained model
	- Preprocess new inputs

```python
final_model = tf.keras.Sequential([norm_layer, model])
X_new = X_test[:3]
y_pred = final_model(X_new)
```

<img src="/images/Pasted image 20260204105817.png" alt="image" width="500">
- Apply an adapt `Normalization` layer to the input feature of each batch in a dataset

```python
dataset = dataset.map(lambda X, y: (norm_layer(X), y))
```

- Write custom Keras layers

```python
import numpy as np

class MyNormalization(tf.keras.layers.Layer):
	def adapt(self, X):
		self.mean_ = np.mean(X, axis=0, keepdims=True)
		self.std_ = np.std(X, axis=0, keepdims=True)
		
	def call(self, inputs):
		eps = tf.keras.backend.epsilon()
		return (inputs - self.mean) / (self.std_ + eps)
```

## The Discretization Layer

- Convert age feature to 3 categories, less than 18, 18 t0 50, and 50 or over

```python
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])
age_cat = discretize_layer(age)
age_cat
<tf.Tensor: shape=(6, 1), dtype=int64, numpy=array([[0],[2],[2],[1],[1],[0]])>
```

- Instead of bin boundaries, provide number of bins `num_bins=3`
- Use one-hot encoding on the discretized values

## The CategoryEncoding Layer

- One-hot encode age features

```python
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)
onehot_laer(age_cat)
<tf.Tensor: shape=(6, 3), dtype=float32, numpy=
array([[0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]], dtype=float32)>
```

- Use count encoding, to contain the number of occurrences of each category
- Multi hot encoding and count encoding loses information
- For multi-hot encoding concatenate the outputs

```python
# two category (non-concate)
two_age_cat = np.array([1, 0], [2, 2], [2, 0])
onehot_layer(two_age_cat)
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[1., 1., 0.],
       [0., 0., 1.],
       [1., 0., 1.]], dtype=float32)>

# two category (concate)
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3+3)
onehot_layer(two_age_cat + [0, 3])
<tf.Tensor: shape=(3, 6), dtype=float32, numpy=
array([[0., 1., 0., 1., 0., 0.],
       [0., 0., 1., 0., 0., 1.],
       [0., 0., 1., 1., 0., 0.]], dtype=float32)>
```

## The StringLookup Layer

```python
cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
<tf.Tensor: shape=(4, 1), dtype=int64, numpy=array([[1], [3], [3], [0]])>

# one-hot encoding
str_look_layer = tf.keras.layers.StringLookup(output_mode="one_hot")
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
array([[0., 1., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.]], dtype=float32)>
```
- Ordered by frequency, reverse alphabetical order
0 → `[OOV]`
1 → Paris
2 → San Francisco
3 → Auckland

- Finds that there are 3 distinct categories
- Unknown categories are mapped to 0

- Each unknown category (OOV) will get mapped pseudo-randomly to one of the OOV buckets, using a hash function

```python
str_lookup_layer = tf.keras.layers.StringLookup(num_oov_indices=5)
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Foo"], ["Bar"], ["Baz"]])
<tf.Tensor: shape=(4, 1), dtype=int64, numpy=array([[5], [7], [4], [3], [4]])>
```
0 → OOV bucket 1
1 → OOV bucket 2
2 → OOV bucket 3
3 → OOV bucket 4
4 → OOV bucket 5
5 → Paris
6 → San Francisco
7 → Auckland

## The Hashing Layer

- For each category, the `Hashing` layer computes a hash, modulo the number of buckets

```python
hashing_layer = tf.keras.layers.Hashing(num_bins=10)
hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]])
<tf.Tensor: shape=(4, 1), dtype=int64, numpy=array([[0], [1], [9], [1]])>
```

## Encoding Categorical Features Using Embeddings


- An embedding is a dense representation of some higher-dimensional data
	- Compute a small dense vector from a large set of categorical data
- Initialized randomly, trained by gradient descent
- Representation learning
	- Word embeddings
<img src="/images/Pasted image 20260204105827.png" alt="image" width="500">
<img src="/images/Pasted image 20260204105837.png" alt="image" width="500">
- Word embeddings capture biases
- Embedding matrix
	- One row per category
	- One column per embedding dimension

```python
tf.random.set_seed(42)
embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2)
embedding_layer(np.array[2, 4, 2])
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[-0.04663396,  0.01846724],
       [-0.02736737, -0.02768031],
       [-0.04663396,  0.01846724]], dtype=float32)>
```

- Embed a categorical text attribute

```python
tf.random.set_seed(42)
ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(ocean_prox)
lookup_and_embed = tf.keras.Sequential([
	str_lookup_layer,
	tf.keras.layers.Embedding(input_dim=str_lookup_layer.vocabulary_size(),
		output_dim=2)
])

lookup_and_embed(np.array([["<1H OCEAN"], ["ISLAND"], ["<1H OCEAN"]]))
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[-0.01896119,  0.02223358],
       [ 0.02401174,  0.03724445],
       [-0.01896119,  0.02223358]], dtype=float32)>
```

- Create a Keras model that can process a categorical test feature along with regular numerical features and learn an embedding for each category

```python
X_train_num, X_train_cat, y_train = [...]
X_valid_num, X_valid_cat, y_valid = [...]

num_input = tf.keras.layers.Input(shape=[8], name="num")
cat_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name="cat")
cat_embeddings = lookup_and_embed(cat_input)
encoded_inputs = tf.keras.layers.concatenate([num_input, cat_embeddings])
outputs = tf.keras.layers.Dense(1)(encoded_inputs)
model = tf.keras.models.Model(inputs=[num_input, cat_input], outputs=[outputs])
model.compile(loss="mse", optimizer="sdg")
history = model.fit((X_train_num, X_train_cat), y_train, epochs=5,
		validation_data=((X_valid_num, X_valid_cat), y_valid))
```

- takes two inputs, which contains  numerical features per instance, and a categorical input, that contains a single categorical text input per instance
- Encodes each ocean-proximity to trainable embedding
- Concatenates the numerical inputs and embeddings
- Compile the model and train it

## Text Preprocessing

- `TextVectorization`

```python
train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
text_vec_layer = tf.keras.layers.TextVectorization()
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])
array([[2, 1, 0, 0],
       [6, 2, 1, 2]])>
```

- Vocab was learned from the 4 sentences in the training data
	- OOV -> 1
	- "be" -> 2
	- "to" -> 3
	- Padding of 0
- Preserve the case and punctuation, `standarize=None`
- Prevent splitting, `split=None`


- Term-frequency x inverse-document-frequency (TF-IDF)
	- Words that occur frequently in the training data are down-weighted, and rare words up-weighted

```python
text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])
<tf.Tensor: shape=(2, 6), dtype=float32, numpy=
array([[0.96725637, 0.6931472 , 0. , 0. , 0. , 0.        ],
       [0.96725637, 1.3862944 , 0. , 0. , 0. , 1.0986123 ]], dtype=float32)>
```

## Using Pretrained Language Model Components

- Model components are called modules
- Contain both pre-processing code and pretrained weights

```python
import tensorflow_hub as hub
hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
sentence_embeddings = hub_layer(tf.constant(["To be", "Not to be"]))
sentence_embeddings.numpy().rount(2)
array([[-0.25,  0.28,  0.01,  0.1 ,  [...] ,  0.05,  0.31],
       [-0.2 ,  0.2 , -0.08,  0.02,  [...] , -0.04,  0.15]], dtype=float32)
```

- This module is a sentence encoder
	- Takes strings as input and encodes each one as a single vector
- Hugging Face

## Image Preprocessing Layers

- Keras API for image preprocessing layers
	- `tf.keras.layers.Resizing`
	- `tf.keras.layers.Rescaling`
	- `tf.keras.layers.CenterCrop`

```python
from sklearn.datasets import load_sample_images

images = load_sample_images()["images"]
crop_image_layer = tf.keras.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)
```

- Data augmentation
	- `RandomCrop`
	- `RandomFlip`
	- `RandomTranslation`
	- `RandomRotation`
	- `RandomZoom`
	- `RandomHeight`
	- `RandomWeight`
	- `RandomConstrast`


# The TensorFlow Datasets Project

- TensorFlow Datasets (TFDS) project makes is easy to load common datasets

```python
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]

# train model
for batch in msist_train.shuffle(10_000, seed=42).batch(32).prefetch(1):
	images = batch["image"]
	labels = batch["label"]
```

- Each item in the dataset is a dictionary containing both features and labels
- Transform the dataset to be a tuple containing two elements

```python
mnist_train = mnist_train.shuffle(buffer_size=10_000, size=42).batch(32)
msnist_Train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefecth(1)
```

- Split the data
	- 90% for training
	- 10% for validation
	- 100% for testing

```python
train_set, valid_set, test_set = tfds.load(
	name="mnist",
	split=["train[:90%]", "train[90%:]", "test"],
	as_superivsed=True
)

train_set = train_set.shuffle(buffer_size=10_000, seed=42).batch(32).prefetch(1)
valid_set = valid_set.batch(32).cache()
test_set = test_set.batch(32).cache()
tf.random.set_seed(42)
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=[28, 28]),
	tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorcal_crossentropy", optimizer="nadam",
			metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)
```