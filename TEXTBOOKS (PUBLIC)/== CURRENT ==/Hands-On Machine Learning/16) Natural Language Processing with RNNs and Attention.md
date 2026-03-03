- NLP tasks
	- Text classification
	- Translation
	- Summarization
	- Question answering
- Character RNN
- Stateless RNN
- Stateful RNN
- Sentiment analysis
- Neural machine translation (NMT)

- Attention mechanisms
- Transformer

- Models
	- GPT
	- BERT

# Generating Shakespearean Text Using a Character RNN

## Creating the Training Dataset

```python
import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare" 
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespear_url)
with open(filepath) as f:
	shakespeare_text = f.read()
	
print(shakespeare_text[:80])
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

- Use `TextVectorization` layer to encode this text
- Split to character-level encoding, and use lower case letters

```python
text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
```
0 => padding tokens
1 => unknown characters

```python
encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() - 2
dataset_size = len(encoded) # chars = 1,115,394
```

- Convert this long sequence into a dataset of windows that we can use to train a sequence-to-sequence RNN

```python
def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
	ds = tf.data.Dataset.from_tensor_slices(sequences)
	ds = ds.window(length + 1, shift=1, drop_remainder=True)
	ds. = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
	if shuffle:
		ds = ds.shuffle(buffer_size=100_000, seed=seed)
	ds = ds.batch(batch_size)
	return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)
```
- Takes a sequence as input, and creates a dataset containing all the window of desired length
- Increases the length by once
- Shuffle the windows, batches them, splits then into input/output pairs, and activates prefetching

<img src="/images/Pasted image 20260204110756.png" alt="image" width="500">

```python
# training, valid, test
length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True,
			seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)
```

- Set the window length to 100


## Building and Training the Char-RNN Model

- Build and train a model with one `GRU` layer composed of 128 units

```python
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
	tf.keras.layers.GRU(128, return_sequences=True),
	tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compiile(loss="spares_categorical_crossentropy", optimizer="nadam",
		metrics=["acrruacy"])
nidek_ckpt = tf.keras.callbacks.ModelCheckpoint(
	"my_shakespear_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10, 
			callbacks=[model_ckpt])
```

- Use the `Embedding` layer to encode the character IDs
	- Input is a 2D tensor `[batch size, window length]`
	- Output is a 3D tensor `[batch size, window length, embedding size]`
- Use a `Dense` layer for the output layer with 39 distinct characters in the text
- Compile the model with cross-entropy loss and a Nadam optimizer
- Train models for several epochs and save the best model with `ModelCheckpoint`

- Wrap the final model layer, as the first layer plus `Lambda` layer to subtract 2 from the character Ids (remove padding and unknown tokens)

```python
shakespeare_model = tf.keras.Sequential([
	text_vec_layer,
	tf.keras.layers.Lambda(lambda X: X - 2),
	model
])

# predict the next char
y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)
text_vec_layer.get_vocabulary()[y_pred + 2]
'e'
```

## Generating Fake Shakespearean Text

- Greedy decoding
	- Generate new text using the char-RNN model
	- Iteratively guess the next letter
- Instead, sample the next characters randomly, with a probability equal to the estimated probability, using `tf.random.categorical()`
	- Generate a more diverse and interesting text

```python
log_probas = tf.math.log([[0.5, 0.4, 0.1]])
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)
<tf.Tensor: shape=(1, 8), dtype=int64, numpy=array([[0, 1, 0, 2, 1, 0, 0, 1]])>
```

- Divide the logits by a number called temperature,
- A temperature close to zero favours high-probability characters, while a high temperature gives all characters an equal probability
	- Used for more rigid and precise text (math equations)

```python
def next_char(text, temperature=1):
	y_proba = shakespeare_model.predict([text])[0, -1:]
	rescaled_logits = tf.math.log(y_proba) / temperature
	char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
	return text_vec_layer.get_vocabulary()[char_id + 2]
	
# get the next char and append to the text
def extend_text(text, n_char=50, temperature=1):
	for _ in range(n_chars):
		text += next_char(text, temperature)
	return text
	
# generate text with diff. temperature
tf.random.set_seed(42)
print(extend_text("To be or not to be", temperature=0.01))
To be or not to be the duke
as it is a proper strange death,
and the

print(extend_text("To be or not to be", temperature=1))
To be or not to behold?
second push:
gremio, lord all, a sistermen,

print(extend_text("To be or not to be", temperature=100))
To be or not to bef ,mt'&o3fpadm!$
wh!nse?bws3est--vgerdjw?c-y-ewznq
```

- Nucleus sampling
	- To generate more convincing text, a common technique is to sample only from the top k characters, or from the smallest set of top characters whose total probability exceeds some threshold
- Beam search


## Stateful RNN

- Stateless RNNs
	- At each training iteration the model starts with a hidden state full of zeros, then is updates this state at each time step
- Stateful RNNs
	- Preserve final state after processing a training batch and use it as the initial state for the next raining batch
- Use sequential and non-overlapping input sequences
	- `shift=length`
	- No shuffle
- Set a batch size of 1

```python
def to_dataset_for_stateful_rnn(sequence, length):
	ds = tf.data.Dataset.from_tensor_slices(sequences)
	ds = ds.window(length + 1, shift=length, drop_remainder=True)
	ds = ds.flat_map(lambda window: window.batch(length =1)).batch(1)
	return ds.map(lamda window: (window[:, :-1], window[:, 1:])).prefetch(1)
	
stateful_train_set = to_dataset_for_stateful_rnn(encoded[:1_000_000], length)
stateful_valid_set = to_dataset_for_stateful_rnn(encoded[1_000_000:1_060_000],
                                                 length)
statefule_test_set = to_dataset_for_stateful_rnn(encoded[1_060_000:], length)
```

![[Pasted image 20260303082855.png]]

```python
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16,
				batch_input_shape=[1, None]),
	tf.keras.layers.GRU(128, return_sequences=True, stateful=True),
	tf.keras.layers.Dense(n_tokens, activation="softmax")
])

# reset the state, at the end of each epoch
class ResetStatesCallback(tf.keras.callbacks.Callback):
	def on_epoch_begin(self, epoch, log):
		self.model.reset_states()
		
# compoile and train model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
				metrics=["accuracy"])
history = model.fit(stateful_train_set, validation_data=stateful_valid_set,
			epochs=10, callbacks=[ResetStatesCallback(), model_ckpt])
```

- The model can make predictions for batches of the same size during training
- Create an identical stateless model, and copy the stateful model's weights to this model

- Sentiment neuron

# Sentiment Analysis

```python
import tensorflow_datasets as tfds

raw_train_set, raw_valid_set, raw_test_test = tfds.load(
	name="imdb_reviews",
	split=["train[:90%]", "train[90%:]", "test"],
	as_superivsed=True
)
tf.random.set_seed(42)
train_set = raw_train_set.shuffle(5000, seed=42).batch(32).prefetch(1)
valid_set = raw_valid_set.batch(32).prefetch(1)
test_set = raw_test_set.batch(32).prefetch(1)

for review, label in raw_train_set.take(4):
	print(review.numpy().decode("utf-8"))
	print("Label:", label.numpy())
This was an absolutely terrible movie. Don't be lured in by Christopher [...]
Label: 0
I have been known to fall asleep during films, but this is usually due to [...]
Label: 0
Mann photographs the Alberta Rocky Mountains in a superb fashion, and [...]
Label: 0
This is the kind of film for a snowy Sunday afternoon when the rest of the [...]
Label: 1
```

- To analyze sentiment, preprocess the text by splitting into words instead of characters
- Tokenize and detokenize text at the subword level
- Byte pair encoding (BPE)
	- Splitting the whole training set into individual characters (including spaces)
	- Then repeatedly merging the more frequently adjacent pairs until the vocabulary reaches the desired size
- Subword regularization
	- Improves accuracy and robustness
	- Add randomness in tokenization during training
- For IMDb, use 1000 tokens with 998 more frequent words plus a padding token and a token for unknown words

```python
vocab_size =1000
text_vec_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
text_vec_layer.adapt(train_set.map(lambda reviews, lambdas: reviews))

# create and train model
embed_size = 128
tf.random.set_seed(42)
model = tf.keras.Sequential([
	text_vec_layer,
	tf.keras.layers.Embedding(vocab_size, embed_size),
	tf.keras.layers.FRU(128),
	tf.keras.layers.Dense(1, activation="sigmoid")
])
modile.compile(loss="binary_crossentropy", optimizer="nadam"),
			metrics=["accuracy"]
history = model.fit(train_set, validation_data=valid_set, epochs=2)
```

- The output will be the estimated probability that the review expresses a positive sentiment
- The reviews have different lengths, and adds padding tokens to make them as long as the longest sequence in the batch

## Masking

- Making the model ignore padding tokens
	- `mask_zero=true`
- `Embedding` layer creates a mask tensor equal to `tf.math.not_equal(inputs, 0)`
- Some layers need to update the mask before propagating it to the next layer
	- Computes mask
		- Takes inputs and previous mask
		- Finds the updated mask

```python
inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
tokens_ids = text_vec_layer(inputs)
mask = tf.math.not_equal(token_ids, 0)
Z = tf.keras.layers.Emedding(vocab_size, embed_size)(token_ids)
Z = tf.keras.layers.GRU(128, dropout=0.2)(Z, mask=mask)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(Z)
model = tf.keras.Model(inputs=[inputs], outptus=[outputs])

# ragged tensors
text_vec_layer_ragged = tf.keras.layers.TextVectorization(
	max_tokens_vocab_size, ragged=True)
text_vec_layer_ragged.adapt(train_set.map(lambda reviews, labels: reviews))
text_vec_layer_ragged(["Great movie!", "This is DiCaprio's best role."])
<tf.RaggedTensor [[86, 18], [11, 7, 1, 116, 217]]>

text_vec_layer(["Great movie!", "This is DiCaprio's best role."])
<tf.Tensor: shape=(2, 5), dtype=int64, numpy=
array([[ 86,  18,   0,   0,   0],
       [ 11,   7,   1, 116, 217]])>
```

- Not possible to use ragged tensors as targets while running on the GPU
- Visualize the embeddings in TensorBoard as they are being learned


## Reusing Pretrained Embeddings and Language Models

- Reuse word embeddings from a large text corpus
- Pretrained models
	- Word2vec embeddings
	- GloBVe embeddings
	- FastText embeddings
- A word has a single representation
- Embeddings from Language Models (ELMo)
	- Contextualized word embeddings learned from the internal states of a deep bidirectional language model

```python
import os
import tensorflow_hub as hub

os.environ["TFHUB_CACHE_DIR"] = "my_tfhub_cache"
model = tf.keras.Sequential([
	hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
	trainable=True, dtype=tf.string, input_shape([]),
	tf.keras.layers.Dense(64, activation="relu"),
	tf.keras.layers.Dense(1, activation="sigmoid")
])
modle.compile(loss="binary_crossentropy", optimizer="nadam",
				metrics=["accuracy"])
model.fit(train_set, validation_data=valid_set, epochs=1)
```

- By default TensorFlow Hub modules are saved to a temporary directory
- Downloaded every time it is run

# An Encoder-Decoder Network for Neural Machine Translation

- English sentences are fed as inputs to the encoder
- The decoder outputs the Spanish translation
- Spanish translations are also used as inputs to the decoder turning training, but shifted back by one step
- Teach forcing
	- Decoder is given the input is should have output previous step
	- Decoder is given the start-of-sequence (SOS) token, and the decoder is expected to end the sequence with an end-of-sequence (EOS) token
- Each word is initially represented by its ID
- Word embedding are then fed to the encoder and decoder
- Decoder outputs a score for each word in the output vocab
- Softmax activation function turns these scores into probabilities
	- Word with the highest probability is the output

![[Pasted image 20260303111856.png]]

- After training (inference time), the target sentence will not feed to the decoder
- Feed the word that is has just output at the previous step
- Feeding the decoder the previous target token to feeding the previous output token during training

![[Pasted image 20260303112012.png]]





## Bidirectional RNNs

## Beam Search
# Attention Mechanisms

## Attention Is All You Need: The Original Transformer Architecture

# An Avalanche of Transformer Models

# Vision Transformers

# Hugging Face's Transformers Library