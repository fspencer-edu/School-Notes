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


```python
# download dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = tf.keras.utils.get_file("spa-eng.zip", origin=url, cache_dir="datasets",
                               extract=True)
text = (Path(path).with_name("spa-eng") / "spa.txt").read_text()

# each lines contains a Eng, and Spa translation separated by a tabe
# remove spanish characters, and parse the sentence pairs, and shuffle
import numpy as np
text = text.replace("¡", "").replace("¿", "")
pairs = [line.split("\t") for ilne in text.splitlines()]
np.random.shuffle(pairs)
sentences_en, sentences_es = zip(*pairs)

for i in range(3):
	print(sentences_en[i], "=>", sentences_es[i])
How boring! => Qué aburrimiento!
I love sports. => Adoro el deporte.
Would you like to swap jobs? => Te gustaría que intercambiemos los trabajos?

# create two text vec layers (one per langauge)
vocab_size = 1000
max_length = 50
text_vec_layer_en = tf.keras.layers.TextVectorization(
	vocab_size, output_sequence_length=max_length)
text_vec_layer_es = tf.keras.layers.TextVectorization(
	vocab_size, output_sequence_length=max_length)
text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {S} endofseq"] for s in sequences_es)
```

- Input sequence will be passed with zeros to reach 50 tokens long
- Longer sequences will be cropped to 50
- The added strings are SOS and EOS tokens

```python
>>> text_vec_layer_en.get_vocabulary()[:10]
['', '[UNK]', 'the', 'i', 'to', 'you', 'tom', 'a', 'is', 'he']
>>> text_vec_layer_es.get_vocabulary()[:10]
['', '[UNK]', 'startofseq', 'endofseq', 'de', 'que', 'a', 'no', 'tom', 'la']

# train and valid set
X_train = tf.constant(sentences_en[:100_000])
X_valid = tf.constant(sentences_en[100_000:])
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])
Y_train = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[:100_000]])
Y_valid = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[100_000:]])

# build model
encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

# encode sentences
embed_size = 128
encoder_input_ids = text_vec_layer_en(encoder_inputs)
decoder_input_ids = text_vec_layer_es(decoder_inputs)
encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
			mask_zero=True)
decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
			mask_zero=True)
encoder_embeddings = encoder_embeddings_layer(encoder_input_ids)
decoder_embeddings = decoder_embeddings_layer(decoder_input_ids)

# create the encoder and pass embedded inputs
encoder = tf.keras.layers.LSTM(512, return_state=True)
encoder_outputs, *encoder_state = encoder(encoder_embeddings)

# Use double state from LSTM layer as the initial state of the decoder
decoder = tf.keras.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

# pass decoder's outputs through a Dense layer with softmax act. fn. to get proba
output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
Y_proba = output_layer(decoder_outputs)
```

- Sampled softmax
	- Look only at the logits output by the model for the correct word and for a random sample of incorrect words, then compute an approximation of the loss based on these logits
- Tie the weights of the output layer to the transpose of the decoder's embedding matrix
	- Reduces the model parameters
	- Orthogonal matrix

```python
model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
			outputs=[Y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
			metrics=["accuracy"])
model.fit((X_train, X_train_dec), y_train, epochs=10,
			validation_data=((X_valid, X_valid_dec), Y_valid))
```
- After training, the decoder expects the input to be the word that was predicted at the previous time step
	- Autoregressive models
- Write a custom memory call that keeps track of the previous output and feeds it to the next encoder

```python
def translate(sentence_en):
	translation = ""
	for word_idx in range(max_length):
		X = np.array([sentence_en])
		X_dec = np.array(["startofseq " + translation])
		y_proba = model.predict((X, X_dec))[0, word_idx]
		predicted_word_id = np.argmax(y_proba)
		predicted_word = text_vec_layer_es.get_vocabulary()[predicted_word_id]
		if predicted_word == "endofseq":
			break
		translation += " " + predicted_word
	return translation .strip()
	
>>> translate("I like soccer")
'me gusta el fútbol'
>>> translate("I like soccer and also going to the beach")
'me gusta el fútbol y a veces mismo al bus'
```

- To increase the training set size add more `LSTM` layers to both the encoder and decoder

## Bidirectional RNNs

- Causal (cannot look into the future)
	- At each time step, a regular recurrent layer only look as past and present inputs before generating its output
		- Seq2seq in decoder
		- Forecasting
- For classification tasks it is preferable to look ahead at the next words before encoding a word
	- Seq2seq in encoder
- Bidirectional recurrent layer
	- Run 2 recurrent layers on the same inputs, one reading the words from left, and other from right

![[Pasted image 20260303114540.png]]

- Wrap a recurrent layer

```python
encoder = tf.keras.layers.Bidirectional(
	tf.keras.layers.LSTM(256, return_state=True))
```
- Bidirectional layer will create a clone of the `LSTM` layer (but in the reverse direction)
- Run both and concatenate their outputs
- This layer will return 4 states instead of 2
	- The final short-term and long-term (forward, and backward)
	- Decoder takes 2 states (short and long-term)
		- Decoder is unidirectional

```python
# concat. the short and long terms states
encoder_outputs, *encoder_state = encoder(encoder_embeddings)
encoder_state = [tf.concat(encoder_state[::2], axis=-1)],
				tf.concat(encoder_state[1::2], axis=-1)
```

## Beam Search

- Keeps track of a short list of the k most promising sentences, and at each decoder step it tries to extend them by one word
	- $k$ parameter is the beam width
- Compute the probabilities of each of the 3000 two-word sentences
- Multiply the estimated conditional probability of each word by the estimated probability of the sentence it completes

![[Pasted image 20260303195219.png]]

- This model is bad at translating long sentences, from the limited short-term memory of RNNs
- Attention mechanisms used to address this problem


# Attention Mechanisms

- The path of a word to its translation is passes through many steps before it is used
- Decoders can focus on the appropriate words (as encoded by the encoder) at each time step
	- Path from an input word to its translation is shorter
- Bilingual evaluation understudy (BLEU) score
	- Compares each translation produced by the model with several good translations produced by humans
		- Count number of n-grams that appear in any of the target translations and adjust the score to take into account the frequency of the produced n-grams in the target
- Instead of sending the encoder's final hidden state to the decoder, as well as the previous target at each step
- Send all of the encoder's output (aggregates) to the decoder as well
- At each time step, the decoder's memory cell computes a weighted sum of all the encoder outputs
	- $\alpha_{(t,i)}$

![[Pasted image 20260303195827.png]]

- Weights are generated by a small network called an alignment model (attention layer)
	- Trained jointly with the rest of the encoder-decoder model
- Starts with a dense layer composed of a single neuron that processes each of the encoder's outputs, with decoder's previous hidden state
	- Outputs a score (energy) for each encoder output
	- Measures how well each output is aligned with the decoder's previous hidden state
- Scores go through a softmax layer to get a final weight for each encoder output
- Bahdanau attention
	- All weights for a given decoder time step add up to 1
	- Concatenates the encoder output with decoder's previous hidden state (concatenate attention/additive attention)

- Luong attention/multiplicative attention
	- Measure the similarity between on of the encoder's outputs and the decoder's previous hidden state
		- Computes the dot product of these two vectors
	- Both vectors have the same dimensionality
- "General" dot product approach
	- Encoder outputs first go through a fully connected layer (without a bias term) before the dot products are computed

- Attention mechanisms

![[Pasted image 20260303200431.png]]

```python
encoder = tf.keras.layers.Bidirectional(
	tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
)
```
- Create the attention layer and pass it the decoder's states and encoder's outputs
- Write a memory cell for decoder's states

```python
attention_layer = tf.keras.layers.Attention()
attention_outputs = attention_layer([decoder_outputs, encoder_outputs])
output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")
Y_proba = output_layer(attention_outputs)

# train model
>>> translate("I like soccer and also going to the beach")
'me gusta el fútbol y también ir a la playa'
```

- Attention layer provides a way to focus attention on the model on part of the inputs
	- Differentiable memory retrieval mechanism
- Directionally lookup
	- Subject, verb
- Computes a similarity measure between the query at each key in the dictionary, and use the softmax function to convert these similarity scores to weights that add up to 1
- `Attention` and `AdditiveAttention` layers both except a list as input
	- Queries, keys, and values
- Decoder outputs are the queries
	- Returns a weighted sum of the encoder outputs that are most similar to the decoder output
- Encoder outputs are the keys and values

## Attention Is All You Need: The Original Transformer Architecture

- Transformer
	- Model is not recurrent
		- Reduces vanishing or exploding gradients
	- NMT without recurrent or convolutional layers, just attention mechanisms
- Each embedding layer outputs a 3D tensor (batch size, sequence length, embedding size)
- Tensors are transformed, but remain the same shape

![[Pasted image 20260303201317.png]]

- At inference time, call the transformer multiple times, producing the translations one word at a time, with partial translations to the decoder
- The encoder transforms the inputs, until it captures the meaning of the word

Encoder: "like" => to be fond of

Decoder: "SOS me gusta el fútbol" -> "el + 1" => "fútbol"

- Both encoder and decoder contain modules that are stacked N times
- Each word is treated independently from each other
	- Encoder's multi-head attention layer updates each word representation by attending to all words in the same sentence
- Decoder's masked multi-head attention layer does the same thing, but when it processes a word (causal layer)
- Decoder's upper multi-head attention layer is where the decoder pays attention to the words in the English sentence
	- Cross-attention
- Positional encodings are dense vectors

## Positional Encodings

- A positional encoding is a dense vector that encodes the position of a word within a sentence
- Encoder all the positions form 0 to max sequence length in batch, then add to word embeddings
- Broadcasting will ensure that the positional encodings get applied to every input sequence

```python
max_length = 50
embed_size = 128
pos_embed_layer = tf.keras.layer.Embedding(max_length, embed_size)
batch_max_len_enc = tf.shape(encoder_embeddings)[1]
encoder_in = encoder_embeddings + pos_embed_layer(tf.range(batch_max_len_enc))
batch_max_len_dec = tf.shape(decoder_embeddings)[1]
decoder_in = decoder_embeddings + pos_embed_layer(tf.range(batch_max_len_dec))
```

- Assumes embeddings are regular tensors, not ragged tensors
- Instead of trainable positional encodings, use fixed positional encodings, based on the sine and cosine functions at different frequencies

- Sine/cosine positional encodings

![[Pasted image 20260303202605.png]]

- Model as access to the absolute position for each word in the sentence because there is a unique positional encoding for each position
- Precompute encoding matrix in the constructor

```python
class PositionalEncoding(tf.keras.layes.Layer):
	def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
		super().__init__(dtype=dtype, **kwargs)
		assert embed_size % 2 == 0, "embed_size must be even"
		p, i = np.meshgrid(np.arange(max_length),
					2 * np.arange(embed_size // 2))
		pos_emb = np.empty((1, max_length, embed_size))
		pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
		pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
		self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
		self.supports_masking = True
		
	def call(self, inputs):
		batch_max_length = tf.shape(inputs)[1]
		return inputs + self.pos_encdings[:, :batch_max_length]
		
# use this layer to add the pos. encoding to the encoder's inputs
pos_embed_layer = PositionalEncoding(max_length, embed_size)
encoder_in = pos_embed_layer(encoder_embeddings)
decoder_in = pos_embed_layer(decoder_embeddings)
```

## Multi-head Attention

- Scaled dot product attention layer

![[Pasted image 20260303203305.png]]

$Q$ = matrix containing one row per query
$K$ = matrix containing one row per key
$V$ = matrix containing one row per value
$QK^T$ = Contains one similarity score for each query.key pair
$1/\sqrt{d_{keys}}$ = scales down similarity score

- `Attention` layer's input are like Q, K, and V, with an extra batch dimension

![[Pasted image 20260303203538.png]]


- 



# An Avalanche of Transformer Models

# Vision Transformers

# Hugging Face's Transformers Library