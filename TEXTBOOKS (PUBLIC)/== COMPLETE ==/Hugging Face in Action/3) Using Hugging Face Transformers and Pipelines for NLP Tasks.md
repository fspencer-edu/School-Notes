
## Introduction to the Transformer Architecture

- Transformer architecture
	- Self-attention
	- Process input data in parallel
	- RNNs
	- Long short-term memory (LSTM)

<img src="/images/Pasted image 20260514223153.png" alt="image" width="500">

- Self-attention
	- Models weight importance of words irrespective of distance
	- Long-range dependencies
	- Parallelize input sequences
- Encoder
	- Builds representations from inputs
	- Sentence classification
- Decoder
	- Generate target sequences
	- Text generation

**Key Components**
- Input embedding
	- Convert input tokens to dense vectors of fixed size
	- Positional encodings to retrain information about order
- Encoder
	- Multi-head attention
		- Attention scores between each pair of input tokens
	- Feed forward neural network (FFN)
		- 2 linera transformations
		- Rectified Linear Unit (ReLU)
- Decoder
	- Masked multi-head attention
		- Prevents attending to future tokens in the target sequence
	- Multi-head attention
		- Allows decoder to attend to output
- Positional encoding
	- Add positional info of each token
	- Sine and cosine functions
- Layer normalization and residual connections
	- Layer normalization
		- Stabilizes and accelerates training
	- Residual connections
		- Adds the input of each sublayer to its output to help with gradient flow
		- Prevents vanishing/exploding gradients
- Final linear and softmax layers
	- Decoder
	- Produce probabilities for the next token

### Tokenization

- Token
	- Chuck of text that a model processes as a single unit
	- Represent an individual word, punctuation, or other linguistic elements
- Tokenization
	- Process of converting a text or sentence to smaller units

**Tokenization Strategies**
- Word
	- Splits text into words on whitespace, or punctuation characters
- Subword
	- Break text into smaller linguistic units
	- Prefix, suffixes, or root words
	- Complex morphology tasks
	- Machine translation
- Character-level
	- Segments text into individual characters
		- Letter, digits, and punctuation marks
		- Meticulous analysis


- Subword tokenization using BERT model

```python
from transformer import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
input_text = "What is unhappiness?"
tokens = tokenizer.tokenize(input_text, return_tensors="pt")

print(f"{tokens=}")

tokens = ['what', 'is', 'un', '##ha', '##pp', '##iness', '?']

```

- BERT
	- Bidirectional Encoder Representations from Transformers
	- Designed for NLP
	- Question answering, text classification , NER, speech tagging, text summarization , sentiment analysis, language translation, text generation, coreference resolution, paraphrase detection, semantic search, textural entailment, dialogue systems


- `##` prefix
	- Indicates that the token is a continuation of the previous one in the original word

### Token Embeddings
### Positional Encoding
### Transformer Block
### Softmax

## Working with the Transformers Library

### Transformers Pipelines
### Using Models
### Using a Transformers Pipeline
## Using Transformers for NLP Tasks
### Text Classification
### Text Generation
### Text Summarization
### Text Translation

### Zero-Shot Classification


