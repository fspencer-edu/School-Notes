- Local LLM querying for text based data
- LLM querying for structured tabular data

## Using GPT4All to query with your own data

### Installing the required packages

- `langChain`
- `gpt4all`
- `faoss-cpu`
	- Facebook AI Similarity search
- `huggingface-hub`
- `Sentence-transformers`

```python
$ pip install langchain
$ pip install gpt4all
$ pip install faiss-cpu
$ pip install huggingface-hub
$ pip install sentence-transformers
```

### Import modules from LangChain package

```python
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import GPT4All
```
### Loading the PDF documents

```python
documents =
    PyPDFLoader('./LocalDataForTraining/Invoice1.pdf').load_and_split()
```

- Chunks are returns ad `Document` objects
### Splitting text into chunks

- `RecursiveCharacterTextSplitter`
	- Split the document into chunks of a specific size
- Chunking
	- Process of breaking text into smaller, meaningful units
	- Model works with in a specific content length (4096 tokens = 3500 words)

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,
                                               chunk_overlap = 64)
texts = text_splitter.split_documents(documents)
```

### Embedding

- Embedding
	- Representation of words, phrases, or sentences as vectors in a high-dimensional space
- Word embeddings
	- Capture meaning and relationship of individual words
- Sentence embeddings
	- Meaning of entire sentences of phrases

```python
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2')
faiss_index = FAISS.from_documents(texts, embeddings)

faiss_index.save_local("./index")
```

- Maps sentences an paragraphs to a 384-dimensional dense vector space
- To embedding files
	- `index.faiss`
	- `index.pkl`

### Loading the embeddings

```python
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2')
faiss_index = FAISS.load_local("./index", embeddings)
```

- Recreate the embedding model used to build the index

### Downloading the model

```python
from gpt4all import GPT4All
llm = GPT4All("mistral-7b-openorca.Q4_0.gguf")
```
### Asking questions

```python
# download model
from gpt4all import GPT4All
llm = GPT4All("mistral-7b-openorca.Q4_0.gguf")

# prompt template
template = """
Please use the following context to answer the question concisely 
and without including the context in your answer.
Context: {context}
Question: {question}
Answer:
"""

def ask_question(question):

    matched_docs = faiss_index.similarity_search(question, 4)#1

    context = ""

    for doc in matched_docs:#2
        context += doc.page_content + " \n\n"

    prompt = PromptTemplate(template = template,#3
        input_variables=["context", "question"]).partial(
            context = context)

    chain = prompt | llm | StrOutputParser()#4
    return chain.invoke({"question": question})
    
while True:
    print(ask_question(input('Question: ')))
```

![[Pasted image 20260518150355.png]]

### Loading multiple documents

```python
documents =
    PyPDFLoader('./LocalDataForTraining/Invoice1.pdf').load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,
                                               chunk_overlap = 64)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2')
faiss_index = FAISS.from_documents(texts, embeddings)

# load multiple documents
import os

pdf_folder_path = "./LocalDataForTraining/"
pdf_dir = os.listdir(pdf_folder_path)

pdf_dir.remove('.DS_Store')  #1
loaders = [PyPDFLoader(os.path.join(pdf_folder_path, fn))
              for fn in pdf_dir]  #2
              
# splitting
all_documents = []

for loader in loaders:
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,
                                                   chunk_overlap = 64)
    documents = text_splitter.split_documents(documents)
    all_documents.extend(documents)
    
# embedding
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2')

faiss_index = FAISS.from_documents(all_documents, embeddings)
faiss_index.save_local("./index")
```
### Loading CSV Files

- `CSVLoader`

```python
from langchain.document_loaders import CSVLoader
documents = CSVLoader('./Titanic_train.csv').load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,
                                               chunk_overlap = 64)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
                 model_name = 'sentence-transformers/all-MiniLM-L6-v2')
faiss_index = FAISS.from_documents(texts, embeddings)

def ask_question(question):
    matched_docs = faiss_index.similarity_search(question, 4)  #1

    context = ""
    for doc in matched_docs:  #2
        context += doc.page_content + " \n\n"

    prompt = PromptTemplate(template = template,  #3
        input_variables=["context", "question"]).partial(
            context = context)

    chain = prompt | llm | StrOutputParser()  #4
    return chain.invoke({"question": question})
```
### Loading JSON files

- `jq`
	- Lightweight, command line JSON processor

```python
pip install jq

from langchain.document_loaders import JSONLoader

documents = JSONLoader('./nobel_laureates.json',
                       jq_schema='.laureates[]',
                       text_content=False).load_and_split()
documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,
                                               chunk_overlap = 64)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2')  #1

faiss_index = FAISS.from_documents(texts, embeddings)  #2

template = """
Please use the following context to answer the question concisely
 and without including the context in your answer.
Context: {context}
Question: {question}
Answer:
"""

def ask_question(question):
    matched_docs = faiss_index.similarity_search(question, 4)  #3

    context = ""
    for doc in matched_docs:  #4
        context += doc.page_content + " \n\n"

    prompt = PromptTemplate(template = template,  #5
        input_variables=["context", "question"]).partial(
            context = context)

    chain = prompt | llm | StrOutputParser()  #6
    return chain.invoke({"question": question})  

while True:  
    print(ask_question(input('Question: ')))
```

## Using LLMs to write code to analyze your data

- LLMs lack the necessary context size to process an entire document unless it is exceptionally short
- Analyzer large private datasets
	- Load data programmatically using pandas
	- Prompt the LLM with the schema of the data
	- Find the query to the solve the problem
	- Using the response, execute the response to get the answers
- Implementation
	- Use local model
	- Cloud based model

### Preparing/Loading the JSON file

- Use `json_normalize()` to load the JSON file and split each person's details into individual columns

```python
df = pd.read_json('famous_people.json')

import json
import pandas as pd
from pandas import json_normalize

with open('famous_people.json', 'r') as json_file:
    json_data = json.load(json_file)

df = json_normalize(json_data, 'famous_people')  #1
df
```
### Using a model

- Ask an LLM to propose a solution to query your data

```python
from langchain.llms import GPT4All

model = 'mistral-7b-openorca.Q4_0.gguf'
llm = GPT4All(model = model)

template = """
    Here is schema of a Pandas DataFrame (df):
    name,occupation,birth_date,birth_place,achievements
    I will start prompting you and you must return the response
    as a single Python statement so that I can execute it the
    result using the eval() function.

    For your info I have loaded the JSON file as a df using the
    following code:  
    with open('famous_people.json', 'r') as json_file:
    json_data = json.load(json_file)
    df = json_normalize(json_data, 'famous_people')  #1
    Question: {question}
"""
```
### 
