
- LangChain
	- Customizes NLP application by linking different components based on specific requirements
- LlamaIndex
	- Connect an LLM to proprietary data
## Introducing LLMs

- LLM
	- Model designed to understand and generate humanlike text based on patterns and structures
		- Size and scale
		- Pretraining
		- Fine tuning

**Trainable Parameters**

- Trainable parameters
	- Variables within a ML or deep learning model that are adjusted during the training process to enable the model to make accurate predictions or preform a specific task
		- Weights
		- Baises

**Tokens**

![[Pasted image 20260516111946.png]]

- Token
	- A chuck of text that a model processes as a single unit
	- Subword-level tokenization

## Introducing LangChain

- LangChain components
	- Prompt templates
	- LLM
	- Agents
	- Memory

### Installing LangChain

```python
!pip install langchain
```
### Creating a prompt template

- Prompt template
	- Structures the instruction or query given to the model to obtain the desired outputs
	- String template that accepts a list of parameters from users to be used to generate a prompt for an LLM

```python
from langchain import PromptTemplate

template = '''
Question: {question}
Answer:
'''

prompt = PromptTemplate(
    template = template,
    input_variables = ['question']
)
prompt

# output
PromptTemplate(input_variables=['question'],
               template='\nQuestion: {question}\nAnswer: ')

```
### Specifying an LLM

- Create a `read` token

```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'Your_HuggingFace_Token'

!pip install langchain-huggingface

from langchain_huggingface import HuggingFaceEndpoint

hub_llm = HuggingFaceEndpoint(
	    endpoint_url="https://api-inference.huggingface.co/models/
		HuggingFaceH4/zephyr-7b-alpha",
	    temperature = 1
)
```

- Direct Preference Optimization (DPO)

### Creating an LLM chain

- Combine prompt template and model to create a chain

```python
from langchain_core.output.parsers import StrOutputParser

llm_chain = prompt | hub_llm | StrOutputParser()
```

- Pipeline style chaining
### Running the chain

- Inferencing is done on Hugging Face's server

```python
qn = "What is an apple"
print(llm_chain.invoke(qn))

qn = "What types are there?"
print(llm_chain.invoke(qn))
```
### Maintaining a conversation

- LLM need the response question and answers
- Provide a history of the conversion to the prompt

```python
template = '''
Current conversation: {history}
Human: {question}
AI:
'''

prompt = PromptTemplate(
    template = template,
    input_variables = ['question','history']
)

llm_chain = prompt | hub_llm | StrOutputParser()

# continuous response
history = ''
while True:
    qn = input('Question: ')
    if qn == 'quit':
        break    
    response = llm_chain.invoke({'question':qn, 'history':history})
    history = response
    print(history)
```

### Using the RunnableWithMessageHistory Class

- Modify the prompt to maintain a conversation with the LLM
- Manage the history of interactions between user and model

```python
import os
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables.history import RunnableWithMessageHistory

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'your_hugging_face_token'

template = '''
Question: {question}
Answer:
'''

prompt = PromptTemplate(
    template = template,
    input_variables = ['question']
)

hub_llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha",
    temperature = 1
)

class SessionHistory:  #1
    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)  #2

    def get_messages(self):
        return self.messages

session_history = SessionHistory()  #3

def get_session_history():  #4
    return session_history

llm_chain = RunnableWithMessageHistory(  #5
    prompt | hub_llm | StrOutputParser(),
    get_session_history = get_session_history
)

while True:  #6

    user_question = input("Ask a question (type 'exit' to stop): ")#7


    if user_question.lower() == "quit":#8
        print("Ending conversation.")
        break


    input_data = {"question": user_question}#9

    response = llm_chain.invoke(input_data)  #10

    session_history.add_messages([  #11
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": response}
    ])

    #L display the response
    print(f"AI: {response}")
```

- Text completion

```python
import os
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'Your_HuggingFace_Token'

template = '''
Complete this: {question}
'''

prompt = PromptTemplate(
    template = template,
    input_variables = ['question']
)
prompt

hub_llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/
models/HuggingFaceH4/zephyr-7b-alpha",
    temperature = 1
)

llm_chain = prompt | hub_llm | StrOutputParser()

while True:
    qn = input('Question: ')
    if qn == 'quit':
        break    
    response = llm_chain.invoke(qn)
    print(response)
```

## Connecting LLMs to Your Private Data

- LlamaIndex
	- Adds RAG (retrieval augmented generation)

### Installing the packages

```python
!pip install llama_index
!pip install llama-index-embeddings-huggingface
!pip install llama-index-llms-huggingface
```
### Preparing the documents

### Loading the documents

- `SimpleDirectoryReaderClass`
	- Component that facilitates reading and indexing documents from a directory

```python
from llama_index.core import SimpleDirectoryReader

loader = loader = SimpleDirectoryReader(
    input_dir="./Training Documents",
    recursive=True,
    required_exts=[".pdf"],
)

documents = loader.load_data()
```
- Directory input
- Recursive loading
- file type filtering
### Using an embedding model

- Vector embedding
	- Numerical representation of objects

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

- Bag of Graph Embeddings (BGE)
	- Generate embeddings for English text

### Indexing the documents

- `VectorStoreIndex`
	- Creates an index an saves the vector embeddings on disk

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    embed_model = embedding_model,
)

index.storage_context.persist(persist_dir=".")
```
5 Files
- `image__vector_store.json`
- `default__vector_store.json`
- `graph_store.json`
- `index_store.json`
- `docstore.json`

### Loading the embeddings

- `StorageContent`

```python
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir=".")
index = load_index_from_storage(storage_context,
                                embed_model = embedding_model)
```
### Using an LLM for querying

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

if torch.backends.mps.is_available():#1
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(#2
    "meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct").to(device)


    huggingface_llm = HuggingFaceLLM(#3
        model=model,
        tokenizer=tokenizer,
    )

    query_engine = index.as_query_engine(llm=huggingface_llm)#4
```

**Using the GPU**

- `to()`
	- Method moves a model or tensor to a specific device
	- 

### Asking questions

```python
while True:
    question = input("Question: ")
    if question.lower() == "quit": break
    print(query_engine.query(question).response)
```

### Using LlamaIndex with OpenAI

- Run a local LLM using OpenAI model

```python
!pip install langchain_community
!pip install langchain_openai

from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "OpenAI_API_Key"
openai_llm = ChatOpenAI(temperature = 0.7,
                        model_name = "gpt-4o-mini")

query_engine = index.as_query_engine(llm = openai_llm)

while True:
    question = input("Question: ")
    if question.lower() == "quit": break
    print(query_engine.query(question).response)
```
### Creating a web frontend for the app

```python
!pip install gradio

def my_chat_bot(input_text):
    response = query_engine.query(input_text)
    return response.response
    
import gradio as gr


gr.Interface(fn = my_chat_bot,#1
             title = "Enquiry",
             inputs = "text",
             outputs = "text").launch()
```
### Holding a conversion

- `as_chat_engine()`

```python
query_engine = index.as_chat_engine(llm=openai_llm)

def my_chat_bot(input_text):
    response = query_engine.chat(input_text)  #1
    return response.response

import gradio as gr

gr.Interface(fn = my_chat_bot,  #2
             title = "Enquiry",
             inputs = "text",
             outputs = "text").launch()
```

### Creating a chatbot UI

```python
import gradio as gr

with gr.Blocks() as mychatbot:
    chatbot = gr.Chatbot()  #1
    question = gr.Textbox()  #2

    def chat(message, chat_history):
        content = "Responses from chatbot..."  #3
        chat_history.append((message, content))
        return "", chat_history

    question.submit(fn = chat,  #4
                    inputs = [question, chatbot],
                    outputs = [question, chatbot])

mychatbot.launch()
```
- `Blocks`
	- Low-level API for custom web application

```python
import gradio as gr

with gr.Blocks() as mychatbot:
    chatbot = gr.Chatbot()  #1
    question = gr.Textbox()  #2

    def chat(message, chat_history):
        content = my_chat_bot(message)
        chat_history.append((message, content))
        return "", chat_history

--LB_EMPTY_LINE--
    question.submit(fn = chat,#3
                    inputs = [question, chatbot],
                    outputs = [question, chatbot])

mychatbot.launch()
```

![[Pasted image 20260517220446.png]]