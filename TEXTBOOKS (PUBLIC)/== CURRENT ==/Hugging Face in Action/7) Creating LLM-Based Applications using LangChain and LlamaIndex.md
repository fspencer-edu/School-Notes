
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