
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

- 
### 
### 
### 

## Connecting LLMs to Your Private Data