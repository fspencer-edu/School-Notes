## Introducing GPT4All

- Open source project containing several pretrained LLMs
- Run locally using consumer CPUs
- 3 GB to 8 GB
## Installing GPT4All

- End user application
- Python library

### Installing the GPT4All application

- Most models require 4-16 GB of RAM

### Installing the GPT4All Python Library

```python
!pip install gpt4all
```
### Listing all supported models

```python
from gpt4all import GPT4All
GPT4All.list_models()

models = GPT4All.list_models()
[{model['name']:model['filename']} for model in models]
```
### Loading a specific model

```python
gpt = GPT4All("mistral-7b-openorca.Q4_0.gguf")
print(gpt.config)
```
### Asking a question

- `chat_session()`
	- Creates a contextual manager in which you can hold an inference-optimized chat session with a model
- `generate()`
	- To ask question

```python
with gpt.chat_session():
    output = gpt.generate("What is the population of Japan?",
                 max_tokens=2048)
    print(output)
    print(gpt.current_chat_session)
    
with gpt.chat_session():
    response1 = gpt.generate(
        prompt='What is the population of Singapore?',
        temp = 0)
    print(response1)
    print(gpt.current_chat_session)  
    print('===')

    response2 = gpt.generate(
        prompt='Where is it located?',
        temp = 0)
    print(response2)
    print(gpt.current_chat_session)
```

- Use `with` to ask a follow up question
- Save context using `gpt.current_chat_session`

```python
session = []

with gpt.chat_session():
    response1 = gpt.generate(prompt='What is the population of Singapore?',
                             temp = 0)
    print(response1)

    session = gpt.current_chat_session#1

with gpt.chat_session():  

    gpt.current_chat_session = session#2
    response2 = gpt.generate(prompt='Where is it located?', temp = 0)
    print(response2)
```
### Binding with Gradio

```python
from gpt4all import GPT4All

gpt = GPT4All("mistral-7b-openorca.Q4_0.gguf")

def chat(message):
    with gpt.chat_session():
        return gpt.generate(prompt = message,
                            temp = 0)
                            
import gradio as gr

gr.Interface(fn = chat,#1
             inputs = "text",
             outputs = "text").launch()
```

<img src="/images/Pasted image 20260518145314.png" alt="image" width="500">

- Save the current chat session's details in a global variable and set it back every time you ask a follow-up question

```python
import gradio as gr
from gpt4all import GPT4All

gpt = GPT4All("mistral-7b-openorca.Q4_0.gguf")
current_chat_session = []

def chat(message):
    with gpt.chat_session():
        global current_chat_session
        gpt.current_chat_session = current_chat_session
        response = gpt.generate(prompt = message,
                                temp = 0)

        current_chat_session = gpt.current_chat_session
        return response

# bind it to gradio
gr.Interface(fn = chat,
             inputs = "text",
             outputs = "text").launch()
```

- Chatbot UI

```python
import gradio as gr
from gpt4all import GPT4All

gpt = GPT4All("mistral-7b-openorca.Q4_0.gguf")
current_chat_session = []

with gr.Blocks() as mychatbot:  #1
  #2
    chatbot = gr.Chatbot()  #3
    question = gr.Textbox()  #4
    clear = gr.Button("Clear Conversation")  #5


    def clear_messages():#6
        global current_chat_session
        current_chat_session = []  #7


    def chat(message, chat_history):#8
        with gpt.chat_session():
            global current_chat_session
            gpt.current_chat_session = current_chat_session

            response = gpt.generate(prompt = message,
                                    temp = 0)
            current_chat_session = gpt.current_chat_session
            
			chat_history.append((message, response))#9


            return "", chat_history#10


    question.submit(fn = chat,#11
                    inputs = [question, chatbot],
                    outputs = [question, chatbot])


    clear.click(fn = clear_messages,#12
                inputs = None,
                outputs = chatbot,
                queue = False)

mychatbot.launch()
```

