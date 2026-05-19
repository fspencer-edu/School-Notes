- Gradio
	- Open source Python package
	- Build a demo web application

## Basics of Gradio

```python
!pip install gradio
```

- `Interface`
	- Build a simple Gradio application
- Flag options
- Authentication
- Accessible on the local network
- Deploy on HF spaces

### Using Gradio's Interface class

- `Interface`
	- High level class
	- `fn`
	- `title`
	- `inputs`
	- `outputs`

```python
import gradio as gr

def my_chatbot(message):
    return "Hello, " + message

interface = gr.Interface(fn = my_chatbot,  #1
                         title = "Hello, Gradio!",  #2
                         inputs = "text",  #3
                         outputs = "text")  #4

interface.launch()
```

<img src="/images/Pasted image 20260518143038.png" alt="image" width="500">

- Flag buttons are displayed by default
- Port number increments by 1 every time you run the cell in Jupyter Notebook
- Flag button creates a new folder called `.gradio/flagged`
	- Saves files in a csv format
### Configuring flagging options

- Default behaviour
	- Flag button
		- Log inputs, outputs, and timestamps

```python
import gradio as gr

def my_chatbot(message):
    return "Hello, " + message

interface = gr.Interface(fn = my_chatbot,
                         title = "Hello, Gradio!",
                         inputs = "text",
                         outputs = "text",
                         flagging_options =
                             ["correct","wrong","ambiguous"])

interface.launch()
```

<img src="/images/Pasted image 20260518143326.png" alt="image" width="500">

### Configuring authentication

- By default Gradio application is public
- Restrict access using `auth()` in the `launch()` method

```python
import gradio as gr

def my_chatbot(message):
    return "Hello, " + message


interface = gr.Interface(fn = my_chatbot,#1
                         title = "Hello, Gradio!",
                         inputs = "text",
                         outputs = "text",
                         flagging_options =
                             ["correct","wrong","ambiguous"])

interface.launch(auth = ("admin", "secret"))
```

- Enable third-party cookies in the browser

```python
def authentication(username, password):#1


    return (username=='admin' and password=='secret')#2

interface.launch(auth = authentication)
```

### Customizing the server and port

- By default Gradio listens at port 7860
- Binds to `127.0.0.1` by default
- Bind to `0.0.0.0` for public access

```python
interface.launch(server_name = "127.0.0.1", server_port = 5000)#1

interface.launch(server_name = "0.0.0.0", server_port = 5000)#2
```
### Sharing Gradio application

```python
interface.launch(share = True)
```

- Gradio will create a temporary link that allows your friend to access Gradio application
- Share public URL
	- Shared links expire after 72 hours

### Deploying using Hugging Face Spaces

- Create a directory
- Create a file name
- Host to Hugging Face spaces
	- `WRITE` token
- Launch in terminal
	- `gradio deploy`

## Working with Widgets

- Components
	- `Textbox`
	- `Audio`
	- `Images`
	- `TabbedInterface`

### Working with Textbox

- Customize the text box by creating an instance of the `Textbox` class

```python
import gradio as gr

def my_chatbot(message):
    return "Hello, " + message

textbox = gr.Textbox(label = "Message",
                     placeholder = "Your message here",
                     lines = 3)

gr.Interface(fn = my_chatbot,
             inputs = textbox,
             outputs = "text").launch()
```

<img src="/images/Pasted image 20260518144007.png" alt="image" width="500">

### Working with Audio

- Upload an audio stream

```python
import numpy as np
import gradio as gr

def reverse_audio(audio):
    sr, data = audio  #1
    reversed_audio = (sr, np.flipud(data))  #2
    return reversed_audio

mic = gr.Audio(sources = ["upload","microphone"],
               type = "numpy",
               label = "Audio")

interface = gr.Interface(fn = reverse_audio,
                         inputs = mic,
                         outputs = "audio")
interface.launch()
```

<img src="/images/Pasted image 20260518144052.png" alt="image" width="500">

### Working with Images

```python
# scikit-image
from skimage.color import rgb2gray
import numpy as np
import gradio as gr

def convert_image(img):    
    return rgb2gray(img)  #1

image = gr.Image(type="numpy")  #2

interface = gr.Interface(fn = convert_image,
                         inputs = image,
                         outputs = "image")

interface.launch()

# PIL
from skimage.color import rgb2gray
import numpy as np
import gradio as gr

def convert_image(img):
    return img.rotate(-90)  #1

image = gr.Image(type="pil")  #2

interface = gr.Interface(fn = convert_image,
                         inputs = image,
                         outputs = "image")

interface.launch()

# predetermined images
interface = gr.Interface(fn = convert_image,
                         inputs = image,
                         outputs = "image",
                         examples = [
                             "images/durian.jpg",
                             "images/mango.jpg",
                             "images/rambutan.jpg"]
                        )
interface.launch()
```
### Working with selection widgets

- `Dropdown`
- `Slider`

```python
import numpy as np
import gradio as gr

languages = ['English','Japanese','Chinese']

def translate(language_index, value, sentence):
    return languages[language_index], value, sentence

dropdown = gr.Dropdown(['English','Japanese','French','Chinese'],
                       type = "index",
                       label = "Language",
                       value = "English")

slider = gr.Slider(minimum = 1,
                   maximum = 5,
                   step = 1,
                   value = 2,
                   label = "Select a value")

textbox1 = gr.Textbox(type = "text",
                     value = "",
                     label = "Sentence to translate",
                     placeholder = "sentence")

textbox2 = gr.Textbox(type = "text",
                     label = "Translated sentence")

interface = gr.Interface(
    translate,
    [dropdown, slider,textbox1],
    textbox2,
)

interface.launch()
```

- Wrap the return values using a tuple for multiple outputs

```python
def translate(language_index, value, sentence):
    return (languages[language_index], value, sentence), language_index
...
...

interface = gr.Interface(
    translate,
    [dropdown, slider,textbox1],
    [textbox2,"text"],
)
```
### Layout using the TabbedInterface class

```python
import gradio as gr

def convert_image(img):

    return rgb2gray(img)#1

def reverse_audio(audio):
    sr, data = audio
    print(sr)
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

image = gr.Image(type="numpy")

mic = gr.Audio(sources = ["upload","microphone"],
               type = "numpy",
               label = "Audio")

interface1 = gr.Interface(title = 'Reverse Audio',
                          fn = reverse_audio,
                          inputs = mic,
                          outputs = "audio")

interface2 = gr.Interface(title = 'Convert Image',
                          fn = convert_image,
                          inputs = image,
                          outputs = "image")

tabbed = gr.TabbedInterface(  #2
    [interface1, interface2],
    ['Tab 1','Tab 2']  #3
)

tabbed.launch()
```

<img src="/images/Pasted image 20260518144413.png" alt="image" width="500">

## Creating a chatbot UI

- `Blocks`
	- Low-level API to create mode customized applications
	- Define layouts and events

### Creating the basic chatbot UI

```python
import gradio as gr

with gr.Blocks() as mychatbot:  #1
    chatbot = gr.Chatbot(type = "messages")  #2
    textbox = gr.Textbox()  #3
    clear = gr.Button("Clear Conversation")  #4

mychatbot.launch()
```
### Writing the Textbox's submit event

- Create an event handler for the `sumbit` event

```python
import gradio as gr

with gr.Blocks() as mychatbot:    
    chatbot = gr.Chatbot(type="messages")
    textbox = gr.Textbox()
    clear = gr.Button("Clear Conversation")

    def chat(message, chat_history):
        response = "Responses from chatbot..."  #1
        chat_history.append({"role": "user", "content": message})  #2
        chat_history.append({"role": "assistant", "content": response})   #2
        print(chat_history)
        return "", chat_history  #3

    textbox.submit(fn = chat,  #4
                   inputs = [textbox, chatbot],   #4
                   outputs = [textbox, chatbot])  #4

mychatbot.launch()
```
### Clearing the chatbot

```python
import gradio as gr

with gr.Blocks() as mychatbot:
    chatbot = gr.Chatbot(type="messages")
    textbox = gr.Textbox()
    clear = gr.Button("Clear Conversation")

    def chat(message, chat_history):
        response = "Responses from chatbot..."
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        print(chat_history)
        return "", chat_history  #1

    textbox.submit(fn = chat,
                   inputs = [textbox, chatbot],
                   outputs = [textbox, chatbot])

    def clear_messages():
        print("Clearing message...")

    clear.click(fn = clear_messages,
                inputs = None,
                outputs = chatbot,
                queue = False)

mychatbot.launch()
```
