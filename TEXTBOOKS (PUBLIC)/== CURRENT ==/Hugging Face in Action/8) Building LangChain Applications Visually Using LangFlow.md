## What is LangFlow?

- LangFlow
	- Open source library that allows you to build LLM based apps using LangChan through a drag and drop visual interface
	- Built on top of LangChain

### Installing LangFlow

```python
!pip install langflow

$ python -m langflow run
```

- Runs at `http://127.0.0.1:7860 `

### Installing LangFlow using Docker

```python
FROM langflowai/langflow:latest

docker build -t langflow .

docker run -p 7860:7860 langflow
```
### Running LangFlow in the cloud

- Run LangFlow on HF spaces
## Creating a new LangFlow Project

- Components (known as flows)
	- Building blocks of a LangFlow project
	- Prompt templates
	- LLMs
	- Agents
	- Memory

### Adding a Prompt component

- `Prompt`

```python
Human: {question}
AI:
```
### Adding a Models component

- `Models`
	- Use model from HF
	- Add API token
	- Repo ID
### Adding a Chains Component

- `Chains`
	- `ConversationChain`
- Connect the `Prompt` and `HuggingFace` components to the `ConversationChain`

![[Pasted image 20260517221110.png]]
### Add Chat Input and Chat Output Components

![[Pasted image 20260517221132.png]]
### Testing the project
### Maintaining a conversation using the Chat Memory component

- Supply the `Prompt` component with memories

```python
{history}
User: {question}
AI:
```

![[Pasted image 20260517221241.png]]
 

## Asking Questions on your own Data

**Other Components**
- File
- Parse Data
- HuggingFace
- OpenAI
- Promt
- Chat input
- Chat output

### Loading PDF documents using the File component

- `File`
### Splitting long test into small chucks using the Parse Data Component

- `Parse Data`
	- Extract and structure relevant information from raw text or documents before processing them further in the pipeline

### Getting questions using the Prompt component

```python
Answer user's questions based on the document below:

---

{Document}

---

Question:
{Question}

Answer:
```

![[Pasted image 20260517221502.png]]
### HuggingFace Component

![[Pasted image 20260517221526.png]]


## Using Your Project Programmatically

![[Pasted image 20260517221608.png]]

- Connections
	- Curl
	- Python API
	- JS API
	- Python
	- Chat widget HTML
	- Tweaks

### cURL

- Command line tool and library for transferring data with URLs

```python
curl -X POST \
    "http://127.0.0.1:7861/api/v1/run/a195037a-1cac-
4f9d-9737-4613107b0374?stream=false" \
    -H 'Content-Type: application/json'\
    -d '{"input_value": "What did I buy",
    "output_type": "chat",
    "input_type": "chat",
    "tweaks": {
  "File-US1KX": {},
  "ParseData-ljEsG": {},
  "ChatInput-rAmmk": {},
  "Prompt-teE98": {},
  "HuggingFaceModel-gSuPr": {},
  "ChatOutput-mO1kr": {},
  "OpenAIModel-xs0J4": { "openai_api_key": "OpenAI API Key" }
}}'
```
### Python code

- Download project as a JSON file
- `load_flow_from_json()`
	- Run it programmatically without having the LangFlow project running

```python
from langflow.load import run_flow_from_json
TWEAKS = {
  "File-US1KX": {},
  "ParseData-ljEsG": {},
  "ChatInput-rAmmk": {},
  "Prompt-teE98": {},
  "HuggingFaceModel-gSuPr": {},
  "ChatOutput-mO1kr": {},
  "OpenAIModel-xs0J4": { "openai_api_key": "OPENAI API Key" }
}

result = run_flow_from_json(flow="Querying a local document.json",
                            input_,
                            fallback_to_env_vars=True,
                            tweaks=TWEAKS)

print(result)
```
