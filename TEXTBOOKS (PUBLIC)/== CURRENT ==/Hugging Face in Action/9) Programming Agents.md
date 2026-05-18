
- Agents are large language models (LLMs)
	- Perform tasks
	- Plan
	- delegate
	- Break complex problems into smaller subtasks
- Smolagents
	- A lightweight, minimalistic agent framework for quick experimentation
- LangGraph
	- Multistep workflows

## What are Agents?

- Understands natural language
- Reasons and plans
- Acts using known tools
- Delivers results

## Developing Agents Using Smolagents

- An agent is a system that combine a language model with tools to perform tasks by generating and executing Python code

```python
pip install smolagents
```

### Using built-in tools: DuckDuckGoSearchTool

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, 
  HfApiModel
model = HfApiModel()                                                 #1
agent = CodeAgent(tools = [DuckDuckGoSearchTool()], model = model)   #2
response = agent.run("How long does it take to travel from " +
                     "New York to Los Angeles by train?")            #3
print(response)  
```

- `CodeAgent`
	- Build an agent that can reason through problems by generating and executing Python code
- `ollama pull qwen2:7b`
- Ollama
	- Open source platform that enables you to run LLMs directly on a local machine

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

model = LiteLLMModel(
    model_id = "ollama/qwen2:7b",
    api_base = "http://127.0.0.1:11434",
    num_ctx = 8192,
)
agent = CodeAgent(tools = [DuckDuckGoSearchTool()], model = model)
response = agent.run("How long does it take to travel from " +
                     " New York to Los Angeles by train?")       
print(response)
```

### Using built-in tools: PythonInterpreterTool

- Allows the agent to execute Python code dynamically

```python
from smolagents import CodeAgent, PythonInterpreterTool, LiteLLMModel

model = LiteLLMModel(
    model_id="gpt-4o-mini",        
    api_base="https://api.openai.com/v1",
)

agent = CodeAgent(tools=[PythonInterpreterTool()], model=model)
response = agent.run("Calculate the 10th Fibonacci number.")
print(response)
```
### Writing custom tools

- `@tool`

```python
from smolagents import CodeAgent, LiteLLMModel, tool
import requests

@tool
def get_weather_info(city: str) -> str:
    """Retrieve the current weather information for a given city.
    Args:
        city: The name of the city to get the weather information for.
    Returns:
        str: A description of the current weather and temperature
        in the city.
    """
    api_key = "<API_KEY>"                                              #1
url = f"http://api.openweathermap.org/data/2.5/weather?
        q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        return f"The weather in {city} is {weather} with a 
                 temperature of {temperature}°C."
    else:
        return f"Could not retrieve weather information for {city}."

model = LiteLLMModel(
    model_id="ollama/qwen2:7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(tools=[get_weather_info], model=model)
response = agent.run("What is the current weather for Singapore?")
print(response)
```

- Tool function must include a docstring describing its parameters

## Developing Agents with LangChain

- LangChain
	- Flexible modular architecture that allows developers to create complex agents by composing components
		- Prompts
		- Memory
		- Tools
		- Chains

```python
!pip install langchain langchain-openai
    langchain-community google-search-results
```

### Using the built-in Tool class

- Agents interact with external functionality through a standardize tool interface
	- `BaseTool`
	- `SerpAPIWrapper`
		- Real time search API that allows developers to programmatically access and extract search results from search engines
- Agents cannot directly use utility wrappers
- A wrapper must be embedded in a `Tool` instance

```python
import os
os.environ["SERPAPI_API_KEY"] = "<SERPAPI_KEY>"

from langchain.tools import Tool
from langchain_community.utilities.serpapi import SerpAPIWrapper

search = SerpAPIWrapper()   #1
tools = [                   #2
    Tool(
        name = "Search",
        func = search.run,
        description = "Useful for when you need to answer questions 
        about current events or search for specific information on 
        the web. Input should be a search query."
    )
]
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"

llm = ChatOpenAI(model = "gpt-4o-mini",          #1
                 temperature = 0)
agent = initialize_agent(                        #2
    tools = tools,
    llm = llm,
agent_type = 
        AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True                               #3
)

response = agent.invoke("Who is Wei-Meng Lee?")  #4
print(response)
```
### 
### 
### 
## Developing Agents using LangGraph
