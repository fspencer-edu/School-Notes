
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

**Built In Tools**

```python
from langchain_community.utilities.bing_search
  import BingSearchAPIWrapper                            #1
from langchain_community.utilities.duckduckgo_search
  import DuckDuckGoSearchAPIWrapper                      #2
from langchain_community.utilities.google_search
  import GoogleSearchAPIWrapper                         #3
from langchain_community.utilities.wikipedia
  import WikipediaAPIWrapper     
```

**Fetch Weather Information**

```python
import os
import requests
from langchain_openai import ChatOpenAI
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.tools import Tool, tool
from langchain.agents import AgentType, initialize_agent

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"
os.environ["SERPAPI_API_KEY"] = "<SERPAPI_KEY>"

llm = ChatOpenAI(temperature=0)

search = SerpAPIWrapper()
search_tool = Tool(
    name = "Search",
    func = search.run,
description = "Useful for when you need to answer questions 
about current events or search for specific information on 
    the web. Input should be a search query."
)

@tool                                                                   #1
def get_weather_info(city: str) -> str:
    """Retrieve the current weather information for a given city.
    Args:
        city: The name of the city to get the weather information for.
    Returns:
        str: A description of the current weather and temperature in
        the city.
    """

    api_key = "<OPENWEATHERMAP_API_KEY>"                                #2
url = f"http://api.openweathermap.org/data/2.5/
        weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        summary = (
            f"Weather in {city}:\n"
            f"Condition: {weather}\n"
            f"Temperature: {temperature}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s"
        )     
        return summary                                         #3
    else:
        return f"Could not retrieve weather information for {city}."
tools = [search_tool, get_weather_info]                        #4
agent = initialize_agent(                                      #5
    tools = tools,
    llm = llm, 
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)
```

## Developing Agents using LangGraph

- LangGraph
	- More flexible and feature rich framework
	- Building complex stateful agents
	- Creating an agent capable of answering user questions using reasoning
	- Integrating an external tool to enable the agent to answer questions
	- Integrating memory

### What is LangGraph

- Python framework developed by LangGain
- Structure logic as a directed graph

- Applications
	- Multi-turn chatbots with memory
	- Decision trees or branching logic
	- Complex tool-using agents
	- Data enrichment or extract, transform, load (ETL) pipelines
	- Modular conversational flows

```python
!pip install langgraph
```

### LangGraph agent basics

- External tools
	- Web search APIs
	- Database connectors

```python
import os
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"      
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) 
tools = []  

agent_executor = create_react_agent(llm, tools)
```

<img src="/images/Pasted image 20260518142301.png" alt="image" width="500">

- ReAct
	- Method in which the agent thinks step by step

### Using LangGraph with tools

- Connect the agent to an external tools

```python
from langchain_core.tools import Tool
from langchain_community.utilities import SerpAPIWrapper

os.environ["SERPAPI_API_KEY"] = "<SERPAPI_KEY>"          #1
serpapi = SerpAPIWrapper()                               #2
search_tool = Tool(
    name = "SerpAPI",                                    #3
    func = serpapi.run,                                  #4
description = "A search engine tool to query real-time information
                   from the web."
)
tools = [search_tool]                            
agent_executor = create_react_agent(llm, tools)
```

<img src="/images/Pasted image 20260518142500.png" alt="image" width="500">

### Using LangGraph with a custom tools

```python
import requests

def get_weather_info(city: str) -> str:
    """Retrieve the current weather information for a given city."""
    api_key = "7453d5cfeaea020958539f22da95d849"           #1
url = f"http://api.openweathermap.org/data/2.5/
        weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url) 
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        summary = (
            f"Weather in {city}:\n"
            f"Condition: {weather}\n"
            f"Temperature: {temperature}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s"
        )     
        return summary                                      #2
    else:
        return f"Could not retrieve weather information for {city}."
        
weather_tool = Tool(                                        #1
    name = "GetWeather",                                    #2
    func = get_weather_info,                                #3
    description = "A tool to fetch the weather information for a city"
)

tools = [search_tool, weather_tool]
agent_executor = create_react_agent(llm, tools)
```

### Using LangGraph with memory

- Message passing state

```python
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List, add_messages]
    
def run_agent(query: str, state: State = None) -> tuple[str, State]:  #1
    response = agent_executor.invoke({"messages": [("user", query)]})
    pprint(response)
    if state is None:
        state = {"messages": []}
    state["messages"].append(("user", query))                          #2
    response = agent_executor.invoke(state)                            #3
    state = {"messages": response["messages"]}                         #4
    return response["messages"][-1].content, state  
    
conversation_state = {"messages": []}                                 #1
while True: 
    query = input("Question: ")
    if query.lower()=="quit": break
    answer, conversation_state = run_agent(query, conversation_state)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
```