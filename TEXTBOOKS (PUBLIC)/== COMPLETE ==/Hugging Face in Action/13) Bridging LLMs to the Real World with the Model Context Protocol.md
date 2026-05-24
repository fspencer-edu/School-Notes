- Model Context Protocol (MCP)
	- Provides a standard way for LLMs to access and use outside data

## What is MCP?

- MCP
	- Open standard created by Anthropic
	- JSON-RPC 2.0
	- Connects LLMs to external services

### The problems MCP solves

- Inconsistent tool access
- Unreliable data retrieval
- Complex prompt engineering
- Fragmented integrations

- Components
	- Tools
	- Resources
	- Prompts

### Understanding MCP

<img src="/images/Pasted image 20260518220244.png" alt="image" width="500">

- Translator
	- Handles communication by converting each provider's unique languages to a common one

<img src="/images/Pasted image 20260518220317.png" alt="image" width="500">

- MCP workflow
	- MCP client
		- LLM
	- MCP server
	- Service
		- Features of data the client wants to use

- Consistent services
	- Database
	- Files
	- Images

<img src="/images/Pasted image 20260518220450.png" alt="image" width="500">

### MCP server deployment

- MCP server is simply a standard application that can run in different configurations
	- Locally
	- Remotely

<img src="/images/Pasted image 20260518220554.png" alt="image" width="500">

<img src="/images/Pasted image 20260518220607.png" alt="image" width="500">

**Server-sent Events**

- Server sent events (SSE)
	- Web standard that enables a server to push real-time data to a client over a single HTTP connectionPersistent connection
	- SSE are similar to WebSockets, although unidirectional
	- Only server can send data to the client

### Components in an MCP server

- Components
	- `Tools`
		- Actionable capabilities that MCP servers provide LLMs
		- Specialized functions
	- `Resources`
		- External data or entities
	- `Prompts`
		- MCP system function as predefined templates for standardized LLM interactions

<img src="/images/Pasted image 20260518220852.png" alt="image" width="500">

## Building a MCP Server

### Installing uv

- `uv`
	- Fast Python package and project manager written in Rust, to install Python packages

```python
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Initializing the project

```python
$ uv init MCP_Demo
Initialized project `mcp-demo` at `/Volumes/SSD/MCP_Demo`

$ cd MCP_Demo
```
### Installing the packages

- Use the official Python SDK for MCP servers and clients

```python
$ uv add "mcp[cli]" httpx PyMuPDF
```
### Creating the MCP server

```python
$ nano server.py

from mcp.server.fastmcp import FastMCP
import httpx
import fitz                   #1
import os

mcp = FastMCP("MCP Demo")     #2

# ==============================================
# resources, tools, and prompts to be added here
#
#       <to be added in next few sections>
#
# ==============================================

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    
$ uv run server.py
```
- Create an instance of the `FastMCP` class
- Start the `FastMCP` server with the transport parameter set to `stdio`

### Inspecting the MCP server

```python
$ uv run mcp dev server.py
```

<img src="/images/Pasted image 20260518221738.png" alt="image" width="500">


### Implementing Resources

```python
# server.py

# ==============================================
# resources, tools, and prompts to be added here

#==========
# Resources
#==========
@mcp.resource("text://{file_path}")  
def get_file(file_path: str) -> str:
    actual_path = os.path.abspath(file_path)                           #1
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Error: File '{actual_path}' not found!")  
    with open(actual_path, "r", encoding="utf-8") as file:
        return file.read()

@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "Version 1.1"
    
@mcp.resource("pdf://{file_path}")
def get_pdf_data(file_path: str) -> str:  
    text = ""
    actual_path = os.path.abspath(file_path)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Error: File '{actual_path}' not found!")
    with fitz.open(actual_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text
```

- `Resources`
	- `get_file()`
		- Retrieves the contents of a text file given its file path
	- `get_config()`
		- Returns static application configuration data
	- `get_pdf_data()`
		- Extracts and returns the text content from a PDF file given its file path

### Implementing Tools

```python
#======
# Tools
#======

@mcp.tool()
async def fetch_weather(city: str, units: str = "metric") -> dict:  
    API_KEY = "xxxxxxxxxxxxxxxxxx"                        #1
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "units": units,
                "appid": API_KEY
            }
        )      
        
        
@mcp.tool()
def get_pdf(file_path: str) -> str:  
    return get_pdf_data(file_path)

@mcp.tool()
def get_text(file_path: str) -> str:  
    return get_file(file_path)
```
### Implementing a prompt

```python
#=======
# Prompt
#=======
# Add a weather_report prompt template
@mcp.prompt()
def weather_report(city: str) -> str:
    return f"""
    Please provide a weather report for {city}.

    You can use the fetch_weather tool to get current weather data.
    If needed, you can convert temperature units using the
    convert_temperature tool.

    Please include:
    - Current temperature
    - Weather conditions
    - Humidity
    - Wind speed
    - Any relevant weather advice for the conditions
    """
```
### Testing the components

- `@mcp.prompt`
	- Registers a function as an MCP prompt

```ython
$ uv run mcp dev server.py
```
## Testing the MCP server using Claude Desktop

- Claude Desktop
	- Integrates MCP to enhance AI-driven tasks and workflows

### Configuring Claude Desktop to use the MCP server

```python
$ nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

{
  "mcpServers": {
    "weather": {
      "command": "/Users/weimenglee/.local/bin/uv",
      "args": [
        "--directory",
        "/Volumes/SSD/MCP_Demo",
        "run",
        "server.py"
      ]
    }
  }
}
```
### Improving the MCP server

```python
from mcp.server.fastmcp import FastMCP
import httpx
import fitz  # for PyMuPDF
import os

import sys

API_KEY = os.getenv('OPENWEATHER_API_KEY')              #1
if not API_KEY:
    print("Error: OPENWEATHER_API_KEY environment variable must be set",
       file=sys.stderr)
    sys.exit(1)

# Create an MCP server
mcp = FastMCP("MCP Demo")
...
...
#======
# Tools
#======
@mcp.tool()
async def fetch_weather(city: str, units: str = "metric") -> dict:
    async with httpx.AsyncClient() as client:
        # API_KEY = "xxxxxxxxxxxx"
        # Using OpenWeatherMap API
        response = await client.get(
...
...
```

## Trying third party MCP servers

- Third party MCP servers
	- Location service
	- Time service

## Get My Location

```python
$ nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

{
  "mcpServers": {
    "weather": {
      "command": "/Users/weimenglee/.local/bin/uv",
      "args": [
        "--directory",
        "/Volumes/SSD/MCP_Demo",
        "run",
        "server.py"
      ],
      "env": {
        "OPENWEATHER_API_KEY": "xxxxxxxxxxxx"
      }
    },
    "get-location": {
      "command": "npx",
      "args": [
        "-y",
        "@mcpcn/mcp-get-location"
      ],
      "env": {}
    }
  }
}
```

- Location Service server is written in Node.js

### mcp-datetime

- Obtain the current time in geographical locations

```python
$ nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

{
  "mcpServers": {
    "weather": {
      "command": "/Users/weimenglee/.local/bin/uv",
      "args": [
        "--directory",
        "/Volumes/SSD/Dropbox/MCP_Demo",
        "run",
        "server.py"
      ],
      "env": {
        "OPENWEATHER_API_KEY": "xxxxxxxxxxxx"
      }
    },
    "get-location": {
      "command": "npx",
      "args": [
        "-y",
        "@mcpcn/mcp-get-location"
      ],
      "env": {}
    },
    "mcp-datetime": {
      "command": "/Users/weimenglee/.local/bin/uvx",
      "args": ["mcp-datetime"]
    }
  }
}
```
**UV and UVX**

- `uv`
	- Python package and project manager
	- Handles dependencies an virtual environments
- `uvx`
	- Executes Python applications in isolated, temporary environments
	- 

### 
### 