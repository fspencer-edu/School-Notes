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

![[Pasted image 20260518220244.png]]

- Translator
	- Handles communication by converting each provider's unique languages to a common one

![[Pasted image 20260518220317.png]]

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

![[Pasted image 20260518220450.png]]

### MCP server deployment

- MCP server is simply a standard application that can run in different configurations
	- Locally
	- Remotely

![[Pasted image 20260518220554.png]]

![[Pasted image 20260518220607.png]]

**Server-sent Events**

- 

### 
### 
## Building a MCP Server
## Testing the MCP server using Claude Desktop
## Trying third party MCP servers
