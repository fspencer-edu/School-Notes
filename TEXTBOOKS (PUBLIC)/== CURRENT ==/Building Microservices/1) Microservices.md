- Domain Driven design
	- System models
- Continuous delivery
	- Production
- Hexagonal architecture
- Virtualization platforms
- Infrastructure automation

# What Are Microservices?

- Small, autonomous services that work together

## Small, and Focused on Doing One Thing Well

 - Modular monolithic codebases
- Cohesion
	- Related code group together
- Organizational alignment

## Autonomous

- All communication between services are via network calls
- Service exposes an API
- Collaborating services communicate with the APIs

# Key Benefits

## Technology Heterogeneity

- Use different technologies for each service
- Different programming languages

## Resilience

- If one component of a system fails, the failure does not cascade
- Service boundaries

## Scaling

- Scale individual services

## Ease of Deployment

- Change a single service a and deploy it independently of the system

## Organizational Alignment

- Align architecture to organizational structure
- Shift ownership of services
## Composability

- Systems are able to be reused

## Optimizing for Replaceability

- Individual services can be replaced, and are easier to manage

# What About Service-Oriented Architecture?

- Service-oriented architecture (SOA)
	- Design approach where multiple services collaborate to provide and end set of capabilities
	- Separate OS
	- Communication via network calls
- Issues with SOA
	- Communication protocols (SOAP)
	- Vendor middleware
	- Lack of guidance about service granularity

# Other Decompositional Techniques

## Shared Libraries

- Break down a codebase into multiple libraries
- Libraries are a way to share functionality between teams and services
	- Loss heterogeneity
	- Cannot deploy new library without deploying entire process

## Modules

- Module decomposition techniques
	- Lifecycle management of modules
	- Deployed into a running process
- Open Source Gateway Initiative (OSGI)
	- Allow plug-ins to be installed in the Eclipse Java IDE
	- Retrofit a module concept in Java via a library

- Erlang
	- Modules are baked into the language at runtime

 - Process boundary separation does enforce clean hygiene

# No Silver Buller

- Distributed transactions
- CAP theorem

# Summary