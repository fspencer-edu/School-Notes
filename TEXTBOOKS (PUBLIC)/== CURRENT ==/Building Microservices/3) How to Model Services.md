
# Introduction MusicCorp

- MusicCorp
	- Recently a brick-and-mortar retailer
	- Now focuses on online
	- Shipping CDs

# What Makes a Good Service?

- Loose coupling
- High cohesion

## Loose Coupling

- A change to one service should not require a change to another
- Knows as little as it needs to about the service with which it collaborates
## High Cohesion

- Related behaviour sits together
- Unrelated behaviour sits elsewhere
- Find boundaries within problem domain that help ensure that related behaviour is in one place, and that communication with other boundaries is as loose as possible

# The Bounded Context

- Domain driven design
	- Create systems that model real world domains
	- Ubiquitous language
	- Repository abstractions
	- Bounded context
		- Any given domain consists of multiple bounded contexts
		- Within each context is a model that does not need to be communicated outside
		- Models that are shared externally with other bounded contexts

- MusicCorp
	- Managing orders being shipped out
	- Taking delivery of new stock
	- Truck

## Shared and Hidden Models

 - Finance and warehouse departments are two separate bounded contexts
 - Both have an explicit interface to the outside world
 - Details that only they need to know

<img src="/images/Pasted image 20260424135104.png" alt="image" width="500">

- Stock item becomes a shared module between the two contexts

## Modules and Services

- Modular boundaries become excellent candidates for microservices

## Premature Decomposition
 
- Prematurely decomposing a system into microservices can be costly

# Business Capabilities

- Contexts have the capability to become key operations that will be exposed over the wire to other collaborators

# Turtles All the Way Down

- Think of coarser-grained contexts, then subdivide along nested contexts
- Microservices
	- Hidden inside another boundary
	- Popped up into top-level contexts

# Communication in Terms of Business Concepts

- Terms and ideas that are shared between parts of the organization should be reflected in the interfaces
- Forms being sent between microservices, are forms that are sent around an orgnaization

# Techniques Boundary

- Onion architecture
	- RPC-batching mechanisms

# Summary