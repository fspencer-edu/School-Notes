
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

![[Pasted image 20260424135104.png]]

- Stock item becomes a shared module between the two contexts
- 

## Modules and Services
## Premature Decomposition
 


# Business Capabilities

# Turtles All the Way Down

# Communication in Terms of Business Concepts

# Techniques Boundary

# Summary