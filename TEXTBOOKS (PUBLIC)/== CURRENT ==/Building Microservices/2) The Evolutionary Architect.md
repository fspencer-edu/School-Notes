
# Inaccurate Comparisons

- Architects
	- Defining the vision for an entire program or entire organization
	- Impact on the quality of the system build
	- Organization's ability to respond to change

# An Evolutionary Vision for the Architecture

# Zoning

# A Principled Approach

## Strategic Goals

- High level goals
- May not include technology
- Company or division level

## Principles

- Rules made in order to align what you are doing to some large goal
- Heroku's 12 Factors
	- Set of design principles structured around the goal of helping to create application on the Heroku platform
- Constraint
	- Something that is very hard to change

## Practices

- How to ensure principles are being carries out
- Detailed, practical guidance for performing tasks
- Technology-specific
- Low level
- Coding guidelines

## Combining Principles and Practices

## A Real-World Example

![[Pasted image 20260424132622.png]]

# The Required Standard

- Defined "good citizen" service in the system
- Capabilities of a service to ensure that system is manageable
- Defining clear attributes that each service should have

## Monitoring

- Coherent, cross-service views of system health
- System-wide view
- Push mechanisms
- Graphite
	- Metrics
- Health
	- Nagois
- Polling system that scrapes data from nodes

## Interfaces

- Small number of defined interface technologies
- REST protocols
- Pagination of resources
- Versioning of end points

## Architectural Safety

- Each downstream service gets its own connection ppol

# Governance Through Code

## Exemplars

- Written documentation
## Tailored Service Template


# Technical Debt

# Exception Handling

# Governance and Leading from teh Centre

# Building a Team

# Summary