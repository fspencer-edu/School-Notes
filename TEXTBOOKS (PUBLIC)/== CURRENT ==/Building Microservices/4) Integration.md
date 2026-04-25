# Locking for the Ideal Integration Technology

- SOAP
- XML-RPC
- REST
- Protocol buffers

## Avoid Breaking Changes

- If microservices adds new fields to a piece of data, existing customers should no be impacted

## Keep APIs Technology Agnostic

- New tools, frameworks, and languages are implemented to work faster and more effectively
- Avoid integration technology that dictates what technology stacks can be implemented to the microservices

## Make Service Simple for Customers

- Allow customers full freedom in their technology choice
- Client library

## Hide Internal Implementation Detail

- Any technology that pushes us to expose internal representation detail should be avoided

# Interfacing with Customers

- Customer creation
	- CRUD operations
- Enrolling a new customer pipeline
	- Payments
	- Emails

# The Shared Database

- Database integration
	- If other services want information form a service, they reach into the database

![[Pasted image 20260424225659.png]]

- Issues
	- External parties are able to view and bind to internal implementation details
	- Data structures in the DB are accessible
	- Schema is rigid
	- Requires regression testing
	- Customers are tied to a specific technology choice
		- Relation DB
		- Non-relational DB
- Logic association is coupled with multiple customers and services

# Synchronous vs. Asynchronous

**Synchronous**
- A call is made to a remote server
- Blocks until operation completes

**Asynchronous**
- Caller does not wait for the operation to complete before returning
- Useful for long-running jobs
- Low latency
- Responsive UI

- Request/response
	- A client initiates a request and waits for the response
	- Can work for both
- Even-based collaboration
	- Asynchronous
	- Business logic is not centralized
	- Task is pushed out more evenly to various collaborators
	- Highly decoupled

# Orchestration vs. Choreography

- Creating a customer
	- A new record is created
	- Postal system sends out a welcome pack
	- Send a welcome email to the customer

![[Pasted image 20260424230449.png]]

**Orchestration**
- Rely on a central brain to guide and drive the process
- Customer service can become too much of a central governing authority
- Becomes a central point where logic starts to live

![[Pasted image 20260424230705.png]]

**Choreography**
- Inform each pair of the system of its jobs
- Customer service emits an event in an asynchronous manner
- More decoupled
- The explicit view of the business process is now only implicitly reflected in the system
- Additional work is needed to monitor and track the processes
- Build a monitoring system the matches the view of the business process, then tracks what each of the services does as independent entities

![[Pasted image 20260424230801.png]]

- Systems that tend more toward the choreographed approach are more loosely coupled, and flexible to change
- Each service should be smart enough to understand its role in the entire system
- Synchronous called are simpler
- Longer lived processes
	- Asynchronous requests with callbacks

# Remote Procedure Calls

- Remote procedure call (RPC)
- Refers to the technique of making a local call and having it execute on a remove service somewhere
	- SOAP
		- XML format
	- Thrift
	- Protocol buffers
		- Binary format
- TCP
	- Offers guarantees about delivery
- UDP
	- Does not offer guarantees but has a lower overhead
- RCP implementations generate client and server stubs
- High performance systems
- Service talks internally
- Strict contracts

## Technology Coupling

- Java RMI
	- Heavily tied to a specific platform
- Thrift and protocol buffers
	- Support for alternative languages

## Local Calls are not like Remote Calls

- Core idea of RPC is to hide the complexity of a remote call
- Cost of marshalling and un-marshalling payloads can be significant
- API design for remote vs local interfaces
- Assume that the network is plagued
	- Failure modes

## Brittleness

- Defining a servie endpoint using Java RMI

```java
import java.rmi.Remote;
import java.rmi.RemoteException;

public interface CustomerRemote extends Remote {
	public Customer findCustomer(String id) throws RemoteException;
	
	public Customer createCustomer(String firstname, String surname, String emailAddress)
		throws RemoteException;
}
```

- Clients that want to consume the new method meed the new stubs
- Binary stub generation
	- Cannot separate client and server deployments
	- Lock step releases
- Encapsulate requires passing around dictionary types as parameters and extracting fields 
- Objects used as part of binary serialization across the wire can be through of as expand only types

## Is RPC Terrible?

- Do not abstract remote calls to the point where the network is completely hidden
- Ensure that you can evolved the server interface without having to insist on lock-step upgrades to clients
- Client libraries are often used in the context of RPC
- Make sure clients are not aware of network calls

# REST

- Representational State Transfer (REST)
- An architectural style inspired by the Web
- Resources are the service
- The server create different representations of a `Customer` on request
- A resource is completely decoupled from how it is stored internally
	- JSON representation
- Different 

# Implementing Asynchronous Event-Based Calibration

# Services as State Machines

# Reactive Extensions

# DRY and the Peril of Code Reuse in a Microservice World

# Access by Reference

# Versioning

# User Interfaces

# Integrating with Third-Party Software

# Summary