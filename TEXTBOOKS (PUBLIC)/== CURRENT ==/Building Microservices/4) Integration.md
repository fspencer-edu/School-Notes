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
- Different styles of REST
- Most commonly used over HTTP
	- Serial or USB

## REST and HTTP

- HTTP verbs
	- GET
	- POST
	- PUT
- Conceptually there is one endpoint in the form of a `Customer` response, and operations are carried out into the HTTP protocol
- HTTP also brings a large ecosystem of supporting tools and technologies
- HTTP caching proxies, load balancers, monitoring tools
- Security controls
	- Auth
	- Client certs
- HTTP can be used to implement RPC
	- SOAP gets routed over HTTP, with little specification

## Hypermedia as the Engine of Application State

- Hypermedia as the engine of application state (HATEOAS)
- Hypermedia
	- A concept whereby a piece of content contains links to various other pieces of content in a variety of formats
	- Clients should perform interactions (leading to state transitions) with the server via these links to other resources

- Hypermedia controls used on an album listing

```xml
<album>
  <name>Give Blood</name>
  <link rel="/artist" href="/artist/theBrakes" /> 1
  <description>
    Awesome, short, brutish, funny and loud. Must buy!
  </description>
  <link rel="/instantpurchase" href="/instantPurchase/1234" /> 2
</album>
```
- 2 hypermedia controls
	- `artist`
	- `instantpurchase`
- Controls to decouple the client and server yields significant benefits over time that greatly offset the small increase in the time it takes to get these protocols to run
- Navigation of controls can be complex
- Use links to allow consumers to navigate API endpoints
	- Reduced coupling

## JSON, XML, or Something Else?

- JSON
- XML
- Hypertext Application Language (HAL)

- Navigation
	- Xpath
	- Jsonpath

## Beware Too Much Convenience

## Downsides to REST over HTTP

- REST
	- Cannot easily generate a client stub a protocol like in RPC
- Some web server frameworks do not support all the HTTP verbs
- REST over HTTP payloads can be more compact than SOAP because is supports alternative formats like jsON or binary
	- Some overhead
	- Not great for large volumes of traffic
	- Low latency communications
- Server-to-server communications
	- UDP
	- RPC frameworks
- RPC implementations may require advanced serialization and deserialization mechanisms

# Implementing Asynchronous Event-Based Calibration

## Technology Choices

1) Microservices to emit events
2) Consumers to find out those events have happened

- Message brokers
	- RabbitMQ
	- Handle both problems
	- Producers use an API to publish an event to the broker
	- Broker handles subscriptions, allowing consumers to be informed of events
	- Complex development process
- Middleware
	- Queues
- ATOM
	- REST-compliant specification that defined semantics for publishing feed of resources
	- HTTP is not good at low latency

## Complexities of Asynchronous Architectures

- Ensure good monitoring
- Correlation Ids
	- Trace requests across process boundaries
- Completing consumer pattern
- Bad messages to message dead letter queue

# Services as State Machines

- Customer service controls all lifecycle events associated with the customer itself

# Reactive Extensions

- Reactive extensions also called Rx
- A mechanism to compose the results of multiple calls together and run operations on them
- Blocking or non-blocking calls
- Observe the outcome of an operation and react when something changes
- RxJava
	- `map`
	- `filter`
- Compose multiple calls together, making handling concurrent calls to downstream services mush easier

# DRY and the Peril of Code Reuse in a Microservice World

- DRY
	- Don't repeat yourself
- Pull duplicated code into abstractions that we can then calls from multiple places
- Shared logging libraries

## Client Libraries

- People that create both the server API and the client API
	- Danger that logic that should exist on server starts leaking into the client
- AWS
	- SOAP or Rest web service calls can be made directly, but everyone ends up using just one of the various SDKs
	- Client is in charge of when the upgrade happens
- Netflix client libraries
	- Service discovery
	- Failure modes
	- Logging
	- Other libraries not ties to the service
- Separate client code to handle the underlying transport protocol, from destination services

# Access by Reference

- Microservice will encompass the lifecycle of our core domain entities
- If a memory of a resource is passed make sure the reference of the original resource is also passed
- Accessing by reference can cause load on the customer service too be too great
	- Cache to reduce load
- Some services might not need to known about an entire resource

# Versioning

## Defer It for a Long as Possible

- Integration technology

_Sample Response from the customer service_
```xml
<customer>
  <firstname>Sam</firstname>
  <lastname>Newman</lastname>
  <email>sam@magpiebrain.com</email>
  <telephoneNumber>555-1234-5678</telephoneNumber>
</customer>
```

- Use XPath to pull out specific fields
- Tolerant reader

_A restructured Customer resource_
```xml
<customer>
  <naming>
    <firstname>Sam</firstname>
    <lastname>Newman</lastname>
    <nickname>Magpiebrain</nickname>
    <fullname>Sam "Magpiebrain" Newman</fullname>
  </naming>
  <email>sam@magpiebrain.com</email>
</customer>
```

- Postel's Law
	- Robustness principle
	- Be conservative in what you do, be liberal in what you accept from others

## Catch Breaking Changes Early

- Consumer-driven contracts
- Tests

## Use Semantic Versioning

- A specification that allows versioning matching and easier integration
- Each version number is in the form `MAJOR.MINOR.PATCH`
- `MAJOR` increments, means it is backward incompatible
- `MINOR` increments, new functionality has been added that should be backward compatible
- `PATCH` state that bug fixes have been made to existing functionality

## Coexist Different Endpoints

- Maintain the ability to release microservice independently of each other
- Avoid forcing customers to upgrade in lock-step
- Coexist both the old and new interfaces in the same running service
- Deploy a new version of the service that exposes both the old and new versions of the endpoint
- When consumers are no longer using the old endpoint, remove it along with any associated code

![[Pasted image 20260426130225.png]]

- Expand and contract pattern
	- Phase breaking change in
- HTTP or URI version numbers in request headers
```python
/v1/customer/
/v2/customer/
```

- RPC
	- Protocol buffers with methods in different namespaces

```python
v1.createCustomer
v2.createCustomer
```

## Use Multiple Concurrent Service Versions

- Route older consumers to older versions, and newer users to the newer versions
- Cost of changing older consumers it too high
	- Legacy devices
- Use of more `nginx` scripts
- Manage persistent state
- Customers created by either version of the service need to be stored and made visible to all services

![[Pasted image 20260426130653.png]]

- Coexisting concurrent service versions
	- Blue/green deployment
	- Canary releases
- The longer it takes for you to get consumers upgraded to the newer version and released, the more you should look to coexist different endpoints in the same microservice rather than coexist entirely different versions

# User Interfaces

## Toward Digital

- Organizations have started to move away from thinking that web or mobile should be treated differently
- Granular APIs
- Compositional layers
	- Places where we eave together various strands of the capabilities we offer

## Constraints

- Web applications
	- Browser version
	- Resolution
- Mobile
	- Server communication
	- Bandwidth
	- Battery
	- Touch gestures
	- SMS service

## API Composition

- Web
	- JS HTTP requests
- API gateway
	- Expose calls that aggregate multiple underlying calls

![[Pasted image 20260426131230.png]]

## UI Fragment Composition

- Map everything back to UI controls
- Pull these fragments in to create a UI
- Assemble a series of coarser grained parts of UI
	- Assemble entire planes of thick client application, or a set of pages for a website
- Fragments are served up from server-side apps that are in turn making the appropriate API calls

![[Pasted image 20260426131436.png]]

- 
## Backends for Frontends

## A Hybrid Approach


# Integrating with Third-Party Software

## Lack of Control

## Customization

## Integration Spagetti

## On Your Own Terms

### Example: CMS as a Service

### Example: The multirole CRM System

## The Strangler Pattern

# Summary