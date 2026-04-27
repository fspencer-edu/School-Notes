# It's All About Seams

- Services should be highly cohesive and loosely coupled
- Working Effectively with Legacy Code
	- Defines the concept of a seam
	- Portion of the code that can be treated in isolation and worked on without impacting the rest of the codebase
	- Service boundaries
- Namespaces groups similar code together
	- `package` in Java

# Breaking Apart MusicCorp

- Identify high-level bounded contexts
- Backend of MusicCorp
	- Catalog
	- Finance
	- Warehouse
	- Recommendations
- Create packages representing these contexts
	- Move the existing code into them
	- Refactoring
- Remaining code will identify misses bounded contexts
- Analyze dependencies between packages
- Packages should represent similar contexts in real-life organizational groups

# The Reasons to Split the Monolith

- Incremental approach

## Pace of Change

- Split large or potential large code bases

## Team Structure

- Split given geographical regions

## Security

- Split to tighten up protection of sensitive information

## Technology

- Split highly used services

# Tangled Dependencies

- Acyclical graph of dependencies

# The Database

- Split database as well
# Getting the Grips with the Problem

- Repository layer
	- Backed by a framework like Hibernate, to bind code to the database, making it easy to map objects or data structures to and form the database

![[Pasted image 20260427001656.png]]

- Hibernate
	- Mapping file per bounded context
- SchemaSpy
	- Generate graphical representations of the relationships between tables

# Example: Breaking Foreign Key Relationships

![[Pasted image 20260427001844.png]]

- Stop the finance code from reaching into the line item table, from the catalog code
- Remove database integration
- Expose the data via an API call in the catalog package that the finance code can call

![[Pasted image 20260427001944.png]]

- 2 database calls to generate the report
- Removal of foreign key relationships
	- Implement other consistency checks across services
	- Trigger actions to clean up related data

# Example: Shared Static Data

![[Pasted image 20260427002126.png]]

- Duplicate table for each of our packages, with the long-term view that it will be duplicated within each service
	- Potential consistency challenge
- Treat the static data as a property file deployed as part of a the service, or as an enumeration
	- Easier to push out changes to config files, than alter live database tables
- Push static data into a service of its own right

# Example: Shared Data

- Shared mutable data

![[Pasted image 20260427002348.png]]

- Add the customer concrete to the current abstract concept
- Create a new package called `Customer`
- Use an API to expose `Customer` to other packages

![[Pasted image 20260427002503.png]]

# Example: Shared Tables

![[Pasted image 20260427002532.png]]

![[Pasted image 20260427002543.png]]

- Split the table into two

# Refactoring Databases

## Staging the Break

- Split out the schema, but keep the service together before splitting the application code into separate microservices

![[Pasted image 20260427003650.png]]

- A separate schema will increase the number of database called to perform a single action
- Pull the data back from two locations, and join in memory
- break transaction integrity when two schemas are moved
- Once DB separation is finished, split the application code
# Transactional Boundaries

- Transaction
	- Insert data
	- Update multiple tables
- A transaction allows us to group together multiple different activities that take our system from one consistent state to another
- Monolithic schemas
	- Create or updates will be done within a single transactional boundary
- Microservices
	- Multiple transactions to complete a task

![[Pasted image 20260427003952.png]]

- The order placing process spans two separate transactional boundaries

![[Pasted image 20260427004026.png]]

## Try Again Later

- Retry transactions in a queue or logfile
- Eventual consistency
	- Accept that the system will get itself into a consistent state at some point in the future
	- Long-lived operations

## Abort the Entire Operation

- Reject the entire operation
- Put system back into a consistent state
- Compensating transaction
	- Kicking off a new transaction to wind back what just happened
	- Report the operation failed
- If compensating transaction fails
	- Retry
	- Allow backend process to clean up the inconsistency later on
	- Maintenance screen, or automated process

## Distributed Transactions

- Distributed transactions try to span multiple transactions within them
- Transaction manager
	- Orchestrate the various transactions being done by underlying systems
- Tries to ensure that everything remains in a consistent state
- Communicating network boundaries
- Short-lived transactions
- Two-phase commit
	- First voting phase
		- Each cohort in the transaction tells the manager to go ahead
		- Relies on all parties halting until central process continues
	- Commit when all cohorts vote yes
- Technology stacks
	- Java's Transaction API

## So What to Do?

- Distributed transactions are hard to get right and can inhibit scaling
- In process order
	- Logical processing of the order from end to end

# Reporting

- Split up how and where the data is stored
- More difficult to report failures and log

# The Reporting Database

- Monolithic service
	- All data is stored in one large database
- Reporting across all the information is fairly easy
	- Join across the data via SQL queries
- Schema of the database is a shared API between the running service and any reporting system
- Change in schema is difficult to manage
- Limit options as to how the database can be optimized
	- Backing the live system
	- Reporting the system

![[Pasted image 20260427004953.png]]

# Data Retrieval via Service Calls

- Pull the required data from the source system via API calls
- Make multiple calls and assemble data
- Becomes harder as when volumes become large
- Reporting systems rely on third-party tools that except to retrieve data in a certain way
- Cache headers to the resources exposed by the service
	- Reverse proxy
	- Potentially expensive cache miss from long tail of data
	- Expose batch APIs

# Data Pump

- Rather than having a reporting system pull the data, have the data pushed to the reporting system
- pulling
	- HTTP overhead

![[Pasted image 20260427005451.png]]

- Cron job
- Version control
- Artifact to be deployed on application deployment

![[Pasted image 20260427005624.png]]

- A segmented schema may be less worthwhile, given the challenges of managing changes in the database

## Alternative Destinations

- Data pump to populate JSON files in AWS S3

# Event Data Pump

![[Pasted image 20260427005812.png]]

- 

# Backup Data Pump

# Toward Real Time

# Cost of Change


# Understanding Root Causes

# Summary