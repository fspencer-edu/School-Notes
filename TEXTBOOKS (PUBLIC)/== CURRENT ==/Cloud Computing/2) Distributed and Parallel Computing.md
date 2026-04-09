- Parallel and distributed computing has made is possible to simulate large infrastructures
- Coordination is required to provide a service, share data, and store data that is too large to fit on a single machine
- Concurrent computation
	- A single program is execute by multiple processors with shared memory


- Middleware
	- Software that acts as a bridge between OS or database and applications
- Message passing
	- Communication between processing
	- Parallel programming and object-oriented design
	- Sends messages (packets) to recipient
- EJB container
	- Enterprise Jave Beans (EJB components)
	- Java server components that contain business logic
	- Provides local and remove access
- JMS queue
	- Staging area that contains messages that have been send and are waiting to be read
- XML
	- Extensible Markup Language
	- Defined a set of rules for encoding documents
- WS-I
	- Web Services Interoperability
	- Organization that establishes best practices for web services
- SOAP
	- Simple Object Access Protocol (SOAP)
	- Message protocol specification for exchanging structured information in web services
- REST
	- Representational state transfer
	- Software architectural style that defined a set of constrains to be used for creating web service
- Hyperthreading
	- Used by microprocessors that allows a single microprocessor to act like two separate processors to the OS
- Remote Method Invocation (RMI)
	- An API which allows an object to invoke a method on an object that exists in another adrress space
- Parallelism
	- Type of computation in which many computations or operations are carried out in parallel

# Distributed Computing

- A distributed system
	- Network of autonomous computers that communicate with each other
	- Computers are independent and do not physical share memory or processors
	- Have different roles depending on the computer's hardware and software properties

## Client-Server System

- Begin with UNIX (1970s)
- WWW
- Server provides a service
	- Respond to service requests from clients
- Multiple clients communicate with the server to consumer its products
	- Use the data provided in response to perform a task
- There is a single point of failure
- Used for service-oriented applications

## Peer-to-peer architecture

- Jobs are divided among all the components of the system
- All computers send and received data
- All contribute processing power and memory
- Peers need to be able to communicate with each other reliably
- Have an organized network structure
- Maintain enough information about locations of other components to send messages to destinations
- Data transfer/storage
- Skype, video-chat


# Properties of Distributed Systems

## Modularity

- Modularity is the idea that the components of a system should be black boxes with respect to each other
- Easier to understand, expand
- Replace defective components
- Localize malfunctions or bugs

## Message Passing

- Components communicate with each other using message massing
- Sender, recipient, and content
- Messages may need to be send over a network
	- Bit streams
	- Complex data structures
- A message protocol is a set of rules for encoding and decoding messages

# Performance Consideration in Distributed Computing

- High level architecture of remoting protocols
- Synchronous and asynchronous simple object access protocol (SOAP)


## High Level Architecture of Remoting Protocols

- Remote Method Invocation (RMI)
	- Jave Enterprise Edition (JEE)
- Design of the interface for the service
- Implement the methods specified in the interface
- Generate the stub and the skeleton
- Register the service by name and locations
- Use the service in an application

- Java Messaging Service (JMS)
	- Second most commonly used protocol in JEE
	- Asynchronous protocol
	- Based on queues where listeners are used to react on messages

**Main Components of JMS**
1. Publisher
	1. Responsible to publish message to queue
2. MQ Server/Message broker
	1. Holds messages in MQ server
3. Subscriber
	1. Performs the task on the message posted

**Types of Communication Supported**
- Point to point
- Publish-Subscribe
- JMS API
- Message Driven Bean
- Spring Framework

## SOAP

- A REST
	- An API that uses HTTP requests for client-server communication
	- Implement caching by HTTP proxies

# Parallel Computing

- Parallel computing
	- Type of computation in which many calculations are carried out simultaneously
		- Bit-level
		- Instruction-level
		- Data
		- Task parallelism
- Used in high-performance computing
- Load balancing
	- Frequency of a microprocessor can be adjusted depending on the needs
- Parallel
	- A computational task is broken down in several, similar subtasks that can be performed independently, and combined, upon completion
- Concurrency
	- The various processes often do not address related tasks
- All stand-alone computers are parallel from a hardware perspective with multiple functional units, multiple execution units/cores, multiple hardware threads

![[Pasted image 20260409153924.png]]

- Computer development milestones
	- Mechanical
	- Electro-magnetic parts
- Element of modern computers
	- Hardware
	- Instruction sets
	- Application programs
	- System software
	- User interface
- Computer architecture
	- Von Neumann to multicomputer and multiprocessors
- Performance of a computer system
	- Improved by hardware and software
	- Efficient resource management


# Performance Consideration in Parallel Computing

## Parallelism Degree

- Pipelining
	- A set of data processing elements connected in series
	- Output of one element is the input of the next one
- The $T(n, N)$ needed to execute more complex operations such as sorting a list of N numbers, or inverting an $N*N$ matrix is given as a performance measure for a machine of parallelism $n$
- O-notation
- The performance of an individual processor or processing element can be measured by the instruction rate or bandwidth $b_1$, using Million Instructions Per Second (MIPS)
	- $b_d$ => data bandwidth is typically Mflops
	- Gflops
- Interprocessor communication mechanisms, strongly influence overall system performance

## Speedup

- Speedup, $Spn(n)$
- Ratio of the total execution time $T(1)$ on a sequential computer to the corresponding execution time $T(n)$ on the parallel computer for a task

$Sp(n) = T(1)T(n)$

$T(1) \leq n*T(n)$
$Sp(n) \leq n$

## Efficiency

- A closely related performance measure
- Expressed as $E_n$
	- Which is the speedup per degree of parallelism
- Also indicate processor utilization

$E(n) = S(n)n$

- Speed up and efficiency provides estimates of the performance change that can be expected in a parallel processing system by increasing the parallelism degree
- Depend on running program, and can change from program to program

# Amdahl's Law

- A program or an algorithm Q may sometimes by characterized by its degree of parallelism $n_i$
	- Min value of n for which the efficiency and speedup reach their max values at time $i$ during execution of Q
$n_i$ => max level of parallelism that can be exploited

- The overall speedup of a system is limited by the part that cannot be improved

$speedup = \frac{1}{(1-p)+ \frac{p}{n}}$
$p$ = fraction of program that can be improved
$1-p$ = part that cannot be improved
$n$ = speedup of improved part


![Amdahl's law - Wikipedia](https://upload.wikimedia.org/wikipedia/commons/e/ea/AmdahlsLaw.svg)

# Types of Parallelism

- Parallel execution or processing involved the division of tasks into several smaller tasks
- Improve response times and increases throughput
- Requirements
	- Computer system/server with built in multiple processors and message facilitation
	- OS capable of managing multiple processors
	- Clustered nodes with application software
		- Oracle Real Application Clusters (RAC)

## Bit-level

- Increasing the word size reduces the number of instructions the process must execute to perform an operation on variables whose sizes are greater than the length of the worth
- 16, 32, vs 64 -bit processors

## Task-level

- Decomposition of a task into sub-tasks and then allocated each sub-task simultaneously and often cooperatively
- Parallelism of various tasks and communication between

## Instruction-level

- All modern processors have multi-stage instruction pipelines
- Each stage corresponds to a different action the processor performs on the instruction in that stage

# Flynn's Classical Taxonomy

- Distinguishes multi=process computer architectures according to how they can be classified
	- Instruction Stream
	- Data Stream

- 2 possible states
	- Single
	- Multiple
- Classification based on instruction and data streams
- Classification based on the structure of computers
- Classification based on memory access
- Classification based on grain size

**Single Instruction, Single Data (SISD)**
- A serial computer
- Only one instruction by CPU during a clock cycle
- Only one data stream
- Deterministic execution
- Older generation mainframes, minicomputers, single core

**Single Instruction, Multiple Data (SIMD)**
- Parallel computer
- All processing units execute the same instruction at any given clock cycle
- Each processing unit can operation on a different data element
- High degree or regularity, graphics/image processing
- Synchronous (lockstep) and deterministic execution
- Processor arrays and vector pipelines
- Most modern computers, with GPUs

**Multiple Instruction, Single Data (MISD)**
- A type of parallel computer
- Each process unit operates on the data independently via separate instruction streams
- A single data stream is fed into multiple process units
- Few actual examples
- Multiple frequency filters operating on a single stream
- Cryptography

**Multiple Instruction, Multiple Data (MIMD)**
- Parallel computer
- Every process may execute a different instruction stream
- Syhchrononous or asynchronous, deterministic or non-deterministic
- Most common type of parallel computer
- Networked parallel computer clusters
- Multi-processor SMP computer, multi-core PC

# Classes of Parallel Computers

- Parallel computers can be classified according to the level at which the hardware supports parallelism
	- Distance between basic computing nodes

## Distributed computing

- A distributed computer
	- Distributed memory multiprocessor
	- Processing elements are connected by a network
	- Highly scalable
## Cluster computing

- Many computers connected on a network and perform like a single entity
- Each computer is called a node
- Improve computational speed, and data integrity
- Load balancing is more difficult
- Beowulf cluster
	- Multiple identical commercial off-the-shelf computers connected with TCP/IP Ethernet LAN
- Use hardware for networking
## Grid computing

- A group of networked computers which work together as. a virtual super computer
	- Analyzing huge sets of data
- Typically have not time dependency
- Use computers which are part of the grid only when idle and operators can perform tasks unrelated to the grid
- Connects geographically dispersed, heterogeneous computers to functions as a VM
- Long-running batch processing tasks
- Decentralized
- Fixed, predetermined capacity

## Cloud computing

- Provides on demand, virtualized resource centralized by cloud providers
	- Servers
	- Storage
	- Databases
	- Networkning
	- Software
	- Analytics
- Flexible resources and economy of scale
## Massively parallel computing

- MPP
	- A single computer with many networked processors
	- Similar characteristics as cluster, but have specialized interconnect networks
	- Larger than clusters

## Specialized parallel computing

- Uses multiple processing elements simultaneously—including specialized hardware like GPUs, FPGAs, and ASICs—to solve complex, compute-intensive problems by breaking them into smaller, independent tasks

## Vector processors

- CPU or computer that can execute the same instruction on large sets of data
- High-level operations that work on linear arrays of number of vecotrs

## Multi-core computing

- A processor that includes multiple processing units on the same chip
- Issue multiple instructions per clock cycle from multiple instruction streams
