
## Process Simplification

**A cycle of an application:**
1. Application developer request resources from operations engineer
2. Resources are provisioned and handed over to developers
3. Developers script and tool their deployment
4. Operations engineers and developers tweak the development repeatedly
5. Additional application dependencies are discovered by developers
6. Operation engineers work to install the additional requirements
7. Loop over step 5 and 6
8. The application is deployed

![[Pasted image 20251217140941.png]]


**Heroku**
- Push-to-deploy systems like Heroku have shown developers help control dependencies
- Whole environment, not just the container engine
- Docker provides a clean separation of responsibilities and encapsulation of dependencies
- Docker also allows more fine-grained control by putting developers in control of everything

- By using an image repository as the hand-off point, Docker allows the responsibility of building the application image to be separated from the development and operation of the container

**New Cycle with Docker**
1. Developers build the Docker image and ship to the registry
2. Operations engineers provide configuration details to the container and provision resources
3. Developers trigger deployment

![[Pasted image 20251217141514.png]]

- Docker allows all the dependency issues to be discovered during the development and test cycles

## Board Support and Adoption

- The majority of large public clouds offer support for Docker
	- Docker runs in AWS via multiple products
		- Elastic Container Service (ECS)
		- Elastic Container Service for Kubernetes (EKS)
		- Fargate
		- Elastic Beanstalk
- Docker can also be used on
	- Google AppEngine
	- Google Kubernete Engine
	- Red Hat OpenShift
	- IBM Cloud
	- Microsoft Azure
	- Rackspace Cloud
	- Docker Cloud
- Docker's containers are becoming the lingua franca between cloud providers
- Docker client runs directly on most major OS
- Server can run on Linux or Windows server
- Docker has released easy-to-use implementation for macOS and Windows
	- Runs on Linux kernel underneath
- Docker has traditionally been developed on the Ubuntu Linux distribution, but most Linux distros and other major OS are supported
- The Open Container Initiative (OCI) 2015
	- Full specification of Docker image format
	- Certification process
	- 4 runtimes claiming to implement the spec
		- `runc` - Docker
		- `railcar` - Oracle
		- Kata Containers from Intel, Hyper, and OpenStack
		- gVisor runtime from Google


## Architecture

- Docker heavily leverages kernel mechanisms
	- iptables
	- Virtual bridging
	- cgroups
	- namespaces
	- filesystem drivers

### Client/Server Model

- Client
- Server/daemon
- Registry
	- Stores Docker images and their metadata
- Docker daemon can run on any number of servers in the infrastructure
- Clients drive all communications and are responsible for telling servers what to do

![[Pasted image 20251217142506.png]]

- `docker` client
- `dockerd` server
- Instead of monolithic structure, the server uses
	- `docker-proxy`
	- `runc`
	- `containerd`
	- `docker-init`

**Network Ports and Unix Sockets**
- `docker` and `dockerd` talk to each other over network socket
- Docker has registered two ports with IANA for use by the Docker daemon and client
	- TCP ports 2375 (unencrypted) and 2376 (encrypted SSL)
- Unix socket can be located in different paths on different OS

**Robust Tooling**
- The older, deprecated project was called Docker Swarm
- Newer version, is Sward mode

- Docker has launched its own orchestration toolset, include Compose, Machine, and Swarm
	- Creates a cohesive deployment story for developers
- Overshadowed by Kubernetes and Apache Mesos

**Docker CLI**
- The CLI is the main interface of Docker
- Go program that compiles and runs on all common architectures and OS
	- Build a container image
	- Pull images from a registry to a Docker daemon or push to registry
	- Start a container on a Docker server in foreground or background
	- Retrieve the Docker logs from a remote server
	- Start a CLI shell inside a running container on a remote server
	- Monitor statistics about container
	- Get a process listing from container

**Docker Engine API**
- Docker daemon has a remote web application programming interface (API)
	- Mapping deployed Docker containers to server
	- Automated deployments
	- Distributed schedulers
- Docker maintains SDKs for Python and Go

**Container Networking**
- Docker initially supported a single networking model
- Now supports a robust assortment of configurations that handle most application requirements
- Most people run containers in the default configuration, called bridge
- mode
	- Each container behaves on a private network
	- Docker servers acts as virtual bridge and the containers are clients behind it
	- A bridge repeats traffic from one side to another
- Each container has its own virtual Ethernet interface connected to the Docker bridge and its won IP address
- Traffic passes over a proxy that is also part of the Docker daemon before getting to the container

![[Pasted image 20251217144943.png]]

- Docker allocates the private subnet from an unused private subnet block
- Bridged to the host's local network through an interface on the server called `docker0`
- All container are on a network together and can take to each other directly
- To get host or outside world, they go over the `docker0` virtual bridge interface

## Getting the Most from Docker

- Docker's architecture aims it squarely at applications that are either stateless or where the state is externalized into data stores like databases or caches
- Databases that run well in Docker can be deployed, although is difficult
- Best use cases
	- Web frontends
	- Backend APIs
	- Short-running tasks
- Start with a stateless application

**Containers are Not Virtual Machines**
- Docker containers are lightweight wrappers around a single Unix process
- Process might spawn others
- Containers are also ephemeral
	- They come and go more than traditional VM
- VM are a stand-in for real hardware
- VMs are often long lived in nature
- Docker can run natively, therefore there is no need for a virtual machine to be run on the system

![[Pasted image 20251217145743.png]]

**Limited Isolation**

- Containers are isolated from each other
- Containers can compete for resources on production machines
- Limits on CPU and memory are encouraged
- Many containers share one or more common filesystem layers
- Containerized processes are just processes on the Docker server
- Runs on same exact instance of Linux kernel as OS
- Apply further isolation
	- SELinux
	- AppArmor

- Many containers use UID 0 to launch processes
- Although the container is contained, it can access the kernel and cause security vulnerabilities

**Containers Are Lightweight**
- A newly created contained from an existing image is 12 kb
- VM require hundreds of thousands of mb
- New container is a reference to a layered filesystem image and metadata

**Toward an Immutable Infrastructure**
- 

## The Docker Workflow

## Wrap-Up