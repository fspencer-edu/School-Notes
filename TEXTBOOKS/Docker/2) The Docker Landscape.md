
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
- 

## Getting the Most from Docker

## The Docker Workflow

## Wrap-Up