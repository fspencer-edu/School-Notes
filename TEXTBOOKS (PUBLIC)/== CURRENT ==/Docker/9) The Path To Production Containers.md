## Getting to Production

**Production on Docker**
1. Locally build and test a Docker image
2. Build an official image for testing and deployment, from CI or build system
3. Push the image to a registry
4. Deploy your Docker image to your server, then configure and start the container

**Workflow**
- Orchestrate the deployment of images and creation of containers on production servers
	- Repeatable process
	- Handle configuration
	- Deliver an executable artifact that can be started


## Docker's Role in Production Environments

- The platform is a system that wraps around more than one instance of Docker and presents a common interface across Docker instances
	- Mesos
	- Kubernetes
	- Docker Swarm
	- Deployment system
	- Separate monitoring system
	- Separate orchestration system

![[Pasted image 20260311113154.png]]

### Job Control

- Job control makes up the table stakes for a modern deployment
- Traditionally left to OS, or Linux init system
- Configure OS from running a process
	- Restarting
	- Reloading
	- Lifecycle
- In Linux system a `cron` is used to start and stop jobs on a timed basis
- Docker engine provides a strong set of primitives around job control
	- `docker start`
	- `docker stop`
	- `docker run`
	- `docker kil`

### Resource Limits

- In Linux systems `cgorups` are used to limit resources
- `ulimit` in Java, Ruby, or Python VM
- In cloud systems, virtual servers limit the resources around a single applications
- Apply a set of resource controls to containers
	- Memory, disk space, IO

### Networking


### Configuration

- 2 levels of configuration for an application
	- Lowest is the Linux environment
		- Chef, Puppet, Ansible
	- Configuration of application
- Dockerfile can be used to build the same 2 level configurations

### Packaging and Delivery

- A containerized system has major advantages over traditional one
- Consistent package, container image, and standardized Docker registry

### Logging

- Docker can collect all the logs from the containers and ship them somewhere

### Monitoring

- Standardized health-checks
- Alternative systems
	- Nagios
	- Zabbix
- In dynamic systems, issues are moved to a more automated process that belong in the platform

### Scheduling

- Containers are easy to move around
	- Better resource usage, reliability, self-healing services, and dynamic scaling
- Older systems
	- One-service-per-server model
- Cloud systems use one-service-per-server model
- Autoscaling with AWS handles dynamic behaviour


### Distributed schedulers
- Blue-green style
	- Launch new generation of an application alongside the old generation, then slowly migrate from old to new stack
- Traditional systems
	- Apache Mesos
		- A resource pool abstraction that lets you run multiple frameworks on the same cluster of hosts
		- Zookeeper
	- Mesos has support for Docker
		- System scheduling is handled by the framework
			- HubSpot's Singularity
			- Mesosphere's Marathon
			- Mesos's Chronos
			- Apache Aurora
- Schedulers handle different kinds of workloads
	- Long-running services
	- One-off commands
	- Scheduled jobs
- Kubernetes is another popular scheduler from Google
	- Built on Docker and runs Docker containers
- Docker Swarm
	- Build as a Docker native system


### Orchestration

- 


### Service Discovery

### Production Wrap Up


## Docker and the DevOps Pipeline

## Wrap-Up