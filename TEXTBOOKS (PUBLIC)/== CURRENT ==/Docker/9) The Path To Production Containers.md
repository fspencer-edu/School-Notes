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

- Ability to command and organize applications and deployments across a whole system
- Tools
	- Capistrano
	- Fabric
	- Spotify's Helios
	- Ansible's Docker
	- New Relic's Centurion

### Service Discovery

- Service discovery is the mechanism by which the application finds all the other services and resources it needs on the network
- Stateless, static websites are the only systems that may not needs any service discovery
- In traditional systems
	- Load balancers where one of the primary means for service discovery
	- Load balancers are used for reliability and scaling
	- Track all of the endpoint associated with a particular service
- Static database configurations or application configuration files
- Docker does not address service discovery in your environment
	- In Docker Swarm mode


**Service Discovery Mechanisms**
- Load balancers with well-known addresses
- Round-robin records
- DNS SRV records
- Dynamic DNS systems
- Multicast DNA
- Overlay networks with well-known addresses
- Gossip protocols
	- Cassandra
	- Seneca JS
	- Sidecar
- Bonjour protocol (Apple)
- Apache Zookeeper
- HashiCorp's Consul
- CoreOS's etcd

### Production Wrap Up


## Docker and the DevOps Pipeline

- Ability to test the application and all of its dependencies in the exact operating environment for production
- Build an image, run on development box, then test the same image with the same application version and dependencies before shipping it to production servers

### Overview

- Pool of production servers that run Docker daemons
- Multiple application
- A build server and test worker boxes that are tied to the test server

**Workflow**
1) A build is triggered by some outside means
2) The build server kicks off a Docker build
3) The image is created on the local `docker`
4) The image is tagged with a build number or commit hash
5) A container is configured to run the test suite based on the newly built image
6) The test suite is run against the container and the result is captured by the build server
7) The build is marked as passing or failing
8) Passed builds are shipped to an image store (registry)


![[Pasted image 20260311120010.png]]


- Pushed the latest code to a Git repo
- Post-commit hook that triggers a build on each commit, so that job is kicked off on the build server
- The job on the test server is set up to take to a `docker` on a test worker server
	- Does not have `docker` running
- Run `docker build` against that remote Docker server and it runs the Dockerfile, generating a new image on the remote Docker server
- Once the image has been built, our test job will create and run a new container based on our new production image
- In production, start `supervisor` to start an `nginx` instance and some Ruby unicorn web server instance behind

```python
$ docker run -e ENVIRONMENT=testing -e API_KEY=12345 \
    -i -t awesome_app:version1 /opt/awesome_app/test.sh
```
- `docker run` will exit with the exit status of the command that was invoked in the container
- For additional steps, capture output of the test run into a file, and debug from status messages
- Take the passed build and push that image to the registry
- The registry is the interchange point between builds and deployments
- Take advantage of the client/server model to invoke the test on a different server from the test master server
- Therefore, the system only ships applications that have correctly passed on the test suite
- Container can also be test against any outside dependencies like databases or caches without having to mock them
- Jenkins for CI

### Outside Dependencies

- Memcache
- Redis instance

- Solve the external dependencies to have a clean test environment
- In Docker compose, our build job could express some dependencies between containers

## Wrap-Up