
- Docker was introduced by Solomon Hykes in 2013
- Docker is a tool that promises to easily encapsulate the process of creating a distributable artifact for any application, deploying it at scale into any environment, and streamlining the workflow and responsiveness of agile software organizations


## The Promise of Docker

- Docker's core domain spans
	- KVM
	- Xen
	- OpenStack
	- Mesos
	- Capistrano
	- Fabric
	- Ansible
	- Chef
	- Puppet
	- SaltStack
- Shipping software at the speed expected today is hard
- Docker helps to build a layer of isolation in software that reduces the burden of communication in the world of humans
- Docker is opinionated about software architecture that encourages robust applications
	- Atomic or throwaway containers
- Nothing in the environment of the application will live longer than the application itself
- Applications are not likely to rely on artifacts left by previous releases
- Applications are portable between servers
- Instances of the application container can come and go with little impact on the uptime of the frontend site

### Benefits of the Docker Workflow

- Packaging software in a way that leverages the skills developers already have
- Bundling application software and required OS filesystems together in a single standardized image format
- Using packaged artifacts to test and deliver the exact same artifact to all systems in all environments
- Abstracting software applications from the hardware without sacrificing resources

- Linux container has been around for a few years
- Docker combines architectural and workflow choices
- Docker has been becoming a foundational layer for modern distributed systems

## What Docker Isn't

- Enterprise virtualization
	- VMware
	- KVM
	- VMs contains a complete OS, running on top of a hypervisor that is managed by the underlying OS
- Cloud platform
	- OpenStack
	- CloudStack
	- Docker only handles deploying, running, and managing containers on pre-existing Docker hosts
	- Does not allow you to create new host systems (instances), object stores, block storage, and other resources involved with cloud platform
- Configuration management
	- Puppet
	- Chef
	- Dockerfiles are used to define how a container should look at build time
	- Do not manage the container's ongoing state, or host system
- Deployment framework
	- Capistrano
	- Fabric
	- Docker can't be used to automate a complex deployment process by itself
- Workload management tool
	- Mesos
	- Kubernetes
	- Swarm
- Development environment
	- Vagrant
		- VM management tool for developers to simulate server stacks in production environment


## Important Terminology

**Docker Client**
- This is the `docker` command used to control most of the Docker workflow and talk to remove Docker servers

**Docker Server**
- This is the `dockerd` command that is used to start the Docker server process that builds and launches containers via a client

**Docker Images**
- Docker images consists of one or more filesystem layers and some important metadata that represent all the files required to run a Dockerized application
- A single Docker image can be copied to numerous hosts
- An image has a name and a tag
	- Tag is used to identify a particular release of an image

**Docker Container**
- Linux container that has been instantiated from a Docker image
- Can only exist once
- Create multiple containers from the same image

**Atomic Host**
- Small, finely tuned OS image, like Fedora CoreOS, that supports container hosting and atomic OS upgrades

## Wrap-Up

