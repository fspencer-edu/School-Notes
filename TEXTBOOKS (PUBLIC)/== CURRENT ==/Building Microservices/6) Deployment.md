- CI/CD

# A Brief Introduction to Continuous Integration

- Continuous integration (CI)
	- Builds
	- Version control repo
	- A CI server detects that code has been committed, checks it out, and carries out verifications/tests
- Artifacts
	- Used to further validate deployed services
- Automate the creation of binary artifacts
- Traceability

# Mapping Continuous Integration to Microservices

<img src="/images/Pasted image 20260427220356.png" alt="image" width="500">

- Group services into one large repository
	- Build will trigger events, to produce multiple artifacts, and tie together
	- Lock-step releases
	- All services are affects
	- More time on start up
- Single source tree will all of the code
	- Multiple CI builds mapping to parts of the source tree
	- Check-in/check-out process can be simpler
	- Easy to make changes that couple services together

<img src="/images/Pasted image 20260427220539.png" alt="image" width="500">

- A single CI build per microservice
	- Quickly make and validate a change prior to deployment into production
	- Run only the build and test for the changed service
	- Produce individual artifacts to deploy

<img src="/images/Pasted image 20260427220705.png" alt="image" width="500">

- Test for microservice should live in the source control with the microservice's source code

# Build Pipelines and Continuous Deliver

- Tests
	- Small-scoped
	- Large-scoped
- Build pipeline
	- Different stages in the build process
	- Track progress of software
- Continuous delivery (CD)
	- Get constant feedback on the production readiness of each and every check-in
	- Every stage of software goes through, both manual and automated build pipeline

<img src="/images/Pasted image 20260427220925.png" alt="image" width="500">

- User acceptance testing (UAT)

## Exceptions

- One microservice per build
- Merge to a monolithic service, to develop experience with the domain
- Then properly separate boundaries

# Platform-Specific Artifacts

- First-class artifact
	- Ruby -> gems
	- Java -> JAR and WAR
	- Python -> eggs
- For some applications a process manager running inside Apache or Nginx is needed
- Automated configuration management tools
	- Puppet
	- Chef

# Operating System Artifacts

- Create artifacts that are native to the underlying OS
	- Linux
	- Windows
- Each OS requires their own package manager

# Custom Images

- Create a virtual machine image that bakes in some of the common dependencies
	- Reduce spin-up time

<img src="/images/Pasted image 20260427221821.png" alt="image" width="500">

- Building images can take a long time
- Resulting images can be large
- Packer
	- Tool designed to make creation of images easier

## Images as Artifacts

 - Create a VM image that bake in the dependencies to speed up feedback
 - Bake service into image
 - The service is ready at launch
 - Netflix uses AWS AMIs

## Immutable Servers

- Ensure that services and entire environments can automatically reproduce
- Configuration drift
	- Code in the source control no longer reflects the configuration of the running host
	- Ensure no changes are made to a running server
	- A change should go through a build pipeline in order to create a new machine
	- Disable SSH

# Environments

- Environments
	- SLow test
	- UAT
	- Performance
	- Production
- Differences between test and deployment environments can introduce problems

# Service Configuration

- Build one artifact per environment
	- Configuration inside
- Create one single artifact and manage configuration separately
	- Property files

# Service-to-Host Mapping

- Virtualization
	- A single physical machine can map to multiple independent hosts
	- Each host can hold one or more services

## Multiple Services Per Host

- Virtualization can add overhead that reduces the underlying resources available to the service
- Application container
	- Multiple-service-per -host model
	- Monitoring more difficult
	- Tracking independent CPU
	- Deployment of services is more complex
	- Inhibit autonomy of teams
	- Limit deployment artifact options
- On demand computing
	- Reduced the costs of computing resources
- Improvement in virtualization

## Application Containers

- .NET applications
- Java application

<img src="/images/Pasted image 20260427223122.png" alt="image" width="500">

- Reduces the overhead
- Constrains technology choice
- Analyzing resource use and threads is more complex

## Single Service Per Host

- Avoid side effects of multiple hosts living on a single host
- Monitoring and remediation simpler
- Reduced single point of failure
- More easily scale one service independent from others

<img src="/images/Pasted image 20260427223332.png" alt="image" width="500">

- Use alternative deployment techniques
	- Image-based
	- immutable server pattern
- Increased number of hosts has a downside
	- More servers to manage
	- Cost of running

## PaaS

- Higher level abstraction compared to a single host
- Heroku

# Automation

## 2 Case Studies on the Power of Automation

# From Physical to Virtual

- Find ways of chucking up existing physical machines into smaller parts

## Traditional Virtualization

- Type I virtualization
	- Hardware
- Type II virtualization
	- Software
- Hypervisor
	- Maps resource like CPU and memory from the virtual to physical host
	- Control layer to manipulate the VM
	- Requires CPU, IO, and memory

<img src="/images/Pasted image 20260427224050.png" alt="image" width="500">

## Vagrant

- Used for dev and tests rather than production
- Virtual cloud on laptop
- Define VMs in a test file, and how VMs are networked together
- Create production-like environments on local machines

## Linux Containers

- Process are run by a given user, and have certain capabilities based on permissions
- Processes can spawn other processes
- Containers are faster to provision than full VMs

## Docker

- Built on top of lightweight containers
- Create and deploy apps that are synonymous with images in the VM world
- Host a single VM in Vagrant that runs a Docker instance
- Software is installed as independent docker apps

# A Deployment Interface

- Uniform interface
	- Keep deployment mechanisms as similar as possible from dev to production
- Single, parametrizable command line call to trigger deployment
	- Python Fabric scripts
		- Maps CLI calls to functions
		- SSH calls
	- AWS Boto
	- Ruby Capistrano

```python
deploy artifact=catalog environment=local version=local

deploy artifact=catalog environment=ci version=b456

deploy artifact=catalog environment=integrated_qa version=latest

```
## Environment Definition

- Define environment
- YAML

```python
development:
  nodes:
  - ami_id: ami-e1e1234
    size:   t1.micro
    credentials_name: eu-west-ssh
    services: [catalog-service]
    region: eu-west-1

production:
  nodes:
  - ami_id: ami-e1e1234
    size:   m3.xlarge 
    credentials_name: prod-credentials 
    services: [catalog-service]
    number: 5  
    
catalog-service:
  puppet_manifest : catalog.pp 
  connectivity:
    - protocol: tcp
      ports: [ 8080, 8081 ]
      allowed: [ WORLD ]
```

- 

# Summary