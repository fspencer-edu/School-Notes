
- Every Docker container is based on an image
- Images are the underlying definition of what gets reconstituted into a running container
- Every Docker image consists of one or more filesystem layers that have direct one-to-one mapping to each individual build step
- For image management, Docker relies on storage backend, which communicates with the underlying Linux filesystem to build and manage the multiple layers that combine into a usable image
- Primary storage
	- AUFS
	- BTRFS
	- Device-mapper
	- Overlay2
- Each storage backend provides a fast copy-on-write (CoW) system for image management

## Anatomy of a Dockerfile

- _Dockerfile_ describes all the steps that are required to create an image and would usually be contained within the root directly of the source code repository

```c
FROM node:11.11.0

LABEL "maintainer"="fiona@example.com"
LABEL "rating"="Five Stars" "class"="First Class"

USER root

ENV AP /data/app
ENV SCPATH /etc/supervisor/conf.d

RUN apt-get -y update

# Daemons
RUN apt-get -y install supervisor
RUN mkdir -p /var/log/supervisor

# Supervisor configuration
ADD ./superivsord/conf.d/* $SCPATH/

Application code

ADD *.js* $AP/

WORKDIR $AP

RUN npm install

CMD ["supervisord", "-n"]
```

- Each line creates a new image layer
- The base image is running in Ubuntu Linux Node 11.11.0
- Labels add metadata via key/value pairs that are used for identification
	- `docker inspect`
- `MAINTAINER` is a depreciated field in Dockerfile
- By default, Docker runs all processes as `root`
- Production containers should almost always be run under the context of a non-privileged user
- `ENV` allow you to set shell variables
- `RUN` instructions start and create required files
- `ADD` is used to copy files from local filesystem or a remote URL into image
- `WORKDIR`, changes the working directly

- The order of commands in a Dockerfile can have a very significant impact on ongoing build times
- Changes between every single build are closer to the bottom
- Adding code should be at the end
- `CMD` defines the command that launches the process
- Best practice to try to run only a single process within a container
- Containers should provide a single function so that it remains easy to horizontally scale individual functions within the architecture

### Building an Image

- Clone a repo from Git

```c
git clone https://github.com/spkane/docker-node-hello.git \
    --config core.autocrlf=input
```

- `.dockerignore` file allows you to define files and directories that you do not want uploaded to the Docker host when building an image
- `docker build` excludes the `.git` directory, which contains the whole source code repository
- `package.json` defines the Node.js application and lists any dependencies
- `supervisord` directory contains the configuration files used to start and monitory the application
- Docker server needs to run for the client to communicate with it
- Initiate a new build by running the upcoming command

```c
docker build -t example/docker-node-hello:latest .
```

- To improve the speed of builds, Docker will use a local cache when safe
- Use `--no-cache` to disable cache for a build
- Limits the resource available to builds


## Troubleshooting Broken Builds

- Almost all Docker images are layered on to of other Docker images
- Run an interactive container so that you can try to determine why your build is not working properly
- Every container image is based on the image layer below it
- Run the lower layer using

```bash
docker run --rm -ti 8a773166616c /bin/bash
root@464e8e35c784:/#
```

- Run any commands inside the container

## Running Your Image

```bash
docker run -d -p 8080:8080 example/docker-node-hello:latest
```

- Run Docker image and them map port 8080 into container to port 8080
- New Node.js application should be running in a container on the host
- Verify with `docker ps`
- Determine the Docker host IP address by simply printing out the value of `DOCKER_HOST` environment variables

```bash
echo $DOCKER_HOST

or

docker-machine ip
```

### Environmental Variables

- index.js
```c
var DEFAULT_WHO = "World";
var WHO = process.env.WHO || DEFAULT_WHO;
app.get('/', function (req, res){
	res.send('Hello ' + WHO + '. Wish you were here. \n');
});
```

```bash
docker stop b7145e06083f
docker run -d -p 8080:8080 -e WHO="Sean and Karl" \
    example/docker-node-hello:latest
    
Hello Sean and Karl. Wish you were here.
```

## Custom Base Images

- Base images are the lowest level images that other Docker images will build upon
- Build images using Alpine Linux, which is designed to be small and is popular as a basis for Docker images
- Based on modern, lightweight musl standard library, instead of GNU libc

## Storing Images

- Deployment is the process of pulling an image from a repository and running it on one or more Docker servers

### Public Registries
- Docker provides an image registry for public images that the community wants to share
	- Docker Hub
	- Quay.io
- Cloud vendors like Google also have their own registries
- A downside to registries is that they are not local to the network where the application is being deployed
- Every layer for every deployment might need to be dragged across the internet

### Private Registries
- Host Docker image internally
- Other options include
	- Docker Trusted Registry
	- Quay Enterprise Registry


### Authentication to a Registry

- Docker assumes the registry will be Docker Hub


**Creating a Docker Hub Account**

```c
docker login

ls -la ~/.docker/config.json

docker logout
```

**Pushing Images into a repository**

```c
docker login
docker build -t example/docker-node-hello:latest .

docker tag example/docker-node-hello:latest \
	${<myuser>}/docker-node-hello:latest
	
docker push ${<myuser>}/docker-node-hello:latest

docker pull ${<myuser>}/docker-node-hello:latest
```

### Running a Private Registry

```c
git clone https://github.com/spkane/basic-registry --config core.autocrlf=input

docker run --entrypoint htpasswd spkane/basic-registry:latest \

docker build -t my-registry .

docker run -d -p 5000:5000 --name registry my-registry

docker logs registry
```

**Testing the Private Registry**

## Advanced Building Techniques

 ### Keeping Images Small

- A large number of Docker containers inherit from a base image that contains a minimal Linux distribution

```c
docker image inspect alpine:latest

docker image inspect adejonge/helloworld:latest
```

**Multistage Builds**
```c
# build container
FROM golang:apline as builder
RUN apk update && \
	apk add git && \
	CGO_ENABLED=0 go get -a -ldflags '-s' github.com/adriaandejonge/helloworld
	
# Production container
FROM scratch
COPY --from=builder /go/bin/helloworld /helloworld
EXPOSE 8080
CMD ["/hellowowld"]
```

- Create a directory
- Paste the content from the preceding example into a _Dockerfile_

```bash
mkdir /tmp/multi-build
cd /tmp/multi-build
vi Dockerfile

docker build .
```

- Have a stage based non the public GO image that builds the API
- Another stage based on the Angular container to build the frontend
- Final stage to combine the outputs from both


**Layers Are Additive**
- Cannot delete file

```c
FROM fedora
RUN dnf install -y httpd
CMD ["/usr/sbin/httpd", "-DFOREGROUND"]

docker build .

docker tag 8d29ec43dc5a size1
```

- 3 layers added not size to final image
- Package managers
	- `apk, apt, dnf, yum`
- Cache uses up a huge amount of space
- Add `RUN dnf clean all`
- Cannot make earlier layers in an image smaller by deleting file in subsequent layers
- When a file is edited or deleted, the older version is being masked
- The most common way to deal with this is using `&&`

```c
FROM fedora
RUN dnf install -y httpd && \
	dnf clean all
CMD ["/usr/sbin/httpd", "-DFOREGROUND"]
```
## Wrap-Up
