```python
## === GENERAL ===

# start docker daemon
docker -d

# help
docker --help

docker info

## === IMAGES ===

# build an images from a Dockerfile
docker build -t <image_name> .

# build an image from a Dockerfile without the cache
docker build -t <image_name> . -no-cache

# list local images
docker images

# delete an image
docker rmi <image_name>

# remove all unused images
docker image prune

## === CONTAINERS ===

# create and run a container from an image
docker run --name <container_name> <image_name>

# run container with and publish to port to host
docker run -p <host_port>:<container_port><image_name>

# run a container in the background
docker run -d <image_name>

# start or stop an container
docker start|stop <container_name>

# open a shell inside a running container
docker exec -it <container_name> sh

# fetch and fllow the logs of a contaner
docker logs -f <container_name>

# inspect a running container
docker inspect <container_name>

# list currently running containers
docker ps
docker ps --all

# resource usage stats
docker container stats

# === DOCKER HUB ===

# login
docker login -u <username>

# publish an image to DH
docker push <username>/<image_name>

# search hub for an image
docker search <image_name>

# pull an image
docker pull <image_name>
```