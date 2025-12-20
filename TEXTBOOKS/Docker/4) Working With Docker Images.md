
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

```

## Troubleshooting Broken Builds

## Running Your Image

## Custom Base Images

## Storing Images

## Advanced Building Techniques

## Wrap-Up
