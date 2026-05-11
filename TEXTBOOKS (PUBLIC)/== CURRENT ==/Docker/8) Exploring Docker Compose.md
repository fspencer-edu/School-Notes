
- Running multiple containers requires a proper setup
- Docker compose is used to streamline development tasks

```python
$ git clone https://github.com/spkane/rocketchat-hubot-demo.git \
		--config core.autocrlf=input
```

- The shell script is a `docker-copose.yaml` file

## Configuring Docker Compose

```bash
# example shell script
#!/bin/bash

set -e
set -u

if [ $# -ne 0 ] && [ ${1} == "down" ]; then
  docker rm -f hubot || true
  docker rm -f zmachine || true
  docker rm -f rocketchat || true
  docker rm -f mongo-init-replica || true
  docker rm -f mongo || true
  docker network rm botnet || true
  echo "Environment torn down..."
  exit 0
fi

# Global Settings
export PORT="3000"
export ROOT_URL="http://127.0.0.1:3000"
export MONGO_URL="mongodb://mongo:27017/rocketchat"
export MONGO_OPLOG_URL="mongodb://mongo:27017/local"
export MAIL_URL="smtp://smtp.email"
export RESPOND_TO_DM="true"
export HUBOT_ALIAS=". "
export LISTEN_ON_ALL_PUBLIC="true"
```

- Docker Compose is configured with a single, declarative YAML file for each project

```python
# docker compose file
version: '2'
services:
  mongo:
    build:
      context: ../../mongodb/docker
    image: spkane/mongo:3.2
    restart: unless-stopped
    command: mongod --smallfiles --oplogSize 128 --replSet rs0
    volumes:
      - "../../mongodb/data/db:/data/db"
    networks:
      - botnet
  mongo-init-replica:
    image: spkane/mongo:3.2
    command: 'mongo mongo/rocketchat --eval "rs.initiate({ ..."'
    depends_on:
      - mongo
    networks:
      - botnet
  rocketchat:
    image: rocketchat/rocket.chat:0.61.0
    restart: unless-stopped
    volumes:
      - "../../rocketchat/data/uploads:/app/uploads"
    environment:
	  PORT: 3000
      ROOT_URL: "http://127.0.0.1:3000"
      MONGO_URL: "mongodb://mongo:27017/rocketchat"
      MONGO_OPLOG_URL: "mongodb://mongo:27017/local"
      MAIL_URL: "smtp://smtp.email"
    depends_on:
      - mongo
    ports:
      - 3000:3000
    networks:
      - botnet
  zmachine:
    image: spkane/zmachine-api:latest
    restart: unless-stopped
    volumes:
      - "../../zmachine/saves:/root/saves"
      - "../../zmachine/zcode:/root/zcode"
    depends_on:
      - rocketchat
    expose:
      - "80"
    networks:
      - botnet
  hubot:
	image: rocketchat/hubot-rocketchat:latest
    restart: unless-stopped
    volumes:
      - "../../hubot/scripts:/home/hubot/scripts"
    environment:
      RESPOND_TO_DM: "true"
      HUBOT_ALIAS: ". "
      LISTEN_ON_ALL_PUBLIC: "true"
      ROCKETCHAT_AUTH: "password"
      ROCKETCHAT_URL: "rocketchat:3000"
      ROCKETCHAT_ROOM: ""
      ROCKETCHAT_USER: "hubot"
      ROCKETCHAT_PASSWORD: "bot-pw!"
      BOT_NAME: "bot"
      EXTERNAL_SCRIPTS: "hubot-help,hubot-diagnostics,hubot-zmachine"
      HUBOT_ZMACHINE_SERVER: "http://zmachine:80"
      HUBOT_ZMACHINE_ROOMS: "zmachine"
      HUBOT_ZMACHINE_OT_PREFIX: "ot"
    depends_on:
      - zmachine
    ports:
      - 3001:8080
    networks:
      - botnet
networks:
    botnet:
	    driver: bridge
```

- `version: '2'`
	- Sets docker version
- The rest of the document is divided into 2 sections
	- Services
		- `mongo`
		- `mongo-init-replica`
		- `rocketchat`
		- `zmachine`
		- `hubot`
	- Networks
		- The docker compose creates a single network, named `botnet`, using the default bridge drive, which will bridge the Docker network the the host's networking stack

### MongoDB

- Each named service contains sections that will Docker how to build, configure, and launch that service
- `build`
	- Contains a `context:` key which builds an image from files located in the directory
- `image: spkane/mongo:3.2`
	- Defines the image tag to build, download, and then run
- `restart`
	- Container option
- `command`
	- `command: mongo --smallfiles --oplogSize 128 --replSet rs0`
	- Define the command that the container should run at startup
- `volumes: \ - "../../mongodb/data/db:/data/db`
	- Mount a local directory into the containers
	- Create a volume to persist data
- `networks: -botnet`
	- Which network this container should be attached to

### Rocketchat

- Does not have a `build`section
- Defines an image without a build line, which tells Docker Compose that is cannot build this image and will pull and launch a standard pre-existing Docker image
- `environment`
	- Define environmental variables
	- `${<VAR_NAME>}`
- Configure URLs to point at the service name and internal port for the container
- `depends_on: -mongo`
	- Defines a container that must be running before this container can be started
	- Health function
- `ports: - 3000:3000`
	- Define all the ports mapped from container to host


### zmachine
- `expose: -80`
	- Expose this port to the other containers on the Docker network
	- Not the underlying host

## Launching Services

```python
$ cd rocketchat-hubot-demo/compose/unix

docker-compose config

docker-compose build
Building mongo

docker-compose up -d

docker-compose logs
```
- Docker compose prefixes the network and container name with the name of the directory that contains you `docker-compose.yaml` file


## Exploring RocketChat


## Exercising Docker Compose

- `docker-compose top`
	- Provides insight on running services
- `docker-comopse exec`
	- Provides the container shell
- `start, stop, pause, unpause`
- `docker-compose down`
	- Delete all the containers
## Wrap-Up


