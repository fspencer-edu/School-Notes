
## Docker Client

- Docker client natively supports 64-bit versions of Linux, Windows, and macOS
- Debian systems uses the deb package format
- Red Hat rely on rpm (Red Hat Package Manager) files and Yellowdog updater, Modified (yum), or Dandified yum (dnf) to install similar software packages
- Homebrew for macOS
- Chocolatey for Windows

### Linux

- Run Docker on a modern release of preferred Linux distribution
- 3.8 or later kernel is required

**Ubuntu Linux 17.04 (64-bit)**

- Update docker version
```bash
sudo apt-get remove docker docker-engine docker.ioo
```

- Docker ships in two flavours
	- Community Edition (CE)
	- Enterprise Edition (EE)

- Install software dependencies
```bash
sudo apt-get update
sudo apt-get install \
	apt-transport-https \
	ca-certificates \
	curl \
	software-properties-common
	
curl -fsSL https://dowbload.docker.com/linux.ubuntu/gpg | \
	sudo apt-key add -
	
sudo add-apt-repository \
	"deb [arch=amd64] https://download.docker.com/linux/ubuntu
	$(lbs_release -cs) \
	stable"
```

- Install Docker
```bash
sudo apt-get update
sudo apt-get install docker-ce
```

**macOS**

- GUI installer
	- Download latest Docker Community Edition for Mac installer
	- Relies on the xhyve project to provide native lightweight virtualization layer for the Linux server component

- Homebrew installation
	- CLI package management system
	- Docker relies on Hyper-V for native virtualization layer

- Chocolatey installation
	- Package management system

**Windows 10**

- Download latest Docker Community Edition

## Docker Server

- The docker server is a separate binary from the client and is used to manage most of the work that Docker is used for

**systemd-Based Linux**
- `systemd` is used to manage processes on the system for Fedora and Ubuntu

```bash
sudo systemctl enable docker
sudo systemclt start docker
```

**Non-Linux VM-Based Server**

- Docker machine
	- Tools that makes is easy to set up Docker hosts on bare-metal, cloud, and virtual machine platforms
	- Install for GitHub release page
	- Use VirtualBox

```bash
mkdir ~/bin
curl -L https://github.com/docker/machine/releases/\
	download/v0.13.0/docker-machine-`uname -s`-`uname -m` 
	> ~/bin/docker-machine
export PATH=${PATH}:~/bin
chmod u+rx ~/bin/docker-machine

docker-machine create --driver virtualbox local
```

- Set up environment variables

```bash
docker-machine env local
export DOCKER_TLS_VERIFY="1"
export DOCKER_HOST="tcp://172.17.42.10:2376"
export DOCKER_CERT_PATH="/Users/me/.docker/machine/machines/local"
export DOCKER_MACHINE_NAME="local"
# Run this command to configure your shell:
# eval $(docker-machine env local)

docker-machine config local
--tlsverify
--tlscacert="/home/skane/.docker/machine/machines/local/ca.pem"
--tlscert="/home/skane/.docker/machine/machines/local/cert.pem"
--tlskey="/home/skane/.docker/machine/machines/local/key.pem"
-H=tcp://172.17.42.10:2376

docker-machine ip local
172.17.42.10

docker-machine ssh local
```

**Vagrant**
- Provides multiple hypervisors
- Support testing on images that match production environment
- Docker should always be set up to use only encrypted remote connections

```bash
mkdir docker-host
cd docker-host

git clone https://github.com/coreos/coreos-vagrant.git
cd coreos-vagrant
```

- Create a `config.ign` file

```bash
{
  "ignition": {
    "version": "2.0.0",
    "config": {}
  },
  "storage": {},
  "systemd": {
    "units": [
      {
        "name": "docker-tcp.socket",
        "enable": true,
        "contents": "[Unit]\nDescription=Docker Socket for the API\n..."
      }
    ]
  },
  "networkd": {},
  "passwd": {}
}
```

## Testing the Setup

```bash
# ubuntu
docker run --rm -ti ubuntu:latest /bin/bash

# fedora
docker run --rm -ti fedora:latest /bin/bash

# Apline
docker run --rm -ti alpine:latest /bin/sh
```

## Exploring the Docker Server

- Running the Docker daemon on a Linux system
```bash
sudo dockerd -H unix:///var/run/docker.sock -H tcp://0.0.0.0:2375
```

- Creates a Unix domain socket, and binds to all system IP addresses using the default unencrypted traffic port for docker

`docker run -it --privileged --pid=host debian nsenter -t 1 -m -u -n -i sh`

- Uses a privileged Debian container that contains the `nsenter` to manipulate the Linux kernel namespaces to navigate the filesystem

## Wrap-Up