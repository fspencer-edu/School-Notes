## What Are Containers?

- Container share a single kernel, and isolation between workloads is implemented entirely within that one kernel
	- Operating system virtualization
- `libcontainer` provides a short definition of a container
	- A container is a self-contained execution environment that shares the kernel of the host system and which is isolated from other containers in the system
- Containers are best thought of as an OS-specific technology

**History of Containers**
- Docker is a newer technology
- Conceptually docker is the evolution of containers from a simple system that was added to the Unix kernel in 1970s
- And was developed as modern containers used in many huge internet firms
- Containers are a a way to isolate and encapsulate a part of the running system
- First batch processing system
	- `chroot` restricts a process's view of the underlying filesystem to a single subtree
	- Protects the OS from untrusted server processes like FTP, BIND, and Sendmail

- 1980s and 1990s
	- Unix variants were creates with mandatory access control
	- Sidewinder firewall built on top of BSDI Unix

- 2000
	- FreeBSD 4.0
		- `jail` was designed to allow shared-environment hosting providers to easily and securely create a separation between their processes and those of individual customers

- 2005
	- Solaris Containers, which evolved into Solaris Zones
	- First major commercial implementation of container technology
	- OpenVZ for Linux was released
- 2008
	- Linux Containers (LXC)

## Creating a Container

- `docker run` is a command that wraps two separate steps into one
	- Creates a container from the underlying image
		- `docker create`
	- Executes the container
		- `docker start`

- `-p`
	- Map network ports in the underlying container to the host
- `-e`
	- Used to pass environment variables into the container

### Basic Configuration

**Container Name**
- Docker randomly names container by combining an adjective with the name of a famour person
- Use `--name` to specify name

```c
docker create --name="awesome-service" ubuntu:latest sleep 120
```

- Have only one container with a given name on a Docker host

**Labels**
- Labels are key/value pairs that can be applied to Docker images and containers as metadata
- Automatically inherit all the labels from their parent image

```c
docker run -d --name has-some-labels -l deployer=Ahmed -l tester=Asako \
  ubuntu:latest sleep 1000
  
docker ps -a -f label=deployer=Ahmed

docker inspect 845731631ba4
```

**Hostname**
- Docker copies certain system files on the host
- Uses a bind mount to link that copy of the file into the container

```c
docker run --rm -ti ubuntu:latest /bin/bash
```

- `--rm` argument tells Docker to delete the container when it exits
- `-t` allocates a pseudo-TTY
- `-t` sets an interactive session and to keep STDIN open

- Run the `mount` command within the resulting container

**Domain Name Service**

- Configured DNS resolution is managed via a bind mount between the host and container
- By default, this is an exact copy of the Docker host's `resolv.conf` file
- Use `--dns` and `--dns-search` to override

```c
/dev/sda9 on /etc/resolv.conf type ext4 (rw,relatime,data=ordered)

docker run --rm -ti --dns=8.8.8.8 --dns=8.8.4.4 --dns-search=example1.com \
```

**MAC Address**
- Media Access Control (MAC) address for the container
- A container will receive a calculated MAC address that starts with the 02:42:ac:11 prefix

```c
docker run --rm -ti --mac-address="a2:11:aa:22:bb:33" ubuntu:latest /bin/bash
```

- Possible to cause ARP contention on network if two systems advertise the same MAC address

**Storage Volumes**
- Mounting storage from the Docker host is not generally advisable
	- Ties container to a persistent state
- Use `-v` to mount directories and individual files from the host server into the container

```c
docker run --rm -ti -v /mnt/session_data:/data ubuntu:latest /bin/bash
```

- Volumes are mounted read-write, but can modify by
`
```c
docker run --rm -ti -v /mnt/session_data:/data:ro \
  ubuntu:latest /bin/bash
```

#### SELinux and Volume Mounts

- Use `z` option for mounting volumes to override permission error
	- `z` indicates bind mount content is shared among multiple containers


## Starting a Container

## Auto-Restarting a Container

## Stopping a Container

## Killing a Container

## Pausing and Unpausing a Container

## Cleaning Up Containers and Images

## Windows Containers

## Wrap-Up