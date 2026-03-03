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


- If the container application is designed to write into `/data`, then it will be visible to the host filesystem and remain available when container stops
- Containers should be designed to stateless whenever possible

**Resource Quotas**
- Using Docker, leverage cgroup in the Linux kernel to control the resources that are available to Docker container
- `docker create` and `docker run` support configuring CPU, memory, swap, and storage I/O restrictions

**CPU Shares**

- Docker uses cpu shares to limit usages by applications in containers
- Docker assigned the number 1024 to represent the full pool
- Docker image that contains the `stress` command for pushing a system to its limits

```c
docker run --rm -ti progrium/stress \
  --cpu 2 --io 1 --vm 2 --vm-bytes 128M --timeout 120s
```
- Creates 2 CPU-bound processes, one IO process, and two memory allocation processes


**CPU Pinning**
- Pin a container to one or more CPU cores
- `--cpuset` is a zero-indexed, so first CPU core i 0
- Using the CPU FS (Completely Fair Scheduler) within the Linux kernel, you can alter the CPU quota for a given container by setting the `--cpu-quota` flag to a valid value when launching the container with `docker run`

**Simplifying CPU Quotas**

- Tell Docker how much CPU is available to a container
- `--cpus` command can be set to a floating-point number between 0.01 and the number of CPU cores on the Docker server

```c
docker run -d --cpus=".25" progrium/stress \
    --cpu 2 --io 1 --vm 2 --vm-bytes 128M --timeout 60s
```

**Memory**
- While constraining the CPU only impacts the application's priority for CPU time, the memory limit is a hard limit
- Allocate more memory to a container than the system has actual RAM
- Containers when use swapping
- `--memory` adds a memory constraint

```c
docker run --rm -ti --memory 512m progrium/stress \
    --cpu 2 --io 1 --vm 2 --vm-bytes 128M --timeout 120s
```


- `b, k, m, g`, represents bytes, kilobytes, megabytes, or gigabytes
- `--memory-swap` option to disable or enable swap available to the container
	- `-1` to disable
- `--kernel-memory` to limit the amount of kernel memory available to a container

- An out-of-memory container causes the kernel to behave like a system out of memory
	- Exit code 137 for kernel out-of-memory (OOM) in the `dmesg` output

**Block IO**
- Enable `blkio.weight` cgroup attribute
	- Divide all the available IO between every process within a cgroup slice by 1000
- `--device-read-iops` and `--device-write-ios` are the most effective ways to set limits

**ulimits**
- `ulimit` is used to limit the resources available to a process

## Starting a Container

```c
docker create -p 6379:6379 redis:2.8

docker ps -a
CONTAINER ID  IMAGE                   COMMAND               ...
6b785f78b75e  redis:2.8               "/entrypoint.sh redi  ...
92b797f12af1  progrium/stress:latest  "/usr/bin/stress --v  ...

docker start 6b785f78b75e
```

- Identify contain by the image and creation time

## Auto-Restarting a Container

- Some containers are very-short lives
- Tell Docker to manage restarts by passing `--restart` to `docker run`
	- `no, always, on-failure, unless-stopped`

```c
docker run -ti --restart=on-failure:3 --memory 100m progrium/stress \
    --cpu 2 --io 1 --vm 2 --vm-bytes 128M --timeout 120s
    
docker ps
...  IMAGE                   ...  STATUS                                ...
...  progrium/stress:latest  ...  Restarting (1) Less than a second ago ...
```

## Stopping a Container


- When stopping, it exits the container
- `docker pause` and `Docker unpause`

```c
docker stop 6b785f78b75e
docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES

docker ps -a
CONTAINER ID  IMAGE                  STATUS                   ...
6b785f78b75e  progrium/stress:latest Exited (0) 2 minutes ago ...
```

- As long as the container has not been deleted, you can restart it without recreating it
- Containers are a tree of process that interact with the system
- Send Unix signals to process in the containers that can then respond to them
	- `SIGTERM` signal and waiting for the container to exit

## Killing a Container

- Use `docker kill` to terminate container immediately
- Supports sending any Unix signal
	- Reconnect a remote logging sessions

```c
docker kill 6b785f78b75e
6b785f78b75e

docker kill --signal=USR1 6b785f78b75e
6b785f78b75e
```

## Pausing and Unpausing a Container

- Pausing leverages the `cgroups` freezer, which prevents your process from being scheduled until it unfreezes
- Pausing a container does not send any information to the container about its state change

```c
docker pause 6b785f78b75e
```

## Cleaning Up Containers and Images

- List all the container using `docker ps -a` command a delete any container in the list with `docker rm`
- Stop all containers that are using an image before removing the image itself

```c
docker ps -a
CONTAINER ID  IMAGE                   ...
92b797f12af1  progrium/stress:latest  ...
...
docker rm 92b797f12af1

docker images
REPOSITORY       TAG     IMAGE ID      CREATED       VIRTUAL SIZE
ubuntu           latest  5ba9dab47459  3 weeks ago   188.3 MB
redis            2.8     868be653dea3  3 weeks ago   110.7 MB
progrium/stress  latest  873c28292d23  7 months ago  281.8 MB

docker rmi 873c28292d23
```

- To purge all the images or container from system use `docker system prune`
- To remove all unused images use `docker system prune -a`
- Delete all container on host, `docker rm $(docker ps -a -q)`
- Delete all images on docker host, `docker rmi $(docker images -q)`
- Remove all container that exited with non-zero state, `docker rm $(docker ps -a -q --filter 'exited!=0')`
- Remove all untagged images, `docker rmi $(docker images -q -f "dangling=true")`


- Useful to script `docker` commands to run on a schedule
	- `cron` or `systemd` timer

## Windows Containers

- Windows containers are not compatible with the rest of the Docker ecosystem because they require Windows-specific container images
- Run a Windows container on Windows 10 wth Hyper-V and Docker
- 

## Wrap-Up