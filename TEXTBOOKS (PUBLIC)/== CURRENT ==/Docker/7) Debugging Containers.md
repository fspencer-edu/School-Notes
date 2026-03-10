- All Docker processes share a kernel, and a filesystem, depending on the container configuration

## Process Output

- `docker top`
	- Top running containers

```python
$ docker top 106ead0d55af

UID        PID    PPID    C  STIME  TTY TIME     CMD
root       4548   1033    0  13:29  ?   00:00:00 /bin/sh -c nginx
root       4592   4548    0  13:29  ?   00:00:00 nginx: master process nginx
www-data   4593   4592    0  13:29  ?   00:00:00 nginx: worker process
```
- To maintain the same user information across platforms, dedicate a non-zero UID to containers
- Create a container user as UID `5000` and then create the same user in the base container images
- Paths are also shown relative to the container and hot the host

```python
$ ps axlfww

...  /usr/bin/dockerd -H fd://
...  \_ docker-containerd -l unix:///var/run/docker/l
...  |   \_ docker-containerd-shim b668353c3af5d62350
...  |   |   \_ /usr/bin/cadvisor -logtostderr
...  |   \_ docker-containerd-shim dd72ecf1c9e4c22bf7
...  |       \_ /bin/s6-svscan /etc/services
...  |           \_ s6-supervise nginx.svc
...  |               \_ ./nginx
...  \_ /usr/bin/docker-proxy -proto tcp -host-ip 0.0
...  \_ /usr/bin/docker-proxy -proto tcp -host-ip 0.0
```

- A Docker daemon is running with two instance of the `docker-proxy`
- There is one `containerd` running
	- Main container runtime inside Docker
- Everything else running are Docker containers and processes
	- `docker-containerd-shim`
	- Once instance of `cadvisor`
	- One instance of `nginx` in another contain being supervised by s6 supervisor

```python
# larger tree output
$ ps -ejH

40643 ...   docker
43689 ...     docker
43697 ...     docker
43702 ...     start
43716 ...       java
46970 ...     docker

# concise tree output
$ pstree `pidof dockerd`

dockerd... docker-containe──cadvisor─┬──15*[{cadvisor}]
       ...         │                 └─9*[{docker-containe}]
       ...         └─docker-containe─┬─s─nginx───13*[{nginx}...]
       ...                           └─9*[{docker-containe}]
       
# PID tree
$ pstree -p `pidof dockerd`

dockerd(4086)... ─┬─dockerd(6529)─┬─{dockerd}(6530)
             ...  │               ├─...
             ...  │               └─{dockerd}(6535)
             ...  ├─...
             ...  ├─mongod(6675)─┬─{mongod}(6737)
             ...  │              ├─...
             ...  │              └─{mongod}(6756)
             ...  ├─redis-server(6537)─┬─{redis-server}(6576)
             ...  │                    └─{redis-server}(6577)
             ...  ├─{dockerd}(4089)
             ...  ├─...
             ...  └─{dockerd}(6738)
```


## Process Inspection

- Inspect running processing
	- `strace`
	- `lsof`

## Controlling Processes

- Unless the top-level process in the container, killing a process will not terminate the container itself
- Containers offer an abstraction that tools interoperate with
- Start by running containers in production containers
- Containers can be treated as normal processes and passed with Linux commands
	- `kill`
- When daemons fork into the background, they become children of PID 1 on Unix systems
- Process 1 is special and is usually an `init` processes
	- This can cause zombie children, since the PID 1 is responsible for initializing the container
- Launch a small init process based on the tini project that will act as PID 1 insides the container on startup
- What ever is specified in the `Dockerfile, CMD` will be passed to `tini`
	- Replaces 

## Network Inspection

## Image History

## Inspecting a Container

## Filesystem Inspection

## Wrap-Up