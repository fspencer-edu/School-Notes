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
	- Replaces the `ENTRYPOINT` section

```python
$ docker run -i -t --init alpine:3.6 sh
/ # ps -ef

PID   USER     TIME   COMMAND
    1 root       0:00 /dev/init -- sh
    5 root       0:00 sh
    6 root       0:00 ps -ef

/ # exit
```

- `--init` creates a PID process `/dev/init`

## Network Inspection

- Docker containers can be connected to the network in a number of ways
- Containers are usually connected to the network via the default bridge network that Docker creates
	- Virtual network where the host is the gateway to the rest of the world
- `docker network ls`

```python
$ docker network ls

NETWORK ID          NAME                DRIVER              SCOPE
a4ea6aeb7503        bridge              bridge              local
b5b5fa6889b1        host                host                local
08b8b30a20da        none                null                local
```

- The default bridge network
	- Host
		- For any containers running in `host` network mode
		- Containers share the same network namespace as the host
	- None
		- Disables network access entirely for the container
- `docker-compose`
	- This may create additional networks with different names
- `docker network inspect`
	- Containers attached to networks

```json
$ docker network inspect bridge

[
    {
        "Name": "bridge",
        ...
        "Driver": "bridge",
        "EnableIPv6": false,
        ...
        "Containers": {
            "6a0f439...9a9c3": {
                "Name": "inspiring_johnson",
                ...
                "IPv4Address": "172.17.0.2/16",
                "IPv6Address": ""
            },
            "8720cc2...e91b5": {
                "Name": "zealous_keller",
                ...
                "IPv4Address": "172.17.0.3/16",
                "IPv6Address": ""
            }
        },
        "Options": {
            "com.docker.network.bridge.default_bridge": "true",
            ...
            "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
            "com.docker.network.bridge.name": "docker0",
            ...
        },
        "Labels": {}
    }
]
```

- Bridge network
	- There are two containers attached to `docker0`
	- IP addresses of each container, and the host network they are bound to
- The name and ID are the only references we have in this output that can tie it back to `docker ps` listing
- Containers will normally have their own network stack and IP address
- Since containers have their own network and addresses, they will not show up in the local `netstat`

```python
$ sudo netstat -an

Active Internet connections (servers and established)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
tcp        0      0 10.0.3.1:53             0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN
tcp6       0      0 :::23235                :::*                    LISTEN
tcp6       0      0 :::2375                 :::*                    LISTEN
tcp6       0      0 :::4243                 :::*                    LISTEN
tcp6       0      0 fe80::389a:46ff:fe92:53 :::*                    LISTEN
tcp6       0      0 :::22                   :::*                    LISTEN
udp        0      0 10.0.3.1:53             0.0.0.0:*
udp        0      0 0.0.0.0:67              0.0.0.0:*
udp        0      0 0.0.0.0:68              0.0.0.0:*
udp6       0      0 fe80::389a:46ff:fe92:53 :::*
```

- `netstat -an`
	- Shows ports that are mapped to containers
	- All of the interfaces we are listening to
- Our container is bound to port 23235 on IP address 0.0.0.0

```python
$ netstat -anp

Active Internet connections (servers and established)
Proto ... Local Address           Foreign Address State  PID/Program name
tcp   ... 10.0.3.1:53             0.0.0.0:*       LISTEN 23861/dnsmasq
tcp   ... 0.0.0.0:22              0.0.0.0:*       LISTEN 902/sshd
tcp6  ... :::23235                :::*            LISTEN 24053/docker-proxy
tcp6  ... :::2375                 :::*            LISTEN 954/docker
tcp6  ... :::4243                 :::*            LISTEN 954/docker
tcp6  ... fe80::389a:46ff:fe92:53 :::*            LISTEN 23861/dnsmasq
tcp6  ... :::22                   :::*            LISTEN 902/sshd
udp   ... 10.0.3.1:53             0.0.0.0:*              23861/dnsmasq
udp   ... 0.0.0.0:67              0.0.0.0:*              23861/dnsmasq
udp   ... 0.0.0.0:68              0.0.0.0:*              880/dhclient3
udp6  ... fe80::389a:46ff:fe92:53 :::*     
```


- Docker has a proxy written in GO that is between all of the containers and the outside
- When we look at the output we only see the `docker-proxy`

- Other network inspection commands
	- `tcpdump`

## Image History

- `docker history`
	- Shows each layer that exists in the inspected image
	- The sizes of each layer
	- Commands that were used to build it

```python
$ docker history redis:latest

IMAGE        CREATED     CREATED BY                                    SIZE ...
33c26d72bd74 3 weeks ago /bin/sh -c #(nop)  CMD ["redis-server"]       0B
<missing>    3 weeks ago /bin/sh -c #(nop)  EXPOSE 6379/tcp            0B
<missing>    3 weeks ago /bin/sh -c #(nop)  ENTRYPOINT ["docker-entry… 0B
<missing>    3 weeks ago /bin/sh -c #(nop) COPY file:9c29fbe8374a97f9… 344B
<missing>    3 weeks ago /bin/sh -c #(nop) WORKDIR /data               0B
<missing>    3 weeks ago /bin/sh -c #(nop)  VOLUME [/data]             0B
<missing>    3 weeks ago /bin/sh -c mkdir /data && chown redis:redis … 0B
<missing>    3 weeks ago /bin/sh -c set -ex;   buildDeps='   wget    … 24.1MB
<missing>    3 weeks ago /bin/sh -c #(nop)  ENV REDIS_DOWNLOAD_SHA=ff… 0B
<missing>    3 weeks ago /bin/sh -c #(nop)  ENV REDIS_DOWNLOAD_URL=ht… 0B
<missing>    3 weeks ago /bin/sh -c #(nop)  ENV REDIS_VERSION=4.0.8    0B
<missing>    3 weeks ago /bin/sh -c set -ex;   fetchDeps='ca-certific… 3.1MB
<missing>    3 weeks ago /bin/sh -c #(nop)  ENV GOSU_VERSION=1.10      0B
<missing>    3 weeks ago /bin/sh -c groupadd -r redis && useradd -r -… 330kB
<missing>    3 weeks ago /bin/sh -c #(nop)  CMD ["bash"]               0B
<missing>    3 weeks ago /bin/sh -c #(nop) ADD file:a0f72eb6710fe45af… 79.2MB
```

- Layers are listed in order
	- First at the bottom
	- Most recent at the top
- `--no-trunc`
	- View complete command that was used to build each layer


## Inspecting a Container

- `/var/lib/docker/containers`
	- Contains the configuration
	- Contains SHA hashes (containers)
- This directory contains some files that are bind-mounted directly into the container
	- `hosts, resolv.conf, hostname`
- Contains the JSON logs

## Filesystem Inspection

- Docker uses a layered filesystem that allows it to track the changes in any given container
- Dockerized applications can continue to write things into the filesystem
	- 


## Wrap-Up