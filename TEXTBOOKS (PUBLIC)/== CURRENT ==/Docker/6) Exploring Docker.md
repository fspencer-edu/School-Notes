## Printing the Docker Version

- Sign in with `docker`
- `docker version`

## Server Information

- `docker info`
	- Filesystem backend the Docker server is running
	- Kernel version
	- OS
	- Plug-ins
	- Runtim
	- Containers
	- Images
- `--data-root`
	- Change default root directory to store images and containers
- `runc`
	- Default Docker runtime

## Downloading Image Updates

- `docker pull ubuntu:latest`
	- Latest is the tag

## Inspecting a Container

- `docker run -d -t ubuntu /bin/bash`

```python
$ docker ps
CONTAINER ID  IMAGE         COMMAND     ... STATUS        ...  NAMES
3c4f916619a5  ubuntu:latest "/bin/bash" ... Up 31 seconds
```

- ID => 3c4f916619a5
- The container ID is also given a dynamic name

- `Docker inspect 3c4f916619a5`

```json
[{
    "Id": "3c4f916619a5dfc420396d823b42e8bd30a2f94ab5b0f42f052357a68a67309b",
    "Created": "2018-11-07T22:06:32.229471304Z",
    ...
    "Args": [],
    ...
    "Image": "sha256:ed889b69344b43252027e19b41fb98d36...a9098a6d"
    ...
    "Config": {
        "Hostname": "3c4f916619a5",
        ...
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ],
        "Cmd": [
            "/bin/bash"
        ],
        ...
        "Image": "ubuntu",
        ...
    },
    ...
}]
```

## Exploring the Shell

- `docker run -i -t ubuntu:16.04 /bin/bash`

```python
$ ps -ef
UID        PID  PPID  C STIME TTY          TIME CMD
root         1     0  0 22:12 ?        00:00:00 /bin/bash
root        12     1  0 22:16 ?        00:00:00 ps -ef
```

- Docker containers by default to not start in the background
- Do not start an `init` systems

## Returning a Result

- Docker will redirect its `stdin` to the remove process
- And the remotes processes `stdout` and `stderr` are in the terminal

```python
docker run 8d12decc75fe /bin/false
$ echo $?
1
```

- Docker proxied that result from the container to the local terminal

## Getting Inside a Running Container

### docker exec

- `dockerd` server
- `docker` CLI
	- Support remotely executing a new process via `docker exec`
	- Start a container in the background, then invoke a shall

```python
docker exec -i -t 589f2ad30138 /bin/bash
root@589f2ad30138:/#
```

### nsenter

- `util-linux` package
- Namespace Enter
	- Allows you to enter any Linux namespace
	- Manipulate things in a container as `root` on the server

```python
$ docker run --rm -v /usr/local/bin:/target jpetazzo/nsenter
Unable to find image 'jpetazzo/nsenter' locally
Pulling repository jpetazzo/nsenter
9e4ef84f476a: Download complete
511136ea3c5a: Download complete
71d9d77ae89e: Download complete
Status: Downloaded newer image for jpetazzo/nsenter:latest
Installing nsenter to /target
Installing docker-enter to /target
```

- `-v` exposes the directory into the running container as /target

### docker volumne

- Docker supports a `volume` subcommand that makes is possible to list all of the volumes stored in the root directory and then discovery additional information about time
	- Physically located on the server
- Volumes are not bind-mounted volumes, instead are special data containers that are used for persisting data

```python
$ docker volume ls
DRIVER              VOLUME NAME

$ docker run -d -v /tmp:/tmp ubuntu:latest sleep 120
6fc97c50fb888054e2d01f0a93ab3b3db172b2cd402fc1cd616858b2b5138857

$ docker volume ls
DRIVER              VOLUME NAME
```

- `docker volume create my_data`

- Start a container with this data volume attached

```python
docker run --rm \
	--mount source=my_data, target=/app \
	ubuntu:latest touch /app/my-presistent-data
```

## Logging

- Logging is a critical part of any production application
- Logal logfile
	- `dmesg`
- Linux distribution
	- `systemd`
- Docker captures all of the normal text output from applications in the containers it manages
- Docker daemon and streamed into a configurable logging backend
- System is pluggable and there are many options

### docker logs

- `json-file` plug-in method using `docker logs`
- Application's logs are streamed by the Docker daemon into a JSON file for each container

```python
$ docker logs 3c4f916619a5
2017/11/20 00:34:56 [notice] 12#0: using the "epoll" ...
2017/11/20 00:34:56 [notice] 12#0: nginx/1.0.15
2017/11/20 00:34:56 [notice] 12#0: built by gcc 4.4.7 ...
2017/11/20 00:34:56 [notice] 12#0: OS: Linux 3.8.0-35-generic
```

- Use `--since` option to limit the log output
	- Display only logs after a specified RFC 3339 data, Unix timestamp, or GO duration tring
- `--tail`
	- Specific the number of lines




## Monitoring Docker

## Prometheus Monitoring

## Exploration

## Wrap-Up