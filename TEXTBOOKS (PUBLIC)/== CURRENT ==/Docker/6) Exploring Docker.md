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
	- Display only logs after a specified RFC 3339 data, Unix timestamp, or GO duration string
- `--tail`
	- Specific the number of lines

- The files backing this logging are on the Docker server

```json
{"log":"2018-02-04 23:58:51,003 INFO success: running.\r\n",
"stream":"stdout",
"time":"2018-02-04T23:58:51.004036238Z"}
```

- The `log` field is exactly what was send to `stdout`
- `stream` field tells what the `stdout` and not `stderr`


### More Advanced Logging

- Docker also supports configurable logging backends
	- `json-file`
	- `syslog`
	- `fluentd`
	- `journald`
	- `gelf`
	- `awslogs`
	- `splunk`
	- `etwlogs`
	- `gcplogs`
	- `logentries`

- The `daemon.json` file is the configuration for the `dockerd` server
	- `/etc/docker/` directory
- Docker supports only one configuration at a time
- Most Linux systems have some kind of syslog received
- This protocol in its various forms is supported by most deployments
- New Linux distributions are based on
	- `systemd` init system
	- `journald` for logging

- Most logging plug-in are blocking by default
- Logging back-pressure can cause issues with the applications
- Change this with `--log-opt mode=non-blocking`
- Set a max buffer size for logs to something like `--log-opt max-buffer-size=4m`

### Non-Plug-In Community Options

- Log directly from application
- Have a process manager in the container relay the logs
	- `systemd`
	- `upstart`
	- `supervisor`
	- `runit`
- Run a logging relay in the container that wraps `stdout/stderr` from the container
- Relay the Docker JSON logs to a remote logging framework from the server or another container
- This options hid logs from `docker logs`
- Spotify uses a statically linked Go relay to handle logging `stderr` and `stdout` to syslogs for one process inside the container
	- No dependencies
- `svlogd` daemon from the `runit` init system can collect logs from the process's `stdout` and ship them to remote hosts over UDP
- Logspout
	- Runs in a separate container
	- Takes to the Docker daemon, and logs all of the system's container logs to syslog
	- Does not preclude `dovker logs`
	- Requires log rotation
	- Not block starting containers

## Monitoring Docker

- `docker stats`
- `docker events`


### Container Stats

- `docker stats` is similar to the Linux `top` command
- takes over the terminal and updates the same lines

```python
$ docker stats b668353c3af5
CONTAINER     CPU %    MEM USAGE/LIMIT    MEM %  NET I/O    BLK I/O    PIDS
b668353c3af5  1.50%    60.39MiB/200MiB  30.19% 335MB/9.1GB  45.4MB/0B  17
```

- Container ID
- CPU usage
- Memory usage, and max
- Network and block IO stats
- Active processes inside the container

- A common problem with running production containers is that memory limits can cause the kernel OOM
- IO stats


### Stats API endpoint

- Most Docker daemons will be installed with the API available only on the Unix domain socket and not published on TCP
- Use `curl` from the host to call the API
- Expose the Docker API on the TCP port, over SSL to monitor endpoint in production

```python
$ docker run -d ubuntu:latest sleep 1000
91c86ec7b33f37da9917d2f67177ebfaa3a95a78796e33139e1b7561dc4f244a

$ curl --unix-socket /var/run/docker.sock \
    http://v1/containers/91c86ec7b33f/stats | head -1
```

### Container Health Checks

- Health checks are a build-time configuration item and are created with a check definition in the Dockerfile

```python
$ git clone https://github.com/spkane/rocketchat-hubot-demo.git \
    --config core.autocrlf=input
$ cd rocketchat-hubot-demo/mongodb/docker

# Dockerfile
FROM mongo:3.2

COPY docker-healthcheck /usr/local/bin/

HEALTHCHECK CMD ["docker-healthcheck"]
```

- Healthcheck
	- Starting
	- Healthy
	- Unhealthy

- Healthcheck configuration
	- `--health-interval`
	- `--health-retries`
	- `--no-healthcheck`

### Docker Events

- `dockerd` daemon internally generates an events stream around the container lifecycle
- Use stream to monitor scenarios or in triggering additional actions

```python
$ docker events
2018-02-18T14:00:39-08:00 1b3295bf300f: (from 0415448f2cc2) die

2018-02-18T14:00:39-08:00 1b3295bf300f: (from 0415448f2cc2) stop

2018-02-18T14:00:42-08:00 1b3295bf300f: (from 0415448f2cc2) start
```

### cAdvisor

- Graph implementation
	- DataDog
	- GroundWork
	- New Relic
	- Prometheus
	- Nagios

```python
# cAdvisor
$ docker run \
    --volume=/:/rootfs:ro \
    --volume=/var/run:/var/run:rw \
    --volume=/sys:/sys:ro \
    --volume=/var/lib/docker/:/var/lib/docker:ro \
    --publish=8080:8080 \
    --detach=true \
    --name=cadvisor \
    google/cadvisor:latest

Unable to find image 'google/cadvisor:latest' locally
Pulling repository google/cadvisor
f0643dafd7f5: Download complete
...
ba9b663a8908: Download complete
Status: Downloaded newer image for google/cadvisor:latest
f54e6bc0469f60fd74ddf30770039f1a7aa36a5eda6ef5100cddd9ad5fda350b
```

- See web interface of Docker host on port 8080

<img src="/images/Pasted image 20260310191058.png" alt="image" width="500">

- cAdvisor provides a REST API endpoint, which can be used to monitor systems

```json
$ curl http://172.17.42.10:8080/api/v1.3/containers/
{
  "name": "/",
  "subcontainers": [
    {
      "name": "/docker"
    }
  ],
  "spec": {
    "creation_time": "2015-04-05T00:05:40.249999996Z",
    "has_cpu": true,
    "cpu": {
      "limit": 1024,
      "max_limit": 0,
      "mask": "0-7"
    },
```

## Prometheus Monitoring

- Used to monitor distributed systems
- Reaches out and gathers statistics from endpoints on a timed basis
- Reconfigure the `dockerd` server to enable the experimental features
- Expose metrics to listener on a specific port
	- `--experimental` and `--metrics-addr=`

```python
daemon.json
{
  "experimental": true,
  "metrics-addr": "0.0.0.0:9323"
}

# restart server and test endpoint
$ systemctl restart docker
$ curl -s http://localhost:9323/metrics | head -15
# HELP builder_builds_failed_total Number of failed image builds
# TYPE builder_builds_failed_total counter
```

- Write a config from Prometheus to run in a container

```yaml
# Scrape metrics every 5 seconds and name the monitor 'stats-monitor'
global:
  scrape_interval: 5s
  external_labels:
    monitor: 'stats-monitor'

# We're going to name our job 'DockerStats' and we'll connect to the docker0
# bridge address to get the stats. If your docker0 has a different IP address
# then use that instead. 127.0.0.1 and localhost will not work.
scrape_configs:
  - job_name: 'DockerStats'
    static_configs:
    - targets: ['172.17.0.1:9323']
      
$ docker run -d -p 9090:9090 \
    -v /tmp/prometheus/prometheus.yaml:/etc/prometheus.yaml \
    prom/prometheus --config.file=/etc/prometheus.yaml
```

- This will run the container and volume mount config file

<img src="/images/Pasted image 20260310191603.png" alt="image" width="500">

- Advanced dashboards
	- DockProm

## Exploration

- `docker cp`
	- Copying files in and out of the container
- `docker export`
	- Saving a container's filesystem to a tarball
- `docker save`
	- Saving an image to a tarball
- `docker import`
	- Loading an image from a tarbal

## Wrap-Up