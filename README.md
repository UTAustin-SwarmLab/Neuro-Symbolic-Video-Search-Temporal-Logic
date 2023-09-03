# Neuro Symbolic Video Frame Search

## Installation Guid
**Requirement** <br>
If you have a UT Swarm Lab cluster, don't worry about the requirement below.

* CUDA Driver: >11.8.0
* Docker: Nvidia Driver

**Set up development environment**
1. clone this repo
2. go to make file and change a user input section <br>
e.g: `CODE_PATH := /home/repos/Neuro-Symbolic-Video-Frame-Search/`
3. make pull_docker_image
4. make build_docker_image
5. make run_docker_container_gpu
6. make exec_docker_container
7. (in container) `cd /opt/ns_vfs`
8. (in container) `bash install.sh`

**Enjoy your development in the container** <br>
Please do not stop and rm container. Otherwise, you need to you install dependencies. If your container is stopped or removed, repeat step 5 - 8.