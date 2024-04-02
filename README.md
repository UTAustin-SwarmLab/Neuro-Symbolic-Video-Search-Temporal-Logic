# Neuro Symbolic Video Understanding

## Installation Guide
**Prerequisites**  
No need to worry about the prerequisites below if you are using a UT Swarm Lab cluster.

* CUDA Driver: Version 11.8.0 or higher
* Docker: Nvidia Driver
**Development Environment Setup**

1. Clone this repository.
2. Navigate to the makefile and modify the user input section.
    * For example: `CODE_PATH := /home/repos/Neuro-Symbolic-Video-Frame-Search/`
3. Execute `make pull_docker_image`
4. Execute `make build_docker_image`
5. Execute `make run_docker_container_gpu`
    * Note: If you are a developer: `make run_dev_docker_container_gpu`
6. Execute `make exec_docker_container`
7. Inside the container, navigate to `/opt/Neuro-Symbolic-Video-Frame-Search`
8. Inside the container, execute `bash install.sh`

**Development Inside the Container**
Enjoy your development environment inside the container!

Please avoid stopping and removing the container, as you will need to reinstall the dependencies. If the container is stopped or removed, repeat steps 5 to 8.

## Artifact Directories

1. [Sandeep's Slide]([text](https://docs.google.com/presentation/d/1KkK9HrhHkLzU3cw1VlMGn-zQ5o8GZf4fjPkle2XDZcM/edit#slide=id.g2a282c69453_0_89))
