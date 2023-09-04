# Makefile
# IMPORTANT: Please find the right cuda dev container for your environment
SHELL := /bin/bash

BASE_IMG=nvidia/cuda:11.8.0-devel-ubuntu20.04

# USER INPUT (TODO: PLEASE MODIFY)
CODE_PATH := /home/mc76728/repos/Video-to-Automaton/

# Custom Image
MY_DOCKER_IMG := ${user}ns_vfs
TAG := latest

pull_docker_image:
	docker pull ${BASE_IMG}

build_docker_image:
	docker build --build-arg BASE_IMG=${BASE_IMG} . -f docker/Dockerfile --network=host --tag ${MY_DOCKER_IMG}:${TAG}

run_docker_container:
	docker run --interactive \
			   --detach \
			   --tty \
			   --name ${MY_DOCKER_IMG} \
			   --cap-add=SYS_PTRACE \
			   --ulimit core=0:0 \
			   --volume ${CODE_PATH}:/opt/Neuro-Symbolic-Video-Frame-Search \
			   ${MY_DOCKER_IMG}:${TAG} \
			   /bin/bash

run_docker_container_gpu:
	docker run --interactive \
			   --detach \
			   --tty \
			   --name ${MY_DOCKER_IMG} \
			   --gpus=all \
			   --runtime=nvidia \
			   --cap-add=SYS_PTRACE \
			   --ulimit core=0:0 \
			   --volume ${CODE_PATH}:/opt/Neuro-Symbolic-Video-Frame-Search \
			   ${MY_DOCKER_IMG}:${TAG} \
			   /bin/bashd

exec_docker_container:
	docker exec -it ${MY_DOCKER_IMG} /bin/bash

stop_docker_container:
	docker stop $(MY_DOCKER_IMG)
	docker rm $(MY_DOCKER_IMG)