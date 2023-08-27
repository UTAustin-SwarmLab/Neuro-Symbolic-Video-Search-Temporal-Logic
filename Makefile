# Makefile
# IMPORTANT: Please find the right cuda dev container for your environment
SHELL := /bin/bash
BASE_IMG=nvidia/cuda:12.2.0-devel-ubuntu20.04

# USER INPUT (PLEASE MODIFY)
CODE_PATH := /Users/minkyuchoi/repos/SwarmLab/Video-to-Automaton


# Custom Image
MY_DOCKER_IMG := ${user}video_to_automaton
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
			   --volume ${CODE_PATH}:/opt/Video-to-Automoton \
			   ${MY_DOCKER_IMG}:${TAG} \
			   /bin/bash

stop_docker_container:
	docker stop $(MY_DOCKER_IMG)
	docker rm $(MY_DOCKER_IMG)