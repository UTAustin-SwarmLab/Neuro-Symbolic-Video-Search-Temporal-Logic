# Makefile
# IMPORTANT: Please find the right cuda dev container for your environment
SHELL := /bin/bash

BASE_IMG=syzygianinfern0/demo

# USER INPUT (TODO: PLEASE MODIFY)
# e.g /home/repos/Neuro-Symbolic-Video-Frame-Search
# No space at the end
CODE_PATH :=<YOUR PATH> #e.g /home/repos/Neuro-Symbolic-Video-Frame-Search

# IF YOU WANT TO MOUNT A DATASTORE -> YOU MUST USE DEV MODE.
DS_PATH :=<YOUR PATH>

# Custom Image
DOCKER_IMG := nsv_pkgs
MY_DOCKER_IMG := $(shell echo ${USER} | sed 's/[@.]/_/g')_${DOCKER_IMG}
TAG := latest

pull_docker_image:
	docker pull ${BASE_IMG}

build_docker_image:
	docker build --build-arg BASE_IMG=${BASE_IMG} . -f docker/Dockerfile --network=host --tag ${DOCKER_IMG}:${TAG}

run_dev_docker_container:
	docker run --interactive \
			   --network=host \
			   --detach \
			   --tty \
			   --name ${MY_DOCKER_IMG} \
			   --cap-add=SYS_PTRACE \
			   --ulimit core=0:0 \
			   --volume ${CODE_PATH}:/opt/nsv_pkgs \
			   --volume ${DS_PATH}:/opt/nsv_pkgs/store \
			   ${BASE_IMG}:${TAG} \
			   /bin/bash

run_dev_docker_container_gpu:
	docker run --interactive \
			   --network=host \
			   --detach \
			   --tty \
			   --name ${MY_DOCKER_IMG} \
			   --gpus=all \
			   --runtime=nvidia \
			   --cap-add=SYS_PTRACE \
			   --ulimit core=0:0 \
			   --volume ${CODE_PATH}:/opt/nsv_pkgs \
			   --volume ${DS_PATH}:/opt/nsv_pkgs/store \
			   ${BASE_IMG}:${TAG} \
			   /bin/bash

run_docker_container_gpu:
	docker run --interactive \
			   --network=host \
			   --detach \
			   --tty \
			   --name ${MY_DOCKER_IMG} \
			   --gpus=all \
			   --runtime=nvidia \
			   --cap-add=SYS_PTRACE \
			   --ulimit core=0:0 \
			   --volume ${CODE_PATH}:/opt/nsv_pkgs \
			   --volume ${DS_PATH}:/opt/nsv_pkgs/store \
			   ${BASE_IMG}:${TAG} \
			   /bin/bash

exec_docker_container:
	docker exec -it ${MY_DOCKER_IMG} /bin/bash

stop_docker_container:
	docker stop $(MY_DOCKER_IMG)
	docker rm $(MY_DOCKER_IMG)