ARG BASE_IMG=nvidia/cuda:12.2.0-devel-ubuntu20.04

FROM $BASE_IMG as dev

USER root
#----------------------------------------------#
# Install Default System Dependencies for Debian
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg git vim libglew-dev libglfw3-dev libglm-dev \
    tmux wget unzip unrar curl \ 
    x11-xserver-utils xvfb \
    gcc g++ make cmake 

#----------------------------------------------#
# Install project dependent system dependencies
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
RUN apt-get install -y \
    python3-pip python3.8-venv \
    rustc cargo \
    && apt-get clean 

# # Python
RUN python3 -m pip install --upgrade pip build
RUN python3 -m pip install torch

#----------------------------------------------#
WORKDIR /opt/Video-to-Automoton