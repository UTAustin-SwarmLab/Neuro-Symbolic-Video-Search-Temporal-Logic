ARG BASE_IMG=nvidia/cuda:11.8.0-devel-ubuntu20.04

FROM $BASE_IMG as dev

USER root
#----------------------------------------------#
# Install Default System Dependencies for Debian
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y ffmpeg git vim libglew-dev libglfw3-dev libglm-dev tmux wget unzip unrar curl x11-xserver-utils xvfb make cmake 

# RUN apt-get apt-get install -y \
#         gcc g++ 
#----------------------------------------------#
# Install project dependent system dependencies

ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

RUN apt-get install -y python3-pip python3.8-venv build-essential libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev software-properties-common 

# FOR GROUNDING DINO
RUN apt-get install -y \
    rustc cargo 
    
#---------------------DO NOT CHANGE BELOW---------------------#
# STORMPY PACKAGE INSTALLATION
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt install -y g++-11 \
    && apt-get clean
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-11
# GIT CLONE
WORKDIR /opt
RUN git clone -b stable https://github.com/moves-rwth/carl-storm.git
RUN git clone -b stable https://github.com/moves-rwth/storm.git
RUN git clone -b master https://github.com/moves-rwth/pycarl.git
RUN git clone -b stable https://github.com/moves-rwth/stormpy.git
# INSTALL CARL STORM
WORKDIR /opt/carl-storm
RUN mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CLN_NUMBERS=ON -DUSE_GINAC=ON -DTHREAD_SAFE=ON 
WORKDIR /opt/carl-storm/build
RUN make lib_carl -j 1
# INSTALL STORM
WORKDIR /opt/storm
RUN mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release -DSTORM_DEVELOPER=OFF -DSTORM_LOG_DISABLE_DEBUG=ON -DSTORM_PORTABLE=ON -DSTORM_USE_SPOT_SHIPPED=ON 
WORKDIR /opt/storm/build
RUN make resources -j 1 && make storm -j 1 && make binaries -j 1
RUN export PATH=$PATH:/opt/storm/build/bin
# INSTALL PYCARL
WORKDIR /opt/pycarl
RUN python3 -m pip install -ve .
# INSTALL STORMPY
WORKDIR /opt/stormpy
RUN python3 -m pip install -ve .
#---------------------DO NOT CHANGE ABOVE---------------------#
# PYTHON TORCH
RUN python3 -m pip install --upgrade pip build
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
#----------------------------------------------#
WORKDIR /opt/Neuro-Symbolic-Video-Frame-Search