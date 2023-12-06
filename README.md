# Neuro Symbolic Video Frame Search

## Installation Guide
**Prerequisites** <br>
No need to worry about the prerequisites below if you are using a UT Swarm Lab cluster.

* CUDA Driver: Version 11.8.0 or higher
* Docker: Nvidia Driver
D
**evelopment Environment Setup**

1. Clone this repository.
2. Navigate to the makefile and modify the user input section.

    For example: `CODE_PATH := /home/repos/Neuro-Symbolic-Video-Frame-Search/`

3. Execute `make pull_docker_image`
4. Execute `make build_docker_image`
5. Execute `make run_docker_container_gpu`
6. Execute `make exec_docker_container`
7. Inside the container, navigate to `/opt/Neuro-Symbolic-Video-Frame-Search` 
8. Inside the container, execute `bash install.sh`

**Development Inside the Container** <br>
Enjoy your development environment inside the container!

Please avoid stopping and removing the container, as you will need to reinstall the dependencies. If the container is stopped or removed, repeat steps 5 to 8.

## Dataset
[NuScenes](https://www.nuscenes.org/nuimages#download): A public large-scale dataset for autonomous driving

## Run Script Arugment
```
# Common arguments
--cv_model: ["yolo", "mrcnn", "clip"].
--video_processor: ["regular_video", "tlv_dataset"].
--output_dir: Output directory.

# If you use regular video
--proposition_set: Proposition sets of a video, e.g - "car,bicycle"
--ltl_formula: LTL formula of a video. The format is very sensitive, e.g - 'P>=0.90 ["car" U "bicycle"]'

# Choose carefully
--save_annotation: Only support by Yolo in this version. Default: False.
--manual_confidence_probability: Development Only.
```

## Linear Temporal Logitics
Confiremd Examples:
- 'P{op}probability [F "proposition"]'
- 'P{op}probability [G "proposition"]'
- 'P{op}probability [F "proposition1" | "proposition2"]'
- 'P{op}probability [F "proposition1" & "proposition2"]'