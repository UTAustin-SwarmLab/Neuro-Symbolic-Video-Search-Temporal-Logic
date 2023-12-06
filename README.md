# Neuro Symbolic Video Frame Search

## CV23 Fall Evaluator 
1. Follow the installation guide below.

2. Download data from [GoogleDrive](https://drive.google.com/drive/folders/1_APVcUDID0oEj6m3HVUxWY8bdZnaZWDr?usp=sharing) and put them into `/sample_data`

3. Run script is in `/run_scripts`
    - Example:
    ```
    python3 run_nsvs_tl.py --video_path '/opt/Neuro-Symbolic-Video-Frame-Search/sample_data/benchmark_COCO_ltl_ "bed" & "bottle"_25_2.pkl' --save_annotation True
    ```
    Note: `.pkl` file should be embraced by single quotations - e.g: 'example.pkl'

4. If you want to run on real data use this example script below:
    ```
    python3 run_nsvs_tl.py --cv_model "yolo_clip" --video_processor "regular_video" --save_annotation True --proposition_set "ship_on_the_sea,kissing,man_backhug_woman" --ltl_formula "P>=0.80 [\"ship_on_the_sea\" U \"man_backhug_woman\"]" --video_path "/opt/Neuro-Symbolic-Video-Frame-Search/sample_data/titanic_scene.mp4"
    ```

5. You are only be able to use Yolo with this version of code repo. 

6. If you don't specify the output dir, it will be available in `artifacts/_result`

## Installation Guide
**Prerequisites** <br>
No need to worry about the prerequisites below if you are using a UT Swarm Lab cluster.

* CUDA Driver: Version 11.8.0 or higher
* Docker: Nvidia Driver
**evelopment Environment Setup**

1. Clone this repository.
2. Navigate to the makefile and modify the user input section.

    For example: `CODE_PATH := /home/repos/Neuro-Symbolic-Video-Frame-Search/`

3. Execute `make pull_docker_image`
4. Execute `make build_docker_image`
5. Execute `make run_docker_container_gpu`
    - Note: If you are a developer: `make run_dev_docker_container_gpu`
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