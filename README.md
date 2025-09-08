<div align="center">

# Neuro Symbolic Video Search with Temporal Logic (NSVS-TL)

[![arXiv](https://img.shields.io/badge/arXiv-2403.11021-b31b1b.svg)](https://arxiv.org/abs/2403.11021) [![Paper](https://img.shields.io/badge/Paper-pdf-green.svg)](https://link.springer.com/chapter/10.1007/978-3-031-73229-4_13) [![Website](https://img.shields.io/badge/ProjectWebpage-nsvs--tl-orange.svg)](https://utaustin-swarmlab.github.io/nsvs/) [![GitHub](https://img.shields.io/badge/Code-Source--Code-blue.svg)](https://github.com/UTAustin-SwarmLab/Neuro-Symbolic-Video-Search-Temporal-Logic) [![GitHub](https://img.shields.io/badge/Code-Dataset-blue.svg)](https://github.com/UTAustin-SwarmLab/Temporal-Logic-Video-Dataset)
</div>

## Abstract

The unprecedented surge in video data production in recent years necessitates efficient tools to extract meaningful frames from videos for downstream tasks. Long-term temporal reasoning is a key desideratum for frame retrieval systems. While state-of-the-art foundation models, like VideoLLaMA and ViCLIP, are proficient in short-term semantic understanding, they surprisingly fail at long-term reasoning across frames. A key reason for this failure is that they intertwine per-frame perception and temporal reasoning into a single deep network. Hence, decoupling but co-designing the semantic understanding and temporal reasoning is essential for efficient scene identification. We propose a system that leverages vision-language models for semantic understanding of individual frames but effectively reasons about the long-term evolution of events using state machines and temporal logic (TL) formulae that inherently capture memory. Our TL-based reasoning improves the F1 score of complex event identification by 9-15% compared to benchmarks that use GPT-4 for reasoning on state-of-the-art self-driving datasets such as Waymo and NuScenes. The source code is available on Github.

## Installation Guide
Ensure you have **CUDA 12.4** installed and available on your system.  
On Linux, you can verify with:
```bash
nvcc --version
```

From the root of the repo, run the following to build all STORM dependencies:
```bash
./build_dependency
```

Next, install uv:
```bash
pip install uv
```

Finally, install everything in `pyproject.toml` to build project dependencies:
```bash
uv sync
```


## Running the System

NSVS can be run in two ways: running it with raw mp4 files and input queries or running it via the TLV dataset.

To run it with mp4 files, modify the mp4 file paths and the natural language search query inside `execute_with_mp4.py` and run it with:
```bash
uv run execute_with_mp4
```

To run it with the TLV dataset, first download the dataset from [GitHub](https://github.com/UTAustin-SwarmLab/Temporal-Logic-Video-Dataset). Then, specify the dataset path in `execute_with_tlv.py` and run the program:
```bash
uv run execute_with_tlv
```


## Connect with Me

<p align="center">
  <em>Feel free to connect with me through these professional channels:</em>
<p align="center">
  <a href="https://www.linkedin.com/in/mchoi07/" target="_blank"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=Linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="mailto:minkyu.choi@utexas.edu"><img src="https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=Gmail&logoColor=white" alt="Email"/></a>
  <a href="https://scholar.google.com/citations?user=ai4daB8AAAAJ&hl" target="_blank"><img src="https://img.shields.io/badge/-Google%20Scholar-4285F4?style=flat-square&logo=google-scholar&logoColor=white" alt="Google Scholar"/></a>
  <a href="https://minkyuchoi-07.github.io" target="_blank"><img src="https://img.shields.io/badge/-Website-00C7B7?style=flat-square&logo=Internet-Explorer&logoColor=white" alt="Website"/></a>
  <a href="https://x.com/MinkyuChoi7" target="_blank"><img src="https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=Twitter&logoColor=white" alt="X"/></a>
</p>

## Citation

If you find this repo useful, please cite our paper:

```bibtex
@inproceedings{choi2024towards,
  title={Towards neuro-symbolic video understanding},
  author={Choi, Minkyu and Goel, Harsh and Omama, Mohammad and Yang, Yunhao and Shah, Sahil and Chinchali, Sandeep},
  booktitle={European Conference on Computer Vision},
  pages={220--236},
  year={2024},
  organization={Springer}
}
```
