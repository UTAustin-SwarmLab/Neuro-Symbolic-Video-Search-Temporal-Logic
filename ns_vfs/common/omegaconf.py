from __future__ import annotations
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf

def load_config_from_yaml(config_path: str, read_only=False) -> DictConfig:
    """Load a yaml config file and return a DictConfig object."""
    config = OmegaConf.load(config_path)
    if read_only:
        OmegaConf.set_readonly(config, True)
    return config

def load_config_from_dict(config_dict: dict, read_only=False) -> DictConfig:
    """Load a dictionary and return a DictConfig object."""
    config = OmegaConf.create(config_dict)
    if read_only:
        OmegaConf.set_readonly(config, True)
    return config

def save_config_to_yaml(config: DictConfig, config_path: str) -> None:
    """Save a DictConfig object to a yaml file."""
    with Path(config_path) as fp:
        OmegaConf.save(config=config, f=fp)