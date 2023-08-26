import os

from video_to_automaton.common import omegaconf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split(
    "video_to_automaton"
)[0]

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)

config = omegaconf.load_config_from_yaml(config_path)

config.VERSION_AND_PATH.ROOT_PATH = ROOT_DIR
config.VERSION_AND_PATH.ARTIFACTS_PATH = os.path.join(ROOT_DIR, "artifacts")
