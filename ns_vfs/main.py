import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="config",
    config_name="constrained_video_streaming",
    version_base="1.2.0",
)
def main(cfg: DictConfig):
    print(cfg)
    nsvs_system = hydra.utils.instantiate(cfg.node)
    nsvs_system.start()


if __name__ == "__main__":
    main()
