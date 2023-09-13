from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from ns_vfs.common import omegaconf
from ns_vfs.model.diffusion.pix2pix import PixToPix

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ns_vfs/config/InstructPix2Pix.yaml"
    )
    config = omegaconf.load_config_from_yaml(config_path)

    pix_2_pix = PixToPix(config=config)

    loaded_data = np.load("frame_window_automata.npy", allow_pickle=True).item()

    i = 0

    for key, value in loaded_data.items():
        if value.verification_result == "true":
            plt.imshow(value.frame_image_set[4].frame_image)
            plt.savefig(f"/opt/Neuro-Symbolic-Video-Frame-Search/d{i}_frame_.png")
            dif_img = pix_2_pix.diffuse(value.frame_image_set[4].frame_image)
            plt.imshow(dif_img)
            plt.savefig(f"/opt/Neuro-Symbolic-Video-Frame-Search/d{i}_dif_img.png")
            i += 1

    pix_2_pix.diffuse(input="/opt/instruct-pix2pix/imgs/example.jpg")

    # print(config)
