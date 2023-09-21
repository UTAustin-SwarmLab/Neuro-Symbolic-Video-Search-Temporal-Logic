from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ns_vfs.common import omegaconf
from ns_vfs.model.diffusion.pix2pix import PixToPix

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ns_vfs/config/InstructPix2Pix.yaml"
    )
    artifact_path = Path("/opt/Neuro-Symbolic-Video-Frame-Search/artifacts")
    image_dir = artifact_path / "diffusion_model_debug_output"
    if image_dir.exists() and image_dir.is_dir():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    config = omegaconf.load_config_from_yaml(config_path)

    pix_2_pix = PixToPix(config=config)

    loaded_data = np.load(os.path.join(artifact_path, "frame_window_automata.npy"), allow_pickle=True).item()
    i = 0
    for key, value in loaded_data.items():
        if value.verification_result == "true":
            frame_image = value.frame_image_set[4].frame_image
            plt.imshow(frame_image)
            plt.savefig(f"{image_dir}/idx{i}_original_full_frame.png")
            person_detection = value.frame_image_set[4].object_detection["person"]
            x1, y1, x2, y2 = map(int, person_detection.xyxy[0])
            only_person_image = frame_image[y1:y2, x1:x2]
            plt.imshow(only_person_image)
            plt.savefig(f"{image_dir}/idx{i}_detected_obj_.png")
            dif_img = pix_2_pix.diffuse(only_person_image)
            plt.imshow(dif_img)
            plt.savefig(f"{image_dir}/idx{i}_diffused_detected_obj_.png")
            required_shape = (y2 - y1, x2 - x1, 3)
            dif_array = np.array(dif_img).copy()
            dif_array.resize(required_shape, refcheck=False)
            frame_image[y1:y2, x1:x2] = dif_img.resize((x2 - x1, y2 - y1))
            plt.imshow(frame_image)
            plt.savefig(f"{image_dir}/idx{i}_diffused_full_frame.png")

            i += 1

    pix_2_pix.diffuse(input="/opt/instruct-pix2pix/imgs/example.jpg")

    # print(config)
