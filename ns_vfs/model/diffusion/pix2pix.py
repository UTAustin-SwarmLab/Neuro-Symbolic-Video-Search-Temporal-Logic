import math
import random

import einops
import k_diffusion as K
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast, nn

from ns_vfs.model.diffusion.stable_diffusion.ldm.util import instantiate_from_config

from ._base import Diffusion


class PixToPix(Diffusion):
    def __init__(self, config: OmegaConf):
        self._config = config.general
        self._model_config = config.model
        self._model = self.load_model_from_config(
            self._config, self._config.checkpoint, self._config.vae_ckpt
        )
        self._model.eval().cuda()
        self._model_wrap = K.external.CompVisDenoiser(self._model)
        self._model_wrap_cfg = CFGDenoiser(self._model_wrap)
        self._null_token = self._model.get_learned_conditioning([""])
        self._seed = random.randint(0, 100000) if self._config.seed is None else self._config.seed

    def image_process(self, image):
        if isinstance(image, np.ndarray):
            input_image = Image.fromarray(image)
        else:
            input_image = Image.open(image).convert("RGB")
        width, height = input_image.size
        factor = self._config.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        return ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    def diffuse(self, input: any):
        input_image = self.image_process(input)
        with torch.no_grad(), autocast("cuda"), self._model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self._model.get_learned_conditioning([self._config.edit])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(self._model.device)
            cond["c_concat"] = [self._model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self._null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self._model_wrap.get_sigmas(self._config.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": self._config.cfg_text,
                "image_cfg_scale": self._config.cfg_image,
            }
            torch.manual_seed(self._seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(
                self._model_wrap_cfg, z, sigmas, extra_args=extra_args
            )
            x = self._model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_img = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_img.save("output.jpg")
            return edited_img

    def load_model_from_config(self, config, ckpt, vae_ckpt=None, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        if vae_ckpt is not None:
            print(f"Loading VAE from {vae_ckpt}")
            vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
            sd = {
                k: vae_sd[k[len("first_stage_model.") :]]
                if k.startswith("first_stage_model.")
                else v
                for k, v in sd.items()
            }
        model = instantiate_from_config(self._model_config)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        return model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [
                torch.cat(
                    [
                        cond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                        uncond["c_crossattn"][0],
                    ]
                )
            ],
            "c_concat": [
                torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])
            ],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond
        ).chunk(3)
        return (
            out_uncond
            + text_cfg_scale * (out_cond - out_img_cond)
            + image_cfg_scale * (out_img_cond - out_uncond)
        )
