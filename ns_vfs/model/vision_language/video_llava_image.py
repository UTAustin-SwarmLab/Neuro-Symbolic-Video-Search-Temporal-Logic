from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import SeparatorStyle
from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

if TYPE_CHECKING:
    from PIL.Image import Image as TypeImage
    from videollava.mm_utils import (
        KeywordsStoppingCriteria as TypeKeywordsStoppingCriteria,
    )

from ns_vfs.model.vision_language.video_llava import VideoLlava


class VideoLlavaImage(VideoLlava):
    """Video LLAVA image model."""

    def __init__(
        self,
        device: str,
        model_path: str,
        vision_processor_name: str,
        conversation_mode: str,
        load_8bit: bool = False,
        load_4bit: bool = True,
    ) -> None:
        super().__init__(
            device=device,
            model_path=model_path,
            vision_processor_name=vision_processor_name,
            conversation_mode=conversation_mode,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
        )

    def process_language_input(
        self, language_input: str
    ) -> tuple[torch.Tensor, TypeKeywordsStoppingCriteria]:
        conversation_input = DEFAULT_IMAGE_TOKEN + "\n" + language_input
        self.conversation_module.append_message(
            self.conversation_roles[0], conversation_input
        )
        self.conversation_module.append_message(
            self.conversation_roles[1], None
        )
        prompt = self.conversation_module.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = (
            self.conversation_module.sep
            if self.conversation_module.sep_style != SeparatorStyle.TWO
            else self.conversation_module.sep2
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        return input_ids, stopping_criteria

    def infer(self, vision_input: str | TypeImage, language_input: str) -> any:
        """Infer."""
        image_tensor = self.convert_vision_input_to_tensor(vision_input)
        input_ids, stopping_criteria = self.process_language_input(
            language_input
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=2024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :]
        ).strip()

        breakpoint()
        return outputs

    def convert_vision_input_to_tensor(
        self, vision_input: str | TypeImage
    ) -> torch.Tensor:
        """Convert image to tensor."""
        vision_tensor = self.vision_processor(
            vision_input, return_tensors="pt"
        )["pixel_values"]
        if isinstance(vision_tensor, list):
            vision_tensor = [
                image.to(self.model.device, dtype=torch.float16)
                for image in vision_tensor
            ]
        else:
            vision_tensor = vision_tensor.to(
                self.model.device, dtype=torch.float16
            )
        return vision_tensor


if __name__ == "__main__":
    video_llava_image = VideoLlavaImage(
        device="cuda:0",
        model_path="LanguageBind/Video-LLaVA-7B",
        vision_processor_name="image",
        conversation_mode="llava_v1",
    )

    video_llava_image.infer(
        vision_input=Image.open(
            "/opt/Neuro-Symbolic-Video-Frame-Search/sample_data/IMG_7977.png"
        ).convert("RGB"),
        language_input="'What is unusual about this image?'",
    )
