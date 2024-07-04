from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import SeparatorStyle, conv_templates
from videollava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init

if TYPE_CHECKING:
    from videollava.mm_utils import (
        KeywordsStoppingCriteria as TypeKeywordsStoppingCriteria,
    )

from ns_vfs.model.vision_language._base import VisionLanguageModelBase


class VideoLlava(VisionLanguageModelBase):
    """Video LLAVA model.

    You need to install video LLava from the link below:
    ref: https://github.com/PKU-YuanGroup/Video-LLaVA
    """

    def __init__(
        self,
        device: str,
        model_path: str,
        vision_processor_name: str,
        conversation_mode: str,
        load_8bit: bool = False,
        load_4bit: bool = True,
    ) -> None:
        super().__init__(device)
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.model_name = get_model_name_from_path(model_path)
        self.model_path = model_path
        self.vision_processor_name = vision_processor_name
        self.conversation_module = conv_templates[conversation_mode].copy()
        self.conversation_roles = self.conversation_module.roles
        self.load_model()

    def load_model(self) -> any:
        """Load weight."""
        # model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs
        disable_torch_init()
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_name,
            load_8bit=self.load_8bit,
            load_4bit=self.load_4bit,
            device=self.device,
            cache_dir=str(Path.home()),
        )
        self.vision_processor = self.processor[self.vision_processor_name]

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

    @abc.abstractmethod
    def infer(self, vision_input: any, language_input: str) -> any: ...
