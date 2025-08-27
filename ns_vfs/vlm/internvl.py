import gc
import logging

import numpy as np
import copy
import torch
from torch.nn.functional import softmax
from transformers import AutoModel, AutoTokenizer

from ns_vfs.vlm.internvl_utils import (
    assign_device_map,
    load_video_from_seq_of_frames,
    split_model,
)
from ns_vfs.vlm.obj import DetectedObject

class InternVL:
    """InternVL's Vision Language Model."""

    def __init__(
        self,
        model_name: str = "InternVL2-8B",
        multi_gpus: bool = False,
        device: int = 0,
    ) -> None:
        """Initialization the InternVL."""
        logging.info(
            (
                "You are using the model based on HuggingFace API.",
                "The model will be downloaded to the HuggingFace cache dir.",
            )
        )
        self.model_name = model_name
        self._path = f"OpenGVLab/{model_name}"
        self._num_gpus = torch.cuda.device_count()
        self.device = device
        if multi_gpus:
            device_map = split_model(model_name)
        else:
            device_map = assign_device_map(model_name=model_name, manual_gpu_id=device)
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(self._path, trust_remote_code=True, use_fast=False)

    def reset_model(self) -> None:
        """Reset the model to its initial state using pretrained weights."""
        self.model = AutoModel.from_pretrained(
            self._path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()
        self.model.apply(self.move_tensors_to_gpu)

    def clear_gpu_memory(self) -> None:
        """Clear CUDA cache and run garbage collection to free GPU memory."""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        gc.collect()  # Run garbage collector

    def move_tensors_to_gpu(
        self,
        module: torch.nn.Module,
    ) -> None:
        """Move all tensors in the module to GPU if they are on the CPU."""
        for name, tensor in module.named_buffers():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                module.register_buffer(
                    name,
                    tensor.cuda(self.device),
                    persistent=False,
                )
        for _, param in module.named_parameters():
            if param.device.type == "cpu":
                param.data = param.data.cuda(self.device)

    def detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_description: str,
        threshold: float
    ) -> DetectedObject:
        """Detect objects in the given frame image.

        Args:
            seq_of_frames (list[np.ndarray]): List of video frames to process.
            scene_description (str): Description of the scene.
            threshold (float): Detection threshold.

        Returns:
            DetectedObject: Detected objects with their details.
        """
        parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
        prompt = rf"Is there a {scene_description} present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"

        response, confidence = self.infer_with_video_confidence(
            language=prompt,
            seq_of_frames=seq_of_frames,
        )

        detected = "yes" in response.lower()
        probability = self.calibrate(confidence, false_threshold=threshold)

        return DetectedObject(
            name=scene_description,
            is_detected=detected,
            confidence=round(confidence, 3),
            probability=round(probability, 3),
        )

    def infer_with_video_confidence(
        self,
        language: str,
        seq_of_frames: list[np.ndarray],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """Perform video inference and return response with confidence score.

        Args:
            language (str): The input prompt or question.
            seq_of_frames (list[np.ndarray] | None):
                List of video frames as numpy arrays.
            video_path (str | None): Path to the input video file.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling for generation.

        Returns:
            tuple[str, float]: Generated response and confidence score.
        """

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        pixel_values, num_patches_list = load_video_from_seq_of_frames(
            seq_of_frames=seq_of_frames, device=self.device
        )

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        language = video_prefix + language

        return self.chat_with_confidence(
            self.tokenizer,
            pixel_values,
            language,
            generation_config,
            num_patches_list=num_patches_list,
        )

    def chat_with_confidence(
        self,
        tokenizer: AutoTokenizer,
        pixel_values: torch.Tensor,
        question: str,
        generation_config: dict,
        num_patches_list: list[int] | None = None,
        IMG_START_TOKEN: str = "<img>",
        IMG_END_TOKEN: str = "</img>",
        IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
        verbose: bool = False,
    ) -> tuple[str, float]:
        """Generate a response with confidence score for the given input.

        Args:
            tokenizer: The tokenizer to use.
            pixel_values: Image tensor input.
            question: The input question or prompt.
            generation_config: Configuration for text generation.
            num_patches_list: List of number of patches for video frames.
            IMG_START_TOKEN: Token to mark the start of an image.
            IMG_END_TOKEN: Token to mark the end of an image.
            IMG_CONTEXT_TOKEN: Token for image context.
            verbose: Whether to print verbose output.

        Returns:
            A tuple containing the generated response and its confidence score.
        """
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            context_tokens = IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches
            image_tokens = IMG_START_TOKEN + context_tokens + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda(self.device)
        attention_mask = model_inputs["attention_mask"].cuda(self.device)
        generation_config["eos_token_id"] = eos_token_id
        generation_config["return_dict_in_generate"] = True
        generation_config["output_scores"] = True
        generation_config["output_logits"] = True
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()

        logits_to_compute = np.where(generation_output.sequences[0].detach().cpu().numpy() != eos_token_id)[0]
        confidence = 1.0
        for logit in logits_to_compute:
            token = generation_output.sequences[0, logit].item()
            prob = softmax(generation_output.logits[logit], dim=0)[0, token]
            confidence = prob.item() * confidence
        self.clear_gpu_memory()

        return response, confidence

    def calibrate(
        self,
        confidence: float,
        true_threshold=0.95,
        false_threshold=0.40,
        target_conf=0.60,
        target_prob=0.78,
        p_min=0.01,
        p_max=0.99,
        steepness_factor=0.7,
    ) -> float:
        """Map confidence to probability using a sigmoid function with adjustable steepness.

        Args:
            confidence: Input confidence score
            true_threshold: Upper threshold
            false_threshold: Lower threshold
            target_conf: Target confidence point
            target_prob: Target probability value
            p_min: Minimum probability
            p_max: Maximum probability
            steepness_factor: Controls curve steepness (0-1, lower = less steep)
        """
        if confidence <= false_threshold:
            return p_min

        if confidence >= true_threshold:
            return p_max

        # Calculate parameters to ensure target_conf maps to target_prob
        # For a sigmoid function: f(x) = L / (1 + e^(-k(x-x0)))

        # First, normalize the target point
        x_norm = (target_conf - false_threshold) / (true_threshold - false_threshold)
        y_norm = (target_prob - p_min) / (p_max - p_min)

        # Find x0 (midpoint) and k (steepness) to satisfy our target point
        x0 = 0.30  # Midpoint of normalized range

        # Calculate base k value to hit the target point
        base_k = -np.log(1 / y_norm - 1) / (x_norm - x0)

        # Apply steepness factor (lower = less steep)
        k = base_k * steepness_factor

        # With reduced steepness, we need to adjust x0 to still hit the target point
        # Solve for new x0: y = 1/(1+e^(-k(x-x0))) => x0 = x + ln(1/y-1)/k
        adjusted_x0 = x_norm + np.log(1 / y_norm - 1) / k

        # Apply the sigmoid with our calculated parameters
        x_scaled = (confidence - false_threshold) / (true_threshold - false_threshold)
        sigmoid_value = 1 / (1 + np.exp(-k * (x_scaled - adjusted_x0)))

        # Ensure we still hit exactly p_min and p_max at the thresholds
        # by rescaling the output slightly
        min_val = 1 / (1 + np.exp(-k * (0 - adjusted_x0)))
        max_val = 1 / (1 + np.exp(-k * (1 - adjusted_x0)))

        # Normalize the output
        normalized = (sigmoid_value - min_val) / (max_val - min_val)

        return p_min + normalized * (p_max - p_min)

