import math

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """Builds a transformation pipeline for the given input size."""
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def assign_device_map(model_name, manual_gpu_id=0):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    for layer_idx in range(num_layers):
        device_map[f"language_model.model.layers.{layer_idx}"] = manual_gpu_id

    device_map["vision_model"] = manual_gpu_id
    device_map["mlp1"] = manual_gpu_id
    device_map["language_model.model.tok_embeddings"] = manual_gpu_id
    device_map["language_model.model.embed_tokens"] = manual_gpu_id
    device_map["language_model.output"] = manual_gpu_id
    device_map["language_model.model.norm"] = manual_gpu_id
    device_map["language_model.lm_head"] = manual_gpu_id
    device_map[f"language_model.model.layers.{num_layers - 1}"] = manual_gpu_id

    return device_map


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def move_tensors_to_gpu(module):
    for name, tensor in module.named_buffers():
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
            module.register_buffer(name, tensor.cuda(), persistent=False)
    for _, param in module.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.cuda()


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video_from_seq_of_frames(
    seq_of_frames: list[np.ndarray],
    input_size=448,
    max_num=1,
    device="cuda",
    dtype=torch.bfloat16,
):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for img in seq_of_frames:
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values).to(dtype=dtype, device=device)  # Convert to bfloat16
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values.to(torch.bfloat16))
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list
