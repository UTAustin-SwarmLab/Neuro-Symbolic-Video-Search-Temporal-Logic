from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from ns_vfs.data.frame import BenchmarkRawImageDataset

from ._base import BenchmarkImageLoader


class ImageNetDS(Dataset):
    """ImageNet Dataset."""

    class_labels_dict: dict
    class_mapping_dict: dict
    class_mapping_dict_number: dict
    mapping_class_to_number: dict

    def __init__(
        self,
        imagenet_dir_path: str,
        type: str = "train",
        batch_id: int | str = 1,
        target_size: tuple = (224, 224, 3),
    ):
        """Load ImageNet dataset from file."""
        mapping_path = imagenet_dir_path + "/LOC_synset_mapping.txt"

        self.class_mapping_dict = {}
        self.class_mapping_dict_number = {}
        self.mapping_class_to_number = {}
        self.mapping_number_to_class = {}
        i = 0

        for line in open(mapping_path):
            self.class_mapping_dict[line[:9].strip()] = line[9:].strip()
            self.class_mapping_dict_number[i] = line[9:].strip()
            self.mapping_class_to_number[line[:9].strip()] = i
            self.mapping_number_to_class[i] = line[:9].strip()
            i += 1

        self.length_dataset = 0

        self.image_path = imagenet_dir_path + "/Data/CLS-LOC/" + type + "/"

        self._num_images_per_class = {}

        for root in self.class_mapping_dict.keys():
            files = os.listdir(self.image_path + root)
            self.length_dataset += len(files)
            self._num_images_per_class[root] = len(files)

        self.target_size = target_size

        print(
            "loaded imagenet dataset ({}) with {} images and {} classes: ".format(
                type, self.length_dataset, len(self.class_mapping_dict.keys())
            )
        )

    def __getitem__(self, index):
        """Get item from dataset."""
        # Find the class ID where the index is located
        class_id = 0
        while index >= self._num_images_per_class[self.mapping_number_to_class[class_id]]:
            index -= self._num_images_per_class[self.mapping_number_to_class[class_id]]
            class_id += 1
        # Find the image ID within the class
        class_name = self.mapping_number_to_class[class_id]
        image_id = os.listdir(self.image_path + class_name)[index]
        # Load the image
        image = plt.imread(self.image_path + class_name + "/" + image_id)
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # Resize
        image = cv2.resize(image, self.target_size[:2])

        return image, class_id

    def __len__(self):
        """Get the length of the dataset."""
        return self.length_dataset

    def class_to_class_number(self, id):
        """Get the class ID from the class name."""
        return self.mapping_class_to_number[id]

    def class_number_to_class(self, id):
        """Get the class name from the class ID."""
        return self.mapping_number_to_class[id]

    def class_number_to_class_name(self, id):
        """Get the class name from the class ID."""
        return self.class_mapping_dict_number[id]

    def class_to_class_name(self, id):
        """Get the class name from the class ID."""
        return self.class_mapping_dict[id]


class ImageNetDataloader(BenchmarkImageLoader):
    """Load ImageNet dataset from file."""

    def __init__(
        self,
        imagenet_dir_path: str,
        batch_id: int | str = 1,
    ):
        """Load ImageNet dataset from file."""
        # Create an imagenet dataset
        self.name = "ImageNet2017-1K"
        self.imagenet = ImageNetDS(imagenet_dir_path)
        # Get text labels from metadata
        self.class_labels = list(self.imagenet.class_mapping_dict.values())
        self.data: BenchmarkRawImageDataset = self.process_data(raw_data=self.load_data())

    def load_data(self) -> dict:
        """Load the labels of the data
        Returns:
        any: dataset.
        """  # noqa: D205
        labels = [0 for _ in range(len(self.imagenet))]
        mapped_labels = [0 for _ in range(len(self.imagenet))]
        cum_count = 0
        for idx, (class_, count) in enumerate(self.imagenet._num_images_per_class.items()):
            cum_count += count
            for j in range(cum_count - count, cum_count):
                labels[j] = idx
                mapped_labels[j] = self.imagenet.class_to_class_name(class_)

        data = {"dataset": self.imagenet, "labels": mapped_labels}
        return data

    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""
        return BenchmarkRawImageDataset(
            unique_labels=self.class_labels, labels=raw_data["labels"], dataset=raw_data["dataset"]
        )
