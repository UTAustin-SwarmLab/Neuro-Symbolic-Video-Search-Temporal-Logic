from __future__ import annotations

import abc
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from ns_vfs.data.frame import BenchmarkRawImage, BenchmarkRawImageDataset

from ._base import DataLoader


class BenchmarkImageLoader(DataLoader):
    """Benchmark image loader."""

    class_labels: list
    data: BenchmarkRawImage

    @abc.abstractmethod
    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""


class Cifar10ImageLoader(DataLoader):
    """Load CIFAR 10 image data from file."""

    def __init__(
        self,
        cifar_dir_path: str,
        batch_id: int | str = 1,
    ):
        """Load CIFAR image data from file.

        Args:
        cifar_dir_path (str): Path to CIFAR image file.
            Your cifar_dir_path must be the same as official CIFAR dataset.
            website: http://www.cs.toronto.edu/~kriz/cifar.html
        batch_id (int | str, optional): Batch ID. Defaults to 1.
            If "all", load all batches.
        """
        self.name = "CIFAR10"
        self._cifar_dir_path = Path(cifar_dir_path)
        self._batch_id = f"data_batch_{batch_id}"
        self.class_labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.data: BenchmarkRawImage = self.process_data(
            raw_data=self.load_data(data_path=self._cifar_dir_path / self._batch_id)
        )

    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""
        class_labels = list(map(lambda x: self.class_labels[x], raw_data[b"labels"]))
        data = raw_data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        plt.imshow(data[4])
        plt.savefig("test_1.png")
        return BenchmarkRawImage(unique_labels=self.class_labels, labels=class_labels, images=data)

    def load_data(self, data_path, load_all_batch: bool = False) -> dict:
        """Load CIFAR image data from file.

        Args:
        data_path (str): Path to CIFAR image file.
        Your cifar_dir_path must be the same as official CIFAR dataset.
        load_all_batch (bool, optional): Whether to load all batches. Defaults to False.

        Returns:
        any: CIFAR image data.
        """
        if self._batch_id == "data_batch_all" or load_all_batch:
            # TODO: Concatenate all batches
            pass
        else:
            data = self.unpickle(file=data_path)
        return data

    def unpickle(self, file):
        """Unpickle CIFAR image data."""
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict


class ImageNetDS(Dataset):
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
        return self.length_dataset

    def class_to_class_number(self, id):
        return self.mapping_class_to_number[id]

    def class_number_to_class(self, id):
        return self.mapping_number_to_class[id]

    def class_number_to_class_name(self, id):
        return self.class_mapping_dict_number[id]

    def class_to_class_name(self, id):
        return self.class_mapping_dict[id]


class ImageNetDataloader(BenchmarkImageLoader):
    def __init__(
        self,
        imagenet_dir_path: str,
        batch_id: int | str = 1,
    ):
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
        """
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
