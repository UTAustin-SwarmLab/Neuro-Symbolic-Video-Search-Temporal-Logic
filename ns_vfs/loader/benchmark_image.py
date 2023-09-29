from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ns_vfs.data.frame import BenchmarkRawImage, BenchmarkRawImageDataset

from ._base import DataLoader

import torchvision

from torchvision import datasets
import json

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


class ImageNetDataloader(BenchmarkImageLoader):
    def __init__(
        self,
        imagenet_dir_path: str,
        batch_id: int | str = 1,
    ):
        # Create an imagenet dataset
        self.imagenet = datasets.ImageFolder(imagenet_dir_path)

        # Get text labels from metadata
        self.class_labels_dict = json.load(open("imagenet_class_index.txt"))
        self.class_labels = list(self.class_labels_dict.values())
        self.data: BenchmarkRawImageDataset = self.process_data(
            raw_data=self.load_data()
        )

    def load_data(self) -> dict:
        """Load the labels of the data
        Returns:
        any: dataset
        """
        labels = [self.imagenet[idx][1] for idx in range(len(self.imagenet))]
        mapped_labels = list(map(lambda x: self.class_labels[x],labels))
        data = {
            'dataset': self.imagenet,
            'labels': mapped_labels
        }
        return data
    
    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""
        return BenchmarkRawImageDataset(unique_labels=self.class_labels, 
                                        labels=raw_data['labels'],
                                        dataset=raw_data['dataset'])
