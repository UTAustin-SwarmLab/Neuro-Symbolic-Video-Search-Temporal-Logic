from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ns_vfs.data.frame import BenchmarkRawImage
from ns_vfs.loader import LABEL_OF_INTEREST

from ._base import BenchmarkImageLoader


class Cifar10ImageLoader(BenchmarkImageLoader):
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
        self._original_class_labels = [
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
        self.class_labels = [label for label in self._original_class_labels if label in LABEL_OF_INTEREST]
        self.data: BenchmarkRawImage = self.process_data(
            raw_data=self.load_data(data_path=self._cifar_dir_path / self._batch_id)
        )

    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""
        class_labels = list(map(lambda x: [self._original_class_labels[x]], raw_data[b"labels"]))
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


class Cifar100ImageLoader(BenchmarkImageLoader):
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
        self.name = "CIFAR100"
        self._cifar_dir_path = Path(cifar_dir_path) / "train"
        self._raw_data = self.load_data(data_path=self._cifar_dir_path)
        self._original_class_labels = self.get_original_class_labels()
        self.class_labels = self.class_labels = [
            label for label in self._original_class_labels if label in LABEL_OF_INTEREST
        ]  # YOLO labels

        self.data: BenchmarkRawImage = self.process_data(raw_data=self._raw_data)

    def get_original_class_labels(self):
        meta_file = f"{self._cifar_dir_path.parent}/meta"
        meta_data = self.unpickle(meta_file)
        raw = meta_data[b"fine_label_names"]
        return [x.decode("utf-8") for x in raw]

    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""
        class_labels = list(map(lambda x: [self._original_class_labels[x]], raw_data[b"fine_labels"]))
        # class_labels:
        # image_idx_1: "label"
        # image_idx_2: "label"
        data = np.array(raw_data[b"data"])
        data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
        plt.imshow(data[5])
        plt.savefig("data_loader_sample_image.png")
        return BenchmarkRawImage(unique_labels=self.class_labels, labels=class_labels, images=data)

    def load_data(self, data_path) -> dict:
        """Load CIFAR image data from file.

        Args:
        data_path (str): Path to CIFAR image file.
        Your cifar_dir_path must be the same as official CIFAR dataset.
        load_all_batch (bool, optional): Whether to load all batches. Defaults to False.

        Returns:
        any: CIFAR image data.
        """
        return self.unpickle(file=data_path)

    def unpickle(self, file):
        """Unpickle CIFAR image data."""
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict
