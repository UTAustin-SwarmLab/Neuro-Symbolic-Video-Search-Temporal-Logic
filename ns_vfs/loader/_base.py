from __future__ import annotations
from ns_vfs.data.frame import BenchmarkRawImage
import abc


class DataLoader(abc.ABC):
    """Data Loader Abstract Class."""

    @abc.abstractmethod
    def load_data(self, image_path) -> any:
        """Load raw image from image_path."""

class BenchmarkImageLoader(DataLoader):
    """Benchmark image loader."""

    class_labels: list
    data: BenchmarkRawImage

    @abc.abstractmethod
    def process_data(self, raw_data) -> any:
        """Process raw data to BenchmarkRawImage Data Class."""
