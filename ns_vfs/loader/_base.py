from __future__ import annotations

import abc


class DataLoader(abc.ABC):
    """Data Loader Abstract Class."""

    @abc.abstractmethod
    def load_data(self, image_path) -> any:
        """Load raw image from image_path."""
