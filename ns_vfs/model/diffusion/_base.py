from __future__ import annotations

import abc


class Diffusion(abc.ABC):
    """Abstract base class for diffusion models."""

    @abc.abstractmethod
    def diffuse(self, input: any):
        """Diffuse the input."""
