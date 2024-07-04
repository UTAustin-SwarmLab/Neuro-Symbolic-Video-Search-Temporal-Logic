from __future__ import annotations

import abc


class Node(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    def stop(self) -> None: ...
