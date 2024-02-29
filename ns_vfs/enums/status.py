from __future__ import annotations

import enum


class Status(enum.Enum):
    UNKNOWN = 0
    SUCCESS = 1
    RUNNING = 2
    FAILURE = 3
