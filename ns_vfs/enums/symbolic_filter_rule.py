from __future__ import annotations

import enum


class SymbolicFilterRule(enum.Enum):
    AVOID_PROPOSITION = "avoid_proposition"
    AND_ASSOCIATED_PROPS = "and_associated_props"
