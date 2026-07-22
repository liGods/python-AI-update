"""Only deterministic safety gates belong in this module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .candidate import CandidateDecision
from .context import DecisionContext


@dataclass(frozen=True)
class HardRuleDecision:
    candidate: CandidateDecision | None
    reason: str


def select_hard_rule(
    context: DecisionContext,
    candidates: Sequence[CandidateDecision],
) -> HardRuleDecision | None:
    """Return only an unavoidable safety result; soft policy stays in scoring."""

    if not candidates:
        return HardRuleDecision(None, "no_legal_candidate")
    return None
