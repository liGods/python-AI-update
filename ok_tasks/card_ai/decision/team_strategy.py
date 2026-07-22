"""Public-information teammate protection and card-passing preferences."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .candidate import CandidateDecision
from .context import DecisionContext


def select_team_candidate(
    context: DecisionContext,
    candidates: Sequence[CandidateDecision],
    *,
    rank_index: Callable[[str], int],
) -> CandidateDecision | None | object:
    """Return a teammate-safe legal candidate, or ``None`` to preserve its turn."""

    if context.position == "landlord":
        return _NO_TEAM_DECISION
    teammate_count = context.teammate_count
    terminal = [candidate for candidate in candidates if candidate.terminal]
    if context.target and context.protect_teammate_play:
        if terminal:
            return min(terminal, key=lambda candidate: candidate.score)
        if teammate_count is not None and teammate_count <= len(context.target):
            return None
        return None
    if context.target or teammate_count not in {1, 2}:
        return _NO_TEAM_DECISION
    wanted_type = "solo" if teammate_count == 1 else "pair"
    passing = [candidate for candidate in candidates if candidate.action_type == wanted_type and not candidate.uses_control]
    if not passing:
        return _NO_TEAM_DECISION
    return min(
        passing,
        key=lambda candidate: (
            max((rank_index(card) for card in candidate.effective_action), default=99),
            candidate.score,
        ),
    )


_NO_TEAM_DECISION = object()
