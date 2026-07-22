"""Shared public-information selection core for live and simulator adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .candidate import CandidateDecision
from .context import DecisionContext
from .hard_rules import select_hard_rule
from .scoring import select_soft_candidate
from ok_tasks.card_ai.search.endgame import EndgameSearchResult, search_public_endgame


@dataclass(frozen=True)
class CoreDecision:
    """Selection result expressed only in terms of an existing legal candidate."""

    candidate: CandidateDecision | None
    reason: str
    search: EndgameSearchResult | None = None


class DecisionPolicyCore:
    """Choose among adapter-supplied legal candidates without creating actions."""

    def choose(
        self,
        context: DecisionContext,
        candidates: Sequence[CandidateDecision],
        *,
        is_bomb: Callable[[str], bool],
        rank_index: Callable[[str], int],
        baseline_turns: Callable[[], int],
        pass_projection: Callable[[], Any],
    ) -> CoreDecision:
        hard_decision = select_hard_rule(context, candidates)
        if hard_decision is not None:
            return CoreDecision(hard_decision.candidate, hard_decision.reason)

        selected = select_soft_candidate(
            context,
            candidates,
            is_bomb=is_bomb,
            rank_index=rank_index,
            baseline_turns=baseline_turns,
            pass_projection=pass_projection,
        )
        if selected is None:
            return CoreDecision(None, "soft_pass")

        result = search_public_endgame(context, candidates, selected)
        return CoreDecision(result.candidate, result.reason, result)
