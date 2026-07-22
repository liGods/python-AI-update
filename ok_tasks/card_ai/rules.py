from __future__ import annotations

from collections import Counter
from itertools import product

from ok_tasks.card_ai.rlcard_adapter import action_type as _action_type
from ok_tasks.card_ai.rlcard_adapter import legal_actions as _legal_actions
from ok_tasks.card_ai.rlcard_adapter import to_internal as _to_internal
from ok_tasks.card_ai.rlcard_adapter import to_rlcard as _to_rlcard
from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.schema import CardInstance


WILDCARD = "W"
RANK_ORDER = tuple(CARD_ORDER) + (WILDCARD,)
RANK_INDEX = {rank: index for index, rank in enumerate(RANK_ORDER)}


def rank_key(rank: str) -> int:
    return RANK_INDEX.get(rank, len(RANK_INDEX))


def sorted_ranks(ranks: list[str] | tuple[str, ...]) -> list[str]:
    return sorted(ranks, key=rank_key)


def action_type(ranks: list[str] | tuple[str, ...]) -> str:
    if not ranks:
        return "none"
    return _action_type(_to_rlcard(list(ranks)))


def estimate_route_turns(ranks: tuple[str, ...]) -> int:
    # Delayed import keeps the rule primitives usable while the live adapter is
    # importing its public decision core.
    from ok_tasks.RlCardRuleModel import _post_skill_route_turns

    return _post_skill_route_turns(tuple(ranks))


def enumerate_rank_actions(hand_ranks: list[str], target_ranks: list[str]) -> list[tuple[str, ...]]:
    """Enumerate RLCard actions and expand skill-created wildcards safely."""

    wildcard_count = hand_ranks.count(WILDCARD)
    natural = [rank for rank in hand_ranks if rank != WILDCARD]
    assignments = [()] if wildcard_count == 0 else product(CARD_ORDER, repeat=wildcard_count)
    target = _to_rlcard(target_ranks)
    actions: set[tuple[str, ...]] = set()
    for assignment in assignments:
        assigned_hand = _to_rlcard(natural + list(assignment))
        for action in _legal_actions(assigned_hand, target):
            actions.add(tuple(_to_internal(action)))
    return sorted(actions, key=lambda value: (len(value), tuple(rank_key(rank) for rank in value)))


def select_card_ids(hand: list[CardInstance], ranks: tuple[str, ...] | list[str]) -> tuple[str, ...] | None:
    """Map effective ranks to physical cards, consuming natural cards before wildcards."""

    remaining = list(hand)
    selected: list[str] = []
    for rank in ranks:
        natural_index = next((index for index, card in enumerate(remaining) if card.rank == rank), None)
        if natural_index is None:
            natural_index = next((index for index, card in enumerate(remaining) if card.rank == WILDCARD), None)
        if natural_index is None:
            return None
        selected.append(remaining.pop(natural_index).card_id)
    return tuple(selected)


def validate_card_selection(hand: list[CardInstance], card_ids: tuple[str, ...]) -> bool:
    available = Counter(card.card_id for card in hand)
    requested = Counter(card_ids)
    return bool(card_ids) and all(available[card_id] >= count for card_id, count in requested.items())
