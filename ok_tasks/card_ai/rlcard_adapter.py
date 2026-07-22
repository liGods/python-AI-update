"""RLCard-backed primitives shared by live play and the simulator.

This module deliberately owns only standard 斗地主 interpretation.  Hero
skills and physical-card state stay in the game kernel.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from rlcard.games.doudizhu.judger import DoudizhuJudger
from rlcard.games.doudizhu.utils import CARD_TYPE, ID_2_ACTION, INDEX, contains_cards


INTERNAL_TO_RLCARD = {"X": "B", "D": "R"}
RLCARD_TO_INTERNAL = {"B": "X", "R": "D"}
WILDCARD = "W"
_ACTION_ORDER = {action: index for index, action in enumerate(ID_2_ACTION)}
_CARD_SORT_ORDER = {card: index for index, card in enumerate(tuple(INDEX) + (WILDCARD,))}


def to_rlcard(cards: list[str] | tuple[str, ...]) -> str:
    """Return the canonical RLCard rank string for project card ranks."""

    converted = [INTERNAL_TO_RLCARD.get(card, card) for card in cards]
    return "".join(sorted(converted, key=lambda card: _CARD_SORT_ORDER.get(card, len(_CARD_SORT_ORDER))))


def to_internal(action: str) -> list[str]:
    return [RLCARD_TO_INTERNAL.get(card, card) for card in action]


def action_order(action: str) -> int:
    """Return the stable RLCard action-space order used for deterministic ties."""

    return _ACTION_ORDER.get(action, len(_ACTION_ORDER))


def physical_action(hand: str, effective_action: str) -> str:
    """Map an effective action back to the physical ranks that must be clicked."""

    remaining_natural = Counter(card for card in hand if card != WILDCARD)
    physical: list[str] = []
    wildcard_count = hand.count(WILDCARD)
    for rank in effective_action:
        if remaining_natural[rank] > 0:
            remaining_natural[rank] -= 1
            physical.append(rank)
        elif wildcard_count > 0:
            wildcard_count -= 1
            physical.append(WILDCARD)
        else:
            raise ValueError(f"Effective action cannot be mapped to physical hand: {effective_action}")
    return to_rlcard(physical)


def legal_action_variants(hand: str, target: str = "") -> tuple[tuple[str, str], ...]:
    """Return each effective legal action together with its physical-card mapping."""

    return tuple((action, physical_action(hand, action)) for action in legal_actions(hand, target))


def action_type(action: str) -> str:
    """Normalize RLCard's detailed action names to the project's stable types."""

    if len(action) == 5 and len(set(action)) == 1:
        return "five_bomb"
    types = CARD_TYPE[0].get(action, [])
    if not types:
        return "unknown"
    name = types[0][0]
    if name.startswith("solo_chain_"):
        return "straight"
    if name.startswith("pair_chain_"):
        return "pair_chain"
    if "chain" in name:
        return "airplane"
    return name


def _contains_with_wildcards(hand: str, action: str) -> bool:
    natural = Counter(card for card in hand if card != WILDCARD)
    required = Counter(action)
    missing = sum(max(0, count - natural[rank]) for rank, count in required.items())
    return missing <= hand.count(WILDCARD)


def contains_with_wildcards(hand: str, action: str) -> bool:
    """Return whether an effective action can be made from a wildcard hand."""

    return _contains_with_wildcards(hand, action)


def is_five_bomb(action: str) -> bool:
    return len(action) == 5 and len(set(action)) == 1


def action_beats(candidate: str, target: str) -> bool:
    """Compare two effective rank actions, including the project's five-bomb rule."""

    if is_five_bomb(candidate):
        return not is_five_bomb(target) or INDEX[candidate[0]] > INDEX[target[0]]
    if is_five_bomb(target):
        return False
    for candidate_type, candidate_weight in CARD_TYPE[0].get(candidate, []):
        for target_type, target_weight in CARD_TYPE[0].get(target, []):
            if candidate_type == "rocket":
                return target_type != "rocket"
            if candidate_type == "bomb" and target_type not in {"bomb", "rocket"}:
                return True
            if candidate_type == target_type and int(candidate_weight) > int(target_weight):
                return True
    return False


@lru_cache(maxsize=512)
def legal_actions(hand: str, target: str = "") -> tuple[str, ...]:
    """Enumerate standard and project-extended legal effective actions."""

    counts = Counter(hand)
    wildcard_count = counts[WILDCARD]
    if wildcard_count:
        standard_actions = [
            action for action in ID_2_ACTION if action != "pass" and _contains_with_wildcards(hand, action)
        ]
    elif max(counts.values(), default=0) <= 4:
        playable = DoudizhuJudger.playable_cards_from_hand(hand)
        standard_actions = sorted((action for action in playable if action in _ACTION_ORDER), key=_ACTION_ORDER.__getitem__)
    else:
        standard_actions = [action for action in ID_2_ACTION if action != "pass" and contains_cards(hand, action)]
    actions = [action for action in standard_actions if not target or action_beats(action, target)]
    for rank in INDEX:
        action = rank * 5
        if counts[rank] + wildcard_count >= 5 and action not in actions and (not target or action_beats(action, target)):
            actions.append(action)
    return tuple(actions)
