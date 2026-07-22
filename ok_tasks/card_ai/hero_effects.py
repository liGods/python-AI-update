from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from typing import Any

from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.heroes import HERO_REGISTRY


RANKS = tuple(CARD_ORDER) + ("W",)
RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}


@dataclass(frozen=True)
class EffectOutcome:
    """One public-information branch produced by a skill effect."""

    hand: tuple[str, ...]
    resource_changes: tuple[tuple[str, int], ...] = ()
    probability: float = 1.0
    label: str = "deterministic"
    risk: float = 0.0


@dataclass(frozen=True)
class EffectProfile:
    effect: str
    operation: str
    choice_kind: str
    optional: bool
    min_hand_delta: int
    max_hand_delta: int
    random_ranks: tuple[str, ...] = ()
    resource: str | None = None
    resource_delta: int = 0
    future_value: float = 0.0
    description: str = ""


_OPERATION_GROUPS: dict[str, frozenset[str]] = {
    "gain": frozenset(
        {
            "both_fill_largest_to_three",
            "fill_j_to_three",
            "fill_kings_to_three",
            "fill_random_pair_to_trio",
            "fill_single_to_pair",
            "fill_twos_to_three",
            "gain_a_and_j",
            "gain_equal_above_six",
            "gain_existing_ranks",
            "gain_high_pair_then_wildcard",
            "gain_joker_or_turn_two_solos_to_pairs",
            "gain_random_above_ten",
            "gain_random_card",
            "gain_random_jqk",
            "gain_rank_by_action_size_once_each",
            "gain_three",
            "gain_three_kings",
            "gain_wildcard",
            "repeat_game_start_skills_gain_twos",
        }
    ),
    "discard": frozenset(
        {
            "discard_increasing_count",
            "discard_lowest",
            "discard_lowest_unless_one",
            "discard_matching_ranks",
            "discard_one",
            "discard_opposite_group",
        }
    ),
    "transform": frozenset(
        {
            "adjust_rank_and_future_pass",
            "change_all_solos_by_outcome",
            "copy_play_to_remaining_hand_once_condition",
            "decrease_one_rank_by_two",
            "increase_below_q_by_two",
            "increase_chosen_card",
            "increase_last_card",
            "increase_lowest",
            "increase_random_card",
            "increase_two_lowest",
            "largest_to_random_joker",
            "mirror_ranks_around_nine",
            "transform_above_k_then_pass",
        }
    ),
    "take": frozenset(
        {
            "inspect_and_take_two_lowest",
            "inspect_four_copy_non_joker",
            "rage_six_take_all_jqk",
            "recover_and_increase_until_seven",
            "recover_and_swap_a_j",
            "reveal_and_take_between_pairs",
            "reveal_equal_and_take_any",
            "reveal_player_count_and_draft",
            "take_all_marked_cards",
            "take_attachments",
            "take_beaten_action",
            "take_beating_action_except_bomb",
            "take_largest_pair",
            "take_one_beaten_card",
            "take_one_from_straight",
            "take_one_or_two_largest",
            "take_pair",
            "take_solo",
            "take_up_to_two_previous_and_pass",
        }
    ),
    "give": frozenset({"give_high_card", "give_up_to_two"}),
    "resource": frozenset(
        {
            "add_grudge",
            "gain_food",
            "gain_or_replace_mount",
            "gain_prestige",
            "light_petal",
            "mark_low_pair_as_tigers",
            "mark_random_ambush",
            "mark_random_enemy_rank",
            "protect_next_play",
            "reset_haofu",
            "reset_hulie",
            "reset_qice",
            "reset_qinxue",
            "unyielding_mark",
            "unyielding_rewards",
        }
    ),
    "rule": frozenset(
        {
            "bid_first",
            "combine_observed_history",
            "copy_latest_three_to_wooden_ox",
            "force_up_to_two_responses",
            "four_above_k_as_four_k",
            "play_x_minus_one_threes_take_action",
            "replace_with_guanyu_zhangfei",
            "restrict_response_to_jqk",
            "see_bottom",
        }
    ),
    "swap": frozenset({"swap_hand_with_wooden_ox"}),
    "composite": frozenset(
        {
            "convert_high_to_wildcard_and_give",
            "convert_solo_pair",
            "copy_bottom_then_discard_after_play",
            "copy_one_and_reset_on_mixed_large_play",
            "discard_after_response_gain_after_pass",
            "discard_group_reduce_enemy_max",
            "discard_k_gain_high_pair_and_pass",
            "discard_same_count_at_least_k_transform_previous_to_3334_or_34567_except_bomb",
            "discard_self_or_other",
            "discard_sum_twelve_gain_above_q",
            "discard_three_recover_highest",
            "discard_two_largest_transfer_lead",
            "exchange_low_for_largest_non_joker",
            "gain_two_discard_one",
            "spend_food_for_jokers_or_wildcard",
            "swap_identity_with_landlord_gain_two_twos",
        }
    ),
}


_DELTA_RANGES: dict[str, tuple[int, int]] = {
    "both_fill_largest_to_three": (0, 2),
    "convert_high_to_wildcard_and_give": (-1, -1),
    "convert_solo_pair": (-1, 1),
    "copy_bottom_then_discard_after_play": (0, 1),
    "copy_one_and_reset_on_mixed_large_play": (0, 1),
    "discard_after_response_gain_after_pass": (-1, 1),
    "discard_group_reduce_enemy_max": (-5, -1),
    "discard_increasing_count": (-3, -1),
    "discard_k_gain_high_pair_and_pass": (0, 1),
    "discard_lowest": (-1, -1),
    "discard_lowest_unless_one": (-1, 0),
    "discard_matching_ranks": (-4, 0),
    "discard_one": (-1, -1),
    "discard_opposite_group": (-2, -1),
    "discard_self_or_other": (-1, 0),
    "discard_sum_twelve_gain_above_q": (-3, -1),
    "discard_three_recover_highest": (-2, -2),
    "discard_two_largest_transfer_lead": (-2, -2),
    "exchange_low_for_largest_non_joker": (0, 0),
    "fill_j_to_three": (0, 3),
    "fill_kings_to_three": (0, 3),
    "fill_random_pair_to_trio": (0, 1),
    "fill_single_to_pair": (0, 1),
    "fill_twos_to_three": (0, 3),
    "gain_a_and_j": (2, 2),
    "gain_equal_above_six": (1, 20),
    "gain_existing_ranks": (1, 20),
    "gain_high_pair_then_wildcard": (1, 2),
    "gain_joker_or_turn_two_solos_to_pairs": (1, 2),
    "gain_random_above_ten": (1, 1),
    "gain_random_card": (1, 1),
    "gain_random_jqk": (1, 1),
    "gain_rank_by_action_size_once_each": (1, 1),
    "gain_three": (1, 1),
    "gain_three_kings": (3, 3),
    "gain_two_discard_one": (1, 1),
    "gain_wildcard": (1, 1),
    "give_high_card": (-1, -1),
    "give_up_to_two": (-2, 0),
    "inspect_and_take_two_lowest": (1, 2),
    "inspect_four_copy_non_joker": (1, 1),
    "largest_to_random_joker": (0, 0),
    "rage_six_take_all_jqk": (0, 12),
    "recover_and_increase_until_seven": (1, 2),
    "recover_and_swap_a_j": (1, 20),
    "repeat_game_start_skills_gain_twos": (2, 2),
    "reveal_and_take_between_pairs": (0, 1),
    "reveal_equal_and_take_any": (0, 1),
    "reveal_player_count_and_draft": (0, 3),
    "spend_food_for_jokers_or_wildcard": (1, 2),
    "swap_identity_with_landlord_gain_two_twos": (2, 2),
    "take_all_marked_cards": (1, 20),
    "take_attachments": (1, 2),
    "take_beaten_action": (1, 20),
    "take_beating_action_except_bomb": (1, 20),
    "take_largest_pair": (2, 2),
    "take_one_beaten_card": (1, 1),
    "take_one_from_straight": (1, 1),
    "take_one_or_two_largest": (1, 2),
    "take_pair": (2, 2),
    "take_solo": (1, 1),
    "take_up_to_two_previous_and_pass": (1, 2),
}


_RANDOM_RANKS: dict[str, tuple[str, ...]] = {
    "fill_random_pair_to_trio": tuple(CARD_ORDER[:13]),
    "gain_equal_above_six": tuple(CARD_ORDER[3:]),
    "gain_high_pair_then_wildcard": ("J", "Q", "K", "A", "2"),
    "gain_joker_or_turn_two_solos_to_pairs": ("X", "D"),
    "gain_random_above_ten": tuple(CARD_ORDER[8:]),
    "gain_random_card": tuple(CARD_ORDER),
    "gain_random_jqk": ("J", "Q", "K"),
    "largest_to_random_joker": ("X", "D"),
    "spend_food_for_jokers_or_wildcard": ("X", "D", "W"),
}


_FIXED_GAINS: dict[str, tuple[str, ...]] = {
    "gain_a_and_j": ("A", "J"),
    "gain_three": ("3",),
    "gain_three_kings": ("K", "K", "K"),
    "gain_wildcard": ("W",),
    "repeat_game_start_skills_gain_twos": ("2", "2"),
    "swap_identity_with_landlord_gain_two_twos": ("2", "2"),
}


_RESOURCE_KEYS: dict[str, tuple[str, int]] = {
    "add_grudge": ("grudge", 1),
    "gain_food": ("food", 1),
    "gain_or_replace_mount": ("mount", 1),
    "gain_prestige": ("prestige", 1),
    "light_petal": ("petal", 1),
    "mark_low_pair_as_tigers": ("tiger_pair", 1),
    "mark_random_ambush": ("ambush", 1),
    "mark_random_enemy_rank": ("enemy_rank_mark", 1),
    "protect_next_play": ("protected_play", 1),
    "reset_haofu": ("haofu_ready", 1),
    "reset_hulie": ("hulie_ready", 1),
    "reset_qice": ("qice_ready", 1),
    "reset_qinxue": ("qinxue_ready", 1),
    "unyielding_mark": ("unyielding", 1),
    "unyielding_rewards": ("unyielding_reward", 1),
}


_PLAYER_CHOICE_EFFECTS = frozenset(
    {
        "both_fill_largest_to_three",
        "discard_self_or_other",
        "discard_two_largest_transfer_lead",
        "exchange_low_for_largest_non_joker",
        "force_up_to_two_responses",
        "give_high_card",
        "give_up_to_two",
        "swap_identity_with_landlord_gain_two_twos",
    }
)
_RANK_CHOICE_EFFECTS = frozenset(
    {
        "copy_bottom_then_discard_after_play",
        "decrease_one_rank_by_two",
        "fill_single_to_pair",
        "increase_chosen_card",
        "take_one_from_straight",
    }
)
_CARD_CHOICE_EFFECTS = frozenset(
    effect
    for effect in set().union(_OPERATION_GROUPS["discard"], _OPERATION_GROUPS["give"], _OPERATION_GROUPS["composite"])
    if effect not in _PLAYER_CHOICE_EFFECTS | _RANK_CHOICE_EFFECTS
)
_OPTIONAL_EFFECTS = frozenset(
    effect
    for skills in HERO_REGISTRY.values()
    for skill in skills
    if skill.interactive
    for effect in (skill.effect,)
)


def _operation_by_effect() -> dict[str, str]:
    operations: dict[str, str] = {}
    duplicates: set[str] = set()
    for operation, effects in _OPERATION_GROUPS.items():
        for effect in effects:
            if effect in operations:
                duplicates.add(effect)
            operations[effect] = operation
    if duplicates:
        raise RuntimeError(f"技能效果分类重复: {sorted(duplicates)}")
    return operations


_OPERATIONS = _operation_by_effect()
_REGISTERED_EFFECTS = {skill.effect for skills in HERO_REGISTRY.values() for skill in skills}
if set(_OPERATIONS) != _REGISTERED_EFFECTS:
    missing = sorted(_REGISTERED_EFFECTS - set(_OPERATIONS))
    extra = sorted(set(_OPERATIONS) - _REGISTERED_EFFECTS)
    raise RuntimeError(f"技能效果处理器表与注册表不一致: missing={missing}, extra={extra}")


def _choice_kind(effect: str, operation: str) -> str:
    if effect in _PLAYER_CHOICE_EFFECTS:
        return "player"
    if effect in _RANK_CHOICE_EFFECTS:
        return "rank"
    if effect in _CARD_CHOICE_EFFECTS:
        return "cards"
    if operation in {"swap", "rule"} and effect in _OPTIONAL_EFFECTS:
        return "activate"
    return "automatic"


EFFECT_PROFILES: dict[str, EffectProfile] = {
    effect: EffectProfile(
        effect=effect,
        operation=operation,
        choice_kind=_choice_kind(effect, operation),
        optional=effect in _OPTIONAL_EFFECTS,
        min_hand_delta=_DELTA_RANGES.get(effect, (0, 0))[0],
        max_hand_delta=_DELTA_RANGES.get(effect, (0, 0))[1],
        random_ranks=_RANDOM_RANKS.get(effect, ()),
        resource=_RESOURCE_KEYS.get(effect, (None, 0))[0],
        resource_delta=_RESOURCE_KEYS.get(effect, (None, 0))[1],
        future_value=0.25 if operation in {"resource", "rule"} else 0.0,
        description=f"三人场技能效果 {effect}",
    )
    for effect, operation in sorted(_OPERATIONS.items())
}


def _sort_hand(cards: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(cards, key=lambda rank: (RANK_INDEX.get(rank, len(RANK_INDEX)), rank)))


def _choice_ranks(choice: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not choice:
        return ()
    values = choice.get("ranks", choice.get("cards", ()))
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values or ())


def _remove_ranks(hand: Sequence[str], requested: Sequence[str]) -> tuple[str, ...]:
    remaining = list(hand)
    for rank in requested:
        if rank in remaining:
            remaining.remove(rank)
    return _sort_hand(remaining)


def _default_discard(hand: tuple[str, ...], count: int) -> tuple[str, ...]:
    return _sort_hand(hand[max(0, min(count, len(hand))):])


def _fill_rank(hand: tuple[str, ...], rank: str, target: int = 3) -> tuple[str, ...]:
    return _sort_hand(hand + (rank,) * max(0, target - hand.count(rank)))


def _transform_selected(hand: tuple[str, ...], selected: tuple[str, ...], replacement: str) -> tuple[str, ...]:
    if not hand:
        return hand
    source = selected[0] if selected and selected[0] in hand else hand[0]
    remaining = list(hand)
    remaining.remove(source)
    remaining.append(replacement)
    return _sort_hand(remaining)


def _higher_ranks(rank: str) -> tuple[str, ...]:
    current = RANK_INDEX.get(rank, len(RANK_INDEX))
    return tuple(value for value in RANKS if RANK_INDEX[value] > current and value != "W")


def _draw_outcomes(hand: tuple[str, ...], ranks: tuple[str, ...], count: int, effect: str) -> tuple[EffectOutcome, ...]:
    count = max(0, count)
    if not ranks or count == 0:
        return (EffectOutcome(hand, label=f"{effect}:no_draw"),)
    denominator = float(len(ranks) ** count)
    outcomes = []
    for drawn in combinations_with_replacement(ranks, count):
        multiplicities = Counter(drawn)
        permutations = math.factorial(count)
        for value in multiplicities.values():
            permutations //= math.factorial(value)
        outcomes.append(
            EffectOutcome(
                _sort_hand(hand + drawn),
                probability=permutations / denominator,
                label=f"{effect}:{''.join(drawn)}",
            )
        )
    return tuple(outcomes)


def _replace_many_outcomes(hand: tuple[str, ...], indexes: tuple[int, ...], effect: str) -> tuple[EffectOutcome, ...]:
    if not indexes:
        return (EffectOutcome(hand, label=f"{effect}:no_card"),)
    choices = tuple(_higher_ranks(hand[index]) or (hand[index],) for index in indexes)
    aggregate: Counter[tuple[str, ...]] = Counter()
    for replacements in product(*choices):
        cards = list(hand)
        for index, replacement in zip(indexes, replacements):
            cards[index] = replacement
        probability = math.prod(1.0 / len(options) for options in choices)
        aggregate[_sort_hand(cards)] += probability
    return tuple(
        EffectOutcome(cards, probability=probability, label=f"{effect}:{''.join(cards)}")
        for cards, probability in sorted(aggregate.items())
    )


def _resolve_effect(
    profile: EffectProfile,
    hand: tuple[str, ...],
    *,
    action_ranks: Sequence[str] = (),
    table_ranks: Sequence[str] = (),
    choice: Mapping[str, Any] | None = None,
    resources: Mapping[str, int] | None = None,
    seed: int = 0,
) -> tuple[EffectOutcome, ...]:
    del seed  # Branch enumeration is deterministic; the engine seed selects a branch during actual resolution.
    hand = _sort_hand(hand)
    selected = _choice_ranks(choice)
    resources = dict(resources or {})
    resource_changes = ()
    if profile.resource:
        resource_changes = ((profile.resource, profile.resource_delta),)

    if profile.effect == "unyielding_rewards":
        level = int(resources.get("unyielding", resources.get("不屈", 0)))
        if level == 2:
            counts = Counter(hand)
            singles = tuple(rank for rank, count in counts.items() if count == 1 and rank not in {"X", "D", "W"})
            gained = (max(singles, key=RANK_INDEX.__getitem__),) if singles else ()
        elif level == 3:
            gained = ("2",)
        elif level >= 4:
            gained = ("W",)
        else:
            gained = ()
        return (EffectOutcome(_sort_hand(hand + gained), resource_changes, label=f"{profile.effect}:{level}"),)

    if profile.operation in {"resource", "rule"}:
        return (EffectOutcome(hand, resource_changes, label=profile.effect),)

    if profile.operation in {"discard", "give"}:
        remove_count = max(0, -profile.min_hand_delta)
        post_hand = _remove_ranks(hand, selected) if selected else _default_discard(hand, remove_count)
        return (EffectOutcome(post_hand, resource_changes, label=profile.effect),)

    if profile.operation == "swap":
        zone = tuple(str(rank) for rank in (choice or {}).get("zone_ranks", ()))
        post_hand = _sort_hand(zone) if zone else hand
        return (EffectOutcome(post_hand, resource_changes, label=profile.effect, risk=0.5 if not zone else 0.0),)

    if profile.effect in {"fill_j_to_three", "fill_kings_to_three", "fill_twos_to_three"}:
        rank = {"fill_j_to_three": "J", "fill_kings_to_three": "K", "fill_twos_to_three": "2"}[profile.effect]
        return (EffectOutcome(_fill_rank(hand, rank), resource_changes, label=profile.effect),)
    if profile.effect == "fill_single_to_pair":
        counts = Counter(hand)
        rank = selected[0] if selected else next((value for value in hand if counts[value] == 1), None)
        post_hand = _fill_rank(hand, rank, 2) if rank else hand
        return (EffectOutcome(post_hand, resource_changes, label=profile.effect),)
    if profile.effect == "fill_random_pair_to_trio":
        pairs = tuple(rank for rank, count in Counter(hand).items() if count == 2)
        outcomes = tuple(
            EffectOutcome(
                _fill_rank(hand, rank),
                probability=1.0 / len(pairs),
                label=f"{profile.effect}:{rank}",
            )
            for rank in pairs
        )
        return outcomes or (EffectOutcome(hand, label=f"{profile.effect}:no_pair"),)
    if profile.effect == "both_fill_largest_to_three":
        rank = max(hand, key=lambda value: RANK_INDEX.get(value, -1), default=None)
        return (EffectOutcome(_fill_rank(hand, rank) if rank else hand, label=profile.effect),)

    if profile.operation == "take":
        available = tuple(str(rank) for rank in table_ranks)
        if profile.effect == "recover_and_increase_until_seven":
            recovered = tuple(str(rank) for rank in action_ranks)
            capacity = max(0, 7 - int(resources.get("冲阵回收", resources.get("recovered", 0))))
            recovered = recovered[:capacity]
            if not recovered:
                return (EffectOutcome(hand, label=f"{profile.effect}:limit"),)
            choices = tuple(_higher_ranks(rank) or (rank,) for rank in recovered)
            aggregate: Counter[tuple[str, ...]] = Counter()
            for values in product(*choices):
                probability = math.prod(1.0 / len(options) for options in choices)
                aggregate[_sort_hand(hand + tuple(values))] += probability
            return tuple(
                EffectOutcome(cards, probability=probability, label=f"{profile.effect}:{''.join(cards)}")
                for cards, probability in sorted(aggregate.items())
            )
        if profile.effect == "recover_and_swap_a_j":
            swap = {"A": "J", "J": "A"}
            recovered = tuple(swap.get(str(rank), str(rank)) for rank in action_ranks)
            return (EffectOutcome(_sort_hand(hand + recovered), label=profile.effect),)
        if selected:
            gained = tuple(rank for rank in selected if rank in available)
        elif profile.effect in {"take_pair", "take_largest_pair"}:
            gained = available[:2]
        elif profile.max_hand_delta > 0:
            gained = available[: profile.max_hand_delta]
        else:
            gained = available
        return (EffectOutcome(_sort_hand(hand + gained), resource_changes, label=profile.effect),)

    if profile.operation == "transform":
        if profile.random_ranks:
            probability = 1.0 / len(profile.random_ranks)
            return tuple(
                EffectOutcome(
                    _transform_selected(hand, selected, rank),
                    resource_changes,
                    probability,
                    f"{profile.effect}:{rank}",
                )
                for rank in profile.random_ranks
            )
        if profile.effect == "mirror_ranks_around_nine":
            mirror = {"3": "2", "4": "A", "5": "K", "6": "Q", "7": "J", "8": "T", "T": "8", "J": "7", "Q": "6", "K": "5", "A": "4", "2": "3"}
            return (EffectOutcome(_sort_hand(tuple(mirror.get(rank, rank) for rank in hand)), label=profile.effect),)
        if profile.effect == "increase_last_card" and hand == ("D",):
            return (EffectOutcome(("3", "3", "3", "3"), label=f"{profile.effect}:split_joker"),)
        if profile.effect in {"increase_last_card", "increase_lowest", "increase_chosen_card"}:
            source = selected[0] if selected and selected[0] in hand else (hand[0] if hand else None)
            if source is None:
                return (EffectOutcome(hand, label=f"{profile.effect}:no_card"),)
            replacements = _higher_ranks(source)
            if not replacements:
                return (EffectOutcome(hand, label=f"{profile.effect}:maximum"),)
            probability = 1.0 / len(replacements)
            return tuple(
                EffectOutcome(
                    _transform_selected(hand, (source,), replacement),
                    probability=probability,
                    label=f"{profile.effect}:{source}>{replacement}",
                )
                for replacement in replacements
            )
        if profile.effect == "increase_two_lowest":
            return _replace_many_outcomes(hand, tuple(range(min(2, len(hand)))), profile.effect)
        if profile.effect == "increase_random_card":
            eligible = tuple(index for index, rank in enumerate(hand) if _higher_ranks(rank))
            if not eligible:
                return (EffectOutcome(hand, label=f"{profile.effect}:maximum"),)
            aggregate: Counter[tuple[str, ...]] = Counter()
            for index in eligible:
                replacements = _higher_ranks(hand[index])
                for replacement in replacements:
                    cards = list(hand)
                    cards[index] = replacement
                    aggregate[_sort_hand(cards)] += 1.0 / len(eligible) / len(replacements)
            return tuple(
                EffectOutcome(cards, probability=probability, label=f"{profile.effect}:{''.join(cards)}")
                for cards, probability in sorted(aggregate.items())
            )
        if profile.effect == "increase_below_q_by_two":
            indexes = tuple(index for index, rank in enumerate(hand) if RANK_INDEX.get(rank, 99) <= RANK_INDEX["Q"])
            cards = list(hand)
            for index in indexes:
                cards[index] = RANKS[min(RANK_INDEX[cards[index]] + 2, RANK_INDEX["D"])]
            return (EffectOutcome(_sort_hand(cards), label=profile.effect),)
        if profile.effect == "decrease_one_rank_by_two":
            source = selected[0] if selected and selected[0] in hand else (hand[-1] if hand else None)
            if source is None:
                return (EffectOutcome(hand, label=f"{profile.effect}:no_card"),)
            replacement = RANKS[max(0, RANK_INDEX.get(source, 0) - 2)]
            return (EffectOutcome(_transform_selected(hand, (source,), replacement), label=f"{profile.effect}:{source}>{replacement}"),)
        if profile.effect == "change_all_solos_by_outcome":
            counts = Counter(hand)
            indexes = tuple(index for index, rank in enumerate(hand) if counts[rank] == 1 and rank not in {"X", "D", "W"})
            outcomes = []
            for direction in (-1, 1):
                cards = list(hand)
                for index in indexes:
                    cards[index] = RANKS[min(RANK_INDEX["D"], max(0, RANK_INDEX[cards[index]] + direction))]
                outcomes.append(EffectOutcome(_sort_hand(cards), probability=0.5, label=f"{profile.effect}:{direction:+d}"))
            return tuple(outcomes)
        if profile.effect == "adjust_rank_and_future_pass":
            source = selected[0] if selected and selected[0] in hand else (hand[0] if hand else None)
            replacements = _higher_ranks(source) if source else ()
            if replacements:
                probability = 1.0 / len(replacements)
                return tuple(EffectOutcome(_transform_selected(hand, (source,), rank), probability=probability, label=f"{profile.effect}:{rank}") for rank in replacements)
        if profile.effect == "transform_above_k_then_pass":
            candidates = tuple(rank for rank in hand if RANK_INDEX.get(rank, -1) >= RANK_INDEX["K"])
            source = selected[0] if selected and selected[0] in candidates else (candidates[0] if candidates else None)
            if source:
                return (EffectOutcome(_transform_selected(hand, (source,), "K"), label=f"{profile.effect}:{source}>K"),)
        return (EffectOutcome(hand, resource_changes, label=profile.effect),)

    fixed = _FIXED_GAINS.get(profile.effect)
    if fixed:
        return (EffectOutcome(_sort_hand(hand + fixed), resource_changes, label=profile.effect),)
    if profile.effect == "gain_high_pair_then_wildcard":
        petal = int(resources.get("petal", resources.get("花瓣", 0)))
        if petal >= 5:
            return (EffectOutcome(_sort_hand(hand + ("W",)), label=f"{profile.effect}:wildcard"),)
        high_ranks = ("J", "Q", "K", "A", "2")
        return tuple(EffectOutcome(_sort_hand(hand + (rank, rank)), probability=1.0 / len(high_ranks), label=f"{profile.effect}:{rank}") for rank in high_ranks)
    if profile.random_ranks:
        gain_count = max(1, profile.min_hand_delta)
        if profile.effect == "gain_equal_above_six":
            gain_count = max(1, len(action_ranks))
        return _draw_outcomes(hand, profile.random_ranks, gain_count, profile.effect)

    if profile.effect == "gain_existing_ranks":
        ranks = tuple(dict.fromkeys(action_ranks or hand))
        gained = ranks[: max(1, min(len(ranks), profile.max_hand_delta))]
        return (EffectOutcome(_sort_hand(hand + gained), label=profile.effect),)
    if profile.effect == "gain_rank_by_action_size_once_each":
        rank = next((value for value in CARD_ORDER if RANK_INDEX.get(value) == len(action_ranks)), "3")
        return (EffectOutcome(_sort_hand(hand + (rank,)), label=profile.effect),)

    if profile.operation == "composite":
        if profile.effect == "copy_bottom_then_discard_after_play":
            rank = selected[0] if selected else str((choice or {}).get("rank", ""))
            return (EffectOutcome(_sort_hand(hand + ((rank,) if rank in RANKS else ())), label=profile.effect),)
        if profile.effect == "copy_one_and_reset_on_mixed_large_play":
            rank = selected[0] if selected else (hand[0] if hand else None)
            return (EffectOutcome(_sort_hand(hand + ((rank,) if rank else ())), label=profile.effect),)
        if profile.effect == "discard_three_recover_highest":
            post_discard = _remove_ranks(hand, selected) if selected else _default_discard(hand, 3)
            recovered = max(table_ranks, key=lambda rank: RANK_INDEX.get(rank, -1), default=None)
            return (EffectOutcome(_sort_hand(post_discard + ((recovered,) if recovered else ())), label=profile.effect),)
        if profile.effect == "convert_solo_pair":
            operation = str((choice or {}).get("operation", ""))
            rank = selected[0] if selected else (hand[0] if hand else None)
            if rank and operation == "solo_to_pair":
                return (EffectOutcome(_sort_hand(hand + (rank,)), label=profile.effect),)
            if rank and operation == "pair_to_solo":
                return (EffectOutcome(_remove_ranks(hand, (rank,)), label=profile.effect),)
        if profile.effect == "gain_two_discard_one":
            outcomes = []
            for drawn in _draw_outcomes(hand, tuple(CARD_ORDER), 2, profile.effect):
                selected_discard = selected if selected and Counter(selected) <= Counter(drawn.hand) else ()
                post_hand = (
                    _remove_ranks(drawn.hand, selected_discard)
                    if selected_discard
                    else _default_discard(drawn.hand, 1)
                )
                outcomes.append(
                    EffectOutcome(
                        post_hand,
                        probability=drawn.probability,
                        label=f"{drawn.label}:discard",
                    )
                )
            return tuple(outcomes)
        if profile.effect == "discard_after_response_gain_after_pass":
            if action_ranks:
                post_hand = _remove_ranks(hand, selected) if selected else _default_discard(hand, 1)
                return (EffectOutcome(post_hand, label=f"{profile.effect}:response"),)
            return _draw_outcomes(hand, tuple(CARD_ORDER), 1, profile.effect)
        if profile.effect == "discard_group_reduce_enemy_max":
            post_hand = _remove_ranks(hand, selected) if selected else _default_discard(hand, 1)
            return (EffectOutcome(post_hand, label=profile.effect),)
        if profile.effect == "discard_k_gain_high_pair_and_pass":
            post_hand = _remove_ranks(hand, ("K",))
            high_ranks = ("J", "Q", "K", "A", "2")
            return tuple(EffectOutcome(_sort_hand(post_hand + (rank, rank)), probability=1.0 / len(high_ranks), label=f"{profile.effect}:{rank}") for rank in high_ranks)
        if profile.effect == "discard_same_count_at_least_k_transform_previous_to_3334_or_34567_except_bomb":
            post_hand = _remove_ranks(hand, selected) if selected else hand
            return (EffectOutcome(post_hand, label=profile.effect),)
        if profile.effect == "discard_self_or_other":
            target = str((choice or {}).get("target", "self"))
            post_hand = (_remove_ranks(hand, selected) if selected else _default_discard(hand, 1)) if target in {"self", "", "None"} else hand
            return (EffectOutcome(post_hand, label=f"{profile.effect}:{target}"),)
        if profile.effect == "discard_sum_twelve_gain_above_q":
            post_hand = _remove_ranks(hand, selected) if selected else _default_discard(hand, 1)
            return _draw_outcomes(post_hand, ("Q", "K", "A", "2", "X", "D"), 1, profile.effect)
        if profile.effect == "exchange_low_for_largest_non_joker":
            post_hand = _remove_ranks(hand, selected) if selected else _default_discard(hand, 1)
            available = tuple(rank for rank in table_ranks if rank not in {"X", "D", "W"})
            gained = (max(available, key=RANK_INDEX.__getitem__),) if available else ()
            return (EffectOutcome(_sort_hand(post_hand + gained), label=profile.effect, risk=0.25 if not gained else 0.0),)
        if profile.effect == "spend_food_for_jokers_or_wildcard":
            if int(resources.get("food", 0)) <= 0:
                return (EffectOutcome(hand, label=f"{profile.effect}:unavailable", risk=1.0),)
            values = ("X", "D", "W")
            return tuple(EffectOutcome(_sort_hand(hand + (rank,)), (("food", -1),), 1.0 / len(values), f"{profile.effect}:{rank}") for rank in values)
        if profile.effect == "swap_identity_with_landlord_gain_two_twos":
            return (EffectOutcome(_sort_hand(hand + ("2", "2")), label=profile.effect),)
        if profile.effect in {"give_high_card", "give_up_to_two", "discard_two_largest_transfer_lead"}:
            count = max(1, -profile.min_hand_delta)
            chosen = selected or tuple(reversed(hand[-count:]))
            return (EffectOutcome(_remove_ranks(hand, chosen), label=profile.effect),)
        delta = profile.max_hand_delta if profile.max_hand_delta < 0 else profile.min_hand_delta
        if delta < 0:
            return (EffectOutcome(_default_discard(hand, -delta), label=profile.effect, risk=0.25),)
        if delta > 0:
            return (EffectOutcome(_sort_hand(hand + ("3",) * delta), label=profile.effect, risk=0.5),)
        return (EffectOutcome(hand, resource_changes, label=profile.effect, risk=0.25),)

    if profile.operation == "gain":
        delta = max(0, profile.min_hand_delta)
        return (EffectOutcome(_sort_hand(hand + ("3",) * delta), resource_changes, label=profile.effect, risk=0.5),)
    raise AssertionError(f"未实现的技能效果操作: {profile.operation}")


EffectHandler = Callable[..., tuple[EffectOutcome, ...]]


def _make_handler(profile: EffectProfile) -> EffectHandler:
    def handler(
        hand: tuple[str, ...],
        *,
        action_ranks: Sequence[str] = (),
        table_ranks: Sequence[str] = (),
        choice: Mapping[str, Any] | None = None,
        resources: Mapping[str, int] | None = None,
        seed: int = 0,
    ) -> tuple[EffectOutcome, ...]:
        return _resolve_effect(
            profile,
            hand,
            action_ranks=action_ranks,
            table_ranks=table_ranks,
            choice=choice,
            resources=resources,
            seed=seed,
        )

    handler.__name__ = f"handle_{profile.effect}"
    return handler


EFFECT_HANDLERS: dict[str, EffectHandler] = {
    effect: _make_handler(profile) for effect, profile in EFFECT_PROFILES.items()
}


def effect_handler_for(effect: str) -> EffectHandler:
    """Return an explicit handler; unknown effects are contract violations."""

    try:
        return EFFECT_HANDLERS[effect]
    except KeyError as exc:
        raise KeyError(f"技能效果没有注册处理器: {effect}") from exc


def validate_effect_contract() -> tuple[str, ...]:
    errors: list[str] = []
    if len(EFFECT_PROFILES) != len(_REGISTERED_EFFECTS):
        errors.append("effect_profile_count")
    if set(EFFECT_HANDLERS) != _REGISTERED_EFFECTS:
        errors.append("effect_handler_coverage")
    for effect, profile in EFFECT_PROFILES.items():
        if profile.operation not in _OPERATION_GROUPS:
            errors.append(f"invalid_operation:{effect}")
        outcomes = EFFECT_HANDLERS[effect](("3", "3", "4"), action_ranks=("3",), table_ranks=("4", "4"))
        if not outcomes:
            errors.append(f"empty_outcomes:{effect}")
        elif not math.isclose(sum(outcome.probability for outcome in outcomes), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"invalid_probability:{effect}")
    return tuple(errors)


__all__ = [
    "EFFECT_HANDLERS",
    "EFFECT_PROFILES",
    "EffectHandler",
    "EffectOutcome",
    "EffectProfile",
    "effect_handler_for",
    "validate_effect_contract",
]
