from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from typing import Any

from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.hero_effects import EFFECT_HANDLERS, EFFECT_PROFILES, EffectOutcome
from ok_tasks.card_ai.heroes import HERO_REGISTRY, normalize_hero_name, skill_by_rule_id
from ok_tasks.card_ai.schema import POSITIONS, HeroSkillSpec, LegalAction


MODE_ID = "landlord_3p"
RANKS = tuple(CARD_ORDER) + ("W",)
RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}
CONTROL_RANKS = frozenset({"2", "X", "D", "W"})
DEFERRED_TRIGGERS = frozenset(
    {
        "after_play_and_unbeaten",
        "beaten_or_pass",
        "game_start_or_unbeaten_play",
        "low_solo_pair_beaten",
        "own_play_beaten",
    }
)
BENEFICIAL_TARGET_EFFECTS = frozenset(
    {"both_fill_largest_to_three", "discard_two_largest_transfer_lead", "give_high_card", "give_up_to_two"}
)
HARMFUL_TARGET_EFFECTS = frozenset(
    {"discard_self_or_other", "exchange_low_for_largest_non_joker", "force_up_to_two_responses"}
)
RouteEvaluator = Callable[[tuple[str, ...]], int | float | tuple[int | float, ...]]

SUPPORTED_TRIGGERS = frozenset(
    {
        "active", "active_after_two_marks", "active_grudge_two", "active_with_lead",
        "active_without_trio_or_bomb", "after_beating", "after_bomb", "after_play",
        "after_play_and_unbeaten", "after_play_under_six", "after_renyi_twice",
        "after_response_or_pass", "after_solo_or_pair", "after_straight", "after_trio_attachment",
        "after_xinghua", "ambush_resolution", "any_action_over_four", "any_all_below_five",
        "beaten_or_pass", "bidding", "exact_plus_one_response_over_four", "first_hand_below_eight",
        "first_hand_below_twelve", "game_start", "game_start_or_ambush_played",
        "game_start_or_joker_play", "game_start_or_unbeaten_play", "global_new_action_type",
        "largest_play_resolution", "lose_haofu_card", "low_solo_pair_beaten", "mark_changed",
        "new_action_type", "new_action_type_contains_three", "no_legal_response", "other_non_solo",
        "other_pair", "other_straight", "other_trio_attachment", "own_play_beaten", "pass_response",
        "pass_under_four", "pass_with_one", "petal_three_or_five", "play_contains_two",
        "play_qice_card", "prestige_three", "repeat_action_type", "response", "response_over_two",
        "third_no_legal_response", "three_distinct_action_types", "three_same_action_types",
        "unbeaten_other_solo",
    }
)
_REGISTERED_TRIGGERS = {skill.trigger for skills in HERO_REGISTRY.values() for skill in skills}
if SUPPORTED_TRIGGERS != _REGISTERED_TRIGGERS:
    missing = sorted(_REGISTERED_TRIGGERS - SUPPORTED_TRIGGERS)
    extra = sorted(SUPPORTED_TRIGGERS - _REGISTERED_TRIGGERS)
    raise RuntimeError(f"技能触发器表与注册表不一致: missing={missing}, extra={extra}")


def _normalise_rank(rank: object) -> str:
    return {"B": "X", "R": "D"}.get(str(rank), str(rank))


def _sort_hand(cards: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted((_normalise_rank(card) for card in cards), key=lambda rank: (RANK_INDEX.get(rank, 99), rank)))


def _normalise_history(history: object) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for index, event in enumerate(history if isinstance(history, (list, tuple)) else ()):
        if isinstance(event, Mapping):
            result.append(dict(event))
        elif isinstance(event, (list, tuple)):
            result.append({"kind": "play", "turn": index, "ranks": list(event)})
    return tuple(result)


def _relations(position: str, landlord: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    seats = ("landlord", "landlord_down", "landlord_up")
    if position == landlord:
        return (), tuple(seat for seat in seats if seat != position)
    return tuple(seat for seat in seats if seat not in {position, landlord}), (landlord,)


@dataclass(frozen=True)
class HeroDecisionContext:
    """Public-information-only input shared by live rules and simulation."""

    hand: tuple[str, ...]
    table_cards: tuple[str, ...] = ()
    position: str = "landlord_down"
    landlord: str = "landlord"
    hero: str | None = None
    allies: tuple[str, ...] = ()
    enemies: tuple[str, ...] = ()
    public_heroes: tuple[tuple[str, str | None], ...] = ()
    public_skill_uses: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    public_card_counts: tuple[tuple[str, int], ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    skill_uses: Mapping[str, int] = field(default_factory=dict)
    marks: Mapping[str, int] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    hand_card_ids: tuple[str, ...] = ()
    card_sources: tuple[str, ...] = ()
    table_owner: str | None = None
    table_relation: str = "unknown"
    pressure: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    turn_index: int = 0
    mode_id: str = MODE_ID

    def __post_init__(self) -> None:
        if len(self.hand_card_ids) not in {0, len(self.hand)}:
            raise ValueError("hand_card_ids 必须与手牌数量一致")
        if len(self.card_sources) not in {0, len(self.hand)}:
            raise ValueError("card_sources 必须与手牌数量一致")
        if self.mode_id != MODE_ID:
            raise ValueError(f"武将策略当前仅支持三人场: {self.mode_id}")

    @property
    def resources(self) -> dict[str, int]:
        values = {str(key): int(value) for key, value in self.marks.items() if isinstance(value, int)}
        aliases = {
            "不屈": "unyielding",
            "冲阵回收": "recovered",
            "田": "food",
            "花瓣": "petal",
            "威望": "prestige",
            "睚眦": "grudge",
        }
        for source, target in aliases.items():
            if source in values:
                values[target] = values[source]
        for key, value in self.extra.get("resources", {}).items() if isinstance(self.extra.get("resources"), Mapping) else ():
            if isinstance(value, int):
                values[str(key)] = value
        return values

    @property
    def enemy_card_counts(self) -> tuple[int, ...]:
        values = dict(self.public_card_counts)
        return tuple(values[enemy] for enemy in self.enemies if values.get(enemy, 0) > 0)

    @property
    def ally_card_counts(self) -> tuple[int, ...]:
        values = dict(self.public_card_counts)
        return tuple(values[ally] for ally in self.allies if values.get(ally, 0) > 0)

    @classmethod
    def from_legacy_state(cls, state: Mapping[str, Any]) -> "HeroDecisionContext":
        position = str(state.get("position", "landlord_down"))
        landlord = str(state.get("landlord", "landlord"))
        allies, enemies = _relations(position, landlord)
        explicit_enemy_counts = tuple(
            int(value)
            for value in state.get("enemy_card_counts", ())
            if isinstance(value, int) and value >= 0
        )
        opponent_counts = tuple(
            int(value)
            for value in state.get("opponent_card_counts", ())
            if isinstance(value, int) and value >= 0
        )
        public_counts: list[tuple[str, int]] = []
        if explicit_enemy_counts:
            public_counts.extend(zip(enemies, explicit_enemy_counts))
            teammate_count = state.get("teammate_card_count", state.get("teammate_count"))
            if allies and isinstance(teammate_count, int):
                public_counts.append((allies[0], teammate_count))
        else:
            other_seats = tuple(seat for seat in ("landlord", "landlord_down", "landlord_up") if seat != position)
            public_counts.extend(zip(other_seats, opponent_counts))

        hero_state = state.get("hero_state", {}) if isinstance(state.get("hero_state"), Mapping) else {}
        table_relation = "ally" if state.get("table_is_teammate") else "enemy" if state.get("table_is_enemy") else "unknown"
        pressure = state.get("table_pressure", state.get("pressure_context", {}))
        raw_hand = tuple(_normalise_rank(value) for value in state.get("hand_cards", ()))
        order = tuple(sorted(range(len(raw_hand)), key=lambda index: (RANK_INDEX.get(raw_hand[index], 99), raw_hand[index], index)))
        hand = tuple(raw_hand[index] for index in order)
        raw_ids = tuple(str(value) for value in state.get("hand_card_ids", ()))
        raw_sources = tuple(str(value) for value in state.get("card_sources", ()))
        hand_ids = tuple(raw_ids[index] for index in order) if len(raw_ids) == len(raw_hand) else ()
        sources = tuple(raw_sources[index] for index in order) if len(raw_sources) == len(raw_hand) else ()
        extra = dict(hero_state.get("extra", {})) if isinstance(hero_state.get("extra"), Mapping) else {}
        if isinstance(state.get("pending_interaction"), Mapping):
            extra["pending_interaction"] = dict(state["pending_interaction"])
        return cls(
            hand=hand,
            table_cards=_sort_hand(state.get("table_cards", ())),
            position=position,
            landlord=landlord,
            hero=normalize_hero_name(state.get("hero")),
            allies=allies,
            enemies=enemies,
            public_heroes=tuple(
                (str(seat), normalize_hero_name(hero))
                for seat, hero in (
                    state.get("public_heroes", {}).items()
                    if isinstance(state.get("public_heroes"), Mapping)
                    else ()
                )
            ),
            public_skill_uses={
                str(seat): {str(name): int(uses) for name, uses in values.items() if isinstance(uses, int)}
                for seat, values in (
                    state.get("public_skill_uses", {}).items()
                    if isinstance(state.get("public_skill_uses"), Mapping)
                    else ()
                )
                if isinstance(values, Mapping)
            },
            public_card_counts=tuple(public_counts),
            history=_normalise_history(state.get("history", ())),
            skill_uses=dict(hero_state.get("skill_uses", {})) if isinstance(hero_state.get("skill_uses"), Mapping) else {},
            marks=dict(hero_state.get("marks", {})) if isinstance(hero_state.get("marks"), Mapping) else {},
            extra=extra,
            hand_card_ids=hand_ids,
            card_sources=sources,
            table_owner=str(state["trick_owner"]) if state.get("trick_owner") else None,
            table_relation=table_relation,
            pressure=dict(pressure) if isinstance(pressure, Mapping) else {},
            seed=int(state.get("seed", 0) or 0),
            turn_index=int(state.get("turn_index", 0) or 0),
        )

    @classmethod
    def from_engine(cls, engine: Any, actor: str | None = None) -> "HeroDecisionContext":
        state = engine.state
        actor = actor or state.current_player
        player = state.players[actor]
        ordered_cards = tuple(sorted(player.hand, key=lambda card: (RANK_INDEX.get(card.rank, 99), card.rank, card.card_id)))
        allies, enemies = _relations(actor, state.landlord)
        public_counts = tuple(
            (position, len(other.hand)) for position, other in state.players.items() if position != actor
        )
        relation = "self" if state.trick_owner == actor else "ally" if state.trick_owner in allies else "enemy" if state.trick_owner in enemies else "unknown"
        extra = dict(player.extra)
        if state.pending_interaction:
            extra["pending_interaction"] = dict(state.pending_interaction)
        return cls(
            hand=tuple(card.rank for card in ordered_cards),
            table_cards=_sort_hand(state.target_ranks),
            position=actor,
            landlord=state.landlord,
            hero=player.hero,
            allies=allies,
            enemies=enemies,
            public_heroes=tuple((position, other.hero) for position, other in state.players.items()),
            public_skill_uses={
                position: dict(other.skill_uses) for position, other in state.players.items()
            },
            public_card_counts=public_counts,
            history=_normalise_history(state.history),
            skill_uses=dict(player.skill_uses),
            marks=dict(player.marks),
            extra=extra,
            hand_card_ids=tuple(card.card_id for card in ordered_cards),
            card_sources=tuple(card.source for card in ordered_cards),
            table_owner=state.trick_owner,
            table_relation=relation,
            pressure={"urgent": min((count for _, count in public_counts), default=17) <= 3},
            seed=state.seed,
            turn_index=state.turn_index,
        )


@dataclass(frozen=True)
class SkillChoice:
    choice_id: str
    rule_id: str
    skill: str
    effect: str
    kind: str
    activate: bool = True
    card_ids: tuple[str, ...] = ()
    ranks: tuple[str, ...] = ()
    target: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires_ui_verification: bool = False

    @property
    def skip(self) -> bool:
        return not self.activate or self.kind == "skip"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parameters"] = dict(self.parameters)
        return value


@dataclass(frozen=True)
class ProjectionBranch:
    hand: tuple[str, ...]
    probability: float
    remaining_turns: int
    label: str
    risk: float = 0.0
    resource_changes: tuple[tuple[str, int], ...] = ()
    card_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillProjection:
    legal: bool
    action_ranks: tuple[str, ...]
    post_hand: tuple[str, ...]
    post_card_sources: tuple[str, ...]
    expected_remaining_turns: float
    worst_remaining_turns: int
    expected_remaining_cards: float
    worst_remaining_cards: int
    terminal: bool
    enemy_emergency_block: bool
    enemy_finish_risk: int
    expected_skill_risk: float
    worst_skill_risk: float
    ally_control_cost: int
    target_relation_cost: int
    external_skill_cost: float
    skill_resource_value: float
    control_card_cost: int
    high_card_cost: int
    triggered_rules: tuple[str, ...] = ()
    triggered_skills: tuple[str, ...] = ()
    resource_changes: tuple[tuple[str, float], ...] = ()
    random_branches: tuple[ProjectionBranch, ...] = ()
    choice: SkillChoice | None = None
    reason: str = ""
    rejection_reasons: tuple[str, ...] = ()

    @property
    def score_key(self) -> tuple[Any, ...]:
        deterministic_ranks = tuple(RANK_INDEX.get(rank, 99) for rank in self.action_ranks)
        return (
            0 if self.legal else 1,
            0 if self.terminal else 1,
            0 if self.enemy_emergency_block else 1,
            round(self.expected_remaining_turns, 6),
            self.worst_remaining_turns,
            round(self.expected_skill_risk, 6),
            round(self.worst_skill_risk, 6),
            self.ally_control_cost,
            self.enemy_finish_risk,
            self.target_relation_cost,
            round(self.external_skill_cost, 6),
            -round(self.skill_resource_value, 6),
            self.control_card_cost,
            self.high_card_cost,
            deterministic_ranks,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["score_key"] = list(self.score_key)
        value["random_branches"] = [branch.to_dict() for branch in self.random_branches]
        value["choice"] = self.choice.to_dict() if self.choice else None
        return value


def _action_payload(action: LegalAction | Sequence[str] | str | None) -> tuple[tuple[str, ...], str, str]:
    if isinstance(action, LegalAction):
        effective_ranks = action.parameters.get("effective_ranks")
        ranks = action.ranks if effective_ranks is None else tuple(str(rank) for rank in effective_ranks)
        return _sort_hand(ranks), action.action_type, action.kind
    if action is None or action == "" or action == "pass":
        return (), "none", "pass"
    ranks = tuple(action) if isinstance(action, str) else tuple(action)
    ranks = _sort_hand(ranks)
    return ranks, _infer_action_type(ranks), "play"


def _physical_action_ranks(action: LegalAction | Sequence[str] | str | None) -> tuple[str, ...]:
    if isinstance(action, LegalAction):
        physical_ranks = action.parameters.get("physical_ranks")
        ranks = action.ranks if physical_ranks is None else tuple(str(rank) for rank in physical_ranks)
        return _sort_hand(ranks)
    return _action_payload(action)[0]


def _infer_action_type(ranks: Sequence[str]) -> str:
    ranks = _sort_hand(ranks)
    if not ranks:
        return "none"
    counts = Counter(ranks)
    if len(ranks) == 1:
        return "solo"
    if len(ranks) == 2:
        if set(ranks) == {"X", "D"}:
            return "rocket"
        return "pair" if len(counts) == 1 else "invalid"
    if len(ranks) == 3 and len(counts) == 1:
        return "trio"
    if len(ranks) >= 4 and len(counts) == 1:
        return "bomb"
    if len(ranks) == 4 and sorted(counts.values()) == [1, 3]:
        return "trio_solo"
    if len(ranks) == 5 and sorted(counts.values()) == [2, 3]:
        return "trio_pair"
    natural = [RANK_INDEX[rank] for rank in ranks if rank in RANK_INDEX and rank not in CONTROL_RANKS]
    if len(ranks) >= 5 and len(counts) == len(ranks) and natural and natural == list(range(min(natural), min(natural) + len(natural))):
        return "straight"
    if len(ranks) >= 6 and len(ranks) % 2 == 0 and set(counts.values()) == {2}:
        values = sorted(RANK_INDEX[rank] for rank in counts)
        if values == list(range(values[0], values[0] + len(values))):
            return "pair_chain"
    if len(ranks) >= 6 and set(counts.values()) == {3}:
        return "airplane"
    return "other"


def _remove_action(hand: tuple[str, ...], action_ranks: tuple[str, ...]) -> tuple[str, ...] | None:
    remaining = list(hand)
    for rank in action_ranks:
        if rank not in remaining:
            return None
        remaining.remove(rank)
    return _sort_hand(remaining)


def estimate_remaining_turns(hand: tuple[str, ...]) -> int:
    if not hand:
        return 0
    if _infer_action_type(hand) not in {"invalid", "other", "none"}:
        return 1
    counts = Counter(hand)
    groups = len(counts)
    natural = tuple(rank for rank in CARD_ORDER[:12])

    def longest(minimum: int, length: int) -> int:
        best = current = 0
        for rank in natural:
            if counts[rank] >= minimum:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best if best >= length else 0

    sequence_saving = max(0, longest(1, 5) - 1, longest(2, 3) - 1, longest(3, 2) - 1)
    trio_groups = sum(count >= 3 for count in counts.values())
    attachments = sum(count in {1, 2} for count in counts.values())
    return max(1, groups - sequence_saving - min(trio_groups, attachments))


def _route_turns(hand: tuple[str, ...], evaluator: RouteEvaluator | None) -> int:
    if evaluator is None:
        return estimate_remaining_turns(hand)
    value = evaluator(hand)
    if isinstance(value, tuple):
        value = value[0]
    return max(0, int(math.ceil(float(value))))


def _history_action_types(context: HeroDecisionContext, actor_only: bool = False) -> tuple[str, ...]:
    values = []
    for event in context.history:
        if event.get("kind") != "play":
            continue
        if actor_only and event.get("actor") not in {None, context.position}:
            continue
        values.append(str(event.get("action_type", _infer_action_type(event.get("ranks", ())))))
    return tuple(values)


def _public_beaten_probability(context: HeroDecisionContext, action_ranks: tuple[str, ...], action_type: str) -> float:
    if not action_ranks or action_type in {"rocket", "bomb"}:
        return 0.0
    highest = max((RANK_INDEX.get(rank, 99) for rank in action_ranks), default=99)
    higher_public = sum(
        RANK_INDEX.get(rank, -1) > highest
        for event in context.history
        for rank in event.get("ranks", ())
    )
    possible_higher = max(0, RANK_INDEX.get("D", 14) - highest)
    enemy_pressure = min(context.enemy_card_counts or (17,))
    count_factor = min(1.0, max(0.2, enemy_pressure / 10.0))
    seen_factor = max(0.1, 1.0 - higher_public / max(1, possible_higher * 4))
    type_factor = 0.75 if action_type in {"solo", "pair"} else 0.35
    return round(min(0.9, possible_higher / 12.0 * count_factor * seen_factor * type_factor), 6)


def _first_own_play(context: HeroDecisionContext) -> bool:
    return not any(event.get("kind") == "play" and event.get("actor") == context.position for event in context.history)


def _trigger_probability(
    spec: HeroSkillSpec,
    context: HeroDecisionContext,
    action_ranks: tuple[str, ...],
    action_type: str,
    kind: str,
    resources: Mapping[str, int],
    post_hand: tuple[str, ...] | None = None,
) -> float:
    if spec.trigger not in SUPPORTED_TRIGGERS:
        raise KeyError(f"技能触发器没有注册处理器: {spec.trigger}")
    if spec.limit is not None and int(context.skill_uses.get(spec.name, 0)) >= spec.limit:
        return 0.0
    trigger = spec.trigger
    is_play = kind == "play" and bool(action_ranks)
    is_pass = kind == "pass"
    response = bool(context.table_cards)
    beaten_probability = _public_beaten_probability(context, action_ranks, action_type)
    prior_types = _history_action_types(context)
    own_types = _history_action_types(context, actor_only=True)
    new_type = action_type not in prior_types
    game_start = bool(context.extra.get("game_start_pending"))

    if trigger in DEFERRED_TRIGGERS and kind == "play" and post_hand is not None and not post_hand:
        return 0.0
    if spec.hero == "典韦" and spec.name == "不屈":
        remaining_count = len(post_hand) if post_hand is not None else len(context.hand) - len(action_ranks)
        if remaining_count < 3 or context.extra.get("不屈失效") or resources.get("unyielding", 0) >= 4:
            return 0.0
    if spec.hero == "陆逊" and spec.name in {"破蜀", "御魏"}:
        remaining_count = len(post_hand) if post_hand is not None else len(context.hand) - len(action_ranks)
        if remaining_count == 1:
            return 0.0

    if trigger == "after_play":
        return float(is_play)
    if trigger == "after_straight":
        return float(is_play and action_type == "straight")
    if trigger == "after_solo_or_pair":
        return float(is_play and action_type in {"solo", "pair"})
    if trigger == "after_trio_attachment":
        return float(is_play and action_type in {"trio_solo", "trio_pair"})
    if trigger == "after_beating":
        return float(is_play and response)
    if trigger == "response":
        return float(is_play and response)
    if trigger == "response_over_two":
        return float(is_play and response and len(action_ranks) >= 2)
    if trigger == "exact_plus_one_response_over_four":
        action_main = max((RANK_INDEX.get(rank, -1) for rank in action_ranks), default=-1)
        table_main = max((RANK_INDEX.get(rank, -1) for rank in context.table_cards), default=-1)
        return float(is_play and response and len(action_ranks) >= 4 and action_main == table_main + 1)
    if trigger == "after_response_or_pass":
        return float(response and (is_play or is_pass))
    if trigger == "repeat_action_type":
        last = context.extra.get("last_action_type") or (own_types[-1] if own_types else None)
        return float(is_play and action_type == last)
    if trigger == "any_action_over_four":
        return float(is_play and len(action_ranks) >= 4)
    if trigger == "any_all_below_five":
        return float(is_play and action_ranks and all(rank in {"3", "4", "5"} for rank in action_ranks))
    if trigger in {"new_action_type", "global_new_action_type"}:
        return float(is_play and new_type)
    if trigger == "new_action_type_contains_three":
        return float(is_play and new_type and "3" in action_ranks)
    if trigger == "play_contains_two":
        return float(is_play and "2" in action_ranks)
    if trigger == "play_qice_card":
        return float(is_play and bool(set(action_ranks) & set(context.extra.get("qice_ranks", ()))))
    if trigger == "after_play_under_six":
        return float(is_play and len(action_ranks) <= 6)
    if trigger == "after_xinghua":
        return float(bool(context.extra.get("xinghua_resolved")) or context.skill_uses.get("星华", 0) > 0)
    if trigger == "after_bomb":
        return float(is_play and action_type in {"bomb", "rocket"})
    if trigger == "three_same_action_types":
        return float(is_play and len(own_types) >= 2 and own_types[-2:] == (action_type, action_type))
    if trigger == "three_distinct_action_types":
        return float(is_play and len(own_types) >= 2 and len(set(own_types[-2:] + (action_type,))) == 3)
    if trigger in {"own_play_beaten", "low_solo_pair_beaten"}:
        if trigger == "low_solo_pair_beaten" and (action_type not in {"solo", "pair"} or any(RANK_INDEX.get(rank, 99) > RANK_INDEX["A"] for rank in action_ranks)):
            return 0.0
        return beaten_probability
    if trigger == "beaten_or_pass":
        return 1.0 if is_pass else beaten_probability
    if trigger == "after_play_and_unbeaten":
        return round(float(is_play) * (1.0 - beaten_probability), 6)
    if trigger == "largest_play_resolution":
        largest = max((RANK_INDEX.get(rank, -1) for rank in context.hand), default=-1)
        return float(is_play and largest in {RANK_INDEX.get(rank, -2) for rank in action_ranks})
    if trigger == "first_hand_below_eight":
        return float(is_play and _first_own_play(context) and len(action_ranks) <= 8)
    if trigger == "first_hand_below_twelve":
        return float(is_play and _first_own_play(context) and len(action_ranks) <= 12)
    if trigger == "game_start_or_joker_play":
        return float(game_start or (is_play and any(rank in {"X", "D"} for rank in action_ranks)))
    if trigger == "game_start_or_ambush_played":
        ambush_ranks = set(context.extra.get("ambush_ranks", ()))
        return float(game_start or bool(ambush_ranks & set(action_ranks)))
    if trigger == "game_start_or_unbeaten_play":
        return 1.0 if game_start else round(float(is_play) * (1.0 - beaten_probability), 6)
    if trigger == "pass_with_one":
        return float(is_pass and len(context.hand) == 1)
    if trigger == "pass_response":
        return float(is_pass and response)
    if trigger == "pass_under_four":
        return float(is_pass and response and len(context.table_cards) <= 4)
    if trigger in {"no_legal_response", "third_no_legal_response"}:
        no_response = bool(context.extra.get("no_legal_response", False))
        if trigger == "third_no_legal_response":
            prior = int(context.extra.get("no_legal_response_count", context.marks.get("无牌响应", 0)))
            return float(is_pass and response and no_response and prior + 1 >= 3)
        return float(is_pass and response and no_response)
    if trigger == "mark_changed":
        return float(resources.get("unyielding", context.marks.get("不屈", 0)) > context.marks.get("不屈", 0))
    if trigger == "petal_three_or_five":
        return float(resources.get("petal", context.marks.get("petal", 0)) in {3, 5})
    if trigger == "prestige_three":
        return float(resources.get("prestige", context.marks.get("prestige", 0)) >= 3)
    if trigger == "after_renyi_twice":
        return float(context.skill_uses.get("仁义", 0) >= 2)
    if trigger == "game_start":
        return float(game_start)
    if trigger in {"other_straight", "other_pair", "other_trio_attachment", "other_non_solo", "unbeaten_other_solo", "ambush_resolution", "lose_haofu_card", "bidding"}:
        return 0.0
    if trigger in {"active", "active_with_lead", "active_after_two_marks", "active_grudge_two", "active_without_trio_or_bomb"}:
        return 0.0
    return 0.0


TriggerHandler = Callable[[HeroSkillSpec, HeroDecisionContext, LegalAction | Sequence[str] | str | None], float]


def _make_trigger_handler(trigger: str) -> TriggerHandler:
    def handler(
        spec: HeroSkillSpec,
        context: HeroDecisionContext,
        action: LegalAction | Sequence[str] | str | None = None,
    ) -> float:
        if spec.trigger != trigger:
            raise ValueError(f"触发器 {trigger} 不能处理 {spec.trigger}")
        action_ranks, action_type, kind = _action_payload(action)
        return _trigger_probability(spec, context, action_ranks, action_type, kind, context.resources)

    handler.__name__ = f"trigger_{trigger}"
    return handler


TRIGGER_HANDLERS: dict[str, TriggerHandler] = {
    trigger: _make_trigger_handler(trigger) for trigger in sorted(SUPPORTED_TRIGGERS)
}


def _spec_for(context: HeroDecisionContext, skill: HeroSkillSpec | str) -> HeroSkillSpec:
    if isinstance(skill, HeroSkillSpec):
        return skill
    by_rule = skill_by_rule_id(skill)
    if by_rule is not None:
        return by_rule
    for spec in HERO_REGISTRY.get(context.hero or "", ()):
        if spec.name == skill or spec.effect == skill:
            return spec
    raise KeyError(f"武将 {context.hero or '未知'} 没有技能: {skill}")


def _choice(
    spec: HeroSkillSpec,
    kind: str,
    index: int,
    *,
    activate: bool = True,
    card_ids: Sequence[str] = (),
    ranks: Sequence[str] = (),
    target: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> SkillChoice:
    return SkillChoice(
        choice_id=f"{spec.rule_id}:{kind}:{index}",
        rule_id=spec.rule_id,
        skill=spec.name,
        effect=spec.effect,
        kind=kind,
        activate=activate,
        card_ids=tuple(card_ids),
        ranks=_sort_hand(ranks),
        target=target,
        parameters=dict(parameters or {}),
        requires_ui_verification=bool(spec.interactive and not spec.ui_verified),
    )


def _skill_choice_window_available(
    spec: HeroSkillSpec,
    context: HeroDecisionContext,
    action: LegalAction | Sequence[str] | str | None,
) -> bool:
    pending = context.extra.get("pending_interaction")
    if isinstance(pending, Mapping) and pending.get("skill") == spec.name:
        return True
    trigger = spec.trigger
    if trigger == "active":
        if spec.effect == "spend_food_for_jokers_or_wildcard":
            return context.resources.get("food", 0) > 0
        return True
    if trigger == "active_with_lead":
        return not context.table_cards
    if trigger == "active_after_two_marks":
        return context.resources.get("enemy_rank_mark", 0) >= 2
    if trigger == "active_grudge_two":
        return context.resources.get("grudge", 0) >= 2
    if trigger == "active_without_trio_or_bomb":
        counts = Counter(context.hand)
        return not any(count >= 3 for count in counts.values()) and not ({"X", "D"} <= set(context.hand))
    action_ranks, action_type, kind = _action_payload(action)
    return _trigger_probability(spec, context, action_ranks, action_type, kind, context.resources) > 0


def enumerate_skill_choices(
    context: HeroDecisionContext,
    skill: HeroSkillSpec | str,
    action: LegalAction | Sequence[str] | str | None = None,
) -> tuple[SkillChoice, ...]:
    """Enumerate every public legal choice, including skip for optional skills."""

    spec = _spec_for(context, skill)
    if spec.limit is not None and int(context.skill_uses.get(spec.name, 0)) >= spec.limit:
        return ()
    profile = EFFECT_PROFILES[spec.effect]
    pending = context.extra.get("pending_interaction")
    choices: list[SkillChoice] = []
    pending_matches = isinstance(pending, Mapping) and pending.get("skill") == spec.name
    if not _skill_choice_window_available(spec, context, action):
        if spec.optional or profile.optional:
            return (_choice(spec, "skip", 0, activate=False, parameters={"skip": True}),)
        return ()
    if not pending_matches and not context.extra.get("selection_hand_is_post_action"):
        played_ranks = _physical_action_ranks(action)
        _, _, action_kind = _action_payload(action)
        if action_kind == "play" and played_ranks:
            ids = context.hand_card_ids or tuple(f"h{index}" for index in range(len(context.hand)))
            sources = context.card_sources or tuple("deck" for _ in context.hand)
            remaining_cards = list(zip(context.hand, ids, sources))
            legal_selection = True
            for rank in played_ranks:
                index = next(
                    (index for index, (value, _, _) in enumerate(remaining_cards) if value == rank),
                    None,
                )
                if index is None:
                    legal_selection = False
                    break
                remaining_cards.pop(index)
            if legal_selection:
                context = replace(
                    context,
                    hand=tuple(card[0] for card in remaining_cards),
                    hand_card_ids=tuple(card[1] for card in remaining_cards),
                    card_sources=tuple(card[2] for card in remaining_cards),
                )
    if pending_matches:
        for index, option in enumerate(pending.get("options", ())):
            if not isinstance(option, Mapping):
                continue
            choices.append(
                _choice(
                    spec,
                    profile.choice_kind,
                    index,
                    card_ids=option.get("card_ids", ()),
                    ranks=option.get("ranks", (option["rank"],) if option.get("rank") else ()),
                    target=option.get("target"),
                    parameters=option,
                )
            )
    elif spec.effect in {
        "gain_two_discard_one",
        "take_beaten_action",
        "take_beating_action_except_bomb",
        "take_one_beaten_card",
    }:
        choices.append(_choice(spec, "activate", 0))
    elif spec.effect == "discard_opposite_group":
        _, action_type, _ = _action_payload(action)
        counts = Counter(context.hand)
        ids = context.hand_card_ids or tuple(f"h{index}" for index in range(len(context.hand)))
        if action_type == "solo":
            for rank, count in counts.items():
                if count < 2:
                    continue
                indexes = tuple(index for index, value in enumerate(context.hand) if value == rank)[:2]
                choices.append(
                    _choice(
                        spec,
                        "cards",
                        len(choices),
                        card_ids=tuple(ids[index] for index in indexes),
                        ranks=(rank, rank),
                    )
                )
        elif action_type == "pair":
            for index, rank in enumerate(context.hand):
                if counts[rank] == 1:
                    choices.append(
                        _choice(spec, "cards", len(choices), card_ids=(ids[index],), ranks=(rank,))
                    )
    elif spec.effect == "convert_solo_pair":
        counts = Counter(context.hand)
        ids = context.hand_card_ids or tuple(f"h{index}" for index in range(len(context.hand)))
        for rank, count in counts.items():
            indexes = tuple(index for index, value in enumerate(context.hand) if value == rank)
            if count == 1:
                choices.append(
                    _choice(
                        spec,
                        "cards",
                        len(choices),
                        card_ids=(ids[indexes[0]],),
                        ranks=(rank,),
                        parameters={"operation": "solo_to_pair", "rank": rank},
                    )
                )
            elif count == 2:
                choices.append(
                    _choice(
                        spec,
                        "cards",
                        len(choices),
                        card_ids=(ids[indexes[0]],),
                        ranks=(rank,),
                        parameters={"operation": "pair_to_solo", "rank": rank},
                    )
                )
    elif profile.choice_kind == "cards":
        max_cards = min(3, len(context.hand))
        if spec.effect in {
            "copy_one_and_reset_on_mixed_large_play",
            "discard_one",
            "take_one_beaten_card",
            "take_one_from_straight",
        }:
            sizes = (1,)
        elif spec.effect in {"discard_three_recover_highest", "discard_sum_twelve_gain_above_q"}:
            sizes = (3,)
        else:
            sizes = tuple(range(1, max_cards + 1))
        ids = context.hand_card_ids or tuple(f"h{index}" for index in range(len(context.hand)))
        for size in sizes:
            for indexes in combinations(range(len(context.hand)), size):
                ranks = tuple(context.hand[index] for index in indexes)
                if spec.effect == "discard_sum_twelve_gain_above_q":
                    numeric = {**{str(value): value for value in range(3, 10)}, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15, "X": 16, "D": 17, "W": 0}
                    if sum(numeric.get(rank, 0) for rank in ranks) != 12:
                        continue
                choices.append(_choice(spec, "cards", len(choices), card_ids=tuple(ids[index] for index in indexes), ranks=ranks))
    elif profile.choice_kind == "rank":
        source = context.table_cards if spec.effect == "take_one_from_straight" else context.hand
        for rank in dict.fromkeys(source):
            choices.append(_choice(spec, "rank", len(choices), ranks=(rank,), parameters={"rank": rank}))
    elif profile.choice_kind == "player":
        if spec.effect == "swap_identity_with_landlord_gain_two_twos":
            targets = (context.landlord,) if context.position != context.landlord else ()
        elif spec.effect == "discard_self_or_other":
            targets = (context.position,) + context.enemies + context.allies
        else:
            targets = context.enemies + context.allies
        card_sizes: tuple[int, ...] = ()
        if spec.effect == "discard_two_largest_transfer_lead":
            card_sizes = (2,) if len(context.hand) >= 2 else ()
        elif spec.effect == "give_up_to_two":
            card_sizes = tuple(size for size in (1, 2) if size <= len(context.hand))
        elif spec.effect in {"exchange_low_for_largest_non_joker", "give_high_card"}:
            card_sizes = (1,) if context.hand else ()
        ids = context.hand_card_ids or tuple(f"h{index}" for index in range(len(context.hand)))
        for target in targets:
            if not card_sizes:
                choices.append(_choice(spec, "player", len(choices), target=target, parameters={"target": target}))
                continue
            for size in card_sizes:
                if spec.effect == "discard_two_largest_transfer_lead":
                    indexes_options = (
                        tuple(
                            sorted(
                                range(len(context.hand)),
                                key=lambda index: (RANK_INDEX.get(context.hand[index], -1), index),
                                reverse=True,
                            )[:2]
                        ),
                    )
                else:
                    indexes_options = combinations(range(len(context.hand)), size)
                for indexes in indexes_options:
                    ranks = tuple(context.hand[index] for index in indexes)
                    choices.append(
                        _choice(
                            spec,
                            "player",
                            len(choices),
                            card_ids=tuple(ids[index] for index in indexes),
                            ranks=ranks,
                            target=target,
                            parameters={"target": target},
                        )
                    )
    else:
        choices.append(_choice(spec, "activate", 0))

    allow_skip = bool(pending.get("optional")) if pending_matches else bool(spec.optional or profile.optional)
    if allow_skip:
        choices.append(_choice(spec, "skip", len(choices), activate=False, parameters={"skip": True}))
    return tuple(choices)


def _project_card_sources(
    context: HeroDecisionContext,
    action_ranks: tuple[str, ...],
    post_hand: tuple[str, ...],
    skill_source: str,
) -> tuple[str, ...]:
    sources = context.card_sources or tuple("deck" for _ in context.hand)
    remaining = list(zip(context.hand, sources))
    for rank in action_ranks:
        index = next((index for index, (value, _) in enumerate(remaining) if value == rank), None)
        if index is not None:
            remaining.pop(index)
    assigned: list[str | None] = [None] * len(post_hand)
    for output_index, rank in enumerate(post_hand):
        index = next((index for index, (value, _) in enumerate(remaining) if value == rank), None)
        if index is not None:
            _, source = remaining.pop(index)
            assigned[output_index] = source
    # Rank transforms retain an existing entity source; only net-new cards receive a skill source.
    for output_index, source in enumerate(assigned):
        if source is None and remaining:
            _, retained_source = remaining.pop(0)
            assigned[output_index] = retained_source
    return tuple(source or skill_source for source in assigned)


def _branches_to_projection(
    context: HeroDecisionContext,
    action_ranks: tuple[str, ...],
    outcomes: Sequence[EffectOutcome],
    *,
    route_evaluator: RouteEvaluator | None,
    physical_action_ranks: tuple[str, ...] | None = None,
    choice: SkillChoice | None = None,
    triggered_rules: Sequence[str] = (),
    triggered_skills: Sequence[str] = (),
    resource_value: float = 0.0,
    target_relation_cost: int = 0,
    external_skill_cost: float = 0.0,
    legal: bool = True,
    kind: str = "skill",
) -> SkillProjection:
    if not outcomes:
        outcomes = (EffectOutcome(context.hand),)
    physical_ranks = action_ranks if physical_action_ranks is None else physical_action_ranks
    total_probability = sum(max(0.0, outcome.probability) for outcome in outcomes) or 1.0
    source_label = f"skill:{triggered_rules[-1]}" if triggered_rules else "skill:projection"
    branches = tuple(
        ProjectionBranch(
            hand=_sort_hand(outcome.hand),
            probability=max(0.0, outcome.probability) / total_probability,
            remaining_turns=_route_turns(_sort_hand(outcome.hand), route_evaluator),
            label=outcome.label,
            risk=outcome.risk,
            resource_changes=outcome.resource_changes,
            card_sources=_project_card_sources(context, physical_ranks, _sort_hand(outcome.hand), source_label),
        )
        for outcome in outcomes
    )
    expected_turns = sum(branch.probability * branch.remaining_turns for branch in branches)
    expected_cards = sum(branch.probability * len(branch.hand) for branch in branches)
    expected_risk = sum(branch.probability * branch.risk for branch in branches)
    worst_risk = max(branch.risk for branch in branches)
    representative = min(branches, key=lambda branch: (branch.remaining_turns, len(branch.hand), branch.hand))
    resources: Counter[str] = Counter()
    for branch in branches:
        for key, delta in branch.resource_changes:
            resources[key] += branch.probability * delta
    enemy_min = min(context.enemy_card_counts or (17,))
    is_response = bool(context.table_cards)
    next_position = POSITIONS[(POSITIONS.index(context.position) + 1) % len(POSITIONS)]
    enemy_responds_next = next_position in context.enemies
    enemy_block = bool(
        enemy_min <= 3
        and kind == "play"
        and (
            not is_response
            or context.table_relation != "ally"
            or enemy_responds_next
        )
    )
    ally_cost = int(kind == "play" and is_response and context.table_relation == "ally" and bool(action_ranks))
    finish_risk = int(kind == "play" and not context.table_cards and len(action_ranks) in set(context.enemy_card_counts))
    control_cost = sum(rank in CONTROL_RANKS for rank in physical_ranks) if kind == "play" else 0
    high_cost = max((RANK_INDEX.get(rank, 99) for rank in physical_ranks), default=0) if kind == "play" else 0
    terminal = bool(branches and all(not branch.hand for branch in branches))
    reason = "技能结算后直接胜利" if terminal else f"技能结算后预计{expected_turns:.2f}手，最坏{max(branch.remaining_turns for branch in branches)}手"
    return SkillProjection(
        legal=legal,
        action_ranks=action_ranks,
        post_hand=representative.hand,
        post_card_sources=representative.card_sources,
        expected_remaining_turns=round(expected_turns, 6),
        worst_remaining_turns=max(branch.remaining_turns for branch in branches),
        expected_remaining_cards=round(expected_cards, 6),
        worst_remaining_cards=max(len(branch.hand) for branch in branches),
        terminal=terminal,
        enemy_emergency_block=enemy_block,
        enemy_finish_risk=finish_risk,
        expected_skill_risk=round(expected_risk, 6),
        worst_skill_risk=round(worst_risk, 6),
        ally_control_cost=ally_cost,
        target_relation_cost=target_relation_cost,
        external_skill_cost=round(external_skill_cost, 6),
        skill_resource_value=resource_value,
        control_card_cost=control_cost,
        high_card_cost=high_cost,
        triggered_rules=tuple(triggered_rules),
        triggered_skills=tuple(triggered_skills),
        resource_changes=tuple(sorted(resources.items())),
        random_branches=branches,
        choice=choice,
        reason=reason,
    )


def _target_relation_cost(context: HeroDecisionContext, choice: SkillChoice) -> int:
    if choice.target is None:
        return 0
    relation = "self" if choice.target == context.position else "ally" if choice.target in context.allies else "enemy" if choice.target in context.enemies else "unknown"
    if choice.effect == "swap_identity_with_landlord_gain_two_twos":
        return 0 if choice.target == context.landlord else 3
    if choice.effect in BENEFICIAL_TARGET_EFFECTS:
        return 0 if relation == "ally" else 1 if relation == "self" else 3
    if choice.effect in HARMFUL_TARGET_EFFECTS:
        return 0 if relation == "enemy" else 1 if relation == "self" else 3
    return 0 if relation in {"self", "enemy"} else 1


def _external_skill_impact(
    context: HeroDecisionContext,
    action_ranks: tuple[str, ...],
    action_type: str,
    kind: str,
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """Estimate only publicly triggered effects belonging to other players."""

    if kind != "play" or not action_ranks:
        return 0.0, (), ()
    prior_types = _history_action_types(context)
    rules: list[str] = []
    skills: list[str] = []
    total_cost = 0.0
    uses_by_seat = context.public_skill_uses
    for seat, hero in context.public_heroes:
        if seat == context.position or hero not in HERO_REGISTRY:
            continue
        relation_sign = -1.0 if seat in context.allies else 1.0 if seat in context.enemies else 0.0
        if relation_sign == 0.0:
            continue
        for spec in HERO_REGISTRY[hero]:
            uses = int(uses_by_seat.get(seat, {}).get(spec.name, 0))
            if spec.limit is not None and uses >= spec.limit:
                continue
            triggered = False
            probability = 1.0
            if spec.trigger == "any_action_over_four":
                triggered = len(action_ranks) >= 4
            elif spec.trigger == "any_all_below_five":
                triggered = all(rank in {"3", "4", "5"} for rank in action_ranks)
            elif spec.trigger == "global_new_action_type":
                triggered = action_type not in prior_types
            elif spec.trigger == "other_pair":
                triggered = action_type == "pair"
            elif spec.trigger == "other_straight":
                triggered = action_type == "straight"
            elif spec.trigger == "other_trio_attachment":
                triggered = action_type in {"trio_solo", "trio_pair"}
            elif spec.trigger == "other_non_solo":
                triggered = action_type != "solo"
            elif spec.trigger == "unbeaten_other_solo":
                triggered = action_type == "solo"
                probability = 1.0 - _public_beaten_probability(context, action_ranks, action_type)
            if not triggered or probability <= 0.0:
                continue
            profile = EFFECT_PROFILES[spec.effect]
            structural_value = 0.5
            if profile.operation in {"gain", "take"}:
                structural_value += 0.5
            if profile.operation == "resource":
                structural_value += 0.35
            structural_value += min(1.0, max(0, profile.max_hand_delta) * 0.1)
            total_cost += relation_sign * probability * structural_value
            rules.append(spec.rule_id)
            skills.append(f"{seat}:{spec.name}")
    return round(total_cost, 6), tuple(rules), tuple(skills)


def apply_skill_choice(
    context: HeroDecisionContext,
    choice: SkillChoice,
    *,
    base_hand: Sequence[str] | None = None,
    action: LegalAction | Sequence[str] | str | None = None,
    route_evaluator: RouteEvaluator | None = None,
) -> SkillProjection:
    """Apply a complete skill choice without mutating the caller's game state."""

    action_ranks, _, kind = _action_payload(action)
    hand = _sort_hand(base_hand if base_hand is not None else context.hand)
    if choice.skip:
        return _branches_to_projection(
            context,
            action_ranks,
            (EffectOutcome(hand, label=f"{choice.skill}:skip"),),
            route_evaluator=route_evaluator,
            choice=choice,
            kind=kind,
        )
    spec = _spec_for(context, choice.rule_id)
    pending = context.extra.get("pending_interaction")
    if isinstance(pending, Mapping) and pending.get("skill") == spec.name:
        pending_effect = str(pending.get("effect", ""))
        selected = choice.ranks or tuple(str(rank) for rank in choice.parameters.get("ranks", ()))
        if not selected and choice.parameters.get("rank"):
            selected = (str(choice.parameters["rank"]),)

        def remove_exact(cards: tuple[str, ...], requested: tuple[str, ...]) -> tuple[str, ...] | None:
            if not (Counter(requested) <= Counter(cards)):
                return None
            remaining = list(cards)
            for rank in requested:
                remaining.remove(rank)
            return _sort_hand(remaining)

        post_hand: tuple[str, ...] | None
        if pending_effect in {"discard_one", "discard_group"}:
            post_hand = remove_exact(hand, selected)
        elif pending_effect in {"copy_bottom", "gain_rank"}:
            post_hand = _sort_hand(hand + selected[:1]) if selected else None
        elif pending_effect == "take_cards":
            post_hand = _sort_hand(hand + selected) if selected else None
        elif pending_effect == "fill_both_largest":
            largest = max(hand, key=lambda value: RANK_INDEX.get(value, -1), default=None)
            post_hand = _sort_hand(hand + (largest,) * max(0, 3 - hand.count(largest))) if largest else hand
        elif pending_effect == "convert_group":
            operation = str(choice.parameters.get("operation", ""))
            if operation == "solo_to_pair" and selected and hand.count(selected[0]) == 1:
                post_hand = _sort_hand(hand + (selected[0],))
            elif operation == "pair_to_solo" and selected and hand.count(selected[0]) == 2:
                post_hand = remove_exact(hand, (selected[0],))
            else:
                post_hand = None
        elif pending_effect == "ganglie":
            discard_ranks = tuple(str(rank) for rank in choice.parameters.get("discard_ranks", ()))
            if discard_ranks:
                post_discard = remove_exact(hand, discard_ranks)
                recover_rank = str(choice.parameters.get("recover_rank", ""))
                recovered = (recover_rank,) if recover_rank in RANKS else ()
                post_hand = _sort_hand(post_discard + recovered) if post_discard is not None else None
            else:
                # The authoritative engine's existing interaction supplies only the recovered
                # rank because its UI-independent baseline always discards the three lowest.
                recovered = selected[:1]
                post_hand = _sort_hand(hand[min(3, len(hand)) :] + recovered) if recovered else None
        else:
            post_hand = None
        if post_hand is None:
            return _branches_to_projection(
                context,
                action_ranks,
                (EffectOutcome(hand, label=f"{spec.name}:illegal_choice", risk=1.0),),
                route_evaluator=route_evaluator,
                choice=choice,
                legal=False,
                kind=kind,
            )
        profile = EFFECT_PROFILES[spec.effect]
        return _branches_to_projection(
            context,
            action_ranks,
            (EffectOutcome(post_hand, label=f"{spec.name}:{pending_effect}"),),
            route_evaluator=route_evaluator,
            choice=choice,
            triggered_rules=(spec.rule_id,),
            triggered_skills=(spec.name,),
            resource_value=profile.future_value,
            target_relation_cost=_target_relation_cost(context, choice),
            kind=kind,
        )

    hand_selected_effects = {
        "convert_high_to_wildcard_and_give",
        "decrease_one_rank_by_two",
        "discard_group_reduce_enemy_max",
        "discard_increasing_count",
        "discard_matching_ranks",
        "discard_one",
        "discard_opposite_group",
        "discard_same_count_at_least_k_transform_previous_to_3334_or_34567_except_bomb",
        "discard_self_or_other",
        "discard_sum_twelve_gain_above_q",
        "discard_two_largest_transfer_lead",
        "exchange_low_for_largest_non_joker",
        "give_high_card",
        "give_up_to_two",
        "increase_chosen_card",
    }
    if choice.ranks and spec.effect in hand_selected_effects and not (Counter(choice.ranks) <= Counter(hand)):
        return _branches_to_projection(
            context,
            action_ranks,
            (EffectOutcome(hand, label=f"{spec.name}:illegal_choice", risk=1.0),),
            route_evaluator=route_evaluator,
            choice=choice,
            legal=False,
            kind=kind,
        )
    handler = EFFECT_HANDLERS[spec.effect]
    payload = dict(choice.parameters)
    if choice.ranks:
        payload["ranks"] = choice.ranks
    if choice.card_ids:
        payload["card_ids"] = choice.card_ids
    if choice.target:
        payload["target"] = choice.target
    outcomes = handler(
        hand,
        action_ranks=action_ranks,
        table_ranks=action_ranks if spec.effect == "discard_three_recover_highest" else context.table_cards,
        choice=payload,
        resources=context.resources,
        seed=context.seed,
    )
    profile = EFFECT_PROFILES[spec.effect]
    relation_cost = _target_relation_cost(context, choice)
    return _branches_to_projection(
        context,
        action_ranks,
        outcomes,
        route_evaluator=route_evaluator,
        choice=choice,
        triggered_rules=(spec.rule_id,),
        triggered_skills=(spec.name,),
        resource_value=profile.future_value + max(0, profile.resource_delta) * 0.25,
        target_relation_cost=relation_cost,
        kind=kind,
    )


def select_skill_choice(
    context: HeroDecisionContext,
    skill: HeroSkillSpec | str,
    action: LegalAction | Sequence[str] | str | None = None,
    *,
    route_evaluator: RouteEvaluator | None = None,
    live: bool = False,
) -> SkillChoice | None:
    """Choose the best complete resolution; unverified live UI cancels or pauses."""

    spec = _spec_for(context, skill)
    choices = enumerate_skill_choices(context, spec, action)
    if not choices:
        return None
    if live and (not spec.ui_verified or not spec.live_verified):
        return next((choice for choice in choices if choice.skip), None)
    action_ranks = _physical_action_ranks(action)
    _, _, action_kind = _action_payload(action)
    base_hand = _remove_action(context.hand, action_ranks) if action_kind == "play" else context.hand
    if base_hand is None:
        return None
    projections = tuple(
        apply_skill_choice(
            context,
            choice,
            base_hand=base_hand,
            action=action,
            route_evaluator=route_evaluator,
        )
        for choice in choices
    )
    best_index = min(range(len(choices)), key=lambda index: (projections[index].score_key, choices[index].choice_id))
    return choices[best_index]


def _choose_optional_outcomes(
    context: HeroDecisionContext,
    spec: HeroSkillSpec,
    hand: tuple[str, ...],
    action: LegalAction | Sequence[str] | str | None,
    route_evaluator: RouteEvaluator | None,
) -> tuple[EffectOutcome, ...]:
    if spec.effect == "inspect_and_take_two_lowest" and not isinstance(context.extra.get("pending_interaction"), Mapping):
        return (EffectOutcome(hand, label="游侠:等待公开最低牌"),)
    choices = enumerate_skill_choices(context, spec, action)
    candidates: list[tuple[tuple[Any, ...], tuple[EffectOutcome, ...]]] = []
    for choice in choices:
        projection = apply_skill_choice(
            context,
            choice,
            base_hand=hand,
            action=action,
            route_evaluator=route_evaluator,
        )
        outcomes = tuple(
            EffectOutcome(
                branch.hand,
                branch.resource_changes,
                branch.probability,
                branch.label,
                branch.risk,
            )
            for branch in projection.random_branches
        )
        candidates.append((projection.score_key + (choice.choice_id,), outcomes))
    if not candidates:
        return (EffectOutcome(hand, label=f"{spec.name}:skip"),)
    return min(candidates, key=lambda item: item[0])[1]


@dataclass
class _EvaluationBranch:
    hand: tuple[str, ...]
    resources: dict[str, int]
    resource_changes: tuple[tuple[str, int], ...] = ()
    probability: float = 1.0
    label: str = "base"
    risk: float = 0.0
    resource_value: float = 0.0


def evaluate_play(
    context: HeroDecisionContext,
    action: LegalAction | Sequence[str] | str | None,
    *,
    route_evaluator: RouteEvaluator | None = None,
) -> SkillProjection:
    """Project a play through every registered skill before scoring it."""

    action_ranks, action_type, kind = _action_payload(action)
    physical_ranks = _physical_action_ranks(action)
    remaining = _remove_action(context.hand, physical_ranks) if kind == "play" else context.hand
    invalid_pass = kind == "pass" and not context.table_cards
    if remaining is None or action_type == "invalid" or invalid_pass:
        return _branches_to_projection(
            context,
            action_ranks,
            (EffectOutcome(context.hand, label="illegal_action", risk=1.0),),
            route_evaluator=route_evaluator,
            physical_action_ranks=physical_ranks,
            legal=False,
            kind=kind,
        )
    external_cost, external_rules, external_skills = _external_skill_impact(
        context, action_ranks, action_type, kind
    )
    if context.hero not in HERO_REGISTRY:
        return _branches_to_projection(
            context,
            action_ranks,
            (EffectOutcome(remaining, label="no_hero"),),
            route_evaluator=route_evaluator,
            physical_action_ranks=physical_ranks,
            triggered_rules=external_rules,
            triggered_skills=external_skills,
            external_skill_cost=external_cost,
            kind=kind,
        )

    branches = [_EvaluationBranch(remaining, dict(context.resources))]
    triggered_rules: list[str] = []
    triggered_skills: list[str] = []
    for spec in HERO_REGISTRY[context.hero]:
        profile = EFFECT_PROFILES[spec.effect]
        next_branches: list[_EvaluationBranch] = []
        triggered_any = False
        for branch in branches:
            probability = _trigger_probability(
                spec,
                context,
                action_ranks,
                action_type,
                kind,
                branch.resources,
                branch.hand,
            )
            if probability <= 0.0:
                next_branches.append(branch)
                continue
            triggered_any = True
            if probability < 1.0:
                next_branches.append(
                    _EvaluationBranch(
                        branch.hand,
                        dict(branch.resources),
                        branch.resource_changes,
                        branch.probability * (1.0 - probability),
                        f"{branch.label}|{spec.name}:not_triggered",
                        branch.risk,
                        branch.resource_value,
                    )
                )
            if spec.optional:
                branch_context = replace(
                    context,
                    hand=branch.hand,
                    hand_card_ids=(),
                    card_sources=(),
                    extra={
                        **context.extra,
                        "resources": dict(branch.resources),
                        "selection_hand_is_post_action": True,
                    },
                )
                outcomes = _choose_optional_outcomes(
                    branch_context, spec, branch.hand, action, route_evaluator
                )
            else:
                outcomes = EFFECT_HANDLERS[spec.effect](
                    branch.hand,
                    action_ranks=action_ranks,
                    table_ranks=(
                        action_ranks
                        if spec.effect == "discard_three_recover_highest"
                        else context.table_cards
                    ),
                    resources=branch.resources,
                    seed=context.seed,
                )
            for outcome in outcomes:
                next_resources = dict(branch.resources)
                for resource, delta in outcome.resource_changes:
                    next_resources[resource] = next_resources.get(resource, 0) + int(delta)
                outcome_resource_value = profile.future_value + sum(
                    max(0, delta) * 0.25 for _, delta in outcome.resource_changes
                )
                next_branches.append(
                    _EvaluationBranch(
                        outcome.hand,
                        next_resources,
                        tuple(branch.resource_changes) + tuple(outcome.resource_changes),
                        branch.probability * probability * outcome.probability,
                        f"{branch.label}|{outcome.label}",
                        max(branch.risk, outcome.risk),
                        branch.resource_value + outcome_resource_value,
                    )
                )
        branches = next_branches
        if triggered_any:
            triggered_rules.append(spec.rule_id)
            triggered_skills.append(spec.name)

    total_probability = sum(max(0.0, branch.probability) for branch in branches) or 1.0
    resource_value = sum(
        max(0.0, branch.probability) / total_probability * branch.resource_value
        for branch in branches
    )
    outcomes = tuple(
        EffectOutcome(
            branch.hand,
            branch.resource_changes,
            branch.probability,
            branch.label,
            branch.risk,
        )
        for branch in branches
    )

    return _branches_to_projection(
        context,
        action_ranks,
        outcomes,
        route_evaluator=route_evaluator,
        physical_action_ranks=physical_ranks,
        triggered_rules=tuple(triggered_rules) + external_rules,
        triggered_skills=tuple(triggered_skills) + external_skills,
        resource_value=resource_value,
        external_skill_cost=external_cost,
        kind=kind,
    )


def evaluate_actions(
    context: HeroDecisionContext,
    actions: Sequence[LegalAction | Sequence[str] | str],
    *,
    route_evaluator: RouteEvaluator | None = None,
) -> tuple[SkillProjection, ...]:
    return tuple(evaluate_legal_action(context, action, route_evaluator=route_evaluator) for action in actions)


def evaluate_legal_action(
    context: HeroDecisionContext,
    action: LegalAction | Sequence[str] | str,
    *,
    route_evaluator: RouteEvaluator | None = None,
) -> SkillProjection:
    if not isinstance(action, LegalAction) or action.kind in {"play", "pass"}:
        return evaluate_play(context, action, route_evaluator=route_evaluator)
    if not action.skill:
        return evaluate_play(context, action, route_evaluator=route_evaluator)
    spec = _spec_for(context, action.skill)
    skip = bool(action.parameters.get("skip"))
    choice = _choice(
        spec,
        "skip" if skip else EFFECT_PROFILES[spec.effect].choice_kind,
        int(action.parameters.get("option_index", 0)),
        activate=not skip,
        card_ids=action.card_ids,
        ranks=action.ranks,
        target=action.target,
        parameters=action.parameters,
    )
    return apply_skill_choice(context, choice, action=action, route_evaluator=route_evaluator)


def select_best_action(
    context: HeroDecisionContext,
    actions: Sequence[LegalAction],
    *,
    route_evaluator: RouteEvaluator | None = None,
) -> tuple[LegalAction, SkillProjection, tuple[SkillProjection, ...]]:
    if not actions:
        raise ValueError("武将策略没有收到合法动作")
    projections = evaluate_actions(context, actions, route_evaluator=route_evaluator)
    best_index = min(range(len(actions)), key=lambda index: (projections[index].score_key, actions[index].action_id))
    return actions[best_index], projections[best_index], projections


def trigger_handler_for(trigger: str) -> TriggerHandler:
    try:
        return TRIGGER_HANDLERS[trigger]
    except KeyError as exc:
        raise KeyError(f"技能触发器没有注册处理器: {trigger}") from exc


def validate_policy_contract() -> tuple[str, ...]:
    errors: list[str] = []
    registered_effects = {skill.effect for skills in HERO_REGISTRY.values() for skill in skills}
    if registered_effects != set(EFFECT_HANDLERS):
        errors.append("effect_handlers")
    if _REGISTERED_TRIGGERS != set(TRIGGER_HANDLERS):
        errors.append("trigger_handlers")
    rule_ids = [skill.rule_id for skills in HERO_REGISTRY.values() for skill in skills]
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("duplicate_rule_ids")
    for hero, skills in HERO_REGISTRY.items():
        context = HeroDecisionContext(hand=("3", "4"), hero=hero)
        for spec in skills:
            if spec.effect not in EFFECT_PROFILES:
                errors.append(f"missing_projection:{spec.rule_id}")
            try:
                enumerate_skill_choices(context, spec)
            except Exception as exc:  # pragma: no cover - returned as a registry diagnostic
                errors.append(f"choice_error:{spec.rule_id}:{type(exc).__name__}")
    return tuple(errors)


__all__ = [
    "HeroDecisionContext",
    "ProjectionBranch",
    "SkillChoice",
    "SkillProjection",
    "TRIGGER_HANDLERS",
    "apply_skill_choice",
    "enumerate_skill_choices",
    "estimate_remaining_turns",
    "evaluate_actions",
    "evaluate_legal_action",
    "evaluate_play",
    "select_best_action",
    "select_skill_choice",
    "trigger_handler_for",
    "validate_policy_contract",
]
