from __future__ import annotations

import random
from typing import Protocol

from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.hero_policy import HeroDecisionContext, select_best_action
from ok_tasks.card_ai.heroes import AUTHORITATIVE_RULE_IDS, HERO_REGISTRY
from ok_tasks.card_ai.rules import estimate_route_turns, rank_key
from ok_tasks.card_ai.schema import LegalAction, POSITIONS
from ok_tasks.card_ai.decision.context import DecisionContext
from ok_tasks.card_ai.decision.core import DecisionPolicyCore


_DECISION_CORE = DecisionPolicyCore()


class Policy(Protocol):
    policy_id: str

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction: ...


def _projection_safely_dominates(candidate, baseline) -> bool:
    if not candidate.legal:
        return False
    if not baseline.legal:
        return True
    if candidate.terminal != baseline.terminal:
        return candidate.terminal
    if candidate.terminal:
        return True
    if candidate.enemy_emergency_block != baseline.enemy_emergency_block:
        return candidate.enemy_emergency_block

    route_no_worse = (
        candidate.expected_remaining_turns <= baseline.expected_remaining_turns
        and candidate.worst_remaining_turns <= baseline.worst_remaining_turns
        and candidate.expected_remaining_cards <= baseline.expected_remaining_cards
        and candidate.worst_remaining_cards <= baseline.worst_remaining_cards
        and candidate.expected_skill_risk <= baseline.expected_skill_risk
        and candidate.worst_skill_risk <= baseline.worst_skill_risk
        and candidate.ally_control_cost <= baseline.ally_control_cost
        and candidate.enemy_finish_risk <= baseline.enemy_finish_risk
        and candidate.target_relation_cost <= baseline.target_relation_cost
        and candidate.external_skill_cost <= baseline.external_skill_cost
    )
    fully_no_worse = (
        route_no_worse
        and candidate.skill_resource_value >= baseline.skill_resource_value
        and candidate.control_card_cost <= baseline.control_card_cost
        and candidate.high_card_cost <= baseline.high_card_cost
    )
    strictly_better = (
        candidate.expected_remaining_turns < baseline.expected_remaining_turns
        or candidate.worst_remaining_turns < baseline.worst_remaining_turns
        or candidate.expected_remaining_cards < baseline.expected_remaining_cards
        or candidate.worst_remaining_cards < baseline.worst_remaining_cards
        or candidate.expected_skill_risk < baseline.expected_skill_risk
        or candidate.worst_skill_risk < baseline.worst_skill_risk
        or candidate.ally_control_cost < baseline.ally_control_cost
        or candidate.enemy_finish_risk < baseline.enemy_finish_risk
        or candidate.target_relation_cost < baseline.target_relation_cost
        or candidate.external_skill_cost < baseline.external_skill_cost
        or candidate.skill_resource_value > baseline.skill_resource_value
        or candidate.control_card_cost < baseline.control_card_cost
        or candidate.high_card_cost < baseline.high_card_cost
    )
    return fully_no_worse and strictly_better


class StableRulePolicy:
    policy_id = "stable_rule_v3"

    def __init__(self, skill_focused: bool = True):
        self.skill_focused = skill_focused
        self.last_decision: dict | None = None

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")
        if not self.skill_focused:
            skip = next(
                (action for action in actions if action.kind == "interaction" and action.parameters.get("skip")),
                None,
            )
            if skip is not None:
                return skip
            non_skill = [action for action in actions if action.kind != "skill"]
            if non_skill:
                actions = non_skill
        hero = engine.observe(actions[0].actor).hero
        if hero is not None and any(
            skill.rule_id not in AUTHORITATIVE_RULE_IDS for skill in HERO_REGISTRY.get(hero, ())
        ):
            return self._select_projected_action(engine, actions)
        if all(action.kind in {"play", "pass"} for action in actions):
            selected = self._select_public_card_action(engine, actions)
            if selected is not None:
                return selected
        return self._select_projected_action(engine, actions)

    def _select_public_card_action(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction | None:
        """Use only ``engine.observe`` to adapt simulator card decisions to the live core."""

        observation = engine.observe(actions[0].actor)
        position = observation.observer
        other_positions = tuple(seat for seat in POSITIONS if seat != position)
        public_counts = dict(zip(other_positions, observation.opponent_card_counts))
        enemies = other_positions if position == observation.landlord else (observation.landlord,)
        allies = tuple(seat for seat in other_positions if seat not in enemies)
        state = observation.to_legacy_model_state()
        state.update(
            {
                "position": position,
                "landlord": observation.landlord,
                "enemy_card_counts": [public_counts[seat] for seat in enemies],
                "teammate_card_count": public_counts.get(allies[0]) if allies else None,
                "table_is_teammate": observation.trick_owner in allies,
                "table_is_enemy": observation.trick_owner in enemies,
                "history": [list(event.get("ranks", ())) for event in observation.history if event.get("kind") == "play"],
                "policy_id": self.policy_id,
            }
        )
        # This lazy import keeps the live rule module independent from simulator policies.
        from ok_tasks.RlCardRuleModel import (
            _build_candidate_records,
            _context_for_score,
            _evaluate_hero_play,
            _is_bomb_action,
            _post_skill_route_turns,
            _to_rlcard,
            build_table_pressure_context,
        )

        hand = _to_rlcard(state["hand_cards"])
        target = _to_rlcard(state["table_cards"])
        enemy_counts = tuple(state["enemy_card_counts"])
        pressure = build_table_pressure_context(state)
        hero_state = state["hero_state"]
        hero_context = _context_for_score(hand, target, observation.hero, hero_state, enemy_counts, pressure)
        records, projection_cache, route_evaluator = _build_candidate_records(
            hand, target, enemy_counts, observation.hero, hero_state.get("last_action_type"),
            self.policy_id, hero_state, pressure, hero_context,
        )
        action_by_physical = {_to_rlcard(action.ranks): action for action in actions if action.kind == "play"}
        records = tuple(record for record in records if record.physical_action in action_by_physical)
        if not records:
            return None
        context = DecisionContext(
            hand=hand,
            target=target,
            enemy_counts=enemy_counts,
            hero=observation.hero,
            last_action_type=hero_state.get("last_action_type"),
            policy_id=self.policy_id,
            protect_teammate_play=bool(state["table_is_teammate"]),
            hero_state=hero_state,
            pressure=pressure,
            hero_context=hero_context,
        )

        def project_pass():
            key = ("pass", "", "none")
            projection = projection_cache.get(key)
            if projection is None:
                projection = _evaluate_hero_play(hero_context, "pass", route_evaluator)
                projection_cache[key] = projection
            return projection

        decision = _DECISION_CORE.choose(
            context,
            records,
            is_bomb=_is_bomb_action,
            rank_index=lambda rank: "3456789TJQKA2BR".index(rank),
            baseline_turns=lambda: _post_skill_route_turns(tuple(state["hand_cards"])),
            pass_projection=project_pass,
        )
        selected_projection = decision.candidate.projection if decision.candidate is not None else project_pass()
        authoritative = observation.hero is None or all(
            skill.rule_id in AUTHORITATIVE_RULE_IDS
            for skill in HERO_REGISTRY.get(observation.hero, ())
        )
        chosen = decision.candidate.physical_action if decision.candidate else "pass"
        self.last_decision = {
            "policy_id": self.policy_id,
            "adapter": "simulator_public",
            "hero": observation.hero,
            "chosen": chosen,
            "proposed": chosen,
            "baseline": chosen,
            "compatibility_gate": "shared_core",
            "authoritative_projection": authoritative,
            "triggered_rules": list(selected_projection.triggered_rules),
            "skill_before_cards": len(hero_context.hand),
            "skill_after_cards": selected_projection.expected_remaining_cards,
            "random_branches": [branch.to_dict() for branch in selected_projection.random_branches],
            "reason": selected_projection.reason,
            "search": (
                {
                    "nodes": decision.search.nodes,
                    "depth": decision.search.depth,
                    "elapsed_ms": decision.search.elapsed_ms,
                    "reason": decision.search.reason,
                    "triggered": decision.search.triggered,
                }
                if decision.search else None
            ),
            "candidates": [
                {
                    "action_id": action_by_physical[record.physical_action].action_id,
                    "physical_action": record.physical_action,
                    "game_stage": record.game_stage,
                    "hand_expansion": dict(record.hand_expansion),
                    "score": list(record.score),
                    "triggered_rules": list(record.projection.triggered_rules),
                    "reason": record.projection.reason,
                }
                for record in records
            ],
        }
        if decision.candidate is None:
            return next((action for action in actions if action.kind == "pass"), None)
        return action_by_physical.get(decision.candidate.physical_action)

    def _select_projected_action(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        """Existing deterministic chooser retained for skill and interaction actions."""
        context = HeroDecisionContext.from_engine(engine)
        proposed, proposed_projection, projections = select_best_action(
            context, actions, route_evaluator=estimate_route_turns
        )
        baseline = LegacyStableRulePolicy(self.skill_focused).select(engine, actions)
        baseline_index = next(index for index, action in enumerate(actions) if action.action_id == baseline.action_id)
        baseline_projection = projections[baseline_index]
        authoritative = context.hero is None or all(
            skill.rule_id in AUTHORITATIVE_RULE_IDS
            for skill in HERO_REGISTRY.get(context.hero, ())
        )
        if not proposed_projection.legal:
            raise ValueError("统一技能策略没有生成合法投影")
        use_proposed = not baseline_projection.legal or proposed.action_id == baseline.action_id or (
            authoritative
            and _projection_safely_dominates(proposed_projection, baseline_projection)
        )
        selected = proposed if use_proposed else baseline
        projection = proposed_projection if use_proposed else baseline_projection
        if not projection.legal:
            raise ValueError("兼容门选择了非法技能投影")
        gate = "accepted" if use_proposed else "legacy_fallback" if authoritative else "projection_shadow"
        self.last_decision = {
            "policy_id": self.policy_id,
            "hero": context.hero,
            "chosen": selected.action_id,
            "proposed": proposed.action_id,
            "baseline": baseline.action_id,
            "compatibility_gate": gate,
            "authoritative_projection": authoritative,
            "triggered_rules": list(projection.triggered_rules),
            "skill_before_cards": len(context.hand),
            "skill_after_cards": projection.expected_remaining_cards,
            "random_branches": [branch.to_dict() for branch in projection.random_branches],
            "reason": projection.reason if use_proposed else f"{'兼容安全门' if authoritative else '非权威投影影子门'}保留冻结动作；统一候选为 {proposed.action_id}",
            "candidates": [
                {
                    "action_id": action.action_id,
                    "score": list(candidate.score_key),
                    "triggered_rules": list(candidate.triggered_rules),
                    "reason": candidate.reason,
                }
                for action, candidate in zip(actions, projections)
            ],
        }
        return selected


class LegacyStableRulePolicy:
    """Frozen pre-skill-projection baseline used by paired hero evaluation."""

    policy_id = "legacy_stable_rule_v2"

    def __init__(self, skill_focused: bool = True):
        self.skill_focused = skill_focused

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")

        interactions = [action for action in actions if action.kind == "interaction"]
        if interactions:
            skip = next((action for action in interactions if action.parameters.get("skip")), None)
            if not self.skill_focused and skip is not None:
                return skip
            return next((action for action in interactions if not action.parameters.get("skip")), interactions[0])

        hand_size = len(engine.state.players[engine.state.current_player].hand)

        def score(action: LegalAction) -> tuple[object, ...]:
            played = len(action.card_ids) if action.kind == "play" else 0
            remaining = max(0, hand_size - played)
            terminal = 0 if action.kind == "play" and remaining == 0 else 1
            if action.kind == "play":
                kind_cost = 0
            elif action.kind == "skill" and self.skill_focused:
                kind_cost = 0
            elif action.kind == "pass":
                kind_cost = 1
            else:
                kind_cost = 2
            bomb_cost = int(action.action_type in {"bomb", "rocket"} and remaining > 0)
            high_card_cost = max((rank_key(rank) for rank in action.ranks), default=-1)
            return (
                terminal,
                remaining,
                kind_cost,
                bomb_cost,
                -len(action.ranks),
                high_card_cost,
                action.action_id,
            )

        return min(actions, key=score)


class RandomLegalPolicy:
    policy_id = "random_legal_v3"

    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")
        return self.random.choice(actions)
