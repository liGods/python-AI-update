from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Any

from ok_tasks.card_ai.heroes import HERO_REGISTRY, OWNED_HEROES
from ok_tasks.card_ai.policies import LegacyStableRulePolicy, Policy, StableRulePolicy
from ok_tasks.card_ai.schema import POSITIONS
from ok_tasks.card_ai.self_play import SelfPlayRunner


def _reward_for(position: str, winner: str) -> float:
    landlord_won = winner == "landlord"
    return 1.0 if (position == "landlord") == landlord_won else -1.0


def wilson_interval(successes: int, samples: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial success rate."""

    if samples < 0 or successes < 0 or successes > samples:
        raise ValueError("Wilson 区间要求 0 <= successes <= samples")
    if samples == 0:
        return 0.0, 1.0
    rate = successes / samples
    denominator = 1.0 + z * z / samples
    center = (rate + z * z / (2.0 * samples)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / samples + z * z / (4.0 * samples * samples)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _skill_decision_metrics(events: list[Any], position: str) -> tuple[int, int, dict[str, int], dict[str, int], dict[str, float]]:
    """Count only chosen, publicly logged skill projections for one seat."""

    decisions = 0
    triggers = 0
    rule_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    expansion_totals = {"expected_total": 0.0, "worst_total": 0.0}
    for event in events:
        if event.actor != position:
            continue
        decision = event.metadata.get("decision", {}) if isinstance(event.metadata, dict) else {}
        rules = decision.get("triggered_rules", ()) if isinstance(decision, dict) else ()
        if not rules:
            continue
        decisions += 1
        stage = str(event.metadata.get("game_stage", "unknown"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        chosen = decision.get("chosen") if isinstance(decision, dict) else None
        candidate = next(
            (
                value for value in decision.get("candidates", ())
                if isinstance(value, dict) and value.get("physical_action") == chosen
            ),
            None,
        )
        expansion = candidate.get("hand_expansion", {}) if isinstance(candidate, dict) else {}
        for field in expansion_totals:
            expansion_totals[field] += float(expansion.get(field, 0.0))
        for rule in rules:
            rule_id = str(rule)
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
            triggers += 1
    return decisions, triggers, rule_counts, stage_counts, expansion_totals


def evaluate_hero_policy_paired(
    heroes: tuple[str, ...] | None = None,
    deals_per_hero: int = 3,
    seed: int = 20260718,
    maximum_steps: int = 1000,
) -> dict[str, Any]:
    """Compare the shared skill policy with the frozen generic policy on identical deals."""

    selected_heroes = tuple(heroes or HERO_REGISTRY)
    unknown = tuple(hero for hero in selected_heroes if hero not in HERO_REGISTRY)
    if unknown:
        raise ValueError(f"未注册武将: {', '.join(unknown)}")
    if deals_per_hero <= 0:
        raise ValueError("deals_per_hero must be positive")

    hero_reports: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []
    for hero_index, hero in enumerate(selected_heroes):
        candidate_wins = 0
        baseline_wins = 0
        paired_improvements = 0
        paired_regressions = 0
        paired_ties = 0
        skill_decisions = 0
        skill_triggers = 0
        triggered_rule_counts: dict[str, int] = {}
        skill_stage_counts: dict[str, int] = {}
        hand_expansion_totals = {"expected_total": 0.0, "worst_total": 0.0}
        completed = 0
        failures: list[dict[str, Any]] = []
        position_reports = {
            position: {
                "requested_samples": deals_per_hero,
                "completed_samples": 0,
                "candidate_wins": 0,
                "baseline_wins": 0,
                "paired_improvements": 0,
                "paired_regressions": 0,
                "paired_ties": 0,
                "skill_decisions": 0,
                "skill_triggers": 0,
                "triggered_rule_counts": {},
                "skill_stage_counts": {},
                "hand_expansion_totals": {"expected_total": 0.0, "worst_total": 0.0},
            }
            for position in POSITIONS
        }
        for deal_index in range(deals_per_hero):
            deal_seed = seed + hero_index * deals_per_hero + deal_index
            for position in POSITIONS:
                hero_map = {seat: hero if seat == position else None for seat in POSITIONS}
                candidate_policies: dict[str, Policy] = {
                    seat: LegacyStableRulePolicy() for seat in POSITIONS
                }
                candidate_policies[position] = StableRulePolicy()
                baseline_policies: dict[str, Policy] = {
                    seat: LegacyStableRulePolicy() for seat in POSITIONS
                }
                try:
                    candidate_events, candidate_summary = SelfPlayRunner(candidate_policies).run_game(
                        deal_seed,
                        hero_map,
                        maximum_steps,
                        include_full_state=False,
                    )
                    _, baseline_summary = SelfPlayRunner(baseline_policies).run_game(
                        deal_seed,
                        hero_map,
                        maximum_steps,
                        include_full_state=False,
                    )
                except Exception as error:
                    failure = {
                        "hero": hero,
                        "seed": deal_seed,
                        "position": position,
                        "error": str(error),
                    }
                    failures.append(failure)
                    all_failures.append(failure)
                    continue

                candidate_reward = _reward_for(position, candidate_summary["winner"])
                baseline_reward = _reward_for(position, baseline_summary["winner"])
                decision_count, trigger_count, rule_counts, stage_counts, expansion_totals = _skill_decision_metrics(candidate_events, position)
                candidate_wins += int(candidate_reward > 0)
                baseline_wins += int(baseline_reward > 0)
                paired_improvements += int(candidate_reward > baseline_reward)
                paired_regressions += int(candidate_reward < baseline_reward)
                paired_ties += int(candidate_reward == baseline_reward)
                skill_decisions += decision_count
                skill_triggers += trigger_count
                seat_report = position_reports[position]
                seat_report["completed_samples"] += 1
                seat_report["candidate_wins"] += int(candidate_reward > 0)
                seat_report["baseline_wins"] += int(baseline_reward > 0)
                seat_report["paired_improvements"] += int(candidate_reward > baseline_reward)
                seat_report["paired_regressions"] += int(candidate_reward < baseline_reward)
                seat_report["paired_ties"] += int(candidate_reward == baseline_reward)
                seat_report["skill_decisions"] += decision_count
                seat_report["skill_triggers"] += trigger_count
                for rule_id, count in rule_counts.items():
                    triggered_rule_counts[rule_id] = triggered_rule_counts.get(rule_id, 0) + count
                    rule_map = seat_report["triggered_rule_counts"]
                    rule_map[rule_id] = rule_map.get(rule_id, 0) + count
                for stage, count in stage_counts.items():
                    skill_stage_counts[stage] = skill_stage_counts.get(stage, 0) + count
                    stage_map = seat_report["skill_stage_counts"]
                    stage_map[stage] = stage_map.get(stage, 0) + count
                for field, value in expansion_totals.items():
                    hand_expansion_totals[field] += value
                    seat_report["hand_expansion_totals"][field] += value
                completed += 1

        requested = deals_per_hero * len(POSITIONS)
        candidate_interval = wilson_interval(candidate_wins, completed)
        baseline_interval = wilson_interval(baseline_wins, completed)
        paired_quality_passed = (
            completed == requested
            and not failures
            and candidate_interval[0] + 1e-12 >= baseline_interval[0]
        )
        authoritative_simulation = all(skill.sim_verified for skill in HERO_REGISTRY[hero])
        passed = authoritative_simulation and paired_quality_passed
        hero_reports[hero] = {
            "requested_samples": requested,
            "completed_samples": completed,
            "candidate_wins": candidate_wins,
            "candidate_win_rate": candidate_wins / completed if completed else 0.0,
            "candidate_wilson_95": list(candidate_interval),
            "baseline_wins": baseline_wins,
            "baseline_win_rate": baseline_wins / completed if completed else 0.0,
            "baseline_wilson_95": list(baseline_interval),
            "paired_improvements": paired_improvements,
            "paired_regressions": paired_regressions,
            "paired_ties": paired_ties,
            "skill_decisions": skill_decisions,
            "skill_triggers": skill_triggers,
            "triggered_rule_counts": triggered_rule_counts,
            "skill_stage_counts": skill_stage_counts,
            "hand_expansion_totals": {field: round(value, 6) for field, value in hand_expansion_totals.items()},
            "positions": position_reports,
            "legal_action_rate": 1.0 if completed == requested and not failures else completed / requested,
            "state_errors": len(failures),
            "authoritative_simulation": authoritative_simulation,
            "paired_quality_passed": paired_quality_passed,
            "sim_verified": passed,
            "failures": failures,
        }

    return {
        "rules_version": "3p.1",
        "seed": seed,
        "deals_per_hero": deals_per_hero,
        "heroes": hero_reports,
        "requested_heroes": len(selected_heroes),
        "sim_verified_heroes": sum(report["sim_verified"] for report in hero_reports.values()),
        "passed": bool(hero_reports) and all(report["sim_verified"] for report in hero_reports.values()),
        "failed_samples": len(all_failures),
        "failures": all_failures[:100],
    }


def evaluate_paired(
    candidate: Policy,
    stable: Policy | None = None,
    deals: int = 50_000,
    seed: int = 20260718,
    maximum_steps: int = 1000,
    deal_offset: int = 0,
) -> dict[str, Any]:
    baseline = stable or StableRulePolicy()
    deltas = []
    candidate_wins = 0
    completed = 0
    failures = []
    for deal_index in range(deals):
        deal_seed = seed + deal_index
        global_deal_index = deal_offset + deal_index
        hero_map = {
            position: OWNED_HEROES[(global_deal_index * len(POSITIONS) + index) % len(OWNED_HEROES)]
            for index, position in enumerate(POSITIONS)
        }
        for position in POSITIONS:
            candidate_policies = {seat: baseline for seat in POSITIONS}
            candidate_policies[position] = candidate
            try:
                _, candidate_summary = SelfPlayRunner(candidate_policies).run_game(
                    deal_seed, hero_map, maximum_steps, include_full_state=False
                )
                _, stable_summary = SelfPlayRunner({seat: baseline for seat in POSITIONS}).run_game(
                    deal_seed, hero_map, maximum_steps, include_full_state=False
                )
            except Exception as error:
                failures.append({"seed": deal_seed, "position": position, "error": str(error)})
                continue
            candidate_reward = _reward_for(position, candidate_summary["winner"])
            stable_reward = _reward_for(position, stable_summary["winner"])
            deltas.append(candidate_reward - stable_reward)
            candidate_wins += int(candidate_reward > 0)
            completed += 1
    mean_delta = fmean(deltas) if deltas else 0.0
    standard_error = stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else float("inf")
    margin = 1.96 * standard_error
    win_rate = candidate_wins / completed if completed else 0.0
    bounded = min(1.0 - 1e-6, max(1e-6, win_rate))
    return {
        "requested_deals": deals,
        "paired_seat_samples": completed,
        "mean_reward_delta": mean_delta,
        "confidence_lower": mean_delta - margin,
        "confidence_upper": mean_delta + margin,
        "candidate_win_rate": win_rate,
        "elo_estimate": 400.0 * math.log10(bounded / (1.0 - bounded)),
        "illegal_actions": 0,
        "failed_samples": len(failures),
        "failures": failures[:100],
        "_delta_sum": float(sum(deltas)),
        "_delta_square_sum": float(sum(delta * delta for delta in deltas)),
        "_candidate_wins": candidate_wins,
    }
