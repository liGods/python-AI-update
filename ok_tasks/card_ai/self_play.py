from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from ok_tasks.card_ai.engine import BaiJiangPaiEngine, SimulationError
from ok_tasks.card_ai.decision.stage import GameStage, StageContext, classify_game_stage
from ok_tasks.card_ai.heroes import HERO_REGISTRY, SIMULATED_HEROES
from ok_tasks.card_ai.policies import Policy, RandomLegalPolicy, StableRulePolicy
from ok_tasks.card_ai.schema import POSITIONS, TrajectoryEvent
from ok_tasks.card_ai.trajectory import TrajectoryWriter, atomic_json


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int
    seed: int = 20260718
    maximum_steps: int = 1000
    include_full_state: bool = True
    verified_heroes_only: bool = True
    hero_pool: tuple[str | None, ...] | None = None


class SelfPlayRunner:
    def __init__(self, policies: dict[str, Policy] | None = None):
        fallback = StableRulePolicy()
        self.policies = {position: (policies or {}).get(position, fallback) for position in POSITIONS}

    def run_game(
        self,
        seed: int,
        heroes: dict[str, str | None] | None = None,
        maximum_steps: int = 1000,
        include_full_state: bool = True,
    ) -> tuple[list[TrajectoryEvent], dict[str, Any]]:
        engine = BaiJiangPaiEngine.create(seed, heroes, game_id=f"selfplay_{seed}")
        events: list[TrajectoryEvent] = []
        started = monotonic()
        for sequence in range(1, maximum_steps + 1):
            actor = engine.state.pending_interaction["actor"] if engine.state.pending_interaction else engine.state.current_player
            observation = engine.observe(actor)
            legal = engine.legal_actions(actor)
            if not legal:
                raise SimulationError(f"非终局状态没有合法动作: {engine.state.game_id}/{sequence}")
            policy = self.policies[actor]
            action = policy.select(engine, legal)
            metadata = {
                "policy_id": policy.policy_id,
                "hero": engine.state.players[actor].hero,
                "game_stage": _public_game_stage(observation),
            }
            decision = getattr(policy, "last_decision", None)
            if decision is not None:
                # Policy explanations contain only public candidate/projection
                # data, so evaluation can attribute observed skill benefits
                # without recording a simulator opponent hand.
                metadata["decision"] = dict(decision)
            if include_full_state:
                metadata["full_state"] = engine.state.to_dict()
            result = engine.step(action)
            events.append(
                TrajectoryEvent(
                    game_id=engine.state.game_id,
                    sequence=sequence,
                    event_type="decision",
                    actor=actor,
                    observation=observation.to_dict(),
                    legal_actions=tuple(candidate.to_dict() for candidate in legal),
                    chosen_action=action.to_dict(),
                    rewards=dict(result.rewards),
                    terminal=result.terminal,
                    metadata=metadata,
                )
            )
            if result.terminal:
                summary = {
                    "game_id": engine.state.game_id,
                    "seed": seed,
                    "steps": sequence,
                    "winner": engine.state.winner,
                    "heroes": {position: player.hero for position, player in engine.state.players.items()},
                    "duration_seconds": round(monotonic() - started, 6),
                    "valid": True,
                }
                return events, summary
        raise SimulationError(f"牌局超过最大步数仍未结束: {engine.state.game_id}")


    def run(self, config: SelfPlayConfig, output_root: str | Path) -> dict[str, Any]:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        hero_names = list(config.hero_pool) if config.hero_pool is not None else [
            hero
            for hero, skills in HERO_REGISTRY.items()
            if not config.verified_heroes_only or (hero in SIMULATED_HEROES and all(skill.verified for skill in skills))
        ]
        summaries = []
        failures = []
        for game_index in range(config.games):
            seed = config.seed + game_index
            heroes = {
                position: hero_names[(game_index * len(POSITIONS) + index) % len(hero_names)]
                for index, position in enumerate(POSITIONS)
            }
            try:
                events, summary = self.run_game(seed, heroes, config.maximum_steps, config.include_full_state)
            except Exception as error:
                failures.append({"seed": seed, "error": str(error), "heroes": heroes})
                continue
            TrajectoryWriter(root / "trajectories" / f"game_{seed}.jsonl.gz").extend(events)
            atomic_json(root / "summaries" / f"game_{seed}.json", summary)
            summaries.append(summary)
        report = {
            "requested_games": config.games,
            "completed_games": len(summaries),
            "failed_games": len(failures),
            "total_steps": sum(summary["steps"] for summary in summaries),
            "failures": failures[:100],
        }
        atomic_json(root / "self_play_summary.json", report)
        return report

    def run_parallel(self, config: SelfPlayConfig, output_root: str | Path, workers: int = 12) -> dict[str, Any]:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        hero_names = list(config.hero_pool) if config.hero_pool is not None else [
            hero
            for hero, skills in HERO_REGISTRY.items()
            if not config.verified_heroes_only or (hero in SIMULATED_HEROES and all(skill.verified for skill in skills))
        ]
        jobs = []
        summaries = []
        reused_games = 0
        for game_index in range(config.games):
            seed = config.seed + game_index
            trajectory_path = root / "trajectories" / f"game_{seed}.jsonl.gz"
            summary_path = root / "summaries" / f"game_{seed}.json"
            if trajectory_path.is_file() and summary_path.is_file():
                try:
                    existing_summary = load_game_summary(summary_path)
                except (OSError, ValueError):
                    existing_summary = None
                if existing_summary and existing_summary.get("valid") and int(existing_summary.get("seed", -1)) == seed:
                    summaries.append(existing_summary)
                    reused_games += 1
                    continue
            heroes = {
                position: hero_names[(game_index * len(POSITIONS) + index) % len(hero_names)]
                for index, position in enumerate(POSITIONS)
            }
            jobs.append((seed, heroes, config.maximum_steps, config.include_full_state, str(root)))
        failures = []
        if jobs:
            job_batches = [jobs[index : index + 16] for index in range(0, len(jobs), 16)]
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                future_map = {executor.submit(_run_game_batch_worker, batch): batch for batch in job_batches}
                for future in as_completed(future_map):
                    try:
                        batch_summaries, batch_failures = future.result()
                        summaries.extend(batch_summaries)
                        failures.extend(batch_failures)
                    except Exception as error:
                        for seed, heroes, _, _, _ in future_map[future]:
                            failures.append({"seed": seed, "error": str(error), "heroes": heroes})
        report = {
            "requested_games": config.games,
            "completed_games": len(summaries),
            "reused_games": reused_games,
            "newly_generated_games": len(summaries) - reused_games,
            "failed_games": len(failures),
            "total_steps": sum(summary["steps"] for summary in summaries),
            "workers": workers,
            "failures": failures[:100],
        }
        atomic_json(root / "self_play_summary.json", report)
        return report


def _public_game_stage(observation: Any) -> str:
    """Classify a trajectory event from the actor's observation only."""

    others = tuple(seat for seat in POSITIONS if seat != observation.observer)
    counts = dict(zip(others, observation.opponent_card_counts))
    enemies = others if observation.observer == observation.landlord else (observation.landlord,)
    allies = tuple(seat for seat in others if seat not in enemies)
    seen = [rank for event in observation.history for rank in event.get("ranks", ())]
    seen.extend(observation.target_ranks)
    stage = classify_game_stage(
        StageContext(
            own_card_count=len(observation.hand),
            position=observation.observer,
            enemy_card_counts=tuple(counts[seat] for seat in enemies),
            teammate_card_count=counts.get(allies[0]) if allies else None,
            table_has_cards=bool(observation.target_ranks),
            table_is_teammate=observation.trick_owner in allies,
            seen_bombs=sum(seen.count(rank) >= 4 for rank in set(seen) if rank not in {"X", "D", "B", "R"}),
            seen_jokers=sum(rank in {"X", "D", "B", "R"} for rank in seen),
            seen_twos=seen.count("2"),
            one_turn_finish_risk=min((counts[seat] for seat in enemies), default=17) <= 1,
        )
    )
    return GameStage(stage).value


def load_game_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_game_worker(
    seed: int,
    heroes: dict[str, str | None],
    maximum_steps: int,
    include_full_state: bool,
    output_root: str,
) -> dict[str, Any]:
    bucket = seed % 10
    if bucket == 0:
        policies = {position: RandomLegalPolicy(seed + index) for index, position in enumerate(POSITIONS)}
    elif bucket < 3:
        policies = {position: StableRulePolicy(skill_focused=False) for position in POSITIONS}
    else:
        policies = {position: StableRulePolicy(skill_focused=True) for position in POSITIONS}
    events, summary = SelfPlayRunner(policies).run_game(seed, heroes, maximum_steps, include_full_state)
    root = Path(output_root)
    TrajectoryWriter(root / "trajectories" / f"game_{seed}.jsonl.gz").extend(events)
    atomic_json(root / "summaries" / f"game_{seed}.json", summary)
    return summary


def _run_game_batch_worker(jobs: list[tuple[int, dict[str, str | None], int, bool, str]]):
    summaries = []
    failures = []
    for job in jobs:
        seed, heroes, _, _, _ = job
        try:
            summaries.append(_run_game_worker(*job))
        except Exception as error:
            failures.append({"seed": seed, "error": str(error), "heroes": heroes})
    return summaries, failures
