from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class QualityMetrics:
    recognition_attempts: int = 0
    recognition_failures: int = 0
    submit_attempts: int = 0
    submit_failures: int = 0
    illegal_actions: int = 0
    unknown_random_clicks: int = 0
    completed_games: int = 0
    uninterrupted_games: int = 0

    @property
    def recognition_success_rate(self) -> float:
        return 1.0 - self.recognition_failures / max(1, self.recognition_attempts)

    @property
    def submit_success_rate(self) -> float:
        return 1.0 - self.submit_failures / max(1, self.submit_attempts)

    @property
    def total_operations(self) -> int:
        return self.recognition_attempts + self.submit_attempts

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "recognition_success_rate": self.recognition_success_rate,
            "submit_success_rate": self.submit_success_rate,
            "total_operations": self.total_operations,
        }


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reasons": list(self.reasons), "metrics": dict(self.metrics)}


class RuntimeQualityGate:
    def __init__(
        self,
        minimum_operations: int = 10_000,
        minimum_success_rate: float = 0.999,
        minimum_uninterrupted_games: int = 100,
    ):
        self.minimum_operations = minimum_operations
        self.minimum_success_rate = minimum_success_rate
        self.minimum_uninterrupted_games = minimum_uninterrupted_games

    def evaluate(self, metrics: QualityMetrics) -> QualityGateResult:
        reasons = []
        if metrics.total_operations < self.minimum_operations:
            reasons.append(f"有效识别/提交操作不足 {self.minimum_operations} 次")
        if metrics.recognition_success_rate < self.minimum_success_rate:
            reasons.append(f"识别成功率 {metrics.recognition_success_rate:.4%} 低于 {self.minimum_success_rate:.2%}")
        if metrics.submit_success_rate < self.minimum_success_rate:
            reasons.append(f"提交成功率 {metrics.submit_success_rate:.4%} 低于 {self.minimum_success_rate:.2%}")
        if metrics.uninterrupted_games < self.minimum_uninterrupted_games:
            reasons.append(f"连续无人干预对局不足 {self.minimum_uninterrupted_games} 局")
        if metrics.illegal_actions:
            reasons.append(f"检测到 {metrics.illegal_actions} 次非法动作")
        if metrics.unknown_random_clicks:
            reasons.append(f"检测到 {metrics.unknown_random_clicks} 次未知界面点击")
        return QualityGateResult(not reasons, tuple(reasons), metrics.to_dict())


def collect_runtime_quality(
    runs_root: str | Path,
    maximum_operations: int = 10_000,
    maximum_events: int = 250_000,
) -> QualityMetrics:
    metrics = QualityMetrics()
    event_paths = sorted(Path(runs_root).glob("*/games/game_*/events.jsonl"), reverse=True)
    remaining = maximum_events
    consecutive_games = 0
    streak_open = True
    for path in event_paths:
        if remaining <= 0:
            break
        game_failed = False
        game_completed = False
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if remaining <= 0 or metrics.total_operations >= maximum_operations:
                break
            remaining -= 1
            try:
                event = json.loads(line)
            except ValueError:
                game_failed = True
                continue
            event_type = event.get("event_type")
            if event_type == "decision_state":
                metrics.recognition_attempts += 1
            elif event_type == "ocr_failed":
                metrics.recognition_attempts += 1
                metrics.recognition_failures += 1
                game_failed = True
            elif event_type == "action_submitted":
                metrics.submit_attempts += 1
            elif event_type == "submit_failed":
                metrics.submit_attempts += 1
                metrics.submit_failures += 1
                game_failed = True
            elif event_type == "illegal_action":
                metrics.illegal_actions += 1
                game_failed = True
            elif event_type == "unknown_random_click":
                metrics.unknown_random_clicks += 1
                game_failed = True
            elif event_type == "game_end" and event.get("status") == "completed":
                game_completed = True
        if game_completed:
            metrics.completed_games += 1
        if streak_open and game_completed and not game_failed:
            consecutive_games += 1
        else:
            streak_open = False
        if metrics.total_operations >= maximum_operations:
            break
    metrics.uninterrupted_games = consecutive_games
    return metrics
