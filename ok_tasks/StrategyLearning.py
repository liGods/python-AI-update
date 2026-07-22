import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ok_tasks.PolicyOptimizer import POLICY_PROFILES, wilson_lower_bound
from ok_tasks.ReplayEvaluator import read_jsonl


REVIEW_SCHEMA_VERSION = 1


def _atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _opening_strength_band(events):
    bidding = next((event for event in events if event.get("event_type") == "bidding" and isinstance(event.get("score"), (int, float))), None)
    if bidding is None:
        return "unknown"
    score = float(bidding["score"])
    if score < 20.0:
        return "weak"
    if score < 34.0:
        return "medium"
    if score < 50.0:
        return "strong"
    return "elite"


def _outcome_score(summary):
    if isinstance(summary.get("score_delta"), (int, float)):
        return float(summary["score_delta"]), "game_score"
    return (1.0 if summary.get("won") else -1.0), "normalized_outcome"


def _failure_causes(summary, events):
    event_counts = Counter(event.get("event_type") for event in events)
    causes = []
    if int(summary.get("submit_failures", 0)) or event_counts["submit_failed"]:
        causes.append({"code": "submission_failure", "category": "execution", "evidence": int(summary.get("submit_failures", 0)) or event_counts["submit_failed"], "explanation": "存在出牌提交失败或计划动作被迫取消，结果不能只归因于策略。"})
    if event_counts["ocr_failed"] or event_counts["opponent_count_rejected"]:
        causes.append({"code": "perception_instability", "category": "perception", "evidence": event_counts["ocr_failed"] + event_counts["opponent_count_rejected"], "explanation": "手牌、桌面牌或对手牌数识别不稳定，决策输入可能不完整。"})
    if event_counts["prediction_failed"]:
        causes.append({"code": "prediction_failure", "category": "model", "evidence": event_counts["prediction_failed"], "explanation": "模型推理失败，牌局未能持续执行同一策略。"})
    if summary.get("won"):
        return causes
    decisions = [event for event in events if event.get("event_type") == "decision"]
    teammate_takeovers = [event for event in decisions if event.get("table_is_teammate") and (event.get("chosen") or event.get("final_choice"))]
    if teammate_takeovers:
        causes.append({"code": "teammate_control_conflict", "category": "strategy", "evidence": len(teammate_takeovers), "explanation": "曾压过队友牌权，需要验证接管是否真正减少了团队剩余牌路。"})
    emergency_passes = []
    retained_controls = []
    for decision in decisions:
        pressure = decision.get("table_pressure") if isinstance(decision.get("table_pressure"), dict) else {}
        nearest_enemy = pressure.get("nearest_enemy")
        chosen = decision.get("final_choice", decision.get("chosen", []))
        if isinstance(nearest_enemy, (int, float)) and nearest_enemy <= 2 and not chosen:
            emergency_passes.append(decision.get("round_id"))
        candidates = decision.get("candidates") if isinstance(decision.get("candidates"), list) else []
        if isinstance(nearest_enemy, (int, float)) and nearest_enemy <= 3 and not chosen and any(candidate.get("is_bomb") for candidate in candidates if isinstance(candidate, dict)):
            retained_controls.append(decision.get("round_id"))
    if emergency_passes:
        causes.append({"code": "missed_emergency_block", "category": "strategy", "evidence": emergency_passes, "explanation": "敌方进入两张内收尾阶段时仍选择不出，需要对照测试紧急封锁。"})
    if retained_controls:
        causes.append({"code": "control_release_timing", "category": "strategy", "evidence": retained_controls, "explanation": "敌方临近出完时仍保留炸弹或控制牌，需要验证更早释放控制资源。"})
    if not causes:
        causes.append({"code": "strategy_outplayed", "category": "strategy", "evidence": len(decisions), "explanation": "日志未发现明确识别或执行故障，优先比较相似局面的整局策略表现。"})
    return causes


def _hypothesis_for(review, policy_samples):
    strategic = [cause for cause in review["failure_causes"] if cause["category"] == "strategy"]
    blocked = any(cause["category"] in {"execution", "perception", "model"} for cause in review["failure_causes"])
    alternatives = [name for name in POLICY_PROFILES if name != review["policy_id"]]
    target_policy = min(alternatives, key=lambda name: (policy_samples.get(name, 0), name)) if alternatives else review["policy_id"]
    cause_code = strategic[0]["code"] if strategic else review["failure_causes"][0]["code"] if review["failure_causes"] else "none"
    hypothesis_id = hashlib.sha256(f"{review['comparison_key']}|{cause_code}|{target_policy}".encode("utf-8")).hexdigest()[:16]
    if blocked and not strategic:
        statement = "先消除识别、推理或提交故障，再判断策略优劣；故障局不作为策略晋升证据。"
        status = "blocked_by_quality"
    else:
        statement = f"在 {review['comparison_key']} 的相似局面中，用 {target_policy} 对照当前策略，验证是否提高胜率和平均得分。"
        status = "collecting"
    return {"hypothesis_id": hypothesis_id, "status": status, "cause_code": cause_code, "baseline_policy": review["policy_id"], "candidate_policy": target_policy, "statement": statement}


def review_game(game_folder, policy_samples=None):
    folder = Path(game_folder)
    summary = _read_json(folder / "summary.json")
    if summary.get("status") != "completed" or summary.get("won") is None:
        return None
    events = read_jsonl(folder / "events.jsonl")
    first_state = next((event for event in events if event.get("event_type") == "decision_state"), {})
    last_state = next((event for event in reversed(events) if event.get("event_type") == "decision_state"), {})
    event_counts = Counter(str(event.get("event_type", "unknown")) for event in events)
    position = summary.get("position") or first_state.get("position") or "unknown"
    hero = summary.get("hero") or first_state.get("hero") or "unknown"
    opening_band = _opening_strength_band(events)
    comparison_key = f"position={position}|opening={opening_band}"
    fingerprint_source = json.dumps({"position": position, "hero": hero, "opening": opening_band, "hand": first_state.get("hand_cards", [])}, ensure_ascii=False, sort_keys=True)
    score_delta, score_source = _outcome_score(summary)
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "run_id": folder.parent.parent.name,
        "game_id": summary.get("game_id", folder.name),
        "status": summary.get("status"),
        "won": bool(summary.get("won")),
        "policy_id": summary.get("policy_id", "balanced"),
        "hero": hero,
        "position": position,
        "opening_strength_band": opening_band,
        "comparison_key": comparison_key,
        "exact_fingerprint": hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:16],
        "score_delta": score_delta,
        "score_source": score_source,
        "event_count": len(events),
        "event_type_counts": dict(sorted(event_counts.items())),
        "decision_count": event_counts.get("decision", 0),
        "first_visible_state": first_state,
        "last_visible_state": last_state,
    }
    review["failure_causes"] = _failure_causes(summary, events)
    review["hypothesis"] = _hypothesis_for(review, policy_samples or {}) if not review["won"] else None
    _atomic_json(folder / "review.json", review)
    return review


class StrategyLearningPipeline:
    def __init__(self, runs_root, library_path, minimum_games=20):
        self.runs_root = Path(runs_root)
        self.library_path = Path(library_path)
        self.minimum_games = max(1, int(minimum_games))

    def _load_library(self):
        value = _read_json(self.library_path)
        return {
            "version": int(value.get("version", 1)),
            "candidate_hypotheses": value.get("candidate_hypotheses", {}) if isinstance(value.get("candidate_hypotheses"), dict) else {},
            "validated_strategies": value.get("validated_strategies", {}) if isinstance(value.get("validated_strategies"), dict) else {},
        }

    def _all_completed_games(self):
        return [folder for folder in sorted(self.runs_root.glob("*/games/game_*")) if folder.is_dir() and _read_json(folder / "summary.json").get("status") == "completed"]

    def process(self, current_run, baseline_policy="balanced"):
        game_folders = self._all_completed_games()
        existing_reviews = [value for value in (_read_json(folder / "review.json") for folder in game_folders) if value]
        policy_samples = Counter(review.get("policy_id", "balanced") for review in existing_reviews)
        reviews = [review_game(folder, policy_samples) for folder in game_folders]
        reviews = [review for review in reviews if review is not None]
        library = self._load_library()
        for review in reviews:
            hypothesis = review.get("hypothesis")
            if hypothesis is None:
                continue
            stored = library["candidate_hypotheses"].setdefault(hypothesis["hypothesis_id"], {**hypothesis, "comparison_key": review["comparison_key"], "source_games": []})
            source = review["run_id"] + "/" + review["game_id"]
            if source not in stored["source_games"]:
                stored["source_games"].append(source)
        grouped = defaultdict(lambda: defaultdict(list))
        for review in reviews:
            if not any(cause.get("category") in {"execution", "perception", "model"} for cause in review.get("failure_causes", [])):
                grouped[review["comparison_key"]][review["policy_id"]].append(review)
        comparisons = []
        promoted = []
        for comparison_key, by_policy in sorted(grouped.items()):
            baseline = by_policy.get(baseline_policy, [])
            for candidate_policy, candidate in sorted(by_policy.items()):
                if candidate_policy == baseline_policy:
                    continue
                baseline_wins = sum(int(review["won"]) for review in baseline)
                candidate_wins = sum(int(review["won"]) for review in candidate)
                baseline_rate = baseline_wins / len(baseline) if baseline else None
                candidate_rate = candidate_wins / len(candidate) if candidate else None
                baseline_score = sum(review["score_delta"] for review in baseline) / len(baseline) if baseline else None
                candidate_score = sum(review["score_delta"] for review in candidate) / len(candidate) if candidate else None
                enough = len(baseline) >= self.minimum_games and len(candidate) >= self.minimum_games
                verified = bool(enough and candidate_rate > baseline_rate and candidate_score > baseline_score and wilson_lower_bound(candidate_wins, len(candidate)) > wilson_lower_bound(baseline_wins, len(baseline)))
                comparison = {
                    "comparison_key": comparison_key,
                    "baseline_policy": baseline_policy,
                    "candidate_policy": candidate_policy,
                    "baseline_games": len(baseline),
                    "candidate_games": len(candidate),
                    "baseline_win_rate": baseline_rate,
                    "candidate_win_rate": candidate_rate,
                    "win_rate_delta": None if baseline_rate is None or candidate_rate is None else round(candidate_rate - baseline_rate, 4),
                    "baseline_average_score": baseline_score,
                    "candidate_average_score": candidate_score,
                    "average_score_delta": None if baseline_score is None or candidate_score is None else round(candidate_score - baseline_score, 4),
                    "verified": verified,
                    "message": "验证通过" if verified else f"继续采样，双方至少需要 {self.minimum_games} 局且候选胜率下界和平均得分都更高",
                }
                comparisons.append(comparison)
                if verified:
                    library["validated_strategies"][comparison_key] = {**comparison, "policy_id": candidate_policy}
                    promoted.append(comparison_key)
        for hypothesis in library["candidate_hypotheses"].values():
            validated = library["validated_strategies"].get(hypothesis.get("comparison_key"))
            if validated and validated.get("policy_id") == hypothesis.get("candidate_policy"):
                hypothesis["status"] = "validated"
        library["version"] += 1
        _atomic_json(self.library_path, library)
        current_name = Path(current_run).name
        current_reviews = [review for review in reviews if review["run_id"] == current_name]
        result = {
            "reviewed_games": len(current_reviews),
            "total_comparable_games": len(reviews),
            "hypotheses_generated": sum(1 for review in current_reviews if review.get("hypothesis")),
            "comparisons": comparisons,
            "promoted_strategies": promoted,
            "strategy_library": str(self.library_path),
            "message": f"已复盘 {len(current_reviews)} 局，生成 {sum(1 for review in current_reviews if review.get('hypothesis'))} 个失败假设，验证入库 {len(promoted)} 条策略",
        }
        _atomic_json(Path(current_run) / "learning_summary.json", result)
        return result
