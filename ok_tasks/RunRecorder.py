import json  # 导入 JSON 模块以写入逐事件和总结文件。
from datetime import datetime, timezone  # 导入带时区时间类型以生成稳定时间戳。
from pathlib import Path  # 导入路径类型以创建持久运行目录。
from time import monotonic  # 导入单调时钟以记录不受系统时间调整影响的耗时。


SCHEMA_VERSION = 2  # 定义运行日志结构版本供后续回放兼容。


def _now_iso():  # 生成带时区的毫秒时间戳。
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")  # 返回本地时区的 ISO 时间文本。


def _atomic_json(path, value):  # 使用同目录临时文件原子写入 JSON。
    target = Path(path)  # 将输入路径转换成统一路径对象。
    target.parent.mkdir(parents=True, exist_ok=True)  # 确保目标父目录已经创建。
    temporary = target.with_suffix(target.suffix + ".tmp")  # 在同一目录生成临时文件路径。
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")  # 先完整写入 UTF-8 临时文件。
    temporary.replace(target)  # 原子替换目标文件避免异常退出产生半截 JSON。


class RunRecorder:  # 将一次自动化运行拆分为会话、牌局和逐事件日志。
    def __init__(self, root, session_id, task_name, config=None):  # 初始化持久化运行目录和会话元数据。
        self.root = Path(root) / session_id  # 为本次运行创建唯一持久目录。
        self.root.mkdir(parents=True, exist_ok=True)  # 创建会话目录层级。
        self.session_id = session_id  # 保存会话编号供每条事件关联。
        self.task_name = task_name  # 保存启动本记录器的任务名称。
        self.config = dict(config or {})  # 复制任务配置避免后续界面修改影响本次记录。
        self.started_at = _now_iso()  # 保存会话开始时间。
        self.started_monotonic = monotonic()  # 保存计算总耗时的单调时钟起点。
        self.game_id = None  # 初始化尚未开始具体牌局。
        self.game_index = 0  # 初始化本次运行牌局序号。
        self.game_started_monotonic = None  # 初始化当前牌局耗时起点。
        self.event_sequence = 0  # 初始化当前牌局事件编号。
        self.game_summaries = []  # 保存已经结束的牌局总结。
        self.metrics = {"events": 0, "captures": 0, "submit_failures": 0, "ocr_failures": 0, "unknown_states": 0}  # 初始化会话关键质量指标。
        _atomic_json(self.root / "session.json", self._session_document("running"))  # 立即写入运行中会话文件以支持异常恢复。

    def _session_document(self, status):  # 构造当前会话总览文档。
        return {  # 返回完整且可独立读取的会话对象。
            "schema_version": SCHEMA_VERSION,  # 写入日志结构版本。
            "session_id": self.session_id,  # 写入会话唯一编号。
            "task_name": self.task_name,  # 写入任务名称。
            "status": status,  # 写入 running、completed、stopped 或 failed 状态。
            "started_at": self.started_at,  # 写入会话开始时间。
            "config": self.config,  # 写入本次运行使用的配置快照。
        }  # 完成会话总览构造。

    def start_game(self, hero=None, position=None, policy_id="balanced"):  # 开始一局新的结构化日志。
        if self.game_id is not None:  # 上一局尚未结算时先保留不完整总结。
            self.end_game(None, status="incomplete")  # 将中断牌局标为不完整且不参与训练。
        self.game_index += 1  # 增加本次运行牌局序号。
        self.game_id = f"game_{self.game_index:04d}"  # 生成固定宽度的牌局编号。
        self.game_started_monotonic = monotonic()  # 记录本局开始的单调时钟。
        self.event_sequence = 0  # 为新牌局重置事件编号。
        game_folder = self.root / "games" / self.game_id  # 解析当前牌局独立目录。
        game_folder.mkdir(parents=True, exist_ok=True)  # 创建当前牌局目录。
        self.event("game_start", hero=hero, position=position, policy_id=policy_id)  # 写入本局首条开始事件。
        return self.game_id  # 返回牌局编号供任务状态面板显示。

    def ensure_game(self, **context):  # 确保导航中途恢复时也有可写牌局日志。
        if self.game_id is None:  # 尚未建立牌局时使用当前上下文开始记录。
            self.start_game(context.get("hero"), context.get("position"), context.get("policy_id", "balanced"))  # 创建恢复牌局记录。
        return self.game_id  # 返回现有或刚创建的牌局编号。

    def event(self, event_type, **payload):  # 追加一条可立即恢复的 JSONL 事件。
        self.event_sequence += 1  # 增加当前牌局事件序号。
        document = {  # 构造单条稳定事件对象。
            "schema_version": SCHEMA_VERSION,  # 写入日志结构版本。
            "session_id": self.session_id,  # 写入所属会话编号。
            "game_id": self.game_id,  # 写入所属牌局编号。
            "sequence": self.event_sequence,  # 写入牌局内连续事件编号。
            "timestamp": _now_iso(),  # 写入事件发生时间。
            "elapsed_ms": int((monotonic() - self.started_monotonic) * 1000),  # 写入相对会话开始的毫秒耗时。
            "event_type": event_type,  # 写入稳定事件类型。
            **payload,  # 合并调用方提供的牌局上下文。
        }  # 完成事件对象构造。
        events_path = self.root / "games" / str(self.game_id or "session") / "events.jsonl"  # 解析当前牌局的逐事件文件。
        events_path.parent.mkdir(parents=True, exist_ok=True)  # 确保会话事件也有可写目录。
        with events_path.open("a", encoding="utf-8") as stream:  # 以追加模式打开事件日志避免覆盖已有内容。
            stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")  # 每条事件单独写一行便于中断恢复。
            stream.flush()  # 立即刷新缓冲区减少异常退出的数据损失。
        self.metrics["events"] += 1  # 累加会话事件数量。
        if event_type == "capture":  # 截图事件需要单独统计素材数量。
            self.metrics["captures"] += 1  # 累加事件素材数量。
        if event_type == "submit_failed":  # 提交失败事件需要进入质量指标。
            self.metrics["submit_failures"] += 1  # 累加出牌提交失败次数。
        if event_type == "ocr_failed":  # OCR 失败事件需要进入质量指标。
            self.metrics["ocr_failures"] += 1  # 累加完整识牌失败次数。
        if event_type == "unknown_state":  # 未知界面需要进入资源缺口指标。
            self.metrics["unknown_states"] += 1  # 累加未知状态次数。
        return document  # 返回已写入事件供测试或状态展示。

    def capture_path(self, state, sequence):  # 生成当前牌局事件截图的持久路径。
        safe_state = "".join(character if character.isalnum() or character in "-_" else "_" for character in state)  # 清理状态名中的路径非法字符。
        owner = self.root / "games" / self.game_id if self.game_id is not None else self.root / "session"  # 将导航素材保存到会话目录，将牌局素材保存到当前牌局目录。
        return owner / "captures" / f"{sequence:05d}_{safe_state}.png"  # 返回固定命名的截图路径。

    def end_game(self, won, status="completed", **payload):  # 结束当前牌局并写入独立总结。
        if self.game_id is None:  # 没有活动牌局时不创建虚假结算记录。
            return None  # 返回空值表示没有牌局需要结束。
        duration = 0.0 if self.game_started_monotonic is None else monotonic() - self.game_started_monotonic  # 计算本局运行秒数。
        self.event("game_end", won=won, status=status, duration_seconds=round(duration, 3), **payload)  # 写入本局最后一条事件。
        events_path = self.root / "games" / self.game_id / "events.jsonl"  # 读取刚完成的完整逐事件记录以生成可检索清单。
        events = []  # 初始化本局有效事件列表。
        for line in events_path.read_text(encoding="utf-8").splitlines() if events_path.is_file() else []:  # 容错处理异常退出留下的空行或截断行。
            try:  # 单行损坏不能丢弃此前已经落盘的完整记录。
                value = json.loads(line)  # 解析当前独立事件对象。
            except ValueError:  # 跳过无法恢复的截断行。
                continue  # 继续统计其他完整事件。
            if isinstance(value, dict):  # 只把对象类型写入记录清单。
                events.append(value)  # 保留事件原始顺序。
        event_types = {}  # 汇总各类事件数量便于快速检查记录完整度。
        for event in events:  # 遍历本局全部可恢复事件。
            name = str(event.get("event_type", "unknown"))  # 规范缺失类型的旧日志。
            event_types[name] = event_types.get(name, 0) + 1  # 累加当前事件类型。
        score_delta = payload.get("score_delta")  # 优先使用未来由结算界面识别到的真实得分变化。
        score_source = "game_score" if isinstance(score_delta, (int, float)) else "normalized_outcome"  # 标记得分来源，避免把标准化胜负误称为游戏积分。
        if score_delta is None and won is not None:  # 当前界面没有可靠积分 OCR 时使用可比较的标准化结果。
            score_delta = 1 if won else -1  # 胜一分、负一分用于策略对照，不冒充真实账户积分。
        decision_round_ids = list(dict.fromkeys(event.get("round_id") for event in events if event.get("event_type") in {"decision_state", "decision"} and event.get("round_id")))  # 状态或决定任一存在都保留回合索引并按首次出现顺序去重。
        summary = {"schema_version": SCHEMA_VERSION, "game_id": self.game_id, "status": status, "won": won, "score_delta": score_delta, "score_source": score_source, "duration_seconds": round(duration, 3), "event_count": len(events), "event_type_counts": event_types, "decision_round_ids": decision_round_ids, **payload}  # 构造可汇总且可验证完整度的本局结果。
        _atomic_json(self.root / "games" / self.game_id / "summary.json", summary)  # 原子保存本局总结。
        self.game_summaries.append(summary)  # 将本局加入会话总结列表。
        self.game_id = None  # 清除活动牌局避免同一结算重复写入。
        self.game_started_monotonic = None  # 清除本局耗时起点。
        return summary  # 返回完成的本局总结供统计器使用。

    def finalize(self, status="completed", optimization=None, missing_resources=None, learning=None):  # 结束运行并生成机器与中文总结。
        if self.game_id is not None:  # 任务结束时仍有未结算牌局需要保留现场。
            self.end_game(None, status="incomplete")  # 将未结算牌局明确排除在训练数据之外。
        completed = [game for game in self.game_summaries if game.get("status") == "completed" and game.get("won") is not None]  # 过滤有效结算牌局。
        wins = sum(1 for game in completed if game.get("won"))  # 统计有效胜利局数。
        duration = monotonic() - self.started_monotonic  # 计算整个会话运行秒数。
        summary = {  # 构造本次运行最终总结。
            "schema_version": SCHEMA_VERSION,  # 写入日志结构版本。
            "session_id": self.session_id,  # 写入会话编号。
            "status": status,  # 写入最终运行状态。
            "completed_games": len(completed),  # 写入有效结算局数。
            "wins": wins,  # 写入胜利局数。
            "losses": len(completed) - wins,  # 写入失败局数。
            "win_rate": round(wins / len(completed), 4) if completed else None,  # 写入有效胜率或空值。
            "duration_seconds": round(duration, 3),  # 写入会话总耗时。
            "metrics": dict(self.metrics),  # 写入识别和操作质量指标。
            "optimization": optimization or {},  # 写入策略优化结果或样本不足原因。
            "learning": learning or {},  # 写入逐局复盘、对照验证和策略入库结果。
            "missing_resources": list(missing_resources or []),  # 写入仍未补齐的资源键。
            "games": list(self.game_summaries),  # 写入所有完成和中断牌局总结。
        }  # 完成运行总结构造。
        _atomic_json(self.root / "summary.json", summary)  # 原子保存机器可读总结。
        _atomic_json(self.root / "session.json", {**self._session_document(status), "finished_at": _now_iso()})  # 将会话状态更新为最终状态。
        report_lines = [  # 生成用户可以直接阅读的中文报告行。
            f"自动打牌运行报告：{self.session_id}",  # 写入报告标题。
            f"状态：{status}",  # 写入最终状态。
            f"有效对局：{len(completed)}，胜 {wins}，负 {len(completed) - wins}",  # 写入胜负概览。
            f"胜率：{summary['win_rate'] if summary['win_rate'] is not None else '暂无有效数据'}",  # 写入胜率或无数据提示。
            f"提交失败：{self.metrics['submit_failures']}，OCR 失败：{self.metrics['ocr_failures']}，未知界面：{self.metrics['unknown_states']}",  # 写入关键故障指标。
            f"策略优化：{(optimization or {}).get('message', '未执行')}",  # 写入本次策略优化结果。
            f"AI 复盘：{(learning or {}).get('message', '未执行')}",  # 写入失败分析、对照测试和入库摘要。
            f"仍缺素材：{', '.join(missing_resources or []) if missing_resources else '无'}",  # 写入资源缺口。
        ]  # 完成中文报告内容。
        (self.root / "报告.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")  # 保存 UTF-8 中文报告。
        return summary  # 返回总结供任务界面展示。
