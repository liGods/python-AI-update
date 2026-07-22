import json  # 导入 JSON 模块以读取逐局事件和总结。
from pathlib import Path  # 导入路径类型以遍历持久运行目录。


def read_jsonl(path):  # 容错读取可能因异常退出截断的 JSONL 文件。
    events = []  # 初始化成功解析的事件列表。
    source = Path(path)  # 将输入转换为统一路径对象。
    if not source.is_file():  # 文件不存在时返回空事件流。
        return events  # 避免回放器因缺少不完整牌局日志失败。
    for line in source.read_text(encoding="utf-8").splitlines():  # 逐行读取独立事件对象。
        if not line.strip():  # 跳过空行。
            continue  # 继续处理下一条事件。
        try:  # 捕获最后一行可能被异常退出截断的情况。
            value = json.loads(line)  # 解析当前完整 JSON 对象。
        except ValueError:  # 无法解析的截断行不应影响前面完整事件。
            continue  # 跳过损坏行并保留可恢复部分。
        if isinstance(value, dict):  # 只接受对象类型事件。
            events.append(value)  # 保存有效事件。
    return events  # 返回原始顺序的可回放事件流。


def replay_game(game_folder):  # 验证一局日志中的状态、候选和已选动作是否自洽。
    folder = Path(game_folder)  # 解析牌局目录。
    events = read_jsonl(folder / "events.jsonl")  # 读取本局全部可恢复事件。
    states = {event.get("round_id"): event for event in events if event.get("event_type") == "decision_state" and event.get("round_id")}  # 按回合编号索引模型输入状态。
    decisions = [event for event in events if event.get("event_type") == "decision"]  # 提取包含完整合法候选的决策事件。
    errors = []  # 初始化回放一致性错误列表。
    for decision in decisions:  # 逐回合验证候选和最终动作。
        round_id = decision.get("round_id")  # 读取决策关联的回合编号。
        if round_id not in states:  # 决策缺少对应模型输入时无法完整回放。
            errors.append(f"{round_id}: 缺少 decision_state")  # 记录缺失状态错误。
            continue  # 继续检查本局其他回合。
        candidate_actions = [candidate.get("cards", []) for candidate in decision.get("candidates", [])]  # 提取所有已枚举合法候选。
        chosen = decision.get("chosen", [])  # 读取规则或模型最终选择动作。
        if chosen and chosen not in candidate_actions:  # 非空动作必须来自当时的合法候选集合。
            errors.append(f"{round_id}: 选择动作不在合法候选中")  # 记录非法或日志不一致动作。
    return {"game_id": folder.name, "events": len(events), "decisions": len(decisions), "valid": not errors, "errors": errors}  # 返回本局回放结果。


def replay_run(run_folder):  # 回放一次运行内全部有效和不完整牌局。
    root = Path(run_folder)  # 解析运行会话目录。
    results = [replay_game(folder) for folder in sorted((root / "games").glob("game_*")) if folder.is_dir()]  # 按牌局编号回放所有目录。
    return {"run": root.name, "games": len(results), "decisions": sum(result["decisions"] for result in results), "valid": all(result["valid"] for result in results), "results": results}  # 汇总整次运行回放状态。


def collect_training_games(runs_root):  # 从所有持久运行目录提取可用于动作评分训练的完整牌局。
    games = []  # 初始化按时间顺序排列的训练牌局列表。
    for game_folder in sorted(Path(runs_root).glob("*/games/game_*")):  # 按会话和牌局路径稳定遍历数据集。
        summary_path = game_folder / "summary.json"  # 解析当前牌局总结路径。
        if not summary_path.is_file():  # 没有总结的异常中断牌局不能参与训练。
            continue  # 继续检查下一局。
        try:  # 捕获损坏总结文件。
            summary = json.loads(summary_path.read_text(encoding="utf-8"))  # 读取本局胜负和有效状态。
        except ValueError:  # 无法解析的总结不参与训练。
            continue  # 继续检查下一局。
        if summary.get("status") != "completed" or summary.get("won") is None:  # 只使用真实结算的完整牌局。
            continue  # 排除手动停止和未知界面中断牌局。
        events = read_jsonl(game_folder / "events.jsonl")  # 读取本局完整事件流。
        states = {event.get("round_id"): event for event in events if event.get("event_type") == "decision_state" and event.get("round_id")}  # 索引每回合模型状态。
        samples = []  # 初始化本局训练决策列表。
        for decision in [event for event in events if event.get("event_type") == "decision"]:  # 遍历规则模型的完整候选决策。
            state = states.get(decision.get("round_id"))  # 查找同一回合的模型输入状态。
            candidates = decision.get("candidates", [])  # 读取当时全部合法候选。
            chosen = decision.get("chosen", [])  # 读取当时最终动作。
            chosen_index = next((index for index, candidate in enumerate(candidates) if candidate.get("cards", []) == chosen), None)  # 定位模仿学习标签索引。
            if state is not None and candidates and chosen_index is not None:  # 只保留状态、候选和标签完整的成功决策。
                samples.append({"state": state, "candidates": candidates, "chosen_index": chosen_index})  # 保存训练所需最小数据。
        games.append({"path": str(game_folder), "won": bool(summary.get("won")), "samples": samples})  # 将本局胜负和决策序列加入数据集。
    return games  # 返回按整局切分的数据集供时间划分使用。
