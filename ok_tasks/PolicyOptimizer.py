import json  # 导入 JSON 模块以持久化策略实验统计。
import math  # 导入数学函数以计算 Wilson 胜率区间。
from pathlib import Path  # 导入路径类型以安全保存策略状态。


POLICY_PROFILES = {  # 定义只能修改次级评分的三套安全策略。
    "conservative": {"hero_scale": 0.5, "length_bonus": 0, "high_card_bias": 0},  # 保守策略弱化技能偏好并尽量保留控制牌。
    "balanced": {"hero_scale": 1.0, "length_bonus": 1, "high_card_bias": 1},  # 均衡策略保持当前规则行为。
    "skill_focused": {"hero_scale": 1.5, "length_bonus": 2, "high_card_bias": 1},  # 技能策略在主安全评分相同时更积极触发英雄技能。
}  # 完成固定策略候选定义。


def wilson_lower_bound(wins, games, z=1.96):  # 计算二项胜率的 Wilson 95% 置信下界。
    if games <= 0:  # 没有有效样本时不能估计胜率。
        return 0.0  # 返回保守的零下界。
    rate = wins / games  # 计算观测胜率。
    denominator = 1.0 + z * z / games  # 计算 Wilson 区间分母。
    center = rate + z * z / (2.0 * games)  # 计算校正后的中心项。
    margin = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * games)) / games)  # 计算置信区间半径。
    return (center - margin) / denominator  # 返回置信区间下界。


class PolicyOptimizer:  # 管理整局策略探索、晋升和自动回滚。
    def __init__(self, path, minimum_games=20):  # 加载持久策略状态并设置最少验证局数。
        self.path = Path(path)  # 保存策略状态文件路径。
        self.minimum_games = max(1, int(minimum_games))  # 限制最少验证局数必须为正数。
        self.data = self._load()  # 读取已有策略统计或创建初始状态。

    def _empty_stats(self):  # 创建一套策略的空统计结构。
        return {"games": 0, "wins": 0, "submit_failures": 0, "recent": []}  # 返回胜负、操作失败和最近结果容器。

    def _load(self):  # 容错读取策略状态文件。
        if not self.path.is_file():  # 首次运行尚未生成状态文件。
            return {"version": 1, "active": "balanced", "previous": "balanced", "profiles": {name: self._empty_stats() for name in POLICY_PROFILES}}  # 返回以均衡策略为活动版本的初始状态。
        try:  # 捕获异常退出或手工编辑导致的损坏 JSON。
            value = json.loads(self.path.read_text(encoding="utf-8"))  # 读取 UTF-8 策略状态。
        except (OSError, ValueError):  # 文件不可读或 JSON 无法解析时使用安全默认值。
            value = {}  # 将损坏内容视为空状态。
        active = value.get("active") if value.get("active") in POLICY_PROFILES else "balanced"  # 修复未知活动策略名称。
        profiles = value.get("profiles") if isinstance(value.get("profiles"), dict) else {}  # 读取已有策略统计对象。
        for name in POLICY_PROFILES:  # 确保三套固定策略均有完整统计。
            stats = profiles.setdefault(name, self._empty_stats())  # 为缺失策略创建空统计。
            for key, default in self._empty_stats().items():  # 补齐旧版本缺失字段。
                stats.setdefault(key, list(default) if isinstance(default, list) else default)  # 保留已有值并复制可变默认列表。
        return {"version": int(value.get("version", 1)), "active": active, "previous": value.get("previous", active), "profiles": profiles}  # 返回修复后的完整状态。

    def save(self):  # 原子保存当前策略统计。
        self.path.parent.mkdir(parents=True, exist_ok=True)  # 创建策略状态父目录。
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")  # 创建同目录临时文件。
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")  # 完整写入中文 JSON。
        temporary.replace(self.path)  # 原子替换主状态文件。

    @property  # 将活动策略暴露为只读属性。
    def active_policy(self):  # 返回当前稳定策略编号。
        return self.data["active"]  # 读取已验证或默认均衡策略。

    def choose_policy(self):  # 为下一整局选择稳定策略或受限探索候选。
        profiles = self.data["profiles"]  # 读取三套策略累计统计。
        under_sampled = [name for name in POLICY_PROFILES if name != self.active_policy and profiles[name]["games"] < self.minimum_games]  # 查找尚未达到验证门槛的候选。
        total_games = sum(int(stats["games"]) for stats in profiles.values())  # 计算全部策略累计局数。
        if under_sampled and total_games % 5 == 4:  # 每五局最多安排一局候选以限制探索流量为百分之二十。
            return min(under_sampled, key=lambda name: (profiles[name]["games"], name))  # 优先选择样本最少的候选策略。
        return self.active_policy  # 其余牌局使用当前稳定策略。

    def record_game(self, policy_id, won, submit_failures=0):  # 将一局有效结算归因到整局使用的策略。
        if policy_id not in POLICY_PROFILES or won is None:  # 未知策略和未结算牌局不能参与学习。
            return False  # 返回未记录结果。
        stats = self.data["profiles"][policy_id]  # 读取本局策略累计统计。
        stats["games"] = int(stats["games"]) + 1  # 累加有效结算局数。
        stats["wins"] = int(stats["wins"]) + int(bool(won))  # 根据胜负累加胜利数。
        stats["submit_failures"] = int(stats["submit_failures"]) + int(submit_failures)  # 累加本局提交失败次数。
        stats["recent"].append({"won": bool(won), "submit_failures": int(submit_failures)})  # 保存自动回滚需要的最近结果。
        del stats["recent"][:-20]  # 只保留最近二十局限制状态文件增长。
        self.save()  # 每局结算后立即持久化策略统计。
        return True  # 返回本局统计已经记录。

    def rollback(self):  # 提供用户可调用的显式上一策略回滚入口。
        active_name = self.active_policy  # 保存当前活动策略供交换历史指针。
        previous_name = self.data.get("previous", active_name)  # 读取上一稳定策略名称。
        if previous_name not in POLICY_PROFILES or previous_name == active_name:  # 没有可用旧版本时拒绝虚假回滚。
            return {"changed": False, "active": active_name, "message": "当前没有可回滚的上一策略"}  # 返回清晰的无操作结果。
        self.data["active"] = previous_name  # 恢复上一稳定策略。
        self.data["previous"] = active_name  # 保留当前版本供必要时再次切换。
        self.data["version"] += 1  # 增加策略版本标记人工回滚。
        self.save()  # 原子保存回滚后的活动指针。
        return {"changed": True, "active": previous_name, "message": f"已回滚到 {previous_name}"}  # 返回成功回滚结果。

    def optimize(self):  # 根据累计样本执行安全晋升或自动回滚。
        active_name = self.active_policy  # 保存优化前的活动策略名称。
        active = self.data["profiles"][active_name]  # 读取活动策略累计统计。
        candidates = []  # 初始化达到最少样本数的候选列表。
        for name, stats in self.data["profiles"].items():  # 检查所有固定策略的可比样本。
            if name != active_name and stats["games"] >= self.minimum_games and active["games"] >= self.minimum_games:  # 仅比较双方都达到门槛的策略。
                candidates.append((wilson_lower_bound(stats["wins"], stats["games"]), name, stats))  # 保存候选胜率置信下界。
        active_bound = wilson_lower_bound(active["wins"], active["games"])  # 计算当前策略的胜率置信下界。
        if candidates:  # 至少存在一套达到验证门槛的候选。
            candidate_bound, candidate_name, candidate = max(candidates)  # 选择胜率置信下界最高的候选。
            active_failure_rate = active["submit_failures"] / max(1, active["games"])  # 计算当前策略每局提交失败率。
            candidate_failure_rate = candidate["submit_failures"] / max(1, candidate["games"])  # 计算候选策略每局提交失败率。
            if candidate_bound > active_bound and candidate_failure_rate <= active_failure_rate:  # 只有胜率下界提高且操作质量不差时允许晋升。
                self.data["previous"] = active_name  # 保存上一稳定版本供自动回滚。
                self.data["active"] = candidate_name  # 将通过验证的候选设为活动策略。
                self.data["version"] += 1  # 增加策略版本号供逐局日志关联。
                self.save()  # 原子保存晋升结果。
                return {"changed": True, "active": candidate_name, "message": f"策略已从 {active_name} 晋升为 {candidate_name}"}  # 返回清晰的晋升报告。
        recent = active.get("recent", [])  # 读取当前策略最近二十局结果。
        previous_name = self.data.get("previous", active_name)  # 读取可以回滚的上一稳定策略。
        if len(recent) >= 20 and previous_name in POLICY_PROFILES and previous_name != active_name:  # 只有晋升后完整积累二十局才检查回滚。
            recent_win_rate = sum(int(item["won"]) for item in recent) / len(recent)  # 计算当前策略最近二十局胜率。
            historical_win_rate = active["wins"] / max(1, active["games"])  # 计算当前策略累计胜率作为稳定基准。
            recent_failure_rate = sum(int(item["submit_failures"]) for item in recent) / len(recent)  # 计算最近每局提交失败率。
            historical_failure_rate = active["submit_failures"] / max(1, active["games"])  # 计算累计每局提交失败率。
            if recent_win_rate < historical_win_rate - 0.10 or recent_failure_rate > historical_failure_rate + 0.05:  # 达到计划中的退化门槛时执行回滚。
                self.data["active"] = previous_name  # 恢复上一稳定策略。
                self.data["previous"] = active_name  # 保留退化版本供排查而不删除统计。
                self.data["version"] += 1  # 增加策略版本标记回滚事件。
                self.save()  # 原子保存回滚结果。
                return {"changed": True, "active": previous_name, "message": f"检测到策略退化，已回滚到 {previous_name}"}  # 返回清晰的回滚报告。
        needed = max(0, self.minimum_games - int(active["games"]))  # 计算当前策略距离验证门槛还需多少局。
        self.save()  # 即使没有切换也保存本次累计统计。
        return {"changed": False, "active": active_name, "message": f"策略保持 {active_name}，仍需积累或验证更多样本（当前策略还差 {needed} 局）"}  # 返回样本不足或无更优候选说明。
