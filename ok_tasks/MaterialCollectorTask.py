import json  # 导入 JSON 模块以保存每张素材的元数据。
import re  # 使用正则表达式识别新版选将标题。
from datetime import datetime  # 导入时间类型以生成会话和文件名称。
from pathlib import Path  # 导入路径类型以安全创建素材目录。

import cv2  # 导入 OpenCV 以编码 PNG 和计算感知哈希。

from ok import BaseTask  # 导入一次性任务基类供 GUI 启动素材采集。
from ok.feature.Box import Box  # 根据技能询问区域生成稳定的取消和确定点击框。
from ok.task.exceptions import TaskDisabledException  # 导入手动停止异常以正确完成运行报告。
from ok_tasks.ResourceInventory import missing_resource_keys, requirement_for_trigger  # 导入素材清单以显示并归类资源缺口。
from ok_tasks.RunRecorder import RunRecorder  # 导入持久化逐局运行记录器。


def compute_dhash(frame, hash_size=8):  # 计算画面的差值哈希用于素材去重。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将 BGR 游戏画面转换为灰度图。
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)  # 缩小画面并保留相邻像素关系。
    differences = resized[:, 1:] > resized[:, :-1]  # 比较每行相邻像素得到二进制特征。
    value = 0  # 初始化整数哈希值。
    for bit in differences.flatten():  # 依次遍历全部二进制特征位。
        value = (value << 1) | int(bit)  # 将当前特征位追加到整数哈希中。
    return value  # 返回可快速比较的差值哈希。


def is_near_duplicate(value, existing_values, max_distance):  # 判断当前哈希是否与已保存素材近似。
    return any((value ^ existing).bit_count() <= max_distance for existing in existing_values)  # 使用汉明距离过滤近似重复画面。


def infer_next_hero_slot(slots, frame_width, frame_height):  # 根据新版前三张等距卡片推导第四张武将卡区域。
    if len(slots) < 3:  # 少于三个可靠卡槽时无法安全推导下一位置。
        return None  # 返回空值禁止猜测点击。
    previous, current = slots[-2], slots[-1]  # 使用最靠右的两个已标注卡槽计算水平间距。
    next_x = current.x + (current.x - previous.x)  # 延续新版四卡布局的等距排列。
    next_x = max(0, min(int(next_x), int(frame_width) - current.width))  # 将推导区域限制在当前游戏画面内。
    next_y = max(0, min(int(current.y), int(frame_height) - current.height))  # 保持卡片顶部并限制垂直边界。
    return Box(next_x, next_y, current.width, current.height, name="hero_slot_4_layout_b")  # 返回可用于OCR和点击的第四卡槽。


def save_capture_files(frame, image_path, metadata):  # 将完整画面和元数据写入磁盘。
    image_path.parent.mkdir(parents=True, exist_ok=True)  # 确保当前会话的素材目录存在。
    encoded, buffer = cv2.imencode(".png", frame)  # 使用 OpenCV 将游戏画面编码为无损 PNG。
    if not encoded:  # 检查 PNG 编码是否成功。
        raise RuntimeError(f"无法编码素材截图: {image_path}")  # 编码失败时立即报告明确错误。
    buffer.tofile(str(image_path))  # 使用 tofile 兼容 Windows 中文路径。
    metadata_path = image_path.with_suffix(".json")  # 为截图生成同名 JSON 路径。
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")  # 以 UTF-8 保存可读元数据。
    return metadata_path  # 返回元数据路径供测试和日志使用。


class MaterialCollectorTask(BaseTask):  # 定义用户手动启动的自动素材采集任务。
    def __init__(self, *args, **kwargs):  # 初始化任务名称、说明和用户配置。
        super().__init__(*args, **kwargs)  # 先初始化 ok-script 的任务基础状态。
        self.name = "自动采集牌局素材"  # 设置 GUI 中显示的中文任务名称。
        self.description = "自动进入初级场，使用提示出牌，并保存去重后的牌局截图与元数据。"  # 说明任务的采集范围和行为。
        self.default_config.update({  # 注册所有可持久化的任务设置。
            "Target Rounds": 20,  # 设置默认采集二十局。
            "Auto Navigate": True,  # 默认自动完成大厅、场次和选将流程。
            "Auto Play With Hint": True,  # 默认通过游戏提示功能选择要出的牌。
            "Confirm Skills": True,  # 默认确认游戏弹出的主动技能。
            "Template Threshold": 0.8,  # 设置模板匹配的默认置信度阈值。
            "Poll Interval": 0.4,  # 设置状态轮询间隔以兼顾响应速度和性能。
            "Unknown Timeout": 45.0,  # 设置未知界面持续多久后安全停止。
            "Unknown Grace Period": 45.0,  # 为旧配置保留至少四十五秒的动画和加载宽限期。
            "Duplicate Distance": 3,  # 设置差值哈希的最大去重距离。
            "Output Folder": "material_collector",  # 设置 screenshots 下的素材子目录。
            "Data Root": "data/card_ai/runs",  # 将日志和素材保存到不会被程序启动清理的持久目录。
            "Diagnostic Mode": False,  # 默认执行自动点击，开启后只识别、记录和截图。
            "Pause Unknown Skills": True,  # 默认在交互型英雄未知二级技能界面暂停。
            "Replay Retry Limit": 5,  # 设置结算按钮连续识别失败后的安全恢复次数。
        })  # 完成默认配置注册。
        self.config_description.update({  # 为 GUI 配置项补充中文说明。
            "Target Rounds": "本次需要完成并采集的牌局数量。",  # 解释目标局数配置。
            "Auto Navigate": "自动点击主界面、初级场、中间武将和确认按钮。",  # 解释自动导航配置。
            "Auto Play With Hint": "轮到我方时先点击提示，采集选中状态后再出牌。",  # 解释提示出牌配置。
            "Confirm Skills": "技能确认弹窗出现时选择确认，否则选择取消。",  # 解释技能处理配置。
            "Template Threshold": "模板识别置信度，误识别时提高，漏识别时降低。",  # 解释模板阈值配置。
            "Poll Interval": "每次刷新游戏画面的间隔秒数。",  # 解释轮询间隔配置。
            "Unknown Timeout": "连续无法识别界面时，截图并停止前等待的秒数。",  # 解释未知超时配置。
            "Unknown Grace Period": "未知画面至少等待的秒数；宽限期内不提示未知图片，也不保存未知截图。",  # 解释旧配置同样会获得更长等待时间。
            "Duplicate Distance": "感知哈希去重距离，数值越大保留的相似截图越少。",  # 解释去重距离配置。
            "Output Folder": "素材保存到 screenshots 下的子目录名称。",  # 解释输出目录配置。
            "Data Root": "逐局日志和事件截图的持久保存目录。",  # 解释持久数据目录配置。
            "Diagnostic Mode": "只识别、记录和截图，不执行任何游戏点击。",  # 解释诊断模式不会操作游戏。
            "Pause Unknown Skills": "遇到未标注的英雄二级技能界面时保存现场并暂停。",  # 解释未知技能安全行为。
            "Replay Retry Limit": "结算界面连续找不到再来一局按钮时的最大重试次数。",  # 解释结算恢复门槛。
        })  # 完成配置说明注册。

    def run(self):  # 执行配置局数的自动素材采集流程。
        self._prepare_session()  # 初始化本次采集会话的目录和运行状态。
        self.log_info(f"素材采集开始，输出目录: {self.output_folder}", notify=True)  # 在日志和通知中显示输出目录。
        final_status = "completed"  # 默认将达到目标局数的运行标记为完成。
        try:  # 确保手动停止和运行异常时仍能关闭逐局日志并执行安全优化。
            while self.completed_rounds < self.config.get("Target Rounds", 20):  # 在达到目标局数前持续采集。
                frame = self.next_frame()  # 获取一张新的游戏窗口画面。
                raw_state, matched_box = self._classify_state()  # 使用现有模板识别当前原始界面状态。
                if raw_state != "skill_prompt" and self._should_pause_unknown_skill():  # 交互技能确认后的下一界面优先于普通出牌模板处理，避免二级按钮误命中已有状态。
                    self._capture_state("unknown_skill_interaction", frame, force=True)  # 强制保存二级技能完整现场。
                    self._record_event("unknown_state", state="unknown_skill_interaction", detected_state=raw_state, hero=getattr(self, "current_hero", None))  # 记录英雄、原始分类和未知技能上下文。
                    handler = getattr(self, "_handle_unknown_skill_interaction", None)  # 查询AI子类提供的语义技能处理器。
                    if callable(handler) and handler(frame):  # 只有完整识牌、选项和确认按钮均验证成功时自动处理。
                        self.pending_skill_capture = False  # 已完成交互后清除未知技能锁。
                        continue  # 重新获取界面确认游戏已经继续。
                    self.pending_skill_capture = False  # 防止用户恢复任务后在同一画面反复触发暂停。
                    self.log_warning("发现未标注的英雄二级技能界面，已保存现场并暂停。", notify=True)  # 通知用户进行标注或人工处理。
                    self.pause()  # 使用 BaseTask 已有暂停接口等待用户恢复或停止任务。
                    continue  # 恢复后重新获取新画面而不沿用暂停前状态。
                if raw_state != "unknown" and raw_state != "skill_prompt":  # 技能确认后直接回到已知牌局状态表示没有未知二级界面。
                    self.pending_skill_capture = False  # 清除技能捕获标记避免稍后的普通等待画面被误判。
                state = self._normalize_state(raw_state)  # 将牌局内的空白桌面和动画统一归类为正常等待状态。
                self._track_state_transition(state)  # 记录状态变化并维护回合唯一编号。
                if state not in {"result_win", "result_loss"}:  # 检查当前画面是否已经离开结算界面。
                    self.result_latched = False  # 离开结算后允许下一局结果被计数。
                    self.replay_attempts = 0  # 离开结算后清除再来一局重试计数。
                if state == "unknown":  # 处理所有模板均无法识别的界面。
                    if self._handle_unknown_wait(frame):  # 宽限期结束后才记录、截图并报告真正持续的未知界面。
                        final_status = "stopped_unknown"  # 将运行标记为未知界面安全停止。
                        return  # 结束任务以避免继续误点。
                else:  # 当前画面属于已知游戏状态。
                    self.unknown_since = None  # 清除未知界面的起始时间。
                    self._capture_state(state, frame)  # 只保存已识别状态，避免动画第一帧立即显示成未知图片。
                    if not self.config.get("Diagnostic Mode", False):  # 诊断模式只记录界面而不执行任何点击。
                        self._handle_state(state, matched_box, frame)  # 根据状态执行安全的导航或采集动作。
                self.sleep(self.config.get("Poll Interval", 0.4))  # 等待下一轮并自动响应用户停止任务。
            self.log_info(f"素材采集完成，共完成 {self.completed_rounds} 局。", notify=True)  # 通知用户目标局数已经完成。
        except TaskDisabledException:  # 单独处理用户主动停止任务的正常控制流。
            final_status = "stopped"  # 将运行标记为用户停止而不是程序失败。
            raise  # 继续交给 ok-script 完成停止通知和任务清理。
        except Exception:  # 捕获其他任务层异常以便先写入可恢复报告。
            final_status = "failed"  # 将运行标记为异常结束。
            raise  # 保留原异常交给 ok-script 正常错误处理。
        finally:  # 无论完成、停止或失败都执行会话收尾。
            self._finalize_session(final_status)  # 写入中文报告并允许 AI 子类执行策略优化。

    def _prepare_session(self):  # 创建会话目录并重置本次运行状态。
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")  # 使用当前时间生成唯一会话名称。
        self.run_recorder = RunRecorder(self.config.get("Data Root", "data/card_ai/runs"), session_name, self.name, dict(self.config))  # 创建不会被应用启动清理的会话记录器。
        self.output_folder = self.run_recorder.root  # 将素材输出统一指向持久会话目录。
        self.completed_rounds = 0  # 重置已完成牌局数量。
        self.sequence = 0  # 重置素材文件连续编号。
        self.state_hashes = {}  # 重置按状态保存的感知哈希集合。
        self.unknown_since = None  # 重置未知界面计时器。
        self.last_action_state = None  # 重置上一次执行动作的状态。
        self.last_action_time = 0.0  # 重置上一次执行动作的时间。
        self.result_latched = False  # 重置结算界面的单次计数锁存状态。
        self.in_match = False  # 重置当前是否已经进入正式牌局的状态。
        self.previous_state = None  # 重置状态转换记录。
        self.turn_sequence = 0  # 重置我方操作回合连续编号。
        self.current_round_id = None  # 重置当前我方回合唯一编号。
        self.turn_action_completed = False  # 重置当前回合动作提交锁。
        self.pending_skill_capture = False  # 重置未知二级技能捕获标记。
        self.replay_attempts = 0  # 重置结算按钮重试次数。
        self.missing_resources = missing_resource_keys(self.feature_exists)  # 根据当前模板集计算资源缺口。
        self.info_set("Missing Resources", len(self.missing_resources))  # 在任务状态面板显示缺失资源数量。
        self._record_event("session_ready", missing_resources=self.missing_resources)  # 将资源缺口写入会话日志。

    def _record_event(self, event_type, **payload):  # 安全追加一条结构化逐局事件。
        recorder = getattr(self, "run_recorder", None)  # 兼容绕过正常初始化的纯逻辑测试对象。
        if recorder is None:  # 没有记录器时直接跳过写盘。
            return None  # 返回空值保持旧测试行为。
        return recorder.event(event_type, **payload)  # 将事件立即写入当前牌局 JSONL。

    def _track_state_transition(self, state):  # 记录状态变化并为每个我方回合生成唯一编号。
        previous = getattr(self, "previous_state", None)  # 读取上一帧稳定状态。
        if previous == state:  # 状态未变化时无需重复写入转换事件。
            return  # 保留当前回合锁并结束处理。
        self._record_event("state_transition", previous=previous, current=state)  # 写入可回放的状态转换事件。
        play_states = {"play_lead", "play_follow", "play_no_legal"}  # 定义所有我方可操作回合状态。
        if state in play_states:  # 新进入我方操作状态时建立回合编号。
            self.turn_sequence = getattr(self, "turn_sequence", 0) + 1  # 增加本局我方回合序号。
            recorder = getattr(self, "run_recorder", None)  # 读取当前持久日志记录器。
            game_id = getattr(recorder, "game_id", None) or "recovered"  # 读取当前牌局编号或断点恢复标记。
            session_id = getattr(recorder, "session_id", None) or "session"  # 纳入会话编号，避免每次启动的game_0001落入同一小流量桶。
            self.current_round_id = f"{session_id}_{game_id}_turn_{self.turn_sequence:04d}"  # 生成跨会话、跨牌局唯一回合编号。
            self.turn_action_completed = False  # 允许新回合执行一次决策和提交。
            self.info_set("Round", self.current_round_id)  # 在任务状态面板显示当前我方回合编号。
        elif previous in play_states:  # 离开我方操作状态时清除回合执行锁。
            self.current_round_id = None  # 清除已经结束的回合编号。
            self.turn_action_completed = False  # 为下一回合恢复可执行状态。
        self.previous_state = state  # 保存当前状态供下一帧比较。

    def _should_pause_unknown_skill(self):  # 判断未知画面是否属于需要暂停的英雄二级技能。
        if not self.config.get("Pause Unknown Skills", True):  # 用户关闭安全暂停时继续普通等待分类。
            return False  # 返回无需暂停。
        return bool(getattr(self, "pending_skill_capture", False))  # 只在确认交互型英雄技能后触发一次暂停。

    def _normalize_state(self, raw_state):  # 将原始模板结果转换为状态机可安全处理的稳定状态。
        if raw_state != "unknown":  # 已识别状态不需要额外推断。
            return raw_state  # 保留模板分类结果。
        recorder = getattr(self, "run_recorder", None)  # 读取逐局记录器辅助判断是否已经进入一局游戏。
        active_game = getattr(recorder, "game_id", None) is not None  # 选将完成后即使尚未出现叫分按钮也属于活动牌局。
        if getattr(self, "in_match", False) or active_game:  # 牌局中的空白桌面通常只是其他玩家操作或技能动画。
            return "match_waiting"  # 正常等待，不启动未知超时、不提示未知图片。
        return "unknown"  # 牌局外持续无法识别时才进入安全超时流程。

    def _finalize_session(self, status):  # 关闭素材采集会话并生成中文总结。
        recorder = getattr(self, "run_recorder", None)  # 读取可能已经成功创建的运行记录器。
        if recorder is None:  # 初始化失败发生在记录器创建前时无法写入报告。
            return None  # 返回空值交由 ok-script 报告原异常。
        summary = recorder.finalize(status=status, missing_resources=getattr(self, "missing_resources", []))  # 写入机器总结和中文报告。
        self.info_set("Run Report", str(recorder.root / "报告.txt"))  # 在任务状态面板显示报告路径。
        return summary  # 返回总结供 AI 子类扩展。

    def _classify_state(self):  # 按优先级识别当前游戏界面状态。
        threshold = self.config.get("Template Threshold", 0.8)  # 读取用户设置的模板阈值。
        high_priority_checks = [  # 先识别可能覆盖底部操作按钮的结算、弹窗和选将界面。
            ("result_win", ["Is it a victory"]),  # 优先识别胜利结算界面。
            ("result_loss", ["be defeated", "result_loss_variant"]),  # 使用两套素材识别失败结算界面。
            ("skill_prompt", ["Confirm the use of skills", "Cancel the use of the skill", "skill_use_prompt"]),  # 优先使用按钮模板，并以完整询问区域兜底识别技能弹窗。
            ("hero_select", ["select"]),  # 识别武将选择界面。
            ("bidding", ["1 point", "two points", "three points"]),  # 识别叫分界面。
        ]  # 完成高优先级模板定义。
        for state, feature_names in high_priority_checks:  # 依次检查不会与普通出牌按钮混淆的状态。
            matched_box = self._find_first_feature(feature_names, threshold)  # 查找该状态的第一个有效模板。
            if matched_box is not None:  # 当前状态至少命中一个模板。
                return state, matched_box  # 立即返回覆盖层状态，禁止继续识别底部按钮。
        hero_select_box = self._find_hero_select_by_ocr()  # 新版灰色选定按钮无法稳定匹配旧模板时使用标题和按钮双重OCR。
        if hero_select_box is not None:  # 两处选将文字同时命中后才认可新版选将界面。
            return "hero_select", hero_select_box  # 返回选定按钮框供状态机记录，实际确认仍在选将流程执行。
        no_legal_box = self._find_first_feature(["play_no_legal_controls"], threshold)  # 优先检查三个按钮均不可提交的整体控制区。
        if no_legal_box is not None:  # 游戏已经明确显示当前没有合法跟牌动作。
            return "play_no_legal", no_legal_box  # 返回安全不出状态。
        pass_box = self._find_first_feature(["Not"], threshold)  # 三按钮跟牌布局左侧必须存在“不出”。
        hint_box = self._find_first_feature(["hint"], threshold)  # 三按钮跟牌布局中间存在“提示”。
        follow_play_box = self._find_first_feature(["play_card_layout_b"], threshold)  # 三按钮跟牌布局右侧存在“出牌”。
        if (pass_box is not None and (hint_box is not None or follow_play_box is not None)) or (hint_box is not None and follow_play_box is not None):  # 至少两个位置一致才认可三按钮布局。
            return "play_follow", follow_play_box or pass_box or hint_box  # 跟牌状态始终绑定右侧提交布局。
        lead_play_box = self._find_first_feature(["play_card_layout_a"], threshold)  # 单按钮主动布局只在屏幕中间显示“出牌”。
        if lead_play_box is not None and pass_box is None and follow_play_box is None:  # 排除三按钮布局的左、右位置后才认可主动出牌。
            return "play_lead", lead_play_box  # 返回只有中间按钮的主动牌权状态。
        navigation_checks = [  # 最后识别不包含牌局操作按钮的导航界面。
            ("arena", ["Primary field"]),  # 识别经典模式场次选择界面。
            ("main", ["Enter the game type selection area"]),  # 识别游戏主界面的自由匹配入口。
            ("waiting", ["waiting time"]),  # 识别匹配等待界面。
        ]  # 完成导航状态模板定义。
        for state, feature_names in navigation_checks:  # 依次尝试每一种导航状态。
            matched_box = self._find_first_feature(feature_names, threshold)  # 查找该状态的第一个有效模板。
            if matched_box is not None:  # 当前状态至少命中一个模板。
                return state, matched_box  # 返回识别状态和用于点击的模板框。
        return "unknown", None  # 所有模板均未命中时标记为未知状态。

    def _find_hero_select_by_ocr(self):  # 使用新版标题和底部按钮共同确认四武将选择界面。
        if not hasattr(self, "_executor"):  # 纯逻辑测试或尚未绑定执行器时不能调用实时OCR。
            return None  # 跳过OCR兜底并保持旧模板分类行为。
        titles = self.ocr(0.24, 0.10, 0.76, 0.24, match=re.compile("请.*选择.*武将"), threshold=0.18, log=False) or []  # 只扫描画面上方稳定标题。
        if not titles:  # 单独出现选定文字不足以证明当前处于选将界面。
            return None  # 缺少标题时拒绝误判其他确认弹窗。
        buttons = self.ocr(0.35, 0.75, 0.65, 0.92, match="选定", threshold=0.18, log=False) or []  # 在底部中央读取新版灰色或黄色选定按钮。
        return buttons[0] if buttons else None  # 仅在按钮也识别成功时返回可点击文字框。

    def _click_hero_confirm(self):  # 兼容旧模板和新版按钮皮肤确认选中的武将。
        if self._click_feature("select", after_sleep=1.5):  # 优先使用已有黄色选定按钮模板。
            return True  # 模板点击成功后结束确认流程。
        buttons = self.wait_ocr(0.35, 0.75, 0.65, 0.92, match="选定", threshold=0.18, time_out=1.5, raise_if_not_found=False, log=False) or []  # 等待选中卡片后按钮变为可确认状态。
        if not buttons:  # 新旧按钮均无法可靠识别时禁止猜测坐标。
            return False  # 返回失败供调用方保留现场并重试。
        self.click_box(buttons[0], after_sleep=1.5)  # 点击OCR确认的选定文字中心并等待进入牌局。
        return True  # 报告新版按钮已经安全点击。

    def _find_first_feature(self, feature_names, threshold):  # 在同一帧中查找候选模板列表。
        for feature_name in feature_names:  # 按优先级逐个检查模板。
            if not self.feature_exists(feature_name):  # 跳过当前素材集中不存在的可选模板。
                continue  # 继续检查同一状态的下一个模板。
            matched_box = self.find_one(feature_name, threshold=threshold)  # 在固定标注区域匹配模板。
            if matched_box is not None:  # 当前模板匹配成功。
                return matched_box  # 返回最高置信度的匹配框。
        return None  # 候选模板全部失败时返回空值。

    def _handle_state(self, state, matched_box, frame):  # 执行当前状态对应的安全动作。
        if state in {"play_lead", "play_follow", "play_no_legal"}:  # 保存本轮实际按钮布局供后续提交函数限定点击区域。
            self.current_play_state = state  # 主动回合只允许中间出牌，跟牌回合只允许右侧出牌。
        if not self._action_ready(state):  # 防止同一界面在动画期间被连续点击。
            return  # 尚未到动作冷却时间时等待下一帧。
        if state == "result_win" or state == "result_loss":  # 处理胜利或失败结算界面。
            self.in_match = False  # 结算出现后退出牌局等待状态。
            if not self.result_latched:  # 检查当前结算是否尚未计入完成局数。
                self.result_latched = True  # 锁存当前结算直到画面离开结算状态。
                self.completed_rounds += 1  # 将当前结算计入已完成局数。
                self.info_set("Completed Rounds", self.completed_rounds)  # 在 GUI 中显示实时完成局数。
                self._capture_state(state, frame, force=True)  # 强制保留每一局的最终结算画面。
                recorder = getattr(self, "run_recorder", None)  # 读取逐局记录器供结算收尾。
                if recorder is not None:  # 正常运行已创建记录器时结束本局日志。
                    recorder.end_game(state == "result_win", hero=getattr(self, "current_hero", None), position=getattr(self, "resolved_ai_position", None), policy_id=getattr(self, "current_policy_id", "balanced"), submit_failures=getattr(self, "game_submit_failures", 0))  # 写入胜负、英雄、身份和操作失败数。
            if self.completed_rounds < self.config.get("Target Rounds", 20):  # 尚未达到目标时准备下一局。
                clicked = self._click_feature("Play another round", after_sleep=2.0)  # 尝试点击再来一局并等待界面切换。
                if not clicked:  # 当前帧尚未出现按钮或模板暂时未匹配成功。
                    self.replay_attempts = getattr(self, "replay_attempts", 0) + 1  # 累加当前结算按钮识别失败次数。
                    replay_buttons = self.wait_ocr(0.64, 0.78, 0.96, 0.98, match="再来一局", time_out=1.0, raise_if_not_found=False, log=False) if hasattr(self, "_executor") else []  # 在右下角使用文字识别兼容胜负不同按钮皮肤，并兼容无执行器测试对象。
                    if replay_buttons:  # OCR 已可靠识别到再来一局文字。
                        self.click_box(replay_buttons[0], after_sleep=2.0)  # 点击文字框中心并等待下一局界面。
                        self._record_event("replay_clicked", source="ocr", attempt=self.replay_attempts)  # 记录结算恢复来源和尝试次数。
                    elif self.replay_attempts >= self.config.get("Replay Retry Limit", 5):  # 连续达到安全恢复门槛时退出结算界面。
                        self._capture_state("replay_button_missing", frame, force=True)  # 保存缺失按钮现场供补充模板。
                        self.click_relative(0.04, 0.06, after_sleep=2.0)  # 根据结算截图点击左上角返回按钮回到可恢复流程。
                        self._record_event("replay_recovery", action="back_to_lobby", attempts=self.replay_attempts)  # 记录退回大厅恢复动作。
                        self.replay_attempts = 0  # 清除当前结算重试次数。
                    else:  # 尚未达到恢复门槛时保留当前结算等待下一轮。
                        self.log_warning("结算界面暂未识别到再来一局按钮，将在下一轮继续尝试。")  # 记录可恢复的按钮识别失败。
            return  # 完成结算状态处理。
        if not self.config.get("Auto Navigate", True) and state in {"main", "arena", "hero_select", "bidding"}:  # 尊重关闭自动导航的配置。
            return  # 只采集当前画面而不执行导航点击。
        if state == "main":  # 处理游戏主界面。
            self.click_box(matched_box, after_sleep=1.5)  # 点击自由匹配入口并等待场次界面。
            return  # 完成主界面动作。
        if state == "arena":  # 处理经典模式场次选择界面。
            self.click_box(matched_box, after_sleep=0.5)  # 点击初级场卡片以选中目标场次。
            self.click_relative(0.86, 0.91, after_sleep=2.0)  # 按截图估算点击右下角出战按钮。
            return  # 完成场次选择动作。
        if state == "hero_select":  # 处理武将选择界面。
            recorder = getattr(self, "run_recorder", None)  # 读取逐局记录器准备新牌局。
            if recorder is not None and recorder.game_id is None:  # 当前没有活动牌局时从选将界面开始新记录。
                recorder.start_game(hero=getattr(self, "current_hero", None), position=None, policy_id=getattr(self, "current_policy_id", "balanced"))  # 创建新牌局目录和开始事件。
            self.game_submit_failures = 0  # 为新牌局重置提交失败计数。
            self._select_middle_hero()  # 固定选择三个候选武将中的中间武将。
            return  # 完成选将动作。
        if state == "bidding":  # 处理叫分界面。
            self.in_match = True  # 叫分出现后标记已经进入正式牌局。
            recorder = getattr(self, "run_recorder", None)  # 读取逐局记录器支持从牌局中途断点恢复。
            if recorder is not None:  # 正常运行存在记录器时确保当前牌局已建立。
                recorder.ensure_game(hero=getattr(self, "current_hero", None), position=getattr(self, "resolved_ai_position", None), policy_id=getattr(self, "current_policy_id", "balanced"))  # 建立或复用牌局日志。
            self._click_feature("1 point", after_sleep=1.0)  # 固定叫一分以获得更多完整对局素材。
            self._record_event("bidding", bid=1)  # 记录固定叫一分动作供回放分析。
            return  # 完成叫分动作。
        if state == "skill_prompt":  # 处理技能确认弹窗。
            feature_name = "Confirm the use of skills" if self.config.get("Confirm Skills", True) else "Cancel the use of the skill"  # 根据配置选择确认或取消。
            interactive_heroes = {"夏侯惇", "关羽", "徐盛", "诸葛均", "凌统", "卢植"}  # 定义当前缺少二级交互素材的英雄集合。
            self.pending_skill_capture = feature_name == "Confirm the use of skills" and getattr(self, "current_hero", None) in interactive_heroes  # 仅在确认交互型英雄技能后等待未知二级界面。
            if feature_name == "Confirm the use of skills":  # 任何已确认技能都可能立即获得、弃置、复制或改变手牌。
                self.hand_change_pending = True  # 通知 AI 子类在下次读取手牌时执行连续两帧确认；普通采集任务允许动态字段存在。
            self._record_event("skill_prompt", hero=getattr(self, "current_hero", None), action="confirm" if self.pending_skill_capture else "cancel_or_passive")  # 记录技能确认行为和英雄上下文。
            self._click_skill_prompt_action(feature_name, after_sleep=1.0)  # 优先点击按钮模板，模板漂移时使用完整询问区域中的固定按钮位置。
            return  # 完成技能弹窗动作。
        if state == "play_no_legal":  # 处理游戏明确没有合法跟牌组合的回合。
            self.in_match = True  # 无合法出牌按钮出现时确认当前处于牌局中。
            if getattr(self, "turn_action_completed", False):  # 当前回合已经执行过安全动作时禁止重复点击。
                return  # 等待游戏切换到下一玩家。
            self._click_feature("Not", after_sleep=1.0)  # 直接点击不出，避免无效提示和出牌重试。
            self.turn_action_completed = True  # 锁定当前回合已经完成不出动作。
            self._record_event("action_submitted", round_id=getattr(self, "current_round_id", None), action=[], source="no_legal")  # 记录无合法牌时的不出动作。
            return  # 完成无合法跟牌回合处理。
        if state == "play_follow" and self.config.get("Auto Play With Hint", True):  # 处理开启提示出牌的普通跟牌回合。
            self.in_match = True  # 我方操作按钮出现时确认当前处于牌局中。
            if getattr(self, "turn_action_completed", False):  # 当前回合已经成功提交时禁止再次选牌。
                return  # 等待界面切换确认动作完成。
            self._play_with_hint(frame)  # 点击提示、保存选中状态并提交出牌。
            return  # 完成普通跟牌回合处理。
        if state == "play_lead" and self.config.get("Auto Play With Hint", True):  # 处理地主首次没有提示按钮的主动出牌回合。
            self.in_match = True  # 我方主动出牌按钮出现时确认当前处于牌局中。
            if getattr(self, "turn_action_completed", False):  # 当前回合已经成功提交时禁止再次选择同一手牌。
                return  # 等待牌权界面消失。
            self._play_lowest_single(frame)  # 选择最右侧最小单牌并提交。

    def _select_middle_hero(self):  # 固定选择当前三卡或四卡布局中的第二名武将。
        slot_name = next((name for name in ("hero_slot_2_layout_b", "hero_slot_2_layout_a") if self.feature_exists(name)), None)  # 新版四卡布局优先，旧三卡布局继续兼容。
        if slot_name is not None:  # 检查任一第二武将槽位标注是否可用。
            middle_slot = self.get_box_by_name(slot_name)  # 读取当前布局第二张武将卡片的固定区域。
            self.click_box(middle_slot, after_sleep=0.4)  # 点击中间武将卡片完成选中。
        else:  # 当前素材数据缺少中间武将槽位标注。
            self.click_relative(0.50, 0.48, after_sleep=0.4)  # 使用选将截图估算的屏幕中间坐标兜底。
        selected_frame = self.next_frame()  # 获取中间武将被选中后的新画面。
        self._capture_state("hero_middle_selected", selected_frame, force=True)  # 保存中间武将被选中的界面。
        self.click_relative(0.15, 0.82, after_sleep=0.4)  # 点击左下方空白背景以关闭武将技能说明浮层。
        if self._click_hero_confirm():  # 通过旧模板或新版OCR按钮进入牌局并确认动作已经执行。
            self.in_match = True  # 从此刻起空白桌面属于正常牌局等待，不再误报未知图片。

    def _play_with_hint(self, frame):  # 使用游戏提示功能安全完成一次出牌。
        hint_box = self._find_first_feature(["hint"], self.config.get("Template Threshold", 0.8))  # 查找当前回合的提示按钮。
        if hint_box is None:  # 模板受按钮亮度或动画影响时使用文字识别兜底。
            hint_buttons = self.wait_ocr(0.38, 0.50, 0.62, 0.68, match="提示", time_out=1.5, raise_if_not_found=False, log=False)  # 在底部操作区等待提示文字。
            hint_box = hint_buttons[0] if hint_buttons else None  # 使用第一个识别到的提示按钮框。
        if hint_box is None:  # 当前布局没有可用的提示按钮。
            self._capture_state("play_turn_without_hint", frame, force=True)  # 保存缺少提示按钮的回合现场。
            return  # 不猜测点击位置以避免误出牌。
        selected_frame = frame  # 初始化最后一次提示后的画面用于失败素材保存。
        for attempt in range(3):  # 最多重试三次提示以处理点击被动画或窗口焦点吞掉的情况。
            self.click_box(hint_box, after_sleep=0.55)  # 点击提示并等待游戏抬起合法手牌。
            selected_frame = self.next_frame()  # 获取本次提示完成后的最新画面。
            if self._submit_selected_cards(selected_frame, "hint_without_play_button", capture_failure=False):  # 仅在黄色可用出牌按钮出现后提交。
                self._capture_state("hand_selected", selected_frame, force=True)  # 保存已经确认可提交的自动选牌结果。
                self.turn_action_completed = True  # 锁定当前回合避免状态切换延迟时再次点击提示。
                self.hand_change_pending = True  # 无论真实出牌或失败后不出，英雄技能都可能改变下一份手牌。
                self._record_event("action_submitted", round_id=getattr(self, "current_round_id", None), source="game_hint")  # 记录游戏提示完成的自动动作。
                return  # 出牌按钮已经点击后结束本回合提示流程。
            if attempt < 2:  # 尚有剩余次数时重新定位提示按钮以适配轻微界面移动。
                refreshed_hint = self._find_first_feature(["hint"], self.config.get("Template Threshold", 0.8))  # 从最新画面再次匹配提示按钮。
                if refreshed_hint is not None:  # 新画面仍能识别到提示按钮。
                    hint_box = refreshed_hint  # 使用刷新后的按钮框执行下一次提示。
                self.sleep(0.25)  # 给卡牌动画留下短暂稳定时间再重试。
        self._capture_state("hint_without_play_button", selected_frame, force=True)  # 保存连续提示后仍未形成有效选牌的现场。
        self._record_event("submit_failed", round_id=getattr(self, "current_round_id", None), stage="hint_selection")  # 记录提示后按钮仍不可用的失败阶段。
        self.log_warning("连续三次点击提示后仍未出现黄色出牌按钮，将在下一轮继续重试。")  # 记录可恢复失败并避免误点灰色按钮。

    def _play_lowest_single(self, frame):  # 主动出牌时选择一张在各种手牌数量下都可点击的单牌。
        self.click_relative(0.50, 0.80, after_sleep=0.5)  # 点击底部中央手牌以适配满手牌和少量牌自动居中的布局。
        selected_frame = self.next_frame()  # 获取中央单牌被选中后的新画面。
        self._capture_state("lead_card_selected", selected_frame, force=True)  # 保存地主首次选牌状态。
        self.turn_action_completed = bool(self._submit_selected_cards(selected_frame, "lead_without_play_button"))  # 等待提交主动出牌并锁定成功回合。
        if self.turn_action_completed:  # 主动单牌已经完成提交时写入回放日志。
            self.hand_change_pending = True  # 出牌后可能立即触发英雄加牌、回收或变点动画。
            self._record_event("action_submitted", round_id=getattr(self, "current_round_id", None), source="center_single_fallback")  # 记录少量手牌兼容兜底来源。

    def _play_button_feature_names(self):  # 根据已经确认的回合布局返回唯一允许点击的出牌区域。
        state = getattr(self, "current_play_state", None)  # 读取状态机保存的主动或跟牌状态。
        if state == "play_follow" or state == "play_no_legal":  # 三按钮布局中只有最右侧按钮能够提交牌组。
            return ("play_card_layout_b",)  # 严禁检查中间区域，避免把黄色“提示”当成“出牌”。
        if state == "play_lead":  # 主动牌权界面只有屏幕中间一个出牌按钮。
            return ("play_card_layout_a", "Confirming the Play")  # 使用中间布局区域及其旧名称兼容素材。
        return ("play_card_layout_b", "play_card_layout_a", "Confirming the Play")  # 无状态测试或旧调用优先检查右侧，降低误点提示风险。

    def _find_active_play_button(self, frame):  # 通过黄色背景识别当前布局中唯一可用的出牌按钮。
        if frame is None or frame.size == 0:  # 过滤缺失或空白画面以避免 OpenCV 裁切异常。
            return None  # 返回未找到交由模板或下一轮处理。
        for feature_name in self._play_button_feature_names():  # 只检查当前状态允许提交的按钮位置。
            if not self.feature_exists(feature_name):  # 当前素材集没有对应布局区域时跳过。
                continue  # 继续检查下一种已标注布局。
            try:  # 捕获合并素材中类别存在但固定区域尚未加载的情况。
                region = self.get_box_by_name(feature_name)  # 获取适配当前分辨率的出牌按钮固定区域。
            except ValueError:  # 固定区域无法读取时不影响其他布局检测。
                continue  # 继续检查下一种按钮区域。
            left = max(0, region.x)  # 将按钮左边界限制在当前画面内。
            top = max(0, region.y)  # 将按钮上边界限制在当前画面内。
            right = min(frame.shape[1], region.x + region.width)  # 将按钮右边界限制在当前画面内。
            bottom = min(frame.shape[0], region.y + region.height)  # 将按钮下边界限制在当前画面内。
            if right <= left or bottom <= top:  # 无效矩形无法执行颜色判断。
                continue  # 跳过当前异常标注区域。
            patch = frame[top:bottom, left:right]  # 裁出当前布局的完整出牌按钮区域。
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)  # 转换到 HSV 以稳定区分黄色可用态和灰色禁用态。
            yellow_mask = cv2.inRange(hsv, (8, 60, 120), (45, 255, 255))  # 提取金黄与橙黄色按钮背景像素。
            yellow_ratio = cv2.countNonZero(yellow_mask) / float(yellow_mask.size)  # 计算黄色像素占整个按钮区域的比例。
            if yellow_ratio >= 0.12:  # 黄色背景超过十二个百分点时判定按钮已经可点击。
                return region  # 返回固定按钮框供现有点击流程直接提交。
        return None  # 所有布局均为灰色或尚未出现时返回未找到。

    def _submit_selected_cards(self, selected_frame, failure_state, capture_failure=True):  # 提交已经被提示或主动点击选中的手牌。
        self.last_submit_used_pass = False  # 每次提交前清除上一次“失败后不出”恢复标记。
        has_button_regions = any(self.feature_exists(name) for name in self._play_button_feature_names())  # 只检查当前主动或跟牌布局对应的出牌区域。
        confirm_box = None  # 初始化尚未确认黄色可用按钮。
        verification_frame = selected_frame  # 从选牌后的第一帧开始检查动画状态。
        if has_button_regions:  # 只有当前素材提供真实按钮框时才执行颜色等待。
            for frame_index in range(6):  # 允许卡牌依次抬起后按钮延迟数帧才由灰变黄。
                confirm_box = self._find_active_play_button(verification_frame)  # 每一帧都按实际黄色像素验证可点击状态。
                if confirm_box is not None:  # 按钮已经真正变黄时停止等待。
                    break  # 进入唯一一次提交点击。
                if frame_index < 5:  # 最后一帧之后无需继续取图。
                    self.sleep(0.12)  # 短暂等待选牌动画完成。
                    verification_frame = self.next_frame()  # 使用新画面重新做颜色判断，不复用灰色模板结果。
        if confirm_box is not None:  # 模板成功找到提示后的可用出牌按钮。
            self.click_box(confirm_box, after_sleep=1.0)  # 点击黄色出牌按钮提交已选手牌。
            self.last_submit_used_pass = bool(self._pass_if_submit_failed())  # 多帧确认出牌是否成功，并保留失败后实际不出的结果。
            return True  # 返回已经完成出牌按钮点击。
        if not has_button_regions:  # 旧素材集确实没有任何按钮区域时才允许使用文字识别兼容。
            play_buttons = self.wait_ocr(0.45, 0.48, 0.75, 0.68, match="出牌", time_out=2.0, raise_if_not_found=False, log=False)  # 在缺少模板时等待操作区出牌文字。
            if play_buttons:  # OCR 在旧素材配置中找到了出牌按钮。
                self.click_box(play_buttons[0], after_sleep=1.0)  # 点击旧配置识别到的出牌文字中心。
                self.last_submit_used_pass = bool(self._pass_if_submit_failed())  # 兼容路径同样区分真实出牌与失败后不出。
                return True  # 返回兼容提交已经执行。
        if capture_failure:  # 最终失败或主动出牌调用需要保存异常现场时执行记录。
            self._capture_state(failure_state, selected_frame, force=True)  # 保存选牌后没有识别到可用出牌按钮的异常现场。
            self.game_submit_failures = getattr(self, "game_submit_failures", 0) + 1  # 累加本局提交失败次数供策略质量门槛使用。
            self._record_event("submit_failed", round_id=getattr(self, "current_round_id", None), stage=failure_state)  # 写入失败阶段和回合编号。
            self.log_warning("选牌后未识别到黄色可用出牌按钮，将在下一轮继续重试。")  # 明确记录安全等待原因。
        return False  # 返回未提交以便提示流程再次选牌。

    def _pass_if_submit_failed(self):  # 出牌点击后仍停留在我方回合时使用不出兜底。
        threshold = self.config.get("Template Threshold", 0.8)  # 读取当前模板匹配阈值。
        persistent_pass_box = None  # 初始化持续存在的不出按钮。
        latest_frame = None  # 初始化用于异常素材保存的最新画面。
        retried_submit = False  # 记录是否已经对黄色出牌按钮执行过一次补点。
        for frame_index in range(5):  # 延长到五帧确认网络和出牌动画延迟。
            latest_frame = self.next_frame()  # 获取出牌点击后的最新游戏画面。
            persistent_pass_box = self._find_first_feature(["Not"], threshold)  # 检查我方回合的不出按钮是否仍存在。
            if persistent_pass_box is None:  # 不出按钮已经消失表示出牌成功或回合结束。
                return False  # 无需执行不出兜底。
            active_play_box = self._find_active_play_button(latest_frame)  # 按颜色检查两种布局的黄色出牌按钮是否仍然保留。
            if frame_index >= 1 and active_play_box is not None and not retried_submit:  # 连续两帧仍可提交时判定首次点击可能未生效。
                self.click_box(active_play_box, after_sleep=0.7)  # 再点击一次黄色出牌按钮而不是立即放弃本手牌。
                retried_submit = True  # 防止网络卡顿期间连续重复提交。
                continue  # 使用下一帧确认补点是否成功。
            self.sleep(0.4)  # 短暂等待下一帧确认按钮不是动画残留。
        self._capture_state("play_submit_failed_before_pass", latest_frame, force=True)  # 保存出牌提交失败的现场。
        self.game_submit_failures = getattr(self, "game_submit_failures", 0) + 1  # 累加本局最终提交失败次数。
        self._record_event("submit_failed", round_id=getattr(self, "current_round_id", None), stage="after_retry_before_pass")  # 记录补点后仍失败的操作现场。
        self.click_box(persistent_pass_box, after_sleep=1.0)  # 点击持续存在的不出按钮跳过本回合。
        self.turn_action_completed = True  # 锁定当前回合已经通过不出完成恢复。
        self.hand_change_pending = True  # 不出也可能触发姜维等英雄的弃牌或变点效果。
        self._record_event("action_submitted", round_id=getattr(self, "current_round_id", None), action=[], source="submit_failure_pass")  # 记录提交失败后的不出兜底。
        self.log_warning("黄色出牌按钮补点后仍未成功，已自动选择不出。")  # 记录本次兜底操作。
        return True  # 返回已执行不出兜底。

    def _click_feature(self, feature_name, after_sleep):  # 按名称查找并点击一个已标注模板。
        if not self.feature_exists(feature_name):  # 检查当前模板数据中是否存在目标名称。
            self.log_warning(f"缺少模板: {feature_name}")  # 在日志中报告缺失模板。
            return False  # 返回失败以便调用方停止依赖该动作。
        matched_box = self.find_one(feature_name, threshold=self.config.get("Template Threshold", 0.8))  # 在当前帧查找目标模板。
        if matched_box is None:  # 当前画面未匹配到目标模板。
            return False  # 返回失败并等待下一次状态轮询。
        self.click_box(matched_box, after_sleep=after_sleep)  # 点击匹配框中心并等待界面变化。
        return True  # 返回成功表示点击已经执行。

    def _click_skill_prompt_action(self, feature_name, after_sleep):  # 点击“是否使用技能”区域中的确定或取消按钮。
        if self._click_feature(feature_name, after_sleep=after_sleep):  # 现有按钮模板匹配成功时保持最精确的点击方式。
            return True  # 返回成功避免重复点击区域兜底按钮。
        if not self.feature_exists("skill_use_prompt"):  # 完整技能询问区域尚未配置时不能安全推算按钮位置。
            return False  # 保留画面等待下一轮识别而不执行盲点。
        region = self.get_box_by_name("skill_use_prompt")  # 读取用户标注的技能说明、按钮和倒计时整体区域。
        is_confirm = feature_name == "Confirm the use of skills"  # 区分右侧确定按钮和左侧取消按钮。
        center_x = region.x + region.width * (0.655 if is_confirm else 0.300)  # 使用真实界面中两个按钮的稳定水平中心。
        center_y = region.y + region.height * 0.675  # 使用真实界面按钮的稳定垂直中心。
        target = Box(int(center_x - region.width * 0.09), int(center_y - region.height * 0.17), int(region.width * 0.18), int(region.height * 0.34), name=feature_name)  # 构造覆盖完整按钮而非文字的小区域。
        self.click_box(target, after_sleep=after_sleep)  # 点击完整按钮中心并等待技能询问层关闭或进入二级交互。
        self._record_event("skill_prompt_region_fallback", hero=getattr(self, "current_hero", None), action="confirm" if is_confirm else "cancel", target={"x": target.x, "y": target.y, "width": target.width, "height": target.height})  # 记录模板漂移后的区域兜底点击供回放诊断。
        return True  # 返回成功表示已通过用户标注区域执行动作。

    def _action_ready(self, state):  # 控制相同状态的最短动作间隔。
        now = datetime.now().timestamp()  # 获取当前 Unix 时间戳。
        if self.last_action_state == state and now - self.last_action_time < 2.0:  # 判断同一状态是否仍处于冷却期。
            return False  # 冷却期内拒绝重复点击。
        self.last_action_state = state  # 记录即将执行动作的状态。
        self.last_action_time = now  # 记录动作开始时间。
        return True  # 允许当前状态执行动作。

    def _unknown_timed_out(self):  # 判断未知界面是否持续超过配置时间。
        now = datetime.now().timestamp()  # 获取当前时间用于未知状态计时。
        if self.unknown_since is None:  # 首次遇到未知界面。
            self.unknown_since = now  # 记录未知界面的开始时间。
            return False  # 首帧未知时先等待动画或加载完成。
        configured_timeout = float(self.config.get("Unknown Timeout", 45.0))  # 读取用户设置或新的四十五秒默认值。
        grace_period = float(self.config.get("Unknown Grace Period", 45.0))  # 新增键确保旧配置中的十五秒不会继续过早停止。
        return now - self.unknown_since >= max(configured_timeout, grace_period)  # 只有超过两者较大值才允许报告未知。

    def _handle_unknown_wait(self, frame):  # 在静默宽限结束后一次性记录真正持续的未知界面。
        if not self._unknown_timed_out():  # 宽限期内只等待下一帧，不保存截图也不提示未知图片。
            timeout = max(float(self.config.get("Unknown Timeout", 45.0)), float(self.config.get("Unknown Grace Period", 45.0)))  # 计算实际等待上限供状态面板显示。
            elapsed = max(0.0, datetime.now().timestamp() - self.unknown_since)  # 计算已经等待的秒数。
            self.info_set("Current State", f"等待界面稳定 {elapsed:.0f}/{timeout:.0f} 秒")  # 用正常等待文案替代立即显示 unknown。
            return False  # 通知主循环继续等待和重新识别。
        self._record_event("unknown_state", state="unknown", elapsed_seconds=datetime.now().timestamp() - self.unknown_since)  # 仅记录一次真正超时的未知状态。
        self._capture_state("unknown_timeout", frame, force=True)  # 强制保存停止前的未知现场。
        self.log_warning("未知界面持续超时，已保存现场并停止采集。", notify=True)  # 告知用户任务安全停止的原因。
        return True  # 通知主循环结束本次任务。

    def _capture_state(self, state, frame, force=False):  # 保存当前状态的去重素材和结构化元数据。
        frame_hash = compute_dhash(frame)  # 计算完整游戏画面的感知哈希。
        known_hashes = self.state_hashes.setdefault(state, [])  # 获取当前状态已经保存的哈希列表。
        max_distance = self.config.get("Duplicate Distance", 3)  # 读取用户配置的去重距离。
        if not force and is_near_duplicate(frame_hash, known_hashes, max_distance):  # 检查是否为无需重复保存的近似画面。
            return None  # 跳过重复素材以控制磁盘占用。
        known_hashes.append(frame_hash)  # 记录本次实际保存的画面哈希。
        if len(known_hashes) > 200:  # 限制每种状态的内存哈希数量。
            del known_hashes[:-200]  # 只保留最近二百张素材的哈希。
        self.sequence += 1  # 为本次素材分配递增编号。
        timestamp = datetime.now().isoformat(timespec="milliseconds")  # 生成带毫秒的可读采集时间。
        safe_time = datetime.now().strftime("%H%M%S_%f")[:-3]  # 生成适合 Windows 文件名的时间片段。
        recorder = getattr(self, "run_recorder", None)  # 读取持久运行记录器以生成逐局截图路径。
        image_path = recorder.capture_path(state, self.sequence) if recorder is not None else self.output_folder / f"{self.sequence:05d}_{state}_{safe_time}.png"  # 正常运行按牌局保存，旧测试环境保留原路径。
        trigger = "after_skill_confirm" if state == "unknown_skill_interaction" else "result_loss" if state == "result_loss" else "our_turn" if state in {"play_lead", "play_follow", "play_no_legal", "hand_selected", "ai_hand_selected", "hint_without_play_button", "play_submit_failed_before_pass"} else "in_match" if getattr(self, "in_match", False) else state  # 将采集状态转换成素材清单触发阶段。
        resource_keys = [requirement.key for requirement in requirement_for_trigger(trigger, getattr(self, "current_hero", None))]  # 查找当前英雄和阶段对应的待补素材键。
        metadata = {  # 构建用于后续自动标注的结构化元数据。
            "state": state,  # 记录分类后的游戏状态。
            "captured_at": timestamp,  # 记录素材采集时间。
            "completed_rounds": self.completed_rounds,  # 记录采集时已完成的牌局数。
            "frame_width": int(frame.shape[1]),  # 记录原始游戏画面宽度。
            "frame_height": int(frame.shape[0]),  # 记录原始游戏画面高度。
            "dhash": f"{frame_hash:016x}",  # 以十六进制保存感知哈希。
            "regions": self._collect_regions(),  # 保存已标注动态区域的固定坐标。
            "session_id": getattr(recorder, "session_id", None),  # 保存所属会话编号供素材导入器关联。
            "game_id": getattr(recorder, "game_id", None),  # 保存所属牌局编号供完整回放关联。
            "round_id": getattr(self, "current_round_id", None),  # 保存所属我方回合编号供选牌前后对比。
            "hero": getattr(self, "current_hero", None),  # 保存当前英雄供技能素材自动归类。
            "resource_keys": resource_keys,  # 保存本截图可以用于补齐的素材清单键。
        }  # 完成元数据构建。
        save_capture_files(frame, image_path, metadata)  # 同步写入 PNG 和 JSON 文件。
        self._record_event("capture", state=state, path=str(image_path), force=bool(force), round_id=getattr(self, "current_round_id", None))  # 将截图路径写入逐局事件流。
        self.info_set("Captured Images", self.sequence)  # 在 GUI 中显示已保存素材数量。
        self.info_set("Current State", state)  # 在 GUI 中显示最近识别的界面状态。
        return image_path  # 返回已保存的截图路径。

    def _collect_regions(self):  # 收集后续识牌和身份分析需要的固定区域坐标。
        region_names = [  # 定义需要写入每份元数据的标注区域。
            "Our hero",  # 保存图片中已经配置的自己英雄名称区域。
            "Underdog hero",  # 保存图片中已经配置的下家英雄名称区域。
            "The next-next level hero",  # 保存图片中已经配置的下下家英雄名称区域。
            "Deck of cards",  # 保存未选中手牌区域。
            "Selected area",  # 保存选中状态手牌区域。
            "Playing card area",  # 保存桌面上一手出牌区域。
            "skill_use_prompt",  # 保存是否发动技能的说明、取消、确定和倒计时区域。
            "skill_card_pool",  # 保存技能展示可获取牌及取消/确定按钮的独立区域。
            "opponent_left_card_count",  # 保存左侧对手剩余牌数区域。
            "opponent_right_card_count",  # 保存右侧对手剩余牌数区域。
            "Identity Mark No. 1",  # 保存第一名玩家身份区域。
            "Identity Mark No. 2",  # 保存第二名玩家身份区域。
            "Identity Mark No. 3",  # 保存第三名玩家身份区域。
        ]  # 完成区域名称列表。
        regions = {}  # 初始化区域元数据字典。
        for region_name in region_names:  # 依次读取每一个可选区域。
            if not self.feature_exists(region_name):  # 当前素材集未定义该区域时跳过。
                continue  # 继续处理下一个区域。
            try:  # 捕获个别区域无法解析的情况。
                box = self.get_box_by_name(region_name)  # 从 FeatureSet 读取区域的固定坐标。
            except ValueError:  # 处理区域尚未加载的异常。
                continue  # 跳过当前区域并继续保存其他元数据。
            regions[region_name] = {"x": box.x, "y": box.y, "width": box.width, "height": box.height}  # 保存区域矩形坐标。
        return regions  # 返回全部可用区域。
