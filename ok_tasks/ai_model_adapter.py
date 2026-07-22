import importlib.util  # 动态加载用户提供的训练模型适配脚本。
from collections import Counter  # 统计模型动作与手牌中各点数的数量。
from pathlib import Path  # 安全解析模型适配脚本和权重文件路径。


CARD_ORDER = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "X", "D")  # 定义斗地主点数顺序，小王和大王分别使用 X、D。
CARD_ALIASES = {"10": "T", "0": "T", "小王": "X", "大王": "D", "SJ": "X", "BJ": "D"}  # 兼容 OCR 和常见模型使用的点数别名。


class AiModelError(RuntimeError):  # 定义可由自动化任务安全捕获的模型错误。
    pass  # 不增加额外状态，仅保留明确异常类型。


def normalize_card(value):  # 将 OCR 或模型返回的牌面统一成内部点数。
    text = str(value).strip().upper()  # 去除空白并统一英文字母大小写。
    text = CARD_ALIASES.get(text, text)  # 将十、小王和大王等别名转换成内部表示。
    return text if text in CARD_ORDER or text == "W" else None  # 标准十五点数外允许技能生成的万能牌实体编码。


def validate_action(action, hand_cards):  # 校验训练模型动作是否能够由当前手牌组成。
    if action is None:  # 模型返回空值通常表示推理失败而不是不出。
        raise AiModelError("AI 模型没有返回动作")  # 阻止把推理失败误当成合法的不出。
    if isinstance(action, str):  # 支持模型以空格或逗号分隔的字符串返回动作。
        action = action.replace(",", " ").split() if " " in action or "," in action else list(action)  # 将字符串动作拆分成单张牌。
    normalized = []  # 创建规范化后的动作牌组。
    for card in action:  # 逐张校验模型返回的点数。
        normalized_card = normalize_card(card)  # 转换当前点数的别名。
        if normalized_card is None:  # 检查模型是否返回了未知点数。
            raise AiModelError(f"AI 模型返回未知牌面: {card}")  # 报告具体的非法模型输出。
        normalized.append(normalized_card)  # 保存通过校验的内部点数。
    hand_counter = Counter(normalize_card(card) for card in hand_cards)  # 统计当前手牌中每种点数的数量。
    action_counter = Counter(normalized)  # 统计模型准备打出的每种点数数量。
    unavailable = {card: count for card, count in action_counter.items() if count > hand_counter[card]}  # 找出模型要求但手牌不足的点数。
    if unavailable:  # 当前动作无法由真实手牌组成时拒绝执行。
        raise AiModelError(f"AI 动作不在当前手牌中: {unavailable}")  # 避免错误点击相邻卡牌。
    return normalized  # 返回可以安全映射到屏幕点击的动作。


class TrainedModelAdapter:  # 封装用户训练模型的加载、推理和输出校验。
    def __init__(self, adapter_path, weights_path=None):  # 无外部权重的规则适配器可只提供脚本路径。
        self.adapter_path = Path(adapter_path).expanduser().resolve()  # 解析模型适配脚本的绝对路径。
        self.weights_path = Path(weights_path).expanduser().resolve() if weights_path else None  # 在配置非空时解析权重绝对路径。
        self.module = None  # 初始化尚未加载的适配模块。
        self.model = None  # 初始化尚未加载的训练模型对象。

    def load(self):  # 动态导入适配脚本并加载训练权重。
        if not self.adapter_path.is_file():  # 检查配置中的模型适配脚本是否存在。
            raise AiModelError(f"找不到 AI 适配脚本: {self.adapter_path}")  # 给出可直接定位的缺失路径。
        if self.weights_path and not self.weights_path.is_file():  # 检查配置中的训练权重是否存在。
            raise AiModelError(f"找不到 AI 权重: {self.weights_path}")  # 阻止无权重状态下进入牌局。
        spec = importlib.util.spec_from_file_location("sgbjp_user_ai_model", self.adapter_path)  # 创建隔离的动态模块说明。
        if spec is None or spec.loader is None:  # 检查 Python 是否能够加载该脚本。
            raise AiModelError(f"无法加载 AI 适配脚本: {self.adapter_path}")  # 报告无效脚本格式。
        module = importlib.util.module_from_spec(spec)  # 根据模块说明创建适配模块实例。
        spec.loader.exec_module(module)  # 执行用户适配脚本以注册加载和推理函数。
        load_model = getattr(module, "load_model", None)  # 查找约定的权重加载函数。
        predict = getattr(module, "predict", None)  # 查找约定的动作推理函数。
        if not callable(load_model) or not callable(predict):  # 校验适配脚本是否实现完整接口。
            raise AiModelError("AI 适配脚本必须实现 load_model(weights_path) 和 predict(model, state)")  # 明确说明缺失的接口。
        self.module = module  # 保存已经通过接口检查的适配模块。
        self.model = load_model(str(self.weights_path) if self.weights_path else "")  # 调用适配脚本加载实际训练模型。
        return self  # 返回适配器自身以便任务链式初始化。

    def predict(self, state):  # 使用训练模型根据完整可见状态选择动作。
        if self.module is None:  # 防止调用方在加载模型前进行推理。
            raise AiModelError("AI 模型尚未加载")  # 提示正确的初始化顺序。
        action = self.module.predict(self.model, state)  # 将结构化牌局状态交给训练模型适配脚本。
        return validate_action(action, state.get("hand_cards", []))  # 严格校验并返回可点击的动作牌组。

    def record_game(self, won, submit_failures=0):  # 将完整结算交给支持小流量统计的模型适配器。
        if self.module is None:  # 模型没有成功加载时没有可记录的候选版本。
            return None  # 返回空值保持规则适配器兼容。
        recorder = getattr(self.module, "record_game", None)  # 查询可选的候选模型结算回调。
        if callable(recorder):  # 只有高容量模型实现该回调时才写入小流量统计。
            return recorder(self.model, won, submit_failures)  # 传递胜负和本局提交失败次数。
        return None  # 普通规则适配器无需执行任何动作。
