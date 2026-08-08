"""
Reward函数相关的常量定义。
所有权重、阈值、上限统一在此维护，避免在业务逻辑代码中出现硬编码数字。
"""

# ============ 五个Reward维度的权重（总分1.0） ============
WEIGHT_ANSWER_CORRECTNESS = 0.4   # 答案正确性（F1）
WEIGHT_ANSWER_CEM = 0.2            # 答案包含（CEM，低门槛兜底分）
WEIGHT_FORMAT_COMPLIANCE = 0.2     # 格式合规
WEIGHT_SEARCH_INCENTIVE = 0.1      # 搜索激励
WEIGHT_RETRIEVAL_CORRECTNESS = 0.1# 检索正确

# ============ 答案正确性维度：F1判定阈值 ============
# F1达到该阈值以上，视为"正确"（用于统计准确率指标，不影响F1本身的连续打分）
F1_CORRECT_THRESHOLD = 0.3

# 数值型答案的近似匹配容差（相对误差，如15.3% vs 15%的容差范围）
NUMERIC_MATCH_RELATIVE_TOLERANCE = 0.05  # 5%相对误差内视为数值匹配

# ============ 搜索激励维度 ============
SEARCH_INCENTIVE_PER_SEARCH = 0.05  # 每次有效搜索的奖励
SEARCH_INCENTIVE_MAX_COUNT = 2      # 最多奖励几次搜索（超过不再加分）

# ============生成阶段终止惩罚：触发条件 ============
MAX_ROUNDS = 25  # 最大交互轮数，超过则终止并0分
MAX_TOOL_CALLS_PER_ROUND = 5# 单轮最多工具调用次数，超过则终止并0分

# ============ 无效问题检测（Invalid Detection） ============
# 对照WXG团队真实实现补充：如果模型回答里出现"无法回答/信息不足"这类关键词，
# 且该问题被标注为"应该有答案"（valid_inst=True），触发惩罚——
# 防止模型学会用"我不知道"来逃避低质量搜索带来的低分风险。
INVALID_ANSWER_PENALTY = -0.5

INVALID_ANSWER_KEYWORDS = [
    # 中文
    "无效", "问题无效", "无法回答", "没有答案", "题目有问题",
    "无效问题", "不能回答", "无答案", "题目无效", "问题有误",
    "文中未提及", "信息不足", "无法确定", "无法得知", "没有提供",
    "不能进行网络搜索", "无法访问", "不具备搜索能力", "我不知道",
    "没有找到", "未提及", "无法获知", "无法搜索",
    # 英文
    "invalid", "question is invalid", "no answer", "cannot answer",
    "unable to answer", "not answerable", "no valid answer",
    "not mentioned", "not provided", "insufficient information",
    "unable to determine", "it is not possible", "information is not available",
    "i cannot search", "i do not have access", "cannot be determined",
    "i don't know", "i do not know", "no information",
]

# ============ ReACT格式标签 ============
TAG_THINK_OPEN = "<think>"
TAG_THINK_CLOSE = "</think>"
TAG_SEARCH_OPEN = "<search>"
TAG_SEARCH_CLOSE = "</search>"
TAG_ANSWER_OPEN = "<answer>"
TAG_ANSWER_CLOSE = "</answer>"

REQUIRED_TAG_PAIRS = [
    (TAG_THINK_OPEN, TAG_THINK_CLOSE),
    (TAG_SEARCH_OPEN, TAG_SEARCH_CLOSE),
    (TAG_ANSWER_OPEN, TAG_ANSWER_CLOSE),
]
