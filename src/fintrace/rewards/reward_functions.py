"""
金融多跳检索Agent的Reward函数实现。

五个维度：
1. 答案正确性（F1，支持数值近似匹配）
2. 答案包含（CEM，低门槛兜底分）
3. 格式合规（一票否决）
4. 搜索激励（鼓励主动搜索）
5. 检索正确（奖励搜得准，不只是搜了）

使用方式：调用 compute_total_reward(trajectory) 得到最终标量reward，
用于GRPO训练循环中的reward_func。
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    WEIGHT_ANSWER_CORRECTNESS,
    WEIGHT_ANSWER_CEM,
    WEIGHT_FORMAT_COMPLIANCE,
    WEIGHT_RETRIEVAL_CORRECTNESS,
    NUMERIC_MATCH_RELATIVE_TOLERANCE,
    SEARCH_INCENTIVE_PER_SEARCH,
    SEARCH_INCENTIVE_MAX_COUNT,
    MAX_ROUNDS,
    MAX_TOOL_CALLS_PER_ROUND,
    INVALID_ANSWER_PENALTY,
    INVALID_ANSWER_KEYWORDS,
    TAG_THINK_OPEN,
    TAG_THINK_CLOSE,
    TAG_SEARCH_OPEN,
    TAG_SEARCH_CLOSE,
    TAG_ANSWER_OPEN,
    TAG_ANSWER_CLOSE,
)


@dataclass
class SearchStep:
    """一次搜索动作的记录：query内容 + 工具返回的检索结果文本"""
    query: str
    retrieved_text: str
    # 可选的结构化命中，用于金融数值/单位的精确判断；保留文本字段兼容现有测试和网页检索。
    retrieved_records: list[dict] = field(default_factory=list)


@dataclass
class Trajectory:
    """一条完整的ReACT轨迹，供Reward函数评估用"""
    raw_text: str                # 模型生成的完整原始文本（含所有<think>/<search>/<answer>标签）
    ground_truth: str                # 标准答案
    search_steps: list[SearchStep] = field(default_factory=list)  # 本轨迹中发生的所有搜索动作
    final_answer: Optional[str] = None# 从<answer>标签中解析出的最终答案（若解析失败为None）
    num_rounds: int = 0                    # 实际交互轮数
    tool_calls_in_single_round: int = 0    # 单轮内最多的工具调用次数（用于判定终止条件）
    is_repeated_query: bool = False        # 是否出现重复query
    valid_inst: bool = True
    # 该问题是否"应该有确定答案"（对照真实实现补充）。
    # 大多数训练数据都是valid_inst=True；如果数据构造阶段刻意合成了一批
    # "确实无法从检索结果里找到答案"的题目用于训练模型识别边界，才会设为False。


# =========================================================
# 维度1：答案正确性（F1，权重0.4）
# =========================================================

def _tokenize(text: str) -> list[str]:
    """
    极简分词：按空格和中文字符逐字切分。
    金融场景的文本以中文为主，逐字切分是常见且够用的近似方案；
    如果后续发现效果不理想，可替换为jieba等分词工具。
    """
    text = text.strip()
    tokens = []
    buffer = ""
    for ch in text:
        if ch.isspace():
            if buffer:
                tokens.append(buffer)
                buffer = ""
        elif "\u4e00" <= ch <= "\u9fff":  # 中文字符范围，逐字切分
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(ch)
        else:
            buffer += ch
    if buffer:
        tokens.append(buffer)
    return tokens


def _extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数字（支持百分号、小数），用于数值近似匹配"""
    matches = re.findall(r"-?\d+\.?\d*%?", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.rstrip("%")))
        except ValueError:
            continue
    return numbers


def _numeric_match(pred_text: str, gt_text: str) -> bool:
    """
    判断预测文本和标准答案文本中的数字是否在容差范围内匹配。
    只要预测文本中有任意一个数字命中标准答案中的任意一个数字（相对误差内），就算数值匹配成功。
    """
    return _best_numeric_match_score(pred_text, gt_text) is not None


def _best_numeric_match_score(pred_text: str, gt_text: str) -> Optional[float]:
    """
    在预测文本和标准答案文本的所有数字对之间，寻找相对误差最小、且在容差范围内的匹配，
    返回一个按误差分级的匹配质量分数（0.8~1.0之间），完全相等给1.0，容差边界附近给0.8。
    如果没有任意一对数字落在容差范围内，返回None（表示数值不匹配）。
    """
    pred_numbers = _extract_numbers(pred_text)
    gt_numbers = _extract_numbers(gt_text)
    if not pred_numbers or not gt_numbers:
        return None

    best_relative_error: Optional[float] = None
    for gt_num in gt_numbers:
        if gt_num == 0:
            continue
        for pred_num in pred_numbers:
            relative_error = abs(pred_num - gt_num) / abs(gt_num)
            if relative_error <= NUMERIC_MATCH_RELATIVE_TOLERANCE:
                if best_relative_error is None or relative_error < best_relative_error:
                    best_relative_error = relative_error

    if best_relative_error is None:
        return None

    # 误差为0（完全精确匹配）给1.0；误差越接近容差上限，分数越接近0.8
    # 线性插值：score = 1.0 - 0.2 * (relative_error / tolerance)
    score = 1.0 - 0.2 * (best_relative_error / NUMERIC_MATCH_RELATIVE_TOLERANCE)
    return score


def compute_f1(pred_text: str, gt_text: str) -> float:
    """
    计算预测答案与标准答案的F1 Score。

    金融场景的一个真实坑：标准答案往往很短（如"15.3%"只有1个token），
    而模型答案通常是完整句子（如"该公司营收同比增长15.3%"）。逐token算Precision时，
    句子里的其余词汇会被当作"多余噪音"拖累Precision，导致核心信息说对了却被判低分。

    修复方式：数值近似匹配作为独立的加成分支，与token-F1取较高值，
    而不是仅在token完全不重叠时才作为兜底触发——因为"部分重叠但Precision被拖低"
    这种情况，比"完全不重叠"更常见，也更需要被数值匹配救回来。
    """
    pred_tokens = _tokenize(pred_text)
    gt_tokens = _tokenize(gt_text)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    overlap = pred_set & gt_set

    if overlap:
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(gt_tokens)
        token_f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        token_f1 = 0.0

    # 数值近似匹配作为独立分支，不依赖token是否重叠；按误差大小分级给分，而非固定值
    numeric_score = _best_numeric_match_score(pred_text, gt_text) or 0.0

    return max(token_f1, numeric_score)


def is_invalid_answer(answer: str) -> bool:
    """
    检测答案是否属于"拒绝回答/声称信息不足"类型（对照真实实现补充的维度）。
    只做关键词命中判断，不追求精确NLU理解，够用即可。
    """
    answer_lower = answer.lower()
    return any(keyword in answer_lower for keyword in INVALID_ANSWER_KEYWORDS)


def reward_answer_correctness(trajectory: Trajectory) -> float:
    """
    维度1：答案正确性，F1 × 权重。

    新增无效答案检测（对照WXG真实实现补充）：如果答案命中"无法回答/信息不足"类关键词，
    且该问题被标注为valid_inst=True（本该有确定答案），直接返回惩罚分（负值），
    不再计算F1——防止模型学会用"我不知道"来逃避低质量搜索带来的低分风险。
    """
    if trajectory.final_answer is None:
        return 0.0

    if trajectory.valid_inst and is_invalid_answer(trajectory.final_answer):
        return INVALID_ANSWER_PENALTY

    f1 = compute_f1(trajectory.final_answer, trajectory.ground_truth)
    return f1 * WEIGHT_ANSWER_CORRECTNESS


# =========================================================
# 维度2：答案包含 CEM（权重0.2）
# =========================================================

def reward_answer_cem(trajectory: Trajectory) -> float:
    """
    维度2：CEM（Cover Exact Match）。
    只要模型答案完整包含标准答案的所有关键token，即得满分，不要求精确匹配。
    """
    if trajectory.final_answer is None:
        return 0.0

    gt_tokens = _tokenize(trajectory.ground_truth)
    pred_text = trajectory.final_answer

    for token in gt_tokens:
        if token not in pred_text:
            return 0.0  # 只要有一个关键词没被覆盖，本维度直接0分
    return WEIGHT_ANSWER_CEM


# =========================================================
# 维度3：格式合规（权重0.2，一票否决）
# =========================================================

def check_format_valid(trajectory: Trajectory) -> bool:
    """
    检查ReACT格式标签是否正确配对、闭合、内容非空。
    返回False表示格式错误，将触发整条轨迹reward归零（在compute_total_reward中处理）。

    对照真实业界实现（WXG团队qa_em_format.py的is_valid_react_sequence）核对后修正：
    1. 必须存在<think>标签（此前遗漏，是一个真实的漏洞——没有think标签说明模型完全没有
       走"先思考再行动"这个ReACT范式，应该判定为格式不合法，而不是被漏检）；
    2. 必须至少有<search>或<answer>标签之一（不强制要求一定要有answer——因为本函数评估的
       是"格式是否合法"，不是"轨迹是否已经完成"；如果一条轨迹只搜索没给答案就在compute_total_reward
       层面走"未完成"的另一条判断逻辑，不应该混在格式检查里）。
    """
    text = trajectory.raw_text

    tag_pairs = [
        (TAG_THINK_OPEN, TAG_THINK_CLOSE),
        (TAG_SEARCH_OPEN, TAG_SEARCH_CLOSE),
        (TAG_ANSWER_OPEN, TAG_ANSWER_CLOSE),
    ]

    for open_tag, close_tag in tag_pairs:
        open_count = text.count(open_tag)
        close_count = text.count(close_tag)
        if open_count != close_count:
            return False  # 标签数量不匹配，说明有未闭合的标签

    # 必须存在<think>标签
    if TAG_THINK_OPEN not in text:
        return False

    # 必须至少有一次<search>或<answer>动作
    has_search = TAG_SEARCH_OPEN in text
    has_answer_tag = TAG_ANSWER_OPEN in text
    if not has_search and not has_answer_tag:
        return False

    # 若存在<answer>标签，其内容不能为空（有标签但没写答案，视为格式不完整）
    if has_answer_tag:
        answer_matches = re.findall(
            rf"{re.escape(TAG_ANSWER_OPEN)}(.*?){re.escape(TAG_ANSWER_CLOSE)}",
            text,
            re.DOTALL,
        )
        if answer_matches and not answer_matches[-1].strip():
            return False

    return True


def reward_format_compliance(trajectory: Trajectory) -> float:
    """维度3：格式合规，通过则满分，不通过则0分（触发一票否决交由上层处理）"""
    return WEIGHT_FORMAT_COMPLIANCE if check_format_valid(trajectory) else 0.0


# =========================================================
# 维度4：搜索激励（权重0.1）
# =========================================================

def reward_search_incentive(trajectory: Trajectory) -> float:
    """
    维度4：搜索激励。
    每次有效搜索（query非空、且不是对上一次query的完全重复）加分，最多封顶两次。
    """
    valid_search_count = 0
    seen_queries = set()

    for step in trajectory.search_steps:
        query = step.query.strip()
        if not query:
            continue  # 空query不算有效搜索
        if query in seen_queries:
            continue  # 重复query不算新的有效搜索
        seen_queries.add(query)
        valid_search_count += 1

    capped_count = min(valid_search_count, SEARCH_INCENTIVE_MAX_COUNT)
    return capped_count * SEARCH_INCENTIVE_PER_SEARCH


# =========================================================
# 维度5：检索正确（权重0.1）
# =========================================================

def reward_retrieval_correctness(trajectory: Trajectory) -> float:
    """
    维度5：检索正确。
    只要任意一次搜索返回的结果文本中，命中了标准答案的关键信息（关键词覆盖或数值近似），
    即得满分。
    """
    for step in trajectory.search_steps:
        # 知识库命中优先按规范化字段判定，避免把“15.3亿元”误当作“15.3%”。
        for record in step.retrieved_records:
            value_text = str(record.get("value_text", ""))
            fact_text = str(record.get("fact", ""))
            unit = str(record.get("unit", ""))
            unit_compatible = not (
                "%" in trajectory.ground_truth and "%" not in f"{value_text}{unit}"
            )
            if unit_compatible and (
                _numeric_match(value_text, trajectory.ground_truth)
                or _numeric_match(fact_text, trajectory.ground_truth)
            ):
                return WEIGHT_RETRIEVAL_CORRECTNESS
            gt_tokens = _tokenize(trajectory.ground_truth)
            if gt_tokens and all(token in value_text or token in fact_text for token in gt_tokens):
                return WEIGHT_RETRIEVAL_CORRECTNESS
        retrieved = step.retrieved_text
        # 关键词覆盖检查
        gt_tokens = _tokenize(trajectory.ground_truth)
        covered = all(token in retrieved for token in gt_tokens) if gt_tokens else False
        if covered:
            return WEIGHT_RETRIEVAL_CORRECTNESS
        # 数值近似检查（应对答案是具体数字的情况）
        if _numeric_match(retrieved, trajectory.ground_truth):
            return WEIGHT_RETRIEVAL_CORRECTNESS
    return 0.0


# =========================================================
# 生成阶段终止惩罚：判断是否触发直接0分
# =========================================================

def should_terminate_with_zero_reward(
    trajectory: Trajectory,
    *,
    max_rounds: int = MAX_ROUNDS,
    max_tool_calls_per_round: int = MAX_TOOL_CALLS_PER_ROUND,
) -> bool:
    """
    检查是否触发"生成阶段终止惩罚"条件，触发则整条轨迹直接0分。
    对应的终止条件包括：未产出最终答案、超过最大轮数、标签解析错误、单轮工具调用过多、重复query。

    max_rounds / max_tool_calls_per_round 必须由 rollout 侧传入实际生效的配置值。
    若沿用 constants.py 的默认值而 rollout 用了更严格的 max_rounds，本判据将永远不会触发，
    超轮轨迹只能靠"没有 final_answer"间接归零——语义上等价，但无法在审计里区分终止原因。
    """
    # 过程奖励只能引导检索，不能替代任务完成；没有 answer 的轨迹不可用于优化答案策略。
    if trajectory.final_answer is None:
        return True
    if trajectory.num_rounds > max_rounds:
        return True
    if trajectory.tool_calls_in_single_round > max_tool_calls_per_round:
        return True
    if trajectory.is_repeated_query:
        return True
    if not check_format_valid(trajectory):
        return True
    return False


# =========================================================
# 汇总：计算最终总Reward
# =========================================================

@dataclass
class RewardBreakdown:
    """便于调试和日志记录的分维度打分明细"""
    answer_correctness: float
    answer_cem: float
    format_compliance: float
    search_incentive: float
    retrieval_correctness: float
    total: float
    terminated_with_zero: bool


def compute_total_reward(
    trajectory: Trajectory,
    *,
    max_rounds: int = MAX_ROUNDS,
    max_tool_calls_per_round: int = MAX_TOOL_CALLS_PER_ROUND,
) -> RewardBreakdown:
    """
    计算一条轨迹的最终总Reward，返回分维度明细（便于训练时打日志排查问题）。

    终止阈值与 rollout 的实际配置保持一致，由调用方透传。
    """
    if should_terminate_with_zero_reward(
        trajectory,
        max_rounds=max_rounds,
        max_tool_calls_per_round=max_tool_calls_per_round,
    ):
        return RewardBreakdown(
            answer_correctness=0.0,
            answer_cem=0.0,
            format_compliance=0.0,
            search_incentive=0.0,
            retrieval_correctness=0.0,
            total=0.0,
            terminated_with_zero=True,
        )

    r_correctness = reward_answer_correctness(trajectory)
    r_cem = reward_answer_cem(trajectory)
    r_format = reward_format_compliance(trajectory)
    r_search = reward_search_incentive(trajectory)
    r_retrieval = reward_retrieval_correctness(trajectory)

    total = r_correctness + r_cem + r_format + r_search + r_retrieval

    # clamp到[0, 1]区间（对照真实实现补充）。
    # 这里有个容易被忽略但很重要的效果：invalid_answer_penalty是负值（-0.5），
    # 如果不做clamp，total可能变成负数；真实实现里clamp到>=0，意味着这个惩罚
    # 实际效果是"抵消掉本来能拿到的CEM/格式分等正向分数"，而不是让total真的为负——
    # 这个细节会影响你分析训练曲线时对reward数值的解读，如果没注意到这个clamp，
    # 可能会误以为"惩罚力度不够"（因为看到的分数底线永远是0，不会更低）。
    total = max(0.0, min(1.0, total))

    return RewardBreakdown(
        answer_correctness=r_correctness,
        answer_cem=r_cem,
        format_compliance=r_format,
        search_incentive=r_search,
        retrieval_correctness=r_retrieval,
        total=total,
        terminated_with_zero=False,
    )


# =========================================================
# 从原始生成文本中解析出Trajectory结构（供训练循环调用）
# =========================================================

def parse_trajectory_from_raw_text(
    raw_text: str,
    ground_truth: str,
    tool_call_results: list[tuple[str, str]],
) -> Trajectory:
    """
    从模型生成的原始文本+ 训练循环中实际执行工具调用得到的结果，
    组装成Trajectory对象。

    Args:
        raw_text: 模型生成的完整文本（含所有标签）
        ground_truth: 标准答案
        tool_call_results: 训练循环中按顺序记录的(query, retrieved_text) 列表
    """
    search_steps = [SearchStep(query=q, retrieved_text=r) for q, r in tool_call_results]

    answer_matches = re.findall(
        rf"{re.escape(TAG_ANSWER_OPEN)}(.*?){re.escape(TAG_ANSWER_CLOSE)}",
        raw_text,
        re.DOTALL,
    )
    final_answer = answer_matches[-1].strip() if answer_matches else None

    queries = [q for q, _ in tool_call_results]
    is_repeated = len(queries) != len(set(queries))

    return Trajectory(
        raw_text=raw_text,
        ground_truth=ground_truth,
        search_steps=search_steps,
        final_answer=final_answer,
        num_rounds=len(tool_call_results),
        tool_calls_in_single_round=1,  # 若训练循环支持单轮多工具调用，需在实际解析逻辑中统计
        is_repeated_query=is_repeated,
    )
