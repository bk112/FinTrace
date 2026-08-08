"""
Reward函数的单元测试。

用几个典型场景验证打分逻辑是否符合预期：
1. 完美轨迹（格式对、搜索了、答案对）—— 应接近满分
2. 格式错误的轨迹—— 应被一票否决，total=0
3. 不搜索直接猜答案的轨迹 —— 应该拿不到搜索激励分
4. 数值近似匹配的轨迹（如"15.3%" vs "约15%"）—— 应该有较高的答案正确性分
5. 检索到了但答案没提取对的轨迹

运行方式：python3 test_reward_functions.py
"""

from fintrace.rewards import (
    Trajectory,
    SearchStep,
    compute_total_reward,
)


def print_breakdown(name: str, breakdown):
    print(f"\n=== {name} ===")
    print(f"  answer_correctness : {breakdown.answer_correctness:.4f}")
    print(f"  answer_cem: {breakdown.answer_cem:.4f}")
    print(f"  format_compliance  : {breakdown.format_compliance:.4f}")
    print(f"  search_incentive   : {breakdown.search_incentive:.4f}")
    print(f"  retrieval_correctness: {breakdown.retrieval_correctness:.4f}")
    print(f"  total              : {breakdown.total:.4f}")
    print(f"  terminated_with_zero: {breakdown.terminated_with_zero}")


def test_case_1_perfect_trajectory():
    """场景1：完美轨迹——格式对、搜索了、答案对，应接近满分"""
    raw_text = (
        "<think>需要先查财报数据</think>"
        "<search>某公司2025年Q4营收增速</search>"
        "<think>已找到数据，可以回答</think>"
        "<answer>该公司2025年Q4营收同比增长15.3%</answer>"
    )
    traj = Trajectory(
        raw_text=raw_text,
        ground_truth="15.3%",
        search_steps=[
            SearchStep(
                query="某公司2025年Q4营收增速",
                retrieved_text="根据财报披露，该公司2025年Q4营收同比增长15.3%，主要来自新能源板块贡献",
            )
        ],
        final_answer="该公司2025年Q4营收同比增长15.3%",
        num_rounds=2,
        tool_calls_in_single_round=1,
        is_repeated_query=False,
    )
    breakdown = compute_total_reward(traj)
    print_breakdown("场景1：完美轨迹", breakdown)
    assert breakdown.total > 0.9, "完美轨迹应接近满分"
    assert not breakdown.terminated_with_zero


def test_case_2_format_error():
    """场景2：格式错误（缺少闲合标签）——应被一票否决，total=0"""
    raw_text = (
        "<think>需要先查财报数据"  # 缺少</think>闭合标签
        "<search>某公司营收</search>"
        "<answer>15.3%</answer>"
    )
    traj = Trajectory(
        raw_text=raw_text,
        ground_truth="15.3%",
        search_steps=[SearchStep(query="某公司营收", retrieved_text="15.3%相关信息")],
        final_answer="15.3%",
        num_rounds=1,
        tool_calls_in_single_round=1,
        is_repeated_query=False,
    )
    breakdown = compute_total_reward(traj)
    print_breakdown("场景2：格式错误", breakdown)
    assert breakdown.total == 0.0, "格式错误应导致total=0"
    assert breakdown.terminated_with_zero


def test_case_3_no_search_direct_guess():
    """场景3：不搜索直接猜答案——搜索激励应为0，但答案可能仍部分正确"""
    raw_text = (
        "<think>凭记忆回答</think>"
        "<search></search>"  # 空search，视为没有真正搜索
        "<answer>大概增长了15%</answer>"
    )
    traj = Trajectory(
        raw_text=raw_text,
        ground_truth="15.3%",
        search_steps=[],  # 没有真正的搜索动作
        final_answer="大概增长了15%",
        num_rounds=1,
        tool_calls_in_single_round=1,
        is_repeated_query=False,
    )
    breakdown = compute_total_reward(traj)
    print_breakdown("场景3：不搜索直接猜答案", breakdown)
    assert breakdown.search_incentive == 0.0, "没有搜索动作，搜索激励应为0"
    assert breakdown.retrieval_correctness == 0.0, "没有检索，检索正确性应为0"


def test_case_4_numeric_approximate_match():
    """场景4：数值近似匹配——"约15%"应该被判定为接近"15.3%"，拿到较高但非满分的F1"""
    raw_text = (
        "<think>查到数据了</think>"
        "<search>营收增速</search>"
        "<answer>约15%</answer>"
    )
    traj = Trajectory(
        raw_text=raw_text,
        ground_truth="15.3%",
        search_steps=[SearchStep(query="营收增速", retrieved_text="15.3%数据")],
        final_answer="约15%",
        num_rounds=1,
        tool_calls_in_single_round=1,
        is_repeated_query=False,
    )
    breakdown = compute_total_reward(traj)
    print_breakdown("场景4：数值近似匹配", breakdown)
    assert breakdown.answer_correctness > 0.0, "数值近似匹配应该拿到非零的答案正确性分数"


def test_case_5_repeated_query_terminated():
    """场景5：重复query——应触发终止惩罚，total=0"""
    raw_text = (
        "<think>再搜一次</think>"
        "<search>营收增速</search>"
        "<think>还是搜同一个</think>"
        "<search>营收增速</search>"
        "<answer>15.3%</answer>"
    )
    traj = Trajectory(
        raw_text=raw_text,
        ground_truth="15.3%",
        search_steps=[
            SearchStep(query="营收增速", retrieved_text="15.3%数据"),
            SearchStep(query="营收增速", retrieved_text="15.3%数据"),  # 重复query
        ],
        final_answer="15.3%",
        num_rounds=2,
        tool_calls_in_single_round=1,
        is_repeated_query=True,
    )
    breakdown = compute_total_reward(traj)
    print_breakdown("场景5：重复query", breakdown)
    assert breakdown.total == 0.0, "重复query应触发终止惩罚，total=0"
    assert breakdown.terminated_with_zero


def test_case_6_invalid_answer_penalty():
    """
    场景6（对照WXG真实实现新增）：模型用"信息不足/无法回答"逃避搜索失败——
    应该被判定为invalid答案，答案正确性维度拿到负分惩罚（-0.5），
    但由于最终clamp到[0,1]区间，total应该被拉到0，不会是负数。
    """
    raw_text = (
        "<think>没搜到有用信息</think>"
        "<search>某冷门指标</search>"
        "<answer>抱歉，信息不足，无法回答这个问题</answer>"
    )
    traj = Trajectory(
        raw_text=raw_text,
        ground_truth="15.3%",
        search_steps=[SearchStep(query="某冷门指标", retrieved_text="无关内容")],
        final_answer="抱歉，信息不足，无法回答这个问题",
        num_rounds=1,
        tool_calls_in_single_round=1,
        is_repeated_query=False,
        valid_inst=True,  # 这题本该有确定答案
    )
    breakdown = compute_total_reward(traj)
    print_breakdown("场景6：无效答案惩罚", breakdown)
    assert breakdown.answer_correctness < 0, "命中invalid关键词应该拿到负分惩罚"
    assert breakdown.total == 0.0, "clamp后total应为0，不应是负数"
    assert not breakdown.terminated_with_zero, "这不是终止条件触发的0分，是clamp后的0分，两者要区分清楚"


if __name__ == "__main__":
    test_case_1_perfect_trajectory()
    test_case_2_format_error()
    test_case_3_no_search_direct_guess()
    test_case_4_numeric_approximate_match()
    test_case_5_repeated_query_terminated()
    test_case_6_invalid_answer_penalty()
    print("\n所有测试用例通过！")
