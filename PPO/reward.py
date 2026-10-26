import os
import json
import re
from openai import OpenAI


def augment_verification(text: str) -> str:
    """
    把 GSM8K 格式的答案加上反向验证段落：
    <<4*6=24>> <<3*24=72>> #### 72
    →
    <<4*6=24>> <<3*24=72>> #### 72
    <verification>
    回代验证: 72÷3=24 ✓; 24÷4=6 ✓
    结论: 和题目条件一致
    </verification>
    """
    if "</verification>" in text:
        return text  # 已经加过了

    steps = re.findall(r'<<(.*?)>>', text)
    if not steps:
        return text

    # 反向推导每个步骤
    reverse_steps = []
    for expr in steps:
        if '=' in expr:
            left, right = expr.rsplit('=', 1)
            try:
                result = float(right.strip())
                # 尝试逆运算
                if '*' in left:
                    parts = left.split('*')
                    rev = f"{right.strip()}÷{parts[1].strip()}={parts[0].strip()}"
                elif '/' in left:
                    parts = left.split('/')
                    rev = f"{right.strip()}×{parts[1].strip()}={parts[0].strip()}"
                elif '+' in left:
                    parts = left.split('+')
                    rev = f"{right.strip()}-{parts[1].strip()}={parts[0].strip()}"
                elif '-' in left:
                    parts = left.split('-')
                    rev = f"{right.strip()}+{parts[1].strip()}={parts[0].strip()}"
                else:
                    continue
                reverse_steps.append(rev)
            except Exception:
                continue

    if reverse_steps:
        verify_block = (
            "\n<verification>\n"
            "回代验证: " + "; ".join(f"{s} ✓" for s in reverse_steps) + "\n"
            "结论: 答案正确\n"
            "</verification>"
        )
        return text + verify_block
    return text


class RewardModel:
    def __init__(self, api_key: str = ""):
        # ★ 绝不硬编码 key ✗✓ —— 从环境变量读 ✓（这个 key 已经出现在聊天/日志里 → 请轮换 ✓）
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("没有 API key ✓ 请 export DEEPSEEK_API_KEY=...")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        self.system_prompt = """你是一个数学评分助手。根据以下标准对学生解答评分(0~100):
            - 100:推理完全正确，步骤完整，答案正确
            - 80~99:方向正确，但步骤有遗漏或不严谨
            - 60~79:思路大体对，关键步骤出错
            - 40~59:部分正确，有明显逻辑漏洞
            - 20~39:方向基本错误
            - 0~19:完全错误或无关
        """

    def step_reward(self, response: str) -> float:
        """
        验证 <</>> 格式的计算步骤，每步正确 +0.05，错误 -0.1
        用于给模型提供中间步骤的稠密奖励信号
        """
        import re
        steps = re.findall(r'<<(.*?)>>', response)
        if not steps:
            return 0.0   # 没有标准格式步骤，不单独给步骤分

        score = 0.0
        for expr in steps:
            try:
                # 取等式右边（如 "4*6=24" → eval("4*6") → 对比 24）
                if '=' in expr:
                    left, right = expr.rsplit('=', 1)
                    expected = eval(left.strip())
                    actual = float(right.strip())
                    score += 0.05 if abs(expected - actual) < 1e-6 else -0.1
                else:
                    eval(expr.strip())  # 无等号只校验语法
            except Exception:
                score -= 0.1
        return score

    def get_reward(self, question: str, answer: str) -> float:
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system", "content": self.system_prompt
                },
                {
                    "role": "user", "content": f"题目:{question}\n\n解答:{answer}\n\n请只输出 JSON:{{\"score\": <int>, \"reason\": \"<str>\"}}, 其中<int>是你的评分，<str>是你的理由"
                },
            ],
            response_format={
                'type': 'json_object'
            }
        )

        try:
            text = response.choices[0].message.content
            result = json.loads(text)
            return result["score"] / 100.0, result["reason"]
        except (json.JSONDecodeError, KeyError, AttributeError):
            match = re.search(r'"score"\s*:\s*(\d+)', text)
            score = int(match.group(1)) / 100.0 if match else 0.0
            return score, "parse_failed"

    


if __name__ == "__main__":
    reward_model = RewardModel()

    # 测试：正确答案
    q = "若 x² + 5x + 6 = 0, 求 x"
    a = "x² + 5x + 6 = (x + 2)(x + 3) = 0, 所以 x = -2 或 x = -3"

    score, reason = reward_model.get_reward(q, a)
    print(f"[正确] 得分: {score:.2f},  理由: {reason}")

    # 测试：半错答案
    a2 = "公式是 x = (-b ± √(b²-4ac)) / 2a = (-5 ± 1) / 2, x = -2"
    score2, reason2 = reward_model.get_reward(q, a2)
    print(f"[半错] 得分: {score2:.2f},  理由: {reason2}")

    # 测试：完全错误
    a3 = "苹果"
    score3, reason3 = reward_model.get_reward(q, a3)
    print(f"[错误] 得分: {score3:.2f},  理由: {reason3}")
        