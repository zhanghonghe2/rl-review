"""get_math_prm.py —— 用 Ray + 大模型并发构造**结构化（PRM 用）**的 MATH 解答 ✓✓

## 为什么要有这个文件 ✗

我们之前的三段式（`[Plan]/[Calculation]/[Answer]`）在 1.5B/3B 上**都是负收益** ✗
（GSM8K −7 · MATH −3.6 ~ −5.8 ✓ 且有干净的三组对照 ✓）。
但那是"**只改 prompt**"或"**转述官方解答**" ✗ —— 前者模型不熟 ✓ 后者是外部文风 ✗。

这个文件走**第三条路** ✓✓：让大模型**自己解题**、按**固定 7 段**写出来 ✓
→ 既能拿到**可判分的分段结构** ✓ 又避免"转述" ✗（GSM8K 那次 −15 点的根因 ✓）

## 输出格式（严格 7 段 ✓ 每段都可能被打分 ✓）

    [Problem]      ← ★ 在 **prompt** 里（不要求模型重复 ✗）
    [Overview]     ← 模型从这开始
    [Plan]         P1. P2. ...
    [Solution]     S1. [calc] ... -> <<expr=result>>
                   S2. [symbolic] ... -> {{expr => simplified}}
                   S3. [theorem] ...        ← 无算式的步骤
    [Check]        V1. ... -> <<...>>
    [Answer]       \\boxed{...}

## 验证（**宁缺毋滥 ✓ 但只拒绝"能证明是错的" ✓✓**）

    ✓ 5 段齐全且顺序对 ✓
    ✓ `[Plan]` ≥1 个 `P<n>.` ✓ · `[Solution]` ≥1 个 `S<n>.` ✓
    ✓ 每个 S 步骤带合法类型标签 ✓
    ✓ 所有 `<<...>>` **算得通**（数值/分数/比较链 ✓）
    ✓ `\\boxed{}` == 官方 answer ✓（用 `math_data.answer_math` ✓ sympy ✓）
    ✓ 纯 ASCII ✓（全英文 ✓）
    ✗ 判不了的（如 `{{x-1=0 => x=1; ...}}`）**放行** ✓✓ —— 不因为"我们验不了"就丢数据 ✓

## 用法

    export DEEPSEEK_API_KEY=sk-xxxx
    # 先小跑 5 条看质量 ✓
    python3 get_math_prm.py --src data/math_train.jsonl --out data/math_prm_train.jsonl --limit 5

    # 全量（7 段很长 ✓ 并发 4 路 ✓）
    python3 get_math_prm.py --src data/math_train.jsonl --out data/math_prm_train.jsonl

    # 自测（不需要网络 ✓ 不需要 key ✓ 用你给的两个例子 ✓✓）
    python3 get_math_prm.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =========================================================================== #
# 1. Prompt（严格按给定模板 ✓ 只把 {question} 换掉 ✓）
# =========================================================================== #
# ★★★ 三块拼装 ✓✓ —— 关键是**存盘的 question 不含例子** ✗
#
#   生成时（API 调用）：RULES + EXAMPLES + 收尾  ✓（例子帮模型照格式做 ✓）
#   存盘时（question） ：RULES 只有  ✓✓（训练/评测看到的就是"题目 + 规则" ✓）
#
#   若把例子也存进 question ✗ → 训练时模型看到 3000+ 字的例子 ✓
#   → 学到的是"抄例子" ✗ 而且推理时输入 Distribution 完全变样 ✗✗

RULES = """Answer the following math problem in a strict structured format.

[Problem]
{question}

Requirements:

[Overview]
Briefly describe the problem type, the goal, and the key idea.

[Plan]
List the high-level solution steps in plain English. Do not perform detailed calculations here.

[Solution]
Write numbered steps using S1, S2, S3, ...
Each step should clearly say what it does.

Use these step types when helpful:
- [setup] Define variables or translate the problem into equations.
- [derive] Derive an equation, inequality, or relation.
- [calc] Perform a numerical calculation. Use exactly one <<expression=result>>.
- [symbolic] Perform algebraic or symbolic manipulation. Use exactly one {{{{expression => simplified_expression}}}}.
- [theorem] State a theorem or rule being used.
- [case] Split into cases if necessary.

[Check]
Verify the answer by substitution, reverse calculation, or a sanity check.

[Answer]
The final line must be \\boxed{{answer}}.

Write the solution in English only.

Format rules that are strictly checked:
- Start every [Plan] line with an uppercase P and a number: P1. P2. P3. ...
- Start every [Solution] step with S1. S2. S3. ... and put exactly one bracketed
  type tag right after it, e.g. "S1. [calc] ...".
- Start every [Check] line with an uppercase V and a number: V1. V2. ...
- Write exactly one \\boxed{{...}} on the last line of [Answer], and nothing after it.
- Use plain ASCII only: no Chinese, no full-width punctuation, no unicode math symbols."""

# ★ 例子**只在生成时**加 ✓（不进 question ✓）
EXAMPLES = """
------------------------------------------------------------------------
EXAMPLE 1

[Overview]
This is a quadratic equation problem. The goal is to find all real values of x satisfying the equation. The key idea is to simplify and factor the quadratic.

[Plan]
P1. Divide the equation by 2.
P2. Factor the quadratic expression.
P3. Solve each factor.
P4. Verify the solutions in the original equation.

[Solution]
S1. [symbolic] Divide both sides by 2 -> {{{{2x^2 - 8x + 6 = 0 => x^2 - 4x + 3 = 0}}}}
S2. [symbolic] Factor the quadratic -> {{{{x^2 - 4x + 3 = (x - 1)(x - 3)}}}}
S3. [theorem] A product is zero only if at least one factor is zero.
S4. [symbolic] Solve each factor -> {{{{x - 1 = 0 => x = 1; x - 3 = 0 => x = 3}}}}

[Check]
V1. Substitute x = 1 -> <<2*1*1 - 8*1 + 6 = 0>>
V2. Substitute x = 3 -> <<2*3*3 - 8*3 + 6 = 0>>

[Answer]
\\boxed{{1, 3}}

------------------------------------------------------------------------
EXAMPLE 2

[Overview]
This is a probability problem involving two independent events. The goal is to find the probability that exactly one event occurs.

[Plan]
P1. Compute the probability that the first event occurs and the second does not.
P2. Compute the probability that the first does not occur and the second does.
P3. Add the two mutually exclusive probabilities.

[Solution]
S1. [setup] Let A be the first event with probability 1/4 and B be the second event with probability 1/3.
S2. [calc] Probability that A occurs and B does not occur -> <<(1/4)*(1-1/3)=1/6>>
S3. [calc] Probability that A does not occur and B occurs -> <<(1-1/4)*(1/3)=1/4>>
S4. [calc] Add the two probabilities -> <<1/6+1/4=5/12>>

[Check]
V1. The result is between 0 and 1 -> <<0 <= 5/12 <= 1>>

[Answer]
\\boxed{{\\frac{{5}}{{12}}}}

------------------------------------------------------------------------
Now produce the solution for the problem in the [Problem] section above.
Start your reply directly with [Overview]. Do not repeat the problem statement."""

# 生成用（含例子 ✓）· 存盘用（只规则 ✓✓）
PROMPT_TMPL = RULES + "\n" + EXAMPLES
QUESTION_TMPL = RULES

SYSTEM = ("You are a careful mathematician. Follow the required section format exactly. "
          "Write in plain ASCII English only — never use Chinese or any non-ASCII character.")


# =========================================================================== #
# 2. 解析 + 验证 ✓
# =========================================================================== #
_SECTIONS = ["[Overview]", "[Plan]", "[Solution]", "[Check]", "[Answer]"]
_STEP_TAG = re.compile(r"\[(setup|derive|calc|symbolic|theorem|case)\]")
_STEP = re.compile(r"^\s*S(\d+)\s*[.。:：]", re.M)
_PLAN = re.compile(r"^\s*P(\d+)\s*[.。:：]", re.M)
_ANN = re.compile(r"<<([^<>]*)>>")
_SYM = re.compile(r"\{\{(.*?)\}\}", re.S)
_BOX = re.compile(r"\\boxed\s*\{")


def split_sections(text: str) -> dict[str, str]:
    """按 `[Overview]` 等标记切段 ✓ 缺的段不出现在结果里 ✓"""
    pos = []
    for h in _SECTIONS:
        i = text.find(h)
        if i >= 0:
            pos.append((i, h))
    pos.sort()
    out: dict[str, str] = {}
    for k, (i, h) in enumerate(pos):
        end = pos[k + 1][0] if k + 1 < len(pos) else len(text)
        out[h] = text[i + len(h):end].strip()
    return out


def _boxed(text: str) -> str | None:
    """抽最后一个 `\\boxed{...}` ✓（处理嵌套花括号 ✓）"""
    last, start = None, 0
    while True:
        m = _BOX.search(text, start)
        if not m:
            break
        i, depth, j = m.end(), 1, m.end()
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last, start = text[i:j - 1], j
        else:
            break
    return last


# ── 算术验证 ✓（数值 → 分数 → 比较链 ✓）──────────────────────────────
def _sy(s: str):
    """解析成 sympy 表达式 ✓（支持 `4x` 这种隐式乘法 ✓）失败返回 None ✓"""
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application, parse_expr, standard_transformations)
    except ImportError:
        return None
    s = s.strip().replace("^", "**").replace("\\times", "*").replace("\\cdot", "*")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    s = s.replace("$", "").replace("−", "-").replace("×", "*").replace("÷", "/")
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    tf = standard_transformations + (implicit_multiplication_application,)
    try:
        return parse_expr(s, transformations=tf, evaluate=True)
    except Exception:                                    # noqa: BLE001
        return None


def _eq(lhs: str, rhs: str) -> bool | None:
    """两边**真的**相等吗 ✓ 判不了返回 None ✓"""
    try:
        import sympy
    except ImportError:
        return None
    # ① 先试纯浮点 ✓（最快 ✓ 覆盖大多数 ✓）
    try:
        from data import check_arith
        r = check_arith(lhs, rhs)
        if r is not None:
            return r
    except Exception:                                    # noqa: BLE001
        pass
    # ② sympy ✓（分数 / 根号 / 负号 ✓）
    pa, pb = _sy(lhs), _sy(rhs)
    if pa is None or pb is None:
        return None
    # ★★ 有**自由变量**就判不了 ✗✓ —— 例如 `<<2x^2-8x+6=0>>` 是**方程**不是恒等式 ✓
    #   `simplify(lhs-rhs)` 不为 0 ✗ → 若直接返回 False 就会**误杀好数据** ✗✗（实测踩了 ✓）
    #   → 只有**两边都是纯数值**时才敢判"对/错" ✓✓
    try:
        if getattr(pa, "free_symbols", None) or getattr(pb, "free_symbols", None):
            return None
        # ★★★ 纯数值时用**相对容差** ✗✓ —— 不能用 `simplify(a-b)==0` ✗
        #   实测误杀：`(1+0.10/4)^20 = 1.63861644029` ✓ 模型写了 11 位有效数字 ✓
        #   真值 1.6386164402907... ✓ → 差 1e-12 ✗ ≠ 0 ✗✗ → 被拒 ✗
        #   → 模型**四舍五入**是常态 ✓ 必须容忍 ✓（相对 1e-6 ✓ 够严 ✓）
        try:
            fa, fb = float(sympy.N(pa)), float(sympy.N(pb))
            return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
        except Exception:                                # noqa: BLE001
            pass
        try:
            d = sympy.simplify(pa - pb)
            return bool(d == 0)
        except Exception:                                # noqa: BLE001
            return None
    except Exception:                                    # noqa: BLE001
        return None


_CMP = re.compile(r"(<=|>=|<|>)")


def check_ann(inner: str) -> bool | None:
    """验一个 `<<...>>` ✓ 返回 None = 判不了 ✓（**放行** ✓✓）"""
    s = inner.strip()
    if not s:
        return None
    # ★ 顺序要紧 ✗：`<=` 里含 `=` ✓ 所以**先**判比较链 ✓ 否则会被拆错 ✓
    if _CMP.search(s):
        e = _sy(s)
        if e is None:
            return None
        try:
            return bool(e) if isinstance(e, bool) else bool(bool(e))
        except Exception:                                # noqa: BLE001
            return None
    # ② `expr=result` ✓ 从**最后一个** `=` 拆 ✓
    if "=" in s:
        i = s.rfind("=")
        return _eq(s[:i], s[i + 1:])
    return None


def verify(text: str, gold: str) -> tuple[bool, list[str]]:
    """★ 判这条数据能不能用 ✓ 返回 `(ok, 问题列表)` ✓"""
    bad: list[str] = []

    # ① 纯 ASCII ✓（模板要求全英文 ✓）
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii:
        uniq = "".join(dict.fromkeys(non_ascii))[:8]
        bad.append(f"含非 ASCII 字符 {uniq!r}")

    # ② 五段齐全 + 顺序 ✓
    sec = split_sections(text)
    missing = [h for h in _SECTIONS if h not in sec]
    if missing:
        bad.append(f"缺段 {missing}")
    else:
        order = [_SECTIONS.index(h) for h in sec]
        if order != sorted(order):
            bad.append("段顺序错")
        for h in _SECTIONS:
            if not sec[h].strip():
                bad.append(f"{h} 为空")

    # ③ Plan / Solution 的编号 ✓
    if "[Plan]" in sec and not _PLAN.search(sec["[Plan]"]):
        bad.append("[Plan] 没有 P1. 这类编号")
    if "[Solution]" in sec:
        sol = sec["[Solution]"]
        if not _STEP.search(sol):
            bad.append("[Solution] 没有 S1. 这类编号")
        # 每个 S 步骤都要带类型标签 ✓
        steps = re.split(r"(?=^\s*S\d+\s*[.。:：])", sol, flags=re.M)
        for st in steps:
            if _STEP.search(st) and not _STEP_TAG.search(st):
                bad.append(f"步骤缺类型标签：{st.strip()[:40]!r}")
                break
        # [calc] 必须有 <<>> ✓ · [symbolic] 必须有 {{}} ✓
        if "[calc]" in sol and "<<" not in sol:
            bad.append("[calc] 步骤没有 <<expr=result>>")
        if "[symbolic]" in sol and "{{" not in sol:
            bad.append("[symbolic] 步骤没有 {{expr => result}}")

    # ④ 所有 `<<...>>` 必须算得通 ✓（判不了的放行 ✓）
    for inner in _ANN.findall(text):
        r = check_ann(inner)
        if r is False:
            bad.append(f"算式不成立：<<{inner.strip()[:50]}>>")
            break

    # ⑤ 答案 == 官方 ✓（这是**最硬**的关 ✓✓）
    b = _boxed(text)
    if b is None:
        bad.append("[Answer] 里没有 \\boxed{}")
    else:
        try:
            from math_data import answer_math
            if not answer_math(f"#### {b}", gold):
                bad.append(f"答案不符：boxed={b[:40]!r} vs gold={gold[:40]!r}")
        except Exception:                                # noqa: BLE001
            pass                                         # 判不了 → 放行 ✓

    return (not bad), bad


# =========================================================================== #
# 3. API 一条 ✓
# =========================================================================== #
def gen_one(client, idx: int, raw: str, model: str, temperature: float,
            max_retries: int = 6, gold: str = "") -> dict | None:
    prompt = PROMPT_TMPL.format(question=raw)
    last = None
    last_text = ""
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("空回复")
            ok, why = verify(text, gold)
            if not ok:
                # ★★ 拒因必须**带回 driver** ✗✓ —— Ray worker 里的 print 会被
                #   `log_to_driver=False` **吞掉** ✗✗（实测：5/5 全拒但我们什么都看不到 ✓）
                last = "verify: " + "；".join(why[:3])
                last_text = text
                # ★ 验证不过就**重试**（换一次采样 ✓ 常常能修好 ✓）
                continue
            # ★★ 存盘的 question **不含例子** ✓✓（例子只给生成时看 ✓）
            return {"idx": idx,
                    "question": QUESTION_TMPL.format(question=raw),
                    "solution": text, "raw_question": raw, "gold": gold}
        except Exception as e:                          # noqa: BLE001
            last = e
            wait = min(60.0, (2 ** attempt) + random.random())
            print(f"  [warn] idx={idx} 第 {attempt + 1} 次失败：{type(e).__name__}: "
                  f"{str(e)[:120]} → {wait:.1f}s 后重试", flush=True)
            time.sleep(wait)
    # ★ 放弃时也把**原文**带回去 ✓（落盘到 .rejected.jsonl ✓ 我们能直接看 ✓✓）
    return {"_err": str(last)[:400], "idx": idx, "raw_question": raw,
            "_rejected_text": locals().get("last_text", "")}


# =========================================================================== #
# 4. 自测（用你给的两个例子 ✓ 不联网 ✓）
# =========================================================================== #
_EX1 = r"""[Overview]
This is a quadratic equation problem. The goal is to find all real values of x satisfying the equation. The key idea is to simplify and factor the quadratic.

[Plan]
P1. Divide the equation by 2.
P2. Factor the quadratic expression.
P3. Solve each factor.
P4. Verify the solutions in the original equation.

[Solution]
S1. [symbolic] Divide both sides by 2 -> {{2x^2 - 8x + 6 = 0 => x^2 - 4x + 3 = 0}}
S2. [symbolic] Factor the quadratic -> {{x^2 - 4x + 3 = (x - 1)(x - 3)}}
S3. [theorem] A product is zero only if at least one factor is zero.
S4. [symbolic] Solve each factor -> {{x - 1 = 0 => x = 1; x - 3 = 0 => x = 3}}

[Check]
V1. Substitute x = 1 -> <<2*1*1 - 8*1 + 6 = 0>>
V2. Substitute x = 3 -> <<2*3*3 - 8*3 + 6 = 0>>

[Answer]
\boxed{1, 3}"""

_EX2 = r"""[Overview]
This is a probability problem involving two independent events. The goal is to find the probability that exactly one event occurs.

[Plan]
P1. Compute the probability that the first event occurs and the second does not.
P2. Compute the probability that the first does not occur and the second does.
P3. Add the two mutually exclusive probabilities.

[Solution]
S1. [setup] Let A be the first event with probability 1/4 and B be the second event with probability 1/3.
S2. [calc] Probability that A occurs and B does not occur -> <<(1/4)*(1-1/3)=1/6>>
S3. [calc] Probability that A does not occur and B occurs -> <<(1-1/4)*(1/3)=1/4>>
S4. [calc] Add the two probabilities -> <<1/6+1/4=5/12>>

[Check]
V1. The result is between 0 and 1 -> <<0 <= 5/12 <= 1>>

[Answer]
\boxed{\frac{5}{12}}"""


def _self_test() -> int:
    fails = 0

    def ck(c, m):
        nonlocal fails
        print(f"  {'✓' if c else '✗✗'} {m}")
        fails += 0 if c else 1

    print("① 你给的两个例子 → 必须**都通过** ✓✓")
    ok1, why1 = verify(_EX1, "1, 3")
    ck(ok1, f"例1（二次方程）{'通过' if ok1 else '被拒 ✗ ' + str(why1)}")
    ok2, why2 = verify(_EX2, r"\frac{5}{12}")
    ck(ok2, f"例2（概率）{'通过' if ok2 else '被拒 ✗ ' + str(why2)}")

    print("\n② 分式/比较链的算术验证 ✓")
    ck(check_ann("(1/4)*(1-1/3)=1/6") is True, "<<(1/4)*(1-1/3)=1/6>> 成立 ✓")
    ck(check_ann("1/6+1/4=5/12") is True, "<<1/6+1/4=5/12>> 成立 ✓")
    ck(check_ann("0 <= 5/12 <= 1") is True, "比较链 <<0 <= 5/12 <= 1>> ✓")
    ck(check_ann("1/6+1/4=1/3") is False, "错的算式 → False ✓（要能拒 ✓）")
    ck(check_ann("2x^2 - 8x + 6 = 0") is None, "带变量的 → None（**放行** ✓✓）")

    print("\n③ 该拒的必须拒 ✓")
    ck(verify(_EX1.replace("[Check]", "[Nope]"), "1, 3")[0] is False, "缺段 → 拒 ✓")
    ck(verify(_EX1.replace("\\boxed{1, 3}", "\\boxed{1, 4}"), "1, 3")[0] is False,
       "答案不符 → 拒 ✓✓（最硬的一条 ✓）")
    ck(verify(_EX1.replace("P1.", "第一步"), "1, 3")[0] is False, "Plan 无编号 → 拒 ✓")
    ck(verify(_EX1.replace("[symbolic]", "[乱码]"), "1, 3")[0] is False, "步骤类型不合法 → 拒 ✓")
    ck(verify(_EX1 + "\n注释：这里", "1, 3")[0] is False, "非 ASCII → 拒 ✓")
    ck(verify(_EX1.replace("<<2*1*1 - 8*1 + 6 = 0>>", "<<2*1*1 - 8*1 + 6 = 9>>"), "1, 3")[0]
       is False, "算式不成立 → 拒 ✓")

    print("\n④ 段切分 ✓")
    s = split_sections(_EX1)
    ck(list(s) == _SECTIONS, f"五段都切出来了 ✓（{list(s)}）")
    ck(_boxed(_EX1) == "1, 3", f"boxed = {_boxed(_EX1)!r} ✓")
    ck(_boxed(r"nested \boxed{\frac{1}{\sqrt{2}}}") == r"\frac{1}{\sqrt{2}}", "嵌套花括号 ✓")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓ —— 验证器可以用了 ✓（例1/例2 都放行 ✓ 该拒的都拒住了 ✓）")
    return 0


# =========================================================================== #
# 5. 主流程 ✓
# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/math_train.jsonl",
                    help="题库 jsonl（要有 question + gold ✓）")
    ap.add_argument("--out", default="data/math_prm_train.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="这次最多做几条（试跑 ✓）")
    ap.add_argument("--workers", type=int, default=4, help="Ray 并发路数（默认 4 ✓）")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="★ 别太低 ✗（0.3 时同一题常写出同一种错误 ✓ 重试也修不好 ✗）")
    ap.add_argument("--flush-every", type=int, default=10)
    ap.add_argument("--verify-only", action="store_true", dest="verify_only",
                    help="只验证已有输出 ✓（不调 API ✓）")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    print(f"题库 {len(rows)} 条 ✓（{a.src}）")
    for k in ("question", "gold"):
        if rows and k not in rows[0]:
            raise SystemExit(f"✗ 题库缺字段 {k!r}（现有：{list(rows[0])}）")

    # ── 只验证 ✓ ──────────────────────────────────────────────────
    if a.verify_only:
        n = ok = 0
        for d in rows:
            n += 1
            good, why = verify(d.get("solution", ""), d.get("gold", ""))
            ok += int(good)
            if not good:
                print(f"  ✗ {d.get('raw_question', '')[:50]!r} → {why[:2]}")
        print(f"\n  {ok}/{n} 通过 ✓（{ok / max(n, 1):.1%}）")
        return 0

    # ── 断点续跑 ✓ ────────────────────────────────────────────────
    done: set[int] = set()
    done_q: set[str] = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d.get("idx"), int):
                done.add(d["idx"])
            if d.get("raw_question"):
                done_q.add(d["raw_question"])
        print(f"  已有输出 {a.out}：已完成 {len(done)} 条 ✓")

    end = len(rows) if a.end is None else min(a.end, len(rows))
    idxs = [i for i in range(max(0, a.start), end)
            if i not in done and rows[i]["question"] not in done_q]
    if a.limit is not None:
        idxs = idxs[:a.limit]
    print(f"  待生成 **{len(idxs)}** 条 ✓ · 并发 {a.workers} 路 ✓")
    if not idxs:
        print("  没有要做的 ✓")
        return 0

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("✗✗ 没有 API key ✓ 请 export DEEPSEEK_API_KEY=sk-xxxx", file=sys.stderr)
        return 2

    from openai import OpenAI
    import ray

    @ray.remote(num_cpus=0.25, max_retries=3)
    def task(i: int, q: str, g: str, okey: str, model: str, temp: float):
        c = globals().get("_CLIENT")
        if c is None or globals().get("_CLIENT_KEY") != okey:
            c = OpenAI(api_key=okey, base_url="https://api.deepseek.com")
            globals()["_CLIENT"] = c
            globals()["_CLIENT_KEY"] = okey
        return gen_one(c, i, q, model, temp, gold=g)

    if not ray.is_initialized():
        ray.init(num_cpus=max(1, a.workers), include_dashboard=False,
                 log_to_driver=False, ignore_reinit_error=True)

    futs = [task.remote(i, rows[i]["question"], rows[i].get("gold", ""),
                        key, a.model, a.temperature) for i in idxs]
    print(f"  ★ 已提交 {len(futs)} 个任务 ✓（7 段输出较长 ✓ 每条可能 20~60s ✓）")

    ok = rej = 0
    t0 = time.perf_counter()
    pend = futs
    buf: list[dict] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        with open(a.out, "a", encoding="utf-8") as f:
            for d in buf:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        buf = []

    while pend:
        ready, pend = ray.wait(pend, num_returns=min(a.flush_every, len(pend)), timeout=600)
        for f in ready:
            try:
                d = ray.get(f)
            except Exception as e:                      # noqa: BLE001
                print(f"  [error] task：{type(e).__name__}: {e}", flush=True)
                d = None
            if d is not None and "_err" not in d:
                buf.append(d)
                ok += 1
            else:
                rej += 1
                if d is not None:                      # ★ 拒因/原文都打出来 ✓✓
                    print(f"  ✗ idx={d.get('idx')} → {d.get('_err')}", flush=True)
                    with open(a.out + ".rejected.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
            el = time.perf_counter() - t0
            done_n = ok + rej
            print(f"  进度 {done_n}/{len(futs)} ✓ 成功 {ok} · 放弃 {rej} · "
                  f"{el / max(done_n, 1):.1f}s/条 · 预计剩 "
                  f"{(len(futs) - done_n) * el / max(done_n, 1) / 60:.0f} 分钟", flush=True)
        flush()

    print(f"\n完成 ✓ 写入 {a.out} · 成功 **{ok}** · 放弃 **{rej}** "
          f"（通过率 {ok / max(ok + rej, 1):.1%}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
