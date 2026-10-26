"""math_data.py —— MATH-500 数据转换 + **LaTeX 等价判定** ✓✓

## 为什么需要判定器 ✗

MATH 的答案是 **LaTeX 表达式** ✗，不是数字：

    `\\left( 3, \\frac{\\pi}{2} \\right)`  ·  `\\frac{3}{4}`  ·  `\\sqrt{2}`  ·  `x=7`

→ 我们原来的 `answer_strict` / `answer_lenient` **只会抽数字** ✗ → 在 MATH 上**全是 0** ✗✗
→ 所以必须做**符号等价** ✓：`\\frac{1}{2}` == `0.5` ✓ · `\\sqrt{4}` == `2` ✓

## 两层实现 ✓

    ① 优先用 **`math_verify`**（HuggingFace 官方包 ✓ 就是为这个任务写的 ✓✓）
       pip install math-verify
    ② 没有它就退回**自己写的 sympy 归一化** ✓（覆盖常见情形 ✓ 但不是全覆盖 ✗）

## 用法

    # ① 转换 H4/MATH-500 → 我们的格式
    python3 math_data.py --from-hf --out data/math500.jsonl

    # ② 校验
    python3 math_data.py --verify data/math500.jsonl

    # ③ 自测（不需要网络 ✓ 不需要 GPU ✓）
    python3 math_data.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── ① 从模型输出里抽"候选答案" ✓（和 data.py 的 lenient 标记一致 ✓）──────────
_BOX = re.compile(r"\\boxed\s*\{")
_ANS_MARKERS = (
    r"####\s*([^\n]*)",                                         # 我们自己的方言 ✓
    r"\*\*\s*Answer\s*:?\s*\*\*\s*([^\n]*)",                     # **Answer:** ✓
    r"(?im)^\s*\**\s*Answer\s*:?\s*\**\s*([^\n]*)",              # Answer: ✓
    r"(?i)the\s+(?:final\s+)?answer\s+is\s*[:=]?\s*([^\n.]*)",   # the answer is ✓
)


def _boxed(text: str) -> str | None:
    """抽最后一个 `\\boxed{...}` ✓（要处理**嵌套花括号** ✗ —— 手写括号配对 ✓）"""
    last = None
    start = 0
    while True:
        m = _BOX.search(text, start)
        if not m:
            break
        i = m.end()                     # `{` 之后
        depth = 1
        j = i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[i:j - 1]
            start = j
        else:
            break
    return last


def extract_answer(text: str) -> str | None:
    """★ 从输出里抽答案字符串 ✓（不判等价 ✓ 只抽 ✓）"""
    b = _boxed(text)
    if b is not None and b.strip():
        return b.strip()
    for pat in _ANS_MARKERS:
        hits = re.findall(pat, text)
        for h in reversed(hits):
            h = h.strip()
            if h:
                return h
    return None


# ── ② 判定等价 ✓ ──────────────────────────────────────────────────────
def _norm_latex(s: str) -> str:
    """把 LaTeX 压成 sympy 能吃的字符串 ✓（覆盖常见情形 ✓ 不是全覆盖 ✗）"""
    s = s.strip()
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    # \frac{a}{b} → (a)/(b) ✓（反复替换 ✓ 处理 `\frac` 里的简单花括号 ✓）
    for _ in range(6):
        s2 = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\text", "").replace("\\mathrm", "").replace("\\mbox", "")
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _sympy_equal(a: str, b: str) -> bool | None:
    """自己写的兜底等价判定 ✓ 返回 None = **判不了** ✗"""
    try:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application)
    except ImportError:
        return None
    tf = standard_transformations + (implicit_multiplication_application,)

    def _p(x):
        return parse_expr(_norm_latex(x), transformations=tf, evaluate=False)

    # ① 纯数值：直接比浮点 ✓
    try:
        fa, fb = float(_norm_latex(a)), float(_norm_latex(b))
        return abs(fa - fb) < 1e-6
    except Exception:                                        # noqa: BLE001
        pass
    # ② 元组/坐标：(a,b) ✓ 逐元素 ✓
    ta = _norm_latex(a).strip("()[]").split(",")
    tb = _norm_latex(b).strip("()[]").split(",")
    if len(ta) == len(tb) >= 2:
        try:
            ok = all(abs(float(sympy.N(_p(x) - _p(y)))) < 1e-6 for x, y in zip(ta, tb))
            return ok
        except Exception:                                    # noqa: BLE001
            pass
    # ③ 符号等价：simplify(a-b) == 0 ✓
    try:
        d = sympy.simplify(_p(a) - _p(b))
        return bool(sympy.simplify(d) == 0)
    except Exception:                                        # noqa: BLE001
        return None


_MV = None


def _math_verify_ok(pred: str, gold: str) -> bool | None:
    """用官方 `math_verify` 判 ✓ 返回 None = 没装 ✗"""
    global _MV
    if _MV is None:
        try:
            from math_verify import parse, verify       # type: ignore
            _MV = (parse, verify)
        except Exception:                                # noqa: BLE001
            _MV = False
    if not _MV:
        return None
    parse, verify = _MV
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:                                    # noqa: BLE001
        return None


def answer_math(text: str, gold: str) -> bool:
    """★ MATH 的判据 ✓（抽答案 → 判等价 ✓）

    顺序：`math_verify` ✓ → 自己的 sympy 兜底 ✓ → 都判不了就当**错** ✗（保守 ✓）
    """
    pred = extract_answer(text)
    if not pred:
        return False
    r = _math_verify_ok(pred, gold)
    if r is not None:
        return r
    r2 = _sympy_equal(pred, gold)
    return bool(r2) if r2 is not None else False


# ── ③ 数据转换 ✓ ──────────────────────────────────────────────────────
def convert_hf(out: str, limit: int | None = None) -> int:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps({
                "question": r["problem"],                    # ★ 改名 ✓
                "gold": str(r["answer"]).strip(),            # ★ LaTeX ✓ 不是数字 ✗
                "solution": r.get("solution", ""),
                "subject": r.get("subject", ""),
                "level": r.get("level", 0),
                "uid": r.get("unique_id", ""),
            }, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    print(f"  写出 {n} 条 → {out} ✓")
    return 0


def convert_local(src_dir: str, out_dir: str) -> int:
    """★ 把本地的 MATH 分科文件转成我们的格式 ✓✓

    输入（你的 `ppo_llm/math_data/` ✓）：
        *_train.jsonl   7 科共 **7500** 条 ✓ —— 只有 `problem` / `solution` / `type` / `level` ✗
        *_test.jsonl    7 科共 **5000** 条 ✓ —— 同上 ✗
        math500_test.jsonl  500 条 ✓ —— 自带 `answer` ✓（H4 版本 ✓）
    输出：
        math_train.jsonl   7500 条（gold 从 `solution` 的 `\boxed{}` **抽出来** ✓）
        math_test.jsonl    5000 条（同上 ✓）
        math500.jsonl      500 条（用自带的 `answer` ✓）
    """
    import glob
    os.makedirs(out_dir, exist_ok=True)
    stats = {}

    def _one(fn: str) -> list[dict]:
        rows = []
        for line in open(fn, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            prob = (d.get("problem") or d.get("question") or "").strip()
            if not prob:
                continue
            ans = (d.get("answer") or "").strip()
            if not ans:
                ans = (_boxed(d.get("solution", "")) or "").strip()
            if not ans:
                continue                                 # 只有 2 条 ✓ 丢掉 ✓
            lv = d.get("level", "")
            m = re.search(r"(\d+)", str(lv))
            rows.append({
                "question": prob,
                "gold": ans,
                "solution": d.get("solution", ""),
                "subject": d.get("subject") or d.get("type") or "",
                "level": int(m.group(1)) if m else 0,
                "uid": d.get("unique_id", ""),
            })
        return rows

    for split in ("train", "test"):
        files = sorted(glob.glob(os.path.join(src_dir, f"*_{split}.jsonl")))
        files = [f for f in files if not os.path.basename(f).startswith("math500")]
        rows = []
        for f in files:
            got = _one(f)
            stats[os.path.basename(f)] = len(got)
            rows.extend(got)
        out = os.path.join(out_dir, f"math_{split}.jsonl")
        with open(out, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split:<6} {len(rows):>5} 条 → {out} ✓")
    m5 = os.path.join(src_dir, "math500_test.jsonl")
    if os.path.exists(m5):
        rows = _one(m5)
        out = os.path.join(out_dir, "math500.jsonl")
        with open(out, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  math500 {len(rows):>4} 条 → {out} ✓")
    print("\n  各科：" + " · ".join(f"{k.replace('.jsonl','')}={v}" for k, v in stats.items()))
    return 0


def make_staged(src: str, out: str, style: str = "three") -> int:
    """★ 把题目**包上三段式要求** ✓✓ —— 测"结构对难题有没有用" ✓

    为什么要它 ✗：我们只知道"三段式在 **GSM8K** 上有害 −40 点" ✓
                   但 GSM8K 是**简单题** ✗ → 理论上**难题**受益更大 ✓✓
                   → 而我们从没在 MATH 上测过 ✗ → 这个函数就是那个实验 ✓

    做法：**只改 question** ✓（加一段格式要求 ✓ 题目本体不动 ✓）
          答案仍用 `\boxed{}` ✓（判据 `--reward-mode math` 认它 ✓✓）
    """
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    if style == "three":
        pre = ("Solve the problem in a strict three-stage format. Write plain ASCII only.\n\n"
               "[Plan]\nExplain the solution steps in plain English (no arithmetic).\n\n"
               "[Calculation]\nOne line per step, shaped: "
               "<plain-English step> -> <<expression=result>>\n\n"
               "[Answer]\nWrite the final answer as: \\boxed{...}\n\n"
               "Problem:\n")
    elif style == "plan":                      # 只加 Plan ✓（隔离"计划"这一项 ✓）
        pre = ("First write a short [Plan] in plain English, then solve the problem.\n\n"
               "Problem:\n")
    else:                                      # 只要求打框 ✓（对照 ✓）
        pre = ("Solve the problem. Write the final answer as \\boxed{...}.\n\n"
               "Problem:\n")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "question": pre + r["question"]},
                               ensure_ascii=False) + "\n")
    print(f"  {len(rows)} 条 → {out}（style={style} ✓）")
    return 0


def verify_file(path: str) -> int:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bad = [r for r in rows if not r.get("question") or not r.get("gold")]
    print(f"  {len(rows)} 条 · 缺字段 {len(bad)}")
    for r in rows[:3]:
        print(f"    level={r.get('level')} subject={r.get('subject')!r} "
              f"gold={r['gold']!r}")
    print("  ★ 判定器自测：'\\\\frac{1}{2}' vs '0.5' →",
          answer_math("#### \\frac{1}{2}", "0.5"))
    print("            '(3, pi/2)' vs '\\\\left( 3, \\\\frac{\\\\pi}{2} \\\\right)' →",
          answer_math("#### (3, pi/2)", "\\left( 3, \\frac{\\pi}{2} \\right)"))
    return 0 if not bad else 1


def _self_test() -> int:
    fails = 0

    def ck(c, m):
        nonlocal fails
        print(f"  {'✓' if c else '✗✗'} {m}")
        fails += 0 if c else 1

    print("① 抽 boxed（含嵌套花括号）")
    ck(_boxed(r"so \boxed{\frac{3}{4}}") == r"\frac{3}{4}", "简单")
    ck(_boxed(r"x \boxed{\frac{1}{\sqrt{2}}} ok") == r"\frac{1}{\sqrt{2}}", "嵌套 ✓")
    ck(_boxed(r"a \boxed{1} b \boxed{2}") == "2", "取最后一个 ✓")
    ck(_boxed("no box") is None, "没有 → None ✓")

    print("\n② extract_answer 认多种")
    ck(extract_answer(r"...\boxed{7}") == "7", "boxed ✓")
    ck(extract_answer("#### 7") == "7", "#### ✓")
    ck(extract_answer("**Answer:** 7") == "7", "**Answer:** ✓")
    ck(extract_answer("The answer is 7.") == "7", "the answer is ✓（句号被吃掉 ✓ 更好 ✓）")
    ck(extract_answer("nothing here") is None, "都没有 → None ✓")

    print("\n③ 等价判定（sympy 兜底 ✓ 不依赖 math_verify ✓）")
    ck(_sympy_equal(r"\frac{1}{2}", "0.5") is True, "1/2 == 0.5 ✓")
    ck(_sympy_equal("2", "2.0") is True, "2 == 2.0 ✓")
    ck(_sympy_equal(r"\sqrt{4}", "2") is True, "sqrt(4) == 2 ✓")
    ck(_sympy_equal(r"\frac{3}{4}", "0.75") is True, "3/4 == 0.75 ✓")
    ck(_sympy_equal("7", "8") is False, "7 != 8 ✓")
    ck(_sympy_equal(r"\left( 3, \frac{\pi}{2} \right)", "(3, pi/2)") is True,
       "坐标对 ✓")
    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-hf", action="store_true", dest="from_hf")
    ap.add_argument("--from-local", default=None, dest="from_local",
                    help="本地分科目录（如 ppo_llm/math_data ✓）")
    ap.add_argument("--out-dir", default="data/", dest="out_dir")
    ap.add_argument("--out", default="data/math500.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--make-staged", default=None, dest="make_staged",
                    help="把题目包上三段式要求 ✓（输出用 --out ✓）")
    ap.add_argument("--style", default="three", choices=["three", "plan", "box"])
    ap.add_argument("--verify", default=None)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.make_staged:
        return make_staged(a.make_staged, a.out, a.style)
    if a.verify:
        return verify_file(a.verify)
    if a.from_local:
        return convert_local(a.from_local, a.out_dir)
    if a.from_hf:
        return convert_hf(a.out, a.limit)
    ap.error("需要 --from-hf / --verify / --self-test 之一")


if __name__ == "__main__":
    sys.exit(main())
