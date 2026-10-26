"""make_prm_test.py —— 把测试集题干**包上 PRM 的 prompt** ✓✓

## 为什么要单独一个脚本 ✗

模型是在 `QUESTION_TMPL`（规则 ✓ **不含例子** ✗）上训练的 ✓
→ 评测时**必须用同一个模板** ✓✓ —— 否则输入分布对不上 ✓ 分数没意义 ✗✗

★ 这是我们已经踩过的坑 ✗：生成时用带例子的 prompt ✓、存盘时用不带例子的 ✓
  → 若评测时又用带例子的 ✗ → **评测分布 ≠ 训练分布** ✗✓

## 用法

    python3 make_prm_test.py --src data/math500.jsonl --out data/math500_prm.jsonl
    python3 make_prm_test.py --src data/math500.jsonl --out data/math500_prm.jsonl --show 1
"""
from __future__ import annotations

import argparse
import json
import sys


def wrap(rows: list[dict]) -> list[dict]:
    """把每条的 `question` 换成 `QUESTION_TMPL.format(...)` ✓ 其它字段原样保留 ✓"""
    from get_math_prm import QUESTION_TMPL
    out = []
    for r in rows:
        q = r["question"]
        out.append({**r, "raw_question": q,
                    "question": QUESTION_TMPL.format(question=q)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/math500.jsonl")
    ap.add_argument("--out", default="data/math500_prm.jsonl")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条的题干（检查用 ✓）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    if rows and "question" not in rows[0]:
        raise SystemExit(f"✗ {a.src} 没有 question 字段（现有 {list(rows[0])}）")
    wrapped = wrap(rows)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in wrapped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ★ 自检 ✓：答案/gold 必须原样保留 ✗ 不能因为包 prompt 而丢 ✓
    bad = sum(1 for x, y in zip(rows, wrapped) if x.get("gold") != y.get("gold"))
    lens = sorted(len(r["question"]) for r in wrapped)
    print(f"  {len(wrapped)} 条 → {a.out} ✓")
    print(f"  题干长度：中位 **{lens[len(lens) // 2]}** · 最大 {lens[-1]} ✓")
    print(f"  gold 丢失 **{bad}** 条 ✓（应为 0 ✓）")
    print(f"  ★ 用的是 `QUESTION_TMPL`（规则 ✓ **无例子** ✓）—— 和训练一致 ✓✓")

    if a.show:
        for r in wrapped[:a.show]:
            print("\n  ════ 题干预览 ════")
            print("  " + r["question"][:420].replace("\n", "\n  "))
            print("  ...")
            print(f"  gold = {r.get('gold')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
