"""test_judge.py —— 判分器校准 ✓✓

两件事：
  ① **金标准校准**（默认）：拿训练数据里的金标准分段去判 ✓
     通过率 ≥ 85% → 判分器可信 ✓ → 训练里 14% 就是"模型确实弱" ✓
     通过率 < 60% → ★ 判分器有问题 ✗ → 别拿它训练 ✗✗
  ② **单元直测**（`--unit`）：4 个手写用例，应当 4/4 ✓

## 为什么必须做 ① ✗✓

训练里 `[Overview]` 几乎全判 0 ✓，但分不清：
   ① 判分器太严 ✗ → 好段也判 0 ✓ → 训练信号是垃圾 ✓✗
   ② 模型确实写得差 ✓ → 判 0 是对的 ✓✓
→ 用**金标准**（大模型写的、已过验证器 ✓）去判，就能分开 ✓✓

## 用法

    export DEEPSEEK_API_KEY=sk-xxxx
    python3 test_judge.py                      # 金标准校准，10 条
    python3 test_judge.py --n 30               # 更多条 ✓
    python3 test_judge.py --unit               # 4 个手写用例 ✓
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prm_reward as R

_UNIT = [
    ("[Overview]", "This is an arithmetic problem. The goal is to compute 1+1. "
                   "The key idea is addition.", 1),
    ("[Plan]", "P1. Add one and one.", 1),
    ("[Solution]", "S1. [calc] Add -> <<1+1=2>>", 1),
    ("[Solution]", "S1. [calc] Add -> <<1+1=3>>", 0),
]


def _unit(model: str) -> int:
    j = R.ApiJudge(model=model, cache_path="/tmp/judge_unit.jsonl")
    ok = 0
    for head, content, want in _UNIT:
        got = j("What is 1+1?", head, "", content)
        good = got == want
        ok += good
        print(f"  {'✓' if good else '✗✗'} {head:<11} → {got}（应 {want}）{content[:44]}")
    print(f"\n  {ok}/{len(_UNIT)} 正确 ✓")
    return 0 if ok == len(_UNIT) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prm_split/dev.jsonl")
    ap.add_argument("--n", type=int, default=10, help="判几条金标准 ✓")
    ap.add_argument("--model", default="deepseek-chat",
                    help="★ 默认 deepseek-chat ✓（非推理模型 ✓ 判 0/1 更稳更便宜 ✓）")
    ap.add_argument("--cache", default="runs/prm_judge_calib.jsonl", dest="cache")
    ap.add_argument("--unit", action="store_true", help="只跑 4 个手写用例 ✓")
    a = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("✗✗ 需要 DEEPSEEK_API_KEY ✓", file=sys.stderr)
        return 2
    if a.unit:
        return _unit(a.model)
    if not os.path.exists(a.data):
        print(f"✗✗ 找不到 {a.data} ✓（先跑 split_prm.py ✓）", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.n]
    j = R.ApiJudge(model=a.model, cache_path=a.cache)

    per_head: dict[str, list[int]] = {h: [] for h in R.HEADS}
    tot = ok = 0
    for k, r in enumerate(rows):
        secs = R.section_contents(r.get("solution", ""))
        if not secs:
            print(f"  ⚠️ 第 {k + 1} 条没解析到段 ✗（solution 里没段名 ✓）→ 跳过")
            continue
        scores = R.score_sections(j, r.get("question", ""), secs)
        for (h, _), s in zip(secs, scores):
            if h in per_head:
                per_head[h].append(s)
            tot += 1
            ok += s
        print(f"  第 {k + 1} 条 ✓ {len(secs)} 段 → {scores} ✓ → "
              f"{(r.get('raw_question') or r.get('question', ''))[:44]!r}")

    print("\n  ════ 金标准分段的通过率 ════")
    for h in R.HEADS:
        v = per_head[h]
        if v:
            print(f"    {h:<11} {sum(v)}/{len(v)} = **{sum(v) / len(v):.0%}** ✓")
    rate = ok / max(tot, 1)
    print(f"\n  **总通过率 {ok}/{tot} = {rate:.1%}** ✓\n")

    if rate >= 0.85:
        print("  ✅ **判分器可信** ✓✓ → 训练里 14% 是**模型确实弱** ✓")
        print("     → 可以继续 RL ✓（但先想想：0.254 的起点太低 ✗ 修 SFT 更划算 ✓）")
    elif rate >= 0.60:
        print("  ⚠️ **判分器偏严** ✗ → 信号被压低 ✓ 但仍可用 ✓")
        print("     → 建议 `--bonus 0.5` ✓ 或换更强判分模型 ✓")
    else:
        print("  ✗✗ **判分器不可信** ✗✗ → 金标准都判不过 ✓ → **不要拿它训练** ✗✗")
        print("     → 先修判分 prompt / 换模型 ✓ 再重跑本脚本 ✓✓")
    return 0 if rate >= 0.60 else 1


if __name__ == "__main__":
    sys.exit(main())
