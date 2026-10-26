"""prm_check.py —— **段分到底有没有信息量**（PRM 这条线的生死判定 ✓✓）

## 为什么先做这个 ✗✓

我们上次的 PRM 训练：**段判对率 +11 点** ✓ 但 **准确率 +0.4 点** ✗✗
→ 说明"让判分器满意每一段"与"答对题"**几乎无关** ✓
→ ★ 那么**继续调 PPO 就是白费** ✗✓ —— 因为奖励本身指错了方向 ✓

**先量相关系数** ✓：
    ρ ≥ 0.40 ✓ → 段分有信息 ✓ → 值得按论文改造（`--reward-shape paper`）✓✓
    0.15~0.40 ✓ → 弱 ✓ → β 要很小 ✓
    ρ ≈ 0    ✗✗ → **段分是噪声** ✗ → 先改判分 prompt，再量 ✓✓

## ★ 最有用的其实是第二张表 ✗✓

**"每段单独的区分度"**：
    `[Overview]` 判 1 时答对率 50% · 判 0 时答对率 50% ✗ → **这一段零信息** ✗
→ ★ 那就把它的权重设 0 ✓（而不是五段平均分配 ✗✓）
→ ★★ 这是"按段校准权重"的**测量依据** ✓✓，比拍脑袋设 β 强得多 ✓

## 用法

    export DEEPSEEK_API_KEY=sk-xxxx
    # ★ 金标准（data 里就有 ✓ 马上能跑 ✓）
    python3 prm_check.py --data data/prm_split/dev.jsonl --n 60

    # ★★ 更有价值：拿**模型自己的输出**量（需要先跑 peek/prm_train 存下来 ✓）
    python3 prm_check.py --data runs/prm_ppo_outputs.jsonl --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prm_reward as R


def _spearman(x: list[float], y: list[float]) -> float:
    """不依赖 scipy ✓（定序后算 Pearson on ranks ✓）"""
    n = len(x)
    if n < 3:
        return 0.0
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        k = 0
        while k < n:
            j = k
            while j + 1 < n and v[order[j + 1]] == v[order[k]]:
                j += 1
            avg = (k + j) / 2.0
            for t in range(k, j + 1):
                r[order[t]] = avg
            k = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prm_split/dev.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--cache", default="runs/prm_corr.jsonl")
    ap.add_argument("--gold-field", default="gold", dest="gold_field")
    a = ap.parse_args()

    if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("✗✗ 需要 DEEPSEEK_API_KEY ✓", file=sys.stderr)
        return 2
    if not os.path.exists(a.data):
        print(f"✗✗ 找不到 {a.data} ✓", file=sys.stderr)
        return 2

    from math_data import answer_math
    j = R.ApiJudge(model=a.model, cache_path=a.cache)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.n]

    xs: list[float] = []          # 段判对比例 ✓
    ys: list[int] = []            # 答对 1/0 ✓
    per: dict[str, list[tuple[int, int]]] = {h: [] for h in R.HEADS}
    skipped = 0
    for k, r in enumerate(rows):
        text = r.get("solution") or r.get("text") or ""
        gold = str(r.get(a.gold_field, ""))
        secs = R.section_contents(text)
        if not secs or not gold:
            skipped += 1
            continue
        sc = R.score_sections(j, r.get("question") or r.get("raw_question", ""), secs)
        rm = sum(sc) / max(len(sc), 1)
        # ★ 答对用**机械判据** ✓（这一步不可刷 ✓✓）
        try:
            rv = 1 if answer_math(text, gold) else 0
        except Exception:                            # noqa: BLE001
            rv = 0
        xs.append(rm)
        ys.append(rv)
        for (h, _), s in zip(secs, sc):
            if h in per:
                per[h].append((s, rv))
        if (k + 1) % 10 == 0:
            print(f"  …{k + 1}/{len(rows)} ✓", flush=True)

    if len(xs) < 5:
        print(f"✗✗ 有效样本只有 {len(xs)} 条 ✓（跳过 {skipped} ✓ 需要有 solution + gold ✓）")
        return 1

    rho = _spearman(xs, ys)
    print(f"\n  n = {len(xs)} ✓（跳过 {skipped} ✓）")
    print(f"  段分均值 **{sum(xs) / len(xs):.2f}** ✓ · 答对率 **{sum(ys) / len(ys):.1%}** ✓")
    print(f"\n  ★★ **Spearman ρ = {rho:+.3f}**  ← 段分与答对的相关性 ✓")

    print("\n  ── 每段单独的区分度 ★（判 1 时答对率 vs 判 0 时答对率 ✓）──")
    weight: dict[str, float] = {}
    for h in R.HEADS:
        v = per[h]
        if not v:
            continue
        s1 = [rv for s, rv in v if s == 1]
        s0 = [rv for s, rv in v if s == 0]
        r1 = sum(s1) / len(s1) if s1 else float("nan")
        r0 = sum(s0) / len(s0) if s0 else float("nan")
        # ★ 区分度 = 判1与判0的答对率之差 ✓（0 = 完全无信息 ✗）
        gap = (r1 - r0) if (s1 and s0) else float("nan")
        weight[h] = max(0.0, gap) if gap == gap else 0.0
        f1 = f"{r1:.0%}({len(s1)})" if s1 else "—"
        f0 = f"{r0:.0%}({len(s0)})" if s0 else "—"
        print(f"    {h:<11} 判1 → 答对 {f1:>9} · 判0 → 答对 {f0:>9} · 区分度 "
              f"{'—' if gap != gap else f'{gap:+.0%}'}")

    tot = sum(weight.values())
    if tot > 0:
        print("\n  ★ 建议的**每段权重**（按区分度归一 ✓ 零信息的段给 0 ✓✓）")
        for h, w in weight.items():
            print(f"    {h:<11} **{w / tot:.2f}** ✓")

    print()
    if rho >= 0.40:
        print("  ✅ **段分有信息** ✓✓ → 按论文改成 `rv ±1 + β·段分` ✓（`--reward-shape paper`）")
        print("     小建议：β 取 `0.75`（5 段全对合计 < 1 ✓ 翻不了正负 ✓）")
    elif rho >= 0.15:
        print("  ⚠️ **弱相关** ✓ → 可用 ✓ 但 β 要小（≤ 0.15 ✓）")
    else:
        print("  ✗✗ **段分≈噪声** ✗✗ → **不要拿它训 PPO** ✗")
        print("     → 先改判分 prompt：问「会不会答对」✓ 而不是「这段合不合格」✗")
        print("     → 改完**再跑本脚本** ✓✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
