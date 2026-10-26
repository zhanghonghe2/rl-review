"""prm_probe.py —— **小规模验证**：分段好不好 + 判分器准不准 ✓✓（10 分钟 ✓）

## 为什么先做这一步 ✗✓

我们已经在 PRM 上花了很久 ✓，但**从没验证过这两件事**✗：
    ① **分段**：`block_spans` 切出来的块，**像"一步"吗** ✗？
       → 若一条长答案被切成 **1 块** ✗ → 段分退化成 0/1 ✗✓ → PRM 等于没做 ✓
    ② **判分**：大模型说 1/0，**和人判的一致吗** ✗？
       → 若一致性 < 70% ✗ → 奖励是噪声 ✗✗

→ ★ 花 10 分钟跑这个 ✓，比再训 3 小时便宜 100 倍 ✓✓

## 它做什么

    ① 采 N 题 × K 条（Qwen2.5-Math ✓ 裸题干 ✓ 无 system ✓）
    ② 每条：切块 → 逐块判分（**遇 0 停** ✓）→ 记录
    ③ 输出**一份可读的报告** ✓ → 你只要**抽 15 个块**看一眼 ✓✓

## 用法

    export DEEPSEEK_API_KEY=sk-xxxx
    python3 prm_probe.py --n 6 --k 3 --report runs/prm_probe.md

    # 只看分段（不调 API ✓ 不花钱 ✓）
    python3 prm_probe.py --n 6 --k 3 --no-judge --report runs/prm_seg.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prm_reward as R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--pool", default="data/math500.jsonl")
    ap.add_argument("--n", type=int, default=6, help="几道题 ✓")
    ap.add_argument("--k", type=int, default=3, help="每题采几条 ✓")
    ap.add_argument("--max-new-tokens", type=int, default=2048, dest="mnt")
    ap.add_argument("--judge-model", default="deepseek-chat", dest="judge_model")
    ap.add_argument("--cache", default="runs/prm_probe_cache.jsonl")
    ap.add_argument("--report", default="runs/prm_probe.md")
    ap.add_argument("--no-judge", action="store_true", dest="no_judge",
                    help="★ 只切块不判分 ✓ 不花钱 ✓✓")
    ap.add_argument("--min-chars", type=int, default=80, dest="min_chars")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.85, dest="gpu_util")
    ap.add_argument("--vllm-kv-gb", type=float, default=8.0, dest="kv_gb")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import data as D
    from model import load_tokenizer

    if not os.path.exists(a.pool):
        print(f"✗✗ 找不到 {a.pool} ✓", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()][: a.n]
    tok = load_tokenizer(a.model)
    D.SYSTEM_OVERRIDE = ""                    # ★ 裸题干 ✓ 无 system ✓

    # ── 采样（vLLM 直采 ✓ 快 ✓）──────────────────────────────────
    from vllm_backend import build_from_adapter
    vb = build_from_adapter(a.model, None, gpu_util=a.gpu_util, kv_gb=a.kv_gb,
                            sleep_level=0, max_len=4096)
    id_lists: list[list[int]] = []
    for r in rows:
        ids = tok.apply_chat_template(D.build_messages(r["question"]),
                                     add_generation_prompt=True,
                                     return_tensors="pt")["input_ids"][0]
        id_lists.extend([ids.tolist()] * a.k)
    t0 = time.perf_counter()
    outs = vb.generate(id_lists, temperature=1.0, max_tokens=int(a.mnt), seed=a.seed)
    print(f"  采样 {len(outs)} 条 ✓ 用时 {time.perf_counter() - t0:.0f}s ✓")
    texts = [tok.decode(o, skip_special_tokens=True) for o in outs]

    # ── 切块 + 判分 ✓ ────────────────────────────────────────────
    judge = None
    if not a.no_judge:
        if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print("✗✗ --judge 需要 DEEPSEEK_API_KEY ✓（或用 --no-judge 只看分段 ✓）",
                  file=sys.stderr)
            return 2
        judge = R.ApiJudge(model=a.judge_model, cache_path=a.cache, mode="block")

    from math_data import answer_math
    lines: list[str] = ["# PRM 小规模验证 ✓\n"]
    stats = {"n": 0, "blocks": 0, "single": 0, "correct": 0, "len": []}
    per_item: list[dict] = []
    for j, pr in enumerate(rows):
        for g in range(a.k):
            r = j * a.k + g
            if r >= len(texts):
                continue
            t = texts[r]
            sp = R.block_spans(t, a.min_chars)
            stats["n"] += 1
            stats["blocks"] += len(sp)
            stats["len"].append(len(t))
            if len(sp) <= 1:
                stats["single"] += 1
            got = 0
            detail = []
            if judge is not None:
                contents = [t[x:y].strip() for x, y in sp]
                sc = R.score_sections(judge, pr["question"],
                                      [("step", c) for c in contents])
                got = sum(sc)
                stats["correct"] += got
                detail = list(zip(contents, sc))
            else:
                detail = [(t[x:y].strip(), None) for x, y in sp]
            try:
                ok = bool(answer_math(t, pr.get("gold", "")))
            except Exception:                    # noqa: BLE001
                ok = False
            per_item.append({"q": pr["question"], "text": t, "blocks": detail,
                             "ok": ok, "hits": got, "nb": len(sp)})

    # ── 写报告 ✓（人能读 ✓）──────────────────────────────────────
    n = max(stats["n"], 1)
    lines.append(f"## 总览\n")
    lines.append(f"- 样本 **{stats['n']}** 条（{a.n} 题 × {a.k} 条）")
    lines.append(f"- **平均块数 {stats['blocks'] / n:.1f}** ✓ · "
                 f"**只切成 1 块的占 {stats['single'] / n:.0%}** "
                 f"{'✗✗ 太粗 ✓' if stats['single'] / n > 0.3 else '✓ 可接受 ✓'}")
    lines.append(f"- 平均长度 {sum(stats['len']) / n:.0f} 字")
    if judge is not None:
        lines.append(f"- 被采纳的块（判 1）占比 **{stats['correct'] / max(stats['blocks'], 1):.0%}**")
        agree = sum(1 for x in per_item if (x['hits'] == x['nb']) == x['ok'])
        lines.append(f"- ★ **段全对 vs 答案对 一致率 {agree / n:.0%}** "
                     f"（越高说明段分越接近答对 ✓）")
    lines.append("\n---\n")
    for i, x in enumerate(per_item):
        lines.append(f"## 样本 {i + 1} · 答案{'✓对' if x['ok'] else '✗错'} · "
                     f"{x['nb']} 块 · 采纳 {x['hits']} ✓\n")
        lines.append(f"**题目**：{x['q'][:160]}\n")
        for k2, (c, s) in enumerate(x["blocks"]):
            mark = {1: "**1** ✓", 0: "**0** ✗", None: "—"}[s]
            lines.append(f"**块 {k2 + 1}** → {mark}")
            lines.append(f"> {c[:300]}\n")
        lines.append("---\n")
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  ════ 小结 ════")
    print(f"  平均**块数 {stats['blocks'] / n:.1f}** ✓ · "
          f"只切 1 块的 **{stats['single'] / n:.0%}** "
          f"{'✗ 太粗 ✓ 需要加超长兜底' if stats['single'] / n > 0.3 else '✓'}")
    if judge is not None:
        print(f"  判 1 的块占比 **{stats['correct'] / max(stats['blocks'], 1):.0%}** ✓")
    print(f"  报告 → **{a.report}** ✓（★ 抽 15 个块人工看一眼 ✓✓）")
    if vb is not None:
        vb.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
