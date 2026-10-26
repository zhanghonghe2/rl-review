"""len_probe.py —— **长度 vs 正确性** 诊断 ✓✓

## 要回答的问题 ✗✓

PRM 那轮跑完发现：**长度 571 → 1437（2.6×）** 而准确率只多 1.4 点 ✗
→ ★ 这是"**真的学会了更长的推理**" ✓ 还是"**灌水**" ✗？

判据（本脚本直接算）：
    · 若 **长答案的正确率明显更高** ✓ → 更长推理有效 ✓
    · 若 **长短答案正确率一样** ✗ → 长度是白写的 ✓（段分被"长度"占了便宜 ✓）

## 它做什么

    ① 采 N 题（vLLM 直采 ✓ 快 ✓）→ 每题记 长度 / 对错 / 块数 / 有无 boxed
    ② 按长度分位数**分桶** ✓ → 每桶的正确率 ✓✓
    ③ 输出：桶表 + 基座 vs 适配器的长度对比 ✓

## 用法

    # ① 基座（对照 ✓）
    python3 len_probe.py --model /data/models/Qwen2.5-Math-1.5B-Instruct \\
        --n 500 --out runs/len_base.jsonl

    # ② PRM 训过的（实验组 ✓）
    python3 len_probe.py --model /data/models/Qwen2.5-Math-1.5B-Instruct \\
        --init-lora runs/prm_paper_best --n 500 --out runs/len_prm.jsonl

    # ③ 两个文件对比 ✓
    python3 len_probe.py --compare runs/len_base.jsonl runs/len_prm.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _buckets(rows: list[dict], nb: int = 5) -> list[str]:
    """按**正确/错误分开**再按长度分桶 ✓ 输出可读表格 ✓"""
    if not rows:
        return ["  （没有数据 ✗）"]
    lens = sorted(r["len"] for r in rows)
    qs = [lens[min(len(lens) - 1, int(len(lens) * i / nb))] for i in range(1, nb)]
    out = [f"  {'长度区间':>14} {'题数':>6} {'正确率':>8} {'平均块数':>8} {'有 boxed':>9}"]
    edges = [0] + qs + [10 ** 9]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sub = [r for r in rows if lo < r["len"] <= hi]
        if not sub:
            continue
        acc = sum(r["ok"] for r in sub) / len(sub)
        blk = st.mean([r["blocks"] for r in sub])
        box = sum(r["boxed"] for r in sub) / len(sub)
        tag = f"{lo}~{hi}" if hi < 10 ** 9 else f">{lo}"
        out.append(f"  {tag:>14} {len(sub):>6} {acc:>7.1%} {blk:>8.1f} {box:>8.1%}")
    return out


def _summary(name: str, rows: list[dict]) -> list[str]:
    if not rows:
        return [f"  {name}: 没有数据 ✗"]
    L = [r["len"] for r in rows]
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    out = [f"\n  ══ {name} ══",
           f"    题数 {len(rows)} ✓ · 正确率 **{len(ok) / len(rows):.1%}** ✓",
           f"    长度 中位 **{st.median(L):.0f}** · 均值 {st.mean(L):.0f} "
           f"· p90 {sorted(L)[int(len(L) * 0.9)]} ✓"]
    if ok and bad:
        out.append(f"    ★ 对的平均长度 **{st.mean([r['len'] for r in ok]):.0f}** ✓ · "
                   f"错的 **{st.mean([r['len'] for r in bad]):.0f}** ✓")
        diff = st.mean([r["len"] for r in ok]) - st.mean([r["len"] for r in bad])
        verdict = ("★ 长的更容易对 ✓（更长推理有效）" if diff > 60
                   else "✗ 长短一样 → **长度是白写的** ✗✓" if abs(diff) <= 60
                   else "⚠️ 短的更容易对 ✗（长度在拖后腿 ✓）")
        out.append(f"    长度差 **{diff:+.0f}** → {verdict}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--init-lora", default=None, dest="init_lora")
    ap.add_argument("--test-file", default="data/math500.jsonl", dest="test_file")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=2048, dest="mnt")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--vllm-gpu-util", type=float, default=0.85, dest="gpu_util")
    ap.add_argument("--vllm-kv-gb", type=float, default=8.0, dest="kv_gb")
    ap.add_argument("--out", default=None, help="把每题结果存成 jsonl ✓（供 --compare ✓）")
    ap.add_argument("--compare", nargs=2, default=None,
                    help="★ 直接对比两个 jsonl ✓ 不重跑 ✓")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import data as D
    import prm_reward as R
    from math_data import answer_math

    # ── 对比模式（不跑 GPU ✓）────────────────────────────────────
    if a.compare:
        sets = []
        for p in a.compare:
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            sets.append((os.path.basename(p), rows))
            for l in _summary(os.path.basename(p), rows) + _buckets(rows):
                print(l)
        if len(sets) == 2:
            (n1, r1), (n2, r2) = sets
            m1 = st.median([r["len"] for r in r1])
            m2 = st.median([r["len"] for r in r2])
            print(f"\n  ★★ 长度中位：{n1} **{m1:.0f}** → {n2} **{m2:.0f}** "
                  f"（{m2 / max(m1, 1):.2f}×）✓")
            print(f"  ★★ 正确率：{sum(r['ok'] for r in r1) / max(len(r1), 1):.1%}"
                  f" → {sum(r['ok'] for r in r2) / max(len(r2), 1):.1%} ✓")
        return 0

    # ── 采样模式 ────────────────────────────────────────────────
    from model import load_tokenizer
    from vllm_backend import build_from_adapter
    D.SYSTEM_OVERRIDE = ""
    rows = [json.loads(l) for l in open(a.test_file, encoding="utf-8") if l.strip()][: a.n]
    print(f"  {len(rows)} 题 ✓（{a.test_file}）"
          f" · adapter {a.init_lora or '（基座）'} ✓")
    tok = load_tokenizer(a.model)
    vb = build_from_adapter(a.model, a.init_lora, gpu_util=a.gpu_util,
                            kv_gb=a.kv_gb, sleep_level=0, max_len=4096)
    out_rows: list[dict] = []
    for i in range(0, len(rows), a.chunk):
        part = rows[i:i + a.chunk]
        ids = []
        for pr in part:
            t = tok.apply_chat_template(D.build_messages(pr["question"]),
                                       add_generation_prompt=True,
                                       return_tensors="pt")["input_ids"][0]
            ids.append(t.tolist())
        outs = vb.generate(ids, temperature=1.0, max_tokens=int(a.mnt),
                           seed=int(a.seed) + i)
        for pr, o in zip(part, outs):
            txt = tok.decode(o, skip_special_tokens=True)
            try:
                ok = int(bool(answer_math(txt, str(pr.get("gold", "")))))
            except Exception:                       # noqa: BLE001
                ok = 0
            out_rows.append({"q": pr["question"][:120], "len": len(txt), "ok": ok,
                             "blocks": len(R.block_spans(txt)),
                             "boxed": int("\\boxed" in txt)})
        print(f"    {min(i + a.chunk, len(rows))}/{len(rows)} ✓ "
              f"长度中位 {st.median([r['len'] for r in out_rows]):.0f} ✓ "
              f"正确率 {sum(r['ok'] for r in out_rows) / len(out_rows):.1%} ✓",
              flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  已存 **{a.out}** ✓（用 --compare 对比两个 ✓）")
    for l in _summary(a.init_lora or "基座", out_rows) + _buckets(out_rows):
        print(l)
    vb.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
