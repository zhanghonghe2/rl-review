"""filter_pool.py —— 把 RL 题池筛成"**2~5/8 对的题**" ✓✓（Qwen2.5-Math 论文的做法 ✓）

## 为什么必须筛 ✗✓

我们实测：RL 里 **30% 的步 `grad_norm == 0`** ✗ —— 因为那些题**组内全对或全错** ✓
→ GRPO 的优势 = `(r − 组均值) / 组标准差` ✓ → 组内全同 → **优势恒 0** ✗✗
→ ★ 那一步**完全白跑** ✓（我们实测过 8 步里 2~3 步如此 ✓）

论文（Qwen2.5-Math）的做法 ✓✓：
    **每题采 8 条** ✓ → **只留 2~5 条对的** ✓
    · < 2 对的 → 模型**根本没能力** ✗ → 学不到 ✓
    · > 5 对的 → 模型**已经会了** ✗ → 不用训 ✓
→ ★ 剩下的题**组内有对有错** ✓ → **每步都有梯度** ✓✓

## ★ 这活适合空闲卡 ✓

纯吞吐 ✓ 可断点续跑 ✓（按 idx 记进度 ✓）→ 挂几小时 ✓✓

## 用法

    # 先小跑看比例 ✓
    python3 filter_pool.py --pool data/math_train_hard.jsonl \
        --out data/pool_2to5.jsonl --limit 50 --k 8 --model /data/models/Qwen2.5-1.5B-Instruct

    # 全量（可断点续跑 ✓）
    python3 filter_pool.py --pool data/math_train_hard.jsonl \
        --out data/pool_2to5.jsonl --k 8 --max-new-tokens 1024 \
        --model /data/models/Qwen2.5-1.5B-Instruct --vllm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/math_train_hard.jsonl")
    ap.add_argument("--out", default="data/pool_2to5.jsonl")
    ap.add_argument("--model", default="/data/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--init-lora", default=None, dest="init_lora",
                    help="★ 用哪个模型采（默认基座 ✓；也可传你最好的 adapter ✓）")
    ap.add_argument("--k", type=int, default=8, help="每题采几条 ✓")
    ap.add_argument("--lo", type=int, default=2, help="至少几条对 ✓（含）")
    ap.add_argument("--hi", type=int, default=5, help="至多几条对 ✓（含）")
    ap.add_argument("--chunk", type=int, default=1, dest="chunk",
                    help="★ 每批**题数** ✓ —— ★★ **行数 = chunk × k** ✗✓"
                         "（k=8 时 chunk 8 就是 **64 行** ✗ → logits 37 GiB → 必爆 ✓）")
    ap.add_argument("--max-new-tokens", type=int, default=1024, dest="mnt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reward-mode", default="math", dest="reward_mode",
                    choices=["math", "lenient", "strict"])
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.85, dest="gpu_util")
    ap.add_argument("--vllm-kv-gb", type=float, default=8.0, dest="kv_gb")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.pool):
        print(f"✗✗ 找不到 {a.pool} ✓", file=sys.stderr)
        return 2
    import torch
    import time

    import data as D
    from model import load_tokenizer

    rows = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    print(f"题池 {len(rows)} 题 ✓（{a.pool}）")
    for k in ("question", "gold"):
        if rows and k not in rows[0]:
            raise SystemExit(f"✗ 题池缺字段 {k!r}（现有 {list(rows[0])}）")

    # ★ 断点续跑 ✓（按 idx 记 ✓ 落在 <out>.hits.jsonl ✓）
    hits_path = a.out + ".hits.jsonl"
    hits: dict[int, int] = {}
    if os.path.exists(hits_path):
        for line in open(hits_path, encoding="utf-8"):
            if line.strip():
                try:
                    d = json.loads(line)
                    hits[int(d["idx"])] = int(d["hits"])
                except Exception:                    # noqa: BLE001
                    pass
        print(f"  断点续跑：已完成 {len(hits)} 题 ✓")

    todo = [i for i in range(len(rows)) if i not in hits]
    if not todo:
        print("  全部采过了 ✓ → 直接筛 ✓")
    else:
        print(f"  待采 **{len(todo)}** 题 × {a.k} 条 ✓")
        tok = load_tokenizer(a.model)
        # ★ 数学/结构化数据：**不要 GSM8K 的 system** ✗（它会命令 `#### <number>` ✓ 与 boxed 打架）
        D.SYSTEM_OVERRIDE = ""

        # ★★★ 直接用 vLLM ✓✓ —— **不走 `rollout`** ✗
        #   原因：`rollout` 为了 PPO 会做一次前向重算 logp ✓ → `logits (B,T,V) fp32` ✗
        #        → chunk×k 一大就 37 GiB ✗✗（实测炸过 ✓）
        #   ★ 而本任务**只要文本** ✓ → 重算是纯浪费 ✓ → 去掉它 ✓✓
        #   → 每批可以放心开到 **64 题 × k 条** ✓（vLLM 内部自己 batch ✓✓）
        from vllm_backend import build_from_adapter
        vb = build_from_adapter(a.model, a.init_lora, gpu_util=a.gpu_util,
                                kv_gb=a.kv_gb, sleep_level=0, max_len=4096)
        print(f"  ★ 走 vLLM 直采 ✓（不重算 logp ✓ 每批 {a.chunk} 题 × {a.k} 条 "
              f"= {a.chunk * a.k} 行 ✓✓）")

        if a.reward_mode == "math":
            from math_data import answer_math as _judge
        elif a.reward_mode == "lenient":
            _judge = lambda t, g: D.answer_lenient(t, g)      # noqa: E731
        else:
            _judge = lambda t, g: D.answer_strict(t, g)       # noqa: E731

        done = 0
        t0 = time.perf_counter()
        for s0 in range(0, len(todo), a.chunk):
            part = [rows[i] for i in todo[s0:s0 + a.chunk]]
            # ★ `VLLMBackend.generate` 收 `list[list[int]]` ✓ 返回 `list[list[int]]` ✗（没有 n=k ✗）
            #   → 把每题的 prompt **复制 k 份** ✓ → vLLM 内部一起 batch ✓✓
            id_lists: list[list[int]] = []
            for pr in part:
                msgs = D.build_messages(pr["question"])       # ★ 和训练同一条路径 ✓✓
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             return_tensors="pt")["input_ids"][0]
                id_lists.extend([ids.tolist()] * a.k)
            outs = vb.generate(id_lists, temperature=1.0, max_tokens=int(a.mnt),
                               seed=int(a.seed) + int(todo[s0]))
            for j, pr in enumerate(part):
                got = 0
                for g in range(a.k):
                    r = j * a.k + g
                    if r >= len(outs):
                        continue
                    txt = tok.decode(outs[r], skip_special_tokens=True)
                    try:
                        got += int(bool(_judge(txt, pr["gold"])))
                    except Exception:                 # noqa: BLE001
                        pass
                idx = todo[s0 + j]
                hits[idx] = got
                with open(hits_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"idx": idx, "hits": got}) + "\n")
            done += len(part)
            kept = sum(1 for v in hits.values() if a.lo <= v <= a.hi)
            el = time.perf_counter() - t0
            eta = el / max(done, 1) * (len(todo) - done) / 60
            print(f"  {done}/{len(todo)} ✓ 已评 {len(hits)} · "
                  f"落在 [{a.lo},{a.hi}] 的 **{kept}/{len(hits)}** "
                  f"（{kept / max(len(hits), 1):.0%}）· "
                  f"**{el / max(done, 1):.2f}s/题** · 剩 ~{eta:.0f} 分钟", flush=True)
        if vb is not None:
            vb.shutdown()

    # ── 筛 + 统计 ✓ ──────────────────────────────────────────────
    keep = [i for i in range(len(rows)) if a.lo <= hits.get(i, -1) <= a.hi]
    with open(a.out, "w", encoding="utf-8") as f:
        for i in keep:
            f.write(json.dumps({**rows[i], "_hits": hits[i]}, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(hits.values())
    print(f"\n  ════ 每题 {a.k} 条里答对几条的分布 ════")
    for n in range(a.k + 1):
        bar = "#" * int(40 * c[n] / max(len(hits), 1))
        print(f"    {n}/{a.k}  {c[n]:>5} 题  {bar}")
    print(f"\n  ★ 保留 [{a.lo},{a.hi}] 对的 → **{len(keep)}/{len(rows)} = "
          f"{len(keep) / max(len(rows), 1):.1%}** ✓ → {a.out}")
    print(f"    丢掉：{c[0] + c[1]} 题太难（≤1 对 ✗）· "
          f"{sum(c[n] for n in range(a.hi + 1, a.k + 1))} 题太易（≥{a.hi + 1} 对 ✗）")
    print(f"\n  ★★ 下一步：`--pool-file {a.out}` 换掉 GRPO 的题池 ✓✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
