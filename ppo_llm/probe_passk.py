"""probe_passk.py —— 在决定还要不要继续 RL 之前，先量出可利用的差距。

判据：
    pass@k >> pass@1   → 模型会做、只是不稳定 → RL 有空间（把概率质量移到对的路径上）
    pass@k ≈  pass@1   → 模型根本不会 → RL 没得学 → 换更大模型或放弃 RL

⚠️ 必须在 temperature=1.0 下测，那才是 RL 采样的分布。
   贪心评估量的是最可能的那条，和 RL 优化的不是同一个东西。

用法：
    python3 probe_passk.py --adapter /data/ppo/runs/sft_15b --problems 128 --k 8
    python3 probe_passk.py --problems 128 --k 8            # 不加载 adapter（基座）
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from math import comb

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from config import PPOConfig
from batch import gen_start
from model import MODEL_PATH, LMWithValueHead, load_tokenizer, rollout


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏 pass@k 估计量（Codex 那篇用的那个）。

    n = 采了几次，c = 其中对了几次，k = 要估的 k。
    注意：**不是**「c >= k」——那是"至少对 k 次"，完全不同的量。
    pass@k 的意思是「给 k 次机会，至少一次对」。
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _judge(text: str, gold: str, mode: str) -> int:
    """★ 判据 ✓ —— pass@k 的 **hit** 用它 ✓（headline 指标 ✓）"""
    if mode == "math":
        from math_data import answer_math
        return int(answer_math(text, gold))
    if mode == "lenient":
        return int(D.answer_lenient(text, gold))
    return int(D.answer_strict(text, gold))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--problems", type=int, default=128)
    ap.add_argument("--test-file", type=str, default=None, dest="test_file",
                    help="★ 自定义测试集 jsonl ✓（question + gold 或含 #### 的 answer ✓）"
                         " 默认用官方 test ✓")
    ap.add_argument("--reward-mode", type=str, default="strict", dest="reward_mode",
                    choices=["strict", "lenient", "math"],
                    help="判据：strict 要 #### N ✓ · lenient 认多种方言 ✓ · "
                         "**math** 用 sympy 判 LaTeX 等价 ✓（MATH 类**必须**用它 ✗）")
    ap.add_argument("--shard", type=str, default=None,
                    help="★★ 多卡切分 ✓ 形如 `0/2` 或 `1/2` —— "
                         "第 i/n 份题 ✓（题池按**下标取模**切 ✓ 不改题目 ✓✓）。"
                         " 配合 `CUDA_VISIBLE_DEVICES=0` / `=1` 两个进程 ✓")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64,
                    help="★ 分组大小：每次 rollout/forward 多少题 ✓ —— 它同时决定 "
                         "① 一次喂 vLLM 多少条 ✓ ② **训练器重算的行数** ✗（显存 ∝ 行数 ✗）。"
                         "调大省时间 ✓；≤16 安全 ✓ · 32 需宽裕 ✓ · 128 会 OOM ✗✗")
    ap.add_argument("--vllm", action="store_true", help="用 vLLM 生成（快 ~2× ✓）")
    ap.add_argument("--greedy", action="store_true",
                    help="★ 额外跑一遍贪心 ✓（默认关 ✗ 多花一倍时间 ✓ 且它两边都不动 ✗）")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--vllm-kv-gb", type=float, default=1.5)
    ap.add_argument("--vllm-sleep-level", type=int, default=1, dest="vllm_sleep_level",
                    help="生成后 sleep 释放（0 = 不释放 ✓ 48G 上建议 0 ✓ "
                         "—— 实测这版 level 1 只释放 1G ✗ 反而留碎片 ✓）")
    ap.add_argument("--max-new-tokens", type=int, default=192, dest="mnt",
                    help="必须和 main.py 的评估一致，否则不可比")
    args = ap.parse_args()

    dev = PPOConfig.auto_device()
    print("设备 " + str(dev) + " | 模型 " + args.model.split("/")[-1]
          + " | adapter " + str(args.adapter))
    tok = load_tokenizer(args.model)
    model = LMWithValueHead(args.model, device=dev, lora=True, init_lora=args.adapter)

    test = (D.load_test_file(args.test_file) if args.test_file
            else D.load_gsm8k("test"))[: args.problems]
    # ★★ `--shard i/n` ✓✓ —— 按**下标取模**切 ✓（不是取前一半 ✗）
    #   取前一半会让两张卡的题目难度不均衡 ✗ → 汇总后的 pass@8 会有偏差 ✗✓
    if args.shard:
        try:
            _i, _n = (int(x) for x in args.shard.split("/"))
        except Exception:                            # noqa: BLE001
            raise SystemExit("✗ --shard 格式应为 i/n ✓ 例如 0/2")
        if not (0 <= _i < _n):
            raise SystemExit(f"✗ --shard {args.shard} 不合法 ✗（要 0 <= i < n ✓）")
        _before = len(test)
        test = [t for k, t in enumerate(test) if k % _n == _i]
        print(f"  ★ 分片 {_i}/{_n} ✓：{_before} 题 → **{len(test)}** 题 ✓"
              f"（按下标取模 ✓ 难度均衡 ✓✓）")
    n = len(test)

    # 只设**真实存在**的字段 ✗（我第一次写了 top_k / repetition_penalty，PPOConfig 里没有 ✓）
    # top_p / top_k / repetition_penalty 是在 rollout 内部**硬编码覆盖**的 ✓
    # 所以这里不需要、也**没法**通过 cfg 设置 ✓
    samp = PPOConfig.preset("llm").replace(
        max_new_tokens=args.mnt, kl_coef=0.0, device=dev,
        temperature=1.0, rollout_prompts=args.chunk)

    if (not args.vllm) and args.chunk > 32:
        print(f"  [vllm] ⚠️ --chunk {args.chunk} 偏大 ✗ → 训练器重算要 {args.chunk} 行 ✓ "
              f"lm_head 显存 ∝ 行数 ✗（实测 128 行要 13.9G ✗✗）→ 建议 ≤16 ✓ ✓")
        if args.chunk > 64:
            raise SystemExit(f"--chunk {args.chunk} 几乎必然 OOM ✗ → 请用 ≤32 ✓")

    _VB = None
    if args.vllm:
        from vllm_backend import build_from_adapter
        _VB = build_from_adapter(args.model, args.adapter,
                                 gpu_util=args.vllm_gpu_util, kv_gb=args.vllm_kv_gb,
                                 sleep_level=args.vllm_sleep_level)
    greed = samp.replace(temperature=0.0)

    hits = torch.zeros(n, dtype=torch.long)        # 每题对了几次（宽松判据）
    strict = torch.zeros(n, dtype=torch.long)      # 严格（= 训练奖励那个）
    greed_hits = 0

    def sweep(cfg, seed: int, mode: str) -> None:
        """★★ 有 vLLM 就**直采** ✓✓ —— 原来走 `rollout` ✗ → 顺带重算 logprobs ✗
        → 显存 ∝ 行数 ✗ → chunk 被压到 ≤16 ✗ → 每次只 16 条序列 ✗ → 慢 6 倍 ✗
        ✓ 现在 `vllm_backend.generate` ✓ → 只吃 KV ✓ → chunk 可 64 ✓
        """
        nonlocal greed_hits
        for i in range(0, n, args.chunk):
            part = test[i:i + args.chunk]
            if _VB is not None:
                ids = [tok.apply_chat_template(
                            D.build_messages(pr["question"]),
                            add_generation_prompt=True,
                            return_tensors="pt")["input_ids"][0].tolist()
                       for pr in part]
                outs = _VB.generate(ids, temperature=float(cfg.temperature),
                                    max_tokens=int(cfg.max_new_tokens),
                                    seed=int(seed) + i)
                pairs = [(pr, tok.decode(o, skip_special_tokens=True))
                         for pr, o in zip(part, outs)]
            else:
                b = rollout(model, tok, part, None, cfg, seed=seed,
                            vllm_backend=None)
                Lp = gen_start(b.mask)
                pairs = [(pr, tok.decode(b.input_ids[j, int(Lp[j]):],
                                         skip_special_tokens=True))
                         for j, pr in enumerate(part)]
                del b
            for j, (pr, txt) in enumerate(pairs):
                # ★ **pass@k 的 hit 用同一判据** ✓
                #   原来 hit 用 `reward_fn_base`（抽数字 ✗）→ 在 MATH 上**永远错** ✗✗
                #   现在：hit 和 strict 都走 `--reward-mode` ✓（MATH 没有"宽松版" ✓）
                ok = _judge(txt, pr["gold"], args.reward_mode)
                if mode == "sample":
                    hits[i + j] += ok
                    strict[i + j] += ok
                else:
                    greed_hits += ok
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    _tag = f" · 分片 {args.shard}" if args.shard else ""
    print("采样 " + str(args.k) + " 轮 x " + str(n) + " 题（T=1.0，和 RL 训练同分布）" + _tag)
    for r in range(args.k):
        sweep(samp, seed=1000 + r, mode="sample")
        print("  第 " + str(r + 1) + "/" + str(args.k) + " 轮完成", flush=True)

    if args.greedy:
        print()
        print("贪心 " + str(n) + " 题（对照）")
        sweep(greed, seed=0, mode="greedy")
    else:
        print("  ★ 跳过贪心（`--greedy` 才跑 ✓ 省一半时间 ✓）")

    K = args.k
    p1 = float((hits / K).mean())
    print()
    print("=" * 62)
    print("  题数 " + str(n) + " x " + str(K) + " 次 = " + str(n * K) + " 个样本")
    print("  pass@1（单次采样答对率）  = %.3f" % p1)
    for k in (2, 4, 8):
        if k <= K:
            pk = sum(pass_at_k(K, int(c), k) for c in hits.tolist()) / n
            print("  pass@" + str(k) + "（k 次里至少一次对） = %.3f   （比 pass@1 高 %+.3f）"
                  % (pk, pk - p1))
    print("  严格 pass@1（训练奖励量的） = %.3f" % float((strict / K).mean()))
    # ★★ 没跑贪心时**不能打 0.000** ✗✗ —— 那会被读成"模型贪心全错" ✓
    #   （实测就被读成"回退了" ✗ 而其实是 `--greedy` 默认关 ✓）
    if args.greedy:
        print("  贪心准确率（和 RL 无关的对照） = %.3f" % (greed_hits / n))
    else:
        print("  贪心准确率 = **未测** ✓（`--greedy` 才跑 ✓ 约 +2 分钟 ✓）")
    print()
    print("  每题答对次数分布（共 " + str(K) + " 次）：")
    cnt = Counter(hits.tolist())
    for k in range(K + 1):
        c = cnt.get(k, 0)
        print("    %d/%d 次对: %4d 题  %s" % (k, K, c, "#" * int(c / max(n, 1) * 50)))

    if K < 8:
        print()
        print("=" * 62)
        print("  ⚠️ 只采了 " + str(K) + " 次 → **pass@8 算不出来** ✗")
        print("     上次我在这里打了一句「没有可利用的差距」—— 那是假的 ✗")
        print("     因为 pass_at_k(K, c, min(8,K)) 在 K<8 时恒等于 pass@1 ✓")
        print("     要判断 headroom 请用 --k 8 ✓")
        return 0

    pk8 = sum(pass_at_k(K, int(c), min(8, K)) for c in hits.tolist()) / n
    gap = pk8 - p1
    print()
    print("=" * 62)
    if gap > 0.15:
        print("  pass@8 - pass@1 = %+.3f  -> 有可利用的差距" % gap)
        print("  模型会做、只是不稳定 -> RL 有空间，该去修优化（B=32 那条路）")
    elif gap > 0.05:
        print("  pass@8 - pass@1 = %+.3f  -> 差距不大" % gap)
        print("  空间有限，先别指望大幅提升")
    else:
        print("  pass@8 - pass@1 = %+.3f  -> 几乎没有可利用的差距" % gap)
        print("  这是会不会的问题，不是稳不稳的问题")
        print("  -> RL 学不到东西，该换更大模型（4B）或改任务")
    return 0


if __name__ == "__main__":
    sys.exit(main())
