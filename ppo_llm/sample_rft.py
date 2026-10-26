"""sample_rft.py —— 拒绝采样（RFT / STaR 式自举），把"模型自己做对的解答"变成 SFT 数据。

## 为什么这条路值得走（用我们自己的数字 ✓）

```
pass@1 = 0.390   ← 实际拿到的
pass@8 = 0.758   ← 模型**已经会做**的
        ─────
        差 0.368  → 这 37 点全是"能不能稳定做出来" ✓
```

RL 改的是**概率分布** ✓（我们已看到 T=1 从 0.324 → 0.527 ✓）；
RFT 改的是**数据** ✓ —— 直接教"你已经会的那条路" ✓✓

**★ 而它有个 RL 没有的优点** ✗✓：**它是监督学习，没有轨迹混沌** ✓✓
（我们实测 RL 的 run-to-run 能差 5~7 点 ✗ 而 SFT 在固定数据+固定 seed 下是确定的 ✓）

## 它不是万能的（说清楚 ✗）

* **上限是 pass@8 = 0.758** ✗ —— 它不创造新能力 ✓ 只吃"已经会的" ✓
* **多样性会下降** ✗（这是它的主要代价 ✓）

## 用法

```bash
# 1) 生成（用 fast 批量 ✓ 32 条一次解 ✓）
python3 sample_rft.py --adapter /data/ppo/runs/b32long_lora \
    --pool rl --problems 1500 --k 8 --out runs/rft_b32long.jsonl

# 2) 用它做一轮 SFT（sft.py 现在支持 --data 了 ✓）
python3 sft.py --data runs/rft_b32long.jsonl --limit 6000 --epochs 2 \
    --out /data/ppo/runs/sft_rft1

# 3) 评估 + 探针（和以前完全同口径 ✓）
python3 sft.py --eval-only /data/ppo/runs/sft_rft1
python3 probe_passk.py --adapter /data/ppo/runs/sft_rft1 --problems 128 --k 8
```

## 两个设计选择（都说明理由 ✓）

1. **默认从 RL 池取题** ✓（`--pool rl` = train[1500:] ✓ **SFT 没见过** ✓）
   → 比从 SFT 池取更有价值 ✓（那是自蒸馏 ✓ 也有效但弱一些 ✓）
   ⚠️ 但注意：如果你之后又在**同一个 RL 池**上做 RL ✗，那里的训练奖励会偏高 ✗
      （因为模型见过这些题的答案了 ✓）—— **评估仍然诚实** ✓✓ 因为我们评估在**测试集** ✓

2. **筛选用 gold 严格判据** ✓（`answer_strict` ✓ 即必须含 `#### N` ✓）
   → 不能只靠"多数投票" ✗ —— GSM8K **有标准答案** ✓ 用 gold 更硬 ✓✓
   → 加 `--min-proc 1.0` 可以额外要求"所有算式标注都算对" ✓（更严 ✓ 但数据更少 ✓）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from config import PPOConfig
from batch import gen_start
from model import MODEL_PATH, LMWithValueHead, empty_cache, load_tokenizer, rollout


# =========================================================================== #
# 1. 筛选（单独成函数 ✓ 这样能用精确数值钉住它 ✓）
# =========================================================================== #
def _staged_ok(text: str) -> bool:
    """三段式合规 ✓（和 `sft.staged_ok` 同一判据 ✓ 为了避免重复实现 ✓ 这里就地写一份最小的 ✓）"""
    H = ("[Plan]", "[Calculation]", "[Answer]")
    idx = [text.find(h) for h in H]
    if any(i < 0 for i in idx) or idx != sorted(idx):
        return False
    for k, h in enumerate(H):
        end = idx[k + 1] if k + 1 < len(H) else len(text)
        if not text[idx[k] + len(h):end].strip():
            return False
    return True


def keep_sample(text: str, gold: str, min_proc: float = 0.0,
                mode: str = "strict", require_staged: bool = False
                ) -> tuple[bool, float]:
    """这条采样值不值得进 SFT 数据？返回 `(是否保留, 算式正确比例)`。

    * **必须严格答对** ✓（`answer_strict` = 显式写了 `#### N` 且和 gold 相等 ✓）
      —— 严格而不是宽松 ✗：RFT 的数据会成为 SFT 的目标 ✓
         而 SFT 学的是**完整文本** ✓ → 用宽松判据会把"答案对但没写格式"的样本收进来 ✓
         → 那正好**稀释**了格式 ✓✓（我们好不容易把格式率做到 0.844 ✗）
    * `min_proc`：可选的额外要求 ✓ —— 算式标注的正确比例 ≥ 它 ✓
      （`reward_parts(text, gold, 1.0)[1]` 就是那个比例 ✓ 因为系数取 1 ✓）
    """
    # ★ 判据跟 `--reward-mode` 走 ✓✓
    #   `math`：MATH 是 LaTeX ✓ → `answer_strict`（只认 ####）会**一条都留不下** ✗✗
    if mode == "math":
        from math_data import answer_math
        if not answer_math(text, gold):
            return False, 0.0
    elif mode == "lenient":
        if not D.answer_lenient(text, gold):
            return False, 0.0
    elif not D.answer_strict(text, gold):
        return False, 0.0
    if require_staged and not _staged_ok(text):
        return False, 0.0
    ratio = float(D.reward_parts(text, gold, 1.0)[1])
    return ratio >= min_proc, ratio


# =========================================================================== #
# 2. 自测（不加载模型 ✓ 秒级）
# =========================================================================== #
def _case_keep_sample() -> None:
    """★ 用精确文本钉住筛选逻辑 ✓

    要害：**必须严格判据** ✗ —— 一条"答案对但没写 `####`"的样本如果被收进来 ✓
    就会在 SFT 里教模型"不用写格式" ✗✓（把 0.844 的格式率往下拉 ✓）
    """
    good = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n#### 18"
    no_fmt = "She sells 9 eggs, so the answer is 18."
    wrong = "She sells 9 eggs.\n#### 99"
    bad_arith = "1 + 1 = <<1+1=3>>3\n#### 18"

    ok, ratio = keep_sample(good, "18")
    if not ok:
        raise AssertionError("严格答对且算式正确的样本应该保留 ✗")
    if abs(ratio - 1.0) > 1e-9:
        raise AssertionError(f"算式全对时比例应为 1.0 ✗ 实际 {ratio}")
    print(f"    严格答对 + 算式全对 → 保留 ✓（ratio={ratio}）")

    if keep_sample(no_fmt, "18")[0]:
        raise AssertionError(
            "「答案对但没写 #### 」被收进来了 ✗ —— 这会教模型不写格式 ✓（必须用严格判据 ✓）")
    if keep_sample(wrong, "18")[0]:
        raise AssertionError("答案错的样本被收进来了 ✗")
    print("    无格式 / 答案错 → 丢弃 ✓")

    ok2, r2 = keep_sample(bad_arith, "18", min_proc=1.0)
    if ok2:
        raise AssertionError("--min-proc 1.0 时算式算错的样本应被丢掉 ✗")
    ok3, r3 = keep_sample(bad_arith, "18", min_proc=0.0)
    if not ok3:
        raise AssertionError("--min-proc 0 时算式错但答案对的样本应保留 ✗")
    print(f"    min_proc 生效：同一句 ratio={r3:.2f} → 0.0 时留 ✓ / 1.0 时丢 ✓")


# =========================================================================== #
# 3. 主流程
# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None, help="要采样的 adapter（默认基座）")
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--pool", type=str, default="rl", choices=["rl", "sft"],
                    help="从哪个池取题：rl（默认 ✓ SFT 没见过 ✓）或 sft")
    ap.add_argument("--problems", type=int, default=1500)
    ap.add_argument("--require-staged", action="store_true", dest="require_staged",
                    help="★ 只留**三段式合规**的样本 ✓（`[Plan]`+`[Calculation]`+`[Answer]` ✓）"
                         " —— 造三段式 RFT 数据时用 ✓")
    ap.add_argument("--reward-mode", type=str, default="strict", dest="reward_mode",
                    choices=["strict", "lenient", "math"],
                    help="筛选判据 ✓（MATH 类**必须** math ✗）")
    ap.add_argument("--test-file", type=str, default=None, dest="test_file",
                    help="★ 用这个 jsonl 当采样池 ✓")
    ap.add_argument("--k", type=int, default=8, help="每题采几条")
    ap.add_argument("--chunk", type=int, default=32,
                    help="★ 每次处理**多少道题** ✓（不是多少条 ✗）—— "
                         "实际每次 rollout 的行数 = chunk × k ✗ → 所以它同时决定 "
                         "① 一次喂 vLLM 多少条 ✓ ② **训练器重算的行数** ✗（显存 ∝ 行数 ✗）。"
                         "用 --vllm 时：chunk×k ≤ 24 安全 ✓（如 chunk 3 × k 8 = 24 ✓）")
    ap.add_argument("--max-new-tokens", type=int, default=192, dest="mnt")
    ap.add_argument("--min-proc", type=float, default=0.0, dest="min_proc",
                    help="额外要求算式标注正确比例 ≥ 它（1.0 = 全对）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    # ↑ 不用 required ✗ 否则 --self-test 也会被拦下 ✓（argparse 先校验参数 ✓）
    ap.add_argument("--resume", action="store_true", help="接着上次跑（读 <out>.progress ✓）")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    # ★ 推理走 vLLM ✓（纯生成 ✓ 权重不变 ✓ 所以只同步一次 ✓✓）
    ap.add_argument("--vllm", action="store_true", help="用 vLLM 生成（快 ~2× ✓）")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--vllm-kv-gb", type=float, default=1.5)
    ap.add_argument("--vllm-sleep-level", type=int, default=1, dest="vllm_sleep_level",
                    help="生成后 sleep 释放（0 = 不释放 ✓ 48G 上建议 0 ✓ "
                         "—— 实测这版 level 1 只释放 1G ✗ 反而留碎片 ✓）")
    args = ap.parse_args()

    if args.self_test:
        print("用例：拒绝采样的筛选")
        _case_keep_sample()
        print("\n✅ sample_rft.py 自测通过")
        return 0
    if not args.out:
        ap.error("--out 是必填的（除非用 --self-test）")

    dev = PPOConfig.auto_device()
    if args.test_file:
        pool = D.load_test_file(args.test_file)
        print(f"  采样池：{args.test_file} · {len(pool)} 题 ✓")
    else:
        sft_pool, rl_pool = D.split_train()
        pool = rl_pool if args.pool == "rl" else sft_pool
    problems = pool[: args.problems]
    rows_per_call = args.chunk * args.k
    print(f"设备 {dev} | adapter {args.adapter} | 池 {args.pool}"
          f"（共 {len(pool)} 题）| 取 {len(problems)} 题 × k={args.k}")
    print(f"★ 每次 generate 的行数 = chunk({args.chunk}) × k({args.k}) = {rows_per_call}"
          f"（我们实测过的是 **32** ✓）")
    if rows_per_call > 64:
        print(f"  ⚠️ {rows_per_call} 行远超测过的 32 ✗ —— 显存没验证过 ✓")
        print(f"     建议：--chunk {max(1, 32 // args.k)}（这样每次正好 32 行 ✓，总耗时不变 ✓）")
    if args.k > 1:
        pass

    tok = load_tokenizer(args.model)
    model = LMWithValueHead(args.model, device=dev, lora=True, init_lora=args.adapter)

    cfg = PPOConfig.preset("llm").replace(
        max_new_tokens=args.mnt, kl_coef=0.0, device=dev,
        temperature=1.0, rollout_prompts=args.chunk)

    # ★ 需要 vLLM 就先建好（**一次** ✓ 权重不变 ✓✓）
    # ★★ vLLM 只替"生成" ✗ —— 重算 logprob 仍由**训练器**做 ✓
    #   → 每次 rollout 的行数 = chunk × k ✗ → 太大就 OOM（实测 128 行 → lm_head 要 13.9G ✗✗）
    #   → 自动把"每次调用的题数"压到安全值 ✓✓
    if args.vllm and rows_per_call > 24:
        safe_chunk = max(1, 24 // max(1, args.k))
        print(f"  [vllm] ⚠️ 每次调用 {rows_per_call} 行（chunk {args.chunk} × k {args.k}）"
              f"→ 训练器重算会 OOM ✗ → 自动改成 chunk={safe_chunk}（{safe_chunk * args.k} 行 ✓）")
        args.chunk = safe_chunk
        rows_per_call = args.chunk * args.k

    _VB = None
    if args.vllm:
        from vllm_backend import build_from_adapter
        _VB = build_from_adapter(args.model, args.adapter,
                                 gpu_util=args.vllm_gpu_util, kv_gb=args.vllm_kv_gb,
                                 sleep_level=args.vllm_sleep_level)

    prog_path = args.out + ".progress"
    start = 0
    if args.resume and os.path.exists(prog_path):
        with open(prog_path) as f:
            start = int(f.read().strip() or 0)
        print(f"★ resume：从第 {start} 题继续 ✓（已有 {os.path.getsize(args.out) if os.path.exists(args.out) else 0} 字节）")

    n_kept = n_done = 0
    rng = random.Random(args.seed)
    mode = "a" if (args.resume and start) else "w"
    with open(args.out, mode, encoding="utf-8") as fout:
        for i in range(start, len(problems), args.chunk):
            part = problems[i:i + args.chunk]
            # 每题采 k 条：同一批 prompt 重复 k 次 → 每行独立采样 ✓
            rows = [q for q in part for _ in range(args.k)]
            b = rollout(model, tok, rows, None, cfg,
                        seed=rng.randrange(10 ** 9), vllm_backend=_VB)
            Lp = gen_start(b.mask)
            for j, q in enumerate(rows):
                txt = tok.decode(b.input_ids[j, int(Lp[j]):], skip_special_tokens=True)
                ok, ratio = keep_sample(txt, q["gold"], args.min_proc, mode=args.reward_mode,
                            require_staged=args.require_staged)
                n_done += 1
                if ok:
                    fout.write(json.dumps(
                        {"question": q["question"], "solution": txt.strip(),
                         "gold": q["gold"], "proc": round(ratio, 4)},
                        ensure_ascii=False) + "\n")
                    n_kept += 1
            fout.flush()                                   # ★ 每块就落盘 ✓ 崩了也不丢 ✓
            with open(prog_path, "w") as fp:
                fp.write(str(min(i + args.chunk, len(problems))))
            del b
            empty_cache()
            print(f"  {min(i + args.chunk, len(problems)):>5}/{len(problems)} 题"
                  f" | 保留 {n_kept}/{n_done}（{n_kept / max(n_done, 1):.1%}）", flush=True)

    n_per_q = n_kept / max(len(problems), 1)
    print(f"\n{'=' * 62}")
    print(f"  采样 {n_done} 条 → 保留 **{n_kept}** 条（{n_kept / max(n_done, 1):.1%}）")
    print(f"  平均每题 {n_per_q:.2f} 条（上限 {args.k} ✓ 全对题会贡献 {args.k} 条 ✓）")
    print(f"  写到 {args.out} ✓")
    print(f"\n  下一步：python3 sft.py --data {args.out} --limit {min(n_kept, 99999)} \\")
    print(f"              --epochs 2 --out /data/ppo/runs/sft_rft1")
    print(f"  ⚠️ 它只会把 pass@1 往 **pass@8 = 0.758** 推 ✗（不创造新能力 ✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
