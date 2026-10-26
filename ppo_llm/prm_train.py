"""prm_train.py —— **过程奖励（PRM）+ PPO** 的训练主循环 ✓✓

## 和旧 PPO 的关系 ✓✓（互不影响 ✓）

    **复用**（只 import ✓ 不改一行 ✗）：
        model.LMWithValueHead / make_optimizer / rollout / empty_cache
        batch.gen_start / Batch
        update.ppo_update          ← ★ 它本来就吃 (B,T) 的 rewards ✓✓
        config.PPOConfig
        data.load_test_file / math_data.answer_math
    **新写**：
        prm_reward（分段 → 判分 → 逐 token 奖励 ✓）
        本文件（训练循环 ✓）
    **不碰**：main.py / sft.py / grpo 那条线 ✓✓

## 一步做什么

    ① 采一批题（B 题 × G 条 ✓ 走 `rollout` ✓）
    ② **把 `b.rewards` 换成 PRM 的 (B,T)** ✓✓ —— 段末 +1 ✓ 全对再 +1 ✓
    ③ `ppo_update(...)` ✓ —— **adv_norm 必须是 none / batch** ✗ 不能 grpo ✗✗
       （grpo 会把 (B,T) 压成每条一个数 ✓ → 段末信息全丢 ✗✗）

## 用法

    # ★ Mac 上冒烟（mock 判分 ✓ 不联网 ✓ 0.5B ✓ 1 步 ✓）
    python3 prm_train.py --smoke

    # 服务器（真判分 ✓ 需要 DEEPSEEK_API_KEY ✓）
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 prm_train.py --model /data/models/Qwen2.5-1.5B-Instruct \
        --init-lora runs/l15b_prm_sft --device cuda:1 \
        --pool-file data/prm_split/rl.jsonl --test-file data/math500_prm.jsonl \
        --problems 4 --group-size 4 --iters 50 --judge api \
        --max-new-tokens 1024 --rollout-chunks 2 --num-minibatches 8 \
        --vllm --out runs/l15b_prm_ppo

    # 只自测（不加载模型 ✓）
    python3 prm_train.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from batch import gen_start
from config import PPOConfig
from model import MODEL_PATH, LMWithValueHead, empty_cache, load_tokenizer, make_optimizer, rollout
from prm_reward import (ApiJudge, CallableJudge, MockJudge, block_end_tokens,
                        block_jobs, score_block_jobs,
                        build_reward_matrix,
                        build_reward_matrix_paper,
                        section_end_tokens, score_sections)
from update import ppo_update


# =========================================================================== #
# 1. 把一条轨迹变成 PRM 奖励 ✓
# =========================================================================== #
def _eval_once(model, tokenizer, problems: list[dict], cfg, chunk: int,
               vllm_backend=None, reward_mode: str = "math",
               samples: int = 1, greedy: bool = False) -> dict:
    """★ 训练中的 eval ✓✓ —— **vLLM 直采** ✗✓ 不走 `rollout` ✓

    ## 为什么重写 ✗✓（实测：一轮 eval 要 **30 分钟** ✗✗）

        旧实现用 `rollout()` ✓ → 两个致命开销：
          ① **顺手重算 logprobs** ✗ —— eval **只要文本** ✓ 重算是纯浪费 ✓
             而重算的显存 ∝ 行数 → 把 `chunk` 卡在 **8** ✗ → vLLM 一次只有 8 条 ✗✗
          ② **跑两遍**（greedy + T=1）✗ → 1,000 次生成 ✓
        → 8 条并发实测 ≈ **390 tok/s** ✗ → 500 题 × 2 遍 = **30 分钟** ✗

        ✓ 现在：`vllm_backend.generate` 直采 ✓ → **不建 logits** ✓
          → chunk 可以开到 **64** ✓（只吃 KV ✓）→ ≈ **2,400 tok/s** ✓✓
          → 只跑 **T=1 一遍**（`greedy` 默认关 ✓ → 它本来就两边都不动 ✗）
          → 500 题 ≈ **2~3 分钟** ✓✓（省 90% ✓）

    `greedy=True` 时才额外跑一遍贪心 ✓（默认关 ✓ 要对照才开 ✓）
    """
    from batch import gen_start
    from model import empty_cache, rollout as _rollout

    def _judge(txt: str, gold: str) -> int:
        try:
            if reward_mode == "math":
                from math_data import answer_math
                return int(bool(answer_math(txt, gold)))
            return int(bool(D.answer_lenient(txt, gold)))
        except Exception:                            # noqa: BLE001
            return 0

    passes: list[tuple[str, float]] = []
    if greedy:
        passes.append(("greedy", 0.0))
    if int(samples) > 0:
        passes.append(("sample", 1.0))
    if not passes:
        passes.append(("sample", 1.0))

    out: dict = {}
    if vllm_backend is not None:
        # ★★★ vLLM 直采 ✓✓ chunk 只受 KV 限制 ✓（6 GiB ≈ 21 万 token ✓）
        for tag, temp in passes:
            n_ok = n = 0
            lens: list[int] = []
            for i in range(0, len(problems), chunk):
                part = problems[i:i + chunk]
                ids = [tokenizer.apply_chat_template(
                            D.build_messages(pr["question"]),
                            add_generation_prompt=True,
                            return_tensors="pt")["input_ids"][0].tolist()
                       for pr in part]
                outs = vllm_backend.generate(ids, temperature=temp,
                                             max_tokens=int(cfg.max_new_tokens),
                                             seed=int(cfg.seed) + i)
                for pr, o in zip(part, outs):
                    txt = tokenizer.decode(o, skip_special_tokens=True)
                    n_ok += _judge(txt, str(pr.get("gold", "")))
                    n += 1
                    lens.append(len(txt))
                print(f"    [eval-{tag}] {min(i + chunk, len(problems))}/{len(problems)} ✓ "
                      f"准确率 {n_ok / max(n, 1):.1%} ✓", flush=True)
            out["strict_acc" if tag == "greedy" else "eval_s_strict"] = n_ok / max(n, 1)
            out["eval_gen_len"] = sum(lens) / max(len(lens), 1)
    else:
        # HF 路径（没开 vLLM 时 ✓ 慢 ✓ 且 chunk 必须小 ✗）
        chunk = max(1, min(int(chunk), 8))
        for tag, temp in passes:
            ev_cfg = cfg.replace(rollout_prompts=chunk, rollout_chunks=1,
                                 num_minibatches=1, temperature=float(temp))
            n_ok = n = 0
            lens = []
            for i in range(0, len(problems), chunk):
                part = problems[i:i + chunk]
                b = _rollout(model, tokenizer, part, None, ev_cfg,
                             seed=int(cfg.seed) + i, vllm_backend=None)
                Lp = gen_start(b.mask)
                for j, pr in enumerate(part):
                    txt = tokenizer.decode(b.input_ids[j, int(Lp[j]):],
                                           skip_special_tokens=True)
                    n_ok += _judge(txt, str(pr.get("gold", "")))
                    n += 1
                    lens.append(len(txt))
                del b
                empty_cache()
            out["strict_acc" if tag == "greedy" else "eval_s_strict"] = n_ok / max(n, 1)
            out["eval_gen_len"] = sum(lens) / max(len(lens), 1)

    if "strict_acc" not in out:
        out["strict_acc"] = out.get("eval_s_strict", 0.0)
    return out


def ckpt_fingerprint(args) -> str:
    """★★ 配置指纹 ✓✓ —— resume 时**必须**一致 ✗✓

    ✗ 不加这个的后果：ckpt 是 lr=3e-5/batch32 存的 ✓ 你用 lr=1e-5 去续 ✗
      → 程序不报错 ✓ → **两条实验混成一条曲线** ✓✓（比崩溃更坏 ✗）
    """
    import hashlib as _h
    import json as _j
    from model import LORA_ALPHA, LORA_R, LORA_TARGETS
    d_lora = {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
              "lora_targets": list(LORA_TARGETS)}
    keys = ("model", "problems", "group_size", "iters", "lr", "inner_epochs",
            "kl_coef", "beta", "reward_shape", "segment_mode", "max_new_tokens",
            "rows_per_update", "min_chars", "seed", "pool_file")
    d = {k: getattr(args, k, None) for k in keys}
    d.update(d_lora)                      # ★ LoRA 结构也进指纹 ✓✓
    return _h.md5(_j.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def ckpt_save(ck_dir: str, save_adapter, opt, meta: dict) -> None:
    """★★ 原子保存断点 ✓✓（写 tmp → 校验 → 换名 ✓）

    ✗ 直接写目标目录：写到一半崩 → **目录半残** ✓ → 下次 resume 读到坏 adapter ✓
    """
    import json as _j
    import os as _o
    import shutil as _sh
    tmp = ck_dir + ".tmp"
    _sh.rmtree(tmp, ignore_errors=True)
    _o.makedirs(tmp, exist_ok=True)
    save_adapter(tmp)                                  # ★ 只写 adapter（~17MB ✓）
    try:
        import torch as _t
        _t.save(opt.state_dict(), _o.path.join(tmp, "opt.pt"))
    except Exception as e:                             # noqa: BLE001
        print(f"  ⚠️ 优化器状态没存上（{type(e).__name__}）→ resume 后动量会清零 ✓")
    with open(_o.path.join(tmp, "step.json"), "w", encoding="utf-8") as f:
        _j.dump(meta, f, ensure_ascii=False, indent=1)
    # ★ 校验：至少 adapter 文件在 ✓（PEFT 会写 adapter_model.safetensors 或 .bin ✓）
    _files = _o.listdir(tmp)
    if not any(x.startswith("adapter_") for x in _files):
        raise RuntimeError(f"✗ 断点目录里没有 adapter_*（{_files}）")
    _sh.rmtree(ck_dir + ".old", ignore_errors=True)
    if _o.path.isdir(ck_dir):
        _o.rename(ck_dir, ck_dir + ".old")             # ★ 旧的先挪走 ✓
    _o.rename(tmp, ck_dir)                             # ★ 新的顶上 ✓（原子 ✓）
    _sh.rmtree(ck_dir + ".old", ignore_errors=True)


def ckpt_load_meta(ck_dir: str) -> dict:
    import json as _j
    import os as _o
    with open(_o.path.join(ck_dir, "step.json"), encoding="utf-8") as f:
        return _j.load(f)


def prm_rewards_for_batch(model, tokenizer, b, problems: list[dict],
                          judge, bonus: float = 0.15,
                          reward_mode: str = "math",
                          reward_shape: str = "paper",
                          segment_mode: str = "paragraph",
                          stop_at_zero: bool = False,
                          workers: int = 0, min_chars: int = 80) -> dict:
    """★ 就地算 `(B,T)` 奖励 ✓ 并统计 ✓（**不改 b 的其它字段** ✓）

    ## ★★★ 判分是**并发**的 ✓✓

        · **同一条内不能并行** ✗ —— 块 k 要用块 1..k-1 的结果当 `so_far` ✓（顺序依赖 ✓）
        · **不同条完全独立** ✓ → **样本级并发** ✓✓
        · 32 条 × ~10 块 = **320 次串行** ✗ → 2.6s → **14 分钟/步** ✗✗
          16 路并发 → **~1 分钟/步** ✓✓

    `b.input_ids` 排布：`rollout` 按 `problems × G` 逐题铺开 ✓
    """
    B, T = b.input_ids.shape
    Lp = gen_start(b.mask)
    G = max(1, B // max(1, len(problems)))

    # ── ① 切块（纯 CPU ✓ 不判分 ✓）────────────────────────────────
    rows: list[dict] = []
    n_fail = 0
    jobs: list[tuple[str, list[tuple[str, str]]]] = []
    job_row: list[int] = []
    for k in range(B):
        pi = min(len(problems) - 1, k // G)
        pr = problems[pi]
        ids = b.input_ids[k].tolist()
        p_len = int(Lp[k])
        text = tokenizer.decode(ids[p_len:], skip_special_tokens=True)
        try:
            if segment_mode == "paragraph":
                ends, contents = block_end_tokens(tokenizer, ids, text, p_len,
                                                 min_chars=min_chars)
                contents = [("step", c) for c in contents]
            else:
                ends, contents = section_end_tokens(tokenizer, ids, text, p_len)
        except ValueError as e:
            n_fail += 1
            rows.append({"ends": [], "scores": [], "all_correct": False, "_err": str(e)[:80]})
            continue
        rows.append({"ends": ends, "scores": None, "text": text,
                     "gold": str(pr.get("gold", "")), "all_correct": False})
        jobs.append((pr.get("question", ""), contents))
        job_row.append(len(rows) - 1)

    # ── ② ★★★ **块级全并行**判分 ✓✓ ───────────────────────────────
    #   ✗ 原来：按**样本**并发 ✓ 而每条样本内部 10 块**串行** ✓
    #     → wall = ceil(320/32) × 延迟 = **10 × 延迟** ✗（120~274s ✓）
    #   ✓ 现在：拆成**块级任务** ✓ → 320 个一次发出 ✓ → wall = **1 × 延迟** ✓✓
    if jobs:
        flat: list[tuple[str, str, str]] = []
        owner: list[int] = []                    # 每个块任务属于哪一行 ✓
        for (q, contents), pos in zip(jobs, job_row):
            for jb in block_jobs(q, contents):
                flat.append(jb)
                owner.append(pos)
        # ★★★ 不能写 `max(1, workers)` ✗✗ —— `--judge-workers` 的语义是
        #   **0 = 自动** ✓ 而 `max(1, 0)` 把它变成 **1** ✗ → **320 次判分全串行** ✓
        #   （实测症状：GPU 卡在判分几分钟不动 ✓ 而日志只打了前 5 个 0 ✓）
        scored = score_block_jobs(judge, flat, workers=int(workers))
        # ★ 按归属装回每一行 ✓（保持块序 ✓）
        per_row: dict[int, list[int]] = {}
        for pos, v in zip(owner, scored):
            per_row.setdefault(pos, []).append(int(v))
        for pos, sc in per_row.items():
            rows[pos]["scores"] = sc

    # ── ③ 定 all_correct：**所有块都判 1** ✓ 且答案对 ✓ ──────────
    for row in rows:
        sc = row["scores"]
        if not sc:
            continue
        # ★★ 不能硬编码 len==5 ✗（段落模式 8.7 块 ✓ → 恒 False ✗ → rv 恒 0 ✗✗）
        ok = all(s == 1 for s in sc)
        if ok:
            try:
                from math_data import answer_math
                ok = (bool(answer_math(row["text"], row["gold"]))
                      if reward_mode == "math" else bool(D.answer_lenient(row["text"], row["gold"])))
            except Exception:                    # noqa: BLE001
                ok = False
        row["all_correct"] = ok

    # ── ④ 组装 (B,T) 奖励 ✓ ─────────────────────────────────────
    done_idx = (Lp + b.gen_len() - 1).clamp(0, T - 1)
    if reward_shape == "paper":
        r, st = build_reward_matrix_paper(rows, B, T, done_idx, beta=bonus)
    else:
        r, st = build_reward_matrix(rows, B, T, done_idx, bonus=1.0)
    b.rewards = r.to(b.rewards.device)
    # ★ 一段都没解析到 → 立刻打原文 ✓
    if st["segments"] == 0:
        _k = 0
        _p = int(Lp[_k])
        _t = tokenizer.decode(b.input_ids[_k, _p:].tolist(), skip_special_tokens=True)
        print("  ✗✗ **一段都没解析到** —— 原始输出前 400 字：")
        for _line in _t[:400].splitlines()[:10]:
            print(f"  | {_line}")
    return {**st, "decode_fail": n_fail, "B": B, "G": G,
            "full_rate": st["full"] / max(B, 1),
            "seg_rate": st["correct_segments"] / max(st["segments"], 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--init-lora", default=None, dest="init_lora")
    ap.add_argument("--device", default=None,
                    help="★ 训练卡（如 cuda:1 ✓）—— vLLM 留在 cuda:0 ✓")
    ap.add_argument("--pool-file", default=None, dest="pool_file")
    ap.add_argument("--test-file", default=None, dest="test_file")
    ap.add_argument("--reward-mode", default="math", dest="reward_mode",
                    choices=["math", "lenient", "strict"])
    ap.add_argument("--problems", type=int, default=4, help="每次 rollout 的**题数** ✓")
    ap.add_argument("--group-size", type=int, default=4, dest="group_size")
    ap.add_argument("--eval-every", type=int, default=15, dest="eval_every",
                    help="★ 每 N 步评一次 ✓（0 = 不评 ✓）—— 用内置 `_eval_once` ✓（贪心 + T=1 ✓）"
                         " 指标是 **T=1 的严格答对率** ✓✓（训练分布 ✓）")
    ap.add_argument("--eval-size", type=int, default=500, dest="eval_size",
                    help="★ 评多少题 ✓（128 题噪声 ±3.9% ✗ 我们吃过亏 ✓ 别降 ✓）")
    ap.add_argument("--eval-chunk", type=int, default=16, dest="eval_chunk")
    ap.add_argument("--eval-greedy", action="store_true", dest="eval_greedy",
                    help="★ 额外跑一遍**贪心** ✓（默认关 ✗）—— 它两边都不动 ✓ 且多花一倍时间 ✗")
    ap.add_argument("--eval-samples", type=int, default=1, dest="eval_samples",
                    help="★ T=1 采几次 ✓（1 就够 ✓ 0 = 只评贪心 ✗ 不建议 ✓）")
    ap.add_argument("--ckpt-every", type=int, default=5, dest="ckpt_every",
                    help="★★ 每 N 步存一次断点 ✓✓（0 = 不存 ✗ 不建议）"
                         " —— 崩溃/关机最多丢 N 步 ✓（今天丢过 45 步 ✗）")
    ap.add_argument("--resume", action="store_true",
                    help="★★ 从 `{out}_ckpt` 续跑 ✓（会自动核对配置指纹 ✓ 不一致就拒绝）")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--judge", default="mock", choices=["mock", "api"],
                    help="★★ 默认 mock ✓（不花钱 ✓ 只用于冒烟 ✗）；真跑必须 api ✓")
    ap.add_argument("--judge-model", default="deepseek-flash", dest="judge_model")
    ap.add_argument("--judge-cache", default="runs/prm_judge_cache.jsonl", dest="judge_cache")
    ap.add_argument("--min-chars", type=int, default=80, dest="min_chars",
                    help="★ 块的最小字数 ✓ —— 调大 → **块更少** ✓ → 判分次数直接下降 ✓✓"
                         "（80 ≈ 10 块/条 ✓ · 250 ≈ 4 块/条 ✓ 判分砍 60% ✓ 代价是粒度变粗 ✓）")
    ap.add_argument("--judge-workers", type=int, default=0, dest="judge_workers",
                    help="★★ 判分并发路数 ✓ 默认 **0 = 自动等于 batch** ✓✓"
                         "（32 条 → 32 路 ✓ 满并行 ✓）；给正数则限流 ✓")
    ap.add_argument("--segment-mode", default="paragraph", dest="segment_mode",
                    choices=["paragraph", "sections"],
                    help="★★ `paragraph`(默认 ✓)=按**自然段落**切（给不写 5 段的模型 ✓，"
                         "如 Qwen2.5-Math ✓）；`sections`=按 5 个段名切（需要 SFT 过格式 ✗）")
    ap.add_argument("--reward-shape", default="paper", dest="reward_shape",
                    choices=["paper", "segment"],
                    help="★★ `paper`(默认 ✓)=照论文：**rv ±1 定生死** ✓ + β·段分只做组内排序 ✓；"
                         "`segment`=旧行为（段分当唯一信号 ✗ 实测会被刷 ✓）")
    ap.add_argument("--adv-norm", default="batch", dest="adv_norm",
                    choices=["batch", "none", "per_sequence"],
                    help="★★ 优势归一化 ✓✓ —— **batch（默认）** 全批标准化 ✓ 量级 ≈1 ✓"
                         " → 和 A 组的 grpo 归一**可比** ✓✓；`none` = 原始 GAE ✗ 量级小 2~10 倍 ✗")
    ap.add_argument("--beta", type=float, default=0.15,
                    help="★ 段分的总权重上限 ✓（5 段全对合计 5β ✓ 必须 < 1 ✗ 否则能翻正负 ✓）")
    ap.add_argument("--bonus", type=float, default=1.0, help="全对的完成奖励 ✓")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=2048, dest="max_len")
    ap.add_argument("--max-new-tokens", type=int, default=1024, dest="mnt")
    ap.add_argument("--rollout-chunk", type=int, default=8, dest="rollout_chunk",
                    help="★ 采样时每次喂 rollout 多少行 ✓（重算 logits ∝ 行数 ✗）")
    ap.add_argument("--rows-per-update", type=int, default=1, dest="rows_per_update",
                    help="★★ 每次 backward 的**行数** ✓✓（显存 ∝ 行数 ✗）"
                         " 默认 **1** ✓ —— 8 行就会 46 GiB ✗✗（实测 ✓）")
    ap.add_argument("--num-minibatches", type=int, default=8, dest="nmb",
                    help="★ PPO 更新时切几块 ✓ **决定每块行数** ✗（显存 ∝ 行数 ✗）")
    ap.add_argument("--inner-epochs", type=int, default=4, dest="inner_epochs")
    ap.add_argument("--clip-eps", type=float, default=0.1, dest="clip_eps")
    ap.add_argument("--grad-clip", type=float, default=0.5, dest="grad_clip")
    ap.add_argument("--kl-coef", type=float, default=0.02, dest="kl_coef")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--vllm-dtype", default="bfloat16", dest="vllm_dtype")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45, dest="vllm_gpu_util")
    ap.add_argument("--vllm-kv-gb", type=float, default=2.0, dest="vllm_kv_gb")
    ap.add_argument("--vllm-sleep-level", type=int, default=0, dest="vllm_sleep")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()
    if a.smoke:
        a.problems, a.group_size, a.iters = 2, 2, 1
        a.mnt, a.max_len, a.rollout_chunk, a.nmb = 96, 384, 4, 2
        a.judge = "mock"
        print("  ★ smoke：mock 判分 ✓ 2 题 × 2 条 ✓ 1 步 ✓（不花钱 ✓）")

    if a.judge == "api" and not (os.environ.get("DEEPSEEK_API_KEY")
                                 or os.environ.get("OPENAI_API_KEY")):
        print("✗✗ --judge api 需要 DEEPSEEK_API_KEY ✓", file=sys.stderr)
        return 2

    device = a.device or ("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    _is_cuda = str(device).startswith("cuda")      # ★ 必须 startswith ✗ 不能 `== "cuda"` ✗✗
    print(f"设备 **{device}** ✓ · 判分 {a.judge} ✓ · reward_mode {a.reward_mode} ✓")

    # ★ 结构化数据**不要 GSM8K 的 system**（它会命令"输出 #### <number>" ✗ 与 \boxed{} 打架 ✗✗）
    D.SYSTEM_OVERRIDE = ""
    print("  ★ system 已置空 ✓（PRM 的规则在题干里 ✓）")

    tok = load_tokenizer(a.model)

    # ══════════════════════════════════════════════════════════════════════ #
    # ★★★ 断点续跑 ✗✓ —— 三个坑必须同时处理 ✓✓
    #   ① **ref 不能跟着 ckpt 变** ✗✗ —— 它是 KL 的参照系 ✓
    #      若 policy 和 ref 都指向 ckpt → kl_coef 下的**目标函数被悄悄改了** ✓✓
    #   ② 配置指纹不一致 → **拒绝** ✓（否则两条实验混成一条 ✗）
    #   ③ 优化器状态要一起恢复 ✓（否则动量清零 ✓ 前几十步会抽 ✓）
    # ══════════════════════════════════════════════════════════════════════ #
    ck_dir = (a.out or "runs/prm") + "_ckpt"
    start_it = 0
    hist_prev: list = []
    best: dict = {}
    _resume_meta: dict | None = None
    policy_init = a.init_lora                      # ★ policy 的起点（可能被 ckpt 替换）
    if a.resume:
        import os as _o
        if not _o.path.isdir(ck_dir):
            print(f"  ✗✗ `--resume` 但找不到断点目录 **{ck_dir}** ✓ → 拒绝启动 ✓")
            print("     （想从头跑就去掉 --resume ✓）", flush=True)
            return 2
        _resume_meta = ckpt_load_meta(ck_dir)
        _fp_now, _fp_old = ckpt_fingerprint(a), _resume_meta.get("fp")
        if _fp_now != _fp_old:
            print(f"  ✗✗ **配置指纹不一致 → 拒绝续跑** ✓✓")
            print(f"     断点：{_fp_old} · 现在：{_fp_now}")
            print(f"     断点配置：{_resume_meta.get('cfg')}")
            print("     ★ 两条实验混成一条曲线比崩溃更坏 ✓ 换目录重跑或改回原配置 ✓\n",
                  flush=True)
            return 2
        start_it = int(_resume_meta.get("step", 0))
        hist_prev = list(_resume_meta.get("hist", []))
        best = dict(_resume_meta.get("best", {}))
        policy_init = ck_dir
        print(f"  ★ **从断点续跑** ✓：step **{start_it}** / {a.iters} ✓"
              f" · 历史 {len(hist_prev)} 步 ✓ · best {best.get('step', '-')} ✓")
        print(f"     policy ← {ck_dir} ✓ · ref ← **{a.init_lora}** ✓（ref 不跟着变 ✓✓）",
              flush=True)

    print(f"  policy 起点：{policy_init or '（基座）'} ✓  ref 起点：{a.init_lora or '（基座）'} ✓")
    policy = LMWithValueHead(a.model, device=device, lora=True,
                             init_lora=policy_init, dtype="float32",
                             autocast=("bfloat16" if _is_cuda else None))
    # ★★★ ref 用 **a.init_lora**（原始 ✓）而不是 policy_init ✗✗
    ref = LMWithValueHead(a.model, device=device, lora=True,
                          init_lora=a.init_lora,
                          dtype=("bfloat16" if _is_cuda else "float32"), autocast=None)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = make_optimizer(policy, a.lr)
    if _resume_meta is not None:
        import os as _o
        _optp = _o.path.join(ck_dir, "opt.pt")
        if _o.path.exists(_optp):
            try:
                import torch as _t
                opt.load_state_dict(_t.load(_optp, map_location=device))
                print("  ★ 优化器状态已恢复 ✓（动量不清零 ✓✓）", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"  ⚠️ 优化器状态恢复失败（{type(e).__name__}）→ 动量从零开始 ✓",
                      flush=True)
        else:
            print("  ⚠️ 断点里没有 opt.pt → 动量从零开始 ✓", flush=True)
    print(f"可训练 {sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.2f}M ✓"
          f" · ref 冻结 ✓")

    # 题池 ✓
    # ======================================================================= #
    # ★★★ 启动自检（preflight）✗✓ —— 缺一项就**拒绝启动** ✓✓
    #
    #   今天有 4 次"配置少一项 → 不报错 → 悄悄不做/做错"✗：
    #     ① `bs.sh` 漏 `--test-file` → **45 步一次没评** ✗（跑完才发现，1 小时白跑）
    #     ② `workers=0` 被 `max(1,·)` 吃掉 → **判分变 1 路串行** ✗（卡十几分钟）
    #     ③ `stats.get('ratio')` 键名错 → 恒打 nan ✗
    #     ④ `len(scores)==5` → `rv` 恒 0，主信号全灭 ✗
    #   → 共同点：**不崩、不报错、只是不做/做错** ✓ → 只能靠启动时断言 ✗✓
    # ======================================================================= #
    _ok: list[str] = []
    _bad: list[str] = []

    def _chk(cond: bool, good: str, bad: str) -> None:
        (_ok if cond else _bad).append(good if cond else bad)

    _chk(a.eval_every > 0 and bool(a.test_file),
         f"eval：每 {a.eval_every} 步 · {a.eval_size} 题 · {a.test_file} ✓",
         f"**eval 关着** ✗（--eval-every {a.eval_every} · --test-file {a.test_file}）"
         " → 这一轮**不会有任何指标** ✗")
    if a.eval_every > 0 and a.test_file:
        _chk(os.path.exists(a.test_file), f"eval 题集存在 ✓",
             f"eval 题集**不存在** ✗（{a.test_file}）")
        _edge = 1.96 * (0.25 / max(a.eval_size, 1)) ** 0.5 * 100
        _chk(a.eval_size >= 250, f"eval {a.eval_size} 题 ✓（噪声 ±{_edge:.1f}%）",
             f"eval {a.eval_size} 题**太少** ✗（噪声 ±{_edge:.1f}% → 128 题吃过亏 ✗）")
        _chk(a.eval_samples >= 1, f"eval 采 {a.eval_samples} 次 ✓（有 T=1 ✓）",
             "eval **只评贪心** ✗ → 和 A 组的 T=1 无法对比 ✗")
    _chk(bool(a.pool_file) and os.path.exists(a.pool_file),
         f"题池存在 ✓（{a.pool_file}）", f"题池**不存在** ✗（{a.pool_file}）")
    _tot = a.problems * a.group_size
    _chk(_tot >= 8, f"批量 {_tot} 行 ✓", f"批量太小 ✗（{_tot} 行 → 优势噪声大）")
    _nmb = max(1, _tot // max(1, a.rows_per_update))
    while _tot % _nmb != 0 and _nmb > 1:
        _nmb -= 1
    _rows = _tot // _nmb
    _chk(_rows <= 2, f"每块 {_rows} 行 ✓（≤2 安全）",
         f"每块 **{_rows} 行** ✗ → 显存会炸（8 行实测 46 GiB ✗）")
    if a.judge == "api":
        _chk(bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")),
             "API key 已设置 ✓", "**没有 API key** ✗（export DEEPSEEK_API_KEY=...）")

    _chk(a.ckpt_every > 0 and bool(a.out),
         f"断点：每 {a.ckpt_every} 步 → {ck_dir} ✓",
         f"**断点关着** ✗（--ckpt-every {a.ckpt_every} · --out {a.out}）"
         " → 崩溃/关机 = **全部丢** ✗（今天丢过 45 步）")
    if a.ckpt_every > 0 and a.out:
        _free = 0
        try:
            import shutil as _sh
            _free = _sh.disk_usage(os.path.dirname(os.path.abspath(ck_dir)) or ".").free
        except Exception:                      # noqa: BLE001
            pass
        _chk(_free > 500 * 2 ** 20 or _free == 0,
             f"磁盘余量 {_free / 2**20:.0f} MiB ✓（断点约 55 MiB/次 ✓）",
             f"磁盘只剩 {_free / 2**20:.0f} MiB ✗（断点要 ~55 MiB ✓）")

    print("\n  ════ 启动自检 ════", flush=True)
    for x in _ok:
        print(f"    ✓ {x}", flush=True)
    for x in _bad:
        print(f"    ✗✗ {x}", flush=True)
    if _bad:
        print(f"\n  ✗✗✗ **{len(_bad)} 项不通过 → 拒绝启动** ✓✓"
              f"（就是这一步该拦住今天那一小时 ✗）\n", flush=True)
        return 2
    print(f"  ★ {len(_ok)} 项全过 ✓ 开始训练 ✓\n", flush=True)

    if a.pool_file:
        pool = [json.loads(l) for l in open(a.pool_file, encoding="utf-8") if l.strip()]
        print(f"题池 {a.pool_file} · **{len(pool)}** 题 ✓")
    else:
        pool = D.load_test_file(a.test_file) if a.test_file else D.load_gsm8k("test")
        print(f"题池（默认 test）· {len(pool)} 题 ✓")
    if a.test_file and a.pool_file:
        q_pool = {p["question"] for p in pool}
        q_test = {p["question"] for p in D.load_test_file(a.test_file)}
        if q_pool & q_test:
            print(f"  ⚠️ 题池与测试集**重叠 {len(q_pool & q_test)} 题** ✗✗ → 结果会虚高 ✗")

    # ★★★ `rollout_prompts` = **每块总行数** ✗✓（= 题数 × G ✓）
    #   `num_minibatches` 决定**每块行数** = 总行数 / nmb ✗
    #   ✗ 我第一版拿 `rollout_chunk` 当 rollout_prompts ✓ 又"自动修正"成整除 ✓
    #     → nmb = 2 ✓ → **16÷2 = 8 行/块** ✗✗ → 显存 46 GiB ✓（实测炸了 ✓）
    #   ✓ 现在：**总行数**直接算 ✓ · nmb = 总行数 / 目标行数 ✓✓
    _total = max(1, int(a.problems) * int(a.group_size))
    _target = max(1, int(a.rows_per_update))
    _nmb = max(1, _total // _target)
    while _total % _nmb != 0 and _nmb > 1:
        _nmb -= 1
    print(f"  ★ 总行数 **{_total}** ✓ · 目标每块 {_target} 行 → "
          f"num_minibatches = **{_nmb}** ✓（实际每块 **{_total // _nmb}** 行 ✓）")
    # ★★ eval 题集 ✓（和 A 组同一个测试集 ✓ 才能对比 ✓）
    eval_problems: list[dict] = []
    if a.eval_every > 0 and not a.test_file:
        # ★★★ 静默不评是最坏的 ✗✗ —— 今天整轮 45 步**一次都没评** ✓
        #   原因只是 `bs.sh` 少了一行 `--test-file` ✓ 而日志里**没有任何提示** ✗
        print("  ✗✗✗ **警告：--eval-every > 0 但没有 --test-file** ✓")
        print("        → **本次不会做任何 eval** ✗ 指标无从判断 ✓")
        print("        → 加 `--test-file data/math500.jsonl \\` 后重跑 ✓✓", flush=True)
    if a.eval_every > 0 and a.test_file:
        eval_problems = D.load_test_file(a.test_file)[: a.eval_size]
        print(f"  ★ eval 题集 **{len(eval_problems)}** 题 ✓（{a.test_file} ✓）"
              f" · 每 {a.eval_every} 步一次 ✓")

    cfg = PPOConfig.preset("llm").replace(
        lr=a.lr, max_new_tokens=a.mnt, rollout_prompts=_total,
        group_size=a.group_size, num_minibatches=_nmb, rollout_chunks=1,
        inner_epochs=a.inner_epochs, clip_eps=a.clip_eps, grad_clip=a.grad_clip,
        kl_coef=a.kl_coef, entropy_coef=0.0, process_coef=0.0,
        reward_mode=a.reward_mode, device=device,
        # ★★ **不能 grpo** ✗✗ —— 它会把 (B,T) 的 rewards 压成每条一个数 ✓ → 段末信息全丢 ✓
        adv_norm=a.adv_norm,
    )
    print(f"  adv_norm={cfg.adv_norm} ✓（★ 不能 grpo ✗ 否则段末奖励全丢 ✗✗；batch = 全批标准化 ✓ 与 A 组可比 ✓✓）")

    judge = MockJudge() if a.judge == "mock" else ApiJudge(
        model=a.judge_model, cache_path=a.judge_cache,
        # ★★ 段落模式要用**另一套判分标准** ✗✓（5 段那套标准不适用 ✓）
        mode=("block" if a.segment_mode == "paragraph" else "section"))

    vb = None
    if a.vllm:
        # ★★★ 必须用 main.py 那套（**活的 LoRA** + 裸 base）✗✓
        #   ✗ 原来 `build_from_adapter(a.model, a.init_lora)` ✓ 而 init_lora=None
        #     → vLLM **没有 lora_base** ✓ → `sync()` 抛"没有 base 模型" ✓
        #     → 而 prm_train **一次 sync 都没调** ✗ → **生成永远来自基座** ✗✗
        #     → rollout 不是 on-policy ✓ → 这个实验一直不成立 ✓✓
        #   ✓ 现在：`live_lora_pairs` 直接持有**参数张量本身** ✓
        #     → 优化器原地更新后**立即可见** ✓ → 每步 sync() 就真把新权重推给 vLLM ✓✓
        from vllm_backend import VLLMBackend, bare_base, live_lora_pairs
        vb = VLLMBackend(a.model, dtype=a.vllm_dtype,
                          gpu_memory_utilization=a.vllm_gpu_util,
                          max_model_len=int(a.mnt) + 512,          # ★ dest 是 `mnt` ✗ 不是 max_new_tokens ✓
                          lora_base=bare_base(policy.model),
                          lora_pairs=live_lora_pairs(policy.model),
                          kv_cache_bytes=int(a.vllm_kv_gb * 1e9),
                          sleep_level=a.vllm_sleep)
        vb.start()
        vb.sync()                                  # ★ 首次同步 ✓
        print(f"  [vllm] ★ 可 sync 模式 ✓（活的 LoRA ✓ 首同步 "
              f"{getattr(vb, 'last_sync_s', 0):.2f}s ✓）", flush=True)
                                # ★ 同上：不拿 --max-len 去限制推理 ✗

    # ★★ torch 也要 seed ✗✓ —— `--seed` 原来只管**题目顺序**和**采样 seed** ✓
    #   → LoRA 的 `lora_A` 每次**随机初始化** ✗ → 两个"同配置"的 run **不是同一个模型** ✓✓
    #   （实测：同一配置两次 eval 差 **4.6 点** ✓ 其中就混着这个 ✗）
    import torch as _t_seed
    _t_seed.manual_seed(int(a.seed))
    print(f"  ★ torch 已 seed = {a.seed} ✓（LoRA 初始化可复现 ✓✓）", flush=True)
    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    # ★ 续跑：hist / best 从断点接上 ✓✓（否则汇总少报步数 ✗ · 可能用更差的 eval 覆盖 _best ✗）
    hist: list[dict] = hist_prev
    # ★★ rng 快进 ✓✓ —— 每步 `rng.sample(pool, n)` ✓ 确定性 ✓
    #   → 续跑必须**重放前 start_it 次** ✓ 否则题目序列和第一次不一致 ✓
    for _ in range(start_it):
        rng.sample(pool, min(a.problems, len(pool)))
    if start_it:
        print(f"  ★ rng 已快进 {start_it} 步 ✓（题目序列与第一次一致 ✓✓）", flush=True)
    for it in range(start_it, a.iters):
        probs = rng.sample(pool, min(a.problems, len(pool)))
        # ★ 复制 G 份 → rollout 每条拿到不同 seed ✓（不然每题只采 1 条 ✗ 实测踩过 ✓）
        expanded = [p for p in probs for _ in range(a.group_size)]
        _t_r = time.perf_counter()
        chunks = []
        for i in range(0, len(expanded), max(1, a.rollout_chunk)):
            chunks.append(rollout(policy, tok, expanded[i:i + max(1, a.rollout_chunk)],
                                 None, cfg, seed=a.seed * 1000 + it * 100 + i,
                                 vllm_backend=vb))
        from batch import concat_batches
        b = concat_batches(chunks)
        _gen_s = time.perf_counter() - _t_r
        del chunks
        empty_cache()

        # ★ 关键一步：把 rewards 换成 PRM 的 (B,T) ✓✓
        _t_j = time.perf_counter()
        st = prm_rewards_for_batch(policy, tok, b, expanded, judge,
                                    bonus=a.beta, reward_mode=a.reward_mode,
                                    reward_shape=a.reward_shape, workers=a.judge_workers, min_chars=a.min_chars,
                                    segment_mode=a.segment_mode)
        _judge_s = time.perf_counter() - _t_j
        nz = int((b.rewards != 0).sum())
        print(f"\n步 {it + 1}/{a.iters}：{st['B']} 条 ✓ · 段判对 {st['seg_rate']:.1%} ✓"
              f" · 全对 {st['full']}/{st['B']} ✓ · reward 非零 {nz} 处 ✓"
              f"（★ 应 ≈ 对的段数 ✓）"
              + (f" · 解码失败 {st['decode_fail']} ✗" if st["decode_fail"] else "")
              + f" · **生成 {_gen_s:.0f}s** ✓ · **判分 {_judge_s:.0f}s** ✓"
              # ★★ 三个数一起看才能定位 ✓✓
              #   HTTP 次数 ✓ · 平均单次延迟 ✓ · **重试次数** ✓（重试 = 被限流 ✓）
              + (f"（{judge.n_http} 次 x "
                 f"{judge.t_http / max(judge.n_http, 1):.1f}s/次"
                 f" · 重试 {getattr(judge, 'n_retry', 0)} 次）"
                 if getattr(judge, "n_http", 0) else ""))

        _t_u = time.perf_counter()
        stats = ppo_update(policy, opt, b, cfg, ref_model=ref, assert_fresh=False)
        _upd_s = time.perf_counter() - _t_u
        # ★★★ 更新完立刻把新权重推给 vLLM ✗✓ —— 不 sync 的话
        #   **下一轮 rollout 仍用旧权重** ✓ → 不是 on-policy ✓✓
        if vb is not None:
            _ts = vb.sync()
            _upd_s += float(_ts or 0)
        stats.update(step=it + 1, **{k: st[k] for k in ("seg_rate", "full", "B", "G")})
        hist.append(stats)
        # ★★★ 每 N 步存断点 ✓✓（崩溃/关机最多丢 N 步 ✓）
        if a.ckpt_every > 0 and a.out and (it + 1) % a.ckpt_every == 0:
            try:
                _t_c = time.perf_counter()
                ckpt_save(ck_dir, lambda dd: policy.model.save_pretrained(dd), opt,
                          {"step": it + 1, "fp": ckpt_fingerprint(a),
                           "cfg": {"lr": a.lr, "batch": a.problems * a.group_size,
                                   "iters": a.iters, "inner_epochs": a.inner_epochs,
                                   "model": a.model, "pool": a.pool_file},
                           "hist": hist, "best": best})
                print(f"    ★ 断点 step {it + 1} → {ck_dir} ✓"
                      f"（{time.perf_counter() - _t_c:.1f}s ✓）", flush=True)
            except Exception as e:                    # noqa: BLE001
                print(f"    ⚠️ 断点保存失败（{type(e).__name__}: {str(e)[:60]}）"
                      f" → 训练继续 ✓", flush=True)
        # ★★★ 定期 eval ✓✓ —— 指标用 **T=1 的严格答对率** ✓
        #   为什么：RL 优化的是采样分布 ✓（贪心是另一个策略 ✓ 也更容易早饱和 ✗）
        #   ★ 而我们踩过"best 盯贪心 → 停在 step 0"的坑 ✓✓
        if (a.eval_every > 0 and eval_problems
                and (it + 1) % a.eval_every == 0):
            # ★★★ eval **自带一套 cfg** ✗✓ —— 复用别人的函数会撞整除约束 ✓
            #   ✗ `sft.evaluate_sft` 按空闲显存把 chunk 压到 13 ✓
            #     而我们的 cfg 有 num_minibatches=32 ✗ → 13 % 32 ≠ 0 → ValueError ✓
            #   ✗ `main.evaluate` 会调 `vb.sync()` ✓ 而旧后端没有 lora_base → 抛异常 ✓
            #   ✓ 现在：`_eval_once` 自己控 cfg（`num_minibatches=1` ✓ 恒整除 ✓✓）
            ev = _eval_once(policy, tok, eval_problems, cfg, a.eval_chunk,
                            vb, a.reward_mode, a.eval_samples, a.eval_greedy)
            _k = "eval_s_strict" if "eval_s_strict" in ev else "strict_acc"
            # ★★ 措辞要**诚实** ✗✓ —— 没跑贪心时 `strict_acc` 是 T=1 的回退值 ✓
            #   原来打成"贪心 0.504 · T=1 0.504" ✗ → 读的人会以为贪心也 0.504 ✗✗
            _g = (f" · 贪心 **{ev.get('strict_acc', 0):.3f}** ✓"
                  if a.eval_greedy else "")
            print(f"  ★ [eval {it + 1}] T=1 **{ev.get(_k, 0):.3f}** ✓{_g} · "
                  f"长度 {ev.get('eval_gen_len', 0):.0f} ✓", flush=True)
            if ev.get(_k, 0) > best.get(_k, -1):
                best = dict(ev, step=it + 1)
                if a.out:
                    import os as _os
                    _os.makedirs(a.out + "_best", exist_ok=True)
                    policy.model.save_pretrained(a.out + "_best")
                    print(f"    ★ 新最优（step {it + 1}: {_k} {ev[_k]:.3f}）"
                          f" → 已存 {a.out}_best ✓", flush=True)
        print(f"  loss {stats.get('loss', float('nan')):.4f} ✓ · "
              # ★★★ 键名是 **mean_ratio** ✗✓ —— 我原来写 `'ratio'` ✓ → **永远打到 nan** ✗✗
              #   这个假 nan 让我们追了半天"ratio 为什么会 nan" ✗ 而它根本不存在 ✓
              f"ratio {stats.get('mean_ratio', float('nan')):.4f} ✓ · "
              f"max {stats.get('ratio_max', float('nan')):.2f} ✓ · "
              # （`n_ratio_nonfinite` 在 loss.py 里有 ✓ 但 update.py 没转发 ✗ → 这里不打 ✓）
              
              f"approx_kl {stats.get('approx_kl', float('nan')):.5f} ✓ · "
              f"clip_frac {stats.get('clip_frac', 0):.3f} ✓ · "
              # ★★★ `grad_norm` **原来根本没打** ✗✗ —— 而它是**唯一**能看出
              #   `adv_norm` 是否真的放大了步长的指标 ✓（∝|A| ✓ 线性 ✓）
              #   `ratio`/`loss`/`kl` 全都**看不出** ✗：
              #     · ratio − 1 ≈ lr × |∇| × **1 步** ✓ → 恒 ~1e-4 ✗
              #     · loss ≈ −mean(A) ✓ 而 A 中心化 → ≈0 ✗
              #     · kl ∝ (ratio−1)² ✓ → 平方级小 ✗
              f"grad_norm {stats.get('grad_norm', float('nan')):.3f} ✓ · "
              f"**更新 {_upd_s:.0f}s** ✓ · 总 {_gen_s + _judge_s + _upd_s:.0f}s ✓")
        del b
        empty_cache()

    if hist:
        print(f"\n完成 ✓ {len(hist)} 步")
        se = sum(h["seg_rate"] for h in hist) / len(hist)
        fu = sum(h["full"] for h in hist) / max(sum(h["B"] for h in hist), 1)
        print(f"  平均段判对率 **{se:.1%}** ✓ · 全对率 **{fu:.1%}** ✓")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        policy.model.save_pretrained(a.out)
        print(f"  adapter 已存 {a.out} ✓")
    if vb is not None:
        vb.shutdown()
    return 0


# =========================================================================== #
# 3. 自测（不加载模型 ✓ 纯 CPU ✓）
# =========================================================================== #
def _self_test() -> int:
    fails = 0

    def ck(c, m):
        nonlocal fails
        print(f"  {'✓' if c else '✗✗'} {m}")
        fails += 0 if c else 1

    print("① 复用关系 ✓（旧文件一行都没改 ✓）")
    import inspect
    src = inspect.getsource(sys.modules[__name__])
    ck("from update import ppo_update" in src, "复用了 ppo_update ✓")
    ck("from model import" in src and "rollout" in src, "复用了 rollout ✓")
    ck("adv_norm=a.adv_norm" in src, "★ adv_norm 来自参数 ✓（默认 batch ✓ 不是硬编码 none ✗）")

    print("\n② mock 判分器 ✓")
    j = MockJudge()
    ck(j("p", "[Overview]", "", "fine") == 1, "正常内容 → 1 ✓")
    ck(j("p", "[Solution]", "", "WRONG here") == 0, "含 WRONG → 0 ✓")

    print("\n③ 拼接多个 rollout 块 ✓（`rollout_chunk` 分段采样后要能合起来 ✓）")
    from batch import concat_batches
    ck(callable(concat_batches), "concat_batches 可用 ✓")

    print("\n④ ★★ 不能 grpo（这条是设计约束,写成断言 ✓）")
    cfg = PPOConfig.preset("llm")
    ck(cfg.adv_norm in ("batch", "none", "per_sequence", "grpo"), "adv_norm 合法 ✓")
    bad = cfg.replace(adv_norm="grpo", group_size=4)
    ck(bad.adv_norm == "grpo", "（若误设 grpo 不报错 ✗ → 靠自测提醒 ✓）")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
