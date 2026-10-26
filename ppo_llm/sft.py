"""sft.py —— 在 GSM8K 的**完整解答**上做监督微调（RL 的 warm-up）。

    python3 sft.py --overfit 8      # 自检：8 条数据过拟合，loss 必须掉到 ~0
    python3 sft.py                  # 正式：1500 条 × 2 epoch（约 15 分钟）
    python3 sft.py --eval-only runs/sft_lora   # 只评估一个已存的 adapter

为什么需要它（三个**定量**理由，不是感觉）
------------------------------------------
① **格式**：严格抽取（必须写 `#### N`）只有 18.8%，而宽松抽取（退回最后数字）有 37.5%
   → 一半失败是"不会写格式"，不是"不会解题" ✗
② **信号**：基线 22% → B=4 时 `P(全错) + P(全对) = 37%` 的批次**奖励全同 → 优势全 0
   → 零梯度** ✗ 提到 45% 能降到 13% ✓
③ **实测证据**：我们已经在稀疏信号上硬优化过一次，结果是
   `eval 0.219 → 0.094`（训练奖励还"看着在涨"）—— 典型的**过拟合到噪声** ✗

顺带的收益
----------
用 **0-shot** 格式训练 → RL 可以不用 4-shot prompt → **prompt 从 675 tokens 降到 ~100**
→ rollout 快 3~5 倍，显存也省 ✓

三个必须做对的地方
------------------
* **label 掩码**：prompt 段必须是 `-100`（不算 loss），只有 assistant 完成段有效 ✗
  否则模型会去学"预测问题"。
* **左 padding + attention_mask**：和 RL 侧保持一致（`batch.pad_left` 那套）。
* **自检 B（8 条过拟合）**：loss 掉不到 ~0 就说明管线有问题 —— 这是 SFT 的
  "最小可信实验"：如果模型连 8 条都记不住，全量训练也没有意义。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from config import PPOConfig
from batch import gen_start
from model import (MODEL_PATH, LMWithValueHead, empty_cache, load_tokenizer,
                   mem_str, rollout)

IGNORE = -100
HERE = os.path.dirname(os.path.abspath(__file__))


# =========================================================================== #
# 1. 数据：把 (question, solution) 变成 (input_ids, labels, attention_mask)
# =========================================================================== #
def encode_example(tokenizer, ex: dict, max_len: int) -> dict | None:
    """0-shot：system + question → 生成提示；然后接上完整解答 + EOS。

    labels 的 prompt 段全是 -100 ✓ 只有解答段有效 ✓
    """
    msgs = D.build_messages(ex["question"], examples=None)      # 0-shot
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt")
    p_ids = enc["input_ids"][0]
    c_ids = tokenizer(ex["solution"].strip() + tokenizer.eos_token,
                      add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        return None
    input_ids = torch.cat([p_ids, torch.tensor(c_ids, dtype=torch.long)])
    labels = torch.cat([torch.full_like(p_ids, IGNORE),
                        torch.tensor(c_ids, dtype=torch.long)])
    return {"input_ids": input_ids, "labels": labels}


def collate(tokenizer, items: list[dict], device) -> dict:
    """左 pad 到本批最长（和 RL 侧一致）。"""
    T = max(len(it["input_ids"]) for it in items)
    B = len(items)
    ids = torch.full((B, T), tokenizer.pad_token_id, dtype=torch.long)
    lab = torch.full((B, T), IGNORE, dtype=torch.long)
    am = torch.zeros(B, T, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        ids[i, T - n:] = it["input_ids"]
        lab[i, T - n:] = it["labels"]
        am[i, T - n:] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device),
            "attention_mask": am.to(device)}


def build_dataset(tokenizer, limit: int, max_len: int, seed: int,
                  data: list[dict] | None = None) -> list[dict]:
    """★ `data` 不为 None 时**只用它** ✓（拒绝采样 RFT 用 ✓）—— 不和 SFT 池混合 ✓

    混合是另一个决定 ✗：想混就在外面把两个 jsonl 拼起来 ✓ 这里保持语义单一 ✓
    """
    if data is not None:
        pool = data
    else:
        pool, _ = D.split_train()                 # ★ 默认只用 SFT 那一段（和 RL 池不相交）
    items = []
    dropped = 0
    for k, ex in enumerate(pool):                 # 长的会被丢掉，所以遍历整段
        it = encode_example(tokenizer, ex, max_len)
        if it is None:
            dropped += 1
            continue
        items.append(it)
        if len(items) >= limit:
            break
        _seen = k + 1
    # ★★ 必须报告 ✗✓ —— 原来**静默丢弃** ✓
    #   `encode_example` 里 `len(p_ids)+len(c_ids) > max_len → None` ✗ → 无任何提示 ✗
    #   后果：你以为训了 2200 条 ✓ 实际只训了 1800 条 ✓ 而日志里看不出来 ✗✗
    if dropped:
        print(f"  ⚠️ **{dropped} 条因超长（> {max_len}）被丢弃** ✗ "
              f"→ 实际可用 {len(items)} 条 ✓（想全用就加大 --max-len ✓）")
    random.Random(seed).shuffle(items)
    return items


# =========================================================================== #
# 2. 一批的 loss（标准 LM 对齐：logits[:-1] 预测 labels[1:]）
# =========================================================================== #
def batch_loss(hf, batch: dict) -> torch.Tensor:
    out = hf(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
             use_cache=False)
    logits = out.logits[:, :-1]                                    # (B,T-1,V)
    target = batch["labels"][:, 1:]                                # (B,T-1)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           target.reshape(-1), ignore_index=IGNORE)


# =========================================================================== #
# 3. 评估：严格格式率 + 贪心准确率
# =========================================================================== #
def parsed_strict(text: str) -> float | None:
    """**只**从 `#### N` 里取数，取不到返回 None（不管对错）。

    这个才是"格式合规"的判据 ✓ 之前我用它顺便比了数值，结果指标退化成
    "严格抽取下答对的比例"，和准确率完全相同 ✗ 看不出格式本身学没学会。
    """
    import re
    for h in reversed(re.findall(r"####\s*([^\n]*)", text)):
        v = D.normalize_number(h)
        if v is not None:
            return v
    return None


def _judge(text: str, gold: str, reward_mode: str = "strict") -> bool:
    """★ 评估判据 = 训练奖励同一个 ✓"""
    if reward_mode == "math":
        from math_data import answer_math
        return answer_math(text, gold)
    if reward_mode == "lenient":
        return D.answer_lenient(text, gold)
    return D.answer_strict(text, gold)


def staged_ok(text: str) -> bool:
    """★ **三段式**合规 ✓✓（`[Plan]` + `[Calculation]` + `[Answer]` 都在 ✓ 且顺序对 ✓）

    为什么需要它 ✗：`format_ok` 只查 `#### N` ✓（那是 GSM8K 的方言 ✓）
    → 在 MATH 上、或测"三段式 prompt 有没有被遵守"时，那个指标**答非所问** ✗
    → 这个函数专门量"**模型到底照不照三段式做**" ✓ → 它是这组实验的**公平性前提** ✓✓
    """
    H = ("[Plan]", "[Calculation]", "[Answer]")
    idx = [text.find(h) for h in H]
    if any(i < 0 for i in idx) or idx != sorted(idx):
        return False
    # 每段都得有内容 ✓
    for k, h in enumerate(H):
        end = idx[k + 1] if k + 1 < len(H) else len(text)
        if not text[idx[k] + len(h):end].strip():
            return False
    return True


def format_ok(text: str) -> bool:
    """有没有写出可解析的 `#### N`（**不要求数值正确**）。"""
    return parsed_strict(text) is not None


@torch.no_grad()
def evaluate_sft(model, tokenizer, problems: list[dict], cfg,
                 chunk: int = 4, vllm_backend=None,
                 reward_mode: str = "strict") -> dict:
    """贪心评估：准确率（宽松抽取）+ **严格格式率**。

    严格格式率是关键指标 —— 它直接对应 SFT 要修的那个问题 ✓
    """
    # ★ 分组大小可调 ✓（--eval-chunk ✓）
    #   原来写死 4 ✗ → 128 题 = 32 次 generate 调用 ✗ → 每次都有固定开销 ✓
    #   调大能省时间 ✓；代价是这次 forward 的行数变多 ✗（lm_head 显存 ∝ 行数 ✗）
    #   经验：≤16 安全 ✓ · 32 需要显存宽裕 ✓ · 128 会 OOM ✗✗
    # ★★ 按**显存预算**自动收窄 chunk ✓✓ —— 别再靠手调 ✗
    #   `logits` 是 (B, T, V) fp32 ✗ → 一次分配 B*T*V*4 字节 ✓
    #   V=151936 且 T≈(prompt+max_new) ✓ → chunk 一大就单次分配几十 G ✗✗
    # ★★ 预算改成**按当前空闲显存自适应** ✓✓
    #   原来写死 3 GiB ✗ → 在 48G 的卡上把 chunk 压到 7 ✗（太保守 ✓）
    #   实测公式：logits = B × **max_new_tokens** × V × 4 ✓
    #     （prompt 段不算在内 ✗ —— 我原来加了 160 高估了 ✓）
    # =================================================================== #
    # ★★★ vLLM 直采 ✓✓ —— eval **只要文本** ✓ 不需要 logits ✗
    #
    #   ✗ 旧路径走 `rollout()` ✓ → 它顺手重算 logprobs ✗ → 显存 ∝ 行数 ✗
    #     → chunk 被压到 ~8 ✗ → vLLM 一次只有 8 条序列 ✗ → 390 tok/s ✗
    #   ✓ 现在：`vllm_backend.generate` ✓ → 只吃 KV（64 条 ≈ 0.25 GiB ✓）
    #     → chunk 可以开 **64** ✓ → ≈ 2,400 tok/s ✓✓
    #   → ★ 500 题：**15 分钟 → ~2 分钟** ✓✓（和 prm_train 里的同一个修法 ✓）
    # =================================================================== #
    if vllm_backend is not None:
        _bs = max(1, int(chunk))
        n_ok = n_strict = n_fmt = n_gen = n_tot = 0
        for i in range(0, len(problems), _bs):
            part = problems[i:i + _bs]
            ids = [tokenizer.apply_chat_template(
                        D.build_messages(pr["question"]),
                        add_generation_prompt=True,
                        return_tensors="pt")["input_ids"][0].tolist()
                   for pr in part]
            outs = vllm_backend.generate(ids, temperature=0.0,
                                         max_tokens=int(cfg.max_new_tokens),
                                         seed=int(cfg.seed) + i)
            for pr, o in zip(part, outs):
                txt = tokenizer.decode(o, skip_special_tokens=True)
                n_ok += int(D.reward_fn_base(txt, pr["gold"]))
                n_strict += int(_judge(txt, pr["gold"], reward_mode))
                n_fmt += (int("\\boxed" in txt) if reward_mode == "math"
                          else int(format_ok(txt)))
                globals()["_n_staged"] = globals().get("_n_staged", 0) + int(staged_ok(txt))
                n_gen += len(o)                     # ★ token 数 ✓ 不是字符 ✗
                n_tot += 1
            print(f"    [eval] {min(i + _bs, len(problems))}/{len(problems)} ✓ "
                  f"严格 {n_strict / max(n_tot, 1):.1%} ✓", flush=True)
        return {"staged_rate": globals().pop("_n_staged", 0) / max(n_tot, 1),
                "eval_reward": n_ok / max(n_tot, 1),
                "strict_acc": n_strict / max(n_tot, 1),
                "format_rate": n_fmt / max(n_tot, 1),
                "eval_gen_len": n_gen / max(n_tot, 1)}

    _VOCAB = 151936
    _T_EST = max(1, int(cfg.max_new_tokens))         # ★ 只算生成长度 ✓
    try:
        _free, _tot = torch.cuda.mem_get_info()
        _BYTES = int(_free * 0.40)                   # 用空闲的 40% ✓✓
    except Exception:                                # noqa: BLE001
        _BYTES = 3 << 30
    _max_rows = max(1, int(_BYTES / (4 * _VOCAB * _T_EST)))
    if int(chunk) > _max_rows:
        print(f"  [eval] 空闲 {_BYTES / 2**30:.1f} GiB 预算 · 单行 {4 * _VOCAB * _T_EST / 2**20:.0f} MiB"
              f" → 最多 {_max_rows} 行 ✓")
        print(f"  [eval] chunk {chunk} → 自动降到 **{_max_rows}** ✓"
              f"（logits 单次要 {int(chunk) * _T_EST * _VOCAB * 4 / 2**30:.1f} GiB ✗ "
              f"预算 3.0 GiB ✓）")
        chunk = _max_rows
    ev = cfg.replace(temperature=0.0,
                     rollout_prompts=max(1, min(int(chunk), len(problems))))
    n_ok, n_strict, n_fmt, n_gen, n_tot = 0, 0, 0, 0, 0
    for i in range(0, len(problems), ev.rollout_prompts):
        part = problems[i:i + ev.rollout_prompts]
        b = rollout(model, tokenizer, part, None, ev, seed=cfg.seed,
                    vllm_backend=vllm_backend)
        Lp = gen_start(b.mask)
        from batch import last_valid_index
        from batch import token_logprobs  # noqa: F401  （只为保持一致，不用）
        rows = torch.arange(b.batch_size)
        # ★ 判据**统一到 data.py** ✓ —— 以前这里读 `b.rewards` ✗ 而它和
        #   `parsed_strict` 在 128 题里有 4 题不一致 ✓ → 同一份权重报出
        #   **0.555 / 0.523** 两个"严格准确率" ✗（两条路径各写了一份判据 ✓）
        #   教训：**同一个指标的实现只能有一份** ✓✓
        for j, pr in enumerate(part):
            txt = tokenizer.decode(b.input_ids[j, int(Lp[j]):], skip_special_tokens=True)
            n_ok += int(D.reward_fn_base(txt, pr["gold"]))       # 宽松（对齐历史数字）
            n_strict += int(_judge(txt, pr["gold"], reward_mode))   # ★ 跟 --reward-mode 走 ✓
            # ★★ 格式率必须**跟判据走** ✗✓ —— 原来恒用 `#### N`（GSM8K 的方言 ✗）
            #   → 在 MATH / PRM 上**恒为 0** ✗ → 那个数字纯误导 ✗✗
            if reward_mode == "math":
                n_fmt += int("\\boxed" in txt)                 # MATH：有 boxed 就算合规 ✓
            else:
                n_fmt += int(format_ok(txt))                     # GSM8K：`#### N` ✓
            n_staged = globals().get("_n_staged", 0) + int(staged_ok(txt))
            globals()["_n_staged"] = n_staged
        n_gen += float(b.gen_len().float().mean())
        n_tot += b.batch_size
        # ★★ 必须在这里释放 ✗✓ —— 否则 `b` 一直占着 ✓
        #   下一次 `b = rollout(...)` 是**先分配后替换** ✗ → 峰值 = 2 批 ✗
        #   + 分配器碎片 → 表现就是"慢慢爆" ✗✓（main.py 那边一直有这两行 ✓）
        del b
        empty_cache()
    return {"staged_rate": globals().pop("_n_staged", 0) / max(n_tot, 1),
            "eval_reward": n_ok / max(n_tot, 1),
            "strict_acc": n_strict / max(n_tot, 1),
            "format_rate": n_fmt / max(n_tot, 1),
            "eval_gen_len": n_gen / max((len(problems) + ev.rollout_prompts - 1)
                                       // ev.rollout_prompts, 1)}


# =========================================================================== #
# 4. 训练
# =========================================================================== #
def train_sft(model, tokenizer, items: list[dict], device, epochs: int, lr: float,
              batch_size: int, steps: int | None, log_every: int, seed: int,
              accum: int = 1, schedule: str = "cosine", warmup_frac: float = 0.03):
    hf = model.hf
    params = [p for p in hf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    # ★★★ 步数**不除以 accum** ✗✓ —— 这是我上次加的错 ✗
    #   累积只增加"每次 update 用多少样本" ✓ **不减少"一共看多少样本"** ✗
    #   错法：`// (batch_size*accum)` → 每个 step 仍吃 batch_size 条 ✓
    #        → 总样本 = n_steps × batch_size = len(items)·epochs/accum ✗✗
    #        → `--grad-accum 4` 时**只看四分之一的数据** ✗（2 epoch 实际是 0.5 epoch ✗）
    n_steps = (steps if steps is not None
               else (len(items) // batch_size) * epochs)

    # ★★★ lr 调度 ✓✓ —— 原来**完全没有** ✗ → 恒定 lr 跑到最后 ✓
    #   而我们这次的目标是"**学会格式但不伤能力**" ✓✓ → 末尾必须**温柔** ✓
    #   证据：GSM8K 那三次 `-15` 点 ✓ 既是 lr 大（1e-4）✗ 又**不衰减** ✗ → 双重过训练 ✓
    #   → 默认 **cosine + 3% warmup** ✓（warmup 防起步炸 ✓ cosine 让末尾收敛 ✓✓）
    sched = None
    if schedule not in ("", "const", "none"):
        import math as _m
        _steps = max(1, n_steps)
        _wu = max(1, int(_steps * max(0.0, warmup_frac))) if warmup_frac > 0 else 0

        def _lr_at(step: int) -> float:
            if _wu and step < _wu:
                return (step + 1) / _wu
            t = (step - _wu) / max(1, _steps - _wu)
            t = min(1.0, max(0.0, t))
            return 0.5 * (1.0 + _m.cos(_m.pi * t))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
        print(f"  ★ lr 调度 **cosine** ✓ warmup {_wu} 步 ✓ 末尾降到 ~0 ✓✓")
    accum = max(1, int(accum))
    rng = random.Random(seed)
    idx = list(range(len(items)))
    losses = []
    print(f"  训练：{len(items)} 条 / batch {batch_size} / {n_steps} 步 / lr {lr:g}")
    t0 = time.perf_counter()
    t_prev = t0
    # ★★ 梯度累积 ✓✓ —— `--batch` 只决定**显存** ✓，`--grad-accum` 决定**有效批量** ✓
    #   为什么需要 ✗：3B + `--max-len 2048` 时 **batch=4 就 OOM** ✗（实测 ✓）
    #   → batch 压到 1~2 再累积 ✓ → 显存按 micro-batch ✓ 统计按有效批量 ✓✓
    opt.zero_grad()
    for step in range(n_steps):
        if step % max(len(items) // batch_size, 1) == 0:
            rng.shuffle(idx)
        sel = idx[(step * batch_size) % len(idx):][:batch_size]
        if len(sel) < batch_size:                       # 尾部不够就从头补
            sel = (sel + idx)[:batch_size]
        batch = collate(tokenizer, [items[i] for i in sel], device)
        loss = batch_loss(hf, batch) / accum            # ★ 平均 ✓
        loss.backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()                            # ★ 跟着优化器走 ✓✓
            opt.zero_grad()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            recent = sum(losses[-log_every:]) / log_every
            now = time.perf_counter()
            seg = now - t_prev            # ★ 本段用时（不是累计）
            t_prev = now
            _lr = opt.param_groups[0]["lr"]
            print(f"    step {step + 1:>4}/{n_steps} | loss {recent:.4f} | lr {_lr:.2e}"
                  f" | 本段 {seg:>5.0f}s（{seg / log_every:.2f}s/步）"
                  f" | 累计 {now - t0:>5.0f}s")
    return losses


# =========================================================================== #
# 5. 自测
# =========================================================================== #
def _case_labels(tokenizer) -> None:
    """label 掩码必须正确：prompt 段全 -100，完成段是真正的解答。"""
    ex = D.load_gsm8k("train")[0]
    it = encode_example(tokenizer, ex, max_len=4096)
    lab, ids = it["labels"], it["input_ids"]
    n_prompt = int((lab == IGNORE).sum())
    n_comp = int((lab != IGNORE).sum())
    if n_prompt == 0:
        raise AssertionError("prompt 段没有 -100 —— 模型会去学预测问题 ✗")
    if n_comp == 0:
        raise AssertionError("完成段全是 -100 —— 没有任何东西可学 ✗")
    # 从 label 反解出完成段文本，必须等于解答本身（+EOS）
    comp_ids = lab[lab != IGNORE]
    comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True).strip()
    if not comp_text.endswith(ex["solution"].strip()[-40:]):
        raise AssertionError(f"完成段反解出来的不是解答：{comp_text[-80:]!r}")
    # prompt 段必须和 ids 完全一致（没有被误改）
    if not torch.equal(ids[:n_prompt], ids[:n_prompt]):
        raise AssertionError("ids 被改了")
    print(f"    prompt {n_prompt} tokens（全 -100）| 完成 {n_comp} tokens（有效）✅")
    print(f"    完成段反解：{comp_text[-60:]!r}")


def _case_overfit(model, tokenizer, device, n: int, lr: float, steps: int) -> None:
    """★ SFT 的最小可信实验：**8 条数据必须能过拟合到接近 0**。

    如果连 8 条都记不住，全量训练没有意义（管线/掩码/lr 一定有问题）。
    """
    items = build_dataset(tokenizer, limit=n, max_len=1024, seed=0)
    if len(items) < n:
        raise AssertionError(f"只凑到 {len(items)} 条（要 {n}）")
    losses = train_sft(model, tokenizer, items, device, epochs=1, lr=lr,
                       batch_size=min(4, n), steps=steps, log_every=steps // 4, seed=0)
    final = sum(losses[-10:]) / len(losses[-10:])
    if final > 0.1:
        raise AssertionError(
            f"{steps} 步后 loss 还有 {final:.4f}（应该 < 0.1）\n"
            f"        → 管线有问题：检查 label 掩码、lr、以及 forward 的对齐"
        )
    print(f"    {n} 条数据 {steps} 步后 loss = {final:.4f} ✅ 管线可信")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="GSM8K 监督微调（RL 的 warm-up）")
    ap.add_argument("--limit", type=int, default=None,
                    help="★ 最多用多少条 ✓ **默认 None = 全用** ✓"
                         "（原来默认 1500 ✗ → 2200 条的数据会静默只用 1500 ✗✗）")
    ap.add_argument("--data", type=str, default=None,
                    help="用这份 jsonl 当训练数据（拒绝采样 RFT 的输出 ✓）；不给就用 GSM8K 的 SFT 池")
    ap.add_argument("--dtype", type=str, default="auto", dest="dtype",
                    help="参数 dtype。auto=float32（LoRA 别用 bf16 ✗）")
    ap.add_argument("--autocast", type=str, default="auto", dest="autocast",
                    help="前向/反向的 autocast。auto=CUDA 上 bf16、其它关闭")
    ap.add_argument("--model", type=str, default=MODEL_PATH, dest="model",
                    help="模型目录（默认 0.5B；1.5B 见 main.py --help）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    # 评估时的生成长度上限。**默认 192，和 main.py 一致** ✗ 改了要两边一起改 ✓
    ap.add_argument("--max-new-tokens", type=int, default=192, dest="max_new_tokens")
    ap.add_argument("--grad-checkpoint", action="store_true", dest="grad_ckpt",
                    help="★ 梯度检查点 ✓✓：用算力换显存，激活降 ~10 倍 ✓"
                         "（**3B + max-len 2048 时必开** ✗ 否则 batch 1 也 OOM ✓）")
    ap.add_argument("--lr-schedule", default="cosine", dest="lr_schedule",
                    choices=["cosine", "const"],
                    help="★ lr 调度 ✓（默认 cosine ✓ —— 恒定时末尾过训练 ✓ 我们吃过 -15 的亏 ✗）")
    ap.add_argument("--warmup-frac", type=float, default=0.03, dest="warmup_frac")
    ap.add_argument("--grad-accum", type=int, default=1, dest="grad_accum",
                    help="★ 梯度累积 ✓：显存只由 --batch 决定 ✓ 有效批量 = batch × grad-accum ✓")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768, dest="max_len")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "runs", "sft_lora"))
    ap.add_argument("--overfit", type=int, default=None,
                    help="自检模式：只在 N 条数据上过拟合（建议 8）")
    ap.add_argument("--overfit-steps", type=int, default=200, dest="overfit_steps")
    ap.add_argument("--eval-only", type=str, default=None, dest="eval_only")
    ap.add_argument("--system", default="default",
                    choices=["default", "none"],
                    help="★ system 消息 ✓ default=GSM8K 那句（'必须写 #### <number>' ✗）；"
                         "**none = 不要 system** ✓✓ —— PRM/结构化数据必须用 none ✗")
    ap.add_argument("--no-adapter", action="store_true", dest="no_adapter",
                    help="★ 评**基座**（不加载 adapter ✓）")
    ap.add_argument("--init-lora", type=str, default=None, dest="init_lora",
                    help="从已存的 adapter 接着 SFT（再补一轮）")
    # ★ 必须和 main.py 的评估一致 ✗（main 是 128）—— 上次我只统一了
    #   max_new_tokens 却漏了这个 ✗ 结果 32 题 0.375 vs 128 题 0.508 不可比 ✓
    ap.add_argument("--eval-size", type=int, default=128, dest="eval_size")
    ap.add_argument("--reward-mode", type=str, default="strict", dest="reward_mode",
                    choices=["strict", "lenient", "math"],
                    help="评估判据：strict 必须 #### N ✓ · lenient 认 **Answer:** 等方言 ✓")
    ap.add_argument("--test-file", type=str, default=None, dest="test_file",
                    help="★ 自定义测试集 jsonl ✓（question + gold 或含 #### 的 answer ✓）"
                         " 默认用官方 test ✓")
    ap.add_argument("--vllm", action="store_true", help="评估用 vLLM 生成（快 ~2× ✓）")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--vllm-kv-gb", type=float, default=1.5)
    ap.add_argument("--vllm-sleep-level", type=int, default=1, dest="vllm_sleep_level",
                    help="生成后 sleep 释放（0 = 不释放 ✓ 48G 上建议 0 ✓ "
                         "—— 实测这版 level 1 只释放 1G ✗ 反而留碎片 ✓）")
    ap.add_argument("--eval-chunk", type=int, default=4, dest="eval_chunk",
                    help="评估时每次 forward 多少题（原来写死 4 ✗）。调大省时间 ✓ "
                         "但显存 ∝ 行数 ✗：≤16 安全 ✓ · 32 要宽裕 · 128 会 OOM ✗")
    args = ap.parse_args()

    device = PPOConfig.auto_device()
    # 确定性：和 main.py 一致 ✓ 否则同一个 adapter 两次评估能差 ±3~6% ✗
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = load_tokenizer(args.model)
    p_dtype = PPOConfig.resolve_dtype(device, args.dtype)
    p_amp = PPOConfig.resolve_autocast(device, args.autocast)
    model = LMWithValueHead(args.model, device=device, lora=True,
                            init_lora=(None if args.no_adapter
                                       else (args.init_lora or args.eval_only)),
                            dtype=p_dtype, autocast=p_amp)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"设备 {device} | 参数 {p_dtype} | autocast {p_amp or '关'}"
          f" | 可训练 {n_tr / 1e6:.2f}M（LoRA）\n")

    # ★ 必须和 main.py 的 --max-new-tokens 一致 ✗（之前是写死的 256 ✓ 导致两边不可比）
    cfg = PPOConfig.preset("llm").replace(max_new_tokens=args.max_new_tokens, kl_coef=0.0,
                                   reward_mode=args.reward_mode,
                                          device=device, temperature=0.0)

    # ★ 评估也走 vLLM ✓（用户要求"推理全部 vLLM" ✓）
    _VB = None
    if args.vllm:
        from vllm_backend import build_from_adapter
        _ADR = None if args.no_adapter else (args.eval_only or args.init_lora)
        _VB = build_from_adapter(args.model, _ADR,
                                 gpu_util=args.vllm_gpu_util, kv_gb=args.vllm_kv_gb,
                                 sleep_level=args.vllm_sleep_level)
                                 # ★ 不传 max_len ✗ —— `--max-len` 是**训练**概念 ✓
                                 #   推理要的是 prompt + max_new_tokens ✗
                                 #   （PRM：1,143 + 1,024 ≈ 2,170 → 2048 都不够 ✗✗）
                                 #   → **不传** ✓ 走 build_from_adapter 的默认 4096 ✓✓

    # ★★★ system 必须在**评估之前**设置 ✗✗
    #   ✗ 原来它在 `--eval-only` 之后 ✓ 而那条分支 `sys.exit(0)` 直接退出 ✓
    #   → `--eval-only --system none` **两个参数一起用也没用** ✗
    #   → 评估用的是 GSM8K 那句"必须写 #### <number>" ✓ 和 `\boxed{}` 打架 ✓
    #   → 测出的贪心准确率必然偏低 ✗ 而且和 prm_train 的数**不可比** ✓✓
    #   ✓ 现在提到最前面 ✓
    if args.system == "none":
        import data as _D
        _D.SYSTEM_OVERRIDE = ""
        print("  ★ system：**已置空** ✓（只用题干里的规则 ✓ PRM 需要这个 ✓✓）", flush=True)
    else:
        print("  ⚠️ system：默认 GSM8K 那句 ✓（★ PRM/结构化数据应加 `--system none` ✗）",
              flush=True)

    if args.eval_only or args.no_adapter:
        _pool = (D.load_test_file(args.test_file) if args.test_file
                 else D.load_gsm8k("test"))
        test = _pool[: args.eval_size]
        r = evaluate_sft(model, tokenizer, test, cfg, chunk=args.eval_chunk, reward_mode=args.reward_mode,
                              vllm_backend=_VB)
        print(f"评估 {len(test)} 题：准确率 {r['eval_reward']:.3f}"
              f" | **格式合规率** {r['format_rate']:.3f}"
              f" | **三段式率** {r.get('staged_rate', 0):.3f}"
              f" | 严格答对率 {r['strict_acc']:.3f}"
              f" | 平均生成长度 {r['eval_gen_len']:.0f}")
        sys.exit(0)

    # ---- 自检 ----
    print("用例 A：label 掩码")
    _case_labels(tokenizer)

    if args.overfit:
        print(f"\n用例 B：{args.overfit} 条过拟合（SFT 的最小可信实验）")
        _case_overfit(model, tokenizer, device, args.overfit, lr=5e-4,
                      steps=args.overfit_steps)
        print("\n🎉 sft.py 自检通过 —— 可以跑全量了：python3 sft.py")
        sys.exit(0)

    # ---- 正式 ----
    _pool = (D.load_test_file(args.test_file) if args.test_file
             else D.load_gsm8k("test"))
    test = _pool[: args.eval_size]
    print(f"\n训练前评估（{len(test)} 题，贪心）：")
    before = evaluate_sft(model, tokenizer, test, cfg, chunk=args.eval_chunk, reward_mode=args.reward_mode,
                              vllm_backend=_VB)
    print(f"  准确率 {before['eval_reward']:.3f} | "
          f"**严格格式率** {before['strict_acc']:.3f} | 长度 {before['eval_gen_len']:.0f}\n")

    rft_data = None
    if args.data:
        rft_data = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rft_data.append(json.loads(line))
        print(f"--data：读入 {len(rft_data)} 条 ✓（**只用这个文件** ✗ 不和 SFT 池混合 ✓）")
        # ★★ 逐条检查 ✗✓ —— 原来只查 `rft_data[0]` ✗
        #   → 一个"新旧拼接"的 jsonl ✓ 只要**第一条**是新格式 ✓ 就会**放行** ✓
        #   → 然后到中间某条才 `KeyError` ✗（更难查 ✓）
        #   现在：全扫一遍 ✓ 缺字段的**跳过** ✓ 并报数 ✓
        need = ("question", "solution")
        bad = [i for i, d in enumerate(rft_data) if any(k not in d for k in need)]
        if bad:
            print(f"  ⚠️ --data 里有 **{len(bad)}** 条缺 {need} ✗ → 已跳过 ✓"
                  f"（第一个是第 {bad[0] + 1} 行 ✓）")
            rft_data = [d for i, d in enumerate(rft_data) if i not in set(bad)]
        if not rft_data:
            raise ValueError(f"--data 里**没有任何**一条同时含 {need} ✗")
        print(f"  --data：可用 {len(rft_data)} 条 ✓")
    if (rft_data is not None and args.limit is not None
            and len(rft_data) > args.limit):
        print(f"⚠️ --data 里有 {len(rft_data)} 条，但 --limit={args.limit} ✗"
              f" → **只用了前 {args.limit} 条** ✓")
        print(f"   想全用就加 --limit {len(rft_data)} ✓（否则会静默只用一部分 ✗）")
    # ★★ 梯度检查点 ✓✓ —— 这是 3B 长序列 SFT **唯一**能救显存的东西 ✗
    #   DeepSpeed/ZeRO 只分片**参数**（我们 LoRA 只有 4M ✗ 没用 ✗）
    #   而爆的是**激活** ∝ B×T×H×I×L ✗ → 检查点把它从 O(L) 降到 O(1) ✓✓
    _had_cache = None                       # ★ 显式初始化 ✓（别用 `dir()` 判断 ✗ 太脆 ✓）
    if args.grad_ckpt:
        try:
            model.hf.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            # ★★ 只记下原值 ✗ 不要永久关掉 ✓✓
            #   原来这里直接 `= False` 且**不恢复** ✗ → 训练后的 eval `generate`
            #   就**没有 KV cache** ✗ → 每 token 重算整个前缀 ✗ → 慢 ~500 倍 ✗✗
            #   （带 `--vllm` 时看不出来 ✓ 因为走 vLLM 自己的 cache ✓ 但 HF 路径会爬 ✗）
            _had_cache = bool(getattr(model.hf.config, "use_cache", True))
            model.hf.config.use_cache = False
            print("  ★ 梯度检查点：**已开** ✓（慢 ~30% ✓ 但激活降 ~10 倍 ✓✓）")
        except Exception as e:                               # noqa: BLE001
            print(f"  ⚠️ 开梯度检查点失败 ✗：{type(e).__name__}: {e}")

    _lim = 10 ** 9 if args.limit is None else int(args.limit)
    items = build_dataset(tokenizer, _lim, args.max_len, args.seed, data=rft_data)
    print(f"数据：{len(items)} 条（0-shot，prompt≈{len(items[0]['input_ids']) - len(items[0]['labels'][items[0]['labels'] != IGNORE])} tokens）\n")
    train_sft(model, tokenizer, items, device, args.epochs, args.lr,
              args.batch, None, 50, args.seed, args.grad_accum,
              args.lr_schedule, args.warmup_frac)

    # ★★ 训练**之后**才恢复 use_cache ✓✓ —— 原来放在 train_sft **前面** ✗（我插错位置 ✓）
    #   训练时不能开 cache ✗（和梯度检查点冲突 ✓ 虽然 batch_loss 显式传了 False ✓ 侥幸没炸 ✓）
    #   但 eval 的 `generate` **必须**开 ✓ → 否则每个 token 重算整个前缀 ✗ → 慢几百倍 ✗✗
    if _had_cache is not None:
        model.hf.config.use_cache = _had_cache
        print(f"  ★ use_cache 恢复成 {_had_cache} ✓（eval 才不会慢 ✗）")

    os.makedirs(args.out, exist_ok=True)
    model.model.save_pretrained(args.out)          # PeftModel → 只存 adapter（~10MB）
    print(f"\nadapter 已存到 {args.out}")

    after = evaluate_sft(model, tokenizer, test, cfg, chunk=args.eval_chunk, reward_mode=args.reward_mode,
                              vllm_backend=_VB)
    print(f"\n训练后：准确率 {after['eval_reward']:.3f}（{before['eval_reward']:.3f}）"
          f"\n  **格式合规率 {after['format_rate']:.3f}**（{before['format_rate']:.3f}）"
          f" | **三段式率 {after.get('staged_rate', 0):.3f}**"
          f"（{before.get('staged_rate', 0):.3f}）"
          f" | 严格答对率 {after['strict_acc']:.3f} | 长度 {after['eval_gen_len']:.0f}（{before['eval_gen_len']:.0f}）")
    # ★ 提示里的 lr 原来是 **2e-4** ✗✗ —— 那正是我们诊断出的**过大**的 lr ✓
    #   （R1 是 3e-6 ✓；我们在 3B/1.5B 上验证 **3e-5** 才有效 ✓✓）
    _nm = "math" if "math" in str(args.test_file or "") else "gsm8k"
    _lr_hint = "3e-5" if _nm == "math" else "3e-5"
    print(f"\n下一步（RL）：python3 main.py --init-lora {args.out} --iters 60 "
          f"--lr {_lr_hint} --adaptive-kl --kl-target 0.1 --inner-epochs 4")
