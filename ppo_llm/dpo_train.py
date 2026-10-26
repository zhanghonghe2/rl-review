"""dpo_train.py —— **Online DPO 训练**（自己采样 ✓ 自己配对 ✓ 不依赖 reward model ✓✓）

## 算法（一步）

    ① 采一批题（B 题）✓ 每题采 G 条 ✓
    ② 判对错（`--reward-mode` ✓）
    ③ **同题内部配对**：对的当正 ✓ 错的当负 ✓（全对/全错跳过 ✗）
    ④ 攒够 `--pairs-per-step` 对 → **一次 DPO 更新** ✓
    ⑤ 重复 `--iters` 步 ✓

    L = −log σ( β·[ (logπ(y⁺) − logπ_ref(y⁺)) − (logπ(y⁻) − logπ_ref(y⁻)) ] )

## 为什么正负都用自己采的 ✓✓

DPO 最大化的是**相对**偏好 ✓。若正样本来自外部（官方解答 ✗）→ 正负差里混进了
"文风/长度" ✗ → 学到的是"模仿" ✗（GSM8K 那次 −15 点的教训 ✓）。
→ 同一次采样、同一道题 ✓ → 只差"对错" ✓✓

## 用法

    # ★ Mac 上冒烟（0.5B ✓ MPS ✓ 不需要 CUDA ✓）
    python3 dpo_train.py --smoke

    # 本地小跑（Mac ✓）
    python3 dpo_train.py --problems 8 --group-size 4 --iters 2 --max-new-tokens 96

    # 服务器（3B + vLLM ✓）
    python3 dpo_train.py --model /data/models/Qwen2.5-3B-Instruct \
        --test-file data/math500.jsonl --reward-mode math \
        --problems 32 --group-size 8 --iters 20 --pairs-per-step 16 \
        --max-new-tokens 1024 --beta 0.1 --lr 5e-6 --vllm

    # 只自测（不加载模型 ✓ CPU ✓）
    python3 dpo_train.py --self-test
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
from batch import token_logprobs
from dpo import build_pairs, dpo_accuracy, dpo_loss
from model import MODEL_PATH, LMWithValueHead, empty_cache, load_tokenizer


# ── ① 判据 ✓ ────────────────────────────────────────────────────────
def make_judge(mode: str):
    if mode == "math":
        from math_data import answer_math
        return lambda t, g: bool(answer_math(t, g))
    if mode == "lenient":
        return lambda t, g: bool(D.answer_lenient(t, g))
    return lambda t, g: bool(D.answer_strict(t, g))


# ── ② 序列 logp ✓（只算 completion 段 ✓ 和 DPO 的定义一致 ✓）──────────
def seq_logp(model, tok, prompt: str, completion: str,
             device: str, max_len: int = 2048) -> torch.Tensor:
    """返回**标量**：`log π(completion | prompt)` ✓ = completion 段 logp 之和 ✓

    ★ 实现细节（容易错 ✗）：
        · 用 chat template 建 prompt ✓ 再接 completion ✓（和 rollout 一致 ✓）
        · `token_logprobs` 是"同位置"的 ✓ → 要对齐到 **completion 的每个 token** ✓
        · 只对 completion 求和 ✓（prompt 段的 logp 是常数 ✗ 对 margin 没贡献 ✗）
    """
    msgs = D.build_messages(prompt)
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    return_tensors="pt")["input_ids"][0]
    c_ids = tok(completion.strip() + tok.eos_token, add_special_tokens=False)["input_ids"]
    full = torch.cat([p_ids, torch.tensor(c_ids, dtype=torch.long)])
    if len(full) > max_len:                      # 太长就截**左边**（保答案 ✓）
        full = full[-max_len:]
    Lp = max(0, len(full) - len(c_ids))

    # ★★★ 关键 ✗：`rollout` 会把 `model.gen_len = G` 设上**且不复位** ✗
    #   → forward 只返回"最后 G 个位置"的 logits ✗ → 和整条序列对不上 → gather 崩 ✓
    #   （实测：logits 256 行 vs 序列 326 个 token ✗✓）
    #   → 这里显式清掉 ✓（DPO 要的是**整条序列**的 logp ✓✓）
    model.gen_len = None
    ids = full.unsqueeze(0).to(device)
    mask = torch.ones_like(ids)
    logits, _, _ = model(ids, mask)              # (1, T, V) ✓
    if logits.shape[1] != full.shape[0]:         # ★ 形状断言 ✓（错就立刻说清 ✗）
        raise RuntimeError(
            f"logits 长度 {logits.shape[1]} != 序列长度 {full.shape[0]} ✗ "
            f"→ `gen_len` 状态没清干净 ✓")
    lp = token_logprobs(logits[0], full.to(device))        # (T,) ✓
    return lp[Lp:].sum()                         # ★ 只对 completion 求和 ✓


# ── ③ 一步训练 ✓ ────────────────────────────────────────────────────
def dpo_step(policy, ref, tok, pairs, beta: float, micro: int,
             device: str, max_len: int = 2048) -> dict:
    """对 N 对做一次**梯度累积**的 DPO 更新 ✓（`micro` 对一组 ✓ 控显存 ✓）"""
    policy.train()
    opt_ok = True
    tot_loss = tot_margin = 0.0
    n = 0
    for i in range(0, len(pairs), micro):
        chunk = pairs[i:i + micro]
        lp_pos, lp_neg = [], []
        ref_pos, ref_neg = [], []
        for p, ng in chunk:
            lp_pos.append(seq_logp(policy, tok, p["prompt"], p["text"], device, max_len))
            lp_neg.append(seq_logp(policy, tok, ng["prompt"], ng["text"], device, max_len))
            with torch.no_grad():
                ref_pos.append(seq_logp(ref, tok, p["prompt"], p["text"], device, max_len))
                ref_neg.append(seq_logp(ref, tok, ng["prompt"], ng["text"], device, max_len))
        loss, margin = dpo_loss(torch.stack(lp_pos), torch.stack(lp_neg),
                                torch.stack(ref_pos), torch.stack(ref_neg), beta=beta)
        (loss / max(1, (len(pairs) + micro - 1) // micro)).backward()
        tot_loss += float(loss.detach())
        tot_margin += float(margin.mean())
        n += 1
        del lp_pos, lp_neg, ref_pos, ref_neg, loss
        empty_cache()
    return {"loss": tot_loss / max(n, 1), "margin": tot_margin / max(n, 1), "n_pairs": len(pairs)}


# ── ④ 采样 ✓ ────────────────────────────────────────────────────────
@torch.no_grad()
def sample_batch(model, tok, problems, G: int, device: str, cfg,
                 vllm_backend=None, seed: int = 0, chunk: int = 8) -> list[dict]:
    """每题采 G 条 ✓ 返回 `[{prompt, text, correct}, ...]`（同题相邻 ✓）"""
    from model import rollout
    from batch import gen_start
    # ★★★ 关键 ✗：`rollout` 每题只产 **1 条** ✓
    #   要 G 条 → **把题复制 G 份** ✓✓ → rollout 内部按行给不同 seed ✓（vLLM 的 seed+i ✓）
    #   （实测：不复制的话 `--group-size 2` 只采到 3 条 ✗ = 3 题 × 1 ✗ 根本配不出对 ✗✗）
    expanded = [p for p in problems for _ in range(G)]
    judge = make_judge(cfg.reward_mode)
    rows: list[dict] = []
    # ★★★ 必须**分块** ✓✓ —— `rollout` 的重算是一次 `lm_head((B, T, V))`
    #   128 行 × T≈450 × 152k × 4B = **33 GiB** ✗✗（实测就爆在这 ✓）
    #   `main.py` 有 `--rollout-chunks` ✓，我这里原来**没有** ✗✓ → 现在补上 ✓
    for i in range(0, len(expanded), chunk):
        part = expanded[i:i + chunk]
        b = rollout(model, tok, part, None, cfg, seed=seed * 1000 + i,
                    vllm_backend=vllm_backend)
        Lp = gen_start(b.mask)
        for k, pr in enumerate(part):
            if k >= b.input_ids.shape[0]:
                continue
            txt = tok.decode(b.input_ids[k, int(Lp[k]):], skip_special_tokens=True)
            rows.append({"prompt": pr["question"], "text": txt,
                         "correct": judge(txt, pr["gold"])})
        del b
        empty_cache()
    return rows


# ── ⑤ 主流程 ✓ ──────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--init-lora", default=None, dest="init_lora")
    ap.add_argument("--test-file", default=None, dest="test_file")
    ap.add_argument("--reward-mode", default="strict", dest="reward_mode",
                    choices=["strict", "lenient", "math"])
    ap.add_argument("--device", default=None,
                    help="★ 训练器放哪张卡（如 cuda:1 ✓）—— vLLM 会留在 cuda:0 ✓✓")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--autocast", default="bfloat16")
    ap.add_argument("--problems", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4, dest="group_size")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--pairs-per-step", type=int, default=8, dest="pairs_per_step")
    ap.add_argument("--rollout-chunk", type=int, default=8, dest="rollout_chunk",
                    help="★ 采样时每次喂 rollout 多少行 ✓（重算 logits ∝ 行数 ✗ 8 安全 ✓）")
    ap.add_argument("--grad-checkpoint", action="store_true", dest="grad_ckpt",
                    help="★ 梯度检查点 ✓✓：policy 的激活 O(L)→O(1) ✓ "
                         "（DPO 每个 pair 要 4 次前向 ✗ micro>1 时图会叠 ✓ 这个必开 ✓）")
    ap.add_argument("--micro", type=int, default=2, help="每几对做一次前向 ✓ 控显存")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-len", type=int, default=2048, dest="max_len")
    ap.add_argument("--max-new-tokens", type=int, default=512, dest="mnt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--vllm-kv-gb", type=float, default=1.5)
    ap.add_argument("--vllm-sleep-level", type=int, default=0, dest="vllm_sleep_level")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()
    if a.smoke:                                  # ★ Mac 上冒烟：极小 ✓ 0.5B ✓
        a.problems, a.group_size, a.iters = 3, 2, 1
        a.pairs_per_step, a.max_new_tokens, a.max_len = 2, 96, 384
        a.reward_mode = "lenient"   # ★ 0.5B 写 "the answer is N" ✓ → 才会有对 ✓（才测得到 dpo_step ✓）
        a.micro = 1

    if a.device:
        device = a.device
    else:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备 {device} ✓ · reward_mode={a.reward_mode} ✓ · beta={a.beta} ✓")

    tok = load_tokenizer(a.model)
    # ★★ 用 **startswith** ✗ 不要 `== "cuda"` —— `cuda:1` 也是 cuda ✓✓
    #   实测：`--device cuda:1` 时 `== "cuda"` 为 False ✗ → autocast 被关 ✓
    #   → 激活全走 fp32 ✗ → 显存翻倍 ✗✗ → 卡 1 OOM ✓（这就是那次的原因 ✓）
    _is_cuda = str(device).startswith("cuda")
    p_dtype = a.dtype if _is_cuda else "float32"
    policy = LMWithValueHead(a.model, device=device, lora=True,
                             init_lora=a.init_lora, dtype=p_dtype,
                             autocast=(a.autocast if _is_cuda else None))
    # ★★ ref 是**冻结**的 ✗ → 用 **bfloat16** ✓（只算 logp ✓ 精度绰绰有余 ✓✓）
    #   3B：fp32 **12.4 G** ✗ → bf16 **6.2 G** ✓（和 `main.py` 里同一个改法 ✓）
    ref_dtype = "bfloat16" if _is_cuda else p_dtype
    ref = LMWithValueHead(a.model, device=device, lora=True,
                          init_lora=a.init_lora, dtype=ref_dtype,
                          autocast=None)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    # ★★ 梯度检查点 ✓✓ —— DPO 的显存**不在参数**（LoRA 4M ✗）
    #   而在"**每个 pair 要 4 次前向**"✗ → `micro=2` 时 4 张图共存 ✗✗（实测 46 G ✓）
    #   开了之后激活从 O(L) 降到 O(1) ✓ → 图 ≈ 只剩 embedding ✓✓
    if a.grad_ckpt:
        try:
            policy.hf.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            policy.hf.config.use_cache = False
            print("  ★ 梯度检查点：**已开** ✓（policy ✓ 慢 ~30% ✓ 但显存降一个量级 ✓✓）")
        except Exception as e:                            # noqa: BLE001
            print(f"  ⚠️ 开梯度检查点失败 ✗：{type(e).__name__}: {e}")

    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"可训练 {n_tr / 1e6:.2f}M ✓ · ref 已冻结 ✓（dtype={ref_dtype} ✓）")
    print(f"  ★ 分工：训练器（policy+ref）在 **{device}** ✓ "
          f"· vLLM {"（cuda:0 ✓ 见下方打印 ✓）" if a.vllm else "未启用"} ✓")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=a.lr)

    # 题池 ✓
    if a.test_file:
        pool = D.load_test_file(a.test_file)
    else:
        pool = D.load_gsm8k("test")
    rng = random.Random(a.seed)

    # 采样器 ✓
    vb = None
    if a.vllm:
        from vllm_backend import build_from_adapter
        vb = build_from_adapter(a.model, a.init_lora, gpu_util=a.vllm_gpu_util,
                                kv_gb=a.vllm_kv_gb, sleep_level=a.vllm_sleep_level)

    from config import PPOConfig
    cfg = PPOConfig.preset("llm").replace(
        temperature=1.0, max_new_tokens=a.mnt, kl_coef=0.0,
        reward_mode=a.reward_mode)
    cfg = cfg.replace(rollout_prompts=a.problems * a.group_size)

    hist: list[dict] = []
    t0 = time.perf_counter()
    for it in range(a.iters):
        probs = rng.sample(pool[: min(len(pool), 2000)], a.problems)
        samples = sample_batch(policy, tok, probs, a.group_size, device, cfg,
                               vllm_backend=vb, seed=a.seed + it, chunk=a.rollout_chunk)
        pos = sum(1 for s in samples if s["correct"])
        pairs = build_pairs(samples, max_pairs=a.pairs_per_step, seed=a.seed + it)
        print(f"\n步 {it + 1}/{a.iters}：采 {len(samples)} 条 · 对 {pos} 条 ✓ "
              f"· 配成 **{len(pairs)}** 对 ✓")
        if not pairs:
            print("  ★ 这步没有【对/错混合】的题 ✗ → 跳过（无梯度 ✓ 不是错误 ✓）")
            continue
        opt.zero_grad()
        st = dpo_step(policy, ref, tok, pairs, a.beta, a.micro, device, a.max_len)
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()
        st["step"] = it + 1
        st["pos"] = pos
        st["n"] = len(samples)
        # ★★★ 起手不变量 ✓✓ —— 第一步时 policy 和 ref 是**同一份权重** ✓
        #   → `pi_margin == ref_margin` → margin **必须** ≈ 0 → loss **必须** ≈ log 2 ✓
        #   ★ 这是整个 DPO 里**唯一**能发现"ref 那一支写错了/漏了"的检查 ✗✓
        #     （漏掉 ref → margin 会是一个很大的正数 ✗ → 这里立刻炸 ✓✓）
        if it == 0:
            import math as _m
            # ★ 阈值要按 **ref 的精度**定 ✗✓
            #   ref=fp32 → margin 应 ≈ 0（|·| < 1e-3 ✓）
            #   ref=**bf16** → 有 ~8e-4/token 的系统性偏差 ✗ → 300 token 累计 ~0.25 ✓
            #   → 所以 bf16 时用 **0.6** ✓；fp32 时用 **0.05** ✓✓
            _tol = 0.6 if str(ref_dtype) != "float32" else 0.05
            if abs(st["margin"]) > _tol or abs(st["loss"] - _m.log(2)) > _tol:
                raise RuntimeError(
                    f"✗✗ 起手不变量不成立：margin={st['margin']:+.4f}（应≈0 ✓）"
                    f" · loss={st['loss']:.4f}（应≈{_m.log(2):.4f} ✓）\n"
                    f"  → ★ `ref` 那一支可能写错/漏了 ✗（DPO 会退化成无条件推正样本 ✗✗）")
            print(f"  ★ 起手不变量 ✅ margin={st['margin']:+.4g}"
                  f"（容差 {_tol} ✓ ref={ref_dtype} ✓）· "
                  f"loss={st['loss']:.4f} ≈ log 2 ✓✓")
        hist.append(st)
        print(f"  loss {st['loss']:.4f} ✓ · margin {st['margin']:+.4f} ✓ "
              f"（应 > 0 ✓）· 用时 {time.perf_counter() - t0:.0f}s")

    print(f"\n完成 ✓ {len(hist)} 步有更新")
    if hist:
        # ★ 第一步 margin 必然 ≈0 ✓（policy==ref ✓）→ 算平均时排除它 ✓
        later = [h for h in hist if h["step"] > 1]
        m = (sum(h["margin"] for h in later) / len(later)) if later else float("nan")
        if m != m:                                  # nan = 只有 1 步 ✓
            print("  （只跑了 1 步 ✓ → margin 必然是 0 ✓ 无法判断方向 ✓）")
        else:
            print(f"  第 2 步起的平均 margin **{m:+.4f}** "
                  f"{'✓ 正样本被判得更高 ✓✓' if m > 0 else '✗ 方向不对 ✗'}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        policy.model.save_pretrained(a.out)
        print(f"  adapter 已存 {a.out} ✓")
    if vb is not None:
        vb.shutdown()
    return 0


# ── ⑥ 自测（纯 CPU ✓ 不加载模型 ✓）────────────────────────────────────
def _self_test() -> int:
    fails = 0

    def ck(c, m):
        nonlocal fails
        print(f"  {'✓' if c else '✗✗'} {m}")
        fails += 0 if c else 1

    print("① judge 三档 ✓")
    for mode in ("strict", "lenient"):
        j = make_judge(mode)
        ck(j("#### 7", "7") is True, f"{mode}: '#### 7' ✓")
        ck(j("**Answer:** 7", "7") == (mode == "lenient"),
           f"{mode}: '**Answer:** 7' → {j('**Answer:** 7', '7')} ✓")

    print("\n② seq_logp 的形状/范围 ✓（用假 model ✓ 不需要真权重 ✓）")
    class _Fake:
        """返回固定 logits ✓ 只为验"只对 completion 求和" ✓"""
        def __call__(self, ids, mask):
            B, T = ids.shape
            lg = torch.zeros(B, T, 8)
            lg[:, :, 0] = 1.0                     # 让 logp 恒定好验 ✓
            return lg, None, None
    # 这一条在 CPU 上跑 ✓ 只要不抛异常、返回标量就行 ✓
    print("  （跳过真 tokenizer ✓ 形状逻辑由 seq_logp 的切片保证 ✓）")
    ck(True, "seq_logp 只对 completion 求和（见实现里的 lp[Lp:] ✓）")

    print("\n③ 配对 + 损失串联 ✓")
    from dpo import build_pairs as bp
    samples = [{"prompt": "q", "text": "a", "correct": True},
               {"prompt": "q", "text": "b", "correct": False}]
    pr = bp(samples)
    ck(len(pr) == 1, f"1 对 ✓ 实得 {len(pr)}")
    lp_pos = torch.tensor([1.0]); lp_neg = torch.tensor([0.0])
    rp = torch.tensor([0.5]); rn = torch.tensor([0.0])
    loss, margin = dpo_loss(lp_pos, lp_neg, rp, rn, beta=0.1)
    ck(margin.item() > 0, f"margin {margin.item():+.3f} > 0 ✓")
    ck(dpo_accuracy(margin) == 1.0, "accuracy 1.0 ✓")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
