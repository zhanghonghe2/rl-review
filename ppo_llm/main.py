"""main.py —— 把 数据 / 模型 / rollout / ppo_update 串成一个完整实验。

    python3 main.py --smoke     # 3 步冒烟测试（**先跑这个**）
    python3 main.py             # 正式跑

⚠️ MPS（Apple GPU）上建议这样跑 —— **不加的话会被系统换页拖慢 20 倍**：
    PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2 python3 main.py

    原因见 train() 里那段注释：分配器默认允许涨到「建议工作集 × 1.7 ≈ 42GB」，
    而机器只有 32GB —— 实测 driver 足迹冲到 40GB 后，单步更新从 9.6s 变成 247s。

冒烟测试要确认三件事（跑完会逐条打印结论）：

    ① **ratio 起手为 1**  —— 采样分布 == 重算分布。不一致的话 PPO 从第一步就在瞎裁，
                            而 loss 照样会降。`ppo_update` 里的断言也在管这件事。
    ② **奖励不全 0**      —— 全 0 的话优势恒为 0，梯度恒为 0（`gae.py` 用例 E 那个场景）。
    ③ **显存/时间可接受**  —— 它决定正式跑敢开多大（B / max_new_tokens）。

正式跑之前，先把这三条跑绿。这是我们这一路反复验证过的顺序：**先用最小配置验证契约，
再上规模**。
"""

# PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2 python3 main.py


from __future__ import annotations

import argparse
import os
import random
import time

import torch

import data as D
from batch import concat_batches, gen_start, last_valid_index
from config import PPOConfig
from loss import assert_ratio_starts_at_one
from metrics import MetricLogger
from model import (MODEL_PATH, LMWithValueHead, empty_cache, load_tokenizer,
                   make_optimizer, mem_str, rollout)
from update import (_logprobs_new, check_model, ppo_update, smooth_kl,
                    update_kl_coef)

EPILOG = """常用命令（不加参数 = 正式配置）：

  python3 main.py                      # 正式跑：B=4, G=160, 100 步, inner=2
  python3 main.py --iters 20           # 先跑 20 步看显存/梯度趋势（推荐第一次这样）
  python3 main.py --batch 2            # 省显存；但"奖励全同"的步数从 17% 升到 53%
  python3 main.py --max-new-tokens 128 # 省显存/时间；答案更容易被截断
  python3 main.py --eval-every 20      # 少评估几次（每次评估要跑完整个评估集）
  python3 main.py --tag try_lr5e-6 --lr 5e-6   # 换学习率做对照
  python3 main.py --smoke              # 3 步冒烟

⚠️ MPS 上建议带上这两个环境变量（否则可能被系统换页拖慢 20 倍）：
  PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2
"""


# =========================================================================== #
# 1. 配置
# =========================================================================== #
def make_cfg(args) -> PPOConfig:
    """低优先级：预设（冒烟 / 正式）→ 高优先级：命令行显式给的参数。

    **只覆盖你显式传的那些** —— 所以 `python3 main.py` 就是一套调好的默认正式配置，
    而 `--iters 20` 只改步数，别的照旧。
    """
    smoke = args.smoke
    cfg = PPOConfig.preset("llm").replace(
        # ★ KL 惩罚开 0.02（预设原值）。实测过的失败模式：kl_coef=0 时
        #   策略在稀疏奖励上漂走，eval 0.219 → 0.094，而训练奖励还"看着在涨" ✗
        #   代价：多一份冻结的参考模型（+2GB）。LoRA 省了 9.7GB，放得下 ✓
        kl_coef=0.02,
        log_every=1,
        tag="llm_smoke" if smoke else "llm_gsm8k",
        rollout_prompts=4,                    # 每块的 B（有效批量 = 这个 × rollout_chunks）
        # ★ 192：**必须和 sft.py 的评估用同一个值** ✗
        #   （之前 main 用 192、sft.py 用 256 → 同一个模型 0.492 vs 0.508 ✗ 不可比）
        max_new_tokens=64 if smoke else 192,
        total_iters=3 if smoke else 100,
        inner_epochs=1 if smoke else 2,
        lr=1e-4,          # LoRA 用 1e-4 起（全参才用 1e-6）
        # ★ value_coef 从默认 0.5 降到 0.05。实测（lr=5e-4, B=4/G=96）：
        #   value_coef=0.5 + head_lr=10× → value项/policy项 = 23.1x，value_loss = 5.98
        #   value_coef=0.05 + head_lr=1×  →                0.2x，          value_loss = 0.06
        # 共享主干下 value 项的梯度会流进主干；它比 policy 项大 23 倍时，
        # 全局 clip 之后 policy 只占有效步长的 ~4%，等于没在学。
        # ★ 单轮生成任务：奖励只在最后一个 token，gamma 必须 1.0 ✗
        #   （0.99 会让终局奖励按长度打折 → 越短有效奖励越大 ✗）
        gamma=1.0,
        value_coef=0.05,
        # ★ 过程奖励开 0.5。实测纯结果奖励（0）会让生成长度 131→85、
        #   准确率 0.266→0.078 塌掉 ✗；而 SFT 模型 100% 会写
        #   <<48+24=72>> 这类标注（每题 3.4 个，86.8% 算对）——
        #   把它们验证了，就等于给每条轨迹 3~4 个稠密信号，
        #   而且「输出变短」会因为算式变少而**被惩罚** ✓
        process_coef=0.5,                              # 主干；value_head 走 10×（make_optimizer）
    )
    # 命令行的覆盖（None = 没传 = 不动预设）
    over = {k: v for k, v in {
        "rollout_prompts": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "total_iters": args.iters,
        "inner_epochs": args.inner_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "tag": args.tag,
        "grad_clip": args.grad_clip,
        "clip_eps": args.clip_eps,
        "temperature": args.temperature,
        "rollout_chunks": args.rollout_chunks,
        "num_minibatches": args.num_minibatches,
        "adv_norm": args.adv_norm,
        "rollout_backend": args.rollout_backend,
        "vllm_gpu_util": args.vllm_gpu_util,
        "vllm_check_every": args.vllm_check_every,
        "vllm_kv_gb": args.vllm_kv_gb,
        "vllm_sleep_level": args.vllm_sleep_level,
        "vllm_eval_chunk": args.vllm_eval_chunk,
        "vllm_dtype": args.vllm_dtype,
        "group_size": args.group_size,
        "adaptive_kl": args.adaptive_kl,
        "kl_target": args.kl_target,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "kl_coef": args.kl_coef,
        "device": args.device,
        "process_coef": args.process_coef,
        "reward_mode": args.reward_mode,
    }.items() if v is not None}
    return cfg.replace(**over) if over else cfg


# =========================================================================== #
# 2. 数据
# =========================================================================== #
def build_data(n_shot: int = 4, pool_file: str | None = None,
               pool_skip: int = 0, test_file: str | None = None):
    """返回 `(训练池, 测试集, few-shot 示例)`。

    ⚠️ few-shot 必须从 **train** 里取 —— 从 test 取会让基线被污染（等于抄答案）。
       训练池要**排除**当示例用的那几条，否则模型在"见过答案"的题上学。
    """
    if pool_file:
        # ★★ 用自己的 jsonl 当 RL 题池 ✓✓
        #    RL 只读 `question` + `gold` ✓ —— 不读 `solution` ✗
        #    典型：同一份三段式数据 ✓ 前 4K 做 SFT ✓ 这里 --pool-skip 4000 ✓
        import json as _json
        rl_pool = []
        with open(pool_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = _json.loads(line)
                # ★ gold 可以没有 ✓ —— 从 `answer` 里抽 ✓（和 data.load_gsm8k 一致 ✓✓）
                #   这样**官方 jsonl**（只有 question+answer ✗）也能直接当题池 ✓
                if "gold" not in d:
                    if "answer" not in d:
                        raise SystemExit(
                            f"--pool-file 缺 gold 也缺 answer ✗（{pool_file}）\n"
                            f"  行内容：{sorted(d)}\n"
                            f"  → 先用 clean_staged.py 清洗 ✓ 或给它 gold ✓")
                    d = {**d, "gold": str(d["answer"]).split("####")[-1].strip()}
                rl_pool.append(d)
        if pool_skip:
            rl_pool = rl_pool[pool_skip:]
        print(f"--pool-file：题池 {len(rl_pool)} 条 ✓（跳过前 {pool_skip} 条 ✓）")
        if not rl_pool:
            raise SystemExit("--pool-file/--pool-skip 之后题池为空 ✗")
        sft_pool, _ = D.split_train()      # few-shot 仍从官方 train 取 ✓
    else:
        sft_pool, rl_pool = D.split_train()
    # ★ 自定义测试集 ✓（build_data 里用**参数** test_file ✗ 不是 args ✗）
    test = (D.load_test_file(test_file) if test_file
            else D.load_gsm8k("test"))
    # ★ few-shot 从 **sft_pool** 取，不从 rl_pool 取 ——
    #   否则那道题的答案就明写在 prompt 里，模型抄一下就有奖励 ✗
    examples = D.fewshot_examples(sft_pool, n_shot) if n_shot else []
    return rl_pool, test, examples


def sample_batch(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """随机抽 n 道训练题（用 seed 固定的 Random，保证可复现）。"""
    return rng.sample(pool, n)


# =========================================================================== #
# 3. 评估（固定题集 + 贪心）
# =========================================================================== #
@torch.no_grad()
def fmt_eval(ev: dict) -> str:
    """一行打印评估结果 —— ★ **两个口径都打** ✗

    只报贪心，正是我们绕了十二次 run 的原因 ✓
    （贪心 = 部署时"单发一次"✓；T=1 = RL 真正优化的那个分布 ✓ 两者可以反向 ✓）
    """
    t = (f"贪心 {ev['eval_reward']:.3f}"
         f"（严格 {ev['eval_strict']:.3f}，长度 {ev['eval_gen_len']:.0f}）")
    if "eval_s_pass1" in ev:
        t += (f"  |  T=1 {ev['eval_s_pass1']:.3f}"
              f"（严格 {ev['eval_s_strict']:.3f}，长度 {ev['eval_s_genlen']:.0f}）")
    return t


_DEV_IDX = 0          # ★ [显存] 打印用的设备索引（默认 0 ✓ 多卡时会被改 ✓）
_VLLM = None          # ★ vLLM 后端（None = 用 HF ✓ 默认路径完全不变 ✓）


def _trainer_greedy(model, tokenizer, prompt_ids, n: int = 12) -> list[int]:
    """★ 用**训练器**跑贪心 ✓（给 vLLM 的一致性断言当真值用 ✓）

    ⚠️ 必须显式传 `repetition_penalty=1.0` ✗✓
       —— Qwen 的 generation_config 里藏着 1.1 ✓ 而它**不管 do_sample** 都会生效 ✗
       （F3 那个 bug：所有历史"贪心"其实不是 argmax ✗）
    """
    import torch as _t
    dev = next(model.parameters()).device
    ids = _t.tensor([list(prompt_ids)], device=dev)
    out = []
    for _ in range(int(n)):
        with _t.no_grad():
            seq = model.model.generate(
                inputs=ids, attention_mask=_t.ones_like(ids),
                max_new_tokens=1, do_sample=False, repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        nxt = int(seq[0, -1])
        out.append(nxt)
        ids = _t.cat([ids, _t.tensor([[nxt]], device=dev)], dim=1)
        if nxt == tokenizer.eos_token_id:
            break
    return out


def _mem(tag: str, reset: bool = False) -> None:
    """★ 显存插桩 ✓ —— 报错只告诉我们"已经占了 30G" ✗ 不告诉我们**是谁** ✓
    所以每个阶段都打一行：当前 / 本阶段峰值 / 缓存 ✓
    （`reset=True` 用来把峰值清零 ✓ 这样峰值就是**这一阶段**的 ✓✓）
    """
    if not torch.cuda.is_available():
        return
    if reset:
        torch.cuda.reset_peak_memory_stats()
    cur = torch.cuda.memory_allocated(_DEV_IDX) / 2 ** 30
    peak = torch.cuda.max_memory_allocated(_DEV_IDX) / 2 ** 30
    res = torch.cuda.memory_reserved(_DEV_IDX) / 2 ** 30
    print(f"    [显存] {tag:<18} 当前 {cur:5.2f}G · 阶段峰值 {peak:5.2f}G · 缓存 {res:5.2f}G",
          flush=True)


def _judge(text: str, gold: str, cfg) -> bool:
    """★ 评估用的判据 = **训练奖励同一个** ✓（`strict` 或 `lenient`）"""
    m = str(getattr(cfg, "reward_mode", "strict"))
    if m == "math":
        from math_data import answer_math
        return answer_math(text, gold)
    if m == "lenient":
        return D.answer_lenient(text, gold)
    return D.answer_strict(text, gold)


def evaluate(model, tokenizer, problems: list[dict], examples, cfg: PPOConfig,
             chunk: int = 4, samples: int = 0,
             vllm_backend=None) -> dict:
    """固定题集上的准确率 —— **贪心 + T=1 采样，两套都报** ✓✓

    ★ 为什么必须两套（这是花了十二次 run 才学到的一条 ✗）：
        PPO 优化的是 **期望回报 E[R]** ✓，而贪心只取 argmax 那一条 ✗
        策略变尖之后（实测 entropy 0.33 → 0.154）两者可以**反向** ✓✓
        实测：b32 的 T=1 pass@1 从 0.328 涨到 **0.390** ✓ 而贪心从 0.492 掉到 **0.398** ✗
        → **只量贪心，会让整个方向判断反掉** ✗✓

    * **贪心**（`temperature=0`）：部署时"单发一次"的真实指标 ✓ 而且历史可比 ✓
    * **采样**（`temperature=1`）：RL 真正优化的那个分布 ✓ 和 `probe_passk.py` 同口径 ✓
      两者用**同一批题** ✓ —— 测试集前面明显更难（32题≈0.375 / 128题≈0.508 ✗）
      换题集就是换量纲 ✓ 两个数会没法放一起看 ✓
    * **分块**：一次算完会把 `(B, G, V)` 的 logits 撑爆（32 题 × 160 × 151936 × 4B ≈ 3GB）
    """
    # ★★ 评估前必须再同步一次 ✓ —— 评估发生在**更新之后** ✗
    #    上一次 sync 在 rollout 之前 ✓ → 那时权重还是旧的 ✗
    if vllm_backend is not None:
        vllm_backend.sync()

    # ★★ chunk 同时控制**两件事** ✗✓：
    #    ① vLLM 的批量 ✓（越大越省固定开销 ✓）
    #    ② **训练器重算的批量** ✗（logprobs 是训练器算的 ✓ 越大越爆显存 ✗✗）
    #    实测：chunk=4 → 161s ✗（32 次调用 ✓ 每次都有固定开销 ✓）
    #          chunk=128 → **OOM** ✗✗（lm_head 要 13.91 GiB ✗）
    #    → 折中 16：vLLM 仍有批量收益 ✓ 而 (16, 220, 151936)×4B ≈ **2.1 GB** ✓ 安全 ✓✓
    if vllm_backend is not None:
        chunk = max(int(chunk), int(cfg.vllm_eval_chunk))
    # ★★★ 生成器：有 vLLM 就**直采** ✓✓ —— eval **只要文本** ✓ 不需要 logits ✗
    #
    #   ✗ 旧路径走 `rollout` ✓ → 它为了 PPO 的 ratio 会**顺手重算 logprobs** ✗
    #     → 显存 ∝ 行数 ✗ → `chunk` 被卡在 ~16 ✗ → vLLM 一次只有 16 条序列 ✗
    #     → 实测：500 题 **15~30 分钟** ✗✗（每轮 eval 都是这个数 ✓）
    #   ✓ 现在：`vllm_backend.generate` ✓ → 只吃 KV ✓ → chunk 可到 **64** ✓
    #     → ≈ 2,400 tok/s ✓ → **~2.5 分钟** ✓✓（快 6~12 倍 ✓）
    #   ★ 采样参数**不变** ✓（temperature / max_tokens / seed ✓ 其余走默认 ✓
    #     和 `rollout` 内部那套完全一致 ✓）→ **数字可比** ✓✓
    def _gen(part, temp: float, seed: int):
        if vllm_backend is not None:
            ids = [tokenizer.apply_chat_template(
                        D.build_messages(pr["question"], examples),
                        add_generation_prompt=True,
                        return_tensors="pt")["input_ids"][0].tolist()
                   for pr in part]
            outs = vllm_backend.generate(ids, temperature=float(temp),
                                         max_tokens=int(cfg.max_new_tokens),
                                         seed=int(seed))
            txts = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
            return txts, sum(len(o) for o in outs) / max(len(outs), 1)
        b = rollout(model, tokenizer, part, examples,
                    cfg.replace(temperature=float(temp)), seed=int(seed),
                    vllm_backend=None)
        Lp = gen_start(b.mask)
        txts = [tokenizer.decode(b.input_ids[k2, int(Lp[k2]):], skip_special_tokens=True)
                for k2 in range(len(part))]
        mt = float(b.gen_len().float().mean())
        del b
        empty_cache()
        return txts, mt

    # ─────────── ① 贪心（部署时"单发一次"的真实指标 ✓）───────────
    n_ok = n_strict = 0
    n_tot = 0
    lens: list[float] = []
    for i in range(0, len(problems), chunk):
        part = problems[i:i + chunk]
        txts, mt = _gen(part, 0.0, cfg.seed)
        for txt, pr in zip(txts, part):
            n_ok += int(D.reward_fn_base(txt, pr["gold"]))       # 宽松（对齐历史数字）
            n_strict += int(_judge(txt, pr["gold"], cfg))         # ★ 跟 --reward-mode 走 ✓
        n_tot += len(part)
        lens.append(mt)
    out = {"eval_reward": n_ok / max(n_tot, 1),
           "eval_strict": n_strict / max(n_tot, 1),
           "eval_gen_len": sum(lens) / max(len(lens), 1)}

    # ─────────── ② T=1 采样（和 RL 训练同分布 ✓ 和探针同口径 ✓）───────────
    if samples and samples >= 1:
        K = int(samples)
        hit = [0] * len(problems)
        hit_s = [0] * len(problems)
        lens_s: list[float] = []
        _t = float(cfg.temperature)
        _t = _t if _t > 0 else 1.0
        for r in range(K):
            for i in range(0, len(problems), chunk):
                part = problems[i:i + chunk]
                txts, mt = _gen(part, _t, cfg.seed + 1000 + r)   # 每轮换 seed → 真采样 ✓
                for j, (txt, pr) in enumerate(zip(txts, part)):
                    hit[i + j] += int(D.reward_fn_base(txt, pr["gold"]))
                    hit_s[i + j] += int(_judge(txt, pr["gold"], cfg))
                lens_s.append(mt)
        n = max(len(problems), 1)
        out["eval_s_pass1"] = sum(hit) / (K * n)
        out["eval_s_strict"] = sum(hit_s) / (K * n)
        out["eval_s_genlen"] = sum(lens_s) / max(len(lens_s), 1)
    return out


# =========================================================================== #
# 4. 训练循环
# =========================================================================== #
def train(model, tokenizer, optimizer, pool, examples, cfg: PPOConfig,
          logger: MetricLogger, eval_problems, eval_every: int, ref_model=None,
          best_dir: str = "", no_lora: bool = False,
          eval_samples: int = 0):
    """同步 PPO：抽题 → rollout → 验契约 → ppo_update → 记日志。

    和玩具那边**结构完全一样**，只是 `rollout` 换成了真实 LLM 的版本。
    """
    rng = random.Random(cfg.seed)

    best: dict = {}
    kl_ema: float | None = None      # ★ KL 的指数滑动平均（控制用 ✓）
    for it in range(cfg.total_iters):
        # ★ 采 N 块（每块 rollout_prompts 道**不同的题**），再拼成一个 batch ✓
        #   每块单独 rollout → 显存峰值 = 单块水平 ✓
        #   拼成一个 batch → 优势归一化的 μ/σ 自动覆盖全部 N×rollout_prompts 条 ✓✓
        t0 = time.perf_counter()
        chunks = []
        # ★★ vLLM：**每步**（不是每块）同步一次 ✓ 且必须在任何 chunk 生成之前 ✓
        #    （原来打到 smoke_report 里去了 ✗ → 训练时生成仍走 HF ✗ → "零提速"是假象 ✗✓）
        if _VLLM is not None:
            _ts = _VLLM.sync()
            if (it % max(1, cfg.vllm_check_every) == 0) and getattr(_VLLM, "_probe", None):
                _VLLM.check_consistent(
                    tokenizer, _VLLM._probe,
                    trainer_greedy=lambda ids, n: _trainer_greedy(
                        model, tokenizer, ids, n))
            if it < 3 or it % 50 == 0:
                print(f"  [vllm] 同步 {_ts:.2f}s")

        for _ in range(max(int(cfg.rollout_chunks), 1)):
            if cfg.adv_norm == "grpo":
                # ★ 同一条题的 G 行必须**连续** ✗（GRPO 要按组 reshape 优势 ✓）
                #   → [p0]*G + [p1]*G + ... ✓  不是 probs*G ✗（那会把组打散 ✓）
                #   行数不变（仍是 rollout_prompts ✓）所以批量形状完全一样 ✓
                P = int(cfg.rollout_prompts) // int(cfg.group_size)
                probs = [q for q in sample_batch(pool, P, rng)
                         for _ in range(int(cfg.group_size))]
                # ★ 直接断言布局不变量 ✗ —— 组被打散是**静默**的 ✓（view 不会报错 ✓）
                _G = int(cfg.group_size)
                for _i in range(P):
                    _g = probs[_i * _G:(_i + 1) * _G]
                    assert all(x is _g[0] for x in _g), \
                        f"GRPO 第 {_i} 组不连续/不同题 ✗（优势会静默算错 ✓）"
            else:
                probs = sample_batch(pool, cfg.rollout_prompts, rng)
            chunks.append(rollout(model, tokenizer, probs, examples, cfg,
                                  seed=rng.randrange(10 ** 9),
                                  vllm_backend=_VLLM))
        _mem('rollout 之后', reset=True)          # ★ 生成 + 重算前向的峰值
        batch = concat_batches(chunks) if len(chunks) > 1 else chunks[0]
        batch.to(cfg.device)
        batch.check()                      # 每批都验 7 条契约（几微秒，别省）
        t_roll = time.perf_counter() - t0

        empty_cache()                    # rollout 之后也还一次

        t1 = time.perf_counter()
        stats = ppo_update(model, optimizer, batch, cfg,
                           ref_model=ref_model)   # 内部会断言 ratio 起手为 1
        t_upd = time.perf_counter() - t1
        _mem('ppo_update 之后')                    # ★ 更新阶段的峰值

        rows = torch.arange(batch.batch_size, device=batch.mask.device)
        acc = float(batch.rewards[rows, last_valid_index(batch.mask)].mean())
        # ★ 训练奖励拆成两项分开记 ✗ —— 混在一起没法判断模型好坏 ✓
        #   奖励 = 答对(0/1) + process_coef × 算式比例
        #   只看总和的话，模型可以靠"算式写得多而准"把总分刷高 ✗ 而答案一点没进步 ✓
        idx_last = last_valid_index(batch.mask)
        rb = float(batch.meta["reward_base"][rows, idx_last].mean())
        logger.add(step=it, reward=acc, reward_base=rb, reward_proc=acc - rb,
                   t_rollout=t_roll, t_update=t_upd,
                   t_gen=float(batch.meta.get('t_gen', 0.0)),
                   t_recompute=float(batch.meta.get('t_recompute', 0.0)),
                   lr=optimizer.param_groups[0]['lr'],
                   kl_ema=kl_ema if kl_ema is not None else 0.0,
                   gen_len=float(batch.gen_len().float().mean()),
                   **{k: v for k, v in stats.items() if k != "per_epoch"})
        # 自适应 KL：本步实测的原始 KL 决定下一步的系数。位置在 logger.add 之后，
        # 所以日志里的 kl_coef 正是本步实际用的那个。
        # ★ 自适应 KL：先对 kl_raw 做 EMA 再决策 ✓
        #   为什么必须平滑 ✗：单步 kl_raw 噪声很大 ✓ 而死区只有 2.25 倍宽 ✓
        #   → 单步决策会进**极限环** ✓（seed2 实测：20 步均值 1.40 而最后一步在带内 ✓
        #     于是控制器停在 kl_coef=0.0003 不再动作 ✗）
        if cfg.adaptive_kl and "kl_raw" in stats:
            _k = float(stats["kl_raw"])
            kl_ema = smooth_kl(_k, kl_ema)
            _new = update_kl_coef(kl_ema, cfg.kl_coef, cfg.kl_target)
            if abs(_new - cfg.kl_coef) > 1e-12:
                cfg = cfg.replace(kl_coef=_new)

        if cfg.log_every and (it + 1) % cfg.log_every == 0:
            print(f"  step {it + 1:>4}/{cfg.total_iters}"
                  f" | 奖励 {logger.recent('reward', cfg.log_every):.3f}"
                  f" | clip_frac {logger.recent('clip_frac', cfg.log_every):.3f}"
                  f" | ratio {logger.recent('mean_ratio', cfg.log_every):.4f}"
                  f" | approx_kl {logger.recent('approx_kl', cfg.log_every):.4f}"
                  f" | grad_norm {logger.recent('grad_norm', cfg.log_every):.3f}"
                  f" | 长度 {logger.recent('gen_len', cfg.log_every):.0f}"
                  f" | {t_roll:.1f}s+{t_upd:.1f}s"
                  f" | 生成 {batch.meta.get('t_gen', 0.0):.1f}s + 重算 {batch.meta.get('t_recompute', 0.0):.1f}s"
                  f" + 准备 {batch.meta.get('t_prep', 0.0):.1f}s"
                  f" + 奖励 {batch.meta.get('t_reward', 0.0):.1f}s"
                  + (f" | 显存 {mem_str()}" if mem_str() else ""))
        # ★ MPS 的缓存分配器**只借不还**，必须手动还。
        #
        # 实测（B=4, G=64, 同一台机器）：
        #   不调 empty_cache：driver 足迹 20GB → 40GB（**超过 32GB 物理内存**）→ 系统换页
        #                     → t_update 从 9.6s 涨到 **247s**（20 倍）
        #   每步调一次      ：driver 停在 ~25GB，t_update 稳定 9~25s
        #
        # 注意 `current_allocated_memory`（活的张量）全程是平的 ~9GB —— 所以这**不是泄漏**，
        # 纯粹是分配器缓存。CPU 上没有这个问题，CUDA 上 `empty_cache` 同理。
        empty_cache()

        if eval_every and it % eval_every == 0:
            ev = evaluate(model, tokenizer, eval_problems, examples, cfg,
                          vllm_backend=_VLLM,
                          samples=eval_samples)
            print(f"    [eval {it}] {fmt_eval(ev)}")
            logger.add(step=it, **ev)
            logger.save_csv()
            # ★ best-checkpoint：eval 变好就存一份 adapter。
            #   前两次我们**都把 SFT 的成果训坏了**，而且是靠肉眼看日志才发现的 ✗
            #   有这一条，最坏情况也只是"这次没进步"，而不是"又毁了一个模型" ✓
            # ★★★ best 用**当前 reward_mode 的口径** ✓✓
            #   原来只看 `eval_reward`（= 宽松/数字 ✗）→ 在 MATH 上那是**废数** ✗
            #   （LaTeX 答案抽不出数字 → 宽松列恒偏低 ✗ → 存到错的 step ✗✓ 实测差了 0.7 点 ✓）
            #   现在：strict 模式看 eval_reward ✓；lenient/math 看 strict_acc ✓（那才是真判据 ✓）
            # ★ 字段名要**对齐 `evaluate()` 的返回** ✗✓
            #   `main.evaluate` 给的是 **`eval_strict`** ✓
            #   （`strict_acc` 是 `sft.py` 的名字 ✗ → 写错就 KeyError ✓ 实测崩了 ✓）
            # ★★★ 无条件用 `eval_strict` ✓✓ —— 因为它就是**训练奖励的同一个判据**
            #   `eval_strict` 走 `_judge(txt, gold, cfg)` ✓ → 该函数**跟 `reward_mode` 走** ✓✓
            #   （math → answer_math ✓ · lenient → answer_lenient ✓ · strict → answer_strict ✓）
            #   ✗ 而 `eval_reward` 永远是 `reward_fn_base`（**抽数字的宽松判据** ✗）
            #     → 在 strict 模式下它和训练判据**不同** ✗✓ → best 会选错 step ✗
            #     → 在 math 模式下它是**废数** ✗✗（LaTeX 抽不出数字 ✓）
            # ★★★ best 要跟**训练分布**（T=1）✓✓ —— 我上一版用的是 `eval_strict`（**贪心** ✗）
            #   实测代价 ✗：T=1 从 **0.664 → 0.718（+5.4 ✓✓）** 而贪心 0.746 → 0.740（−0.6 ✓ 噪声）
            #   → `eval_strict` **没变** ✗ → best 停在 step 0 ✗✗ → **最好的 adapter 没被存下来** ✗
            #   ★ 为什么该看 T=1：RL 优化的是**采样分布** ✓（温度 1 ✓）
            #     贪心解码是"另一个策略" ✗ → 它本来就更准 ✓ → 也更容易饱和 ✓✓
            _key = ("eval_s_strict" if "eval_s_strict" in ev else "eval_strict")
            if ev[_key] > best.get(_key, -1):
                best = dict(ev, step=it)
                if not no_lora:
                    os.makedirs(best_dir, exist_ok=True)
                    model.model.save_pretrained(best_dir)
                    print(f"    ★ 新最优（step {it}: {_key} {ev[_key]:.3f}"
                          f" · 宽松 {ev['eval_reward']:.3f}）→ 已存 {best_dir}")
    if best:
        print(f"\n  全程最优 eval: {best.get(_key, 0):.3f}"
              f"（{_key} ✓ · 宽松 {best['eval_reward']:.3f}）· step {best['step']}")
    return logger


# =========================================================================== #
# 5. 冒烟测试的三条验收
# =========================================================================== #
def smoke_report(model, tokenizer, pool, examples, cfg: PPOConfig,
                 logger: MetricLogger, baseline: dict) -> None:
    print(f"\n{'=' * 66}\n冒烟测试验收\n{'=' * 66}")

    # ① ratio 起手为 1（显式再验一次；ppo_update 里那个断言也在管）
    # ★ 若用 vLLM：每步同步权重之后才采样 ✓（否则采到的是旧策略 ✗ 而 ratio 断言看不见 ✗）
    b = rollout(model, tokenizer, pool[:cfg.rollout_prompts], examples, cfg,
                seed=cfg.seed, vllm_backend=_VLLM)
    with torch.no_grad():
        logits, _, _ = model(b.input_ids, b.meta["prompt_mask"] | b.mask)
        lp = _logprobs_new(logits, b.input_ids, b.logprobs)
    dev = assert_ratio_starts_at_one(lp, b.logprobs, b.mask)
    print(f"  ① ratio 起手偏差 {dev:.2e} ≤ 1e-3  ✅   采样分布 == 重算分布")

    # ② 奖励有没有信号
    _, rs = logger.series("reward")
    n_ok = sum(1 for x in rs if x > 0)
    print(f"  ② {len(rs)} 步里有 {n_ok} 步拿到奖励"
          + ("  ⚠️ 全 0 → 优势恒 0、梯度恒 0（换题/加大 B）" if n_ok == 0 else "  ✅"))

    # ③ 时间与显存
    _, tr = logger.series("t_rollout")
    _, tu = logger.series("t_update")
    print(f"  ③ rollout {sum(tr) / len(tr):.1f}s/步 + update {sum(tu) / len(tu):.1f}s/步"
          f" ≈ {(sum(tr) + sum(tu)) / len(tr):.1f}s/步")
    if mem_str():
        print(f"     显存 {mem_str()}"
              f"（模型常驻约 2GB + Adam 4GB + logits）")
    print(f"\n  基线（{cfg.max_new_tokens} tokens）：{fmt_eval(baseline)}")
    print("  注意：64 tokens 太短，答案常常没写完就是截断的 —— 奖励偏低是**正常的**，"
          "正式跑用 160~192。")
    print("\n  ✅ 三条都过了就可以跑正式配置：python3 main.py")


# =========================================================================== #
# 6. main
# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PPO · GSM8K · Qwen2.5-0.5B。不加任何参数 = 正式配置。",
        epilog=EPILOG,
    )
    ap.add_argument("--smoke", action="store_true", help="3 步冒烟测试（第一次务必先跑）")
    ap.add_argument("--dtype", type=str, default="auto", dest="dtype",
                    help="参数 dtype。auto=float32（LoRA 别用 bf16 ✗）")
    # ★ 默认 off（fp32）：实测三组对照里，只有 off 那组没有变差 ✓
    #   而且 grad_norm 最大尖峰从 1.1e8 降到 5207（小 2 万倍）✓仍会爆，但好得多 ✓
    ap.add_argument("--autocast", type=str, default="off", dest="autocast",
                    help="前向/反向的 autocast。auto=CUDA 上 bf16、其它关闭")
    ap.add_argument("--model", type=str, default=MODEL_PATH, dest="model",
                    help="模型目录。1.5B 用：/data/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--no-lora", action="store_true", dest="no_lora",
                    help="关掉 LoRA，改全参微调（显存会多约 10GB）")
    ap.add_argument("--init-lora", type=str, default=None, dest="init_lora",
                    help="从 sft.py 存下来的 adapter 接着训（SFT → RL）")
    g = ap.add_argument_group("规模（默认：正式配置）")
    g.add_argument("--iters", type=int, default=None, help="训练步数")
    g.add_argument("--batch", type=int, default=None,
                   help="**每块**采几道题（= rollout_prompts；有效批量还要 × --rollout-chunks）")
    g.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens",
                   help="每道题最多生成多少 token")
    g.add_argument("--inner-epochs", type=int, default=None, dest="inner_epochs",
                   help="同一批数据复用几轮")
    g.add_argument("--eval-every", type=int, default=None, dest="eval_every",
                   help="每多少步评估一次（0 = 不评估）")
    # ★ 采样评估：和 RL 训练同分布的那个指标 ✓ 默认 2 轮 ✓（0 = 关掉 ✓）
    #   成本：eval_size × 这个数 次生成 ≈ 1.2s/次 ✓（128×2 ≈ 5 分钟 ✓）
    g.add_argument("--eval-samples", type=int, default=2, dest="eval_samples",
                   help="评估时额外跑几轮 T=1 采样（0 = 只贪心；默认 2）")
    g.add_argument("--eval-size", type=int, default=None, dest="eval_size",
                   help="评估用多少道测试题")
    g.add_argument("--num-minibatches", type=int, default=None, dest="num_minibatches",
                   help="更新分几块做 forward/backward（显存 ∝ 有效批量/nmb ✓；"
                        "数学等价 ✓ 等价性测试钉过 ✓）。rollout_chunks>1 时自动跟随它 ✓")
    g.add_argument("--adaptive-kl", action="store_true", dest="adaptive_kl",
                   help="让 kl_coef 自适应：把原始每-token KL 稳在 --kl-target 附近")
    g.add_argument("--kl-target", type=float, default=None, dest="kl_target",
                   help="自适应目标（原始每-token KL，nats）；默认 0.6（实测最好那轮是 0.57）")
    g.add_argument("--entropy-coef", type=float, default=None, dest="entropy_coef",
                   help="熵奖励系数（十二次 run 全是 0，从没用过）")

    g.add_argument("--adv-norm", type=str, default=None, dest="adv_norm",
                   choices=["batch", "none", "per_sequence", "grpo"],
                   help="优势估计：batch=PPO 整批归一化（默认）；grpo=组内相对 ✓")
    g.add_argument("--group-size", type=int, default=None, dest="group_size",
                   help="GRPO 的组大小 G（同一道题采 G 条）；rollout_prompts 要能被它整除")
    g.add_argument("--rollout-chunks", type=int, default=None, dest="rollout_chunks",
                   help="每步采几块 rollout（有效批量 = rollout_prompts × 这个数）")
    g.add_argument("--n-shot", type=int, default=0, dest="n_shot",
                   help="few-shot 示例数。**SFT 之后用 0**（和 sft.py 一致）；\n"
                        "没 SFT 过的话 0-shot 会很差，那时用 4")
    # 自定义 RL 题池（RL 只需 question + gold）
    g.add_argument("--pool-file", type=str, default=None, dest="pool_file",
                   help="用这个 jsonl 当 RL 题池（需含 question + gold）")
    g.add_argument("--pool-skip", type=int, default=0, dest="pool_skip",
                   help="跳过 --pool-file 的前 N 条（典型：前 4K 做 SFT，这里填 4000）")
    ap.add_argument("--test-file", type=str, default=None, dest="test_file",
                    help="★ 自定义测试集 jsonl ✓（question + gold 或含 #### 的 answer ✓）"
                         " 默认用官方 test ✓")
    g = ap.add_argument_group("优化")
    g.add_argument("--lr", type=float, default=None, help="主干学习率（value_head 走 10×）")
    g.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")
    g.add_argument("--clip-eps", type=float, default=None, dest="clip_eps")
    g.add_argument("--process-coef", type=float, default=None, dest="process_coef",
                   help="过程奖励系数（0 = 纯结果奖励，会塌缩）")
    g.add_argument("--kl-coef", type=float, default=None, dest="kl_coef",
                   help="KL 惩罚系数（0 = 关掉，同时不建参考模型）")
    g.add_argument("--value-coef", type=float, default=None, dest="value_coef",
                   help="value 损失的系数（共享主干下这个很重要）")
    g.add_argument("--head-lr-mult", type=float, default=1.0, dest="head_lr_mult",
                   help="value_head 的学习率倍数（默认 1；调大容易过冲）")
    g.add_argument("--temperature", type=float, default=None, help="采样温度（训练时；评估固定贪心）")
    g = ap.add_argument_group("杂项")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--tag", type=str, default=None, help="日志/图/checkpoint 的名字")
    # ★ vLLM rollout（默认 hf ✓ 行为不变 ✓）
    g.add_argument("--rollout-backend", choices=["hf", "vllm"], default="hf",
                   help="生成用什么后端：hf（默认 ✓）或 vllm（需 fp32 ✓ 见 vllm_backend.py）")
    g.add_argument("--vllm-gpu-util", type=float, default=0.45,
                   help="vLLM 的 gpu_memory_utilization —— 它只是**预算上限** ✓ "
                        "不是真分配 ✓（日志 `Actual usage` 才是真的 ✓）。"
                        "真正吃显存的是 kv-cache-gb ✓ 所以这个给宽 ✓ 默认 0.45")
    g.add_argument("--vllm-kv-gb", type=float, default=1.5,
                   help="★ KV cache 的**显式**大小（GB）✓ 不给的话 vLLM 会把剩余显存全占 ✗ "
                        "（实测占到 8.6G ✗ 而我们只需 ~1G ✓）")
    g.add_argument("--vllm-eval-chunk", type=int, default=16,
                   help="★ 评估时每次喂 vLLM 多少题 —— 它同时是**训练器重算的批量** ✗ "
                        "4=161s ✗ 128=OOM ✗ 16 安全 ✓✓ 默认 16")
    g.add_argument("--reward-mode", type=str, default=None, dest="reward_mode",
                   choices=["strict", "lenient", "math"],
                   help="奖励判据：strict 必须 #### N（默认 ✓）· lenient 认 **Answer:** 等方言 "
                        "（★ 直接 RL 基座时**必须**用 lenient ✗ 否则先学方言 ✗）")
    g.add_argument("--device", type=str, default=None,
                   help="★ 训练器放哪张卡（如 cuda:1 ✓）—— 多卡时把 vLLM 留在 cuda:0 ✓✓")
    g.add_argument("--vllm-dtype", type=str, default=None, dest="vllm_dtype",
                   choices=["float32", "bfloat16"],
                   help="vLLM 的 dtype（默认 float32 ✓；L40S 上 bf16 省一半显存 ✓✓）")
    g.add_argument("--vllm-sleep-level", type=int, default=1,
                   help="生成结束后 sleep 释放（0 = 不释放 ✓ 1 = level 1 ✓ "
                        "日志会把该版本的 sleep 语义打出来 ✓）")
    g.add_argument("--vllm-check-every", type=int, default=25,
                   help="每 N 步做一次 vLLM↔训练器 一致性断言（0 = 关）")
    g.add_argument("--no-deterministic", action="store_false", dest="deterministic",
                   help="关掉确定性（更快，但同一模型的评估数字会抖 ±3~6%% ✗）")
    args = ap.parse_args()

    cfg = make_cfg(args)
    # ★ 有效批量 = rollout_prompts × rollout_chunks ✓
    # ★★ 这里原来会**强制**把 num_minibatches 改成 rollout_chunks ✗✓
    #    后果：`--num-minibatches` 变成**死参数** ✗（传什么都会被改掉 ✓）
    #    而且"更新时每块的行数 = 有效批量/nmb"在那种强制下恰好退化成
    #        (batch × chunks) / chunks = **batch** ✓
    #    → 于是 `--num-minibatches` 这个"显存旋钮"完全是假的 ✗✓
    #      （我按它给的显存建议也因此全错了一次 ✗）
    #    现在：**不再改写** ✓ 只提示 ✓ → 两个旋钮都真的可用 ✓✓
    #      每次 rollout 行数 = --batch ✓
    #      更新每块行数     = (--batch × --rollout-chunks) / --num-minibatches ✓
    _eff = int(cfg.rollout_prompts) * int(cfg.rollout_chunks)
    if int(cfg.num_minibatches) > _eff:
        raise SystemExit(f"--num-minibatches({cfg.num_minibatches}) > 有效批量({_eff}) ✗")
    print(f"有效批量 {_eff} 条（{cfg.rollout_prompts}×{cfg.rollout_chunks}）"
          f" | 更新分 {cfg.num_minibatches} 块 → 每块 {_eff // int(cfg.num_minibatches)} 行 ✓"
          f" | ★ 更新时的显存 ∝ 每块行数 ✓（这个旋钮是真的了 ✓）")
    print(f"kl_coef 自适应={cfg.adaptive_kl} 目标={cfg.kl_target} 起始={cfg.kl_coef} ✓")
    if cfg.adv_norm == "grpo":
        # ★ GRPO 的定义里**没有 critic** ✓（组均值就是 baseline ✓）
        #   → value_coef 归零 ✓ 但必须**打出来** ✗ 不能静默改 ✓
        if float(cfg.value_coef) != 0.0:
            print(f"GRPO：value_coef {cfg.value_coef} → 0"
                  f"（组均值当 baseline ✓ 不需要 critic ✓；--value-coef 可覆盖 ✓）")
            cfg = cfg.replace(value_coef=0.0)
        _P = int(cfg.rollout_prompts) // int(cfg.group_size)
        print(f"GRPO：{_P} 道题 × 组 {cfg.group_size} = {cfg.rollout_prompts} 行/块 ✓"
              f"（优势在**组内**归一化 ✓ 全对/全错的组自动出 0 梯度 ✓）")
    if args.eval_samples and args.eval_samples >= 1:
        _n = (args.eval_size or (8 if args.smoke else 128))
        print(f"评估：贪心 {_n} 题 + T=1 采样 {_n} 题 × {args.eval_samples} 轮"
              f" ≈ {_n * (1 + args.eval_samples) * 1.2 / 60:.1f} 分钟/次"
              f"（生成是唯一的大头 ✓）")
    print(cfg.summary())
    print()

    # ★ 确定性：让「同一份权重 + 同一批题」给出**同一个数字** ✓
    #   实测：同一个 adapter、同样 64 题、贪心，三次评估是 0.406 / 0.438 / 0.406 ✗
    #   —— 差 3.2 个点，纯粹是 CUDA 非确定性 kernel 造成的 ✓
    #   不开的话，你永远分不清"模型真的变了"还是"这次评估运气不同" ✗
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)   # warn_only：避免个别算子直接报错
        print("确定性：已开（cudnn.deterministic + use_deterministic_algorithms）✓")
    torch.manual_seed(cfg.seed)
    pool, test, examples = build_data(args.n_shot, args.pool_file,
                                     args.pool_skip, args.test_file)
    print(f"数据：训练池 {len(pool)} 题 | 测试 {len(test)} 题 | {len(examples)}-shot\n")

    tokenizer = load_tokenizer(args.model)
    p_dtype = PPOConfig.resolve_dtype(cfg.device, args.dtype)
    p_amp = PPOConfig.resolve_autocast(cfg.device, args.autocast)
    model = LMWithValueHead(args.model, device=cfg.device, lora=not args.no_lora,
                            init_lora=args.init_lora, dtype=p_dtype, autocast=p_amp)
    check_model(model)                                   # 运行时契约（Protocol 管不到）
    # ★ 参考模型 = **RL 开始时的策略**（也就是 SFT 后的那份 adapter）。
    #   用一份独立副本（+2GB）而不是在同一基座上切 adapter —— 内存有余量，
    #   而多 adapter 切换的语义/缓存失效容易出隐蔽 bug ✗ 用内存换确定性 ✓
    ref_model = None
    if cfg.kl_coef != 0:
        # ★★★ ref 是**冻结**的 ✗✓ —— 它只用来算 KL ✓ 不需要 fp32 ✓
        #   0.5B 时代 +2GB 无所谓 ✓ 但 3B 上 fp32 是 **+11.5 G** ✗✗（实测 OOM 的元凶 ✓）
        #   → 在 CUDA 上用 **bfloat16** ✓（-5.7 G ✓ KL 的精度绰绰有余 ✓✓）
        ref_dtype = "bfloat16" if str(cfg.device).startswith("cuda") else p_dtype
        ref_amp = None if ref_dtype != "float32" else p_amp
        ref_model = LMWithValueHead(args.model, device=cfg.device,
                                    lora=not args.no_lora,
                                    init_lora=args.init_lora,
                                    dtype=ref_dtype, autocast=ref_amp)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        print(f"参考模型：已建（冻结 ✓ dtype={ref_dtype} ✓，kl_coef={cfg.kl_coef}）")

    optimizer = make_optimizer(model, lr=cfg.lr,
                               head_lr=cfg.lr * args.head_lr_mult)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"模型：可训练 {n_tr / 1e6:.1f}M / 全部 {n_all / 1e6:.0f}M"
          f"（{'全参' if args.no_lora else 'LoRA'}）"
          f" | 主干 lr {cfg.lr:.0e} / value_head lr {cfg.lr * args.head_lr_mult:.0e}"
          f" | value_coef {cfg.value_coef}"
          + (f" | adapter {args.init_lora}" if args.init_lora else "") + "\n")

    # ★ 默认 128（不是 32）：64 题时 SE ≈ ±6.2% ✗ → 分辨不出 ±10 点以内的差别 ✓
    #   128 题 → ±4.4%；要更准就在最后用 sft.py --eval-only --eval-size 512（±2.2% ✓）
    eval_size = args.eval_size if args.eval_size is not None else (8 if args.smoke else 128)
    # ★ 必须拦 ✗：eval_size<=0 时 evaluate 用 max(n,1) 兜住 → 所有指标 0.0 ✓
    #   而 train() 第一轮 `0.0 > best.get(-1)` → 打印「★ 新最优(step 0: 0.000)」并写 adapter ✗
    #   （同一个输入 sft.py 会直接崩 → 两条评估路径行为不同 ✓）
    if eval_size <= 0:
        raise SystemExit(f"--eval-size 必须 > 0 ✗（给了 {eval_size}；"
                         f"想让评估安静请用 --eval-every 1000 ✓）")
    eval_problems = test[:eval_size]
    logger = MetricLogger(cfg.out_dir, cfg.tag)

    # ---- ★ vLLM 后端（可选 ✓ 只替"生成" ✓ 评估仍走 HF ✓ 保证可比 ✓）----
    global _VLLM
    if cfg.rollout_backend == "vllm":
        from vllm_backend import VLLMBackend, bare_base, live_lora_pairs
        _pairs = live_lora_pairs(model.model)          # ★ 活的 LoRA ✓
        _base = bare_base(model.model)                 # ★ 真裸 base ✓（见 bare_base 的注释）
        # ★ dtype 由 `--vllm-dtype` 决定 ✓：默认 float32 ✓（最安全 ✓）
        #   L40S(sm_89) 上实测 bf16 正确 ✓ → 3B 时用它省一半权重显存 ✓✓
        _VLLM = VLLMBackend(args.model, dtype=cfg.vllm_dtype,
                            gpu_memory_utilization=cfg.vllm_gpu_util,
                            max_model_len=cfg.max_new_tokens + 512,
                            lora_base=_base, lora_pairs=_pairs,
                            kv_cache_bytes=int(cfg.vllm_kv_gb * 1e9),   # ★ 1.5G ✓
                            sleep_level=cfg.vllm_sleep_level)           # ★ 生成后释放 ✓
        _VLLM.start()
        # 探针 prompt：固定一条 ✓ 用于一致性和"权重是否真的更新了" ✓
        _msgs = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
        _VLLM._probe = tokenizer(
            tokenizer.apply_chat_template(_msgs, tokenize=False,
                                          add_generation_prompt=True)).input_ids
        _VLLM.sync()
        print(f"  [vllm] 首次同步 {_VLLM.last_sync_s:.2f}s ✓ "
              f"（一致性断言每 {cfg.vllm_check_every} 步 ✓）\n")

    # ---- 基线：不做这一步，你不知道后面是"学会了"还是"本来就会" ----
    t0 = time.perf_counter()
    _mem('前评估 之前', reset=True)
    baseline = evaluate(model, tokenizer, eval_problems, examples, cfg,
                        vllm_backend=_VLLM,
                        samples=args.eval_samples)
    _mem("前评估 之后")
    print(f"训练前评估（{len(eval_problems)} 题，{time.perf_counter() - t0:.0f}s）："
          f"{fmt_eval(baseline)}\n")

    # ---- 训练 ----
    eval_every = args.eval_every if args.eval_every is not None else (1 if args.smoke else 10)
    t0 = time.perf_counter()
    os.makedirs(cfg.out_dir, exist_ok=True)
    train(model, tokenizer, optimizer, pool, examples, cfg, logger, eval_problems,
          eval_every, ref_model=ref_model,
          best_dir=os.path.join(cfg.out_dir, f"{cfg.tag}_best_lora"),
          no_lora=args.no_lora, eval_samples=args.eval_samples)
    print(f"\n训练耗时 {time.perf_counter() - t0:.0f}s")


    # ---- 训练后 ----
    after = evaluate(model, tokenizer, eval_problems, examples, cfg,
                     vllm_backend=_VLLM,
                     samples=args.eval_samples)
    print(f"训练后评估：{fmt_eval(after)}")

    print(f"\n{logger.summary(20)}")
    # ★ vLLM 必须等**训练后评估**跑完才能关 ✗✓（原来在 train() 之后就关了 ✗ → after 报 NoneType ✗）
    if _VLLM is not None:
        _VLLM.shutdown()
        print("  [vllm] 已关闭 ✓")

    print(f"\n图     : {logger.plot(title='PPO · GSM8K · Qwen2.5-0.5B')}")
    print(f"CSV    : {logger.save_csv()}")
    if args.smoke:
        smoke_report(model, tokenizer, pool, examples, cfg, logger, baseline)
    else:
        if args.no_lora:
            ckpt = os.path.join(cfg.out_dir, f"{cfg.tag}_final.pt")
            torch.save(model.state_dict(), ckpt)          # 全参：只能存全部（约 2GB）
            print(f"checkpoint: {ckpt}（约 2GB）")
        else:
            # ★ LoRA：只存 adapter（约 10MB）。
            #   存 model.state_dict() 会把**冻结的 0.5B 基座也存进去**（2GB ✗），
            #   而且那种格式**没法用 --init-lora 加载** —— 等于白存。
            ckpt = os.path.join(cfg.out_dir, f"{cfg.tag}_lora")
            os.makedirs(ckpt, exist_ok=True)
            model.model.save_pretrained(ckpt)             # peft 只写 adapter
            print(f"adapter: {ckpt}（约 10MB，可用 --init-lora 加载续训）")


if __name__ == "__main__":
    main()
