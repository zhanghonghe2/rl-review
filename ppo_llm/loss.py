"""PPO 的目标函数：裁剪替代目标 + 价值损失 + KL 惩罚 + 熵奖励。

和 gae.py 一样，这是**纯函数模块**：只吃张量，不认识模型、不认识 optimizer。
损失函数不做优化 —— `.backward()` 和 `.step()` 属于 trainer，不属于这里。

目标函数
--------
记概率比

    r_t = exp( logπ_new(a_t|s_t) - logπ_old(a_t|s_t) )

PPO 的裁剪替代目标：

    L_policy = -E[ min( r_t · A_t , clip(r_t, 1-ε, 1+ε) · A_t ) ]

价值损失（可选裁剪）：

    L_value  = 0.5 · E[ max( (V - R)², (V_clip - R)² ) ]

KL 惩罚（对 reference 模型，防止把预训练能力训崩）与熵奖励：

    L_kl      = β · E[ KL_t ]
    L_entropy = -c_ent · E[ H_t ]

    L_total = L_policy + c_vf · L_value + L_kl + L_entropy

三条纪律
--------
1. **所有 reduction 必须走 `masked_mean`**，分母是有效 token 数，不是 `numel()`。
   这个文件里应该搜不到裸的 `.mean()`（除了 `masked_mean` 内部）。
2. **返回诊断量**。PPO 调参 90% 靠看 `clip_frac` 和 `approx_kl`，一开始就把它们
   和 loss 一起返回，省得回头改签名。
3. **不碰 optimizer**。损失函数只负责算数。

为什么 `min` 里的不对称是关键
----------------------------
`min(surr1, surr2)` 对不同符号的 advantage 效果完全不同，这不是 bug 而是设计：

    A > 0（好动作）：ratio 超过 1+ε 时被裁 → 不敢一次把好动作的概率抬太高
    A < 0（坏动作）：ratio 低于 1-ε 时被裁 → 不敢一次把坏动作的概率砍太狠

`_case_b()` 用四个精确数字把这件事钉住（ratio=3 和 ratio=0.5，各配 A=±1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import sys

import torch
from torch import Tensor

from batch import masked_mean
from config import PPOConfig


# =========================================================================== #
# 1. KL 估计量（逐 token）
# =========================================================================== #
#: k3 里 d 的上截断值。d=10 时 exp 才 22026 ✓；再大就该截住（那是极端离群 ✗）
KL_D_MAX = 10.0


def kl_penalty(
    logp: Tensor,
    logp_ref: Tensor,
    estimator: Literal["k1", "k3"] = "k3",
) -> Tensor:
    """逐 token 的 KL(π_new ‖ π_ref) 估计量。

    令 ``d = logp_ref - logp``（注意方向！）：

    ``"k1"`` → ``d``
        无偏，但**方差最大，而且经常是负数**。当新策略比 reference 更偏向
        采样到的 token（训练早期最常见的情况）时，`d < 0` —— 于是"KL 惩罚"
        变成了"奖励"，符号完全反了。这就是很多实现里"KL 越训越负"的原因。

    ``"k3"`` → ``exp(d) - d - 1``
        恒 ≥ 0（当且仅当 d=0 取等号），方差小得多，是更好的估计量。
        Schulman, "Approximating KL Divergence" (2020)。**默认用它。**

    自己动手验证（写进 ``_case_c()``）：
      * logp == logp_ref          → 两个估计量都 == 0
      * logp > logp_ref（d < 0）  → k1 < 0（负 KL！），k3 > 0
      * logp < logp_ref（d > 0）  → 两个都 > 0
    最后再断言：任何输入下 k3 ≥ 0。

    符号约定写反是 KL 项最常见的隐蔽错误 —— loss 照样下降，只是优化的是错的
    东西。所以这个函数的期望值必须手算对拍，不能"跑起来看着正常"就算数。
    """
    d = logp_ref - logp
    if estimator == "k1":
        return d

    if estimator == "k3":
        # ★ 从上方截断 d：k3 = exp(d) - d - 1，而 exp(d) 的梯度是 exp(d)·∂d ✗
        #   d 稍大（新策略明显更不看好这个 token）就会指数爆炸 ✗
        #   实测 grad_norm 出现过 6498 / 1.1e8，**连 fp32 都有** ✗
        #   d 很负时 exp(d)-d-1 ≈ -d 是线性的 ✓ 不会爆 ✓ 所以只截上方 ✓
        dc = d.clamp(max=KL_D_MAX)
        return dc.exp() - dc - 1

    raise ValueError(f"未知 estimator: {estimator}")



# =========================================================================== #
# 2. 输出容器（这部分是管道，直接给你实现好了）
# =========================================================================== #
@dataclass
class PPOOutput:
    """一次 PPO 更新的结果：5 个损失分量 + 4 个诊断量。"""

    # ---- 损失（参与反传）----
    total: Tensor        # 加权求和后的总损失
    policy: Tensor       # L_policy（已经是负号之后的，直接 backward 就是梯度上升）
    value: Tensor        # L_value（未乘 value_coef）
    kl: Tensor           # β·E[KL]（已乘 kl_coef）
    kl_raw: Tensor       # 原始 E[KL]（**不乘** kl_coef ✓ 自适应控制器用 ✓）
    entropy: Tensor      # **原始** E[H]（≥0，可直接画曲线）；
                         # 进 loss 的是 -c_ent·E[H]，见 total

    # ---- 诊断（detach，不反传）----
    clip_frac: Tensor    # 被裁剪的 token 比例。**持续 > 0.3 说明步子太大**
    mean_ratio: Tensor   # E[r]，应该围绕 1.0
    approx_kl: Tensor    # E[(r-1) - log r]，恒 ≥ 0。**> 0.02 通常该降 lr**
    ratio_max: Tensor    # max(r)，用来发现离群样本
    n_ratio_nonfinite: Tensor = None   # ★ 有多少个有效位置上的 r 是非有限值 ✗（>0 就是有 bug ✓）

    def detached(self) -> dict[str, float]:
        """转成可直接打日志的普通 dict。"""
        return {
            "loss": float(self.total.detach()),
            "policy_loss": float(self.policy.detach()),
            "value_loss": float(self.value.detach()),
            "kl": float(self.kl.detach()),
            "kl_raw": float(self.kl_raw.detach()),
            "entropy": float(self.entropy.detach()),
            "clip_frac": float(self.clip_frac.detach()),
            "mean_ratio": float(self.mean_ratio.detach()),
            "approx_kl": float(self.approx_kl.detach()),
            "ratio_max": float(self.ratio_max.detach()),
            "n_ratio_nonfinite": float(self.n_ratio_nonfinite.detach())
            if self.n_ratio_nonfinite is not None else 0.0,
        }


# =========================================================================== #
# 3. 主损失函数
# =========================================================================== #
def ppo_loss(
    logprobs_new: Tensor,
    logprobs_old: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values_new: Tensor,
    values_old: Tensor,
    mask: Tensor,
    cfg: PPOConfig,
    entropy: Tensor | None = None,
    logprobs_ref: Tensor | None = None,
) -> PPOOutput:
    """对整批数据算一次 PPO 更新。所有张量形状 (B, T)。

    参数
    ----
    logprobs_new : 当前策略在**训练这批数据时**重算的 logprob（带梯度）
    logprobs_old : 行为策略在 **rollout 时**存下来的 logprob（不参与求导）
    advantages, returns : ``compute_gae`` 的输出
    values_new, values_old : 当前 critic 的估计 与 rollout 时的估计
        （`values_old` 只在 `cfg.value_clip_eps is not None` 时用得上）
    mask : 只有生成出来的 token 为 True
    cfg : `PPOConfig`，提供 clip_eps / value_clip_eps / value_coef /
        entropy_coef / kl_coef / kl_estimator
    entropy : (B,T)，逐 token 的策略熵（想开熵奖励才需要传）
    logprobs_ref : (B,T)，reference 模型的 logprob（`cfg.kl_coef > 0` 才需要）

    每一项怎么写
    ------------
    **policy**（注意负号）::

        log_ratio = logprobs_new - logprobs_old
        ratio     = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1-ε, 1+ε) * advantages      # ε = cfg.clip_eps
        policy = -masked_mean(torch.min(surr1, surr2), mask)

    **value**（`cfg.value_clip_eps is None` 时退化成普通 MSE）::

        err = (values_new - returns) ** 2
        若启用裁剪:
            v_clip = values_old + (values_new - values_old).clamp(-εv, +εv)
            err    = torch.maximum(err, (v_clip - returns) ** 2)
        value = 0.5 * masked_mean(err, mask)

    **kl**::

        kl_vec = kl_penalty(logprobs_new, logprobs_ref, cfg.kl_estimator)
        kl     = cfg.kl_coef * masked_mean(kl_vec, mask)
        # cfg.kl_coef == 0 或没传 logprobs_ref → 返回 0（用 torch.zeros 保持 dtype/device）

    **entropy**（注意负号：熵是"越大越好"，所以进 loss 要取负）::

        ent          = masked_mean(entropy, mask)        # 原始熵，≥0，进诊断量
        entropy_term = -cfg.entropy_coef * ent           # 进 loss 的加权项

    **total**::

        total = policy + cfg.value_coef * value + kl + entropy_term

    诊断量（全部包在 ``torch.no_grad()`` 里算）
    ------------------------------------------
    clip_frac   = masked_mean((ratio - 1).abs() > ε, mask)          # 布尔要转 float
    mean_ratio  = masked_mean(ratio, mask)
    approx_kl   = masked_mean((ratio - 1) - log_ratio, mask)        # 恒 ≥ 0
    ratio_max   = (ratio * mask).max()                              # 乘 mask 把 padding 的 0 排掉

    四个坑
    ------
    1. **padding 位置必须彻底不参与**。四项都要除以"有效 token 数"，不是 numel()。
       用 `masked_mean` 就自动对了。`_case_d()` 会用一个"带垃圾 padding 的 batch"
       和一个"压实的 batch"对比，要求两者结果**完全一致**。
    2. **`kl` 和 `entropy` 在关闭时要返回真正的 0 张量**（`torch.zeros((), ...)`），
       不要返回 Python 的 `0` —— 不然 `total` 的 dtype/device 可能出问题，
       `.backward()` 时的图也会变得奇怪。
       （`entropy` 字段报告的是**原始熵 E[H]**，不是加权项 -c·E[H] ——
         前者可解释、能画曲线；后者只是它的线性函数，没有额外信息。）
    3. **`ratio_max` 别忘了乘 mask**。padding 位置的 ratio 是 `exp(0-0) = 1`，
       不乘 mask 的话只要有效位置的最大值 < 1 就会返回 1，掩盖真实情况。
    4. **`min` 的不对称是有意的**，别"顺手改成 `clamp(surr1)`"或者只用 `surr2`。
       `_case_b()` 用 ratio=3 和 ratio=0.5 各配 A=±1 的四个数字钉住它。
    """

    # policy
    log_ratio = logprobs_new - logprobs_old
    ratio = log_ratio.exp()
    
    surr1 = advantages * ratio
    surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
    policy = -masked_mean(torch.min(surr1, surr2), mask)

    
    # value
    err = (values_new - returns) ** 2
    if cfg.value_clip_eps is not None:
        v_clip = values_old + (values_new - values_old).clamp(-cfg.value_clip_eps, cfg.value_clip_eps)
        err = torch.maximum(err, (v_clip - returns) ** 2)
    value = 0.5 * masked_mean(err, mask)

    # kl
    # ★ kl_raw = **原始**每-token KL ✓（自适应 kl_coef 要控制的就是它 ✓）
    #   原来只能靠 `kl / kl_coef` 反推 ✗ —— 系数一旦会变就推不出来了 ✓✓
    if logprobs_ref is not None and cfg.kl_coef != 0:
        kl_vec = kl_penalty(logprobs_new, logprobs_ref, cfg.kl_estimator)
        kl_raw = masked_mean(kl_vec, mask)
        kl = cfg.kl_coef * kl_raw
    else:
        kl_raw = torch.zeros((), dtype=policy.dtype, device=policy.device)
        kl = torch.zeros((), dtype=policy.dtype, device=policy.device)

    # entropy
    # 诊断量记【原始熵】E[H]：正的、可解释、和 ln(V) 可比、能直接画曲线。
    # 进 loss 的是 -c_ent·E[H]（最小化它 = 鼓励探索）。
    # 注意：只要调用方给了 entropy 就把它算出来（哪怕系数是 0），
    #       否则日志里会显示 entropy=0，让你误以为策略熵真的是 0。
    if entropy is not None:
        ent = masked_mean(entropy, mask)
    else:
        ent = torch.zeros((), dtype=policy.dtype, device=policy.device)
    entropy_term = -cfg.entropy_coef * ent

    # total 
    total = policy + cfg.value_coef * value + kl + entropy_term


    with torch.no_grad():
        clip_frac = masked_mean((ratio - 1).abs() > cfg.clip_eps, mask)
        # ★★★ 诊断量抗 nan ✓✓ —— `masked_mean` 是 `(x*m).sum()` ✗
        #   只要**一个** `-inf`，整个 `mean_ratio` 就变 nan ✗✓（我们正是看到 nan ✓）
        #   → 于是 clip_frac/ratio 的自检**全被遮住** ✗
        #   ★ 只修**诊断量** ✗ 不动 loss ✓（loss 用 min(surr)*A ✓ 那里 A=0 会抵消 ✓）
        _nf = ~torch.isfinite(ratio) & mask
        n_nonfinite = _nf.sum().float()
        mean_ratio = masked_mean(
            torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0), mask)
        approx_kl = masked_mean(torch.nan_to_num((ratio - 1) - log_ratio,
                                                 nan=0.0, posinf=0.0, neginf=0.0), mask)
        ratio_max = (torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0) * mask).max()

    out = PPOOutput(
        total=total, 
        policy=policy,
        value=value,
        kl=kl, 
        kl_raw=kl_raw,
        entropy=ent, 
        clip_frac=clip_frac, 
        mean_ratio=mean_ratio,
        n_ratio_nonfinite=n_nonfinite,
        approx_kl=approx_kl,
        ratio_max=ratio_max
    )

    return out


    raise NotImplementedError("写 PPO 的四项损失 + 四个诊断量")


# =========================================================================== #
# 4. 诊断工具（管道，已实现）
# =========================================================================== #
def assert_ratio_starts_at_one(
    logprobs_new: Tensor, logprobs_old: Tensor, mask: Tensor,
    tol: float = 1e-3, tol_max: float = 0.02
) -> float:
    """第一次内层更新**之前**，new 和 old 必须来自同一个策略 → ratio ≡ 1。

    实测最大偏差 > tol，说明 old logprob 和 new logprob 的**计算路径不一致**：
    重新分词、padding 方向不同、dropout 没关、prompt 模板两处各写一遍……
    这是 PPO 实现里最容易悄悄坏掉的一环 —— 它不报错，只是让裁剪目标失效。

    trainer 里建议每批都调一次（开销可忽略），出问题当场炸掉而不是训歪一周。
    """
    ratio = (logprobs_new - logprobs_old).exp()
    dev_t = ((ratio - 1.0).abs() * mask).detach()   # 纯诊断，不需要梯度
    dev = float(dev_t.max())

    # ★ 两条判据，量纲不同 ✗（被 seed1 误杀之后改的 ✓）
    #   **均值**：对"系统性错位"极敏感 ✓ → 容差可以很紧（1e-3 ✓）
    #   **最大值**：会被逐位置的舍入毛刺顶高 ✗ → 必须放宽 ✓
    #   为什么最大值天然高 ✗：`logprobs_old` 来自 rollout 的 forward（整批 ✓），
    #   而 `lp_new` 来自 ppo_update 的**分块** forward（4 行 ✓）
    #   → 张量形状不同 → kernel 归约顺序不同 → 舍入不同 ✓
    #   实测这份噪声 **1e-4 ~ 2.3e-3** ✓（冒烟里不分块时恰好 **0.00e+00** ✓）
    #   而历史真 bug 是 **0.9811** 和 **195** ✓✓ → 差 2~4 个数量级 ✓
    n_mask = max(int(mask.sum()), 1)
    dev_mean = float((dev_t * mask).sum()) / n_mask
    if dev_mean > tol or dev > tol_max:
        # 自动诊断：如果所有超差位置上的 logprobs_old 都恰好是 0，
        # 那几乎一定是 mask 传错了板子 —— 传成了"真实 token"mask，把 prompt 位置
        # 也算进来了，而 prompt 位置的 logprobs_old 在 rollout 里从没被填过（恒为 0）。
        bad = dev_t > tol
        n_bad = int(bad.sum())
        n_zero_old = int((bad & (logprobs_old == 0)).sum())
        # 判别量：**整个 mask 里**有多少位置的 logprobs_old 是 0。
        #   占比高 → 这个字段根本没被填上（rollout 重算时 forward 只回了部分生成段）
        #   只有极个别 → 数值饱和（某个 logit 碾压其他时 log_softmax 恰好给 0.0），不是 bug
        n_zero_in_mask = int(((logprobs_old == 0) & mask).sum())
        n_mask = max(int(mask.sum()), 1)
        hint = ""
        if n_bad and n_zero_old == n_bad:
            hint = (f"\n        ⚠️ 自动诊断：{n_bad} 个超差位置上的 logprobs_old **全为 0**。\n"
                    f"           两种可能：\n"
                    f"           (a) **mask 传错了** —— 这里要的是「生成位置」的 mask"
                    f"（`batch.mask`），\n"
                    f"               不是 `forward` 用的「真实 token」mask"
                    f"（`prompt_mask | mask`）。prompt 位置在 rollout 里没被填过。\n"
                    f"           (b) **mask 是对的，但 rollout 没把 logprobs 填到这些位置上**\n"
                    f"               —— 例如重算时 `forward` 只返回了部分生成段的 logits"
                    f"（`gen_len` 是上一批留下的、比本批短）。\n"
                    f"           判别：mask 内共有 {n_zero_in_mask}/{n_mask} "
                    f"（{n_zero_in_mask / n_mask:.1%}）个位置的 logprobs_old == 0。\n"
                    f"             占比高 → (b)；只有极个别 → 数值饱和，不是 bug。")
        raise AssertionError(
            f"ratio 起始偏差：均值 {dev_mean:.2e}（阈值 {tol:.0e}）"
            f" · 最大 {dev:.4f}（阈值 {tol_max}）\n"
            f"        → old/new logprob 的计算路径不一致！\n"
            f"        常见原因（按可能性排序）：\n"
            f"          1. mask 传错（见下方自动诊断）\n"
            f"          2. rollout 与重算走了两条不同的 tokenize / padding 路径\n"
            f"          3. 生成段和 mask 错位了一个 token"
            f"{hint}"
        )
    return dev


# =========================================================================== #
# 5. 手算用例（这些期望值是我用纸笔算出来的，不要改）
# =========================================================================== #
def _cfg(**kw) -> PPOConfig:
    """造一个干净的配置：默认把所有可选项关掉，方便逐项隔离测试。"""
    base = dict(tag="loss_test", entropy_coef=0.0, kl_coef=0.0, value_coef=0.0,
                clip_eps=0.2, value_clip_eps=None)
    base.update(kw)
    return PPOConfig(**base)


def _case_e() -> None:
    """★ k3 的 exp(d) 必须从上方截断 —— 否则 d 稍大梯度就爆炸 ✓

    实测（1.5B 的 RL，**fp32 也出现** ✗ 所以不是 bf16 的锅）：
        grad_norm 出现过 6498 / 1.1e8 ✗
    机制：k3 = exp(d) − d − 1，d = logp_ref − logp
          新策略明显更不看好某 token 时 d 很大 ✗ → exp(d) 指数爆炸 ✗
          且其梯度是 exp(d)·∂d ✗✗ → 一步就把全局范数顶穿

    两条断言：d 很大时必须被截住 ✓；d 正常时必须**完全等于原公式** ✓
    """
    lp = torch.tensor([0.0], requires_grad=True)
    ref = torch.tensor([50.0])            # d = 50 → exp(50) = 5.2e21 ✗
    kl = kl_penalty(lp, ref, "k3")
    kl_v = float(kl.detach())          # detach 一下，免得刷 requires_grad 警告
    if kl_v > 1e6:
        raise AssertionError(f"没截住：kl = {float(kl):.3e}（应 < 1e6）")
    kl.backward()
    if abs(float(lp.grad)) > 1e6:
        raise AssertionError(f"梯度没截住：{float(lp.grad):.3e}")
    a, b = torch.tensor([0.5]), torch.tensor([0.55])
    got = float(kl_penalty(a, b, "k3"))
    want = float((b - a).exp() - (b - a) - 1)
    if abs(got - want) > 1e-9:
        raise AssertionError(f"正常范围被改动了：got={got} want={want}")
    print(f"    d=50 → kl={kl_v:.1f}（截住 ✓）| d=0.05 → {got:.6f}（原公式 ✓）")
    print(f"    截断值 KL_D_MAX={KL_D_MAX}，exp 最多 {__import__('math').exp(KL_D_MAX):.0f} ✓")


def _case_f() -> None:
    """★ ratio 断言的两条判据 —— 直接复现了 seed1 的误杀 ✓✓

    实测（seed1 崩溃）：最大偏差 **0.0023**（> 旧的 1e-3 容差 ✗）而均值只有 ~2e-4 ✓
    根因：`logprobs_old` 来自 rollout 的**整批** forward ✓
          而 `lp_new` 来自 ppo_update 的**分块** forward ✓
          → 张量形状不同 → kernel 归约顺序不同 → 舍入不同 ✓（**不是 bug** ✗）
    真 bug 的量级：**0.9811** / **195** ✓✓（大 2~4 个数量级 ✓）
    → 所以 **均值 1e-3 + 最大值 0.02** 既不再误杀 ✓ 又仍抓得住真错位 ✓✓
    """
    n = 10
    mask = torch.ones(1, n, dtype=torch.bool)
    lp_old = torch.zeros(1, n)
    d_vec = torch.full((1, n), 1e-5)          # 正常舍入
    d_vec[0, 0] = math.log(1.0 + 0.0023)      # 复现 seed1 的那个毛刺
    try:
        dev = assert_ratio_starts_at_one(lp_old + d_vec, lp_old, mask)
    except AssertionError as e:
        raise AssertionError(f"0.0023 的舍入毛刺又被误杀 ✗（旧代码会这样）：{e}")
    print("    最大 0.0023 / 均值 ~2e-4 -> 通过 ✓（max=" + format(dev, ".4f") + "）")

    bad = torch.zeros(1, n)
    bad[0, 0] = math.log(1.5)
    caught = False
    try:
        assert_ratio_starts_at_one(lp_old + bad, lp_old, mask)
    except AssertionError:
        caught = True
    if not caught:
        raise AssertionError("0.5 的真错位竟然没被抓住 ✗✗")
    print("    真错位（最大 0.5）-> 仍然报错 ✓✓")


def _case_a() -> None:
    """ratio ≡ 1：策略损失必须退化成 ``-masked_mean(A)``。

    这是最基本的一致性检查。ratio=1 时 `surr1 = surr2 = A`，
    所以 policy = -E[A]。如果连这条都不成立，说明 ratio 算错了。

    顺便验证 mask：A = [1,2,3,4] 但只前 3 个有效 → 仍然是 -(1+2+3)/3 = -2.0，
    而不是 -2.5。
    """
    cfg = _cfg()
    lp = torch.zeros(1, 4)
    adv = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, True, False]])
    out = ppo_loss(lp, lp, adv, torch.zeros(1, 4), torch.zeros(1, 4),
                   torch.zeros(1, 4), mask, cfg)

    _assert_scalar("ratio≡1 的策略损失", out.policy, -2.0)
    _assert_scalar("ratio≡1 的 mean_ratio", out.mean_ratio, 1.0)
    _assert_scalar("ratio≡1 的 clip_frac", out.clip_frac, 0.0)
    _assert_scalar("ratio≡1 的 approx_kl", out.approx_kl, 0.0)
    _assert_scalar("total（各项系数为 0 时应等于 policy）", out.total, -2.0)


def _case_b() -> None:
    """裁剪的不对称性：四个精确数字。

    ε = 0.2，所以 clip 区间是 [0.8, 1.2]。

    ratio = 3.0（超出上界）：
        A = +1 → surr1 = 3.0,  surr2 = 1.2 → min = 1.2  → policy = -1.2  （裁掉了）
        A = -1 → surr1 = -3.0, surr2 = -1.2 → min = -3.0 → policy = +3.0  （**没裁**）

    ratio = 0.5（超出下界）：
        A = +1 → surr1 = 0.5,  surr2 = 0.8  → min = 0.5  → policy = -0.5  （**没裁**）
        A = -1 → surr1 = -0.5, surr2 = -0.8 → min = -0.8 → policy = +0.8  （裁掉了）

    看规律：**"好动作被抬太高"和"坏动作被砍太狠"会被裁；反过来不会。**
    这正是 PPO 想要的：限制单次更新的幅度，但不阻止它远离坏动作。

    approx_kl 也一起验：ratio=3 → (3-1) - ln3 = 0.9014；ratio=0.5 → (0.5-1) - ln0.5 = 0.1931
    """
    import math

    cfg = _cfg()
    for ratio_v, adv_v, exp_policy in (
        (3.0, +1.0, -1.2),
        (3.0, -1.0, +3.0),
        (0.5, +1.0, -0.5),
        (0.5, -1.0, +0.8),
    ):
        lp_old = torch.zeros(1, 1)
        lp_new = torch.full((1, 1), math.log(ratio_v))
        adv = torch.full((1, 1), adv_v)
        mask = torch.ones(1, 1, dtype=torch.bool)
        out = ppo_loss(lp_new, lp_old, adv, torch.zeros(1, 1), torch.zeros(1, 1),
                       torch.zeros(1, 1), mask, cfg)
        _assert_scalar(f"ratio={ratio_v} A={adv_v:+.0f} 的策略损失", out.policy, exp_policy)
        _assert_scalar(f"ratio={ratio_v} 的 clip_frac", out.clip_frac, 1.0)
        _assert_scalar(f"ratio={ratio_v} 的 ratio_max", out.ratio_max, ratio_v, tol=1e-4)
        exp_kl = (ratio_v - 1.0) - math.log(ratio_v)
        _assert_scalar(f"ratio={ratio_v} 的 approx_kl", out.approx_kl, exp_kl, tol=1e-4)


def _case_c() -> None:
    """KL 方向：k3 恒非负，k1 可以是负数。

    d = logp_ref - logp

        logp == logp_ref            → d = 0    → k1 = 0,      k3 = 0
        logp = 1.0, logp_ref = 0.5  → d = -0.5 → k1 = -0.5 ❗, k3 = e^-0.5 + 0.5 - 1 = 0.106531
        logp = 0.5, logp_ref = 1.0  → d = +0.5 → k1 = +0.5,   k3 = e^0.5 - 0.5 - 1  = 0.148721

    第二行就是 k1 的病：新策略比 reference **更**偏向采样到的 token 时，
    它给出负的"KL" → 惩罚项变成奖励项。训练早期几乎一直在发生这件事。
    """
    import math

    # 用 (1, 3) 而不是 (3,)：_assert_close 的期望值形状是 (1, N)，本框架里
    # 所有张量也都是 (B, T)。形状契约由上方的循环单独测，这里只测数值。
    lp = torch.tensor([[1.0, 1.0, 0.5]])    # 位置1: logp(1.0) > ref(0.5) → d = -0.5
    ref = torch.tensor([[1.0, 0.5, 1.0]])   # 位置2: logp(0.5) < ref(1.0) → d = +0.5

    # ---- 契约检查：逐 token 函数必须保持形状不变 ----
    # （之前的版本漏了这条，所以 unsqueeze 之类的形状 bug 能溜过去）
    for shape in ((3,), (2, 4)):
        z = torch.zeros(*shape)
        for est in ("k1", "k3"):
            out_shape = tuple(kl_penalty(z, z, est).shape)
            if out_shape != shape:
                raise AssertionError(
                    f"kl_penalty(estimator='{est}') 改变了形状："
                    f"输入 {shape} → 输出 {out_shape}\n"
                    f"        它是**逐 token** 函数，返回形状必须和输入一致"
                    f"（否则逐 token 使用时会被广播成别的形状，静默出错）"
                )
    print("    形状保持：输入 (3,) / (2,4) 都原样返回  ✅")

    k1 = kl_penalty(lp, ref, "k1")
    k3 = kl_penalty(lp, ref, "k3")
    _assert_close("k1（第二个位置是负的）", k1, [0.0, -0.5, 0.5])
    _assert_close("k3", k3, [0.0, math.exp(-0.5) + 0.5 - 1.0, math.exp(0.5) - 0.5 - 1.0],
                  tol=1e-5)
    if bool((k3 < -1e-9).any()):
        raise AssertionError("k3 出现了负值 —— 它必须恒 ≥ 0，检查符号约定")

    # 常数上的恒等：k3 = exp(d) - d - 1，在 d=0 处取最小值 0
    d = torch.linspace(-3, 3, 61)
    vals = kl_penalty(torch.zeros_like(d), d, "k3")     # logp=0, logp_ref=d → d 就是 d
    if bool((vals < -1e-9).any()) or abs(float(vals.min())) > 1e-9:
        raise AssertionError("k3 的最小值应该在 d=0 处取到 0")
    print("    k3 在 d∈[-3,3] 上恒 ≥ 0 且最小值在 d=0 处取到  ✅")


def _case_d() -> None:
    """mask 生效：带垃圾 padding 的 batch 必须和"压实的 batch"结果完全一致。

    这是最有力的一条 mask 测试 —— 它同时抓住两类错误：
      * 用 `.mean()` 而不是 `masked_mean`（分母变成 numel）
      * padding 的脏值漏进统计量

    构造：5 个有效 token，塞进 (2,4) 的 batch 里，padding 位置填 999（垃圾值）。
          再把这 5 个 token 压实成 (1,5) 的 batch。两者的每一项损失都必须相同。
    """
    cfg = _cfg(clip_eps=0.2, value_clip_eps=0.2, value_coef=0.5,
               entropy_coef=0.01, kl_coef=0.1)
    # 5 个有效 token 的真实数据
    valid_lp_new = [0.10, -0.20, 0.30, -0.40, 0.50]
    valid_lp_old = [0.05, -0.10, 0.20, -0.30, 0.40]
    valid_adv = [1.0, -2.0, 3.0, -4.0, 5.0]
    valid_ret = [0.5, 1.5, -0.5, 2.0, 1.0]
    valid_v_new = [0.2, 0.4, 0.6, 0.8, 1.0]
    valid_v_old = [0.1, 0.3, 0.5, 0.7, 0.9]
    valid_ent = [0.5, 0.6, 0.7, 0.8, 0.9]
    valid_lp_ref = [0.08, -0.18, 0.28, -0.38, 0.48]

    # ---- 压实版：(1, 5) ----
    compact = dict(
        lp_new=torch.tensor([valid_lp_new]), lp_old=torch.tensor([valid_lp_old]),
        adv=torch.tensor([valid_adv]), ret=torch.tensor([valid_ret]),
        v_new=torch.tensor([valid_v_new]), v_old=torch.tensor([valid_v_old]),
        ent=torch.tensor([valid_ent]), lp_ref=torch.tensor([valid_lp_ref]),
        mask=torch.ones(1, 5, dtype=torch.bool),
    )
    # ---- 填充版：(2, 4)，前 3 + 后 2 个有效，其余填 999 ----
    def padded(vals):
        t = torch.full((2, 4), 999.0)
        t[0, :3] = torch.tensor(vals[:3])
        t[1, :2] = torch.tensor(vals[3:])
        return t

    padded_batch = dict(
        lp_new=padded(valid_lp_new), lp_old=padded(valid_lp_old),
        adv=padded(valid_adv), ret=padded(valid_ret),
        v_new=padded(valid_v_new), v_old=padded(valid_v_old),
        ent=padded(valid_ent), lp_ref=padded(valid_lp_ref),
        mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
    )

    out_c = ppo_loss(compact["lp_new"], compact["lp_old"], compact["adv"],
                     compact["ret"], compact["v_new"], compact["v_old"],
                     compact["mask"], cfg, entropy=compact["ent"],
                     logprobs_ref=compact["lp_ref"])
    out_p = ppo_loss(padded_batch["lp_new"], padded_batch["lp_old"], padded_batch["adv"],
                     padded_batch["ret"], padded_batch["v_new"], padded_batch["v_old"],
                     padded_batch["mask"], cfg, entropy=padded_batch["ent"],
                     logprobs_ref=padded_batch["lp_ref"])

    dc, dp = out_c.detached(), out_p.detached()
    bad = [k for k in dc if abs(dc[k] - dp[k]) > 1e-4]
    if bad:
        detail = "\n".join(
            f"        {k:<12} 压实={dc[k]:+.6f}  填充={dp[k]:+.6f}  差={dp[k]-dc[k]:+.2e}"
            for k in bad
        )
        raise AssertionError(
            f"带 padding 的 batch 和压实的 batch 结果不一致，这些项不对：\n{detail}\n"
            f"        → padding 位置漏进了 reduction。所有项都要用 masked_mean"
            f"（分母是有效 token 数，不是 numel()）"
        )
    print(f"    压实 vs 填充：9 项全部一致  ✅  (policy={dc['policy_loss']:.6f})")


def _fmt(t: Tensor, nd: int = 4) -> str:
    """把任意形状的张量拍平成人能看的字符串。

    （之前的版本直接写 `[round(x,4) for x in got.tolist()]`，只要 `got` 是二维
      就会 `TypeError: type list doesn't define __round__` —— 断言明明通过了，
      却死在打印上，还把整个测试进程带崩。测试辅助函数也会写错。）
    """
    return str([round(float(v), nd) for v in t.flatten().tolist()])


def _assert_scalar(name: str, got: Tensor, exp: float, tol: float = 1e-5) -> None:
    """断言一个**标量**（0 维）张量。

    `ppo_loss` 的所有输出 —— 5 个损失分量和 4 个诊断量 —— 都是标量，
    因为它们都是 `masked_mean(...)` 的结果（对整个 batch 求了均值）。
    所以这里不能用 `_assert_close`（那个是给 (B,T) 逐 token 值用的）。

    这个检查同时兜住"返回 Python 标量而不是张量"的情况 —— 那样 `.detach()`
    和 `.backward()` 都会失败，而且报错位置离真正的原因很远。
    """
    if not isinstance(got, Tensor):
        raise AssertionError(
            f"{name}: 期望张量，实际是 {type(got).__name__} —— "
            f"关掉的项要返回零张量 `torch.zeros((), ...)`，不要返回 Python 的 0"
        )
    if got.dim() != 0:
        raise AssertionError(
            f"{name}: 损失/诊断量应该是 0 维标量，实际形状 {tuple(got.shape)}\n"
            f"        （如果返回的是 (B,T)，说明少了一层 masked_mean）"
        )
    exp_t = torch.tensor(float(exp), dtype=got.dtype, device=got.device)
    if not torch.isclose(got, exp_t, atol=tol):
        raise AssertionError(
            f"{name} 对不上\n    期望 {exp:+.6f}\n    实际 {float(got):+.6f}"
        )
    print(f"    {name:<34} ✅ {float(got):+.6f}")


def _assert_close(name: str, got: Tensor, exp: list[float], tol: float = 1e-5) -> None:
    """断言 `got` 逐元素等于 `exp`。

    ⚠️ **隐含假设：`got` 是 `(1, N)` 形状**（期望值被造成 `torch.tensor([exp])`）。
    本框架里所有张量都是 `(B, T)`，而这里的用例都是 B=1，所以够用。
    传一维 `(N,)` 输入会得到"形状应为 (1,N)，实际 (N,)"的报错 —— 那是**测试写错了**，
    不是实现错了。形状契约请用独立的形状检查（见 `_case_c`），不要靠这个辅助函数。
    """
    exp_t = torch.tensor([exp], dtype=got.dtype)
    if got.shape != exp_t.shape:
        raise AssertionError(f"{name}: 形状应为 {tuple(exp_t.shape)}，实际 {tuple(got.shape)}")
    if not torch.allclose(got, exp_t, atol=tol):
        raise AssertionError(
            f"{name} 对不上\n    期望 {_fmt(exp_t, 6)}\n    实际 {_fmt(got, 6)}"
        )
    print(f"    {name:<34} ✅ {_fmt(got, 4)}")


if __name__ == "__main__":
    cases = [("用例 A：ratio≡1 的退化", _case_a),
             ("用例 B：裁剪的不对称性（四个精确数字）", _case_b),
             ("用例 C：KL 方向（k3 ≥ 0，k1 可负）", _case_c),
               ("用例 D：mask 生效（压实 vs 填充）", _case_d),
               ("用例 E：k3 的 exp 截断（防梯度爆炸）", _case_e),
               ("用例 F：ratio 断言的双判据（复现 seed1 的误杀）", _case_f)]
    passed = 0
    for title, fn in cases:
        print(f"\n{title}")
        try:
            fn()
            passed += 1
        except NotImplementedError as e:
            print(f"    ⏳ 还没实现：{e}")
        except AssertionError as e:
            print(f"    ❌ {e}")
        except Exception as e:      # 一个用例崩掉不该带走后面所有用例
            print(f"    💥 未预期的异常 {type(e).__name__}: {e}")
    print(f"\n{'=' * 60}")
    if passed == len(cases):
        print(f"🎉 {len(cases)} 个用例全部通过 —— loss.py 可以进下一步了")
    else:
        print(f"进度：{passed}/{len(cases)} 通过（NotImplementedError 是正常的，去写实现吧）")
    # ★ 失败必须让**退出码非 0** ✗（此前被 try/except 吞掉 → echo $? 恒为 0 ✗）
    sys.exit(0 if passed == len(cases) else 1)

