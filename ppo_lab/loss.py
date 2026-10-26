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

import torch
from torch import Tensor

from batch import masked_mean
from config import PPOConfig


# =========================================================================== #
# 1. KL 估计量（逐 token）
# =========================================================================== #
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
        return d.exp() - d - 1

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
    entropy: Tensor      # **原始** E[H]（≥0，可直接画曲线）；
                         # 进 loss 的是 -c_ent·E[H]，见 total

    # ---- 诊断（detach，不反传）----
    clip_frac: Tensor    # 被裁剪的 token 比例。**持续 > 0.3 说明步子太大**
    mean_ratio: Tensor   # E[r]，应该围绕 1.0
    approx_kl: Tensor    # E[(r-1) - log r]，恒 ≥ 0。**> 0.02 通常该降 lr**
    ratio_max: Tensor    # max(r)，用来发现离群样本

    def detached(self) -> dict[str, float]:
        """转成可直接打日志的普通 dict。"""
        return {
            "loss": float(self.total.detach()),
            "policy_loss": float(self.policy.detach()),
            "value_loss": float(self.value.detach()),
            "kl": float(self.kl.detach()),
            "entropy": float(self.entropy.detach()),
            "clip_frac": float(self.clip_frac.detach()),
            "mean_ratio": float(self.mean_ratio.detach()),
            "approx_kl": float(self.approx_kl.detach()),
            "ratio_max": float(self.ratio_max.detach()),
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
    if logprobs_ref is not None and cfg.kl_coef != 0:
        kl_vec = kl_penalty(logprobs_new, logprobs_ref, cfg.kl_estimator)
        kl = cfg.kl_coef * masked_mean(kl_vec, mask)
    else:
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
        mean_ratio = masked_mean(ratio, mask)
        approx_kl = masked_mean((ratio - 1) - log_ratio, mask)
        ratio_max = (ratio * mask).max()

    out = PPOOutput(
        total=total, 
        policy=policy,
        value=value,
        kl=kl, 
        entropy=ent, 
        clip_frac=clip_frac, 
        mean_ratio=mean_ratio,
        approx_kl=approx_kl,
        ratio_max=ratio_max
    )

    return out


    raise NotImplementedError("写 PPO 的四项损失 + 四个诊断量")


# =========================================================================== #
# 4. 诊断工具（管道，已实现）
# =========================================================================== #
def assert_ratio_starts_at_one(
    logprobs_new: Tensor, logprobs_old: Tensor, mask: Tensor, tol: float = 1e-3
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
    if dev > tol:
        # 自动诊断：如果所有超差位置上的 logprobs_old 都恰好是 0，
        # 那几乎一定是 mask 传错了板子 —— 传成了"真实 token"mask，把 prompt 位置
        # 也算进来了，而 prompt 位置的 logprobs_old 在 rollout 里从没被填过（恒为 0）。
        bad = dev_t > tol
        n_bad = int(bad.sum())
        n_zero_old = int((bad & (logprobs_old == 0)).sum())
        hint = ""
        if n_bad and n_zero_old == n_bad:
            hint = (f"\n        ⚠️ 自动诊断：{n_bad} 个超差位置上的 logprobs_old **全为 0**。\n"
                    f"           这几乎可以断定是 **mask 传错了**：这里要的是"
                    f"「生成位置」的 mask（`batch.mask`），\n"
                    f"           不是 `forward` 用的「真实 token」mask"
                    f"（`prompt_mask | mask`）。prompt 位置在 rollout 里没被填过。")
        raise AssertionError(
            f"ratio 起始偏差 {dev:.4f} > {tol}：old/new logprob 的计算路径不一致！\n"
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
             ("用例 D：mask 生效（压实 vs 填充）", _case_d)]
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
