"""dpo.py —— **Online DPO 的核心**（损失 + 配对 ✓ 纯函数 ✓ 可 CPU 自测 ✓✓）

## 为什么先写这个 ✗

这几天一半的 bug 都在"数学对、接线错"上 ✗（ratio 恒 1 ✗ · seed 没传 ✗ · mask 错一位 ✗）。
→ DPO 是**新算法** ✓ → 所以先把**能自测的部分**（loss 和 pair 构造）**钉死** ✓✓，
   再去接训练循环 ✓ —— 出问题时能**排除掉一半** ✓✓

## 算法（online / self-pair ✓）

    每步：采 B 题 × G 条 ✓
      → 每题**内部配对**：对的当正 ✓ 错的当负 ✓（**同一分布** ✓✓ 这是关键 ✗）
      → 攒满 N 对 → **一次 DPO 更新** ✓
      → 全对 / 全错的题**跳过** ✗（没有 pair ✓）

    L = −log σ( β·[ (logπ(y⁺) − logπ_ref(y⁺)) − (logπ(y⁻) − logπ_ref(y⁻)) ] )

## 为什么正样本**必须**是模型自己的 ✓✓（这点最容易搞错 ✗）

    DPO 最大化的是"**正样本相对负样本**的优势" ✓
    → 若正样本来自外部（官方解答 ✗）→ 正负差在**风格/长度**上 ✗ → 学到的是"模仿文风" ✗✗
    → GSM8K 那次 −15 点就是这个错 ✓（官方 0.827 → SFT 0.672 ✗）
    → ★ 正负都来自**同一次采样** ✓ → 只差"对错" ✓✓ → 学到的才是"为什么对" ✓✓
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# =========================================================================== #
# 1. DPO 损失 ✓
# =========================================================================== #
def dpo_loss(
    logp_pos: torch.Tensor,          # (N,) 策略对**正样本**的 logπ ✓
    logp_neg: torch.Tensor,          # (N,) 策略对**负样本**的 logπ ✓
    ref_logp_pos: torch.Tensor,      # (N,) 参考模型对正样本 ✓
    ref_logp_neg: torch.Tensor,      # (N,) 参考模型对负样本 ✓
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 `(loss, reward_margin)` ✓

    `loss` = −log σ(β·[(logp_pos − ref_pos) − (logp_neg − ref_neg)]) ✓
    `margin` = 上面 σ 里面那个量 ✓（诊断用 ✓ 它 > 0 表示"正样本相对更被偏好" ✓✓）

    ★ 注意 `beta` 的方向 ✗：**β 越大越"信任"参考模型** ✓（对偏好差更敏感 ✗）
      β=0.1 是 DPO 论文的默认 ✓；β→0 退化成"不看参考" ✗
    """
    pi_margin = logp_pos - logp_neg                    # (N,) 策略自己的偏好 ✓
    ref_margin = ref_logp_pos - ref_logp_neg           # (N,) 参考模型的偏好 ✓
    margin = pi_margin - ref_margin                    # ★ 相对参考的**增量** ✓
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, margin.detach()


def dpo_accuracy(margin: torch.Tensor, beta: float = 0.1) -> float:
    """★ 诊断：**有多少 pair 是"正样本被判得更高"** ✓（>0.5 才算在学 ✓✓）"""
    return float((beta * margin > 0).float().mean())


# =========================================================================== #
# 2. 从"同一题的多次采样"里配对 ✓
# =========================================================================== #
def build_pairs(
    samples: list[dict],
    max_pairs: int | None = None,
    mode: str = "random",
    seed: int = 0,
) -> list[tuple[dict, dict]]:
    """★ 把"一组采样"变成 DPO 的 `(正, 负)` 对 ✓✓

    Args:
        samples: 每条形如 `{"text": str, "correct": bool, "prompt": str, ...}` ✓
                 **同一题的所有采样必须相邻且在同一个 group 里** ✓（调用方保证 ✓）
        mode: `"random"`（随机取 1 对 ✓）· `"shortest"`（取**最短**的一对 ✓ 省显存 ✓）

    ★ 关键规则 ✗：
        · 只从**同一题**里配 ✓（跨题配对 = 学"哪道题更容易" ✗ 完全错 ✗✗）
        · 只配 **对的 vs 错的** ✓（对 vs 对 没有梯度 ✗）
        · 全对/全错的题**跳过** ✗（没 pair ✓）

    返回：`[(positive, negative), ...]` ✓（负样本在前 ✗ 别搞反 ✓✓）
    """
    import random as _r
    rng = _r.Random(seed)

    # 按 prompt 分组 ✓（保持出现顺序 ✓ 可复现 ✓）
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for s in samples:
        k = str(s.get("prompt", ""))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(s)

    pairs: list[tuple[dict, dict]] = []
    for k in order:
        g = groups[k]
        pos = [s for s in g if s.get("correct")]
        neg = [s for s in g if not s.get("correct")]
        if not pos or not neg:
            continue                                  # ★ 没有混合 → 跳过 ✓
        if mode == "shortest":
            # 最短的一对 ✓（长序列最吃显存 ✓ 也是 DPO 里偏差最小的 ✓）
            p = min(pos, key=lambda s: len(str(s.get("text", ""))))
            n = min(neg, key=lambda s: len(str(s.get("text", ""))))
        else:
            p = rng.choice(pos)
            n = rng.choice(neg)
        pairs.append((p, n))

    if max_pairs is not None and len(pairs) > max_pairs:
        rng.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


# =========================================================================== #
# 3. 自测（纯 CPU ✓ 不需要模型 ✓）
# =========================================================================== #
def _self_test() -> int:
    fails = 0

    def ck(cond, msg):
        nonlocal fails
        print(f"  {'✓' if cond else '✗✗'} {msg}")
        fails += 0 if cond else 1

    print("① loss 的方向（最要紧 ✓✓）")
    # 正样本被判得更高、参考不变 → margin > 0 → loss 应该小 ✓
    lp_pos = torch.tensor([0.9, 0.9])
    lp_neg = torch.tensor([-0.9, -0.9])
    ref_pos = torch.tensor([0.0, 0.0])
    ref_neg = torch.tensor([0.0, 0.0])
    loss_good, m_good = dpo_loss(lp_pos, lp_neg, ref_pos, ref_neg, beta=0.1)
    loss_bad, m_bad = dpo_loss(lp_neg, lp_pos, ref_pos, ref_neg, beta=0.1)
    ck(m_good.mean() > 0, f"正样本更高时 margin > 0（{m_good.mean():.3f}）")
    ck(loss_good < loss_bad, f"loss：正确配对 {loss_good:.4f} < 对调 {loss_bad:.4f} ✓")
    ck(dpo_accuracy(m_good) == 1.0, f"dpo_accuracy = {dpo_accuracy(m_good)} ✓")
    ck(dpo_accuracy(m_bad) == 0.0, f"对调后 accuracy = {dpo_accuracy(m_bad)} ✓")

    print("\n② 完全一样 → margin 0 · loss = log2 = 0.6931 ✓")
    z = torch.zeros(3)
    loss_z, m_z = dpo_loss(z, z, z, z, beta=0.5)
    ck(abs(m_z.mean().item()) < 1e-6, f"margin = {m_z.mean():.6f} ✓")
    ck(abs(loss_z.item() - math.log(2)) < 1e-5,
       f"loss = {loss_z.item():.4f}（应 = log 2 = {math.log(2):.4f}）✓")

    print("\n③ ★ 参考模型也在变：只看**增量** ✓")
    # 策略和参考都偏好正样本 → 增量 0 → margin 0 ✓（说明 DPO 只奖励"比参考更好" ✓）
    p_, n_ = torch.tensor([0.5]), torch.tensor([-0.5])
    loss_same, m_same = dpo_loss(p_, n_, p_.clone(), n_.clone(), beta=1.0)
    ck(abs(m_same.item()) < 1e-6, f"策略==参考 → margin {m_same.item():.6f} ✓")
    # 策略**多**偏好了一点 → margin > 0 ✓
    loss_inc, m_inc = dpo_loss(torch.tensor([0.7]), n_, p_.clone(), n_.clone(), beta=1.0)
    ck(m_inc.item() > 0, f"比参考更偏好 → margin {m_inc.item():.3f} > 0 ✓")

    print("\n④ beta 的作用（大 β → 更敏感 ✓）")
    _, m1 = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), z[:1], z[:1], beta=0.1)
    l1, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), z[:1], z[:1], beta=0.1)
    l2, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), z[:1], z[:1], beta=5.0)
    ck(l2 < l1, f"β=5 的 loss {l2:.4f} < β=0.1 的 {l1:.4f} ✓（更信任偏好 ✓）")

    print("\n⑤ ★★ 配对：只从同题里配、只配 对/错 ✓")
    samples = [
        {"prompt": "Q1", "text": "a1", "correct": True},
        {"prompt": "Q1", "text": "b1", "correct": False},
        {"prompt": "Q2", "text": "a2", "correct": True},     # 全对 → 跳过 ✓
        {"prompt": "Q2", "text": "b2", "correct": True},
        {"prompt": "Q3", "text": "a3", "correct": False},    # 全错 → 跳过 ✓
        {"prompt": "Q3", "text": "b3", "correct": False},
        {"prompt": "Q4", "text": "a4", "correct": False},
        {"prompt": "Q4", "text": "b4", "correct": True},
    ]
    pairs = build_pairs(samples, seed=0)
    ck(len(pairs) == 2, f"只出 2 对（Q1/Q4）✓ 实得 {len(pairs)}")
    ck(all(p["correct"] and not n["correct"] for p, n in pairs),
       "每对都是 (对, 错) ✓✓")
    ck(all(p["prompt"] == n["prompt"] for p, n in pairs),
       "每对**同题** ✓✓（跨题配对是致命的 ✗）")
    qs = sorted(p["prompt"] for p, _ in pairs)
    ck(qs == ["Q1", "Q4"], f"覆盖 {qs} ✓（Q2 全对 / Q3 全错 都被跳过 ✓）")

    print("\n⑥ 可复现（同 seed → 同结果 ✓）")
    a = build_pairs(samples, seed=7)
    b = build_pairs(samples, seed=7)
    ck([(x["text"], y["text"]) for x, y in a] == [(x["text"], y["text"]) for x, y in b],
       "同 seed 结果一致 ✓")

    print("\n⑦ max_pairs 生效 ✓")
    many = []
    for i in range(50):
        many.append({"prompt": f"q{i}", "text": "p", "correct": True})
        many.append({"prompt": f"q{i}", "text": "n", "correct": False})
    got = build_pairs(many, max_pairs=10, seed=0)
    ck(len(got) == 10, f"截到 10 对 ✓ 实得 {len(got)}")

    print()
    if fails:
        print(f"✗✗ {fails} 项失败")
        return 1
    print("✅ 全部通过 ✓✓ —— 损失的数学和配对规则都钉住了 ✓")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
