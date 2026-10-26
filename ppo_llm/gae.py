"""GAE（广义优势估计）+ 优势归一化。

这是一个**纯函数模块**：只吃张量、吐张量，不认识任何模型。
所以你可以在毫秒级用手算值对拍它 —— 这是整个框架里最该先写对的文件。

公式（Schulman et al. 2016）
---------------------------
    δ_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
    A_t = δ_t + γ·λ·(1 - done_t)·A_{t+1}

    λ=0  → A_t = δ_t                    （一步 TD 误差：低方差、高偏差）
    λ=1  → A_t = 回报 - V(s_t)          （无偏、高方差）
    0<λ<1 → 折中

估值目标（value target）用 λ-return：
    returns_t = A_t + V(s_t)

注意 λ<1 时 `returns` **不等于**蒙特卡洛回报，这是正确的 —— 它本来就是
"λ 加权的多步回报"，本来就在偏差和方差之间做交易。别去"修正"它。


和你之前那份 PPO 代码的差异（你说感觉差很远，对照这张表就清楚了）
----------------------------------------------------------------
你原来的递推长这样：

    G = nextStateValue[idx][-1].detach()
    rewardNow = reward[idx]
    for i in reversed(range(len(stateValue[idx]))):
        G = rewardNow + 0.99 * G
        rewardNow = 0
        A = G - stateValue[idx][i]
    ...
    advantageNow = (advantageNow - advantageNow.mean()) / (advantageNow.std() + 1e-8)

| # | 旧代码 | 这里 | 为什么必须改 |
|---|---|---|---|
| 1 | 递推只有 γ，没有 λ | `lam` 是一个显式参数 | 旧代码其实是 **λ=1** 的特例（MC 回报 + bootstrap）。λ 是你唯一能调偏差/方差的旋钮 |
| 2 | 没有 `done` 概念 | `(1-done_t)` 出现**两次** | 旧代码把所有回合都当成"还能继续"，终止步还在 bootstrap → 优势系统性偏大 |
| 3 | 没有 `mask` | 最后 `adv = adv * mask` | padding 位置的δ是垃圾；不清零会污染后面的归一化统计量 |
| 4 | 在**单条序列内**减均值除标准差 | `normalize_advantages(..., mode="batch")` | 旧代码把"这条回答好不好"这个序列级信息整体减掉了，见 `normalize_advantages` 的 docstring |
| 5 | γ=0.99 硬编码 | `gamma` 是参数，默认 1.0 | 单轮生成任务里折扣 = 长度偏置（0.99²⁰⁰≈13%） |
| 6 | 递推写在训练循环里 | 独立的纯函数 | 旧代码无法被测试，所以你没法知道它算错了 |

第 6 条是最根本的：**算法一旦埋在双层循环里，你就失去了对拍它的能力。**
"""

from __future__ import annotations

from typing import Literal

import sys

import torch
from torch import Tensor

from batch import masked_mean, masked_std


# =========================================================================== #
# 1. GAE
# =========================================================================== #
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    mask: Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """反向递推算出优势与 λ-return。

    参数
    ----
    rewards, values, dones, mask : (B, T)
    next_values : (B, T)，``next_values[b, t] = V(s_{t+1})``。
        trainer 里用 ``batch.next_values_from_values()``... 更正：它是 ``batch.py``
        的**模块级函数** ``next_values_from_values(values, mask)``，
        用 mask 定位而不是切片。
    gamma, lam : 见模块 docstring 的公式。

    返回
    ----
    (advantages, returns)，都是 (B, T)，且 padding 位置严格为 0。

    三个坑（先想清楚再写）
    ---------------------
    1. **`(1-done)` 要出现在两个地方，作用不同。**
       - 出现在 δ 里：终止步不能 bootstrap（丢掉 V(s_{t+1})）
       - 出现在 λ 的传播里：回合结束后，后面的 A 必须从 0 重新开始
       少写哪一个，都不会报错，只会让优势偏大。`_case_b()` 专门钉这个。
    2. **反向递推的 carry 初始化成什么？** 以及每个 t 上 carry 是"先算 δ 再乘 λ"还是
       反过来 —— 差一个下标就是另一个算法。
    3. **最后为什么要乘 mask？** 不乘的话 padding 位置会留下非零的 δ/σ 分量，
       污染 `normalize_advantages(mode="batch")` 的均值和标准差。

    实现提示（不是必须）
    -------------------
    形状是 (B, T)，T 通常只有几十到几百，所以**对 T 写一个反向 for 循环**是最清晰
    的写法（B 维向量化）。别为了炫技把它写成花哨的矩阵运算 —— 这个函数的可读性
    比速度重要得多。

    实现完成后，直接跑 ``python3 gae.py`` 会用两组手算值对拍你。
    """

    batch_size = rewards.shape[0]

    advantages = torch.zeros_like(rewards, device=rewards.device)
    returns = torch.zeros_like(rewards, device=rewards.device)

    for sample in range(batch_size):
        advantage = 0
        for idx in range(len(mask[sample]) -1, -1, -1):
            if not mask[sample, idx]:
                continue
            delta = rewards[sample][idx] + gamma * (~dones[sample][idx]) * next_values[sample][idx] - values[sample][idx]
            advantage = delta + gamma * lam * (~dones[sample][idx]) * advantage
            advantages[sample][idx] = advantage

    returns = advantages + values
    return advantages, returns            

# =========================================================================== #
# 2. 优势归一化
# =========================================================================== #
def normalize_advantages(
    adv: Tensor,
    mask: Tensor,
    mode: Literal["batch", "none", "per_sequence"] = "batch",
    eps: float = 1e-8,
) -> Tensor:
    """优势归一化。

    ``"batch"``（默认，正确）
        在**整个 batch 的所有有效 token** 上做 ``(A - μ) / σ``。
        保留"这条轨迹整体好不好"的尺度，只把它放到稳定量纲上。
        μ、σ 用 `masked_mean` / `masked_std`（它们已经会忽略 padding）。

    ``"none"``
        原样返回。critic 本身就是 baseline。

    ``"per_sequence"``（⚠️ 反面教材，故意保留）
        在**每一条序列内部**减均值、除标准差。你旧代码就是这么写的。

        它的致命之处：**归一化后，每条序列的优势均值恰好是 0**。当奖励稀疏
        （只在终局）时，"答对的序列"和"答错的序列"减去各自均值后，形状几乎
        完全一样 —— 模型**无法区分哪条回答是好的**。剩下的只是 critic 在序列内
        的形状，而不是"这条轨迹整体好不好"。

        一句话：`per_sequence` 把**序列之间**的信号（唯一重要的那个信号）抹掉了，
        只留下**序列内部**的起伏。

        注意别用"给奖励加常数"来理解这个差别 —— `batch` 和 `per_sequence` 都是
        仿射归一化，对全局常数**都不变**。正确的判据是：**把"答对"和"答错"两条
        序列放在同一个 batch 里，归一化后它们的优势还区不区分得开。**
        `_case_c()` 就是钉这个的。

    别忘了最后乘回 mask。

    这个函数必须是**纯函数**：返回新张量，**不允许就地修改传进来的 adv**。
    理由有两个：
      * 调用方通常还要拿 `adv` 做日志/诊断，被改掉就再也看不到原始值了；
      * 如果 `adv` 带 autograd 图，就地操作会直接抛错。

    注意 ``adv *= mask``（原地，等于 ``adv.__imul__(mask)``）和
    ``adv = adv * mask``（重新绑定名字）**不是一回事**：
    前者改的是调用方手里的那个对象，后者只改本函数内的局部名字。
    ``_case_d()`` 用"调用前后对比"把这条钉住了。
    """

    if mode == "batch":
        adv = (adv - masked_mean(adv, mask, eps)) / (masked_std(adv, mask, eps) + eps)
        return adv * mask

    if mode == "none":
        return adv * mask

    if mode == "per_sequence": # wrong
        out = adv.clone()  
        batch_size = adv.shape[0]
        for sample in range(batch_size):
            out[sample] = (adv[sample] - masked_mean(adv[sample], mask[sample], eps)) / (masked_std(adv[sample], mask[sample], eps) + eps)
        return out * mask
    
    raise ValueError(f"未知的 mode: {mode}")


# =========================================================================== #
# 3. 手算用例（不要改这里的期望值，它们是我用纸笔算出来的）
# =========================================================================== #
def _case_a() -> None:
    """λ 扫描：同一个输入，λ=0 / 0.5 / 1 三种期望值。

    T=3, B=1, γ=1.0, 全都不终止
        rewards     = [0, 0, 1]
        values      = [0.2, 0.5, 0.8]
        next_values = [0.5, 0.8, 0.0]     # 最后一个位置是 bootstrap

    δ_2 = 1 + 0.0 - 0.8 =  0.2
    δ_1 = 0 + 0.8 - 0.5 =  0.3
    δ_0 = 0 + 0.5 - 0.2 =  0.3
    """
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.2, 0.5, 0.8]])
    next_values = torch.tensor([[0.5, 0.8, 0.0]])
    dones = torch.zeros(1, 3, dtype=torch.bool)
    mask = torch.ones(1, 3, dtype=torch.bool)

    expect = {
        # λ=0：A = δ，returns = δ + V = 一步 TD 目标
        0.0: ([0.3, 0.3, 0.2], [0.5, 0.8, 1.0]),
        # λ=0.5：A_2=0.2; A_1=0.3+0.5·0.2=0.4; A_0=0.3+0.5·0.4=0.5
        0.5: ([0.5, 0.4, 0.2], [0.7, 0.9, 1.0]),
        # λ=1：退化成 MC 回报 - V，returns 应该全部 = 未来奖励和 = 1.0
        1.0: ([0.8, 0.5, 0.2], [1.0, 1.0, 1.0]),
    }
    for lam, (exp_a, exp_r) in expect.items():
        adv, ret = compute_gae(rewards, values, next_values, dones, mask,
                               gamma=1.0, lam=lam)
        _assert_close(f"λ={lam} 优势", adv, exp_a)
        _assert_close(f"λ={lam} λ-return", ret, exp_r)


def _case_b() -> None:
    """done 的两个截断：这是最容易写错、又最难自己发现的一处。

    T=3, B=1, γ=1.0, λ=1.0
        rewards     = [0, 1, 0]
        values      = [0.2, 0.5, 0.9]
        next_values = [0.5, 99.0, 0.4]    # 中间那个 99 是"陷阱值"
        dones       = [F, T, F]           # 第 1 步之后回合终止

    第 1 步已经终止，所以 next_values[1]=99 是**不该被用到**的。
    正确结果：
        δ_1 = 1 + γ·99·(1-1) - 0.5 = 0.5
        A_1 = δ_1 + γλ·(1-1)·A_2 = 0.5          ← λ 传播也被截断
        δ_2 = 0 + 0.4 - 0.9 = -0.5；A_2 = -0.5
        δ_0 = 0 + 0.5 - 0.2 = 0.3；A_0 = 0.3 + 0.5 = 0.8
    """
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 0.5, 0.9]])
    next_values = torch.tensor([[0.5, 99.0, 0.4]])   # 99 绝不能被用上
    dones = torch.tensor([[False, True, False]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    adv, ret = compute_gae(rewards, values, next_values, dones, mask,
                           gamma=1.0, lam=1.0)
    _assert_close("done 截断：优势", adv, [0.8, 0.5, -0.5])
    if float(adv.abs().max()) > 10:
        raise AssertionError("用到了被 done 屏蔽的 bootstrap 值（那个 99）")


def _case_c() -> None:
    """归一化的致命差别：**答对和答错，归一化后还区分得开吗？**

    构造两条共享同一个 critic 剖面的序列，唯一的区别是终局奖励：
        行 0 = 答对（R=1），行 1 = 答错（R=0）

    正确行为：
      * mode="batch"        → 行 0 的优势整体为正、行 1 整体为负 —— **区分得开** ✅
      * mode="per_sequence" → 两行**逐点完全相同**（各自均值被减成 0）—— 区分不开 ❌

    你旧代码用的是后者。这意味着模型从梯度里**根本看不出哪条回答是对的**。

    （别用"给奖励加常数"来想这件事：batch 和 per_sequence 都是仿射归一化，
      对全局常数都不变。判据必须是"两条序列之间还区不区分得开"。）
    """
    values = torch.tensor([[0.1, 0.2, 0.3],
                           [0.1, 0.2, 0.3]])          # 两条序列共享同一个 critic
    next_values = torch.tensor([[0.2, 0.3, 0.3],
                                [0.2, 0.3, 0.3]])
    rewards = torch.tensor([[0.0, 0.0, 1.0],          # 行 0：答对
                            [0.0, 0.0, 0.0]])         # 行 1：答错
    dones = torch.zeros(2, 3, dtype=torch.bool)
    mask = torch.ones(2, 3, dtype=torch.bool)

    adv, _ = compute_gae(rewards, values, next_values, dones, mask, gamma=1.0, lam=1.0)
    print(f"    原始优势：答对 {adv[0].tolist()} / 答错 {adv[1].tolist()}")

    for mode, should_distinguish in (("batch", True), ("per_sequence", False)):
        n = normalize_advantages(adv, mask, mode=mode)
        row_ok, row_bad = n[0], n[1]
        distinguishable = not torch.allclose(row_ok, row_bad, atol=1e-6)
        verdict = "✅ 符合预期" if distinguishable == should_distinguish else "❌ 不符合预期"
        print(f"    mode={mode:<13} 答对均值={float(row_ok.mean()):+.3f} "
              f"答错均值={float(row_bad.mean()):+.3f} "
              f"区分得开={distinguishable}  {verdict}")
        if distinguishable != should_distinguish:
            raise AssertionError(
                f"mode={mode}: 答对/答错{'应该' if should_distinguish else '不应该'}"
                f"区分得开。\n"
                f"      提醒：per_sequence 会把每条序列的均值减成 0，"
                f"于是'答对'和'答错'在梯度里变得一模一样 —— 这正是你旧代码的问题。"
            )


def _case_d() -> None:
    """mask / padding —— **前三个用例都没覆盖到的地方（我写测试时的疏漏，现在补上）**。

    输入刻意做成 ``next_values_from_values()`` 的真实产物：padding 位置
    ``next_value = 0`` 但 ``value`` 非零 → padding 位置的 δ = -V ≠ 0。
    **这才是训练时真正会发生的情况。**

        行 0: mask [T,T,T,F]  rewards [0,0,2,0]  next [0.3,0.3,0,0]  dones [F,F,T,F]
        行 1: mask [T,T,F,F]  rewards [0,1,0,0]  next [0.3,0,0,0]    dones [F,T,F,F]
        values 全 0.3，γ=1，λ=1

    期望：有效优势 = [1.7,1.7,1.7,0.7,0.7]，padding 位置**严格 0**
          batch 归一化有效位置 = [0.8165,0.8165,0.8165,-1.2247,-1.2247]，padding 位置 0
    """
    rewards = torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    values = torch.full((2, 4), 0.3)
    next_values = torch.tensor([[0.3, 0.3, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]])
    dones = torch.tensor([[False, False, True, False], [False, True, False, False]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    problems: list[str] = []
    adv, _ = compute_gae(rewards, values, next_values, dones, mask, gamma=1.0, lam=1.0)

    if bool((adv[~mask].abs() > 1e-6).any()):
        problems.append(
            f"compute_gae 的 padding 位置优势不是 0：{adv[~mask].tolist()}\n"
            f"        → 陷阱 3。提示：`tensor(False) is False` 的结果是 False，"
            f"所以 `if mask[i][j] is False` 永远不会命中；用 `not bool(...)` 或 `~mask`"
        )
    exp_valid = torch.tensor([1.7, 1.7, 1.7, 0.7, 0.7])
    if not torch.allclose(adv[mask], exp_valid, atol=1e-5):
        problems.append(f"有效位置优势错误：期望 {exp_valid.tolist()}，实际 {adv[mask].tolist()}")
    else:
        print(f"    compute_gae padding 清零  ✅ 有效={[round(x, 2) for x in adv[mask].tolist()]}")

    adv_in = adv.clone()
    n = normalize_advantages(adv_in, mask, mode="batch")
    if bool(torch.isnan(n).any() | torch.isinf(n).any()):
        problems.append(f"归一化输出出现 NaN/inf：{n.tolist()}")
    else:
        if bool((n[~mask].abs() > 1e-6).any()):
            problems.append(
                f"归一化后 padding 位置不为 0：{n[~mask].tolist()}"
                f"（docstring 里写着「别忘了最后乘回 mask」）"
            )
        exp_n = torch.tensor([0.816497, 0.816497, 0.816497, -1.224745, -1.224745])
        if not torch.allclose(n[mask], exp_n, atol=1e-4):
            problems.append(
                f"batch 归一化的有效位置数值错误\n"
                f"        期望 {[round(x, 4) for x in exp_n.tolist()]}\n"
                f"        实际 {[round(x, 4) for x in n[mask].tolist()]}\n"
                f"        → 均值和标准差只能吃 mask=True 的 token；分母要加 eps 防除零"
            )
        else:
            print(f"    batch 归一化             ✅ 有效={[round(x, 4) for x in n[mask].tolist()]}")

    # 纯函数检查：不能就地改掉调用方手里的张量
    before = adv.clone()
    normalize_advantages(adv, mask, mode="per_sequence")
    if not torch.allclose(adv, before, atol=1e-9):
        problems.append(
            "normalize_advantages 就地修改了输入张量 → 会悄悄污染调用方手里的 adv\n"
            "        （必须写成纯函数：返回新张量，不改参数）"
        )

    if problems:
        raise AssertionError("\n    ".join(problems))


def _case_e() -> None:
    """零方差：全 batch 的 advantage 完全相同时，不能产生 NaN。

    **这在 RL 里一定会发生**：训练早期模型把所有题都做错（reward 全 0），
    或某次 rollout 恰好全是同一个结果 → std = 0 → `(A-μ)/σ` 变成 0/0 = **NaN**，
    NaN 一旦进梯度就会污染每个参数，训练当场报废。

    所以分母必须写成 `σ + eps` 或 `σ.clamp_min(eps)`。
    """
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.full((1, 3), 0.3)
    next_values = torch.full((1, 3), 0.3)
    dones = torch.zeros(1, 3, dtype=torch.bool)
    mask = torch.ones(1, 3, dtype=torch.bool)

    adv, _ = compute_gae(rewards, values, next_values, dones, mask, gamma=1.0, lam=1.0)
    print(f"    优势完全相同 {[round(x, 3) for x in adv.tolist()[0]]}，std = 0")
    for mode in ("batch", "per_sequence"):
        n = normalize_advantages(adv.clone(), mask, mode=mode)
        if bool(torch.isnan(n).any() | torch.isinf(n).any()):
            raise AssertionError(
                f"mode={mode} 出现 NaN/inf：{n.tolist()}\n"
                f"        零方差是 RL 的常态（全对 / 全错），分母必须加 eps 兜底"
            )
        print(f"    mode={mode:<13} ✅ 无 NaN，输出 {n.tolist()}")


def _assert_close(name: str, got: Tensor, exp: list[float], tol: float = 1e-5) -> None:
    exp_t = torch.tensor([exp], dtype=got.dtype)
    if got.shape != exp_t.shape:
        raise AssertionError(f"{name}: 形状应为 {tuple(exp_t.shape)}，实际 {tuple(got.shape)}")
    if not torch.allclose(got, exp_t, atol=tol):
        raise AssertionError(
            f"{name} 对不上\n    期望 {exp_t.tolist()}\n    实际 {got.tolist()}"
        )
    print(f"    {name:<24} ✅ {got.tolist()}")


if __name__ == "__main__":
    cases = [("用例 A：λ 扫描", _case_a),
             ("用例 B：done 的两个截断", _case_b),
             ("用例 C：归一化的致命差别", _case_c),
             ("用例 D：mask / padding（含纯函数检查）", _case_d),
             ("用例 E：零方差不能出 NaN", _case_e)]
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
    print(f"\n{'=' * 60}")
    if passed == len(cases):
        print(f"🎉 {len(cases)} 个用例全部通过 —— 可以进下一步了")
    else:
        print(f"进度：{passed}/{len(cases)} 通过（NotImplementedError 是正常的，去写实现吧）")
    # ★ 失败必须让**退出码非 0** ✗（此前被 try/except 吞掉 → echo $? 恒为 0 ✗）
    sys.exit(0 if passed == len(cases) else 1)

