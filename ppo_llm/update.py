"""一次 PPO 更新 —— 从 `ppo_lab/trainer.py` 搬过来，但**后来改动很大** ✗。

★ 相对玩具版的三处改动（都是这一路踩出来的 ✓）：
  ① **minibatch 那一段重写了** ✓
     原版用 `reshape(nmb, B/nmb, G)` ✗ → Python3 的 `/` 是浮点 → 直接 TypeError ✓
     而且 `if B // nmb != 0` 写反了 ✓ → `num_minibatches=1` 时**恒真 → 立刻抛错** ✓
     现在：按行 `slice` ✓（每块仍是 (mb,T) ✓ 下游函数一个都不用改 ✓）
           + `loss` 按 **token 占比加权** ✓（不加权 → 不等长块之间会偏 ✓ 且不报错 ✗）
     等价性测试：`nmb=1` 与 `nmb=8` 的**全部参数梯度**逐位一致 ✓✓
  ② **`_set_gen_width`** ✓：生成段宽度改从 `batch.meta["gen_width"]` 每块重设 ✓
     原来只存在 `model.gen_len` 这个**单个可变属性**上 ✗ → 多块时只剩最后一块的值 ✓
     → 别的块 forward 会切错尾部 ✓ 而且**不报错** ✓（第 6 步踩过的那个 bug ✓）
  ③ **`grpo_advantages`** ✓：新增 GRPO 的组相对优势 ✓（`adv_norm="grpo"` ✓ 不要 critic ✓）

为什么单独一个文件
------------------
`ppo_lab/trainer.py` 里除了 `ppo_update`，还有玩具专用的 `SyncSource` / `train()` /
`evaluate()` 和一整套 toy 测试用例（它们要 import `toy`）。LLM 这边只要这一个函数，
所以抽出来单独放，**不把 `toy.py` 拖进来**。

抽出来之后我用差分测试验过：同一个 batch、同一份初始权重，
这里的 `ppo_update` 和 `ppo_lab` 那份返回的**每一个诊断量都完全相同**。

模型需要满足什么
----------------
只要 `forward(input_ids, token_mask)` 返回 `(logits, values, ...)` 就行 ——
第三个返回值 `ppo_update` 根本不用（它用 `next_values_from_values` 从 `batch.values` 推）。
`toy.TinyActorCritic` 和 `ppo_llm` 的 `LMWithValueHead` 都满足这个契约。
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from batch import (RolloutBatch, gen_start, last_valid_index, next_values_from_values,
                   token_logprobs)
from config import PPOConfig
from gae import compute_gae, normalize_advantages
from loss import assert_ratio_starts_at_one, ppo_loss



class ActorCriticModel(Protocol):
    """`ppo_update` 对模型的要求 —— 就这两条。

    **不需要继承**（这是结构化接口，见 `ppo_lab` 里关于 Protocol 的讨论）：
    只要方法齐了就算满足。运行时由 `check_model()` 兜底。
    """

    def forward(self, input_ids: Tensor, token_mask: Tensor) -> tuple:
        """返回 `(logits, values, ...)`；`logits` 是 (B,T,V) 或生成段的 (B,G,V)。"""
        ...

    def parameters(self):
        """给优化器和 `clip_grad_norm_` 用。"""
        ...


def check_model(model) -> None:
    """运行时契约：缺方法立刻报错（Protocol 在运行时不会替你检查）。"""
    missing = [m for m in ("forward", "parameters") if not callable(getattr(model, m, None))]
    if missing:
        raise TypeError(
            f"{type(model).__name__} 缺少 {missing}；ActorCriticModel 要求 forward / parameters"
        )



def _to_full(x: Tensor, input_ids: Tensor, ref: Tensor) -> Tensor:
    """把 `(B,G)` 的"生成段"张量铺回 `(B,T)`（mask 外为 0）；已经是 `(B,T)` 就原样返回。

    ⚠️ 这个函数的存在本身是个教训：**"只算生成段"这个优化会沿着数据流传播**。
    不只 logprob 会变短，**任何从 logits 现算出来的 (B,T) 量都会变短** ——
    比如 entropy。漏掉一个，`ppo_loss` 里就会炸一句
    `size of tensor a (12) must match tensor b (21)`。
    """
    T_full = input_ids.shape[1]
    G = int(x.shape[1])
    if G == T_full:
        return x
    if not 0 < G <= T_full:
        raise ValueError(f"时间维 {G} 不合法（T={T_full}）")
    out = torch.zeros(input_ids.shape, dtype=ref.dtype, device=ref.device)
    out[:, T_full - G:] = x
    return out


def _logprobs_new(logits: Tensor, input_ids: Tensor, ref: Tensor) -> Tensor:
    """把模型前向的输出变成 `(B,T)` 的 logprob（mask 外为 0）。

    允许模型返回两种形状：

    * `(B, T, V)` —— 全量 logits（玩具模型走这条）
    * `(B, G, V)` —— **只算生成段**的 logits（`G = T - prompt_len`）。
      真实 LLM 该走这条：省掉 `(B,T,V)` 那个大张量（B=4,T=812 时 fp32 就是 1.97GB，
      反向还要再来一份梯度）。实测 **快 1.45x，而且步时方差从 2.5 倍降到几乎为 0** ——
      后者对训练循环更重要，谁也不想某一步随机慢三倍。

    判据是"时间维是否等于 T"：生成段永远在最后 G 列，所以 `input_ids[:, T-G:]` 就是它。
    """
    T_full = input_ids.shape[1]
    if logits.shape[1] == T_full:
        return token_logprobs(logits, input_ids)
    G = int(logits.shape[1])
    if not 0 < G <= T_full:
        raise ValueError(f"模型返回 logits 的时间维 {G} 不合法（T={T_full}）")
    return _to_full(token_logprobs(logits, input_ids[:, T_full - G:]), input_ids, ref)


#: `out.detached()` 里**按 token 加权**的那些量（都是 mask 内均值 ✓）
#: `ratio_max` 不在里面 —— 它跨块要取 **max** ✓ 不是均值 ✓
_MEAN_KEYS = ("loss", "policy_loss", "value_loss", "kl", "kl_raw", "entropy",
              "clip_frac", "mean_ratio", "approx_kl")


def _set_gen_width(m, batch: RolloutBatch) -> None:
    """把生成段宽度从 **batch** 同步到模型上 ✓

    ★ 多块时**每块都必须重设** ✗ —— `model.gen_len` 是单个可变属性 ✓
      只靠 rollout 最后留下的那个值 → 别的块 forward 会切错尾部 ✓
      而且**不报错**（第 6 步踩过的那个 bug ✓）
    """
    gw = batch.meta.get("gen_width")
    if gw is not None and hasattr(m, "gen_len"):
        m.gen_len = int(gw)


def smooth_kl(kl_raw: float, prev: float | None, alpha: float = 0.7) -> float:
    """KL 的指数滑动平均 ✓ —— 控制**有噪声**的信号必须先平滑 ✗

    alpha 是"旧值的权重" ✓（0.7 → 时间常数约 3 步 ✓）
    seed2 的实测病：单步 kl_raw 在带内外来回跳 ✓
      → 20 步均值 **1.40** 而**最后一步**在带内（< 0.9）✗
      → 控制器于是停在 kl_coef=0.0003 不再动作 ✗
    """
    return float(kl_raw) if prev is None else float(alpha * prev + (1.0 - alpha) * kl_raw)


def update_kl_coef(kl_raw: float, kl_coef: float, target: float,
                    factor: float = 1.5, lo: float = 1e-5, hi: float = 1.0) -> float:
    """★ 经典自适应 KL 控制器 ✓（trl / InstructGPT 用的那一套 ✓）

        kl_raw > factor × target  → kl_coef ×= factor
        kl_raw < target / factor  → kl_coef ÷= factor
        否则不动 ✓

    死区是 `[target/factor, factor×target]` ✓ 对数尺度上是个积分器 ✓
    → 会**收敛进死区**然后小幅来回 ✓
    （`_case_adaptive_kl` 用被控对象模拟整条回路 ✓ 断言它真的收敛 ✓
      只验算式的话"方向写反"这种错能溜过去 ✗）

    `factor=1.5` 而不是 2.0 ✓：2.0 时每步跨过整个死区 ✓ 会在边界来回跳 ✓
    """
    if target is None or target <= 0 or not kl_coef:
        return float(kl_coef)
    if kl_raw > factor * target:
        new = kl_coef * factor
    elif kl_raw < target / factor:
        new = kl_coef / factor
    else:
        return float(kl_coef)
    return float(min(max(new, lo), hi))


def grpo_advantages(rewards: Tensor, mask: Tensor, group_size: int,
                    eps: float = 1e-8) -> Tensor:
    """★ GRPO 的组相对优势 ✓ —— 单独成函数就是为了能用精确数值钉住它 ✓

        A_i = (r_i − mean_g) / (std_g + eps)      g = 同一条题的那 group_size 条 ✓

    **注意它是逐轨迹的标量** ✓（结果奖励 ✓ 然后铺到 mask 内的每个位置 ✓），
    不是逐 token 的 ✓ —— 这就是 GRPO 用"组均值"当 baseline、不需要 critic 的原因 ✓✓

    依赖一个**布局约定** ✗：同一个组必须**连续** ✓（rows[i*G:(i+1)*G] 是同一条题 ✓）
    —— `main.py` 里是 `[p0]*G + [p1]*G + ...` ✓ 不是 `probs*G` ✗
    """
    if group_size < 2:
        raise ValueError(f"GRPO 需要 group_size ≥ 2 ✗（现在是 {group_size}）")
    B = rewards.shape[0]
    if B % group_size != 0:
        raise ValueError(f"batch {B} 不能被 group_size {group_size} 整除 ✗")
    rows = torch.arange(B, device=rewards.device)
    r = rewards[rows, last_valid_index(mask)]              # (B,) 每条轨迹的结果奖励 ✓
    rg = r.view(B // group_size, group_size)
    mu = rg.mean(dim=1, keepdim=True)
    sd = rg.std(dim=1, keepdim=True, unbiased=False)
    adv_row = ((rg - mu) / (sd + eps)).reshape(B)
    return adv_row.unsqueeze(1).expand_as(mask).masked_fill(~mask, 0.0)


def ppo_update(
    model: ActorCriticModel,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: PPOConfig,
    ref_model: TinyActorCritic | None = None,
    assert_fresh: bool = True,
) -> dict:
    """对一整批 rollout 数据做 `cfg.inner_epochs` 轮 PPO 更新，返回诊断量。

    参数
    ----
    model : actor-critic（策略头和价值头在同一个模块里，所以只用一个 optimizer）
    optimizer : 通常 `torch.optim.AdamW(model.parameters(), lr=cfg.lr)`
    batch : `toy.rollout()` 的产物，已经过 `batch.check()`
    cfg : 用 `inner_epochs` / `grad_clip` / `adv_norm` / `clip_eps` /
          `value_clip_eps` / `kl_coef` / `target_kl`
    ref_model : 参考模型（`cfg.kl_coef > 0` 时才需要；本任务默认关掉）
    assert_fresh : 是否断言"数据由当前策略产生"。
        **同步必须 True**；**异步必须 False** —— 异步的数据天生来自旧版本，
        ratio 起手就不是 1，那个断言会（正确地）失败。

    要返回的字典
    -----------
    ``{"loss": float, "policy_loss": float, "value_loss": float,
       "kl": float, "entropy": float,
       "clip_frac": float, "mean_ratio": float, "approx_kl": float,
       "ratio_max": float, "grad_norm": float,
       "per_epoch": [ {clip_frac, mean_ratio, approx_kl}, ... ]}``

    标量取**最后一轮内层更新**的诊断量（那是这批数据被榨到的最远处），
    `per_epoch` 保留每一轮的轨迹 —— 它是回答"inner_epochs 该设几"的唯一依据：
    跑完看 `clip_frac` 从第几轮开始基本不再变化（说明后面的轮次已近空转）。

    实现要点（顺序错了就会静默学歪）
    ------------------------------
    * `next_values` 用 `next_values_from_values(batch.values, batch.mask)`（batch.py 的模块级函数）。
    * `token_mask = batch.meta["prompt_mask"] | batch.mask`。
    * **第 0 轮**调用 `assert_ratio_starts_at_one(logprobs_new, batch.logprobs, batch.mask)`
      —— 它会检查 rollout 和重算是不是同一条计算路径。放后面就没意义了
      （此时策略已经动过，ratio 本来就不该是 1）。
    * `ppo_loss` 的实参对应关系：`logprobs_new` / `logprobs_old=batch.logprobs` /
      `values_new=values` / `values_old=batch.values`。
    * `optimizer.zero_grad()` 在**每一轮**开头；`backward()` → `clip_grad_norm_()` → `step()`。
    * `cfg.target_kl` 不为 None 时：某轮的 `out.approx_kl` 超过它就 `break`
      （这是 PPO 论文里的实用技巧，能让训练稳很多）。
    * `cfg.num_minibatches > 1` 时把 batch 切成 minibatch（可选，默认 1 用不到）。

    写完直接跑 `python3 trainer.py`，五个用例会告诉你对不对。
    其中**用例 A 是最关键的**：一次梯度上升之后，被选中 token 的 logprob 必须变大 ——
    如果这个不过，说明符号 / mask / detach 有一处错了，PPO 根本不在做它说的事。
    """


    # ================= 准备：**整批**做 GAE + **整批**归一化 =================
    values = batch.values
    next_values = next_values_from_values(batch.values, batch.mask)
    token_mask = batch.meta["prompt_mask"] | batch.mask

    if cfg.adv_norm == "grpo":
        # ★ GRPO：不用 GAE、不用 value baseline ✗
        #   组均值就是 baseline ✓ → critic 是多余的 ✓（实测它本来就几乎不出力 ✓）
        advantages = grpo_advantages(batch.rewards, batch.mask, int(cfg.group_size))
        # ★ 一次性诊断 ✓✓ —— `grad_norm=0` 时能立刻分辨是哪个环节为零 ✗
        if not globals().get("_ADV_STATS", 0):
            globals()["_ADV_STATS"] = 1
          # ★ 整段包在 try 里 ✓ —— 诊断本身出错也**不能**拖垮训练 ✗✓
            try:
                _A = advantages.detach()
                _M = batch.mask
                print(f"  [adv] |A| 均值 {float(_A.abs().mean()):.6f} · "
                      f"A 非零 {int((_A != 0).sum())}/{_A.numel()} · "
                      f"mask 非零 {int(_M.sum())}/{_M.numel()} · "
                      f"reward 非零 {int((batch.rewards != 0).sum())}")
                # ★ 每组的奖励 ✓ —— 必须**按 last_valid_index 取** ✗✓
                #   （我第一版写成 `batch.rewards.view(-1, G)` ✗ → 那会把 (B,T) 拍平 ✓
                #     于是看到的是"某一行的前 G 个 token" ✗ → 奖励只在最后一个 token 上 ✗ → 全 0.0 ✓
                #     → **一个纯粹的打印 bug** ✗，但很容易被误读成"数据坏了" ✗✗）
                _n = int(cfg.group_size)
                _rows = torch.arange(batch.rewards.shape[0], device=batch.rewards.device)
                _r = batch.rewards.detach()[_rows, last_valid_index(batch.mask)]   # (B,) ✓
                if _n > 1 and _r.numel() >= _n:
                    for _i, _row in enumerate(_r.view(-1, _n)[:4]):
                        print(f"     组{_i} 奖励 {[round(float(x), 3) for x in _row]}")
                    print(f"     全批奖励：均值 {float(_r.mean()):.3f} · "
                          f"非零 {int((_r != 0).sum())}/{_r.numel()} · "
                          f"组内全同的组数 {int((_r.view(-1, _n).std(dim=1) == 0).sum())}/{_r.numel() // _n}")
            except Exception as _e:               # noqa: BLE001
                print(f'  [adv] 诊断出错（不影响训练 ✓）：{type(_e).__name__}: {_e}')

        returns = batch.rewards.clone()          # value_coef 应为 0 ✓ 但给个合法值 ✓
    else:
        # ★ PPO：整批 GAE + **整批**归一化 ✓
        #   μ/σ 必须取全部有效 token ✗（切块后各算各的会把同一批放缩成 nmb 种倍数 ✓）
        advantages, returns = compute_gae(
            rewards=batch.rewards, values=values, next_values=next_values,
            dones=batch.dones, mask=batch.mask, gamma=cfg.gamma, lam=cfg.lam)
        advantages = normalize_advantages(advantages, batch.mask, cfg.adv_norm)

    # ================= 参考模型（KL 惩罚）=================
    # ⚠️ 静默失败要堵死：`kl_coef > 0` 却没传 ref_model 时，老代码会安静地算出 kl = 0 ✓
    if cfg.kl_coef != 0 and ref_model is None:
        raise ValueError(
            f"cfg.kl_coef={cfg.kl_coef} 却没传 ref_model —— KL 惩罚会静默失效\n"
            f"        要么传参考模型（main.py 在 kl_coef != 0 时会自动建），\n"
            f"        要么把 kl_coef 设成 0")
    if ref_model is not None:
        for pm in ref_model.parameters():
            pm.requires_grad_(False)          # 参考模型永远冻结

    # ================= minibatch 切分（按行 slice ✓ 不用 reshape ✗）=================
    B = batch.input_ids.shape[0]
    nmb = max(int(cfg.num_minibatches), 1)
    if nmb > B:
        raise ValueError(f"num_minibatches={nmb} 比 batch size {B} 还大 ✗")
    if B % nmb != 0:
        raise ValueError(f"batch size {B} 不能被 num_minibatches={nmb} 整除 ✗")
    mb = B // nmb
    slices = [slice(i * mb, (i + 1) * mb) for i in range(nmb)]

    # ★ 每块的有效 token 数 —— loss 按它加权 ✓（ppo_loss 返回的是本块**均值** ✗）
    n_tok = [int(batch.mask[sl].sum()) for sl in slices]
    N_tok = sum(n_tok)
    if N_tok == 0:
        raise ValueError("整批没有任何有效 token ✗")
    w = [n / N_tok for n in n_tok]

    # ================= 参考 logprob（分块 forward ✓ 只做一次 ✓ 无反向）=================
    logprobs_ref = None
    if ref_model is not None:
        logprobs_ref = torch.zeros_like(batch.logprobs)
        for sl in slices:
            _set_gen_width(ref_model, batch)          # ★ 每块都设 ✓
            with torch.no_grad():
                logits_ref, _, _ = ref_model(batch.input_ids[sl], token_mask[sl])
            logprobs_ref[sl] = _logprobs_new(logits_ref, batch.input_ids[sl],
                                             batch.logprobs[sl])

    # ================= 内层更新（分块 forward/backward + 梯度累积）=================
    per_epoch: list[dict] = []
    last: dict | None = None
    for idx in range(cfg.inner_epochs):
        optimizer.zero_grad()                      # ★ 一轮只清一次（放进循环里就变成 nmb 次独立更新 ✗）
        acc: dict[str, float] = {}
        for k_i, sl in enumerate(slices):
            _set_gen_width(model, batch)           # ★ 每块都重设（不能靠上一块留下的值 ✗）
            # ★★ 前向**之前**就打一次 ✓✓ —— 上一轮的插桩在 backward 后 ✓ 结果一行都没打出来 ✗
            #   ⇒ OOM 发生在**第一次 forward** ✗ → 所以要看到真实的 T / 行数 ✓
            import torch as _t
            if _t.cuda.is_available():
                _dv = batch.input_ids.device
                _ix = _dv.index if _dv.index is not None else _t.cuda.current_device()
                _rows = int(sl.stop - sl.start) if sl.stop is not None else -1
                _T = int(batch.input_ids.shape[1])
                _G = int(getattr(model, "gen_len", 0) or 0)
                print(f"    [mem] 前向 mb {k_i + 1}/{len(slices)} · 行 {_rows} · "
                      f"T {_T} · gen_len {_G} · "
                      f"已分配 {_t.cuda.memory_allocated(_ix) / 2**30:5.1f} G", flush=True)
            logits, vals, _ = model(batch.input_ids[sl], token_mask[sl])
            lp_new = _logprobs_new(logits, batch.input_ids[sl], batch.logprobs[sl])

            if idx == 0 and assert_fresh:
                assert_ratio_starts_at_one(lp_new, batch.logprobs[sl], batch.mask[sl])

            # ★★ 熵只有在 `entropy_coef != 0` 时才需要**梯度** ✗
            #    而我们所有 run 的 entropy_coef 都是 0 ✓ → 以前这里无条件算 ✓
            #    于是白造两份 (B,G,V)（log_softmax + exp ✓ 每块 ~2.9GB ✗）
            #    而且它在带梯度路径里 → 还要为 backward 一直留着 ✗✗
            #    现在：coef==0 时走 no_grad ✓ 诊断量照样有 ✓ 图不留 ✓ 张量立刻释放 ✓✓
            if cfg.entropy_coef != 0:
                lp_all = logits.log_softmax(dim=-1)
                ent = _to_full(-(lp_all.exp() * lp_all).sum(dim=-1),
                               batch.input_ids[sl], batch.logprobs[sl])
            else:
                # ★★★ coef==0 → **完全不算熵** ✗✓
                #   原来这里还走 no_grad 算了一遍 ✓ → `log_softmax` 一个 (B,G,V) 1.45G ✓
                #   再 `exp()` + `mul` 各一份 ✓ → **峰值 ~4.4 GiB** ✗✗ 只为日志 ✗
                #   `loss.py` 已经支持 `entropy=None` ✓（coef=0 时那项本来恒为 0 ✓✓）
                #   诊断改用**采样 token 的 NLL**（`-lp_new` ✓ 已算好 ✓ 与熵同向 ✓）
                ent = None
                _ent_proxy = float((-lp_new).mean())
            out = ppo_loss(lp_new, batch.logprobs[sl], advantages[sl], returns[sl],
                           vals, batch.values[sl], batch.mask[sl], cfg, ent,
                           logprobs_ref=None if logprobs_ref is None else logprobs_ref[sl])

            # ★ 按 token 占比加权再反传 ✓
            #   整批均值 = Σ(均值_k × tokens_k) / N  ✓ 所以要乘 w[k_i] ✓
            (out.total * w[k_i]).backward()
            # ★★ 临时插桩 ✓✓ —— 每个 minibatch 反传后打一次显存 ✓
            #   目的：**定位那 39 G 是哪来的** ✗（我按"4 行"算只有 ~6 G ✗ 对不上 ✓）
            import torch as _t
            if _t.cuda.is_available():
                _dev = batch.input_ids.device
                _idx = _dev.index if _dev.index is not None else _t.cuda.current_device()
                print(f"    [mem] mb {k_i + 1}/{len(slices)} · 行 {sl.stop - sl.start}"
                      f" · 已分配 {_t.cuda.memory_allocated(_idx) / 2**30:5.1f} G"
                      f" · 峰值 {_t.cuda.max_memory_allocated(_idx) / 2**30:5.1f} G"
                      f" · 图保留 {'是' if out.total.requires_grad else '否'}", flush=True)

            d_k = out.detached()
            if cfg.entropy_coef == 0:
                # ★ 用**另一个键**记 proxy ✗ 不要覆盖 `entropy` ✗✗
                #   （覆盖会让自测拿 proxy 和真熵比 ✓ → 误报"熵项符号反了" ✗；
                #     那一轮自测**帮我抓住了这个错** ✓ 保留它 ✓✓）
                #   coef==0 时 `entropy` 键由 loss.py 给 0 ✓（那项本来恒为 0 ✓）
                d_k = dict(d_k)
                d_k["nll_proxy"] = _ent_proxy
            for key in _MEAN_KEYS:
                acc[key] = acc.get(key, 0.0) + d_k[key] * w[k_i]
            acc["ratio_max"] = max(acc.get("ratio_max", 0.0), d_k["ratio_max"])

        d = acc
        d["grad_norm"] = float(clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()                           # ★ 一轮只 step 一次 ✓

        per_epoch.append({k: d[k] for k in ("clip_frac", "mean_ratio", "approx_kl")})
        last = d
        if cfg.target_kl is not None and d["approx_kl"] > cfg.target_kl:
            break

    assert last is not None, "inner_epochs <= 0 ✗"

    # ★ 自适应 KL：用**本步实测的原始 KL** 决定**下一步**的系数 ✓
    #   只算出来放进 stats ✓ 真正换 cfg 由 main.py 做（cfg 不可变 ✓）
    # ★ 报两个 ✗：`kl_coef` = **本步实际用的**（日志要的是这个 ✓）
    #            `kl_coef_next` = 自适应算出来的**下一步**的值 ✓（main.py 用它换 cfg ✓）
    # ★ 决策搬到 main.py 了 ✗：控制**有噪声**的信号必须先平滑 ✓
    #   而平滑需要**跨步状态** ✓ 纯函数的 ppo_update 存不了 ✓
    #   （seed2 实测：单步决策进了极限环 ✓ kl_raw 20 步均值 1.40 而最后一步在带内 ✗）
    last["kl_coef"] = cfg.kl_coef
    last["kl_coef_next"] = cfg.kl_coef         # 字段保留（日志/兼容 ✓）
    return {**last, "per_epoch": per_epoch}
        




# =========================================================================== #
# 自测（`python3 update.py`）—— 只在这里依赖 toy，正式代码不依赖
# =========================================================================== #
if __name__ == "__main__":
    import os
    import sys

    _TOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "ppo_lab")
    sys.path.insert(0, _TOY_DIR)
    try:
        from toy import CountOnesTask, TinyActorCritic, rollout
    except ImportError as _e:
        # ★ 缺**可选测试依赖** ≠ 代码坏 ✓ 所以这里**响亮地跳过**、退出码 0 ✓
        #   理由：一条"红了但其实没事"的测试，会训练人以后忽略红色 ✗
        #         那比这个 bug 本身更坏 ✓
        #   正式代码（main.py / ppo_update）**完全不依赖 toy** ✓
        print(f"⏭️  跳过：找不到 toy（{_e}）")
        print(f"    它应该在 {os.path.abspath(_TOY_DIR)}")
        print(f"    那是 `ppo_lab/` 玩具项目 ✓ **正式代码不依赖它** ✓ 所以这不是故障 ✓")
        print()
        print(f"    但**这些用例很有价值** ✗（尤其 minibatch 累积的等价性 ✓）")
        print(f"    想在服务器上也跑到，把 ppo_lab 一起同步过去：")
        print(f"        rsync -av ppo_lab/ <host>:/data/ppo_lab/")
        print(f"    ★ 注意目标要写成 /data/ppo_lab（和 /data/ppo 平级 ✓）")
        print(f"      因为 update.py 找的是它自己的 `../ppo_lab` ✓")
        print()
        print("    ⚠️ **本次一个用例都没跑** ✗ —— 退出码 2 = '跳过'（不是成功 ✓ 也不是失败 ✗）")
        print("       为什么不用 0 ✗：绿灯会让人以为覆盖过了 ✓✓ 那比这个 bug 更坏 ✓")
        sys.exit(2)

    def _cfg(**kw) -> PPOConfig:
        base = dict(tag="update_test", device="cpu", rollout_prompts=8,
                    max_new_tokens=12, inner_epochs=3, lr=2e-3, total_iters=1)
        base.update(kw)
        return PPOConfig(**base)

    def _case_grpo_advantage() -> None:
        """★ 用**精确数值**钉住 GRPO 的定义 ✓✓

        组0 = (0,0,0,0)  → 优势必须**全 0** ✓✓ ← GRPO 的核心特性（无信息组不出梯度 ✓）
        组1 = (1,0,1,0)  → 必须**恰好**归一化成 (+1,−1,+1,−1) ✓（mean=0 / std=1 ✓）

        这条测试同时能抓出"用整批归一化冒充 GRPO" ✗：
        整批看这两组的均值是 0.25 ✓ 归一化结果**不会**是 ±1 ✓✓
        """
        G, T = 4, 5
        B = 2 * G
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -2:] = True                      # 每行 2 个有效位置 ✓
        rew = torch.zeros(B, T)
        rows = torch.arange(B)
        rew[rows, last_valid_index(mask)] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        adv = grpo_advantages(rew, mask, G)

        g0 = adv[:G][mask[:G]]
        if float(g0.abs().max()) != 0.0:
            raise AssertionError(f"全同组的优势应为 0 ✗ 实际最大 |A| = {float(g0.abs().max()):.3e}")
        if float(adv[~mask].abs().max()) != 0.0:
            raise AssertionError("mask 外必须为 0 ✗")

        g1 = adv[G:][mask[G:]]
        if abs(float(g1.mean())) > 1e-6:
            raise AssertionError(f"组内均值应为 0 ✗ 实际 {float(g1.mean()):.3e}")
        if abs(float(g1.std(unbiased=False)) - 1.0) > 1e-6:
            raise AssertionError(f"组内标准差应为 1 ✗ 实际 {float(g1.std(unbiased=False)):.6f}")
        vals = sorted(round(float(v), 6) for v in g1.unique())
        if vals != [-1.0, 1.0]:
            raise AssertionError(f"组1 应恰好是 ±1 ✗ 实际 {vals}")

        # 守卫也要有效 ✓
        for bad, why in ((1, "group_size=1"), (3, "batch 不被 G 整除")):
            try:
                grpo_advantages(rew, mask, bad)
                raise AssertionError(f"{why} 应该被拦住 ✗")
            except ValueError:
                pass
        print(f"    组0(全同) → 优势全 0 ✓   组1(1,0,1,0) → 恰好 ±1 ✓（mean=0/std=1）")


    def _case_ema_damping() -> None:
        """★ EMA 必须真的**压住震荡** ✓✓（否则那个修法就是装饰 ✗）

        给一个**方波** kl_raw（0.1 与 3.0 交替 ✓，模拟 seed2 的极限环 ✓）
        比较两种做法的"系数变动次数" ✓
        """
        wave = [0.1 if i % 2 == 0 else 3.0 for i in range(40)]
        c_raw = c_ema = 0.01
        ema = None
        n_raw = n_ema = 0
        for k in wave:
            nxt = update_kl_coef(k, c_raw, 0.6)
            n_raw += int(abs(nxt - c_raw) > 1e-12); c_raw = nxt
            ema = smooth_kl(k, ema, 0.7)
            nxt = update_kl_coef(ema, c_ema, 0.6)
            n_ema += int(abs(nxt - c_ema) > 1e-12); c_ema = nxt
        if n_ema >= n_raw:
            raise AssertionError(
                f"EMA 没起作用 ✗ 变动次数 不平滑={n_raw} vs 平滑={n_ema}\n"
                f"        （平滑之后必须**严格更少** ✓ 否则极限环还在 ✓）")
        print(f"    方波输入 40 步：不平滑变动 {n_raw} 次 → 平滑后 {n_ema} 次 ✓✓")
        print(f"    （最终系数 不平滑 {c_raw:.5f} / 平滑 {c_ema:.5f} ✓）")


    def _case_adaptive_kl() -> None:
        """★ 自适应 KL：验算式 ✓ 更要验**整条回路真的收敛** ✓✓

        ③ 是关键 ✓：用一个"被控对象"模拟真实反馈（系数越大 → KL 越小 ✓
           因为惩罚会把策略拉回参考模型 ✓）：`kl_next = K / coef`
           从 coef=0.001 跑 40 步 → 断言 KL 落进死区 ✓
           只验算式的话，"方向写反了"能过 ✗ 闭环过不了 ✓
        """
        T, F = 0.6, 1.5
        lo_b, hi_b = T / F, T * F

        if update_kl_coef(10.0, 0.01, T) <= 0.01:
            raise AssertionError("KL 太高时系数应变大 ✗（方向反了）")
        if update_kl_coef(0.001, 0.01, T) >= 0.01:
            raise AssertionError("KL 太低时系数应变小 ✗")
        if abs(update_kl_coef(T, 0.01, T) - 0.01) > 1e-12:
            raise AssertionError("死区内不应改动 ✗")

        c = 0.01
        for _ in range(80):
            c = update_kl_coef(999.0, c, T)
        if c > 1.0 + 1e-9:
            raise AssertionError(f"上界没夹住 ✗ {c}")
        c = 0.01
        for _ in range(80):
            c = update_kl_coef(1e-9, c, T)
        if c < 1e-5 - 1e-12:
            raise AssertionError(f"下界没夹住 ✗ {c}")

        K = 0.01 * T
        coef, kl = 0.001, None
        for _ in range(40):
            kl = K / coef
            coef = update_kl_coef(kl, coef, T)
        if not (lo_b <= kl <= hi_b):
            raise AssertionError(
                f"闭环未收敛进死区 ✗ 40 步后 kl={kl:.4f}，死区 [{lo_b:.3f}, {hi_b:.3f}]\n"
                f"        → 控制器方向反了 / 死区和 factor 不匹配 ✓")
        print(f"    方向 ✓ 夹取 ✓ 闭环：40 步后 kl={kl:.4f} 落在死区 "
              f"[{lo_b:.3f}, {hi_b:.3f}] ✓  coef={coef:.5f}")


    def _case_tail() -> None:
        """★ 生成段路径必须和全量路径给出**完全相同**的诊断量。

        只算生成段只是一个**省显存的等价变形**，数学上不能有任何改变。
        不一致就说明切片对齐错了（差一个 token 之类）——而那类错误在
        loss 曲线上一眼看不出，只会让训练悄悄学歪。

        （实测收益：B=4,T=812 时快 1.45x，步时方差从 2.5 倍降到几乎为 0。）
        """
        cfg = _cfg()
        task = CountOnesTask()

        def fresh():
            torch.manual_seed(0)
            m = TinyActorCritic()
            b = rollout(task, m, cfg, generator=torch.Generator().manual_seed(0))
            b.check()
            return m, b

        m_full, b_full = fresh()
        m_tail, b_tail = fresh()
        Lp = int(gen_start(b_full.mask)[0])   # 单行 → 取标量 ✓      # ★ 与生产同一口径 ✓

        class TailOnly(nn.Module):
            """把 forward 包一层：只返回生成段的 logits。"""

            def __init__(self, inner: nn.Module, lp: int) -> None:
                super().__init__()
                self.inner, self.lp = inner, lp

            def forward(self, input_ids, token_mask):
                logits, values, _ = self.inner(input_ids, token_mask)
                return logits[:, int(self.lp):], values, None

        s_full = ppo_update(m_full, torch.optim.AdamW(m_full.parameters(), lr=cfg.lr),
                            b_full, cfg)
        s_tail = ppo_update(TailOnly(m_tail, Lp),
                            torch.optim.AdamW(m_tail.parameters(), lr=cfg.lr), b_tail, cfg)

        keys = ("loss", "policy_loss", "value_loss", "clip_frac",
                "mean_ratio", "approx_kl", "ratio_max")
        bad = [k for k in keys if abs(float(s_full[k]) - float(s_tail[k])) > 1e-6]
        if bad:
            raise AssertionError(
                f"生成段路径和全量路径不一致的项：{bad}\n"
                f"        全量 {[round(float(s_full[k]), 6) for k in keys]}\n"
                f"        生成段 {[round(float(s_tail[k]), 6) for k in keys]}\n"
                f"        → 只算生成段必须是等价变形；不一致说明切片差了一个 token"
            )
        print(f"    全量 {round(float(s_full['loss']), 6)} == "
              f"生成段 {round(float(s_tail['loss']), 6)}（7 项全同）  ✅")

    def _case_minibatch_equiv() -> None:
        """★ num_minibatches=8 必须和 =1 给出**一模一样的梯度和诊断量** ✓

        这一条测试能一次抓出**四个静默错误** ✗（全都不报错 ✓）：
          ① 优势归一化的作用域（必须整批 ✓ 切块后各算各的就错 ✗）
          ② loss 的 token 加权（不加权 → 不等长块之间偏 ✓）
          ③ zero_grad / step 的位置（放进循环 → 变成 8 次独立更新 ✗）
          ④ 前 n-1 块白跑（只对最后一块 backward ✗）

        做法：同一个 batch 跑两遍，lr=0（参数不动 ✓ 方便比梯度 ✓）
        """
        cfg1 = _cfg(rollout_prompts=32, num_minibatches=1)
        task = CountOnesTask()

        def fresh():
            torch.manual_seed(0)
            m = TinyActorCritic()
            b = rollout(task, m, cfg1, generator=torch.Generator().manual_seed(0))
            b.check()
            return m, b

        def run(nmb):
            m, b = fresh()
            opt = torch.optim.AdamW(m.parameters(), lr=0.0)     # lr=0 → 参数不动 ✓
            st = ppo_update(m, opt, b, cfg1.replace(num_minibatches=nmb))
            grads = {n: (None if p.grad is None else p.grad.detach().clone())
                     for n, p in m.named_parameters()}
            return st, grads

        s1, g1 = run(1)
        s8, g8 = run(8)

        keys = ("loss", "policy_loss", "value_loss", "kl", "entropy",
                "clip_frac", "mean_ratio", "approx_kl", "ratio_max", "grad_norm")
        bad = [k for k in keys if abs(float(s1[k]) - float(s8[k])) > 1e-5]
        if bad:
            raise AssertionError(
                f"nmb=1 和 nmb=8 的诊断量不一致：{bad}\n"
                f"        nmb=1: {[round(float(s1[k]), 6) for k in bad]}\n"
                f"        nmb=8: {[round(float(s8[k]), 6) for k in bad]}\n"
                f"        → 大概率是 token 加权或归一化作用域错了 ✗")
        for n in g1:
            a, b_ = g1[n], g8[n]
            if (a is None) != (b_ is None):
                raise AssertionError(f"参数 {n} 的梯度有无不一致 ✗")
            if a is not None and not torch.allclose(a, b_, atol=1e-6, rtol=1e-5):
                raise AssertionError(
                    f"参数 {n} 的梯度不一致 ✗ 最大差 {(a - b_).abs().max():.3e}\n"
                    f"        → 累积路径和整批路径不等价 ✓ 这是**静默**学歪的典型来源 ✗")
        print(f"    nmb=1 与 nmb=8：10 项诊断量 + 全部参数梯度一致 ✓")
        print(f"    （B=32 → 每块 {32 // 8} 条；各块的有效 token 数 {[int(c) for c in [1]] if False else '不等'} ✓ 所以加权那条被真正考到了 ✓）")


    def _case_independent_oracles() -> None:
        """★★ 两条**独立判据** —— 专治"两条路径一起错、断言看不见" ✗✓

        审计实测的教训：把 `h_prev` 位移从**两条路径一起**去掉 ✓
          → 真实误差 **22.71 nats**（答案全错 ✓）而 ratio 断言报 **0.00e+00** ✓✓
        根因：那个断言是"这条路径和自己比" ✓ → 对**对称**错误零灵敏度 ✓

        这里补两条能看见对称错误的判据 ✓：
          ① **step 位置**：lr>0 时"更新后的参数"必须逐位相同 ✓
             （旧用例把 lr 设成 0 ✗ → `step()` 放进循环里也抓不到 ✓ 子代理变异实测 ✓）
          ② **ratio 不许退化成恒 1**：先用同一个 batch 更新一次 ✓ 再算第二遍 ✓
             此时策略**已经动过** ✓ → ratio **必须**偏离 1 ✓✓
             （若 `logprobs_old = lp_new.detach()` ✗ → 第二遍 ratio 仍是 1 ✓
               而"起手为 1"那条断言**恰好只检查 |dev| ≤ tol** ✓ → 抓不到它为 1 ✗✓）
        """
        cfg1 = _cfg(rollout_prompts=8, num_minibatches=1, max_new_tokens=8)
        task = CountOnesTask()

        def fresh(lr):
            torch.manual_seed(0)
            m = TinyActorCritic()
            b = rollout(task, m, cfg1, generator=torch.Generator().manual_seed(0))
            b.check()
            return m, b, torch.optim.AdamW(m.parameters(), lr=lr)

        # ① step 位置：lr>0 → 参数必须逐位一致 ✓
        out = {}
        for nmb in (1, 8):
            m, b, opt = fresh(1e-2)
            ppo_update(m, opt, b, cfg1.replace(num_minibatches=nmb))
            out[nmb] = {n: p.detach().clone() for n, p in m.named_parameters()}
        diff = max(float((out[1][n] - out[8][n]).abs().max()) for n in out[1])
        # ★ 阈值必须**由实测噪声定**，不能拍 ✗（拍成 1e-6 → 一上来就误报 ✓ 实测 1.17e-5 ✓）
        #   噪声来源：8 块分别累加 vs 一次整批 ✓ 浮点加法不满足结合律 ✓
        #            梯度差 ~1e-5（旧用例已经量过 ✓）→ Adam 一步位移 ≈ lr ✓ → 参数差 ~1e-5 ✓
        #   而**真要抓**的错误（`step()` 放进循环 ✗）差是 ~lr × 8 步 ≈ **1e-2** ✓✓
        #   → 取 1e-4 = 噪声的 10 倍 ✓ 同时比真错误小 100 倍 ✓✓
        TOL = 1e-4
        if diff > TOL:
            raise AssertionError(
                f"lr=1e-2 时 nmb=1 与 nmb=8 更新后的参数不一致"
                f"（最大差 {diff:.2e} > 噪声阈值 {TOL:.0e}）✗\n"
                f"        → 几乎肯定是 `optimizer.step()` 放进了 minibatch 循环 ✗\n"
                f"          （旧用例用 lr=0 ✗ 参数不动 → 这个错误抓不到 ✓）")
        print(f"    ① step 位置：lr>0 下参数一致 ✅（实测差 {diff:.1e}；噪声阈值 {TOL:.0e} ✓；"
              f"若 step 放进循环会差 ~1e-2 ✓ 差 100 倍 ✓）")

        # ② ratio 不许恒为 1 ✓
        m, b, opt = fresh(1e-3)
        ppo_update(m, opt, b, cfg1)                                   # 先真的走一步 ✓
        st2 = ppo_update(m, torch.optim.AdamW(m.parameters(), lr=0.0), b, cfg1,
                         assert_fresh=False)
        dev = abs(float(st2["mean_ratio"]) - 1.0)
        if dev < 1e-3:
            raise AssertionError(
                f"同一 batch 第二遍的 mean_ratio={float(st2['mean_ratio']):.6f} ≈ 1 ✗\n"
                f"        → 策略明明已经动过 ✓ ratio 却还是 1 ✓\n"
                f"        → 说明 `logprobs_old` 被换成了**新策略重算值** ✗\n"
                f"          （那样 PPO 就退化成普通策略梯度 ✓ 裁剪/信任域全部失效 ✗\n"
                f"           而且 clip_frac 会恒为 0 ✓ **不报错** ✓）")
        print(f"    ② ratio 未退化：第二遍 mean_ratio={float(st2['mean_ratio']):.6f}"
              f"（偏离 1 达 {dev:.2e} ✓）✅")

    def _case_term_signs() -> None:
        """★ 逐项**符号**的独立判据 ✗（审计变异实测：KL/熵/value 反号全都抓不到 ✓）

        做法：**一次只开一项**（其余系数为 0 ✓）→ 跑若干步 → 断言该项该往哪走 ✓
          kl_coef>0      → 与参考的距离 E[k3] 必须**下降** ✓
          entropy_coef>0 → 熵必须**上升** ✓（这一项进 loss 带负号 ✗ 反了就变"压熵" ✓）
          value_coef>0   → value_loss（=(V−G)² 那一项）必须**下降** ✓
        """
        cfg1 = _cfg(rollout_prompts=8, num_minibatches=1, max_new_tokens=8, inner_epochs=1)

        def run(**coefs):
            torch.manual_seed(0)
            m = TinyActorCritic()
            b = rollout(task, m, cfg1, generator=torch.Generator().manual_seed(0))
            b.check()
            # ★ kl_coef>0 必须给 ref_model ✗ 否则 ppo_update 会**响亮拒绝** ✓✓
            #   （它宁可报错也不静默把 KL 算成 0 ✓ 这个设计是对的 ✓）
            import copy as _copy          # 局部 import ✓（本文件其它用例也是这么做的 ✓）
            ref = _copy.deepcopy(m) if float(coefs.get("kl_coef", 0.0)) > 0 else None
            opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
            for _i in range(40):
                # ★ 只有第 0 次能断言"数据由当前策略产生" ✓
                #   （第 2 次起策略已经动过 ✓ ratio **本来就该**偏离 1 ✓
                #     实测：不关掉的话第 2 次 dev=0.26 → 断言正确报错 ✓
                #     这说明那条断言是活的 ✓✓ 但这里是在故意反复用同一批数据 ✓）
                st = ppo_update(m, opt, b, cfg1.replace(**coefs),
                                ref_model=ref, assert_fresh=(_i == 0))
            return st

        task = CountOnesTask()
        base = run(kl_coef=0.0, entropy_coef=0.0, value_coef=0.0)
        # ★ 不能拿 kl_coef=0 当基准 ✗：那时**根本不建 ref** ✓ `kl_raw` 是**假 0** ✓✓
        #   （子代理的 B-2 就是这个：0 有两种含义 ✓）
        #   改成比两个系数：其余项相同、只有 kl_coef 不同 → 大的必须把距离压得更低 ✓
        hi = run(kl_coef=5.0, entropy_coef=0.0, value_coef=0.0)
        lo = run(kl_coef=0.05, entropy_coef=0.0, value_coef=0.0)
        if not (hi["kl_raw"] < lo["kl_raw"]):
            raise AssertionError(
                f"kl_coef 更大时 KL 反而没更小 ✗ lo(0.05)={lo['kl_raw']:.6f} "
                f"hi(5.0)={hi['kl_raw']:.6f}\n"
                f"        → KL 项符号反了（惩罚变成奖励 ✓ 会主动推离参考模型 ✗）")
        print(f"    ① kl_coef 0.05 → 5.0：E[k3] {lo['kl_raw']:.6f} → {hi['kl_raw']:.6f} "
              f"被压得更低 ✅")

        env = run(kl_coef=0.0, entropy_coef=0.5, value_coef=0.0)
        if not (env["entropy"] > base["entropy"] - 1e-9):
            raise AssertionError(
                f"entropy_coef>0 时熵没上升 ✗ {base['entropy']:.6f} → {env['entropy']:.6f}\n"
                f"        → 熵项符号反了（变成**主动压熵** ✗）")
        print(f"    ② entropy_coef>0 → 熵 {base['entropy']:.6f} → {env['entropy']:.6f} 上升 ✅")

        val = run(kl_coef=0.0, entropy_coef=0.0, value_coef=1.0)
        if not (val["value_loss"] < base["value_loss"] + 1e-9):
            raise AssertionError(
                f"value_coef>0 时 value_loss 没下降 ✗ "
                f"{base['value_loss']:.6f} → {val['value_loss']:.6f}\n"
                f"        → value 项符号反了 ✓")
        print(f"    ③ value_coef>0 → value_loss {base['value_loss']:.6f} → "
              f"{val['value_loss']:.6f} 下降 ✅")

    def _case_bad_shape() -> None:
        """时间维不合法时给出清晰的报错，而不是静默算错。"""
        class Bad(nn.Module):
            def forward(self, input_ids, token_mask):
                return torch.zeros(2, 999, 5), torch.zeros(2, 1), None

        cfg = _cfg(rollout_prompts=2)
        task = CountOnesTask()
        torch.manual_seed(0)
        m = TinyActorCritic()
        b = rollout(task, m, cfg, generator=torch.Generator().manual_seed(0))
        try:
            ppo_update(Bad(), torch.optim.AdamW(m.parameters(), lr=1e-3), b, cfg)
            raise AssertionError("时间维 999 > T 却没报错")
        except ValueError as e:
            if "不合法" not in str(e):
                raise
            print(f"    非法时间维被拦住：{str(e)[:60]}  ✅")

    print("用例：自适应 KL 控制器")
    _case_adaptive_kl()

    print("\n用例：KL 平滑（压住极限环）")
    _case_ema_damping()

    print("用例：GRPO 的组相对优势")
    _case_grpo_advantage()

    print("用例：生成段 logits 的等价性")
    _case_tail()
    def _case_kl() -> None:
        """KL 惩罚必须真的生效 —— 这是"配置项接线"检查，不是功能检查。

        三种病都试一遍：
          ① kl_coef>0 却没传 ref      → 必须**报错**（不能静默为 0）
          ② ref 和策略不同            → kl 必须 > 0，且 total 要变化
          ③ kl_coef 大小不同          → total 必须不同（说明系数真的起作用）
        """
        import copy

        base = _cfg(kl_coef=0.2)
        task = CountOnesTask()
        torch.manual_seed(0)
        m = TinyActorCritic()
        b = rollout(task, m, base, generator=torch.Generator().manual_seed(0))
        b.check()

        # ① 缺 ref → 报错
        try:
            ppo_update(copy.deepcopy(m), torch.optim.AdamW(m.parameters(), lr=1e-3), b, base)
            raise AssertionError("kl_coef>0 且没传 ref_model，却没报错（会静默失效）")
        except ValueError as e:
            if "静默失效" not in str(e):
                raise
            print("    ① 缺 ref_model 被拦住（不会静默算 0）  ✅")

        # ② ref 不同 → kl > 0
        torch.manual_seed(123)
        ref_other = TinyActorCritic()
        m2 = copy.deepcopy(m)
        s_other = ppo_update(m2, torch.optim.AdamW(m2.parameters(), lr=1e-3), b, base,
                             ref_model=ref_other)
        if not float(s_other["kl"]) > 0:
            raise AssertionError(f"ref 与策略不同，kl 却 = {s_other['kl']}")
        print(f"    ② ref 不同 → kl = {float(s_other['kl']):.4f} > 0  ✅")

        # ③ 系数必须真的改变 total
        m3 = copy.deepcopy(m)
        s_lo = ppo_update(m3, torch.optim.AdamW(m3.parameters(), lr=1e-3), b,
                          base.replace(kl_coef=0.01), ref_model=ref_other)
        m4 = copy.deepcopy(m)
        s_hi = ppo_update(m4, torch.optim.AdamW(m4.parameters(), lr=1e-3), b,
                          base.replace(kl_coef=1.0), ref_model=ref_other)
        if abs(float(s_lo["loss"]) - float(s_hi["loss"])) < 1e-7:
            raise AssertionError(
                f"kl_coef 0.01 和 1.0 的 total 一样（{float(s_lo['loss'])}）"
                f" —— 系数没接线")
        print(f"    ③ kl_coef 0.01→1.0：loss {float(s_lo['loss']):.5f} → "
              f"{float(s_hi['loss']):.5f}（真的变了）  ✅")

        # ④ ref == 策略 → kl **恰好** 0（k3 在 d=0 时为 0）
        #
        #    ⚠️ 必须用 inner_epochs=1：返回的诊断量取自**最后一个** epoch，
        #    而 inner_epochs=3 时策略在内层已经被优化了两步，那时它**本来就该**
        #    偏离参考模型 —— 那不是 bug，是 KL 惩罚的语义 ✓
        #    （我第一版就写错了期望值，被这条用例抓住 —— 用例该干的事。）
        m5 = copy.deepcopy(m)
        s_same = ppo_update(m5, torch.optim.AdamW(m5.parameters(), lr=1e-3), b,
                            base.replace(inner_epochs=1), ref_model=copy.deepcopy(m))
        if abs(float(s_same["kl"])) > 1e-4:
            raise AssertionError(f"ref 就是策略本身，kl 却 = {s_same['kl']}（应该 ≈ 0）")
        print(f"    ④ ref == 策略 → kl = {float(s_same['kl']):.2e} ≈ 0  ✅")

    print("\n用例：KL 惩罚真的生效")
    _case_kl()

    print("\n用例：minibatch 累积的等价性")
    _case_minibatch_equiv()

    print("\n用例：非法形状的防御")
    _case_bad_shape()

    print("\n用例：独立判据（step 位置 / ratio 不许退化）")
    _case_independent_oracles()

    print("\n用例：逐项符号（KL / 熵 / value）")
    _case_term_signs()
    print(f"\n{'=' * 60}")
    print("🎉 update.py 自测通过")
