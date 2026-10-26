"""PPO 的编排层：把 rollout / GAE / loss 串成一个训练循环。

到这里四个部件都齐了（`batch` 管契约、`gae` 管优势、`loss` 管目标、`toy` 管环境），
`trainer` 要做的事本质只有一件：**按正确的顺序调用它们**。

`ppo_update()` 是整个项目里唯一一个"算法本体"级别的循环 ——
它就是你之前 `main.py` 里那四行 `for _ in range(4):` 的规范化版本。
**这个函数留给你写**；编排部分（评估、训练循环、日志）我已经写好并验证过。

一次 `ppo_update` 的完整顺序
---------------------------
    1. next_values = next_values_from_values(batch.values, batch.mask)
    2. advantages, returns = compute_gae(...)
    3. advantages = normalize_advantages(advantages, batch.mask, cfg.adv_norm)   ← 跨 batch！
    4. token_mask = meta["prompt_mask"] | batch.mask        ← forward 需要的输入
    5. for epoch in range(cfg.inner_epochs):
         logits, values, _ = model(batch.input_ids, token_mask)    ← 要梯度！
         logprobs_new = token_logprobs(logits, batch.input_ids)
         第 0 轮先断言 ratio 起手为 1（assert_ratio_starts_at_one）
         out = ppo_loss(...)
         optimizer.zero_grad(); out.total.backward()
         grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)   ← 裁剪！
         optimizer.step()
    6. 返回诊断量字典（含 per_epoch 轨迹）

M3 的验收标准（我用一份临时参考实现实测的）
----------------------------------------
`preset("toy")` 是 `lr=1e-3, rollout_prompts=32, max_new_tokens=12, inner_epochs=4`，
在 CPU 上 **300 步约 7 秒**，评估准确率（128 道固定题、贪心）大致是：

    step   0 | 0.023      ← 随机初始化，几乎全错
    step  50 | ~0.95
    step 150 | ~1.00
    step 300 | ~1.00      ← 学会数 1 的个数了

**如果你的 `ppo_update` 写对了，`python3 run_toy.py` 应该跑到这个水平。**
跑不上去通常不是"训练不够久"，而是：符号反了 / mask 漏了 / detach 没做 /
ratio 起手不是 1（裁剪把梯度吃光了）。

六个必须想清楚的点
-----------------
1. **`next_values` 从哪来**：`next_values_from_values(batch.values, batch.mask)`
   （注意它是 `batch.py` 里的**模块级函数**，不是 `RolloutBatch` 的方法）
   用 mask 推（不要用切片）；因为 `dones[last]=True`，bootstrap 值其实用不到。
2. **归一化必须跨整个 batch**（`mode="batch"`），不是每条序列内部 ——
   后者会把"这条回答好不好"这个序列级信号整体减掉（见 `gae.py` 的 docstring）。
3. ⚠️ **这个框架有两个 mask，别混**（这是最贵的一个坑）：

   | mask | 含义 | 用在哪 |
   |---|---|---|
   | `batch.mask` | 生成出来的 token | `next_values_from_values` / `compute_gae` / `normalize_advantages` / `ppo_loss` / ratio 断言 —— **共 5 处** |
   | `token_mask` = `batch.meta["prompt_mask"] | batch.mask` | 真实 token（prompt + 生成），只管"隐状态要不要更新" | **只有 `model.forward` 这一处** |

   传错的后果：prompt 位置会被算进优势（静默污染归一化统计量），
   而 ratio 断言会立刻炸 —— 因为 prompt 位置的 `logprobs_old` 在 rollout 里从没被填过（恒为 0），
   却会被拿去和 `logprobs_new` 比。实测偏差 ~0.86，且**超差位置的 logprobs_old 全是 0**。
   建议在代码里用两个名字写清楚：`gen_mask = batch.mask` / `token_mask = ...`。
4. **重算 forward 必须带梯度**（和 rollout 相反！rollout 在 `no_grad` 下记录常数）。
5. **`zero_grad` 在内层每一轮都要调**，不是每批一次。漏了会让梯度跨轮累加，
   等效于把学习率放大成一个奇怪的东西。
6. **`clip_grad_norm_` 在 `backward()` 之后、`step()` 之前**。它的返回值是**裁剪前**
   的梯度范数 —— 正好是你要记进日志的诊断量。
"""

from __future__ import annotations

import warnings
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from batch import (RolloutBatch, last_valid_index, next_values_from_values,
                   token_logprobs)
from config import PPOConfig
from gae import compute_gae, normalize_advantages
from loss import assert_ratio_starts_at_one, ppo_loss
from metrics import MetricLogger
from toy import CountOnesTask, TinyActorCritic, rollout


# =========================================================================== #
# 1. 核心：一次 PPO 更新（★ 留给你实现）
# =========================================================================== #
def ppo_update(
    model: TinyActorCritic,
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


    values, next_values = batch.values, next_values_from_values(batch.values, batch.mask)
    token_mask = batch.meta["prompt_mask"] | batch.mask

    logprobs_old = batch.logprobs
    rewards = batch.rewards

    advantages, returns = compute_gae(rewards=rewards, values=values, next_values=next_values,
                                      dones=batch.dones, mask=batch.mask, gamma=cfg.gamma, lam=cfg.lam)
    advantages = normalize_advantages(advantages, batch.mask, cfg.adv_norm)  

    
    # if cfg.kl_coef > 0:
    #     logprobs_ref = ref_model.

    per_epoch = []

    for idx in range(cfg.inner_epochs):

        logits, values,  next_values = model.forward(batch.input_ids, token_mask)
        logprobs_new = token_logprobs(logits, batch.input_ids)
        if idx == 0 and assert_fresh:
            assert_ratio_starts_at_one(logprobs_new, logprobs_old, batch.mask)
            
        lp_all = logits.log_softmax(dim=-1)              # (B, T, V)
        entropy = -(lp_all.exp() * lp_all).sum(dim=-1)   # (B, T)
        out = ppo_loss(logprobs_new, logprobs_old, advantages, returns, values, batch.values, batch.mask, cfg, entropy)

        loss = out.total
        policy_loss = out.policy
        values_loss = out.value
        kl = out.kl
        entropy = out.entropy
        clip_frac = out.clip_frac
        mean_ratio = out.mean_ratio
        approx_kl = out.approx_kl
        ratio_max = out.ratio_max
        
        optimizer.zero_grad()
        out.total.backward()
        grad_norm = float(clip_grad_norm_(model.parameters(), cfg.grad_clip)) 
        optimizer.step()

        per_epoch.append({"clip_frac": clip_frac, "mean_ratio": mean_ratio, "approx_kl": approx_kl})

        if cfg.target_kl is not None and out.approx_kl > cfg.target_kl:
            break    

    return {"loss": loss, "policy_loss": policy_loss, "value_loss": values_loss,
       "kl": kl, "entropy": entropy,
       "clip_frac": clip_frac, "mean_ratio": mean_ratio, "approx_kl": approx_kl,
       "ratio_max": ratio_max, "grad_norm": grad_norm,
       "per_epoch": per_epoch}
        
   

    raise NotImplementedError("写 PPO 的内层更新循环")


# =========================================================================== #
# 2. 评估（已实现）
# =========================================================================== #
def make_eval_prompts(task: CountOnesTask, n: int = 128, seed: int = 12345
                      ) -> tuple[Tensor, Tensor, Tensor]:
    """造一组**固定**的评估 prompt。

    必须固定：每次评估用不同题目的话，曲线抖动大半来自题目难度，看不出学习。
    seed 故意和训练 seed 不同，避免评估集和训练集重叠。
    """
    g = torch.Generator().manual_seed(seed)
    return task.sample(n, generator=g)


@torch.no_grad()
def evaluate(
    task: CountOnesTask,
    model: TinyActorCritic,
    cfg: PPOConfig,
    prompts: tuple[Tensor, Tensor, Tensor],
) -> dict:
    """贪心评估：固定题集上的准确率 + 平均生成长度。

    贪心（temperature=0）是为了让指标**可比**：采样评估的方差比训练信号还大。
    """
    prompt_ids, prompt_mask, counts = prompts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # temperature=0 会触发"需要探索"的提醒
        greedy_cfg = cfg.replace(temperature=0.0, rollout_prompts=prompt_ids.shape[0])
    b = rollout(task, model, greedy_cfg, prompts=(prompt_ids, prompt_mask, counts))
    Lp = b.shape[1] - cfg.max_new_tokens
    parsed = task.parse_count(b.actions[:, Lp:], b.mask[:, Lp:])
    return {
        "eval_reward": float((parsed == counts).float().mean()),
        "eval_gen_len": float(b.gen_len().float().mean()),
    }


# =========================================================================== #
# 3. 同步训练循环（已实现）
# =========================================================================== #
class RolloutSource(Protocol):
    """**接口声明**（不是基类）—— 训练循环依赖的全部就是这四个方法。

    关于 `Protocol`，三个容易误会的地方：

    1. **它没有任何实现，也不需要实现。** `Protocol` 是"结构化类型"：
       一个类**不需要继承它**，只要方法齐了就自动满足它。
       （`SyncSource` / `AsyncSource` 都没有继承它 —— 看它们的 `__mro__` 就知道。）
    2. **它不能被实例化**（`TypeError: Protocols cannot be instantiated`）。
       方法体写成 `...` 不是"待实现"，而是"这里只有签名"。
    3. **它只在两个地方起作用**：给类型检查器（mypy/pyright）看，
       以及给读代码的人看 —— "训练循环到底需要 source 提供什么"。

    ⚠️ **别继承它**：`class X(RolloutSource)` 会让 X 拿到返回 `None` 的空方法，
    漏写一个方法就会静默传 `None` 下去。要满足它就**直接写同名方法**。

    运行时契约由 `train()` 开头那个 `_check_source()` 兜底 ——
    静态类型检查器抓不到"你少写了一个方法"，但那个检查能。
    """

    def start(self) -> None:
        """启动生产者（同步源是 no-op，异步源在这里起线程）。在训练循环开始前调用。"""
        ...

    def get(self) -> tuple[RolloutBatch, int]:
        """取一批来训练，返回 (batch, lag)。lag = 当前版本 - 该批次的生成版本。"""
        ...

    def advance(self) -> None:
        """训练完一批后调用：版本号 +1（告诉生产者"新权重可用了"）。"""
        ...

    def close(self) -> None:
        """收尾（停线程等）。`train()` 在 finally 里调用，异常时也会执行。"""
        ...


_SOURCE_METHODS = ("start", "get", "advance", "close")


def check_source(source) -> None:
    """运行时契约：少写一个方法就**立刻**报错，而不是训练到一半才 AttributeError。

    `Protocol` 只在静态检查时起作用；Python 运行时不会替你验。
    所以这里补一道，顺便把"需要哪些方法"写死在错误信息里。
    """
    missing = [m for m in _SOURCE_METHODS if not callable(getattr(source, m, None))]
    if missing:
        raise TypeError(
            f"{type(source).__name__} 缺少方法 {missing}\n"
            f"        RolloutSource 要求：{', '.join(_SOURCE_METHODS)}\n"
            f"        （Protocol 不会在运行时替你检查 —— 静态检查器才管这个）"
        )



class SyncSource:
    """同步数据源：每轮用**当前**权重现场采一批。lag 恒为 0。

    它和 `async_runner.AsyncSource` 实现同一套三个方法
    （`start` / `get` / `advance` / `close`），所以 `train()` 完全看不出区别 ——
    **"异步"在这个框架里就是换一个 source，算法代码一行都不用动。**
    """

    def __init__(self, task: CountOnesTask, model: TinyActorCritic, cfg: PPOConfig) -> None:
        self.task, self.model, self.cfg = task, model, cfg
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.version = 0

    def start(self) -> None:
        pass                                  # 同步没有生产者线程

    def get(self) -> tuple[RolloutBatch, int]:
        b = rollout(self.task, self.model, self.cfg, generator=self.gen)
        b.version = self.version
        return b, 0                           # 永远是最新版本

    def advance(self) -> None:
        self.version += 1

    def close(self) -> None:
        pass


def train(
    task: CountOnesTask,
    model: TinyActorCritic,
    cfg: PPOConfig,
    source: "RolloutSource",           # 类型注解：让静态检查器验证接口是否满足
    logger: MetricLogger | None = None,
    eval_prompts: tuple[Tensor, Tensor, Tensor] | None = None,
    eval_every: int | None = None,
) -> tuple[TinyActorCritic, MetricLogger]:
    """**同步和异步共用的训练循环** —— 唯一的区别就是传进来的 `source`。

    这个函数不知道数据是"现场采的"还是"从队列里取的旧的"，它只关心
    `(batch, lag)`。所以下面这段代码一行都不用为了异步而改 ——
    这就是"同步/异步只差数据来源"这句话的代码形态。

    新增记进日志的两项：`lag`（滞后几个版本）和 `version`（该批次的生成版本）。
    **`lag` 曲线是 M4 唯一的观察对象**：它一涨，`clip_frac` 就跟着涨。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    logger = logger or MetricLogger(out_dir=cfg.out_dir, tag=cfg.tag)
    eval_prompts = eval_prompts if eval_prompts is not None else make_eval_prompts(task)
    eval_every = cfg.log_every if eval_every is None else eval_every

    check_source(source)         # 运行时契约：缺方法立刻报错（静态检查器管不到这里）
    model.to(cfg.device)          # 和 batch.to(cfg.device) 配对，否则设备不匹配
    model.train()
    source.start()                # 同步源是 no-op；异步源在这里起生产者线程

    try:
        for it in range(cfg.total_iters):
            batch, lag = source.get()              # ← 同步/异步的全部差异都在这一行
            batch.to(cfg.device)
            batch.check()                          # 每批都验契约，几微秒
            stale = bool(batch.meta.get("stale", False))
            stats = ppo_update(model, optimizer, batch, cfg, assert_fresh=not stale)
            # per_epoch 是列表（每轮的轨迹），不能进标量日志 —— 先摘出来。
            per_epoch = stats.pop("per_epoch", None)
            source.advance()                       # 版本号 +1

            # 训练奖励：用 task 的**原始**奖励（准确率），而不是含逐步代价的 shaped reward，
            # 这样曲线可以直接读成"答对率"。
            Lp = batch.shape[1] - cfg.max_new_tokens
            parsed = task.parse_count(batch.actions[:, Lp:], batch.mask[:, Lp:])
            acc = float((parsed == batch.meta["counts"]).float().mean())
            logger.add(step=it, reward=acc, lag=float(lag), version=float(batch.version),
                       gen_len=float(batch.gen_len().float().mean()), **stats)

            if eval_every and it % eval_every == 0:
                logger.add(step=it, **evaluate(task, model, cfg, eval_prompts))
                logger.save_csv()
            if cfg.log_every and (it + 1) % cfg.log_every == 0:
                print(f"  step {it + 1:>4} | lag {logger.recent('lag', cfg.log_every):.1f}"
                      f" | 训练 {logger.recent('reward', cfg.log_every):.3f}"
                      f" | 评估 {logger.recent('eval_reward', 3):.3f}"
                      f" | clip_frac {logger.recent('clip_frac', cfg.log_every):.3f}"
                      f" | approx_kl {logger.recent('approx_kl', cfg.log_every):.4f}")
    finally:
        source.close()

    model.eval()
    return model, logger


def train_sync(
    task: CountOnesTask,
    model: TinyActorCritic,
    cfg: PPOConfig,
    logger: MetricLogger | None = None,
    eval_prompts: tuple[Tensor, Tensor, Tensor] | None = None,
    eval_every: int | None = None,
) -> tuple[TinyActorCritic, MetricLogger]:
    """同步 PPO：就是 `train()` 加一个 `SyncSource`。"""
    source = SyncSource(task, model, cfg)
    return train(task, model, cfg, source, logger, eval_prompts, eval_every)


# =========================================================================== #
# 4. 测试用例
# =========================================================================== #
def _fresh(task: CountOnesTask, cfg: PPOConfig) -> tuple[TinyActorCritic, RolloutBatch]:
    """固定的模型 + 一批固定的 rollout，供各个用例复用（保证可复现）。"""
    torch.manual_seed(cfg.seed)
    model = TinyActorCritic()
    b = rollout(task, model, cfg, generator=torch.Generator().manual_seed(cfg.seed))
    b.check()
    return model, b


def _cfg(**kw) -> PPOConfig:
    """干净的测试配置：B=8、G=12、小步数、可复现。"""
    base = dict(tag="trainer_test", device="cpu", rollout_prompts=8, max_new_tokens=12,
                inner_epochs=1, lr=1e-2, grad_clip=1.0, total_iters=2, log_every=1)
    base.update(kw)
    return PPOConfig(**base)


def _masked_logprob(model: TinyActorCritic, batch: RolloutBatch) -> float:
    """当前策略在被选中 token 上的平均 logprob（只在生成位置上算）。"""
    with torch.no_grad():
        token_mask = batch.meta["prompt_mask"] | batch.mask
        logits, _, _ = model(batch.input_ids, token_mask)
        lp = token_logprobs(logits, batch.input_ids)
        return float((lp * batch.mask).sum() / batch.mask.sum())


def _case_a() -> None:
    """★ 最关键的用例：一次梯度上升后，被选中 token 的 logprob **必须变大**。

    构造：把 critic 的估值全设为 0、终局奖励全设为 +1 → 优势全为正（用
    `adv_norm="none"` 保住符号）。此时 PPO 的目标是"提高这些动作的概率"，
    所以 logprob 必须上升。

    这个用例不过，说明符号 / mask / detach 有一处错了 —— 而 loss 照样会下降。
    """
    task = CountOnesTask()
    cfg = _cfg(adv_norm="none", inner_epochs=1, value_coef=0.0, entropy_coef=0.0)
    model, batch = _fresh(task, cfg)

    # critic 说"这些都是 0 价值"，而实际拿到 +1 → 优势恒为正
    batch.values = torch.zeros_like(batch.values)
    rows = torch.arange(batch.batch_size)
    last = last_valid_index(batch.mask)
    batch.rewards[rows, last] = 1.0                      # 覆盖掉 -step_cost，保证 A > 0

    lp_before = _masked_logprob(model, batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ppo_update(model, optimizer, batch, cfg)
    lp_after = _masked_logprob(model, batch)

    if not lp_after > lp_before + 1e-4:
        raise AssertionError(
            f"一次梯度上升后 logprob 没有变大：{lp_before:.4f} → {lp_after:.4f}\n"
            f"        优势全是正的，PPO 应该提高这些动作的概率。检查：\n"
            f"          * policy loss 的负号（要最大化替代目标 → loss 取负）\n"
            f"          * advantages 有没有乘对 mask\n"
            f"          * ratio 起手是不是 1（不是 1 的话裁剪会吃掉梯度）"
        )
    print(f"    logprob {lp_before:+.4f} → {lp_after:+.4f}（上升）  ✅")


def _case_b() -> None:
    """ratio 起手必须为 1 —— 而且这个断言要能**真的拦住**被污染的批次。

    两半都测：干净批次必须通过；把 `batch.logprobs` 人为挪开必须报错。
    只测前者的话，"断言写了个恒真的条件"这种错误发现不了。
    """
    task = CountOnesTask()
    cfg = _cfg()
    model, batch = _fresh(task, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ppo_update(model, optimizer, batch, cfg)             # 干净 → 不应报错
    print("    干净批次：ratio 起手为 1，断言不触发  ✅")

    model2, batch2 = _fresh(task, cfg)
    batch2.logprobs = batch2.logprobs + 0.5              # 人为制造"两条计算路径"
    try:
        ppo_update(model2, torch.optim.AdamW(model2.parameters(), lr=cfg.lr), batch2, cfg)
        raise AssertionError("batch.logprobs 被污染了，assert_ratio_starts_at_one 却没拦住 —— "
                             "说明那个断言写成了恒真条件")
    except AssertionError as e:
        if "ratio" not in str(e):
            raise
        print("    污染批次：断言拦住  ✅")


def _case_c() -> None:
    """梯度裁剪真的生效：把 grad_clip 调到极小，参数位移必须明显变小。"""
    task = CountOnesTask()
    cfg = _cfg(adv_norm="none", inner_epochs=1, lr=1e-2)

    def moved(grad_clip: float) -> float:
        model, batch = _fresh(task, cfg)
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        ppo_update(model, opt, batch, cfg.replace(grad_clip=grad_clip))
        return float(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before)))

    d_free = moved(1e9)          # 不裁
    d_clip = moved(1e-6)         # 狠裁
    if not d_clip < d_free * 0.5:
        raise AssertionError(
            f"grad_clip 看起来没生效：不裁时位移 {d_free:.3e}，狠裁时 {d_clip:.3e}（没变小）\n"
            f"        检查 `clip_grad_norm_(model.parameters(), cfg.grad_clip)` "
            f"是否在 backward() 之后、step() 之前"
        )
    print(f"    参数位移：不裁 {d_free:.3e} → 狠裁 {d_clip:.3e}（显著变小）  ✅")


def _case_d() -> None:
    """内层多轮的轨迹：这就是"inner_epochs 该设几"的依据。

    不强行断言趋势（和具体任务/学习率有关），但**必须**满足两条：
      * 每轮的 `approx_kl` 都 ≥ 0（KL 估计量的下限）
      * 最后一轮的策略确实动过（`approx_kl > 0`）
    然后把轨迹打印出来给你看趋势。
    """
    task = CountOnesTask()
    cfg = _cfg(inner_epochs=6, lr=5e-3, adv_norm="none")
    model, batch = _fresh(task, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    stats = ppo_update(model, optimizer, batch, cfg)

    per_epoch = stats.get("per_epoch")
    if not per_epoch or len(per_epoch) != cfg.inner_epochs:
        raise AssertionError(f"返回的 per_epoch 应含 {cfg.inner_epochs} 条，实际 "
                             f"{None if per_epoch is None else len(per_epoch)} 条")
    if any(e["approx_kl"] < -1e-9 for e in per_epoch):
        raise AssertionError("approx_kl 出现负值 —— 它是 (r-1) - log r，必须恒 ≥ 0")
    if not per_epoch[-1]["approx_kl"] > 0:
        raise AssertionError("最后一轮 approx_kl == 0：策略根本没动，检查梯度是不是全被裁掉了")

    print("    epoch |  clip_frac | mean_ratio | approx_kl")
    for i, e in enumerate(per_epoch, 1):
        print(f"      {i:>3} | {e['clip_frac']:>10.4f} | {e['mean_ratio']:>10.4f} | {e['approx_kl']:>9.5f}")
    print("    ↑ 曲线走平的位置就是 inner_epochs 的合理上界（后面的轮次在白算）")


def _case_e() -> None:
    """可复现性：同样的 seed 跑两次，结果必须完全一样。"""
    task = CountOnesTask()
    cfg = _cfg(total_iters=3, log_every=0)
    outs = []
    for _ in range(2):
        torch.manual_seed(cfg.seed)
        model = TinyActorCritic()
        _, logger = train_sync(task, model, cfg, eval_every=0)
        outs.append([round(r["reward"], 6) for r in logger.records])
    if outs[0] != outs[1]:
        raise AssertionError(f"两次运行结果不同：\n        {outs[0]}\n        {outs[1]}")
    print(f"    两次运行的奖励序列完全一致：{outs[0]}  ✅")


def _case_f() -> None:
    """配置项必须**真的生效**：`adv_norm` 和 `entropy_coef`。

    这两个是"静默失效"型 bug：不报错、训练照样跑，只是**你以为开了其实没开**。
    （我的前五个用例全用了 `adv_norm="none"`，等于完全没测归一化 —— 这是测试的洞。）

    测法：
      * 把奖励放大 100 倍 → `adv_norm="batch"` 的 grad_norm 必须远小于 `"none"`
        （两者一样大 = 归一化从来没被调用）
      * `entropy_coef > 0` 时返回的 `entropy` 必须 > 0
        （恒为 0 = `ppo_loss` 的 `entropy` 参数没被传进去）
    """
    task = CountOnesTask()

    # ---- ① adv_norm 真的生效吗 ----
    #
    # ⚠️ 这个测试有两个"必须做对"的隔离，否则它会去量错的东西（我第一版两处都错了）：
    #   (a) **把 value_coef / entropy_coef 关掉**。否则 grad_norm 由 value_loss 主导
    #       （returns 被放大 100 倍 → 价值误差巨大），归一化优势只改动策略那一小块，
    #       比值只有 1.5x，看起来像"没生效"。（实测：不关 1.56x，关掉后 4~5x）
    #   (b) **先给终局奖励填上真有真假的分布**，再放大。否则未训练的模型几乎全答错，
    #       reward×100 大部分是 0，优势尺度根本没被放大多少 —— 测试就没有对比度。
    def gnorm(adv_norm: str) -> float:
        cfg = _cfg(adv_norm=adv_norm, inner_epochs=1, lr=1e-2,
                   value_coef=0.0, entropy_coef=0.0)      # ← 隔离 (a)
        model, batch = _fresh(task, cfg)
        rows, last = torch.arange(batch.batch_size), last_valid_index(batch.mask)
        batch.rewards[rows, last] = torch.linspace(1.0, 0.0, batch.batch_size)  # ← 隔离 (b)
        batch.rewards = batch.rewards * 100.0             # 再放大 100 倍
        batch.values = torch.zeros_like(batch.values)     # critic 说"都是 0 价值"
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        return float(ppo_update(model, opt, batch, cfg)["grad_norm"])

    g_none, g_batch = gnorm("none"), gnorm("batch")
    if not g_batch < g_none * 0.2:
        raise AssertionError(
            f"adv_norm 看起来没生效：none 的 grad_norm={g_none:.3f}，"
            f"batch 的={g_batch:.3f}（归一化后应该小 5 倍以上）\n"
            f"        → `ppo_update` 里是不是漏了 "
            f"`advantages = normalize_advantages(advantages, batch.mask, cfg.adv_norm)`？"
        )
    print(f"    adv_norm：none 梯度 {g_none:.3f} → batch {g_batch:.3f}"
          f"（压低 {g_none / g_batch:.1f}x）  ✅")

    # ---- ② entropy_coef 真的生效吗 ----
    cfg = _cfg(inner_epochs=1, entropy_coef=0.05)
    model, batch = _fresh(task, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    stats = ppo_update(model, opt, batch, cfg)
    if not float(stats["entropy"]) > 1e-3:
        raise AssertionError(
            f"entropy_coef={cfg.entropy_coef}，但返回的 entropy={float(stats['entropy']):.6f}（恒为 0）\n"
            f"        → 说明 `ppo_loss` 的 `entropy=` 参数没被传。要从 logits 现算：\n"
            f"          `lp = logits.log_softmax(-1); entropy = -(lp.exp() * lp).sum(-1)`"
        )
    print(f"    entropy_coef：返回的 entropy = {float(stats['entropy']):.4f}（非零）  ✅")


if __name__ == "__main__":
    cases = [("用例 A：一次更新后 logprob 必须变大（★ 最关键）", _case_a),
             ("用例 B：ratio 起手为 1（正反两向都要测）", _case_b),
             ("用例 C：梯度裁剪真的生效", _case_c),
             ("用例 D：内层多轮的轨迹（inner_epochs 的依据）", _case_d),
             ("用例 E：可复现性", _case_e),
             ("用例 F：配置项必须真的生效（adv_norm / entropy）", _case_f)]
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
        except Exception as e:
            print(f"    💥 未预期的异常 {type(e).__name__}: {e}")
    print(f"\n{'=' * 60}")
    if passed == len(cases):
        print(f"🎉 {len(cases)} 个用例全部通过 —— 可以跑 M3 了：python3 run_toy.py")
    else:
        print(f"进度：{passed}/{len(cases)} 通过（NotImplementedError 是正常的）")
