"""rollout 数据结构、mask 感知的 reduction、以及 padding 的正确姿势。

这个文件是整套代码的地基。它错了，后面全错，而且**大部分错误不会报错**
—— 张量形状是对的，只是数值悄悄错了。所以这里每一条不变量都写成了代码。

核心约定（背下来，后面所有文件都依赖它）
--------------------------------------
1. **所有张量形状都是 (B, T)**：B = batch 里几条序列，T = 补齐后的最大长度。
2. **`mask[b,t]=True` 表示"第 t 个位置是模型生成出来的 token，要训练"。**
   prompt 位置和 padding 位置都是 False。
3. **`values[b,t]` 是"生成第 t 个 token 之前"的状态价值 V(s_t)**，
   和 `logprobs[b,t]`（生成第 t 个 token 的 logprob）**时间对齐**。
   对齐错了不会报错，只会让优势有偏 —— 这是最难查的一类 bug。
4. **`mask=True` 的位置在每一行内必须是连续的**（生成段连着 prompt）。
   `check()` 会检查这一条，`next_values_from_values()` 依赖它。
5. **全框架统一用 left padding 拼 prompt**。这不是审美选择，而是上面第 4 条的要求：
   prompt 左 padding 后，真实 token 右对齐，生成 token 紧接其后，于是整行的
   `prompt真实段 + 生成段` 恰好是一段**连续的** True —— 第 4 条自动成立。
   如果用 right padding，`[真,真,PAD,生成,生成]` 就会出现空洞，第 4 条直接破。
   （附带好处：HF `generate` 本来就要求 left padding。）

   同理，`input_ids` 的长度 T 是**完整序列**（prompt + 生成），而 `logprobs` /
   `values` / `rewards` / `dones` / `mask` 也都是 (B, T)，只是在 prompt 位置上
   被 mask 掉。**保持"所有字段同一个形状"比"省几个字节"重要得多** ——
   形状不统一是静默 bug 的温床。

你上一版的那个 bug（值得反复看）
----------------------------
tokenizer 默认 **right padding**，但提取 logprob/value 用的是"从序列尾部往前数"
（`logits[i, -r_len-suffix_len-1 : -suffix_len-1]`）。结果 batch 里除了最长的
那条，**其余样本取到的全是 PAD 位置** —— 形状完全正确，数值是垃圾，不报错。

本文件的应对是：**彻底不用尾部负索引，一律用 mask 定位。** 这样 padding 在哪边
都不会错。`python3 batch.py` 会把这件事演示给你看。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sys

import torch
from torch import Tensor


# =========================================================================== #
# 1. mask 感知的 reduction
# =========================================================================== #
def masked_sum(x: Tensor, mask: Tensor) -> Tensor:
    """只在 mask=True 的位置求和。"""
    return (x * mask.to(x.dtype)).sum()


def masked_mean(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """只在 mask=True 的位置求均值。

    分母是**有效元素的个数**，不是 `numel()`。这一行就是"padding 不参与训练"的
    全部实现 —— 忘了它，padding 位置的损失就会按它的长度权重污染梯度。

    恒等检查：mask 全 True 时，`masked_mean(x, mask) == x.mean()`。
    """
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(eps)


def masked_std(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """mask 感知的标准差。优势归一化要用。"""
    m = mask.to(x.dtype)
    mean = masked_mean(x, m, eps)
    var = masked_mean((x - mean) ** 2, m, eps)
    return var.clamp_min(0.0).sqrt()


# =========================================================================== #
# 2. padding 工具
# =========================================================================== #
def pad_left(seqs: Sequence[Sequence[int]], pad_id: int = 0) -> tuple[Tensor, Tensor]:
    """左 padding。返回 (ids, mask)，mask=True 表示真实 token。

    真实 token 贴在**右端**，于是"序列最后一个位置"就是最后一个真实 token ——
    这正是 HF `generate` 需要的布局（新 token 从同一个位置往后接）。
    """
    max_len = max(len(s) for s in seqs) if seqs else 0
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, max_len - len(s):] = torch.as_tensor(list(s), dtype=torch.long)
            mask[i, max_len - len(s):] = True
    return ids, mask


def pad_right(seqs: Sequence[Sequence[int]], pad_id: int = 0) -> tuple[Tensor, Tensor]:
    """右 padding。真实 token 贴在**左端**。

    本框架不用它训练（生成任务里 tail 对齐更自然），但**保留它是为了演示
    "右 padding + 尾部负索引"这个组合会怎么悄悄错**。见文件底部的演示。
    """
    max_len = max(len(s) for s in seqs) if seqs else 0
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, :len(s)] = torch.as_tensor(list(s), dtype=torch.long)
            mask[i, :len(s)] = True
    return ids, mask


def last_valid_index(mask: Tensor) -> Tensor:
    """每一行最后一个 mask=True 的位置下标，形状 (B,)。

    用 `mask * (arange+1)` 取最大值再减一，避免对 bool 做 argmax 时"全 False 行"
    返回 0 的坑。`check()` 已经保证每行至少有一个 True。
    """
    positions = torch.arange(1, mask.shape[1] + 1, device=mask.device)
    return (mask.long() * positions).max(dim=1).values - 1


def gen_start(mask: Tensor) -> Tensor:
    """每行**生成段的起点**（用于解码 ✓）

    ★ 为什么必须有这个函数 ✗：以前 6 处评估器都写
        `Lp = b.input_ids.shape[1] - cfg.max_new_tokens`     ← 假设生成恰好写满 cap ✗
      而真实生成长度 ≈100 而 cap=192 ✓ → prompt 又只有 ~77 ✓
      → 这个表达式会变成**负数** ✗ → Python 负索引 → 只解码出**最后十几个 token** ✗
      → "评估器看到的文本" ≠ "奖励计算所用的文本" ✓✓
      实测后果：`sft.py` 同一份权重报出 **0.555（b.rewards ✓）与 0.523（错切片 ✗）**
      两个"严格准确率" ✓ 差异 ≈3%（写完 `#### N` 后又接着说几句的回答被切掉了 ✓✓）

    用 mask 就与长度、padding 方向、cap 全都无关 ✓✓
    """
    assert mask.dtype == torch.bool, "mask 必须是 bool"
    # ★ 全 False 的行必须**报错** ✗ 而不是静默返回 0 ✓
    #   （argmax 在空行上返回 0 ✓ 于是解码会从第 0 个 token 开始 → 把整个 prompt 当成生成 ✓
    #     而这个错误**不会报错** ✓ 只会让评估数字悄悄偏掉 ✓✓）
    empty = ~mask.any(dim=1)
    if bool(empty.any()):
        raise AssertionError(
            f"gen_start：第 {empty.nonzero().flatten().tolist()} 行 mask 全 False ✗"
            f"（没有生成 token → 不能解码 ✓）")
    return mask.float().argmax(dim=1)      # 左/右 padding 下都是"第一个生成位置" ✓


def next_values_from_values(
    values: Tensor, mask: Tensor, bootstrap: float | Tensor = 0.0
) -> Tensor:
    """从 `values` 推出 `next_values`，**用 mask 而不是切片**。

    对每一行：`next_values[t] = values[t+1]`，最后一个有效位置用 `bootstrap`
    （"回合终止后的价值"，通常取 0；也可以用 critic 对终止状态的估计）。

    为什么不用 `values[:, 1:]` 再 pad：那样等于假设 mask 的形状是规则的，
    一旦 padding 方向变了或者序列长度不一，就会静默错位。这里显式用 mask。
    """
    assert mask.shape == values.shape, "values 与 mask 形状必须一致"
    nxt = torch.zeros_like(values)
    nxt[:, :-1] = values[:, 1:]
    rows = torch.arange(values.shape[0], device=values.device)
    last = last_valid_index(mask)
    if isinstance(bootstrap, Tensor):
        nxt[rows, last] = bootstrap.to(values.dtype)
    else:
        nxt[rows, last] = bootstrap
    # padding 位置的 next_value 归零，避免污染（GAE 里还会乘 mask，双保险）
    nxt = nxt * mask.to(values.dtype)
    return nxt


# =========================================================================== #
# 3. 内存高效的单 token logprob
# =========================================================================== #
def token_logprobs(logits: Tensor, target_ids: Tensor) -> Tensor:
    """给定 (B, T, V) 的 logits 和 (B, T) 的目标 token，返回 (B, T) 的 logprob。

    **不要这么写**：`torch.log_softmax(logits, dim=-1).gather(...)`
    那会 materialize 一个完整的 (B, T, V) 中间张量。V=152k、B=4、T=600 时
    约 1.5GB —— 这就是你在 MPS 上跑不长序列的原因。

    正确做法：只对**被选中的那一个 token** 做归一化。
        logprob = logits[target] - logsumexp(logits, dim=-1)
    两者数学上完全等价，显存省掉约 V 倍。
    """
    selected = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    # ★★★ `torch.logsumexp(logits, -1)` 会在内部**再生成一份 (B,T,V)** ✗✗
    #   实测（3B · B=16 · T≈870 · V=152k）：它要一次分配 **7.91 GiB** ✗ → OOM ✓
    #   → 按 **T 分块** 算 ✓✓ 峰值降到 ~1/8 ✓（结果逐位相同 ✓：logsumexp 是逐行独立的 ✓）
    # ★ 分块要**与形状无关** ✗ —— 本函数既被 (B, T, V) 调用 ✓（rollout 重算 ✓）
    #   也被 (T, V) 调用 ✓（DPO 里 seq_logp ✓）→ 原来写死 `[:, i:i+step]` + `cat(dim=1)` ✗
    #   → 一维时 `cat(dim=1)` 直接 IndexError ✗（实测 ✓）
    #   现在：在**倒数第二维**（T）上切 ✓，结果拼在**最后一维** ✓✓
    T = logits.shape[-2]
    step = 64 if T > 64 else T
    parts = []
    for i in range(0, T, step):
        parts.append(torch.logsumexp(logits[..., i:i + step, :], dim=-1))
    if len(parts) == 1:
        return selected - parts[0]
    return selected - torch.cat(parts, dim=-1)


# =========================================================================== #
# 4. rollout 批次
# =========================================================================== #
def concat_batches(batches: list["RolloutBatch"]) -> "RolloutBatch":
    """把多次 rollout 的结果拼成一个 batch（右侧补齐到共同的 T ✓）。

    ★ 为什么能直接补：
      `mask` / `advantages` / GAE 全都**只在 mask 内**起作用 ✓
      补齐位置 mask=False ✓ → GAE 走到那里就停 ✓ → 天然被排除 ✓✓
      （同一批内 gen_len 不等本来就支持 ✓ 这里只是把 T 也不等也吃进来 ✓）

    ★ 为什么要拼成一个 batch、而不是传 list：
      优势归一化是 `(A − μ) / σ`，μ/σ 必须取**全部 32 条**的所有 mask 内 token ✓
      拼成一个 batch 之后，现有的 `normalize_advantages(..., "batch")` 自动就是全局的 ✓✓
      传 list 反而要自己重写归一化 ✗ —— 那正是最容易静默出错的地方 ✗
    """
    if not batches:
        raise ValueError("concat_batches 收到空列表 ✗")
    if len(batches) == 1:
        return batches[0]

    T = max(b.input_ids.shape[1] for b in batches)
    dev = batches[0].input_ids.device
    pad_id = int(batches[0].input_ids[0, 0])          # 只用它占位，mask 会挡住 ✓

    def cat_rows(vals, fill):
        """按行拼。**只有 2 维的才需要补到共同的 T** ✓ —— 而且补在**左边** ✓✓

        ★★ 必须左补，这是端到端冒烟才抓到的 bug ✗
           真实 LLM 的生成长度是**左 padding**（真实 token 在右边 ✓）
           第一版右补 → pad 落到**中间** ✗ → `forward` 里取"最后 gen_len 个位置"
           时取到的是 pad ✗ → ratio 起始偏差 195.3 ✗✗ 整段错位 ✓
           toy 的 batch 是**右对齐**的 ✓ → 单元测试里右补依然连续 ✓ 所以它没暴露 ✗
           → 这也是"单元测试过了 ≠ 端到端对了"的实例 ✓

        1 维的（比如 toy 的 `meta["counts"]` —— 每行一个计数 ✓）直接拼 ✓
        第一版把 `pad` 用在所有 meta 上 ✗ → 1 维解包 `b, t_ = t.shape` 直接炸 ✓
        """
        outs = []
        for v in vals:
            if v.dim() == 2 and v.shape[1] != T:
                o = torch.full((v.shape[0], T), fill, dtype=v.dtype, device=v.device)
                o[:, T - v.shape[1]:] = v          # ★ 左补 ✓ 与模型的左 padding 一致 ✓
                outs.append(o)
            else:
                outs.append(v)
        return torch.cat(outs, dim=0)

    for b in batches:
        if b.input_ids.shape[0] == 0:
            raise ValueError("有空 batch ✗")
        for name in ("input_ids", "actions", "logprobs", "values", "rewards", "dones", "mask"):
            if getattr(b, name).dim() != 2:
                raise ValueError(f"{name} 必须是 (B,T) 二维 ✗ 实际 {tuple(getattr(b, name).shape)}")

    meta = {}
    for k in batches[0].meta:
        vals = [b.meta.get(k) for b in batches]
        if all(isinstance(v, Tensor) for v in vals):
            fill = False if vals[0].dtype == torch.bool else 0
            meta[k] = cat_rows(vals, fill)
        elif k == "gen_width":
            # ★ 取 **max** —— 不能要求各块相等 ✗（我第一版就是那么写的 ✓ 结果真实 run 炸了 ✗）
            #
            #   `gen_width` 是那一块里**实际最长的生成** ✗ 不是 `max_new_tokens` ✓
            #   实测：同一个 max_new_tokens=192 下，8 块是 169 / 172 / 180 / 192 ✓ 全不同 ✓
            #   （有的块没人写满 192 → 那块就短 ✓）
            #
            #   取 max 是**精确等价**的 ✓✓：
            #     forward 在位置 t 的 logits 只依赖 t 之前的前缀 ✓ 与 G 无关 ✓
            #     mask 只标真实生成 ✓ → 多算出来的那些位置被排除 ✓ 不进 loss ✓
            #     —— 这正是 `update.py::_case_tail` 验证过的
            #        「只算生成段是等价变形」✓ 我却在 concat 这里画蛇添足加了个限制 ✗
            meta[k] = max(int(v) for v in vals)
        else:
            meta[k] = vals[0]

    return RolloutBatch(
        input_ids=cat_rows([b.input_ids for b in batches], pad_id),
        actions=cat_rows([b.actions for b in batches], 0),
        logprobs=cat_rows([b.logprobs for b in batches], 0.0),
        values=cat_rows([b.values for b in batches], 0.0),
        rewards=cat_rows([b.rewards for b in batches], 0.0),
        dones=cat_rows([b.dones for b in batches], False),
        mask=cat_rows([b.mask for b in batches], False),
        meta=meta,
    )
@dataclass


class RolloutBatch:
    """一次 rollout 的全部内容。所有张量形状都是 (B, T)。

    字段
    ----
    input_ids : (B,T) long   完整序列（prompt + 生成），trainer 靠它重算 new logprob
    actions   : (B,T) long   生成位置上的 token id（mask=False 的位置填 0 即可）
    logprobs  : (B,T) float  **行为策略**在这些 token 上的 logprob（rollout 时存下来的）
    values    : (B,T) float  V(s_t)，与 logprobs 时间对齐
    rewards   : (B,T) float  逐步奖励；终局稀疏奖励就只在最后一个有效位置非零
    dones     : (B,T) bool   True = 这一步之后回合终止（**终止步不能 bootstrap**）
    mask      : (B,T) bool   要训练的生成 token
    version   : int          产生这批数据的策略版本号 —— 异步的 staleness 靠它算
    advantages/returns : compute_gae 之后填，之前为 None
    """

    input_ids: Tensor
    actions: Tensor
    logprobs: Tensor
    values: Tensor
    rewards: Tensor
    dones: Tensor
    mask: Tensor
    version: int = 0
    advantages: Tensor | None = None
    returns: Tensor | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        """形状一致性检查。**让它早失败**，别让错形状的张量静默传播到 loss 里。"""
        shapes = {
            "input_ids": tuple(self.input_ids.shape),
            "actions": tuple(self.actions.shape),
            "logprobs": tuple(self.logprobs.shape),
            "values": tuple(self.values.shape),
            "rewards": tuple(self.rewards.shape),
            "dones": tuple(self.dones.shape),
            "mask": tuple(self.mask.shape),
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"RolloutBatch 各字段形状必须一致，实际为 {shapes}")
        if self.input_ids.dim() != 2:
            raise ValueError(f"所有字段必须是 (B,T) 二维，实际 {self.input_ids.dim()} 维")
        self.dones = self.dones.bool()
        self.mask = self.mask.bool()

    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, int]:
        return int(self.input_ids.shape[0]), int(self.input_ids.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_tokens(self) -> int:
        """有效（要训练的）token 总数。日志里所有平均值的分母都该是它。"""
        return int(self.mask.sum())

    def gen_len(self) -> Tensor:
        """每条序列生成了多少 token，形状 (B,)。用于统计长度分布／截断率。"""
        return self.mask.sum(dim=1)

    def to(self, device: str | torch.device) -> "RolloutBatch":
        """把所有张量搬到 device —— **包括 meta 里的**。

        meta 里放着 `prompt_mask` / `counts`，它们要和 mask 在同一个设备上，
        否则 `check()` 里那句 `mask & prompt_mask` 会直接抛设备不匹配。
        （我第一版漏了 meta，于是"batch 搬到 MPS、meta 留在 CPU"就炸了。）
        """
        for name in ("input_ids", "actions", "logprobs", "values", "rewards",
                     "dones", "mask", "advantages", "returns"):
            v = getattr(self, name)
            if isinstance(v, Tensor):
                setattr(self, name, v.to(device))
        for k, v in self.meta.items():
            if isinstance(v, Tensor):
                self.meta[k] = v.to(device)
        return self

    def index(self, idx: Tensor) -> "RolloutBatch":
        """切 minibatch：按样本维度取子集。"""
        def take(v):
            return v[idx] if isinstance(v, Tensor) else v

        return RolloutBatch(
            input_ids=take(self.input_ids), actions=take(self.actions),
            logprobs=take(self.logprobs), values=take(self.values),
            rewards=take(self.rewards), dones=take(self.dones), mask=take(self.mask),
            version=self.version, advantages=take(self.advantages),
            returns=take(self.returns), meta=self.meta,
        )

    def with_advantages(self, advantages: Tensor, returns: Tensor) -> "RolloutBatch":
        """GAE 算完后把优势和回报挂回批次（原地，返回 self 方便链式调用）。"""
        self.advantages = advantages
        self.returns = returns
        return self

    # ------------------------------------------------------------------ #
    def check(self, *, verbose: bool = False) -> None:
        """内部一致性检查。**训练时建议每步都调**（几微秒），它抓的都是静默 bug。

        检查项：
        1. 每行至少有一个要训练的 token（否则这条样本白采了）
        2. 生成位置在每行内连续（`next_values_from_values` 依赖它）
        3. `dones` 只在要训练的位置上为 True
        4. `rewards` 在不要训练的位置上必须为 0
        5. `actions` 在生成位置上必须等于 `input_ids`
        6. 每行的 `dones` 恰好有 1 个 True（回合并须终止一次）
        7. `logprobs` / `values` 等浮点字段是 detached 的（不带 autograd 图）
        """
        m = self.mask
        if not bool(m.any(dim=1).all()):
            bad = (~m.any(dim=1)).nonzero().flatten().tolist()
            raise ValueError(f"第 {bad} 行没有任何要训练的 token（白采了）")

        # 连续性：每行 True 的个数 == (last_valid - first_valid + 1)
        idx = torch.arange(m.shape[1], device=m.device).expand_as(m)
        first = torch.where(m, idx, torch.full_like(idx, m.shape[1])).min(dim=1).values
        last = last_valid_index(m)
        span = (last - first + 1)
        if not bool((span == m.sum(dim=1)).all()):
            raise ValueError("生成位置在行内不连续：next_values_from_values 会算错")

        if bool((self.dones & ~m).any()):
            raise ValueError("dones 出现在非生成位置（padding 或 prompt 上不该有终止）")
        if bool(((self.rewards != 0) & ~m).any()):
            raise ValueError("rewards 在非生成位置上非零（会被当成真实奖励）")
        # actions 在生成位置上必须就是那些 token（LLM 的动作空间 = 词表）。
        # ⚠️ 注意：**当前算法里没有任何地方读 actions** —— PPO 的 ratio 用的是 logprob，
        #    不是 action id。所以它填错了也不会报错，正因为如此才需要在这里钉住。
        if bool((self.actions[m] != self.input_ids[m]).any()):
            raise ValueError(
                "actions 在生成位置上必须等于 input_ids（LLM 的动作空间就是词表；"
                "最省事的填法就是 actions = input_ids.clone()）"
            )
        # 机会性校验：meta 里带了 prompt_mask 就查"生成位置不能落在 prompt 上"。
        #
        # 契约：`meta["prompt_mask"]` 是 **(B, T)**（和 mask 同形状），
        #       True **只标真实 prompt 位置** —— 生成段和 padding 都是 False。
        #       （注意别和"整行的真实 token 掩码"搞混：那等于 prompt_mask | mask，
        #         拿它来比是循环论证，什么也验不出来。我第一次就写错了这个。）
        #
        # 为什么这条重要：RolloutBatch 里没有 prompt_len，check() 自己不知道 prompt
        # 到哪儿结束。而最常犯的错是把 mask 写成 `input_ids != PAD` —— 那样 prompt
        # 位置也被当成"要训练的 token"，**前面所有检查都会放过它**，结果是模型被
        # 顺手训练成"复述 prompt"，而 loss 曲线看着还挺正常。
        prompt_mask = self.meta.get("prompt_mask")
        if isinstance(prompt_mask, Tensor):
            if tuple(prompt_mask.shape) != tuple(m.shape):
                raise ValueError(
                    f"meta['prompt_mask'] 必须是 (B,T)={tuple(m.shape)}（和 mask 同形状），"
                    f"实际是 {tuple(prompt_mask.shape)}。\n"
                    f"        常见原因：存成了 (B, Lp) 的'只有 prompt 段'版本。"
                    f"请在末尾补上生成段的 False："
                    f"`cat([prompt_mask, zeros_like(gen_mask)], dim=1)`"
                )
            if bool((m & prompt_mask.bool()).any()):
                raise ValueError(
                    "mask 和 prompt_mask 有重叠：prompt 位置不该被当成生成 token\n"
                    "        （常见原因：mask 写成了 `input_ids != PAD`。"
                    "正确的是用生成时的 `active` 标志，也就是 `gen_mask`）"
                )
        # 每行的回合必须终止一次：dones 恰好 1 个 True（自然 STOP 或截断）。
        # ⚠️ 全 False 时 GAE 会把"回合结束"当成"还能继续"，terminal 奖励不再终止回报，
        #    优势被 V(s_T) 污染 —— 而这个错误**不会报错**，只会让优势有偏。
        #    （这条校验原来漏了，所以"dones 全 False"的 bug 能一路溜过去。）
        n_done = self.dones.sum(dim=1)
        if not bool((n_done == 1).all()):
            bad = (n_done != 1).nonzero().flatten().tolist()
            raise ValueError(
                f"第 {bad} 行的 dones 个数不是 1（实际 {(n_done[bad]).tolist()}）："
                f"每行的回合必须终止一次\n"
                f"        自然输出 STOP 或被 max_new_tokens 截断，都要在"
                f"**最后一个有效生成位置**标 dones=True"
            )
        if self.advantages is not None:
            if bool((self.advantages[~m].abs() > 1e-6).any()):
                raise ValueError("padding 位置的优势必须为 0（否则污染归一化统计量）")
        # rollout 记录的是**行为策略的常数快照**，不能带 autograd 图。
        # 为什么严重：`ppo_loss` 里 `log_ratio = logprobs_new - logprobs_old`，
        # 如果 old 带着图，backward() 会**同时也**从 old 那条路算梯度 ——
        # 等价于把 (logπ_new − logπ_old(θ)) 对 θ 求导，凭空多出一项 −∂logπ_old/∂θ，
        # 而 PPO 要求 old 是常数（detach）。后果：梯度方向错 + 显存翻倍。
        # 根因通常是 rollout 没有包在 `torch.no_grad()` 里。
        for _name in ("logprobs", "values", "rewards", "advantages", "returns"):
            _t = getattr(self, _name)
            if isinstance(_t, Tensor) and _t.is_floating_point() and _t.requires_grad:
                raise ValueError(
                    f"{_name} 带着 autograd 图（requires_grad=True）：rollout 记录的是"
                    f"行为策略的常数快照，必须是 detached 的\n"
                    f"        常见原因：rollout 没包在 `torch.no_grad()` 里"
                )
        if verbose:
            print(f"  [check] B={self.batch_size} T={self.shape[1]} "
                  f"有效 token={self.num_tokens} 平均生成长度={float(self.gen_len().float().mean()):.1f}")


# =========================================================================== #
# 5. 自我演示：为什么"右 padding + 尾部负索引"会悄悄错
# =========================================================================== #
def _demo() -> None:
    print("=" * 68)
    print("演示：同一个 batch，两种 padding，同一个'尾部切片'写法")
    print("=" * 68)

    # 三条长度不同的序列
    seqs = [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3]]
    gen_lens = [len(s) for s in seqs]

    for side, pad_fn in (("left", pad_left), ("right", pad_right)):
        ids, mask = pad_fn(seqs, pad_id=0)
        print(f"\n--- {side} padding ---")
        print(f"ids  =\n{ids}")
        print(f"mask =\n{mask}")
        # 假装 logprob 就是"位置下标 + 100"，这样一眼能看出取到了哪一段
        fake = (torch.arange(ids.shape[1]).float() + 100).expand_as(ids)
        print(f"逐位置的值 =\n{fake}")
        for i, g in enumerate(gen_lens):
            tail = fake[i, -g:]                       # ← 尾部负索引（上一版的做法）
            bymask = fake[i][mask[i]]                 # ← mask 定位（本框架的做法）
            flag = "✅ 一致" if torch.equal(tail, bymask) else "❌ 错位！取到了 PAD"
            print(f"  样本{i} (len={g}): 尾部切片={tail.tolist()}  mask定位={bymask.tolist()}  {flag}")

    print("\n结论：**mask 定位在两种 padding 下都对；尾部切片只在 left padding 下对。**")
    print("       上一版代码是 right padding（tokenizer 默认）+ 尾部切片 → 静默错位。\n")

    print("=" * 68)
    print("演示：masked_mean 和裸 mean 的差别")
    print("=" * 68)
    x = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, False, False, False]])
    print(f"x.mean()            = {float(x.mean()):.4f}   ← 被 padding 稀释了")
    print(f"masked_mean(x, mask)= {float(masked_mean(x, mask)):.4f}   ← 正确\n")

    print("=" * 68)
    print("演示：一个合法的 RolloutBatch + check()")
    print("=" * 68)
    # prompt 用 left padding（框架统一约定），生成段紧接其后 →
    # 整行的 mask 是"一段连续的 True"，满足第 4 条不变量
    prompts, prompt_mask = pad_left([[1, 2, 3], [1], [1, 2]])          # (3, 3)
    gen_vals = [                                                        # 不等长
        [4, 5],          # 生成了 2 个
        [4],             # 1 个
        [4, 5],          # 2 个
    ]
    G = max(len(g) for g in gen_vals)
    B = len(gen_vals)
    Lp = prompts.shape[1]
    T = Lp + G

    # 逐生成 token 的字段：先铺成全长，再填生成段
    gen_ids = torch.zeros(B, G, dtype=torch.long)
    gen_mask = torch.zeros(B, G, dtype=torch.bool)
    for i, g in enumerate(gen_vals):
        gen_ids[i, :len(g)] = torch.tensor(g)
        gen_mask[i, :len(g)] = True

    def full_prompt_part(dtype, fill=0.0):
        """prompt 位置上的占位（值为 0 / False，反正会被 mask 掉）。"""
        return torch.full((B, Lp), fill, dtype=dtype)

    input_ids = torch.cat([prompts, gen_ids], dim=1)                    # (B, T)
    actions = torch.cat([full_prompt_part(torch.long), gen_ids], dim=1)
    logprobs = torch.zeros(B, T)
    values = torch.zeros(B, T)
    # ⚠️ mask 只标"生成出来的 token"，**prompt 位置必须是 False**。
    #    （我第一版这里写成了 cat([prompt_mask, gen_mask])，把 prompt 也算进了
    #     mask，于是 gen_len() 返回的是"prompt+生成"的长度 —— 形状对、语义错，
    #     正是这个文件要防的那类 bug。）
    mask = torch.cat([torch.zeros_like(prompt_mask), gen_mask], dim=1)  # (B, T)
    # 终局奖励只放在每条序列的最后一个生成位置上
    rewards = torch.zeros(B, T)
    dones = torch.zeros(B, T, dtype=torch.bool)
    for i, g in enumerate(gen_vals):
        rewards[i, Lp + len(g) - 1] = 1.0
        dones[i, Lp + len(g) - 1] = True

    batch = RolloutBatch(
        input_ids=input_ids, actions=actions, logprobs=logprobs, values=values,
        rewards=rewards, dones=dones, mask=mask, version=3,
    )
    batch.check(verbose=True)
    print(f"  mask =\n{mask.int()}")
    print(f"  gen_len() = {batch.gen_len().tolist()}  （只数生成段，应为 {[len(g) for g in gen_vals]}）")
    assert batch.gen_len().tolist() == [len(g) for g in gen_vals], "gen_len 把 prompt 也数进去了"
    nxt = next_values_from_values(batch.values, batch.mask, bootstrap=0.0)
    print(f"  next_values 形状 {tuple(nxt.shape)}（用 mask 推出来的，没用切片）")
    print("\n看第 1 行：mask = [0,0,0,1,0] —— 真实生成 token 只有第 3 列，")
    print("第 4 列是补齐用的 padding。如果按'从尾部数 G=2 个位置'去取，")
    print("就会把第 4 列这个 padding 当成生成 token —— 这就是上一版的 bug。")

    # 故意构造一个坏批次，确认 check() 真的会拦住
    print("\n--- 故意构造一个坏批次（dones 放在了 prompt 位置上）---")
    bad = RolloutBatch(
        input_ids=input_ids, actions=actions, logprobs=logprobs, values=values,
        rewards=rewards, dones=dones.clone(), mask=mask, version=3,
    )
    bad.dones[0, 0] = True        # 第 0 行第 0 列是 prompt，不在 mask 里
    try:
        bad.check()
        print("  ❌ check() 没拦住，说明校验写漏了")
    except ValueError as e:
        print(f"  ✅ check() 拦住：{e}")

    print("\n✅ batch.py 自检通过")


def _case_concat() -> None:
    """★ `concat_batches` 必须**左补** ✓ —— 这条用例的写法本身有一段教训 ✗

    第一版我用 toy 的 batch 造用例 ✓ 但 **toy 是右对齐的** ✓
    当时写的是右补 → 在 toy 上 mask 依然连续 ✓ **测试通过了** ✓
    可是真实 LLM 是**左 padding**（真实 token 贴右边 ✓）✗
      → 右补让 pad 落到**中间** ✗
      → `forward` 取"最后 gen_len 个位置"时取到 pad ✗
      → ratio 起始偏差 **195.3** ✗✗ **只有端到端冒烟抓到了它** ✓

    → 所以这里改成**自己造左 padding 的 batch** ✓，并直接钉住那个被破坏的不变量：
      **每一行的生成段必须仍然是最右边的 G 个位置** ✓✓
    """
    def mk(T: int, Lp: int, G: int) -> RolloutBatch:
        """左 padding：左边 Lp 个 prompt，右边 G 个生成段 ✓"""
        ids = torch.arange(100, 100 + 2 * T, dtype=torch.long).reshape(2, T)
        mask = torch.zeros(2, T, dtype=torch.bool)
        mask[:, T - G:] = True                       # ★ 生成段贴右边 ✓
        prompt_mask = torch.zeros(2, T, dtype=torch.bool)
        prompt_mask[:, T - G - Lp:T - G] = True
        rewards = torch.zeros(2, T)
        rewards[:, -1] = 1.0
        dones = torch.zeros(2, T, dtype=torch.bool)
        dones[:, -1] = True
        b = RolloutBatch(input_ids=ids, actions=ids.clone(),
                         logprobs=torch.zeros(2, T), values=torch.zeros(2, T),
                         rewards=rewards, dones=dones, mask=mask,
                         meta={"prompt_mask": prompt_mask, "gen_width": G})
        b.check()
        return b

    # ★ G 必须相同（这是设计不变量：forward 取"最后 G 个位置" ✗ 不同 G 无法调和 ✓）
    #   真实情况正是：各块 max_new_tokens 相同 ✓ 而 prompt 长度不同 → T 不同 ✓
    b1, b2 = mk(21, 9, 12), mk(16, 4, 12)            # ★ G 同 ✓ T 不同 ✓
    c = concat_batches([b1, b2])
    n1, t1, t2 = 2, 21, 16
    if tuple(c.input_ids.shape) != (4, 21):
        raise AssertionError(f"形状不对 ✗ {tuple(c.input_ids.shape)}，应为 (4, 21)")

    # ★★ 核心不变量：每行的生成段必须仍是最右边的 G 个位置 ✓✓
    for i in range(4):
        if not bool(c.mask[i, -12:].all()):
            raise AssertionError(f"第 {i} 行生成段不在最右边 ✗ —— 右补就会这样 ✗")
        if bool(c.mask[i, :-12].any()):
            raise AssertionError(f"第 {i} 行生成段左边还有 True ✗ mask 不连续了 ✗")
    if last_valid_index(c.mask).tolist() != [20, 20, 20, 20]:
        raise AssertionError(f"last_valid 不对 ✗ {last_valid_index(c.mask).tolist()}")

    # 左补出来的那几列必须全 False ✓（否则 GAE 会走到 pad 上 ✓）
    if bool(c.mask[n1:, :t1 - t2].any()) or bool(c.meta["prompt_mask"][n1:, :t1 - t2].any()):
        raise AssertionError("补齐列里有 True ✗")
    if float(c.rewards[~c.mask].abs().max()) != 0.0:
        raise AssertionError("mask 外有非零奖励 ✗")
    if bool((c.dones & ~c.mask).any()):
        raise AssertionError("mask 外有 done=True ✗（GAE 会在 pad 上 bootstrap ✗）")

    # 真实内容逐位保留（左补 → 后半整体右移 t1-t2 列 ✓）
    for a, b_, ct, name in ((b1.input_ids, b2.input_ids, c.input_ids, "input_ids"),
                            (b1.logprobs, b2.logprobs, c.logprobs, "logprobs")):
        if not torch.equal(ct[:n1, :t1], a):
            raise AssertionError(f"{name} 前半被改了 ✗")
        if not torch.equal(ct[n1:, t1 - t2:], b_):
            raise AssertionError(f"{name} 后半被改了 ✗")

    c.check()                                        # 7 条契约也要过 ✓
    print(f"    T={t1} 与 T={t2} 拼接 → {tuple(c.input_ids.shape)} ✓")
    print("    生成段仍在最右边 + 补齐列全 False + mask 外奖励/done 为 0 ✓")


def _case_concat_mixed_gen_width() -> None:
    """★ 各块 `gen_width` **不同** → 必须取 max，不能报错 ✗

    背景（**真实 run 才暴露的** ✗）：8 块的 gen_width 是 169 / 172 / 180 / 192 ✓
      —— 因为它是那一块里**实际最长的生成** ✗ 不是 max_new_tokens ✓
    我第一版写的是「必须相等，否则抛错」✗ → 生产直接炸 ✓

    取 max 为什么**精确等价** ✓：
      位置 t 的 logits 只依赖 t 之前的前缀 ✓ 与 G 无关 ✓
      mask 只标真实生成 ✓ → 多算出来的位置被排除 ✓ 不进 loss ✓
      （就是 `_case_tail` 验证过的「只算生成段是等价变形」✓）
    """
    def mk(T: int, Lp: int, G: int) -> RolloutBatch:
        ids = torch.arange(100, 100 + 2 * T, dtype=torch.long).reshape(2, T)
        mask = torch.zeros(2, T, dtype=torch.bool)
        mask[:, T - G:] = True
        pm = torch.zeros(2, T, dtype=torch.bool)
        pm[:, T - G - Lp:T - G] = True
        rw = torch.zeros(2, T); rw[:, -1] = 1.0
        dn = torch.zeros(2, T, dtype=torch.bool); dn[:, -1] = True
        b = RolloutBatch(input_ids=ids, actions=ids.clone(),
                         logprobs=torch.zeros(2, T), values=torch.zeros(2, T),
                         rewards=rw, dones=dn, mask=mask,
                         meta={"prompt_mask": pm, "gen_width": G})
        b.check()
        return b

    b1, b2 = mk(15, 6, 5), mk(18, 6, 9)          # G=5 与 G=9 ✓ 不同 ✓
    c = concat_batches([b1, b2])                  # ← 第一版会在这里抛错 ✗
    if int(c.meta["gen_width"]) != 9:
        raise AssertionError(f"应取 max=9 ✗ 实际 {c.meta['gen_width']}")
    for i in range(4):
        if not bool(c.mask[i, -9:].any()):
            raise AssertionError(f"第 {i} 行在最后 9 个位置里没有真实生成 ✗")
    c.check()
    print(f"    G=5 与 G=9 → 取 max=9 ✓ 不报错 ✓（真实 run 就是这么炸的 ✗）")


def _case_gen_start() -> None:
    """★ gen_start 的自测（原来它**一个自测都没有** ✗）

    左 padding 的真实形状：`[F,F,T,T,T]` → 起点必须是 **2** ✓
    右 padding：`[T,T,T,F,F]` → 起点必须是 **0** ✓
    全 False 行：**必须报错** ✗（不能静默返回 0 → 那会把 prompt 当生成 ✓）
    """
    left = torch.tensor([[False, False, True, True, True],
                         [False, True, True, True, True]])
    got = gen_start(left)
    if got.tolist() != [2, 1]:
        raise AssertionError(f"左 padding 起点应为 [2,1] ✗ 实际 {got.tolist()}")
    right = torch.tensor([[True, True, True, False, False]])
    if gen_start(right).tolist() != [0]:
        raise AssertionError("右 padding 起点应为 [0] ✗")
    try:
        gen_start(torch.zeros(1, 4, dtype=torch.bool))
        raise AssertionError("全 False 行竟然没报错 ✗（会静默把 prompt 当生成 ✓）")
    except AssertionError as e:
        if "竟然没报错" in str(e):
            raise
    print("    gen_start：左 [2,1] ✓ 右 [0] ✓ 全 False 报错 ✓")


if __name__ == "__main__":
    _demo()
    print("\n用例：concat_batches（不同 T、左 padding）")
    _case_concat()
    print("\n用例：concat_batches（各块 gen_width 不同）")
    _case_concat_mixed_gen_width()
    _case_gen_start()

