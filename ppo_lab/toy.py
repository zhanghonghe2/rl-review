"""可验证奖励的玩具任务 + 小模型 —— 让 PPO 能在 3 秒内跑完一次实验。

为什么不用 GSM8K / 真 LLM 起步
----------------------------
真模型一次 rollout 就是几十秒起步，你会把时间全花在等 MPS 上，而不是理解 PPO。
所以 M3 用这个文件：**同样的结构（prompt + 自回归生成 + 终局可验证奖励），
但快到可以随便做实验**。

任务：数出输入里 1 的个数，用 unary 输出
---------------------------------------
    输入 prompt : [BOS] + [IN0/IN1] × n        (n ∈ [4, 8])
    正确回答    : [MARK] × k + [STOP]          (k = 输入里 IN1 的个数)
    每步代价    : -step_cost（默认 -0.01）
    终局奖励    : +1.0（解析出的个数 == k），否则 0.0

词表：PAD=0, BOS=1, IN0=2, IN1=3, MARK=4, STOP=5（V=6）

四个必须满足的性质（达不到 M3 就白做）
------------------------------------
1. **奖励可自动判定** —— 精确匹配，不调任何模型打分
2. **序列长度可变** —— 输出长度 = 答案数值，天然不等长
3. **batch 内长度不一** —— prompt 长度随机，强制走 mask 路径
4. **逐步代价 + 终局奖励** ← 最容易漏，也是最重要的

第 4 条展开（**这里我第一版写错过，留个记录**）：

我最初写的是"没有逐步代价时 λ 在数学上完全不起作用"——**这是错的**。
`gae.py::_case_a` 本身就是反例：那条用例的奖励是纯终局的 [0, 0, 1]（无逐步代价），
而 λ=0 → [0.3, 0.3, 0.2]、λ=0.5 → [0.5, 0.4, 0.2]、λ=1 → [0.8, 0.5, 0.2]，
λ 的影响很大，根本没退化。（这个用例早就通过了，等于我们俩都没注意到那句话是错的。）

正确的说法：

* **没有逐步代价、γ=1 时**，序列内每个 token 的回报是同一个常数 R。
  于是 A_t = R - V(s_t) 的差异**全部来自 critic 的估计误差**，
  中间 token 不携带任何"环境结构"信息（只有终局那一步有真实信号）。
* **加上每步 -0.01 之后**，回报变成 R - c·(剩余步数)，**"还剩多少步"进入了回报**：
  每个 token 都带一部分真实信号，critic 要学"距离结束还有多远"，
  λ 的偏差/方差权衡也才有实际意义。

逐步代价真正不可替代的两个理由：

1. **没有它，提前输出 STOP 没有任何收益** → 模型学不会停止 →
   生成长度会一直钉在 `max_new_tokens` → `rollout` 里那条"早停"路径
   在训练中永远走不到（而它恰恰是最容易写错的地方）。
2. **它才是真实 LLM RL 的结构**：逐 token 的 KL 惩罚就是逐步代价。

布局约定（和 batch.py 的契约严格对齐）
-------------------------------------
* prompt **左 padding**：真实 token 右对齐，于是"prompt 真实段 + 生成段"
  在整行里是一段**连续的** True → 满足 `RolloutBatch.check()` 的连续性要求。
  （右 padding 会在中间留洞，直接违反契约。详见 batch.py 的 docstring。）
* `mask` 只标**生成出来的 token**，prompt 位置和 padding 位置都是 False。
* `values[b,t]` = "生成第 t 个 token **之前**"的 V(s_t)，与 `logprobs[b,t]` 时间对齐。
* 终局奖励和 `dones` 都放在**该行最后一个有效生成位置**。

留给你实现的两个函数（都是契约关键，不是管道）
--------------------------------------------
* `parse_count()` —— 从生成序列解析个数。边界条件比看起来多。
* `rollout()`     —— 组装 RolloutBatch。values/rewards/dones/mask 的对齐都在这儿，
                     也是 `_case_c`（ratio 起手为 1）和 `_case_d`（padding 不变性）
                     能通过的前提。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from batch import RolloutBatch, last_valid_index, pad_left, token_logprobs
from config import PPOConfig

# --------------------------------------------------------------------------- #
# 词表
# --------------------------------------------------------------------------- #
PAD, BOS, IN0, IN1, MARK, STOP = 0, 1, 2, 3, 4, 5
VOCAB = 6
TOKEN_NAMES = {PAD: "_", BOS: "B", IN0: "0", IN1: "1", MARK: "m", STOP: "."}


# =========================================================================== #
# 1. 小模型：一个手写逐步循环的 actor-critic（已实现，重点是那个 mask）
# =========================================================================== #
class TinyActorCritic(nn.Module):
    """embedding + GRUCell + 策略头 + 价值头。

    为什么用手写逐步循环而不是 nn.GRU
    --------------------------------
    因为要**显式处理 padding**：PAD 位置不能更新隐状态，否则 padding 的个数会
    改变后面所有 token 的表示 —— 那就是"padding 影响结果"，也就是 `_case_d` 要抓的错。

    `forward` 返回三个东西，时间对齐如下（第 t 个位置）::

        logits[:, t]      = π(·|s_t)     ← 生成 token t **之前**的策略分布
        values[:, t]      = V(s_t)       ← 生成 token t **之前**的状态价值
        values_next[:, t] = V(s_{t+1})   ← 吃掉 token t **之后**的状态价值

    `values_next` 一并返回，是为了让 GAE 的 bootstrap 语义明确 ——
    不用在别处猜"下一个位置的价值"该怎么取。
    """

    def __init__(self, vocab: int = VOCAB, hidden: int = 64) -> None:
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.emb = nn.Embedding(vocab, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, vocab)
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def step_logits_values(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """给定隐状态，算出这一步的策略 logits 和价值。"""
        return self.pi(h), self.v(h).squeeze(-1)

    def advance(self, h: Tensor, tokens: Tensor) -> Tensor:
        """把 tokens 喂进 GRU，返回新的隐状态（不处理 mask）。"""
        return self.cell(self.emb(tokens), h)

    def forward(self, input_ids: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """整序列前向。返回 (logits, values, values_next)，形状 (B,T,V) / (B,T) / (B,T)。

        `token_mask[b,t]=True` 表示该位置是**真实 token**（prompt 真实段或生成段），
        False 表示 padding —— **False 的位置不更新隐状态**。
        """
        B, T = input_ids.shape
        h = torch.zeros(B, self.hidden, dtype=self.emb.weight.dtype, device=input_ids.device)
        logits, values, values_next = [], [], []
        for t in range(T):
            lg, val = self.step_logits_values(h)          # 用"吃掉 token t 之前"的 h
            h_new = self.advance(h, input_ids[:, t])
            _, val_next = self.step_logits_values(h_new)
            logits.append(lg)
            values.append(val)
            values_next.append(val_next)
            # ⚠️ 只在这里处理 padding：mask 为 False 时保持 h 不变
            h = torch.where(token_mask[:, t:t + 1], h_new, h)
        return (torch.stack(logits, dim=1),
                torch.stack(values, dim=1),
                torch.stack(values_next, dim=1))

    @torch.no_grad()
    def run_prefix(self, prompt_ids: Tensor, prompt_mask: Tensor) -> Tensor:
        """只吃 prompt，返回读完 prompt 之后的隐状态 h。"""
        h = torch.zeros(prompt_ids.shape[0], self.hidden,
                        dtype=self.emb.weight.dtype, device=prompt_ids.device)
        for t in range(prompt_ids.shape[1]):
            h_new = self.advance(h, prompt_ids[:, t])
            h = torch.where(prompt_mask[:, t:t + 1], h_new, h)
        return h


# =========================================================================== #
# 2. 任务：随机 prompt + 确定性构造（测试用）
# =========================================================================== #
@dataclass
class CountOnesTask:
    """数 1 的个数。`encode()` 用于测试里构造确定性输入。"""

    min_len: int = 4
    max_len: int = 8
    step_cost: float = 0.01

    # ---------------------------------------------------------------- #
    def encode(self, patterns: list[list[int]]) -> tuple[Tensor, Tensor, Tensor]:
        """把 0/1 模式列表编成 (prompt_ids, prompt_mask, counts)，**左 padding**。

        patterns = [[1,0,1,1], [0,0,1]] → 两条 prompt，第二条左边补 PAD。
        """
        seqs = [[BOS] + [IN1 if b else IN0 for b in p] for p in patterns]
        ids, mask = pad_left(seqs, pad_id=PAD)
        counts = torch.tensor([sum(p) for p in patterns], dtype=torch.long)
        return ids, mask, counts

    def sample(self, batch: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """随机采 batch 条 prompt，长度在 [min_len, max_len] 之间。"""
        patterns = []
        for _ in range(batch):
            n = int(torch.randint(self.min_len, self.max_len + 1, (1,), generator=generator))
            bits = torch.randint(0, 2, (n,), generator=generator).tolist()
            patterns.append(bits)
        return self.encode(patterns)

    # ---------------------------------------------------------------- #
    def parse_count(self, gen_ids: Tensor, gen_mask: Tensor) -> Tensor:
        """从生成序列解析出"模型认为有几个 1"，返回 (B,) 的 long 张量。

        ⚠️ 边界条件比看起来多，实现前先把这四种情况想清楚：

        1. **正常结束**：`[m, m, m, STOP]` → 3
        2. **STOP 之后还有内容**：`[STOP, m, m, m]` → 0
           （STOP 之后的 MARK **必须忽略**，否则模型可以"先停再说"）
        3. **从不输出 STOP**（撞上 max_new_tokens）：`[m, m, PAD, PAD]` → 数到有效位置末尾 → 2
        4. **padding 位置上的垃圾 token**：`gen_mask=False` 的位置**必须忽略**，
           不管它的 id 是 MARK 还是别的

        提示：可以像下面这样一次算完，也可以写循环 —— 但**必须用 gen_mask**，
        不能假设"STOP 一定在最后"或者"有效位置一定连续到末尾"。
        """

        out = torch.zeros([gen_ids.shape[0]])

        for sample in range(gen_ids.shape[0]):
            count = 0
            ids_sample, mask_sample = gen_ids[sample], gen_mask[sample]    
            for idx in range(ids_sample.shape[0]):
                if mask_sample[idx] == False:
                    continue
                else:
                    c = ids_sample[idx].item()
                    if c == STOP:
                        break
                    elif c == MARK:
                        count += 1
                    else:
                        continue
            out[sample] = count

        return out

        raise NotImplementedError("解析生成序列里的 MARK 个数（注意四个边界条件）")

    def reward(self, gen_ids: Tensor, gen_mask: Tensor, counts: Tensor) -> Tensor:
        """精确匹配奖励：解析出的个数 == 真实个数 → 1.0，否则 0.0。形状 (B,)。

        （想加"部分分"的话在这里改 —— 但要清楚它会改变奖励的结构：
          部分分是密集奖励，会让 GAE 的 λ 和终局稀疏奖励下的行为完全不同。
          先用精确匹配跑通，别一上来就加 shaping。）
        """
        parsed = self.parse_count(gen_ids, gen_mask)
        return (parsed == counts).to(torch.float32)


# =========================================================================== #
# 3. rollout：把"生成 + 奖励 + 组装批次"串起来
# =========================================================================== #
def rollout(
    task: CountOnesTask,
    model: TinyActorCritic,
    cfg: PPOConfig,
    generator: torch.Generator | None = None,
    prompts: tuple[Tensor, Tensor, Tensor] | None = None,
) -> RolloutBatch:
    """采一批 prompt（或用传入的）→ 自回归生成 → 组装成 `RolloutBatch`。

    参数
    ----
    cfg : 用 `cfg.rollout_prompts`（batch 大小）、`cfg.max_new_tokens`、
          `cfg.temperature`（>0 采样；<=0 贪心，测试里用贪心保证确定性）
    generator : 随机源，保证可复现
    prompts : 可选 (prompt_ids, prompt_mask, counts)，不传就 `task.sample(...)`

    要填的字段（**这些对齐关系就是 batch.py 的契约**）
    ------------------------------------------------
    input_ids : `cat([prompt_ids, gen_ids], dim=1)`，形状 (B, Lp+G)
    actions   : 生成位置的 token id（prompt 位置填 0 即可，反正被 mask 掉）
    logprobs  : **行为策略**在每个生成 token 上的 logprob —— rollout 时算一次存下来
    values    : 每个生成位置"生成该 token 之前"的 V(s_t)
    mask      : 只标生成出来的 token（= 生成时的 active 位）
    rewards   : 每个有效生成位置放 `-task.step_cost`，
                **最后一个有效位置额外加上终局奖励** `task.reward(...)`
    dones     : **最后一个有效位置** True，其余 False
    version   : 0（同步；异步时由 runner 覆盖）
    meta      : 必须放进 `{"prompt_mask": prompt_mask_full, "counts": counts}`，
                其中 `prompt_mask_full` 是 **(B, T)** —— 和 mask 同形状，
                **True 只标真实 prompt 位置**（生成段和 padding 都是 False）。
                要的是 "整行版" 而不是 (B, Lp) 的 "只有 prompt 段" 版：
                在末尾补上生成段的 False 即可 ——
                `cat([prompt_mask, zeros_like(gen_mask)], dim=1)`。
                用途：`check()` 靠它验证"mask 没把 prompt 位置算进来"；
                测试靠它重建整行掩码（`prompt_mask | mask`）。

    两个关键的自洽要求（`_case_c` / `_case_d` 会验）
    ---------------------------------------------
    1. **生成时的 logprob 必须和 `model.forward` 重算的完全一致**。
       所以两者要共用同一条计算路径：都用 `step_logits_values` + 同样的 mask 处理。
       一旦"生成用一套、重算用另一套"，ratio 起手就不是 1，裁剪目标直接失效。
    2. **每行在第一个 STOP 之后要停止更新隐状态**（active 变 False），
       并且后续位置的 `mask` 为 False —— 这与 `forward` 里"padding 不更新 h"是同一件事，
       两边必须一致。

    实现提示
    --------
    * 先 `h = model.run_prefix(prompt_ids, prompt_mask)` 拿到读完 prompt 的隐状态。
    * 然后**循环 `G = cfg.max_new_tokens` 步** —— 这就是"生成到 STOP 或最大长度"的
      **批量化实现**：batch 里各行停的时刻不同（有的第 3 步就输出 STOP，有的跑满 G 步），
      所以不能用 `while 直到 STOP`，只能循环固定 G 次 + 用 `active` 掩码逐步排除已结束的行。
      每步：算 logits/values → 采样（或 argmax）→ 记 logprob → 喂回 GRU。
      三行关键：`gen_mask[:, k] = active`（记录这步是否真生成了）、
      `h = torch.where(active[:, None], h_new, h)`（冻结已结束行的隐状态）、
      `active = active & (tok != STOP)`（推进状态机）。
      可选优化：`if not active.any(): break`，剩下的列全是 mask=False。
    * 注意区分**自然结束**（输出了 STOP）和**被截断**（跑满 G 步）。本任务里两种都按
      `dones[last]=True` 处理是合理的 —— 因为判分就是按截断后的输出做的。
      通用 RL 里时间限制截断通常应该 bootstrap，但那是以后要做的实验，先别改。
    * `forward` 需要的是**整行真实 token 掩码** = `prompt_mask | mask`（两者都是 (B,T)）。
      ⚠️ `meta["prompt_mask"]` 是 **prompt-only**：末尾要补 **False**，
      `cat([prompt_mask, torch.zeros_like(gen_mask)], dim=1)`。
      **千万别把 `gen_mask` 拼进去** —— 那样它就和 `mask` 在生成段上重叠，`check()` 会（正确地）报错。
    * 最后一个有效位置的下标：`last_valid_index(gen_mask)`（batch.py 里的模块级函数）。
    """

    if prompts is None:
        prompts = task.sample(cfg.rollout_prompts)
    prompt_ids, prompt_mask, counts = prompts
    batch_size = prompt_ids.shape[0]
    
    h = model.run_prefix(prompt_ids, prompt_mask)
    active = torch.ones(batch_size, dtype=torch.bool, device=prompt_ids.device)
    
    gen_ids = torch.zeros([batch_size, cfg.max_new_tokens], dtype=torch.long, device=prompt_ids.device)
    gen_mask = torch.zeros([batch_size, cfg.max_new_tokens], dtype=torch.bool, device=prompt_ids.device)
    gen_prob = torch.zeros([batch_size, cfg.max_new_tokens], dtype=torch.float, device=prompt_ids.device)
    gen_values = torch.zeros([batch_size, cfg.max_new_tokens], dtype=torch.float, device=prompt_ids.device)

    with torch.no_grad():
        for k in range(cfg.max_new_tokens):
            logits, value = model.step_logits_values(h)
            if cfg.temperature > 0:
                tok = torch.multinomial(torch.softmax(logits / cfg.temperature, -1), 1).squeeze(-1)
            else:
                tok = logits.argmax(-1)
                
            lp = torch.log_softmax(logits, dim=-1) # [batch_size, vocab]
            logprob = lp.gather(1, tok.unsqueeze(-1)).squeeze(-1) #[batch_size]
            gen_ids[:, k] = torch.where(active, tok, PAD)
            gen_mask[:, k] = active
            gen_prob[:, k] = logprob
            gen_values[:, k] = value
            h = torch.where(active[:, None], model.advance(h, tok), h)
            active = active & (tok != STOP)     
            if not active.any(): 
                break  

    input_ids = torch.cat([prompt_ids, gen_ids], dim=1) # [batch_size, len_prompt + len_gen]
    action = torch.zeros_like(input_ids, device=input_ids.device)
    action[:, prompt_ids.shape[1]: ] = gen_ids

    logprobs = torch.zeros_like(input_ids, dtype=torch.float, device=input_ids.device)
    logprobs[:, prompt_ids.shape[1]: ] = gen_prob

    values = torch.zeros_like(input_ids, dtype=torch.float, device=input_ids.device)
    values[:, prompt_ids.shape[1]: ] = gen_values

    rewards = torch.zeros_like(input_ids, dtype=torch.float, device=input_ids.device)
    # rewards[:, -1] = task.reward(gen_ids, gen_mask, counts)

    dones = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
    # dones[:, -1] = ~active

    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
    mask[:, prompt_ids.shape[1]: ] = gen_mask

    last = last_valid_index(mask)  
    rows = torch.arange(batch_size)

    rewards[:, prompt_ids.shape[1]:] = -task.step_cost * gen_mask
    rewards[rows, last] += task.reward(gen_ids, gen_mask, counts)
    dones[rows, last] = True 


    prompt_mask = torch.cat([prompt_mask, torch.zeros_like(gen_mask)], dim=-1)

    # active = active.unsqueeze(-1)


    return RolloutBatch(
        input_ids=input_ids,
        actions=action,
        logprobs=logprobs,
        values=values, 
        rewards=rewards,
        dones=dones,
        mask=mask,
        version=0,
        # advantages=torch.Tensor([0]),
        # returns=torch.Tensor([0]),
        meta={"prompt_mask": prompt_mask, "counts": counts}
    )
    

    raise NotImplementedError("生成 + 奖励 + 组装 RolloutBatch")


# =========================================================================== #
# 4. 测试用例
# =========================================================================== #
def _model(seed: int = 0) -> TinyActorCritic:
    """测试用：固定随机初始化，保证用例**可复现**。

    （之前直接 `TinyActorCritic()`，权重每次不同 → 有的运行某行提前输出 STOP、
      有的不输出 → 同一个用例时过时不过。测试必须可复现，否则"红/绿"没有意义。）
    """
    torch.manual_seed(seed)
    return TinyActorCritic()


def _case_a() -> None:
    """奖励判定：解析的四个边界条件。

    ⚠️ 两个列容易抄混，注意它们的区别：
      * `真实个数`    = 题目答案（prompt 里 1 的个数），和环境有关
      * `期望解析个数` = **模型说了几**（从生成序列解析出来的），和答案无关
    前四例模型都答对了，所以这两列**恰好相等**；⑤⑥ 是答错的，它们才分岔。
    （我第一版就是把 ⑤⑥ 的"期望解析"抄成了"真实个数"。）

    ⑤⑥ 故意答错，用来确认奖励真的会判 0 —— 不然奖励可能恒为 1 而看不出来。
    """
    task = CountOnesTask()
    M, S, P = MARK, STOP, PAD
    T, F = True, False
    #        真实个数  生成序列             gen_mask      期望解析个数  期望奖励
    cases = [
        (3, [M, M, M, S],          [T, T, T, T],  3, 1.0),   # ① 正常
        (0, [S, M, M, M],          [T, T, T, T],  0, 1.0),   # ② STOP 之后的 MARK 必须忽略
        (2, [M, M, P, P],          [T, T, F, F],  2, 1.0),   # ③ 从不输出 STOP → 数到末尾
        (1, [M, S, M, M],          [T, T, T, T],  1, 1.0),   # ④ STOP 在中间
        (2, [M, M, M, S],          [T, T, T, T],  3, 0.0),   # ⑤ 说了 3，实际 2 → 0
        (0, [M, S, P, P],          [T, T, F, F],  1, 0.0),   # ⑥ 说了 1，实际 0 → 0
    ]
    for i, (true_k, gen, gm, exp_parsed, exp_r) in enumerate(cases, 1):
        # ---- 先验"测试数据自己是否自相矛盾" ----
        # 奖励必须由 (期望解析, 真实个数) 唯一决定。这条能挡住"抄错列"这类错误 ——
        # 否则测试失败会被误读成"实现错了"，白白去追一个不存在的 bug。
        want_r = 1.0 if exp_parsed == true_k else 0.0
        if abs(want_r - exp_r) > 1e-9:
            raise AssertionError(
                f"第 {i} 例的测试数据自相矛盾：期望解析 {exp_parsed}、真实个数 {true_k}"
                f" → 奖励应为 {want_r}，但写成了 {exp_r}（这一行是测试数据写错了，不是实现错）"
            )
        gen_ids = torch.tensor([gen])
        gen_mask = torch.tensor([gm])
        counts = torch.tensor([true_k])
        parsed = task.parse_count(gen_ids, gen_mask)
        r = task.reward(gen_ids, gen_mask, counts)
        if int(parsed[0]) != exp_parsed or abs(float(r[0]) - exp_r) > 1e-6:
            raise AssertionError(
                f"第 {i} 例：生成 {gen} mask {gm}，真实个数 {true_k}\n"
                f"        期望解析 {exp_parsed} 奖励 {exp_r}；"
                f"实际解析 {int(parsed[0])} 奖励 {float(r[0])}"
            )
        print(f"    第{i}例 parsed={int(parsed[0])} reward={float(r[0]):.1f}  ✅")


def _case_b() -> None:
    """rollout 的结构不变量：长度可变、对齐正确、check() 通过。

    这是"能用"的最低标准：
      * batch 内生成长度**不全都一样**（否则 mask 路径没被真正走通）
      * `rewards` 只在 mask 位置非零，且**只有最后一个有效位置**带终局奖励
      * `dones` 只在最后一个有效位置为 True
      * `RolloutBatch.check()` 通过（mask 连续、dones/rewards 不在 padding 上）
    """
    cfg = PPOConfig.preset("toy_debug").replace(rollout_prompts=16, max_new_tokens=12)
    task = CountOnesTask()
    model = _model(cfg.seed)
    batch = rollout(task, model, cfg, generator=torch.Generator().manual_seed(cfg.seed))

    batch.check(verbose=True)
    gl = batch.gen_len()
    if int(gl.unique().numel()) < 2:
        raise AssertionError(f"batch 内所有序列生成长度都一样（{gl.tolist()}）—— "
                             f"这个任务的意义就是变长，检查生成循环是不是哪里截断了")
    print(f"    生成长度分布：min={int(gl.min())} max={int(gl.max())} "
          f"不同取值 {int(gl.unique().numel())} 种  ✅")

    # rewards：非最后位置应该恰好是 -step_cost
    last = last_valid_index(batch.mask)
    rows = torch.arange(batch.batch_size)
    for b in range(batch.batch_size):
        vs = batch.rewards[b][batch.mask[b]]
        # ⚠️ gen_len 可能 == 1（第 0 步就输出 STOP）→ vs[:-1] 是空张量，
        #    对空张量调 .max() 会 RuntimeError。先判长度再取 max。
        if vs.numel() > 1 and vs[:-1].abs().sub(task.step_cost).abs().max() > 1e-6:
            raise AssertionError(
                f"第 {b} 行的非终局步骤代价不是 -{task.step_cost}：{vs.tolist()}\n"
                f"        → 装配 rewards 时漏了「逐步代价」那一行。要在加终局奖励**之前**写：\n"
                f"          `rewards[:, Lp:] = -task.step_cost * gen_mask`\n"
                f"        （不是可选项：没有它，提前输出 STOP 就没有收益，"
                f"模型学不会停止，而且和真实 LLM RL 的结构不一致）"
            )
        if not bool(batch.dones[b, last[b]]):
            raise AssertionError(f"第 {b} 行的最后一个有效位置没有 dones=True")
        if int(batch.dones[b].sum()) != 1:
            raise AssertionError(f"第 {b} 行有 {int(batch.dones[b].sum())} 个 dones，应该恰好 1 个")
    print(f"    每步代价 + 终局奖励 + dones 都只落在最后一个有效位置  ✅")
    print(f"    本批奖励：{batch.rewards[rows, last].tolist()}")

    # ---- cfg.temperature 和 generator 是否真的被尊重 ----
    # 贪心（temperature=0）应该**无视**随机源；采样（>0）应该**用**随机源。
    # 这条能抓住"rollout 一直 argmax、完全没探索"这类问题 —— 它对数值正确性
    # 毫无影响（用例 C 照样过），但会让 PPO 的探索彻底消失。
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # temperature=0 会触发"PPO 需要探索"的提醒
        cfg_greedy = cfg.replace(temperature=0.0)
    fixed = task.encode([[1, 1, 0, 1], [0, 1, 0, 1, 1], [1, 0, 1]])
    a = rollout(task, model, cfg_greedy, torch.Generator().manual_seed(1), fixed)
    b = rollout(task, model, cfg_greedy, torch.Generator().manual_seed(2), fixed)
    if not torch.equal(a.input_ids, b.input_ids):
        raise AssertionError("temperature=0（贪心）时，不同随机源应该给出**相同**结果")
    c = rollout(task, model, cfg, torch.Generator().manual_seed(1), fixed)
    d = rollout(task, model, cfg, torch.Generator().manual_seed(2), fixed)
    if torch.equal(c.input_ids, d.input_ids):
        raise AssertionError(
            "temperature>0（采样）时，不同随机源却给出**相同**结果\n"
            "        → rollout 忽略了 cfg.temperature（或 generator），一直在做 argmax。\n"
            "        后果：每个 prompt 永远只产生一条轨迹，优势估计退化（PPO 需要探索）。"
        )
    print("    temperature / generator 都被尊重（贪心无视随机源，采样使用随机源）  ✅")


def _case_c() -> None:
    """**ratio 起手必须为 1**（T1）—— 这是整个 PPO 最容易悄悄坏掉的一环。

    rollout 时记录的 logprob，必须和"重新 forward 算一遍"的 logprob **逐位相等**。
    同理 `values` 也要相等（value 的对齐错了不会报错，只会让优势有偏）。

    不通过通常意味着：
      * 生成和重算走了两条不同的代码路径（比如一个用左 padding、另一个用右 padding）
      * padding 位置影响了隐状态（`torch.where` 漏了）
      * 生成段的位置和 mask 的位置错位了一个 token
    """
    cfg = PPOConfig.preset("toy_debug").replace(rollout_prompts=12)
    task = CountOnesTask()
    model = _model(cfg.seed)
    batch = rollout(task, model, cfg, generator=torch.Generator().manual_seed(0))

    # meta['prompt_mask'] 是 (B, T)：True 只标真实 prompt 位置。
    # forward 需要的是"整行真实 token 掩码" = prompt_mask | mask（两者都是 (B,T)，直接或）。
    prompt_mask = batch.meta.get("prompt_mask")
    if prompt_mask is None:
        raise AssertionError("rollout 没有把 prompt_mask 放进 batch.meta（契约要求 (B,T)）")
    if tuple(prompt_mask.shape) != tuple(batch.mask.shape):
        raise AssertionError(f"meta['prompt_mask'] 应为 (B,T)={tuple(batch.mask.shape)}，"
                             f"实际 {tuple(prompt_mask.shape)}")
    token_mask = prompt_mask | batch.mask

    with torch.no_grad():          # 只做一致性对拍，不需要建图
        logits, values, _ = model(batch.input_ids, token_mask)
        lp_re = token_logprobs(logits, batch.input_ids)
        d_lp = ((lp_re - batch.logprobs).abs() * batch.mask).max()
        d_v = ((values - batch.values).abs() * batch.mask).max()
    if float(d_lp) > 1e-4:
        raise AssertionError(
            f"重算的 logprob 和 rollout 记录的最大偏差 {float(d_lp):.4f}\n"
            f"        → 生成和重算不是同一条计算路径。ratio 起手就不是 1，"
            f"PPO 的裁剪目标会失效"
        )
    if float(d_v) > 1e-4:
        raise AssertionError(f"重算的 values 和 rollout 记录的最大偏差 {float(d_v):.4f}（对齐错了）")
    print(f"    重算 logprob 最大偏差 {float(d_lp):.2e}  ✅（ratio 起手 = 1）")
    print(f"    重算 values  最大偏差 {float(d_v):.2e}  ✅")


def _case_d() -> None:
    """padding 不变性（T2）：单独跑 vs 和长序列拼 batch，结果必须一模一样。

    用**贪心**（temperature=0）保证确定性，这样两个 batch 的生成结果可以逐位比较。
    如果拼 batch 就变了，说明 padding 影响了计算 —— `batch.py` 那整页的教训就在这里。
    """
    import warnings

    # temperature=0 会触发 config 里"PPO 需要探索"的警告 —— 那是给训练用的提醒，
    # 这里的贪心只是为了让两个 batch 的对比**确定**，所以临时静音。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = PPOConfig.preset("toy_debug").replace(rollout_prompts=1, temperature=0.0,
                                                    max_new_tokens=12)
    task = CountOnesTask()
    model = _model(cfg.seed)

    short = [1, 1, 0]           # 答案 2
    long_ = [1, 0, 1, 1, 0, 1, 1, 1]   # 答案 6，明显更长

    solo = task.encode([short])
    both = task.encode([short, long_])

    b_solo = rollout(task, model, cfg, prompts=solo)
    b_both = rollout(task, model, cfg, prompts=both)

    # ⚠️ 两次的 T 不同（第二条 prompt 更长），所以**不能用绝对位置切片比较**，
    #    必须用 mask 把"生成段"提出来再比 —— 这正是 batch.py 反复强调的那一点。
    gen_solo = b_solo.input_ids[0][b_solo.mask[0]]
    gen_both = b_both.input_ids[0][b_both.mask[0]]
    lp_solo = b_solo.logprobs[0][b_solo.mask[0]]
    lp_both = b_both.logprobs[0][b_both.mask[0]]

    same_len = gen_solo.numel() == gen_both.numel()
    same_ids = same_len and bool((gen_solo == gen_both).all())
    d_lp = (lp_solo - lp_both).abs().max() if same_len else torch.tensor(float("inf"))
    if not same_ids or float(d_lp) > 1e-5:
        raise AssertionError(
            f"同一条 prompt 单独跑 vs 拼 batch 结果不一致\n"
            f"        生成长度 {gen_solo.numel()} vs {gen_both.numel()}，"
            f"生成 token 相同={same_ids}，logprob 最大偏差 {float(d_lp):.2e}\n"
            f"        → padding 影响了计算。检查 `torch.where(token_mask[:, t:t+1], h_new, h)` "
            f"是否在**每一步**都生效"
        )
    print(f"    单独跑 vs 拼 batch：生成 token 相同 {same_ids}，生成长度 {gen_solo.numel()}，"
          f"logprob 最大偏差 {float(d_lp):.2e}  ✅")


if __name__ == "__main__":
    cases = [("用例 A：奖励判定的四个边界", _case_a),
             ("用例 B：rollout 结构不变量", _case_b),
             ("用例 C：ratio 起手必须为 1", _case_c),
             ("用例 D：padding 不变性", _case_d)]
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
        print(f"🎉 {len(cases)} 个用例全部通过 —— toy.py 可以进 trainer.py 了")
    else:
        print(f"进度：{passed}/{len(cases)} 通过（NotImplementedError 是正常的）")
