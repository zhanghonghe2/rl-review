"""模型（actor-critic）+ rollout —— 真实 LLM 版本。

和玩具的分工一一对应：

    toy.TinyActorCritic   ←→   LMWithValueHead
    toy.rollout           ←→   rollout

**契约完全一样**，所以 `update.ppo_update` 一行都不用改：
`forward(input_ids, token_mask)` 返回 `(logits, values, ...)`，第三个值没人用。

这个文件里有三块：
    ① `load_tokenizer` —— 左 padding + pad_token（契约要求）
    ② `LMWithValueHead` —— 共享主干 + lm_head + value_head
    ③ `rollout` —— ★ **留给你实现**（骨架 + 契约 + 验收标准都在 docstring 里）

    python3 model.py     # 跑自测
"""

from __future__ import annotations

import contextlib
import os
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch import gen_start, RolloutBatch, last_valid_index, token_logprobs, last_valid_index
from config import PPOConfig
from data import build_messages, gated_reward, reward_fn, reward_parts
from update import _logprobs_new

# 默认设备。做成模块级常量 + 构造参数，**不要**在类里直接引用全局 ——
# 否则想用 CPU 跑个快速自测都得改源码。
# ⚠️ 用 PPOConfig.auto_device()：它按 **cuda → mps → cpu** 挑。
#   自己写 `"mps" if mps else "cpu"` 的话，在**有 CUDA 的机器上会静默退回 CPU**
#   —— 能跑，但慢 10~50 倍，而且不会报错。
_DEVICE = PPOConfig.auto_device()


def empty_cache() -> None:
    """把缓存分配器占着不放的显存还给系统。**两端都安全。**

    MPS 上**必须**调：实测不调会让 driver 足迹从 20GB 涨到 40GB（超过 32GB 物理
    内存 → 系统换页），单步更新时间从 9.6s 变成 **247s** ✗
    CUDA 上分配器自己管得更好，调了也无害 ✓（用 `is_available()` 分开判断，
    不会在 CUDA 机器上调到不存在的 `torch.mps`）
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def mem_str() -> str:
    """当前设备已分配的显存（GB）字符串；取不到就返回空串。"""
    if torch.backends.mps.is_available():
        return f"{torch.mps.driver_allocated_memory() / 1e9:.1f}G"
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / 1e9:.1f}G"
    return ""


def to_pt_dtype(name: str | None):
    """字符串 → torch.dtype；None 原样返回。"""
    return None if name is None else getattr(torch, name)

# =========================================================================== #
# ★★★ LoRA 结构 —— 提到**模块级** ✓✓ 让 `prm_train` 的**配置指纹**能读到 ✗✓
#
#   ✗ 原来硬编码在 `LoraConfig(...)` 里 ✓ → 指纹**看不见它** ✓
#     → 改成叠 MLP 之后，**旧 ckpt 的指纹仍然匹配** ✗✗
#     → resume 会拿旧结构的 adapter 去配新结构 ✓ → 形状不匹配 / 或静默用错 ✓
#   ✓ 现在暴露成常量 ✓ → 指纹里带上它 ✓ → 结构一变就**拒绝续跑** ✓✓
# =========================================================================== #
LORA_R: int = 16
LORA_ALPHA: int = 32
# ★ 只挂 attention：**4.36M** ✓（现在）— 加上 MLP：**18.5M** ✓（4.2 倍 ✓）
LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj")


class LMWithValueHead(nn.Module):
    """共享主干 + lm_head + value_head。**默认用 LoRA 训练。**

    为什么默认 LoRA（MPS 32GB 上这是最有效的一刀，实测 B=4/T=746/G=160 一步）：

        全参：可训练 494.0M，活张量 **12.61GB**，5.05s
        LoRA：可训练   2.2M，活张量  **2.86GB**，4.01s

    省的**不只是优化器状态** —— 梯度和计算图里保存的张量也一起缩了（所以比
    "只算 Adam 状态"的估计省得多）。顺带还快一点，因为反向的工作量小了。

    LoRA 下的三层结构（很容易搞混）：

        self.model        对外那层 —— `rollout` 里 `model.model.generate(...)` 用它
        self.hf           底层 CausalLM —— `.model` 拿 hidden、`.lm_head` 拿 logits
        self.value_head   始终**全参**训练（它只有 897 个参数）

    梯度检查点在**这个规模上没用**（实测 21.29G vs 21.28G）—— 瓶颈是 `(B,G,V)`
    的 logits 和优化器状态，不是 backbone 激活（hidden 只有 896）。
    """

    def __init__(self, model_path: str, device: str = _DEVICE, lora: bool = True,
                 init_lora: str | None = None, dtype: str = "float32",
                 autocast: str | None = None):
        super().__init__()
        # ★ 加载任何东西**之前**先查路径 ✗ —— 否则 peft 会把"不存在的相对路径"
        #   当成 HuggingFace 仓库名去联网，报出一串连接错误 ✗✓
        #   （实测：`--init-lora runs/sft_15b` 在别的目录跑时，报的是一堆
        #     HEAD https://huggingface.co/runs/sft_15b/... ✗ 完全看不出真因）
        if init_lora is not None and not os.path.isdir(init_lora):
            raise FileNotFoundError(
                f"init_lora={init_lora!r} 不是一个目录 ✗\n"
                f"        解析成绝对路径：{os.path.abspath(init_lora)}\n"
                f"        peft 会把不存在的路径当成 HuggingFace 仓库名 → 去联网下载 ✗\n"
                f"        常见原因：没在项目目录下跑（cd 到项目目录，或用绝对路径 ✓）")
        if model_path and not os.path.isdir(model_path) and LOCAL_FILES_ONLY:
            raise FileNotFoundError(
                f"模型目录不存在：{model_path!r}\n"
                f"        （解析成 {os.path.abspath(model_path)}）\n"
                f"        要么用 --model 指到正确位置，要么设 PPO_ALLOW_DOWNLOAD=1 允许联网下载")
        # 0.5B 太小，fp32 完全放得下（2GB），不需要 bf16。
        # bf16 参数配 AdamW 更新精度差（缺 fp32 master weight），小模型得不偿失。
        # 参数 dtype：默认 fp32（两端一致 ✓）。**不要**给 LoRA 换 bf16 ✗
        # —— 可训练参数按 1e-4 量级更新，bf16 在 0.01 附近的分辨率就是 1e-4 ✗
        pt_dtype = to_pt_dtype(dtype)
        self._amp_dtype = to_pt_dtype(autocast)      # None = 关掉 autocast ✓
        hf = AutoModelForCausalLM.from_pretrained(model_path, dtype=pt_dtype,
                                                  local_files_only=LOCAL_FILES_ONLY)

        if lora:
            # 延迟 import：不用 LoRA 时就不需要装 peft
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            if init_lora:
                # 从 SFT 存下来的 adapter 接着训（`sft.py` 的产物）。
                # 用 PeftModel.from_pretrained 而不是手工 load_state_dict ——
                # 它会**从目录里读回 r / alpha / target_modules**，
                # 不用在这边硬编码一遍（硬编码早晚会和 sft.py 不一致 ✗）。
                hf = PeftModel.from_pretrained(hf, init_lora, is_trainable=True)
            else:
                hf = get_peft_model(hf, LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=LORA_R,
                    lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                    target_modules=list(LORA_TARGETS), bias="none"))
            self.model = hf.to(device)
            self.hf = self.model.base_model.model     # 移动之后再取，别留 CPU 上的旧引用
            for p in self.model.parameters():         # 基座冻结
                p.requires_grad_(False)
            for n, p in self.model.named_parameters():
                if "lora_" in n:                      # 只训适配器
                    p.requires_grad_(True)
        else:
            self.model = hf.to(device)
            self.hf = self.model

        self.value_head = nn.Linear(self.hf.config.hidden_size, 1,
                                    device=device, dtype=pt_dtype)
        # ★ 零初始化 value head —— **不是可选的美化，是共享主干能不能学的前提**
        #
        # 实测（同一批数据，只改这一处初始化）：
        #   默认初始化: V∈[-2.5,16.1], value_loss 18.2, 主干梯度 value项/policy项 = **20.6 倍**
        #   零初始化  : V≡0,           value_loss  0.25, 主干梯度 value项/policy项 = **0 倍**
        #
        # 为什么 20.6 倍要命：`clip_grad_norm_` 用**全局**范数，value 项把范数撑起来后，
        # policy 那一份会被同一个系数压小 20 倍 → 学不动。
        # 为什么零初始化还顺带做了 critic warm-up：W_v = 0 时
        #   ∂v_loss/∂主干 = W_vᵀ·(V − target) = 0
        # 所以第一步反向**完全不往主干送梯度**。
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        # 生成段长度。由 rollout 设置在**重算之前**，让 forward 只对生成段算 logits。
        # None = 全序列（慢但正确）。
        self._is_cuda = device == "cuda" or str(device).startswith("cuda")
        self.gen_len: int | None = None

    def _amp(self):
        """autocast 上下文；不启用时是 no-op ✓（两端通用）。

        `self._is_cuda` 在 `__init__` 里存好 —— 不在 forward 里遍历参数去猜设备 ✗
        """
        if self._amp_dtype is None or not self._is_cuda:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def forward(self, input_ids, token_mask):
        """返回 `(logits, values, None)`。**注意这里有个必须的移位。**

        契约是"**同位置**"的（和 `toy.TinyActorCritic.forward` 一致）：

            logits[t] = 产生 input_ids[t] 的那个分布
            values[t] = V(s_t)，即"吃掉 token t **之前**"的状态价值

        `toy` 就是这么写的 —— 先 `lg, val = step_logits_values(h)`，**然后**才
        `h = advance(h, input_ids[:, t])`。而 HF 的 `h[t]` **已经吃过 token t**，
        所以两个都要右移一位。

        不移位的后果（实测，用"逐个前缀跑"作 ground truth）：
            logits 不移位 → logprob 偏差 **21.0**，ratio 起手偏差 **1.000** → PPO 第一步就废
            values 不移位 → V(s_{t+1}) 冒充 V(s_t)，GAE 整体错一位，**而这个不报错**
        """
        with self._amp():          # CUDA 上矩阵乘走 bf16 ✓ 参数仍是 fp32 ✓
            out = self.hf.model(input_ids=input_ids, attention_mask=token_mask)
        h = out.last_hidden_state.float()                          # (B,T,H) h[t] 已吃过 t

        # ★ 右移一位：h_prev[t] = h[t-1] = "吃过 token 0..t-1" 的状态
        #   位置 0 补零（它永远是 padding 或 prompt 位置，被 mask 挡掉，无意义）
        h_prev = torch.zeros_like(h)
        h_prev[:, 1:] = h[:, :-1]

        # ★★★ 参数 dtype 必须和输入对齐 ✗✓
        #   正常路径有 autocast 兜底 ✓ 但 **ref 模型是 bf16 且 autocast=None** ✗
        #   → `value_head(h_prev)` 就变成 Float × BFloat16 ✗（实测 RuntimeError ✓）
        #   → 这里把输入临时 cast 到**参数**的 dtype ✓（输出仍然 .float() ✓ 下游不变 ✓✓）
        _h = h_prev
        _wd = self.value_head.weight.dtype
        if _h.dtype != _wd:
            _h = _h.to(dtype=_wd)

        with self._amp():
            values = self.value_head(_h).squeeze(-1)                # (B,T) **全序列**，GAE 要用

            h_lm = _h[:, -self.gen_len:] if self.gen_len else _h
            _lm_d = self.hf.lm_head.weight.dtype
            if h_lm.dtype != _lm_d:
                h_lm = h_lm.to(dtype=_lm_d)
            logits = self.hf.lm_head(h_lm)
        # ★ 输出强制回 fp32：下游（GAE / ratio / KL / loss / 熵）全部沿用原来的
        #   fp32 路径 → **不引入任何新的 dtype 混用风险** ✓
        #   （dtype 不匹配在 MPS 上会直接杀进程且没有回溯，我们踩过一次 ✗）
        return logits.float(), values.float(), None   # 第三个值 ppo_update 不用


def make_optimizer(model: "LMWithValueHead", lr: float,
                   head_lr: float | None = None) -> torch.optim.Optimizer:
    """共享主干的两个头配**分组学习率**。

    这是"分离 actor/critic"最大的好处之一，而它**不需要真的拆开模型**：
    主干是预训练好的、只需要微调（小 lr）；value head 是随机初始化的、
    要追上一个不断变强的策略（大 lr，通常 10~100 倍）。

    共享主干本来会**剥夺**你这个自由度（一个 lr 管所有参数）—— 分组把它还回来了。
    """
    # 只把 requires_grad=True 的交给 Adam：
    #   全参模式 → 全部；LoRA 模式 → 2.2M 个适配器参数（冻结的基座不进去，
    #   省掉它们的优化器状态）。
    trunk = [p for p in model.model.parameters() if p.requires_grad]
    head = list(model.value_head.parameters())
    return torch.optim.AdamW([
        {"params": trunk, "lr": lr},
        {"params": head, "lr": head_lr if head_lr is not None else lr * 10},
    ])


def load_tokenizer(model_path: str, local_files_only: bool = True):
    """加载 tokenizer，并把**契约要求的两个设置**做掉。

    1. `padding_side = "left"` —— 全框架的硬性要求：mask 必须在每行**末尾**连续，
       右 padding 会让 mask 中间出现空洞（`batch.check()` 会拦）。
    2. `pad_token` 兜底 —— Qwen2.5 自带 `<|endoftext|>`（≠ eos），所以不会有
       "pad token is same as eos token" 那个警告；但换个模型就可能没有，兜一下。
    """
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# =========================================================================== #
# rollout —— ★ 留给你实现
# =========================================================================== #
def keep_process_reward(has_eos_row: bool, gen_len_row: int, batch_width: int) -> float:
    """★ 截断门：这一行的**过程项**要不要保留（1.0 = 保留）

    契约：**只准依赖行内事实 `has_eos_row`** ✓✓
    绝对**不准**依赖 `gen_len_row` / `batch_width` ✗ —— 那两个是**批内相对量** ✓

    为什么这条契约值钱 ✗：原实现写成 `int(gl) >= G`（G = 这一批的实际宽度 ✗）→
      · 批里没有截断行时 G = max(gen_len) → **自然结束、又恰好最长的那行被无故扣掉过程项** ✗
      · 更严重：**同一道题的奖励随它和谁拼在一批而变** ✗✗
        → 这解释了为什么改 --batch / --rollout-chunks 会大幅改变结果 ✓
          （不只是数据变了，**奖励函数本身变了** ✓✓）

    那两个不许用的参数**故意留在签名里** ✓ —— 它们是这条契约的载体 ✓
    并由 `_case_c` 用不同批次组合钉死它们对返回值没有影响 ✓✓
    """
    _ = gen_len_row, batch_width          # 明确声明不用 ✓ 见 docstring ✓
    return 1.0 if has_eos_row else 0.0


def rollout(
    model: "LMWithValueHead",
    tokenizer,
    problems: list[dict],
    examples: list[dict] | None,
    cfg: PPOConfig,
    seed: int | None = None,
    vllm_backend=None,          # ★ 可选：vLLM 后端 ✓（None = 原来的 HF 路径 ✓ 行为不变 ✓）
) -> RolloutBatch:
    """采一批：GSM8K 题目 → prompt → 生成 → 组装 `RolloutBatch`。

    契约和 `toy.rollout` **完全一样**（7 个 `(B,T)` 字段 + `meta`），
    所以写完之后 `update.ppo_update` 直接能用。

    五步
    ----
    1. **建 prompt**：`build_messages(question, examples)` → `tokenizer.apply_chat_template(
       ..., add_generation_prompt=True, return_tensors="pt")`。
       ⚠️ 它返回的是 **BatchEncoding（dict）**，不是 tensor —— 要取 `["input_ids"]`
       和 `["attention_mask"]`（transformers 5.x，我实测过）。
       逐条编码，然后左 pad 到本批最长（`batch.pad_left`）。
    2. **生成**：`with torch.no_grad(): model.model.generate(...)`
       * ⚠️ **采样参数必须和重算 logprob 的分布一致**：
         `do_sample=True, temperature=1.0, top_p=None`，不要 `repetition_penalty`。
         而且 `model.model.generation_config` 里**可能已经塞了** temperature/top_p，
         `generate` 会合并它们 —— 显式传参覆盖，并注意那条
         `The following generation flags are not valid` 警告。
       * 传 `attention_mask`（左 padding 时必须，否则会看到
         `The attention mask is not set and cannot be inferred...` 警告）。
       * ⚠️ **`generate` 不接受 `generator=`**（transformers 5.x 会报
         `model_kwargs are not used by the model: ['generator']`）。
         要复现就用 `torch.manual_seed(seed)`（包在 `torch.random.fork_rng()` 里
         不污染全局 RNG）—— 所以下面签名的参数是 `seed: int | None`，不是 Generator。
       * `gen_ids = out[:, Lp:]`（左 padding 下，prompt 永远在前 Lp 列）。
    3. **拼装**：⚠️ 其实 **`generate` 返回的 `out` 已经是完整序列**（prompt + 生成），
       所以 `input_ids = out` 就行 —— 再 `cat` 一遍是**恒等操作**
       （实测 `out == cat([prompt_ids, out[:, Lp:]])` 为 True）。
       **真正需要拼的是 `attention_mask`**：
       `full_mask = cat([attention_mask, ones(B,G)], dim=1)`
       —— 重算 forward 要用它，否则左 padding 那几列会被 attend 到。
    4. **重算** logprobs / values：一次 forward 过完整序列（**要梯度吗？不要** ——
       这里只需要记录常数快照，`torch.no_grad()` 下跑，且契约第 7 条要求 detached）。
       `logprobs = _logprobs_new(logits, input_ids, logits)` ——
       **不能**直接 `token_logprobs(logits, input_ids)`，因为 `forward` 可能只返回生成段。
       ⚠️ 这一步就是 `assert_ratio_starts_at_one` 要检查的东西：只要它和生成时用的
       分布一致，ratio 起手就该是 1。
    5. **奖励**：`reward_fn(text, gold)` → **只在最后一个有效位置**写 0/1；
       `dones` 同理（**每行恰好 1 个 True**，契约第 6 条）。
       真实 LLM **没有逐步代价**，别照抄 toy 的 `-0.01`。

    最后别忘了：`model.gen_len = 生成长度` —— 它让 `forward` 只对生成段算 logits
    （实测快 1.45x，且步时方差从 2.5 倍降到几乎为 0）。不设也能跑，只是慢。

    验收（`python3 model.py` 会逐条检查）
    ------------------------------------
    * `batch.check()` 全过（7 条契约）
    * `assert_ratio_starts_at_one` **不响** ← 采样分布对上了
    * 奖励**不全为 0** ← 否则 PPO 一点梯度都没有
    """

    _t_prep0 = time.perf_counter()          # ★ 计时：准备（渲染/分词/左pad/拼批）
    messages, golds = [], []
    for problem in problems:
        message = build_messages(problem["question"], examples)
        messages.append(message)
        golds.append(problem["gold"])

    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, return_tensors="pt")
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    # ★ 把输入搬到**模型所在**的 device。tokenizer 产出的一定在 CPU，
    #   而 `generate` 不会替你搬。不搬的话 MPS 上会报
    #   `Placeholder storage has not been allocated on MPS device!`
    #   —— 和"设备不匹配"完全看不出关系，很容易查错方向。
    #   以**模型**的 device 为准（不是 cfg.device，两者可能不一致）。
    dev = next(model.parameters()).device
    input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)

    Lp = input_ids.shape[1]
    with torch.random.fork_rng(devices=[]): 
        torch.manual_seed(cfg.seed if seed is None else seed)
        with torch.no_grad():
            # 采样参数：**四个都要显式覆盖** generation_config 里的
            # temperature=0.7 / top_p=0.8 / top_k=20 / repetition_penalty=1.1。
            # cfg.temperature == 0 → 贪心（评估用：固定题集 + 贪心才可比）。
            sample_kw = (
                dict(do_sample=True, temperature=cfg.temperature,
                     top_p=None, top_k=None, repetition_penalty=1.0)
                if cfg.temperature > 0
                # ★★ greedy 分支也**必须**显式覆盖 repetition_penalty ✗
                #    Qwen2.5 的 generation_config.json 里写着 repetition_penalty=1.1 ✓
                #    而 RepetitionPenaltyLogitsProcessor 只看 `!= 1.0` 就加入 ✓
                #    且在 `_sample` 里**无条件**先跑 logits_processor ✗
                #    （只有 temperature/top_k/top_p 被 `if do_sample:` 挡住 ✓）
                #    → 于是 `do_sample=False` 的"贪心"其实是**对罚过的 logits 取 argmax** ✗✓
                #    实测（0.5B）同一 prompt：不传 → 'The sum of two plus two is four.'
                #                            传 1.0 → 'The answer is 4.'   ← 完全不同 ✗
                #    后果：**所有"贪心"评估数字都不是 argmax** ✓ 而 best-checkpoint 也按它选 ✗
                else dict(do_sample=False, repetition_penalty=1.0)
            )
            _t_prep = time.perf_counter() - _t_prep0   # ★ 准备阶段耗时
            _t_gen0 = time.perf_counter()        # ★ 计时：生成
            if vllm_backend is None:
                out = model.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,     # 左 padding 必须传
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **sample_kw,
                )
            else:
                # ★★ 显式标记 ✓✓ —— 接线打错位置时（我就犯过 ✗）日志上一眼可见 ✗✓
                _VLOG = globals().setdefault("_VLLM_CALLS", [0])
                _VLOG[0] += 1
                if _VLOG[0] <= 3:
                    print(f"  [rollout] ★ 本次生成走 **vLLM** ✓（第 {_VLOG[0]} 次）")
                # ★★ vLLM 只替"生成"这一步 ✓✓
                #    → 返回 token ids ✓ → 下面**照原样**拼回 out（input_ids/mask/gen_len）
                #    → `logprobs_old` 仍然由**训练器重算** ✓ → ratio 起手仍为 1 ✓✓
                #    ★ 注意：prompt 是**左 padding** 的 ✓ → 要按 mask 取出真实 token ✓
                outs = []
                for row in range(input_ids.shape[0]):
                    real = attention_mask[row].bool()
                    pid = input_ids[row][real].tolist()      # ★ 去掉左 pad ✓
                    outs.append(pid)
                # ★★★ 必须把 `seed` 传下去 ✗✗
                #   原来漏了 ✗ → generate 用默认 0 ✓ → 每次调用都是 "0 + 行号" ✗
                #   → **每一轮的同一条题拿到同一个种子** ✗ → 8 次采样完全相同 ✗✗
                #   实测后果（probe_passk k=8）：808 题 0/8 · 511 题 8/8 · 中间全 0 ✗
                #     pass@1 == pass@8 == 0.387 ✗✓（分布是二值的 ✓）
                gen_ids = vllm_backend.generate(
                    outs,
                    temperature=float(cfg.temperature),
                    max_tokens=int(cfg.max_new_tokens),
                    seed=int(seed) if seed is not None else 0,
                )
                # ★★★ 拼成和 HF **完全一致**的布局 ✓✓ —— 这里的偏移错一次就全盘皆错 ✗✗
                #
                #   HF 那条路用的是**左 padding** ✓：
                #       prompt 占 [Lp−len(pid), Lp)     ← 右对齐 ✓ 前面补 pad ✓
                #       生成段占 [Lp, Lp+len(g))        ← 从 Lp 开始 ✓
                #   而 ③ 拼装和 ⑤ 奖励都是**按 Lp 读**的 ✓：
                #       `input_ids[:, Lp:]`（生成段 ✓）`Lp + gen_len − 1`（奖励位 ✓）
                #
                #   ★ 我之前写成了"从 0 开始放 prompt"✗（左对齐 ✗）→ 只有**最长的那一行**
                #     恰好 len(pid)==Lp 时才碰巧对 ✓ → 单行探针永远对 ✓✓ 而多行不等长全错 ✗✗
                #     实测后果：reward 和 mask 错位 → GRPO 的组内全 0 → **grad_norm = 0** ✗✓
                G = max(1, max(len(g) for g in gen_ids))
                Lp = input_ids.shape[1]                 # ★ padded prompt 宽度 ✓
                dev = input_ids.device
                full = torch.full((len(outs), Lp + G), tokenizer.pad_token_id,
                                  dtype=torch.long, device=dev)
                am = torch.zeros_like(full)
                for r, (pid, g) in enumerate(zip(outs, gen_ids)):
                    n_p = len(pid)
                    assert n_p <= Lp, f"prompt {n_p} > Lp {Lp} ✗"
                    full[r, Lp - n_p:Lp] = torch.tensor(pid, device=dev)   # ★ 右对齐 ✓
                    am[r, Lp - n_p:Lp] = 1
                    if g:
                        full[r, Lp:Lp + len(g)] = torch.tensor(g, device=dev)  # ★ 从 Lp ✓
                        am[r, Lp:Lp + len(g)] = 1

                # ★ HF 的 `generate` 这里返回**裸张量** ✓（下面 ③ 里就是 `input_ids = out` ✓）
                #   → 所以必须给张量 ✗ 不能包对象 ✗✓（包了会在 `input_ids.shape` 炸 ✗）
                out = full

    _t_gen = time.perf_counter() - _t_gen0     # ★ 生成耗时
    _t_re0 = time.perf_counter()               # ★ 计时：重算前向
    # ---- ③ 拼装：`out` 已经是完整序列；真正要拼的是 attention_mask ----
    input_ids = out
    B, T = input_ids.shape
    G = T - Lp

    # 每行的**有效生成长度**：EOS 之后 generate 会填 pad_token_id
    # ⚠️ 不能直接 argmax：全 False 的行会返回 0（长度被算成 1）
    eos_mask = input_ids[:, Lp:] == tokenizer.eos_token_id            # (B,G)
    has_eos = eos_mask.any(dim=1)                                     # (B,)
    gen_len = torch.where(has_eos,
                          eos_mask.float().argmax(dim=1) + 1,         # +1：EOS 也是真实动作
                          torch.full((B,), G, device=input_ids.device))
    gen_mask = torch.arange(G, device=input_ids.device)[None, :] < gen_len[:, None]   # (B,G)
    attention = torch.cat([attention_mask, gen_mask.long()], dim=1)                   # (B,T)

    actions = torch.zeros_like(input_ids)
    actions[:, Lp:] = input_ids[:, Lp:]          # 契约：actions[mask] == input_ids[mask]

    # ★★ 重算之前，先把 gen_len 设成**本批**的 G。
    #    原来它靠上一批留下的值，而上一批的生成段长度可能不同（例如全行提前 EOS 停在 12），
    #    forward 就会只回 12 列，`_logprobs_new` 把 logprob 铺到"最后 12 列" →
    #    生成段前面那一大截全是 0 → ppo_update 里 ratio 断言炸，
    #    而且**自动诊断会误报成"mask 传错了"**（因为超差位置的 logprobs_old 恰好都是 0）。
    # ★★ 三个"静默为零"的诊断 ✓✓ —— grad_norm=0 时**立刻**能看出是哪个环节 ✗
    if not globals().get("_SHAPE_STATS", 0):
        globals()["_SHAPE_STATS"] = 1
        _m = int(mask.sum()) if "mask" in dir() else -1
        print(f"  [shape] batch: T={T} Lp={Lp} G={G} · gen_len 均值 "
              f"{float(gen_len.float().mean()):.1f} · gen_len 最小 {int(gen_len.min())} "
              f"· has_eos {int(has_eos.sum())}/{B}")
        print(f"  [shape] gen_mask 非零 {int(gen_mask.sum())}/{gen_mask.numel()} · "
              f"prompt_mask 非零 {int(attention_mask.sum())}/{attention_mask.numel()}")
    model.gen_len = G

    # ---- ④ 重算 logprob / values：no_grad（契约第 7 条要求不带图）----
    # ⚠️ 传给 forward 的是 **attention mask**（真实 token = prompt 真实段 + 生成真实段），
    #    不是生成 mask —— 这是玩具里那个「两个 mask」的坑的同一个版本。
    with torch.no_grad():
        logits, values, _ = model(input_ids, attention)
        # ⚠️ 必须用 _logprobs_new：因为 forward 可能只返回**生成段**的 logits
        #    （model.gen_len 被上一次 rollout 设过），直接用 token_logprobs 会形状不匹配。
        #    它不是 softmax+gather（那是概率），而是真正的 logprob。
        logprobs = _logprobs_new(logits, input_ids, logits)

    _t_re = time.perf_counter() - _t_re0       # ★ 重算耗时
    # ---- ⑤ 奖励 / dones：每行最后一个**有效生成位置** ----
    texts = [tokenizer.decode(g, skip_special_tokens=True) for g in input_ids[:, Lp:]]
    last = Lp + gen_len - 1                                          # (B,)
    rows = torch.arange(B, device=input_ids.device)

    # ★ 结果项 / 过程项**分开算**（日志要分开记 ✗ 合在一起就没法判断模型好坏 ✓）
    #
    # 过程奖励的「截断门」：没吐 EOS（写满 max_new_tokens 被打断）的轨迹，
    # **过程项清零** ✗ 不加这一条会出事：截断的输出照样能拿满过程项 →
    # "一直写下去"能刷分 → EOS 概率被压到 0 → **永不停止**
    #（实测：EOS 率 98.4% → 0%，准确率 31% → 7.8% ✗）
    _t_rw0 = time.perf_counter()        # ★ 计时：奖励（正则 + 算式解析 ✓ 头号嫌疑 ✓）
    base_l, proc_l = [], []
    for i, (t, prob, gl) in enumerate(zip(texts, problems, gen_len.tolist())):
        b_, p_ = reward_parts(t, prob["gold"], cfg.process_coef, mode=cfg.reward_mode)
        base_l.append(b_)
        # ★★ 截断门必须用「**这一行有没有吐 EOS**」✗ 而不是「`gl >= G`」✓
        #
        #    原来写的是 `int(gl) >= G`，而 `G = T - Lp` 是**这一批**实际生成的宽度
        #    （= 批内最长的那一行）✗ —— 于是判据变成了**批内相对量**：
        #      · 批里没有截断行时（很常见），G = max(gen_len)
        #        → **自然结束、又恰好最长的那一行被无故扣掉过程项** ✗
        #        （偏偏它常是写得最完整、`<<>>` 标注最全的那一行 ✗ 反向激励）
        #      · 更严重：**同一道题的奖励取决于它和谁拼在一批** ✗✗
        #        → 这正解释了为什么改 `--batch` / `--rollout-chunks` 会大幅改变结果 ✓
        #          （不只是数据变了，**奖励函数本身变了** ✓✓）
        #    `has_eos` 上面已经算好了 ✓ 它是**行内局部**的事实 ✓ 与批次组成无关 ✓✓
        proc_l.append(keep_process_reward(bool(has_eos[i]), int(gl), int(G)) * p_)
    dev = input_ids.device
    rewards = torch.zeros_like(input_ids, dtype=torch.float)
    rewards[rows, last] = torch.tensor([a + b for a, b in zip(base_l, proc_l)],
                                       dtype=torch.float, device=dev)
    _t_rw = time.perf_counter() - _t_rw0   # ★ 奖励耗时

    reward_base = torch.zeros_like(input_ids, dtype=torch.float)
    reward_base[rows, last] = torch.tensor(base_l, dtype=torch.float, device=dev)

    dones = torch.zeros_like(input_ids, dtype=torch.bool)
    dones[rows, last] = True                                         # 每行恰好 1 个

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[:, Lp:] = gen_mask                                          # 行内连续，且不含 EOS 后的 pad


    return RolloutBatch(
        input_ids=input_ids,
        actions=actions,
        logprobs=logprobs,
        values=values,
        rewards=rewards,
        dones=dones,
        mask=mask,
        meta={"prompt_mask": torch.cat([attention_mask.bool(),   # (B,T)：只含真实 prompt，
                                            torch.zeros_like(gen_mask)], dim=1),
              "reward_base": reward_base,          # 结果项（不含过程项）✓ 日志用
              # ★ 生成段宽度 G 存在 **batch** 上 ✗→✓（int ✓ batch.to() 有 Tensor 守卫，安全 ✓）
              #   原来它只存在 model.gen_len 上 —— 那是**单个可变属性** ✓
              #   多块 rollout 之后它只剩最后一块的值 → 别的块 forward 切错尾部，且不报错 ✗
              "gen_width": int(G),
              # ★ 拆分计时：vLLM 只能加速"生成"那部分 ✗
              #   量出占比才知道接 vLLM 的收益上界 ✓
              "t_gen": float(_t_gen), "t_recompute": float(_t_re),
              "t_prep": float(_t_prep), "t_reward": float(_t_rw)},
    )


# =========================================================================== #
# 自测
# =========================================================================== #
# 默认模型目录。**服务器上不用改代码** —— 用环境变量或 --model 覆盖 ✓
#     PPO_MODEL=/root/models/Qwen2.5-1.5B-Instruct python3 main.py ...
MODEL_PATH = os.environ.get(
    "PPO_MODEL", "/data/models/Qwen2.5-0.5B-Instruct")  # ★ 可用 env 覆盖 ✓
# 服务器上如果要让 transformers 自己下载（本地没这个目录），设 PPO_ALLOW_DOWNLOAD=1
LOCAL_FILES_ONLY = os.environ.get("PPO_ALLOW_DOWNLOAD", "") != "1"


def _case_a() -> None:
    """模型本身：形状 / dtype / 零初始化。**不需要生成，秒级。**"""
    dev_expected = "mps" if torch.backends.mps.is_available() else "cpu"
    m = LMWithValueHead(MODEL_PATH)
    dt = next(m.model.parameters()).dtype
    if dt != m.value_head.weight.dtype:
        raise AssertionError(f"dtype 不一致：主干 {dt} vs value_head {m.value_head.weight.dtype}"
                             f"（MPS 上会抛 Metal 断言，进程当场死）")
    dev = m.value_head.weight.device
    B, T = 2, 40
    ids = torch.randint(0, 1000, (B, T)).to(dev)
    am = torch.ones(B, T, dtype=torch.long).to(dev)
    logits, values, third = m(ids, am)
    for name, got, want in [("logits", tuple(logits.shape), (B, T, m.model.config.vocab_size)),
                            ("values", tuple(values.shape), (B, T))]:
        if got != want:
            raise AssertionError(f"{name} 形状 {got} != {want}")
    if not torch.isfinite(values).all():
        raise AssertionError("values 里有 NaN/Inf")
    if values.abs().max() > 1e-6:
        raise AssertionError(f"value_head 没零初始化：V 范围 [{values.min():.3f}, {values.max():.3f}]"
                             "（默认初始化会让 value 项的主干梯度是 policy 项的 20 倍，"
                             "全局裁剪把策略梯度压掉）")
    print(f"    dtype {dt} | device {dev} | logits {tuple(logits.shape)} | "
          f"values {tuple(values.shape)} | V≡0 ✅")

    opt = make_optimizer(m, lr=1e-6, head_lr=1e-4)
    lrs = [g["lr"] for g in opt.param_groups]
    if lrs != [1e-6, 1e-4]:
        raise AssertionError(f"分组 lr 不对：{lrs}")
    print(f"    分组优化器：主干 {lrs[0]:.0e} / value_head {lrs[1]:.0e} ✅")


def _case_greedy_penalty() -> None:
    """★ 贪心必须与 generation_config 的 repetition_penalty **无关** ✓✓

    Qwen2.5 的 generation_config.json 写着 repetition_penalty=1.1 ✓
    而 `RepetitionPenaltyLogitsProcessor` 只看 `!= 1.0` 就加入、且在 `_sample` 里
    **无条件**先跑（只有 temperature/top_k/top_p 被 `if do_sample:` 挡住）✗
    → `do_sample=False` 的"贪心"= **对罚过的 logits 取 argmax** ✗
    → 后果：**所有"贪心"评估数字都不是 argmax** ✓ 而 best-checkpoint 也按它选 ✗

    判据：把 generation_config 的 repetition_penalty 改成 **1.5** ✓
          输出必须**一字不变** ✓✓（旧代码会变 ✓ 所以这条有牙齿 ✓）
    """
    dev = PPOConfig.auto_device()
    path = os.environ.get("PPO_MODEL", MODEL_PATH)
    tk = load_tokenizer(path)
    mm = LMWithValueHead(path, device=dev, lora=True, init_lora=None)
    c0 = PPOConfig.preset("llm").replace(max_new_tokens=24, kl_coef=0.0, device=dev,
                                         temperature=0.0, rollout_prompts=1)
    from data import load_gsm8k          # model.py 只 import 了函数名 ✓ 没有 D ✓
    pr = [load_gsm8k("test")[0]]

    def greedy_text():
        b_ = rollout(mm, tk, pr, None, c0, seed=0)
        Lp_ = int(gen_start(b_.mask)[0])
        return tk.decode(b_.input_ids[0, Lp_:], skip_special_tokens=True)

    before = greedy_text()
    old = mm.model.generation_config.repetition_penalty
    try:
        mm.model.generation_config.repetition_penalty = 1.5
        after = greedy_text()
    finally:
        mm.model.generation_config.repetition_penalty = old
    if before != after:
        raise AssertionError(
            "贪心受了 generation_config 的 repetition_penalty 影响 ✗\n"
            f"        rep={old} → {before!r}\n"
            f"        rep=1.5  → {after!r}\n"
            "        → 必须在 `do_sample=False` 分支里显式传 repetition_penalty=1.0 ✓")
    print(f"    贪心对 generation_config 的 penalty 免疫 ✅（rep={old} 与 1.5 输出一致）")


def _case_c() -> None:
    """★ 截断门**必须与批次组成无关** ✓✓ —— 这正是那个真实 bug 的判据 ✗

    原实现 `int(gl) >= G`（G = 这一批的实际宽度 ✗）在下面的输入上会给出
    **不同**的答案 ✓ → 同一道题的过程奖励取决于它和谁拼在一批 ✗✗
        (gen_len=50, batch_width=50) → 旧实现清零 ✗   ← 可怜的最长那行
        (gen_len=50, batch_width=80) → 旧实现保留 ✓
    新实现只吃 has_eos ✓ → 四种组合必须**全等** ✓✓
    """
    got = {keep_process_reward(True, gl, bw)
           for gl, bw in ((50, 50), (50, 80), (10, 50), (100, 200))}
    if got != {1.0}:
        raise AssertionError(
            "截断门依赖了批内量 ✗ 四种批次组成给出 " + str(got) +
            " → 同一道题的奖励会随批次变化 ✗✗（必须只吃 has_eos ✓）")
    print("    有 EOS：四种批次组成 (50/50, 50/80, 10/50, 100/200) → 全为 1.0 ✅")

    got2 = {keep_process_reward(False, gl, bw) for gl, bw in ((50, 50), (50, 80), (100, 100))}
    if got2 != {0.0}:
        raise AssertionError("没吐 EOS 的行没被清零 ✗ " + str(got2))
    print("    无 EOS：一律清零（防一直写下去刷分）✅")


def _case_b() -> None:
    """rollout 的契约 + 三条验收（需要你先实现 rollout）。"""
    from batch import token_logprobs
    from loss import assert_ratio_starts_at_one
    import data as D

    cfg = PPOConfig.preset("llm").replace(rollout_prompts=2, max_new_tokens=64,
                                          kl_coef=0.0, device="cpu")
    train, test = D.load_gsm8k("train"), D.load_gsm8k("test")
    examples = D.fewshot_examples(train, 4)
    m = LMWithValueHead(MODEL_PATH, device=cfg.device)
    tok = load_tokenizer(MODEL_PATH)

    # ★ 调**两次**：第一次之后 `model.gen_len` 就被设上了，第二次 forward 只返回生成段。
    #   （这个洞让一个真 bug 溜过去过一次：rollout 里直接用 token_logprobs 会形状不匹配。）
    _ = rollout(m, tok, test[:cfg.rollout_prompts], examples, cfg, seed=cfg.seed)
    b = rollout(m, tok, test[:cfg.rollout_prompts], examples, cfg, seed=cfg.seed)

    b.check()                                            # ← 7 条契约
    print(f"    batch.check() ✅  B={b.batch_size} T={b.shape[1]} "
          f"平均生成长度 {float(b.gen_len().float().mean()):.1f}")

    pm = b.meta.get("prompt_mask")
    if pm is None:
        raise AssertionError("meta 里没有 prompt_mask（契约要求：标记真实 prompt 位置）")
    overlap = int((pm & b.mask).sum())
    if overlap:
        raise AssertionError(f"prompt_mask 和 mask 有 {overlap} 个位置重叠 —— "
                             f"prompt_mask 必须**不含**生成段（也不含左 padding）")
    print(f"    prompt_mask 干净（和生成段零重叠）✅")

    # ★ 跨**不同生成长度**再各跑一次：这抓的是"重算时用了上一批的 gen_len"。
    #   上一批短、这一批长的话，生成段前面一大截 logprob 会是 0（字段没填上），
    #   而 7 条通用契约全都会通过。
    for G in (32, 96):
        bb = rollout(m, tok, test[:cfg.rollout_prompts], examples,
                     cfg.replace(max_new_tokens=G), seed=cfg.seed)
        frac = float(((bb.logprobs == 0) & bb.mask).sum()) / float(bb.mask.sum())
        if frac > 0.05:
            raise AssertionError(
                f"max_new_tokens={G} 时，mask 内有 {frac:.1%} 的 logprobs 是 0\n"
                f"        → 字段没被填上：重算 forward 用的生成段长度和本批不一致\n"
                f"        （`model.gen_len` 必须在重算**之前**就设成本批的 G）"
            )
        bb.check()
        print(f"    max_new_tokens={G:>3} 也能跑（mask 内 logprobs 为 0 的占比 {frac:.2%}）✅")

    # ★ 领域专用检查：EOS 之后的填充位**不能**进 mask。
    #   （框架的 check() 抓不到这个 —— 那 7 条是通用的，不认识 eos/pad 是什么 token。
    #     而 generate 会在提前结束的行后面填 pad_token_id，用 ones 当 mask 就会把这些
    #     位置当成真实动作送进 loss，配上一堆 -14~-20 的垃圾 logprob。）
    from batch import last_valid_index
    _last = last_valid_index(b.mask)
    _rows = torch.arange(b.batch_size)
    _is_last = torch.zeros_like(b.mask)
    _is_last[_rows, _last] = True                      # 最后一列(通常是 EOS)是合法的
    _n_pad = int(((b.input_ids == tok.pad_token_id) & b.mask & ~_is_last).sum())
    if _n_pad:
        raise AssertionError(
            f"mask 里有 {_n_pad} 个 pad_token 位置 —— EOS 之后的填充位不该被训练\n"
            f"        → full_mask 的生成段要用 gen_mask（arange(G) < gen_len），不能用 ones\n"
            f"        （generate 会在提前结束的行后面填 pad_token_id；用 ones 的话这些位置\n"
            f"         会以 -14~-20 的垃圾 logprob 混进 loss，而 check() 那 7 条全都会通过）"
        )
    print("    生成段里没有混进 EOS 后的 pad ✅")

    from update import _logprobs_new          # 兼容「全量 logits」和「只算生成段」两种返回
    with torch.no_grad():
        tm = pm | b.mask
        logits, _, _ = m(b.input_ids, tm)
        lp = _logprobs_new(logits, b.input_ids, b.logprobs)
    assert_ratio_starts_at_one(lp, b.logprobs, b.mask)   # ← 采样分布一致性
    print("    ratio 起手为 1（采样分布 == 重算分布）✅")

    # 截断门（这里 base 模型不写 <<>>，只验不崩；无关性由 _case_c 钉住 ✓）
    assert keep_process_reward(True, 50, 50) == 1.0
    assert keep_process_reward(False, 50, 50) == 0.0
    print("    截断门：有 EOS 保留 / 无 EOS 清零 ✅")


    # ⚠️ 不能写 `b.rewards[b.mask].view(B, -1)` —— 各行生成长度不同，view 会炸
    #    （这是「假设等长」的老毛病；用 last_valid_index 就通用）
    lv = last_valid_index(b.mask)
    last = b.rewards[torch.arange(b.batch_size), lv]
    n_ok = int((last > 0.5).sum())
    print(f"    本批奖励 {last.tolist()}（对了 {n_ok}/{b.batch_size}）"
          + ("  ⚠️ 全 0 → 这一步没有梯度（换一批题或加大 B）" if n_ok == 0 else "  ✅"))


if __name__ == "__main__":
    cases = [("用例 A：模型形状 / dtype / 零初始化", _case_a),
             ("用例 C：截断门与批次组成无关（纯函数 ✓ 秒级）", _case_c),
             ("用例 D：贪心与 generation_config 的 penalty 无关", _case_greedy_penalty),
             ("用例 B：rollout 契约 + ratio 起手为 1 + 奖励不全 0", _case_b)]
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
            print(f"    💥 未预期的异常 {type(e).__name__}: {str(e)[:200]}")
    print(f"\n{'=' * 60}")
    if passed == len(cases):
        print(f"🎉 {len(cases)} 个用例全部通过 —— 可以写 main.py 了")
    else:
        print(f"进度：{passed}/{len(cases)} 通过（NotImplementedError 是正常的）")
