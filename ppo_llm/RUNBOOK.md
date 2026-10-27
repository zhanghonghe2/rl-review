# 启动命令手册（Runbook）

> 所有 RL 命令的入口、每个参数的作用、以及我们实际用的值。
> 配套：`REPORT.md`（结果）· `RECORD.md`（历史与基线表）· `PPO_LLM_PRM.md`（PPO+PRM 原理）。
> **每个参数都注明了"我们用的值"** —— 直接照抄即可复现。

---

## 0. 三种算法怎么选

| 想做 | 脚本 | 奖励 | 有无 critic | 我们用它得到的结论 |
|---|---|---|---|---|
| **GRPO** | `main.py --adv-norm grpo` | 结果 0/1 | ✗（组内均值当基线）| 1.5B/MATH-500：贪心 0.424 → **0.560** |
| **PPO**（带 critic）| `main.py --adv-norm batch` | 结果 0/1 | ✓（value head + GAE）| 小模型上收益不稳定，后来主要用 GRPO |
| **PRM**（PPO + 过程奖励）| `prm_train.py` | 0/1 **+ 逐段大模型判分** | ✓ | 1.5B-Instruct：pass@1 0.342 → **0.473**（5σ）|
| **online-DPO** | `dpo_train.py` | 成对偏好（同题内自采）| ✗（自带 ref）| 机制对，但天花板 = pass@8 |

**前置条件**（所有命令通用）：
```bash
export DEEPSEEK_API_KEY=sk-...     # PRM 的判分器用；DPO/GRPO 不需要
cd ppo_llm
python3 selfcheck.py               # 35 个文件：调用/签名/参数名一致性
```

**★ 铁律**：任何正式 run 之前，先用 **2 步 + 1 次 eval** 跑通（约 6 分钟）：

```bash
ITERS=2 EVAL_EVERY=1 bash bs.sh
```
> 我们的静默 bug（少传 `--test-file`、并发默认成 1、`len(scores)==5`）**全部**是这 6 分钟能抓到的。

---

## 1. 冷启动 SFT（可选，但强烈建议）

```bash
CUDA_VISIBLE_DEVICES=0 python3 sft.py \
    --model /data/models/Qwen2.5-1.5B-Instruct \
    --data data/gsm8k_sft.jsonl --limit 1500 \
    --epochs 2 --lr 2e-4 --batch 4 --grad-accum 4 --max-len 768 \
    --system default --lr-schedule cosine \
    --eval-size 128 --test-file data/gsm8k_test.jsonl --reward-mode strict \
    --vllm --eval-chunk 64 \
    --out runs/sft_lora
```

| 参数 | 作用 | 我们的值 |
|---|---|---|
| `--data` | 用哪份 jsonl 当 SFT 数据 | 官方 1500 或拒绝采样输出 |
| `--limit` | 最多用多少条（**默认 None = 全用**）| 1500 |
| `--system` | system 消息：`default`=GSM8K 的"必须写 `#### <number>`" | `default`（GSM8K）· **`none`**（PRM/MATH，别和 `\boxed{}` 打架）|
| `--lr-schedule` | `cosine`（默认）/ `const` | `cosine`（恒定会在末尾过训练，吃过 −15 的亏）|
| `--grad-accum` | 梯度累积；有效批量 = `batch × grad-accum` | 4（有效 16）|
| `--overfit N` | 自检：只在 N 条上过拟合 | 8（`loss → 0.0000` 才算管线可信）|
| `--eval-only` | 只评估不训练 | 配合 `--no-adapter` 评**基座** |

---

## 2. GRPO（我们跑出第一个干净正结果的命令）

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --model /data/models/Qwen2.5-1.5B-Instruct \
    --device cuda:1 \
    --pool-file data/gsm8k_train.jsonl --pool-skip 1500 \
    --n-shot 0 --test-file data/math500.jsonl \
    --iters 60 \
    --batch 16 --rollout-chunks 1 \
    --group-size 8 --adv-norm grpo \
    --inner-epochs 4 --num-minibatches 4 \
    --lr 3e-5 --clip-eps 0.1 --grad-clip 0.5 \
    --adaptive-kl --kl-target 0.1 \
    --max-new-tokens 1024 \
    --eval-every 15 --eval-size 500 --eval-samples 2 \
    --rollout-backend vllm --vllm-dtype bfloat16 \
    --vllm-gpu-util 0.85 --vllm-kv-gb 6 --vllm-eval-chunk 64 \
    --tag grpo_1p5b
```

| 参数 | 作用 | 我们的值 / 备注 |
|---|---|---|
| `--adv-norm grpo` | ★ 组内标准化（同题 G 条互为基线）→ **不需要 critic** | `grpo` |
| `--group-size 8` | 同一道题采几条 | 8（`--batch` 要能被它整除）|
| `--batch` | 每块采几道题 | 16（**有效批量 = batch × rollout-chunks**）|
| `--rollout-chunks` | 每步采几块 | 1 → 有效批量 16 |
| `--inner-epochs 4` | 同一批数据复用几轮 | 4（GRPO 常用；PPO 我们用 2）|
| `--adaptive-kl --kl-target 0.1` | KL 系数自适应，把每-token KL 稳在 0.1 | 0.1（最好那轮实测 0.57 ✗ 见 RECORD）|
| `--n-shot 0` | few-shot 示例数 | **SFT 之后必须 0**（和 sft.py 一致，否则 prompt 不一致）|
| `--pool-skip 1500` | 跳过池子前 N 条 | 1500（前 1500 给了 SFT，**防泄漏**）|
| `--eval-samples 2` | 评估时额外跑几轮 T=1 | 2（**必须 ≥1**：只看贪心会得出反的结论）|
| `--test-file` | ★ 自定义测试集 | `data/math500.jsonl`（**漏了它 = 一次 eval 都不跑** ✗）|

---

## 3. PPO（带 critic / value head）

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --model /data/models/Qwen2.5-1.5B-Instruct \
    --device cuda:1 \
    --pool-file data/gsm8k_train.jsonl --pool-skip 1500 --n-shot 0 \
    --test-file data/math500.jsonl \
    --iters 45 --batch 8 --rollout-chunks 4 \
    --adv-norm batch --inner-epochs 2 \
    --value-coef 0.5 --process-coef 0 --kl-coef 0.02 \
    --lr 3e-5 --clip-eps 0.1 --grad-clip 0.5 \
    --max-new-tokens 1024 \
    --eval-every 15 --eval-size 500 --eval-samples 2 \
    --rollout-backend vllm --vllm-dtype bfloat16 \
    --tag ppo_1p5b
```

| 参数 | 作用 | 我们的值 |
|---|---|---|
| `--adv-norm batch` | ★ 全批所有有效 token 上标准化 → 量级 ≈1，**保留 token 级结构** | `batch`（**别用 `per_sequence`**：每条序列内标准化会让"答对/答错"形状一样 → 零信号 ✗）|
| `--value-coef 0.5` | value 损失权重 | 0.5 |
| `--process-coef` | 过程奖励系数（`main.py` 里的简易版）| **0**（纯结果奖励；真正的 PRM 走第 5 节）|
| `--kl-coef 0.02` | 固定 KL 系数（不用自适应时）| 0.02 |
| `--num-minibatches` | 更新分几块做 forward/backward；**显存 ∝ 有效批量/nmb** | 4（数学等价，只是省显存）|
| `--head-lr-mult` | value head 的 lr 倍数 | 1.0（调大容易过冲）|

---

## 4. online-DPO

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 dpo_train.py \
    --model /data/models/Qwen2.5-1.5B-Instruct \
    --device cuda:1 \
    --test-file data/math500.jsonl --reward-mode math \
    --problems 32 --group-size 8 \
    --iters 20 --pairs-per-step 16 \
    --beta 0.1 --lr 5e-6 \
    --max-new-tokens 1024 --max-len 2048 \
    --grad-checkpoint --micro 2 --rollout-chunk 8 \
    --vllm --vllm-kv-gb 6 \
    --out runs/dpo_online
```

| 参数 | 作用 | 我们的值 / 备注 |
|---|---|---|
| `--test-file` | ★★ **训练题池**（不是评测集 ✗）| 不给则默认 `D.load_gsm8k("test")` → **等于在 test 上训练** ✗✗ 务必显式给 |
| `--problems` × `--group-size` | 每步采 `problems × group_size` 条，同题内组 pair | 32 × 8 |
| `--pairs-per-step` | 攒够多少对做一次更新 | 16 |
| `--beta 0.1` | DPO 温度（KL 约束强度）| 0.1 |
| `--lr 5e-6` | DPO 的 lr 要比 PPO 小一个量级 | 5e-6 |
| `--grad-checkpoint` | ★ 每个 pair 要 4 次前向（policy/ref × chosen/rejected）→ 必须开 | 开 |
| `--micro` | 每几对做一次前向（控显存）| 2 |

**它的判据**（我们就是靠这个判断"对没对"）：
```
起手不变量：policy==ref → margin ≈ 0、loss ≈ log2 = 0.6931 ✓
margin 轨迹：−0.144 → +0.020 → +0.268   ← 确实在学 ✓
但准确率不涨 → 天花板 = pass@8（pair 全来自当前策略，只能重分配概率 ✗）
```

---

## 5. PRM：PPO + 过程奖励（当前主力）

```bash
MODEL=/data/models/Qwen2.5-1.5B-Instruct \
POOL=data/math_train_hard.jsonl \
OUT=runs/prm_1p5b MNT=1024 \
ITERS=45 EVAL_EVERY=15 \
bash bs.sh
```

展开就是（`bs.sh` 的内容）：

```bash
nohup python3 prm_train.py \
    --model "$PPO_MODEL" --device cuda:1 \
    --pool-file "$POOL" --test-file data/math500.jsonl \
    --problems 8 --group-size 4 \
    --iters 45 --ckpt-every 5 \
    --judge api --judge-model deepseek-chat \
    --segment-mode paragraph --reward-shape paper --beta 0.15 \
    --adv-norm batch --inner-epochs 2 --rows-per-update 1 \
    --lr 3e-5 --kl-coef 0.02 --clip-eps 0.1 --grad-clip 0.5 \
    --max-new-tokens 1024 --max-len 2048 --rollout-chunk 16 \
    --eval-every 15 --eval-size 500 --eval-chunk 64 \
    --vllm --vllm-gpu-util 0.85 --vllm-kv-gb 6 \
    --out "$OUT" > "$OUT.log" 2> "$OUT.err" &
```

### 5.1 奖励相关

| 参数 | 作用 | 我们的值 |
|---|---|---|
| `--reward-shape paper` | ★★ `paper` = `rv(±1) 定生死` + `β/N·段分`（字典序）；`segment` = **只用段分** ✗（被刷过）| `paper` |
| `--beta 0.15` | 段分**总权重上限**（必须 <1，否则能翻正负）| 0.15（≈ rv 的 13%；要真测 PRM 需 ≥0.5）|
| `--segment-mode paragraph` | 切块方式：`paragraph`（自然段落，适合不写 5 段的通用模型）/ `sections`（5 段式）| `paragraph` |
| `--min-chars 80` | 块的最小字数；**调大 → 块更少 → 判分次数直线下降** | 80（≈10 块/条）|
| `--judge api` | `mock`（默认，不花钱，只冒烟）/ `api`（真跑必须）| `api` |
| `--judge-model` | 判分模型 | `deepseek-chat` |
| `--judge-workers 0` | 判分并发；**0 = 自动等于 batch** | 0（32 条 → 32 路）|
| `--judge-cache` | 判分缓存文件 | `runs/prm_judge_cache.jsonl`（**判分提示词改了要删它** ✗）|

### 5.2 PPO / 显存相关

| 参数 | 作用 | 我们的值 |
|---|---|---|
| `--problems` × `--group-size` | 有效批量（题 × 每条几样本）| 8 × 4 = **32** |
| `--rows-per-update 1` | ★★ 每次 backward 的**行数**；显存 ∝ 行数 | **1**（32 行分 32 次；OOM 的根因就是这里）|
| `--rollout-chunk 16` | 采样时每次喂几行；**重算 logits 的显存 ∝ 行数** | 16（32 就爆 ✗）|
| `--adv-norm batch` | 与 GRPO 同一口径（不然 A/B 不可比）| `batch` |
| `--inner-epochs 2` | 同一批数据复用几轮 | 2 |
| `--vllm-kv-gb 6` | ★ KV cache 的**显式**大小（GB）| 6（不给会占满剩余显存 ✗）|
| `--vllm-gpu-util 0.85` | ✅ 只是**预算上限**，不是真分配 | 0.85 |

### 5.3 断点/评测

| 参数 | 作用 | 我们的值 |
|---|---|---|
| `--ckpt-every 5` | 每 N 步原子写断点（adapter + 优化器 + 配置指纹）| 5 |
| `--resume` | 从 `{out}_ckpt` 续跑；**指纹不一致直接拒绝** | 需要时给 |
| `--eval-every 15` | 每 N 步评一次（0 = 不评）| 15 |
| `--eval-size 500` | 评多少题（128 题噪声 ±3.9% ✗）| 500 |
| `--eval-chunk 64` | eval 时每次喂 vLLM 多少题（**已直采，不受 logits 限制**）| 64 → 500 题 ≈ **2.5 分钟** |
| `--eval-greedy` | 额外跑一遍贪心（默认关：多花一倍时间且两边都不动）| 关（要贪心用 `sft.py --eval-only`）|

---

## 6. 工具命令（流程的一部分）

```bash
# ① 筛训练池：按 base 的 pass@8 ∈ [1,7] 保留
python3 filter_pool.py --pool data/math_train_hard.jsonl --out data/pool_1to7.jsonl \
    --model /data/models/Qwen2.5-Math-1.5B-Instruct \
    --k 8 --lo 1 --hi 7 --chunk 32 --limit 2000 --max-new-tokens 1024 --vllm

# ② pass@k 探针（★ 别忘了 --adapter，否则测的是基座 ✗）
python3 probe_passk.py --adapter runs/prm_1p5b_best \
    --model /data/models/Qwen2.5-1.5B-Instruct --vllm-kv-gb 8 \
    --test-file data/math500.jsonl --problems 500 --k 8 --chunk 128 \
    --max-new-tokens 1024 --reward-mode math --vllm --vllm-sleep-level 0
#   --greedy 才跑贪心 ✓；--shard 0/2 / 1/2 可两卡分片 ✓

# ③ 分段+判分的小样本验证（不花 GPU）
python3 prm_probe.py --problems 17            # 看到块长/块数/段全对率
python3 prm_probe.py --problems 17 --no-judge # 只看切块

# ④ 自检
python3 selfcheck.py                 # 调用/签名/参数名
python3 prm_train.py --self-test     # 奖励矩阵 / GAE / 分段
```

---

## 7. 跨命令通用参数（速查）

| 参数 | 作用 | 常见值 |
|---|---|---|
| `--device cuda:1` | ★ 训练器放哪张卡；**vLLM 永远跟 `current_device`=0** | `cuda:1`（vLLM 在 0，训练在 1）|
| `--model` | 模型目录 | `/data/models/Qwen2.5-{1.5B,3B}-Instruct`、`Qwen2.5-Math-1.5B-Instruct` |
| `--init-lora` | 从已有 adapter 继续训（SFT → RL）| `runs/sft_lora` |
| `--no-lora` | 关 LoRA 改全参（显存 +10 GB）| 没用过 |
| `--max-new-tokens` | 每题最多生成多少 token | 1024（通用模型）/ 2048（Math）|
| `--max-len` | 上下文上限（prompt + 生成）| 2048 |
| `--temperature` | 训练采样温度 | 1.0 |
| `--lr` | 主干 lr（value_head 走 10×）| 3e-5（PPO/GRPO）· 5e-6（DPO）· 2e-4（SFT）|
| `--clip-eps` | PPO 裁剪 ε | 0.1（比论文 0.2 紧）|
| `--grad-clip` | 梯度裁剪 | 0.5 |
| `--seed` | 种子（题目顺序 + 采样）| 0 |
| `--tag` / `--out` | 日志/断点的名字 | 每次实验换，**别复用**（会覆盖 `_best` ✗）|

---

## 8. 口径与验收标准（不看这些，数字没有意义）

```
判据   ：**严格** = \boxed{} / #### 抽取后符号等价（sympy，相对容差 1e-6）
题数   ：**500 题**（MATH-500 全集）→ 单次 eval 的 SE ≈ **2.0 点**
指标   ：贪心 + **T=1**（PPO 优化的是期望回报，只看贪心会得出反的结论 ✗）
可分辨 ：1 eval ≈ 5.8 点 · 3 种子 ≈ 3.3 点 → **E < R 的实验不要做**
```

| 阶段 | 验收标准 |
|---|---|
| 启动 | `可训练 18.5M` ✓ · `题池 N` ✓ · `test 500` ✓ · API key ✓ |
| 第 1 步 | `奖励 0.3~0.6` · `全对 20~60%` · `grad_norm 0.1~1` |
| 每步 | 生成 ≤45s · 判分 ≤10s · 更新 ≤30s（batch 32）|
| eval | **500 题 ≤ 3 分钟** |
| 断点 | 每 5 步落盘；崩溃最多丢 5 步 |

**★ `ratio ≈ 1.000x` 是正常的**：日志记的是"第 1 轮 step 之前"，那时只更新过 1 次。
判断"有没有在学"看 **eval**，不要看 `ratio` / `loss`（`loss ≈ −mean(A)` 恒 ≈0）。

---

## 9. 最容易踩的五个坑

| # | 坑 | 症状 | 防法 |
|---|---|---|---|
| 1 | 漏 `--test-file` | 45 步**一次 eval 都没跑** ✗ | 启动自检会打印 `test 500` ✓ |
| 2 | `--judge-workers` 默认 0 被 `max(1,...)` 吃掉 | 320 次判分串行 → 十几分钟 | 日志里看 `块级并发 W 路` |
| 3 | `--rollout-chunk` / `--rows-per-update` 设大 | OOM（32 行 logits ≈ 40 GiB ✗）| 保持 16 / 1 |
| 4 | 复用 `--out` | 覆盖上一轮的 `_best` ✗ | 每次换 tag |
| 5 | 忘了 `sync()`（只在自写脚本里）| 生成永远用旧权重 → 静默 off-policy | 用现成脚本 ✓ |
