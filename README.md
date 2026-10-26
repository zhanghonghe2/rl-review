# RL Review — 小模型数学推理的 PPO / GRPO / PRM 实验

从零实现的 PPO / GRPO / 过程奖励（PRM）栈，用来在 2×L40S 上做小模型（1.5B / 3B）的数学推理 RL。

## 先看这两个文件

| 文件 | 内容 |
|---|---|
| **[`ppo_llm/REPORT.md`](ppo_llm/REPORT.md)** | ★ 技术报告：主结果、测量学、工程加速、**静默 bug 分类学**、**失败的实验** |
| **[`ppo_llm/RECORD.md`](ppo_llm/RECORD.md)** | 全过程记录 + 顶部**基线表**（模型 × 测试集 × 贪心/pass@1/pass@8）|
| [`ppo_llm/PPO_LLM_PRM.md`](ppo_llm/PPO_LLM_PRM.md) | ★ **大模型 PPO + 过程奖励的实现说明**（公式、超参、诊断、坑）|
| [`ppo_llm/PRINCIPLES.md`](ppo_llm/PRINCIPLES.md) | 从事故里长出来的工作准则（含成本/收益判据）|

## 一句话结果

同一套 PPO + PRM 配方：

- **Qwen2.5-1.5B-Instruct（MATH-500）**：pass@1 **0.342 → 0.473**，贪心 **0.424 → 0.538**，
  `8/8` 题数 **21 → 97**、`0/8` **158 → 124**。
- **Qwen2.5-Math-1.5B**（基座已 0.664）：只涨 T=1（+4.6），**贪心不动** →
  **效应大小由基座离天花板多远决定**，不是由奖励形式决定。

## 目录

```
ppo_llm/          主实现（PPO/GRPO/PRM、评测、数据工具）
  prm_train.py      过程奖励版 PPO（块级并行判分、断点续跑、内置快速 eval）
  prm_reward.py     段落切分 / 字符→token 映射 / 判分器 / 奖励矩阵
  main.py           GRPO / PPO 主循环
  filter_pool.py    按 pass@8 筛训练池
  probe_passk.py    pass@k 探针（支持多卡分片）
ppo_lab/          早期 PPO 玩具实现（带手算验证）
dodge_bandit/     最小可验证的 PPO 实验
PPO/              早期异步 PPO 尝试
```

## 运行（需要 GPU + vLLM）

```bash
export DEEPSEEK_API_KEY=sk-...      # PRM 的判分器用；不设则只能用 --judge mock
cd ppo_llm
ITERS=2 EVAL_EVERY=1 bash bs.sh     # ★ 先跑 2 步验证（约 6 分钟）
bash bs.sh                          # 正式 45 步
```

数据与模型权重不在仓库里（见 `.gitignore`）：
`ppo_llm/data/`、`ppo_llm/math_data/`、`ppo_llm/runs/`、`PPO/Qwen*/`。

## 注意

- 所有 `RNG` 都可以用 `--seed` 固定；**但 `prm_train` 的 torch seed 是后加的**，
  早于该修复的实验不可复现（见 REPORT §5）。
- 指标口径：**严格**（`\boxed{}` 符号等价）· 500 题 · `pass@1` = T=1 单样本。
  不同口径（宽松/严格）差约 30 点，不要混比。
