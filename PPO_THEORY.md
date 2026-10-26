# PPO 公式理论：从零推导

> 这份文档从最基础的问题定义开始，**每一步都给出推导**（不写"可以证明"）✓
> 并且**每一步都连回 `ppo_llm/` 的代码和实测数字** ✓
> 目的：看完之后你能自己判断"这个实现哪里可能错" ✓

---

## 目录

1. [把问题写清楚：我们的 MDP 长什么样](#1)
2. [策略梯度：从目标函数到可估的梯度](#2)
3. [为什么要 baseline，以及 advantage](#3)
4. [同策略的困境与重要性采样](#4)
5. [TRPO 的思路，PPO 的简化](#5)
6. [PPO-clip 为什么是 `min`：四种情形的精确推导](#6)
7. [完整损失函数：四项与它们的符号](#7)
8. [优势估计：GAE 的完整推导](#8)
9. [★ 为什么单轮任务必须 `γ=1`](#9)
10. [完整算法：伪代码 ↔ 我们的代码](#10)
11. [去掉 critic 的变体：GRPO / RLOO](#11)
12. [实现里最容易错的地方（每条都是我们踩过的）](#12)
13. [公式 → 我们的代码/数字 对照表](#13)
14. [常见理解型问答](#14)

---

<a name="1"></a>
## 1. 把问题写清楚：我们的 MDP 长什么样

### 1.1 一般形式

强化学习用 **MDP**（马尔可夫决策过程）描述：$(S, A, P, r, \gamma)$

- $s_t \in S$：时刻 $t$ 的状态
- $a_t \in A$：动作
- $P(s_{t+1}\mid s_t, a_t)$：环境转移（**我们无法控制** ✓）
- $r_t \in \mathbb{R}$：奖励
- $\gamma \in [0,1]$：折扣因子

策略 $\pi_\theta(a\mid s)$ 是我们要学的东西。目标是

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\Big[\sum_{t=0}^{T-1}\gamma^t r_t\Big]
\;=\; \mathbb{E}_{\tau \sim \pi_\theta}\big[G_0\big]
$$

其中 $\tau = (s_0,a_0,s_1,a_1,\dots)$ 是一条轨迹（episode）。

### 1.2 ★ 我们的具体情形（GSM8K 上的语言模型 RL）

这是**最关键的一步理解** ✗：把语言生成映射到 MDP 上。

| MDP 概念 | 我们的对应物 | 代码位置 |
|---|---|---|
| 状态 $s_t$ | **prompt + 已经生成的 $t$ 个 token** | `input_ids[:, :t]` |
| 动作 $a_t$ | **下一个 token**（词表 151936 选 1）| `actions[:, t]` |
| 策略 $\pi_\theta(a\mid s)$ | `softmax(logits[t])` | `lm_head` 输出 |
| 环境转移 $P$ | **确定性**：拼上这个 token | 无（$P$ 是恒等）|
| 一整个 episode | **一整条回答**（推理 + `#### N`）| 一行 batch |
| 奖励 $r_t$ | **只在最后一步**给（其余 0）| `rewards[rows, last]` |
| 折扣 $\gamma$ | **必须 = 1**（见 §9）| `cfg.gamma` |

**两个后果** ✓✓：

**(a) 环境是确定的** ✓ → 所以转移概率 $P$ 在策略梯度里**根本不会出现** ✓✓
（这就是为什么语言模型的 RL 只有策略项，没有模型项 ✓）

**(b) 奖励只在轨迹末尾** ✓ → 这是一条**稀疏**奖励 ✓✓
→ 这一个事实导致了后面所有的工程问题：价值函数、GAE、baseline、方差…… ✓

### 1.3 我们的奖励具体是什么

```python
reward = 答对(0/1) + process_coef × (算式标注算对的比例)     # data.py::reward_parts
```

- **结果项** $r^{\text{out}} \in \{0,1\}$：最终答案对不对（严格判据要求显式 `#### N`）
- **过程项** $r^{\text{proc}} \in [0,1]$：GSM8K 自带的 `<<算式=结果>>` 标注里算对的比例
- 取 $\lambda = $ `process_coef`，总奖励 $R = r^{\text{out}} + \lambda r^{\text{proc}}$

⚠️ 这个"混合奖励"有个实测教训 ✗：**它会掩盖真实进步** ✓
（某个 run 训练奖励 0.7286 最高 ✓ 而 eval 只有 0.297 最差 ✗ —— 因为它靠过程项刷分 ✓）
→ 所以后来把日志**拆成** `reward_base` / `reward_proc` ✓✓

---

<a name="2"></a>
## 2. 策略梯度：从目标函数到可估的梯度

### 2.1 目标函数的梯度长什么样

直接对 $J(\theta)$ 求导很难 ✗，因为**数据分布本身依赖 $\theta$** ✓（换了 $\theta$ 就换了采样的轨迹 ✓）

**第一步：写出期望的显式形式。**

$$
J(\theta) = \sum_{\tau} P_\theta(\tau)\, R(\tau),
\qquad
P_\theta(\tau) = \rho(s_0)\prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t)\, P(s_{t+1}\mid s_t,a_t)
$$

**第二步：用 score function 恒等式。**

对任意 $p_\theta(x) > 0$：

$$
\nabla_\theta p_\theta(x) = p_\theta(x)\,\nabla_\theta \log p_\theta(x)
$$

> **推导**：$\log p_\theta$ 对 $\theta$ 求导得 $\frac{\nabla_\theta p_\theta}{p_\theta}$，两边乘 $p_\theta$ 即得 ✓

**第三步：把梯度推过求和号。**

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \sum_\tau \nabla_\theta P_\theta(\tau)\, R(\tau) \\
&= \sum_\tau P_\theta(\tau)\, \nabla_\theta \log P_\theta(\tau)\, R(\tau) \qquad \text{(用第二步)}\\
&= \mathbb{E}_{\tau\sim\pi_\theta}\big[\, \nabla_\theta \log P_\theta(\tau)\, R(\tau) \,\big]
\end{aligned}
$$

**第四步：展开 $\log P_\theta(\tau)$。**

$$
\log P_\theta(\tau) = \log \rho(s_0) + \sum_{t} \log \pi_\theta(a_t\mid s_t) + \sum_t \log P(s_{t+1}\mid s_t,a_t)
$$

**求梯度时后两项消失** ✓✓：
- $\log\rho(s_0)$ 与 $\theta$ 无关 ✓
- **$\log P$ 与环境有关、与 $\theta$ 无关** ✓✓ ← 这是关键的一步

于是

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\Big(\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big)\Big(\sum_{t}\gamma^t r_t\Big)\Big]
$$

### 2.2 拆成逐步的形式（因果性）

利用"$t$ 时刻的动作不能影响 $t$ 之前的奖励"（因果性 / 无未来泄漏）：

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot \Psi_t\Big]\;}
$$

其中 $\Psi_t$ 可以是：

| $\Psi_t$ | 名字 | 偏差 | 方差 |
|---|---|---|---|
| $\sum_{t'\ge t}\gamma^{t'-t} r_{t'}$ | 回报 $G_t$（REINFORCE ✓）| **无偏** ✓ | **很大** ✗ |
| $G_t - b(s_t)$ | 带 baseline | 无偏 ✓（§3）| 更小 ✓ |
| $Q^\pi(s_t,a_t)$ | 动作价值 | 无偏 ✓ | 小 ✓ |
| $Q^\pi - V^\pi = A^\pi$ | **advantage** ✓✓ | 无偏 ✓ | **最小** ✓ |

**这就是 PPO 的出发点** ✓✓：用 advantage $A_t$ 代替回报 ✓

### 2.3 最朴素的实现：REINFORCE

```
1. 用 π_θ 采一批轨迹
2. 算每条轨迹的回报 G_t
3. θ ← θ + α · Σ_t ∇log π_θ(a_t|s_t) · G_t
```

⚠️ **问题** ✗：方差极大 ✓
- $G_t$ 的尺度随任务变化 ✓（我们这里 $G_t \in [0, 1.3]$ ✓ 还算好 ✓）
- 如果奖励全是正的 ✓ 那 $\nabla\log\pi$ 会被"一致地往一个方向推" ✗ → 不知道哪个动作**相对更好** ✓
- **实测** ✗：我们第一次 RL 尝试（纯结果奖励 + 无 baseline 之外的结构）就出现**长度塌缩** ✓
  `131 → 85 tokens` ✓ —— 因为偶然对的**短**轨迹被强化了 ✓✓

**→ 必须有 baseline** ✓ 见下一节 ✓

---

<a name="3"></a>
## 3. 为什么要 baseline，以及 advantage

### 3.1 定理：任何只依赖状态的 baseline 都不引入偏差 ✓✓

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\big[\nabla_\theta\log\pi_\theta(a\mid s)\cdot b(s)\big] = 0
\quad\text{对任意 } b(s)
$$

**推导** ✓：

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\, b(s)\big]
&= b(s)\sum_a \pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)\\
&= b(s)\sum_a \pi_\theta(a\mid s)\,\frac{\nabla_\theta \pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}\\
&= b(s)\sum_a \nabla_\theta \pi_\theta(a\mid s)\\
&= b(s)\,\nabla_\theta \underbrace{\sum_a \pi_\theta(a\mid s)}_{=\,1}\\
&= b(s)\cdot \nabla_\theta 1 = 0 \qquad\blacksquare
\end{aligned}
$$

**三个推论** ✓：

1. **减去任何 $b(s)$ 都不改变梯度的期望** ✓✓ → 只降方差 ✓
2. **$b$ 不能依赖动作 $a$** ✗✓ —— 否则上面第一步的 $b(s)$ 提不到求和号外 ✓ 就会有偏 ✓✓
   （这正是 GRPO 除以组内 std 会引入偏置的原因 ✓ 见 §11 ✓）
3. **最好的 $b(s)$ 是 $V^\pi(s)$** ✓ → 此时 $\Psi_t = Q - V = A$ ✓

### 3.2 Advantage 的定义与性质

$$
A^\pi(s,a) \;=\; Q^\pi(s,a) - V^\pi(s)
$$

| 符号 | 定义 |
|---|---|
| $V^\pi(s) = \mathbb{E}_{\pi}[G_t\mid s_t=s]$ | 状态价值（这个状态**平均**能拿多少）|
| $Q^\pi(s,a) = \mathbb{E}_{\pi}[G_t\mid s_t=s,a_t=a]$ | 动作价值 |
| $A^\pi(s,a)$ | 这个动作比"平均水平"好多少 ✓ |

**性质** ✓：
- $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)] = \mathbb{E}[Q] - V = V - V = 0$ ✓
- $A>0$ → 好动作 → 提高它的概率 ✓
- $A<0$ → 坏动作 → 降低它的概率 ✓
- **正负都有** ✓ → 不会像"奖励全正"那样把所有动作一起推 ✗✓

### 3.3 落到我们的场景（单轮、末奖励）

$$
V^\pi(s_t) = \mathbb{E}\big[R \mid \text{前 } t \text{ 个 token}\big]
$$

因为我们只在末尾给奖励 $R$ ✓ 且 $\gamma=1$ ✓ → 回报就是从当前前缀出发能拿到的期望奖励 ✓

**代码对应** ✓：`model.py` 里 `value_head` 输出 `values[t]` ✓ → 就是 $V(s_t)$ ✓
⚠️ **对齐约定** ✗：`values[t]` 必须来自"**还没吃掉第 $t$ 个 token**"的隐状态 ✓✓
（我们的 `forward` 用 `h_prev[:, 1:] = h[:, :-1]` 实现右移一位 ✓ —— 不移位时实测 logprob 偏差 **21.0** ✓）

---

<a name="4"></a>
## 4. 同策略的困境与重要性采样

### 4.1 问题

策略梯度要求数据来自**当前** $\pi_\theta$ ✓
但 PPO 想对**同一批数据**做多次更新 ✓（因为采样很贵 ✓✓ —— 我们实测一次 rollout 6~31 秒 ✓ 而一次更新只有 4 秒 ✓）

一旦更新了一次 ✓ 数据就是**旧策略** $\pi_{\theta_{\text{old}}}$ 采的了 ✗ → 不在分布上 ✓

### 4.2 重要性采样

对任意函数 $f$：

$$
\mathbb{E}_{x\sim p}[f(x)] = \int p(x) f(x)\,dx = \int q(x)\frac{p(x)}{q(x)}f(x)\,dx = \mathbb{E}_{x\sim q}\Big[\frac{p(x)}{q(x)}f(x)\Big]
$$

（只要 $q(x) > 0$ 在 $p(x)>0$ 处成立 ✓）

应用到轨迹上：

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_{\theta_{\text{old}}}}\Big[\underbrace{\frac{P_\theta(\tau)}{P_{\theta_{\text{old}}}(\tau)}}_{\text{比值}}R(\tau)\Big]
$$

### 4.3 ★ 比值可以拆成逐 token 的乘积

$$
\frac{P_\theta(\tau)}{P_{\theta_{\text{old}}}(\tau)}
= \frac{\rho(s_0)\prod_t \pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)}
        {\rho(s_0)\prod_t \pi_{\theta_{\text{old}}}(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)}
= \prod_{t=0}^{T-1}\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

**环境项完全抵消** ✓✓（因为 $P$ 不含 $\theta$ ✓）

定义**逐 token 比值**：

$$
r_t(\theta) \;=\; \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
= \exp\Big(\log\pi_\theta(a_t\mid s_t) - \log\pi_{\theta_{\text{old}}}(a_t\mid s_t)\Big)
$$

**代码对应** ✓✓（`loss.py`）：

```python
log_ratio = logprobs_new - logprobs_old
ratio     = log_ratio.exp()
```

### 4.4 ★ 最重要的一条不变量：$\theta=\theta_{\text{old}}$ 时 $r_t \equiv 1$

$$
r_t(\theta_{\text{old}}) = \frac{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} = 1
$$

**为什么它是最重要的一条** ✓✓：
- 它保证"重算的 logprob"和"采样时存的 logprob"走的是**同一条计算路径** ✓✓
- 如果它们不一致 ✗ → 重要性采样的前提就破了 ✗ → **裁剪目标失效** ✗ → 训练悄悄学歪 ✓ 而且**不报错** ✗✓

所以我们在 `ppo_update` 的**第一次**内层更新**之前**断言它 ✓✓：

```python
assert_ratio_starts_at_one(logprobs_new, batch.logprobs, batch.mask)   # 容差 1e-3
```

**实测踩坑** ✗：`model.gen_len` 陈旧时 ratio 起始偏差 **0.9811** ✓（第 6 步炸掉 ✓）
**而且这条断言是后面所有改造的地基** ✓✓：批量累积、GRPO、多卡，全都建立在"$r$ 起手为 1"上 ✓

### 4.5 一个诚实的说明（近似在哪）✗

上面严格推出的是**逐轨迹**的比值 $\prod_t r_t$ ✓
而 PPO 的标准实现（包括我们 ✓）用的是：

$$
L = \mathbb{E}\Big[\frac{1}{\sum_t m_t}\sum_t m_t \cdot \min\big(r_t A_t,\ \text{clip}(r_t)A_t\big)\Big]
$$

即**逐 token 的比值** ✓ 而不是乘积 ✓✓ —— 这是一个**近似** ✓（把序列级 IS 变成 token 级 ✓）
它在实践中工作良好 ✓ 且方差小得多 ✓✓
但它带来一个副作用：**token 级平均会给长序列更多权重** ✗（§11.3 详述 ✓）

---

<a name="5"></a>
## 5. TRPO 的思路，PPO 的简化

### 5.1 代理目标（surrogate）

用旧数据估新策略的性能：

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_{t}\Big[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}A_t\Big]
= \mathbb{E}_t\big[r_t(\theta)\,A_t\big]
$$

（CPI = Conservative Policy Iteration ✓）

**注意** ✓：这个式子只在 $\theta \approx \theta_{\text{old}}$ 时才可信 ✓
一旦 $\pi_\theta$ 偏离太远 ✗ → 比值方差爆炸 ✗ → 代理目标和真实目标脱节 ✓✓

### 5.2 TRPO 的约束形式

$$
\max_\theta\ \mathbb{E}_t\big[r_t(\theta)A_t\big]
\quad\text{s.t.}\quad
\mathbb{E}_t\big[\mathrm{KL}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s_t)\,\|\,\pi_\theta(\cdot\mid s_t)\big)\big] \le \delta
$$

**它为什么对** ✓：有理论保证"在 KL 球内，代理目标的提升 $\Rightarrow$ 真实目标的提升" ✓（带一个 $\frac{2\epsilon\gamma}{(1-\gamma)^2}$ 之类的系数 ✓）
**它的代价** ✗：每步要解一个带约束的优化 ✓（共轭梯度 + 线搜索 ✓）→ 实现复杂 ✓ 我们不会手写 ✓

### 5.3 PPO 的两种简化

| 版本 | 做法 |
|---|---|
| **PPO-penalty** | 把约束变成惩罚：$L = \mathbb{E}[r_t A_t] - \beta\,\mathrm{KL}$ ✓ ← **我们用的就是这一项** ✓✓ |
| **PPO-clip** ✓✓ | 直接**裁剪比值** ✓ 不需要二阶信息 ✓ ← 我们用的也是这个 ✓ |

$\beta$ 怎么取 ✗ → 就是"自适应 KL 系数" ✓（trl / InstructGPT 的做法 ✓ 我们已实现 ✓）

---

<a name="6"></a>
## 6. PPO-clip 为什么是 `min`：四种情形的精确推导

### 6.1 目标形式

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\Big(r_t(\theta)A_t,\ \mathrm{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)A_t\Big)\Big]
$$

取 $\epsilon = 0.2$ ✓（我们的 `cfg.clip_eps` ✓）

**为什么要 `min` 而不是直接 `clip`** ✗✓ —— 这是最容易误解的一处 ✓
`clip(r, 1-ε, 1+ε)·A` 单独用是**错**的 ✗（它会丢掉"远离坏动作"的梯度 ✓）

### 6.2 四种情形逐个算（用具体数字 ✓）

取 $\epsilon = 0.2$ → 裁剪区间 $[0.8, 1.2]$ ✓

**情形 1：$r = 3.0$（远超上界），$A = +1$（好动作）**

$$
rA = 3.0,\qquad \mathrm{clip}(r)A = 1.2
$$
$$
\min(3.0,\ 1.2) = \boxed{1.2}\quad\text{→ 裁掉了 ✓}
$$

**直觉** ✓：好动作的收益被**封顶** ✓ → 不允许"靠把概率抬到离谱来多拿分" ✓✓

**情形 2：$r = 3.0$，$A = -1$（坏动作）**

$$
rA = -3.0,\qquad \mathrm{clip}(r)A = -1.2
$$
$$
\min(-3.0,\ -1.2) = \boxed{-3.0}\quad\text{→ 没裁 ✓}
$$

**直觉** ✓：坏动作**仍然被完整惩罚** ✓ → 鼓励策略远离它 ✓✓
（如果用了 `clip(r)·A` 就会变成 $-1.2$ ✗ → 惩罚被削弱了 ✗）

**情形 3：$r = 0.5$（远低于下界），$A = +1$**

$$
rA = 0.5,\qquad \mathrm{clip}(r)A = 0.8
$$
$$
\min(0.5,\ 0.8) = \boxed{0.5}\quad\text{→ 没裁 ✓}
$$

**直觉** ✓：一个本来该提高概率的动作 ✓ 现在概率还很低 ✓ → **要继续把它拉上来** ✓✓

**情形 4：$r = 0.5$，$A = -1$**

$$
rA = -0.5,\qquad \mathrm{clip}(r)A = -0.8
$$
$$
\min(-0.5,\ -0.8) = \boxed{-0.8}\quad\text{→ 裁掉了 ✓}
$$

**直觉** ✓：坏动作的概率已经压得够低了 ✓ → **不再给额外的奖励** ✓✓
（否则策略会为了"刷低概率"而过度压低 ✓ 那会毁掉探索 ✓）

### 6.3 规律总结

$$
\boxed{\text{被裁掉的只有两种情形：}\ (r>1+\epsilon,\ A>0)\ \text{和}\ (r<1-\epsilon,\ A<0)}
$$

**一句话** ✓：**"好动作被抬太高"和"坏动作被砍太狠"会被裁 ✓；反过来不会** ✓✓
→ PPO 限制的是**单次更新的幅度** ✓ 但不阻止策略**远离坏动作** ✓✓

### 6.4 ★ 这四个数字就是我们的测试用例

`loss.py::_case_b()` 用**完全一样**的四组数字钉住实现 ✓✓：

| ratio | A | 期望的 loss 贡献 | 是否被裁 |
|---|---|---|---|
| 3.0 | +1 | $-1.2$ | ✓ 裁 |
| 3.0 | −1 | $+3.0$ | ✗ 不裁 |
| 0.5 | +1 | $-0.5$ | ✗ 不裁 |
| 0.5 | −1 | $+0.8$ | ✓ 裁 |

（注意 `policy = -min(...)` ✓ 有负号 ✓ —— 因为我们**最小化** loss ✓）

同时它还断言：$r=3$ 时 $\mathrm{approx\_kl} = (3-1)-\ln 3 = 0.9014$ ✓；$r=0.5$ 时 $(0.5-1)-\ln 0.5 = 0.1931$ ✓

---

<a name="7"></a>
## 7. 完整损失函数：四项与它们的符号

我们的 `ppo_loss` 返回的就是这一条 ✓（`loss.py`）：

$$
\boxed{
\mathcal{L} =
\underbrace{\mathcal{L}^{\text{policy}}}_{\text{裁剪代理目标}}
+ c_v\underbrace{\mathcal{L}^{\text{value}}}_{\text{价值回归}}
+ \underbrace{\mathcal{L}^{\text{KL}}}_{\text{参考约束}}
+ c_e\underbrace{\mathcal{L}^{\text{entropy}}}_{\text{熵奖励（负号）}}
}
$$

### 7.1 策略项（有负号 ✗）

$$
\mathcal{L}^{\text{policy}} = -\,\mathbb{E}_t\Big[\min\big(r_t A_t,\ \mathrm{clip}(r_t)A_t\big)\Big]
$$

**为什么有负号** ✓：优化器**最小化** ✓；而策略梯度是**最大化**代理目标 ✓ → 所以取负 ✓✓
**这是最经典的符号错误来源** ✗✓ —— 如果符号反了 ✓ 模型会**主动学坏** ✓ 而且 loss 看起来还在下降 ✗

### 7.2 价值项

$$
\mathcal{L}^{\text{value}} = \frac{1}{2}\,\mathbb{E}_t\Big[\max\big((V_\theta(s_t)-G_t)^2,\ (\mathrm{clip}(V_\theta,V_{\text{old}}\pm\epsilon_v)-G_t)^2\big)\Big]
$$

- $\frac{1}{2}$ 是约定 ✓（求导后抵消平方的 2 ✓）
- $G_t$ 是**回报目标**（用 GAE 的 $\lambda$-回报 ✓ 见 §8.4）
- 价值裁剪防止 critic 单步变化过大 ✓（$c_v = 0.05$ ✓）

⚠️ **我们踩过的坑** ✗✗：$c_v$ 与 value head 的学习率**会联手抢走策略的梯度** ✓
实测（`RECORD.md` §2.5）：分别反传两项，看落在主干上的梯度范数之比

| 配置 | value 项 / policy 项 | value_loss |
|---|---|---|
| $c_v=0.5$，head lr $= 10\times$ | **23.1×** ✗✗ | 5.98 |
| $c_v=0.05$，head lr $= 1\times$ | **0.1×** ✓ | 0.06 |

**机制** ✓：全局 `clip_grad_norm_` 用**总范数** ✓ → value 项占 23 倍时 ✓ 策略只拿到约 $1/24$ 的有效步长 ✗ → 等于没在学 ✓
**修法** ✓：降 $c_v$ + head lr 不放大 ✓ 并**断言**这个比值在合理范围 ✓

### 7.3 KL 惩罚（对参考模型）

$$
\mathcal{L}^{\text{KL}} = \beta\,\mathbb{E}_t\Big[\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_{\text{ref}}(\cdot\mid s_t)\big)\Big]
$$

**逐 token 的两个估计量** ✓（`kl_penalty` ✓）：

设 $d = \log\pi_{\text{ref}}(a_t\mid s_t) - \log\pi_\theta(a_t\mid s_t)$

| 估计量 | 公式 | 性质 |
|---|---|---|
| **k1** | $d$ | 无偏 ✓ 但**可以为负** ✗ → 惩罚间歇性变成奖励 ✗ |
| **k3** ✓ | $e^{d} - d - 1$ | **恒 ≥ 0** ✓ 但 $e^d$ 会爆 ✗ |

**k1 为什么可以为负** ✗：当新策略比参考**更**看好采样到的 token 时 ✓ $d<0$ ✓ → "KL"是负的 ✗✓

**k3 为什么会爆** ✗✗：$d$ 稍大就指数爆炸 ✓ 而且它的梯度是 $e^d\cdot\partial d$ ✓
实测：`grad_norm` 冲到 **1.1e8** ✗（100 步里 11 步 > 3 ✓，**fp32 也出现** ✓ 所以不是精度问题 ✓）

**修法** ✓✓：把 $d$ 从上方截断 ✓（我们取 `KL_D_MAX = 10` ✓）：

```
d = 50  →  kl = 22015.5（截住 ✓）      不截的话 exp(50) = 5.2e21 ✗（差 16 个数量级 ✓）
d = 0.05 → 0.001271（正常范围完全没被动 ✓）
```
实测效果：`grad_norm` 峰值 **6498 → 10.0** ✓✓（小 650 倍 ✓）

### 7.4 熵奖励（注意负号 ✗）

$$
\mathcal{L}^{\text{entropy}} = -\,c_e\,\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot\mid s_t))\big]
$$

**负号的理由** ✓：熵**越大越好**（鼓励探索 ✓）而我们要**最小化** loss ✓ → 所以取负 ✓✓
**我们报告的是原始熵** ✓（正的、可解释 ✓）而进 loss 的是 $-c_e\cdot\mathcal{H}$ ✓ —— 两者差一个线性变换 ✓

⚠️ **我们十二次 run 的 $c_e$ 全是 0** ✗✓（从没用过 ✓）
**而实测** ✗：entropy 从 0.33 一路跌到 **0.1396** ✓（越来越尖 ✓）→ 且贪心出现退化 ✓ → 这是下一步该动的旋钮 ✓

### 7.5 mask：所有期望都是"只对生成位置"

上标 $\mathbb{E}_t$ 里的 $t$ **只遍历生成出来的 token** ✓✓ 不是整条序列 ✓
**这就是 `masked_mean` 存在的理由** ✓：

$$
\mathrm{masked\_mean}(x, m) = \frac{\sum_t m_t x_t}{\sum_t m_t}
\quad\text{而不是}\quad \frac{1}{T}\sum_t x_t
$$

**实测代价** ✗：用裸 `mean()` 会被 padding 稀释 ✓（示例：真值 1.0 被算成 0.5 ✓）
`loss.py::_case_d()` 用"压实 batch vs 带垃圾 padding 的 batch"钉住这条 ✓✓（9 项必须完全一致 ✓）

---

<a name="8"></a>
## 8. 优势估计：GAE 的完整推导

### 8.1 两种极端估计

| 估计 | 公式 | 偏差 | 方差 |
|---|---|---|---|
| **蒙特卡洛** | $G_t - V(s_t)$ | **无偏** ✓ | **大** ✗（要跑完整条轨迹 ✓）|
| **TD(0)** | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | **有偏** ✗（依赖 $V$ 准不准 ✓）| **小** ✓ |

GAE 用**指数加权**在两者之间插值 ✓✓

### 8.2 推导

定义 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

定义 GAE：

$$
A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}
$$

**递推形式**（实现用的就是这个 ✓✓）：

$$
\boxed{\,A_t = \delta_t + \gamma\lambda\,A_{t+1}\,}
$$

> **推导** ✓：
> $\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l} = \delta_t + (\gamma\lambda)\sum_{l\ge0}(\gamma\lambda)^{l}\delta_{t+1+l} = \delta_t + \gamma\lambda A_{t+1}$ ✓
> （所以实现是**从后往前**扫一遍 ✓）

### 8.3 两个端点的验证

**$\lambda = 0$** ✓：

$$
A_t = \delta_t
$$
→ 退化成 TD(0) ✓

**$\lambda = 1$** ✓：

$$
A_t = \sum_{l\ge0}\gamma^l \delta_{t+l}
$$

展开并望远镜求和：

$$
\begin{aligned}
\sum_{l\ge0}\gamma^l\delta_{t+l}
&= \sum_{l\ge0}\gamma^l r_{t+l} + \sum_{l\ge0}\gamma^{l+1}V(s_{t+l+1}) - \sum_{l\ge0}\gamma^{l}V(s_{t+l})\\
&= \sum_{l\ge0}\gamma^l r_{t+l} + \underbrace{\sum_{l\ge1}\gamma^{l}V(s_{t+l})}_{=A} - \underbrace{\Big(V(s_t) + \sum_{l\ge1}\gamma^{l}V(s_{t+l})\Big)}_{=B}\\
&= \sum_{l\ge0}\gamma^l r_{t+l} - V(s_t) \qquad\blacksquare
\end{aligned}
$$

→ 退化成**蒙特卡洛减去 baseline** ✓✓（$A$ 和 $B$ 里的求和项完全抵消 ✓）

#### ★ 有限回合的边界：为什么必须有 `(1-done)` 截断 ✗✓

上面的望远镜求和隐含一个条件 ✗：**求和必须跑到"值函数为 0 的地方"** ✓
在**终止状态**，价值**按定义为 0** ✓（$V(s_T) := 0$ ✓）→ 于是有限回合里：

$$
\sum_{l=0}^{T-t}\gamma^l\,\delta_{t+l} \;=\; \sum_{l=0}^{T-t}\gamma^l\,r_{t+l} \;-\; V(s_t)
$$

**如果不把终止处的 bootstrap 去掉** ✗：

$$
\sum_{l=0}^{L}\gamma^l\,\delta_{t+l}
= \sum_{l=0}^{L}\gamma^l\,r_{t+l}
+ \underbrace{\gamma^{L+1}V(s_{t+L+1})}_{\text{多出来的这一项}}
- V(s_t)
$$

**数值验证** ✓（$\gamma=\lambda=1$，6 步，随机 $V$）：

| 量 | 值 |
|---|---|
| $\sum_l \delta_l$ | 2.194152 |
| $\sum_l r_l - V(s_0)$ | 1.630476 |
| **差** | **0.563676** |
| $V(s_6)$ | **0.563676** ← ★ 差值**正好等于**终止处的价值 ✓✓ |

→ 把终止的 bootstrap 置零（即 `(1-done)` ✓）之后，两边**逐位相等**（差 $0.00\text{e}{+}00$）✓✓

**★ 这就是 §8.5 那两处 `(1-done)` 截断存在的理由** ✓✓ —— 它不是"实现细节" ✗
它是**望远镜求和成立的前提** ✓✓ —— 漏掉就会让 $V$ 跨 episode 泄漏 ✗ 而且**不报错** ✓

（写这份文档时我先写了个数值验证脚本 ✗，结果两边不等 ✓ 差点以为推导错了 ✗
 一查：差值恰好是 $V(\text{终止})$ ✓ —— **是我的验证脚本漏了终止条件** ✓ 不是推导 ✗
 这件事本身正好说明了这一节为什么重要 ✓✓）


**所以 $\lambda \in [0,1]$ 是一个连续的偏差–方差旋钮** ✓✓（我们取 $\lambda = 0.95$ ✓）

### 8.4 那 `returns` 是什么

价值网络要回归的目标 ✓（$\lambda$-回报）：

$$
G_t^\lambda = A_t^{\text{GAE}} + V(s_t)
$$

**为什么这样写** ✓：因为 $A_t = G_t^\lambda - V(s_t)$ ✓（定义 ✓）→ 移项即得 ✓
**代码对应** ✓：`gae.py` 同时返回 `advantages` 和 `returns` ✓ 而 `loss.py` 用 `returns` 做价值回归 ✓

### 8.5 ★ 两处 `(1-done)` 截断（不是一处 ✗）

我们每个 episode **恰好有一个终止** ✓（末 token ✓）→ `dones[rows, last] = True` ✓

**第一处** ✓：算 $\delta_t$ 时，最后一步不能 bootstrap：

$$
\delta_t = r_t + \gamma\,(1-d_t)\,V(s_{t+1}) - V(s_t)
$$

（否则会把"下一回合的开头"当成"这一回合的延续" ✗✓）

**第二处** ✓：GAE 的递推跨过终止必须断掉：

$$
A_t = \delta_t + \gamma\lambda\,(1-d_t)\,A_{t+1}
$$

**漏掉任何一处** ✗ → 值会跨 episode 泄漏 ✓ → 不报错 ✓ 只是学歪 ✓

### 8.6 优势归一化（作用域是**最容易错**的地方 ✗✗）

$$
\hat{A}_t = \frac{A_t - \mu}{\sigma + \epsilon},\qquad
\mu = \mathrm{mean}(\{A_t : m_t = 1\}),\quad \sigma = \mathrm{std}(\cdot)
$$

**作用域必须覆盖"这一批全部有效 token"** ✓✓

**实测教训** ✗：
- 如果**按 minibatch 各自归一化** ✗ → 同一批数据被放缩成 8 种不同倍数 ✓ → 累积出来的梯度 ≠ 真批量梯度 ✓ **而且不报错** ✓✓
- 如果**按同一条题的组**归一化 ✓ → 那是 GRPO ✓（另一回事 ✓ 见 §11 ✓）

**我们的防呆** ✓✓：`update.py::_case_minibatch_equiv()` 断言
**`num_minibatches=1` 与 `=8` 的全部参数梯度逐位一致** ✓✓ —— 这一条能一次抓出四个静默错误 ✓

---

<a name="9"></a>
## 9. ★ 为什么单轮任务必须 `γ = 1`

这是我们**真实踩过并量化过**的一个 bug ✓✓

### 9.1 推导

单轮生成任务里奖励只在**最后一步**给 ✓ 其他步 $r_t = 0$ ✓
所以某一步 $t$ 的回报是：

$$
G_t = \gamma^{T-t}\cdot R \qquad (T = \text{总步数})
$$

**★ 折扣因子随长度变化** ✗✓：

| 长度 $T$ | $\gamma^{T}$（$\gamma=0.99$）|
|---|---|
| 80 | $0.99^{80} = 44.8\%$ |
| 160 | $0.99^{160} = 20.0\%$ |
| 300 | $0.99^{300} = 4.9\%$ |

**注意比较的对象** ✓：我们比的是**不同长度轨迹的"有效奖励"** ✓
→ 同样答对的答案 ✓ 短的得到 $0.448R$ ✓ 长的只得到 $0.20R$ ✗✓
→ **策略被鼓励去缩短回答** ✗✗ —— 而且这个压力**藏在 $\gamma$ 里** ✓ 日志上完全看不见 ✓✓

### 9.2 实测后果

第一次 RL（0.5B）的段均值：

| 段 | 奖励均值 | 生成长度 |
|---|---|---|
| 1-20 | 0.163 | 130 |
| 21-40 | 0.125 | 132 |
| 61-80 | 0.188 | 93 |
| 81-100 | 0.075 | **85** |

→ 长度单调下降 ✓ 斜率约 $-0.63$ token/步 ✓ → 然后崩溃 ✓

### 9.3 正确取值

$$
\boxed{\gamma = 1 \quad\text{（单轮生成任务）}}
$$

**理由** ✓：一个 episode 内部**没有"未来的新起点"** ✓ → 不需要折扣 ✓
**折扣的正当用途** ✗：无限/长回合里让回报有限 ✓ —— 我们每条轨迹都有限且终止 ✓ → 不需要 ✓

**并且要有一条带牙齿的测试** ✓✓：`config.py::_case_gamma()` 断言单轮任务的 $\gamma$ 必须为 1 ✓

---

<a name="10"></a>
## 10. 完整算法：伪代码 ↔ 我们的代码

### 10.1 伪代码

```
初始化：策略 π_θ（+ 可选的 critic V_φ），参考模型 π_ref（冻结 ✓）
for iter in 1..N:
    # ── 1. 采样（同策略 ✓）──
    抽一批题 D
    τ ~ π_θ(·|D)                      # 采样：T=1.0 ✓ 左 padding ✓
    记录 logπ_old(a_t|s_t)             # ★ 行为策略的 logprob ✓
    算奖励 R(τ)（结果项 + 过程项 ✓）

    # ── 2. 优势 ──
    V = V_φ(s_t)                       # critic 前向 ✓
    A, G = GAE(R, V, (1-done), γ, λ)   # §8 ✓
    Â = normalize(A)                   # ★ 作用域 = 整批 ✓

    # ── 3. 内层多轮更新（同批数据复用 ✓）──
    for epoch in 1..E:
        for minibatch in split(D):
            r = exp(logπ_θ - logπ_old)          # ★ 起手必须 ≡ 1 ✓
            L = -min(r·Â, clip(r)·Â)            # 策略 ✓
                + c_v · value_loss              # 价值 ✓
                + β · KL(π_θ ‖ π_ref)           # 参考 ✓
                - c_e · H(π_θ)                  # 熵 ✓
            backward(L); clip_grad_norm(); step()
```

### 10.2 ★ 我们的实现里三个"结构性"的决定

**(a) 整个更新做成一个纯函数** ✓ `update.py::ppo_update(model, optimizer, batch, cfg, ...)`
→ 好处：能用**逐参数梯度**做等价性测试 ✓✓

**(b) 优势的归一化在切 minibatch **之前**做** ✓✓
→ 保证 $\mu,\sigma$ 覆盖整批 ✓（§8.6 ✓）

**(c) 内层轮数 $E$ 与 minibatch 数分开控制** ✓
- `cfg.inner_epochs`：同一批数据看几遍 ✓（我们取 2 ✓）
- `cfg.num_minibatches`：一次 forward 吃多少行 ✓（显存 ∝ 批量/这个数 ✓）
→ ⚠️ 但 $E>1$ 意味着第二遍时数据已经 off-policy ✗（策略动过了 ✓）→ 这也是方差的来源之一 ✓

### 10.3 我们加的四道防呆（都不是教科书内容 ✓）

| 防呆 | 抓什么 |
|---|---|
| `assert_ratio_starts_at_one` ✓ | 采样路径 ≠ 重算路径 ✗（§4.4 ✓）|
| `batch.check()` 7 条契约 ✓ | mask 不连续 / `dones` 位置错 / 奖励漏出 mask ✗ |
| `_case_minibatch_equiv` ✓ | 归一化作用域 / token 加权 / zero_grad 位置 ✗ |
| `_case_padded_vs_compact` ✓ | 用裸 `mean()` 而不是 `masked_mean` ✗ |

**共同点** ✗✓：它们抓的 bug **全都不报错** ✓✓ —— 只会让训练悄悄学歪 ✓

---

<a name="11"></a>
## 11. 去掉 critic 的变体：GRPO / RLOO

### 11.1 动机（我们的实测数据 ✓）

| 量 | 实测值 |
|---|---|
| `value_loss` | 0.0143 ~ 0.0202（**稳定而小** ✓）|
| value 项对主干的梯度 / policy 项 | **0.01 ~ 0.1 倍** ✓ |

→ **critic 几乎没在出力** ✗✓ → 那能不能干脆不要它 ✗？

### 11.2 GRPO（Group Relative Policy Optimization）

对**同一道题**采 $G$ 条 ✓ 组内做相对归一化：

$$
A_i = \frac{r_i - \mathrm{mean}_g(r)}{\mathrm{std}_g(r) + \epsilon}
$$

**优点** ✓：不需要 critic ✓（组均值就是 baseline ✓）
**隐患** ✗：**除以 $\mathrm{std}$ 会引入偏置** ✗✓（回忆 §3.1 的推论 2：baseline 不能依赖动作 ✓ 而 std 依赖整组 ✓）
另外：**组内全同**（0/8 或 8/8）时 → $A_i \equiv 0$ ✓ → 该题没有梯度 ✓✓
→ 这其实是个**特性** ✓：自动过滤"根本不会"和"早就稳了"的题 ✓✓
（我们的探针说中间混合的题占 **67%** ✓ 正好只在这些上出梯度 ✓）

**实测教训** ✗：用 $\lambda=0.3$ 的过程项跑 GRPO ✓ → `reward_proc/0.3 = 0.887` ✓ 且末尾奖励冲到 **1.26**（上限 1.3 ✓）而答案准确率在掉 ✗✓
→ 机制：组内 0/1 答案常常全同 ✓ → 那么组内差异就由**稠密的过程项**决定 ✗✓ → **GRPO 把 shaping 放大了** ✓✓

### 11.3 RLOO（REINFORCE Leave-One-Out）✓✓

$$
A_i = r_i - \frac{1}{k-1}\sum_{j\ne i} r_j
$$

**它为什么无偏** ✓✓：把"组内其他样本的均值"当作 baseline ✓
→ 而由 §3.1，**任何不依赖当前动作的 baseline 都不引入偏置** ✓✓
（严格说："留一均值"依赖其他样本 ✓ 但不依赖 $a_i$ 本身 ✓ → 在我们的采样设定下无偏 ✓）

**它比 GRPO 好在哪** ✓：不做 $\mathrm{std}$ 除法 ✗ → 无偏 ✓ 且方差更低 ✓✓
**改动量** ✗：我们已经有 `grpo_advantages` ✓ → RLOO 只是把归一化那两行换掉 ✓（约 3 行 ✓✓）

### 11.4 ★ token 级平均带来的长度偏置 ✗

我们的 loss 是 **token 级**平均 ✓：

$$
\mathcal{L} = \frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}{\sum_{b,t} m_{b,t}}
$$

→ 一条 **200 token** 的序列贡献 200 个 token ✓ 一条 **50 token** 的贡献 50 个 ✓
→ **长序列在梯度里权重更大** ✗✓

在 PPO 里这是个设计选择 ✓；但在 GRPO 里它是**已知偏置** ✓（Dr.GRPO 的出发点 ✓）
修法 ✓：先在**每条序列内部**求均值 ✓ 再对**序列**求均值 ✓（即"sequence-level 归一化" ✓ ~10 行 ✓）

---

<a name="12"></a>
## 12. 实现里最容易错的地方（每条都是我们踩过的）

| # | 症状 | 根因 | 修法 | 怎么防 |
|---|---|---|---|---|
| 1 | ratio 起始偏差 0.9811 | `model.gen_len` 陈旧 ✗ 重算 forward 切错尾部 | 重算前先设本批的 $G$ ✓ | 起手断言 ✓ |
| 2 | 训练不动且不报错 | `logprobs_old` 与重算路径不一致 ✗ | 统一 tokenize / padding ✓ | 同上 ✓ |
| 3 | 长度单调缩短 → 崩溃 | 单轮任务用了 $\gamma=0.99$ ✗ | $\gamma=1$ ✓ | `_case_gamma` ✓ |
| 4 | 永不停止（EOS 率 98%→0%）| 截断的输出照样拿满过程奖励 ✗ | **截断门**：无 EOS → 过程项清零 ✓ | 实测 EOS 率 ✓ |
| 5 | 训练奖励高而 eval 低 | 过程项被刷 ✗ | 拆分日志 `reward_base`/`reward_proc` ✓ | 两列分开打 ✓ |
| 6 | eval 打出 0.714（真实 0.27）| 评估读了含过程项的 `batch.rewards` ✗ | 评估只用纯答案奖励 ✓ | 两个口径都报 ✓ |
| 7 | `grad_norm` 冲到 1e8 | k3 的 $e^d$ 爆炸 ✗ | 截断 $d\le10$ ✓ | `_case_e` ✓ |
| 8 | 策略"被钉住"（approx_kl 0.0013）| 固定 `kl_coef` 后期过强 ✗ | **自适应 KL** ✓ | 闭环收敛测试 ✓ |
| 9 | padding 位置漏进统计量 | 裸 `mean()` ✗ | `masked_mean` ✓ | 压实 vs 填充 ✓ |
| 10 | value 项抢走策略梯度（23×）| $c_v$ + head lr 过大 ✗ | 降 $c_v$、head lr=1× ✓ | 梯度比断言 ✓ |
| 11 | minibatch 累积 ≠ 真批量 | 归一化作用域 / token 加权 ✗ | 整批归一化 + token 加权 ✓ | 逐参数梯度等价 ✓ |
| 12 | 同一模型两次评估差 ±3.2 点 | CUDA 非确定性 ✗ | 确定性模式 ✓ | 两次必须一字不差 ✓ |

**共同主题** ✗✓：**12 条里有 10 条不会报错** ✓✓ —— 它们只是让训练悄悄学歪 ✓
**所以"实现 PPO"的难点不在公式 ✗ 在把这些静默错误变成会响的断言** ✓✓

---

<a name="13"></a>
## 13. 公式 → 我们的代码/数字 对照表

| 公式 | 代码 | 我们的取值 / 实测 |
|---|---|---|
| $\gamma$ | `cfg.gamma` | **1.0** ✓（单轮任务 ✓ §9）|
| $\lambda$ | `cfg.lam` | 0.95 |
| $\epsilon$（裁剪）| `cfg.clip_eps` | 0.2 |
| $c_v$ | `cfg.value_coef` | 0.05（从 0.5 降下来 ✓）|
| $c_e$ | `cfg.entropy_coef` | **0.0** ✗（十二次没用过 ✓）|
| $\beta$（KL）| `cfg.kl_coef` | 0.01 → **自适应 0.0029** ✓ |
| $r_t$ | `loss.py: ratio = log_ratio.exp()` | 起手必须 ≡1 ✓ |
| $\hat A$ | `gae.py: normalize_advantages` | 整批 $(\mu,\sigma)$ ✓ |
| GAE 递推 | `gae.py: compute_gae` | $\lambda=0.95$ ✓ 两处 `(1-done)` ✓ |
| $G_t^\lambda$ | `returns` | 价值回归目标 ✓ |
| KL 估计量 | `cfg.kl_estimator` | `k3` + `KL_D_MAX=10` ✓ |
| 逐 token 期望 | `masked_mean` | 只对 mask 内 ✓ |
| 参考模型 | `ref_model`（冻结 ✓）| 每步算一次 logprob ✓ |
| 内层轮数 | `cfg.inner_epochs` | 2 |
| minibatch | `cfg.num_minibatches` | 8（B=32 时每块 4 行 ✓）|

**训练动态的实测三档** ✓：
```
健康： approx_kl 0.01~0.05 · clip_frac < 0.1 · mean_ratio ≈ 1.00
实测： approx_kl 0.0029~0.0037（**偏低** ✗）· clip_frac 0.0067~0.0144（很低 ✓）· mean_ratio 1.0011~1.0024 ✓
→ 所以严格说：我们的步长是**偏保守**的 ✗ 而不是偏大 ✓（§5.5 的 lr 换算 ✓）
```

---

<a name="14"></a>
## 14. 常见理解型问答

**Q1：为什么要 `clip`，直接限制学习率不行吗？**
不行 ✗。学习率限制的是**参数**的变化 ✓；而 PPO 关心的是**策略分布**的变化 ✓。
两者不是一回事 ✓：LoRA 下参数只动 0.29%，但分布可能已经变了很远 ✓
`approx_kl` 就是直接量"分布变了多少"的 ✓（我们实测它和 lr 成 $lr^2$ 关系 ✓）

**Q2：为什么 `min` 里要保留未裁剪项？**
见 §6.2 的情形 2 和 3 ✓。只用 `clip(r)·A` 会**丢掉"继续远离坏动作"和"继续拉高好动作"的梯度** ✗
`min` 的作用是：**只在"占便宜"的方向上裁剪** ✓✓

**Q3：advantage 归一化会改变梯度方向吗？**
不会 ✗（它是个正的缩放 + 平移 ✓ 而平移恰好等于减了一个 baseline ✓ → 由 §3.1 不引入偏置 ✓）
但**作用域**会决定"相对谁" ✓✓ —— 按整批归一化 vs 按组归一化是**两个不同的算法** ✓

**Q4：为什么需要参考模型？**
防止策略为了奖励**跑得太远** ✗（语言能力退化、输出乱码、格式崩坏 ✓）
它就是"信任域"的一种实现 ✓（TRPO 的约束的实用替代 ✓）
**实测** ✓：$\beta$ 太大 → 学不到东西 ✓（0.05 时训练奖励从 0.5 崩到 0.15 ✗）；太小 → 漂移失控 ✓
→ 所以应该**自适应** ✓✓（把实测的原始 KL 稳定在目标附近 ✓）

**Q5：`approx_kl` 和 `kl` 有什么区别？**
- `approx_kl` $= \mathbb{E}[(r-1)-\log r]$ ✓：**新旧策略**之间的距离 ✓ → 量的是**这一步走了多远** ✓
- `kl` $= \beta\cdot\mathbb{E}[\text{k3}(d)]$ ✓：**当前策略与参考模型**的距离 ✓ → 量的是**总共走了多远** ✓
两者完全不同 ✓✓ 混用会得出反的结论 ✓（我们曾把"$kl$ 涨"当成"$approx\_kl$ 涨" ✗）

**Q6：为什么 advantage 用 GAE 而不是直接 $G_t - V(s_t)$？**
因为 $G_t$ 的方差随 $T$ 增长 ✗ → 我们的 $T$ 有 100~200 ✓ → 方差会很可观 ✓
GAE 用 $\lambda$ 做指数加权 ✓ 在几乎不引入偏差的前提下大幅降方差 ✓✓（§8.3 ✓）

**Q7：GRPO 和 PPO 的本质区别是什么？**
**只有一处** ✓✓：**advantage 在哪个范围内做相对比较** ✓
- PPO：在**整批（很多道不同的题）**内 ✓
- GRPO：在**同一道题的那 $G$ 条**内 ✓✓
其余（裁剪、KL、熵）都一样 ✓
→ 这个区别在"同一批里题目难度差异大"时很关键 ✓：PPO 的优势里混入了"这题本身难不难" ✗

**Q8：为什么单轮任务里"奖励只在末 token"很重要？**
它让 $V(s_t)$ 的含义变成"从前 $t$ 个 token 出发，最终答对的概率" ✓✓
也让它成为**非常难估**的量 ✓（一个前缀后面能走的路很多 ✓）
→ 这就是"critic 出力很少"（实测 0.01~0.1 倍 ✓）的深层原因 ✓
→ 也是 GRPO/RLOO 这类"不要 critic"的方法**特别适合**它的原因 ✓✓

---

## 附：一页速查（面试/复查用）

$$
\begin{aligned}
&\text{目标：} && J(\theta)=\mathbb{E}_{\pi_\theta}\Big[\textstyle\sum_t\gamma^t r_t\Big]\\
&\text{策略梯度：} && \nabla_\theta J = \mathbb{E}\big[\nabla_\theta\log\pi_\theta(a_t|s_t)\,A_t\big]\\
&\text{baseline 无偏：} && \mathbb{E}_a[\nabla\log\pi_\theta(a|s)\,b(s)]=0\\
&\text{比值：} && r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)},\qquad r_t(\theta_{\text{old}})\equiv1\\
&\text{裁剪目标：} && \mathcal{L}^{\text{policy}}=-\mathbb{E}\big[\min(r_tA_t,\ \mathrm{clip}(r_t,1\pm\epsilon)A_t)\big]\\
&\text{TD 误差：} && \delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)\\
&\text{GAE：} && A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}\\
&\text{价值目标：} && G_t^\lambda=A_t+V(s_t)\\
&\text{总计：} && \mathcal{L}=\mathcal{L}^{\text{policy}}+c_v\mathcal{L}^{\text{value}}+\beta\,\mathrm{KL}-c_e\,\mathcal{H}\\
&\text{单轮任务：} && \boxed{\gamma=1}
\end{aligned}
$$

---

*配套阅读：`RECORD.md`（这份实现踩过的所有坑 + 实测数字）· 附录 A（代码地图）*
