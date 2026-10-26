# DPO 推导（从 RLHF 目标到 DPO 损失）

> 目的：把 DPO 从头推一遍 ✓，每一步都写清楚"为什么" ✓。
> 最后两节是**我们踩过的坑** ✓ —— 那些不是数学问题，是实践问题 ✗。
>
> 参考：Rafailov et al., *Direct Preference Optimization: Your Language Model is
> Secretly a Reward Model*, NeurIPS 2023。

---

## 0. 符号

| 符号 | 含义 |
|---|---|
| $x$ | prompt（题目）|
| $y$ | 一条完整回答（completion）|
| $\pi_\theta(y\mid x)$ | 当前策略（要训的模型）|
| $\pi_{\text{ref}}(y\mid x)$ | 参考策略（冻结的 SFT 模型）|
| $r(x,y)$ | 隐式奖励（**未知** ✓ 我们从来没见过它 ✗）|
| $\beta$ | KL 强度 / 温度（通常 0.1 ✓）|
| $y_w, y_l$ | 一对回答里的 **winner（好）** 和 **loser（差）** |
| $\sigma(z)=\dfrac{1}{1+e^{-z}}$ | sigmoid ✓ |

**记号约定**：本文里 $\log\pi(y\mid x)$ 指**整条回答**的 log 概率之和
$\sum_t \log\pi(y_t \mid x, y_{<t})$ ✓（DPO 用序列级 ✓，不是 token 级 ✗）。

---

## 1. 起点：带 KL 约束的 RLHF 目标

RLHF 想干的事：

$$
\max_{\pi_\theta}\;
\mathbb{E}_{x\sim\mathcal{D}}\;\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
  r(x,y)\;-\;\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
\right]
\tag{1}
$$

**两项各是什么** ✗：

* $r(x,y)$：奖励（我们希望回答好 ✓）
* $-\beta\log\dfrac{\pi_\theta}{\pi_{\text{ref}}}$：**KL 惩罚** ✓
  —— 不让新策略离参考太远 ✗（否则会去"刷"奖励 ✓ 输出崩掉 ✗✓）

**★ 这一步就是 PPO/GRPO 的训练目标** ✓ —— DPO 的出发点和它们**完全一样** ✓。

---

## 2. 关键一步：这个目标有**闭式最优解**

**(1)** 是对 $y$ 的分布求期望并优化 ✓ —— 这是一个**逐点**（per-$x$）的变分问题 ✓。

固定 $x$，把它写成对**任意分布** $\pi$ 的泛函：

$$
J(\pi)=\sum_{y}\pi(y)\left[r(x,y)-\beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\right]
\tag{2}
$$

**★ 约束**：$\sum_y \pi(y)=1$ ✓。用 Lagrange 乘子 $\lambda$：

$$
\mathcal{L}(\pi,\lambda)=\sum_y\pi(y)\left[r-\beta\log\frac{\pi}{\pi_{\text{ref}}}\right]+\lambda\left(1-\sum_y\pi(y)\right)
$$

对 $\pi(y)$ 求偏导并令其为 0（把 $\sum_y$ 看成独立变量 ✓）：

$$
\frac{\partial\mathcal{L}}{\partial\pi(y)}
= r(x,y)-\beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}-\beta+\lambda = 0
$$

解出：

$$
\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}=\frac{r(x,y)}{\beta}+\frac{\lambda-\beta}{\beta}
$$

两边取指数，把与 $y$ 无关的因子并进一个常数：

$$
\boxed{\;\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)\;}
\tag{3}
$$

其中

$$
Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right)
\tag{4}
$$

是**配分函数**（partition function）✓ —— 它的作用是让 (3) 归一化 ✓。

**★ 直觉** ✓：最优策略 = 参考策略 **按 exp(奖励/β) 重新加权** ✓✓
→ 奖励高的回答概率被放大 ✓；β 越小放得越猛 ✓；β→∞ 就退回参考 ✓。
（这正是"**用奖励重加权参考策略**"那句话的来源 ✓）

---

## 3. 反解：把奖励用策略表达出来

对 (3) 取 log：

$$
\log\pi^*(y\mid x)=\log\pi_{\text{ref}}(y\mid x)+\frac{1}{\beta}r(x,y)-\log Z(x)
$$

移项解 $r$：

$$
\boxed{\;r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)\;}
\tag{5}
$$

**★★ 这一步是整个 DPO 的核心** ✓✓ ——
它说明：**如果你知道最优策略，你就知道奖励** ✓（差一个只依赖 $x$ 的常数 ✓）。

也就是说 —— **策略本身就隐含了一个奖励模型** ✓✓（论文标题那句 "Your Language
Model is Secretly a Reward Model" ✗）。

---

## 4. 偏好模型：Bradley–Terry

人类标注给的是**成对比较** ✓："$y_w$ 比 $y_l$ 好" ✓。
Bradley–Terry 模型假设：

$$
p(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)
\tag{6}
$$

**★ 关键**：它只依赖奖励的**差** ✓ —— 所以 (5) 里那个 $\beta\log Z(x)$ 在相减时
**会自己消掉** ✓✓。这是 DPO 能绕开 $Z(x)$ 的全部原因 ✓。

---

## 5. 把 (5) 代进 (6) → DPO

把 (5) 分别代入 $r(x,y_w)$ 和 $r(x,y_l)$，相减：

$$
r(x,y_w)-r(x,y_l)
=\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\underbrace{+\beta\log Z(x)-\beta\log Z(x)}_{=\,0\;\checkmark}
\tag{7}
$$

于是**未知的 $r$ 和算不出的 $Z$ 全部消失** ✓✓，只剩**两个策略**：

$$
p(y_w\succ y_l\mid x)=
\sigma\!\left(
\beta\left[
\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right)
\tag{8}
$$

现在把"最优策略 $\pi^*$"换成"**待训策略 $\pi_\theta$**" ✓（因为我们要找的就是它 ✓），
用**最大似然**去拟合偏好数据 → 取负对数 → **DPO 损失**：

$$
\boxed{
\mathcal{L}_{\text{DPO}}(\theta)
=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log\sigma\!\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right)
\right]
}
\tag{9}
$$

**★★ 就这么简单** ✓✓ —— **没有奖励模型 ✓ 没有 PPO ✓ 没有 value ✓ 没有采样循环 ✓**
（采样只是为了**产生 pair** ✓，更新本身是**监督式**的 ✓）

---

## 6. 记号的简化（和我们代码一致）

定义**每条回答的"隐式奖励"**：

$$
\hat r_\theta(y\mid x)\;=\;\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
\tag{10}
$$

则 (9) 就是：

$$
\mathcal{L}=-\log\sigma\big(\hat r_\theta(y_w)-\hat r_\theta(y_l)\big)
\tag{11}
$$

再令

$$
\text{margin}\;=\;\big(\log\pi_\theta(y_w)-\log\pi_\theta(y_l)\big)
-\big(\log\pi_{\text{ref}}(y_w)-\log\pi_{\text{ref}}(y_l)\big)
\tag{12}
$$

则 $\mathcal{L}=-\log\sigma(\beta\cdot\text{margin})$ ✓ ——
**这正是 `dpo.py` 里 `dpo_loss()` 的实现** ✓✓：

```python
pi_margin  = logp_pos - logp_neg            # (12) 第一项 ✓
ref_margin = ref_logp_pos - ref_logp_neg     # (12) 第二项 ✓
margin     = pi_margin - ref_margin          # (12) 整体 ✓
loss       = -F.logsigmoid(beta * margin).mean()
```

**★ 自测里那条"策略==参考 → margin=0"** ✓ 就是在验证 (12) 的**第二项没被漏掉** ✓✓
（漏了的话 DPO 会退化成"无条件推高正样本" ✗ → 无限过优化 ✗）。

---

## 7. 梯度长什么样（理解的钥匙）

对 (9) 求梯度。记 $z=\beta\cdot\text{margin}$，则 $\dfrac{d}{dz}\left[-\log\sigma(z)\right]=-\sigma(-z)$：

$$
\nabla_\theta\mathcal{L}
=-\,\beta\;\mathbb{E}\Big[\,
\underbrace{\sigma\!\big(-z\big)}_{\text{权重 }w\in(0,1)}
\big(\nabla_\theta\log\pi_\theta(y_w)-\nabla_\theta\log\pi_\theta(y_l)\big)
\Big]
\tag{13}
$$

**两项各是什么** ✗：

* $\nabla_\theta\log\pi_\theta(y_w)$：**提高**好回答的概率 ✓
* $-\nabla_\theta\log\pi_\theta(y_l)$：**压低**差回答的概率 ✓
* $w=\sigma(-z)\in(0,1)$：**权重** ✓ —— 已经分得很开的 pair（$z$ 大）权重小 ✓；
  分不开的 pair（$z\approx0$）权重大 ✓✓

**★★ 所以 DPO 的梯度是"**加权对比**"** ✓✓：
它不要求"正样本概率高" ✗，只要求"**正样本比负样本更被偏好，且比参考更偏好**" ✓。

**★ 这就是为什么 (a) 正负必须同分布 ✓ (b) 它不怕正样本概率已经 0.94 ✓✓
（和我们 RFT 那次 `loss 0.06` 卡死的情况正好互补 ✓）。**

---

## 8. 和 PPO/GRPO 的关系

| | PPO / GRPO | DPO |
|---|---|---|
| 需要奖励模型 | 需要（或规则 ✓）| **不需要** ✓✓ |
| 需要 critic / value | PPO 需要 ✗ | **不需要** ✓✓ |
| 需要 ref 模型 | 需要（KL ✓）| **需要** ✓（同样的 KL ✓）|
| 需要在线采样 | 是（on-policy ✓）| **可以离线** ✓（也可在线 ✓）|
| 梯度形式 | 策略梯度（含 ratio ✗）| **监督式的对比** ✓✓ |
| 数据 | $(x, y, r)$ | $(x, y_w, y_l)$ ✓ |
| 失败模式 | 高方差 ✓ ratio 炸 ✓ | 偏好数据不干净 ✗ |

**★ 一句话** ✓：**DPO 把"RL 问题"变成了"分类问题"** ✓✓
—— 代价是**丢掉了探索** ✗（它只在给定 pair 上学 ✓）。

---

## 9. ★★ 我们踩过的三个坑（不是数学问题，是数据问题 ✗）

### 坑 1：正负必须**同分布** ✗✗

DPO 最大化的是**相对**偏好 ✓。若：

* 正样本 $y_w$ = **官方解答**（LaTeX ✓ 长 ✓）
* 负样本 $y_l$ = **模型自己的错答案**（短 ✓ 风格不同 ✗）

那么 (13) 里那个差**同时包含了"对错"和"文风"两件事** ✗ →
模型学到的是"**模仿官方文风**" ✗，而不是"为什么这道题对" ✓。

**实际后果**：我们在 GSM8K 上做过类似的事（用官方解答当 SFT 目标 ✗）→ **−15 点** ✗✗。

**正确做法** ✓✓：**正负都来自同一次采样** ✓ —— 只差"对/错"这一个维度 ✓。
（`dpo.py` 的 `build_pairs()` 强制了这一点 ✓，自测里专门查了"不跨题 ✓、必是(对,错) ✓"）

### 坑 2：把 $\beta\log Z(x)$ 忘了 → 退化成"无条件推正样本" ✗

(7) 里那一项**必须确实消掉** ✓。实现上等价于：**`ref` 那一支不能漏** ✗。
漏掉的症状：`margin` 恒 > 0 ✓、loss 一路降 ✓、但模型**过优化** ✓（输出退化成重复某个正样本 ✓）。

**自测**：策略==参考 → `margin` 必须 = 0 ✓（`dpo.py` 的用例 ③ ✓）。

### 坑 3：`beta` 的方向 ✗

$\beta$ 大 → 更"信任"参考 ✓（对偏好差更敏感 ✗）→ loss 更小 ✓；
$\beta\to0$ → 退化成"不看参考" ✗。
**默认 0.1** ✓；显存/稳定性吃紧时可以试 0.05~0.5 ✓。

---

## 10. ★ 为什么"online"（数据要模型自己产）

(9) 是在**给定偏好数据**上的极大似然 ✓ —— 它**不区分**数据是哪个策略产生的 ✗。
但：

* 若 pair 来自**旧策略** ✗ → 它描述的是"**过去**哪里会错" ✓；
* 而我们要改的是**现在**的策略 ✗。

$$\text{offline DPO}=\text{用旧数据做一次大步更新} \quad
\text{online DPO}=\text{用此刻的数据做小步更新}$$

**★ 经验** ✓：**online / iterative DPO 明显优于 offline** ✓✓（每轮重新采样 ✓）。
我们的设计（每步采 B 题 × G 条 ✓ → 同题内部配对 ✓ → 攒 N 对更新一次 ✓）就是这个思路 ✓。

**★★ 顺带** ✓：这也解释了"为什么强模型改写的数据对 DPO 没用" ✗ ——
它来自**别人的策略** ✓，不是"此刻的模型会犯的错" ✓（可那是 SFT 的问题 ✓，
对 DPO 是**另一个**问题：**分布不同** ✗）。

---

## 11. 一页总结

$$
\underbrace{\max_\pi\mathbb{E}\left[r-\beta\log\frac{\pi}{\pi_{\text{ref}}}\right]}_{\text{(1) RLHF 目标}}
\;\Longrightarrow\;
\underbrace{\pi^*\propto\pi_{\text{ref}}e^{r/\beta}}_{\text{(3) 闭式最优}}
\;\Longrightarrow\;
\underbrace{r=\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z}_{\text{(5) 反解奖励}}
$$
$$
\xrightarrow{\text{代入 BT (6)}}
\underbrace{\mathcal{L}=-\log\sigma\!\left(\beta\!\left[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right]\right)}_{\text{(9) DPO}}
$$

**三句话** ✓✓：

1. **RLHF 的最优策略是"按奖励重加权参考策略"** ✓（(3)）
2. **所以奖励可以反解成策略的 log-ratio** ✓（(5)）—— 只差一个与 $y$ 无关的常数 ✓
3. **成对比较只看奖励差** ✓ → 常数消掉 ✓ → **只剩两个策略，可以直接监督** ✓✓

**实践三律** ✗：

* 正负 **同分布** ✓（否则学文风 ✗）
* `ref` 那一支 **不能漏** ✓（否则过优化 ✗）
* **online 采样** ✓（否则用旧数据改新策略 ✗）
