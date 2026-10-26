# A Survey of On-Policy Distillation for LLMs —— 中文精读摘要

> **原文**：Mingyang Song & Mao Zheng（Tencent, Large Language Model Department）. *A Survey of On-Policy Distillation for Large Language Models*. Preprint, arXiv:2604.00626v4 [cs.LG], 18 Jun 2026. 共 89 页：正文 10 章，参考文献自第 76 页起。
>
> **本文性质**：按章节精读后合并的中文摘要，**不是逐句全文翻译**。术语、方法名、模型名、benchmark 名与公式**一律保留英文/原文符号**（如 OPD、f-divergence、forward KL、logit、πmix、GRPO、O(εT²)），公式用 LaTeX 按原文转写、未改写。
>
> **标注约定**：凡属**我（编者）的提炼归纳、而非原文表述**，均以 `【编者按】` 标出。原文自身的局限、以及本摘要未逐条核验的范围，见文末 **附录 C**。

---

## 速览

**一句话**：off-policy 蒸馏让学生只在"完美的教师前缀"上训练，推理时却必须从自己不完美的输出往下写，这个 distribution mismatch（exposure bias）会沿轨迹**二次**累积（O(εT²)）；**On-Policy Distillation (OPD)** 把采样分布从固定语料换成学生自身 policy——学生自己提轨迹、教师对学生真正走过的状态给反馈——把蒸馏从"单次模仿"变成"迭代纠正"，对应 DAgger 把界从 O(εT²) 收到 O(εT)。

**论文的定位**：自称**首个把 OPD 当作独立范式**（而非 classical KD 的一个变体）处理的综述；核心贡献是把 OPD 形式化为**学生采样轨迹上的 f-divergence 最小化**，并用**三条设计轴**（objective function / signal source / training dynamics）组织散落在 KD、RL、imitation learning 三个社区的文献，同时给出 OPD 与 **KL-constrained RL 的等价性**。

**三条最值得记住的结论** 【编者按】
1. **换的是"数据从哪来"，不是"匹配什么"**。所以 OPD 的核心工程代价在于每步都要 fresh rollout，而非换一个 loss。
2. **没有普适最优的 divergence**。应该**按任务与 token 位置自适应**：推理类任务偏 reverse KL（mode-seeking，防幻觉），开放生成偏 forward KL（mode-covering，保多样性）；EOPD / AOPD 是这条路线的前沿。
3. **蒸馏与 RL 的边界正在消失**。GKD 的 token 级 KL、MiniLLM 的 sequence-level reverse KL、G-OPD 的 reward-augmented 形式，都是 **KL-constrained policy optimization 在不同 reward 定义与约束强度下的特例**。

**阅读指引** 【编者按】
- 只要框架：§1–§3（含 Table 1 / Table 2 两张全局对比表）
- 要动手训：§4（目标函数）+ §5（信号来源）+ §6（效率与稳定化）
- 要判断"我该不该上 OPD"：§7 —— 尤其 **7.1 Success Conditions** 与 **7.4 Decision Framework**
- 工业系统落地：§8；前沿与开放问题：§9

---

## 摘要（原文 Abstract 要点）

**核心问题**：off-policy 蒸馏让学生在 flawless teacher prefixes 上训练，推理时却必须从自身不完美输出续写，exposure bias 沿轨迹累积。OPD 让学生自己生成 sequence、教师对这些"学生走过的状态"给反馈，把蒸馏从 single-pass imitation 变成 iterative correction。文献散落于 knowledge distillation、reinforcement learning、imitation learning 三社区，缺统一处理。

**论文的统一化**：把 OPD 形式化为**学生采样轨迹上的 f-divergence 最小化**，沿三条设计轴（objective functions、signal sources、training dynamics）组织，并给出 OPD 与 KL-constrained reinforcement learning 的形式化联系。

**摘要点名的 open problems**：distillation scaling laws、uncertainty-aware teacher feedback、agent-level distillation。

## §1 Introduction
**①核心问题**：推理时分布失配即 interactive imitation learning 的 compounding error（Ross et al., 2011）——自回归生成把失配按序列长度**二次放大**，horizon $T$ 下期望偏差 $O(\epsilon T^2)$；reasoning 任务最致命，一次早期失误可毁掉整段证明。Gudibande et al. (2023)：off-policy 模仿闭源 LLM 常只学到表面风格、未获得 reasoning competence。

关键在于改变**数据从哪来**而非**匹配什么**：OPD 把采样分布换成学生自身 evolving policy，教师反馈按访问级别分三档——white-box 的 full token-level distributions、black-box 的 scalar rewards / pairwise preferences、teacher-free 下从自身输出导出的 contrastive signals。理论回报对应 DAgger：在 learner 自己访问的 state 上查询 expert，把 $O(\epsilon T^2)$ 降到 $O(\epsilon T)$。

**③数字/事实**：DeepSeek-R1 把 671B mixture-of-experts teacher 蒸馏为 1.5B–70B dense students，long chain-of-thought 基本完整；GKD 与同期 MiniLLM 于 mid-2023 把 OPD 带入 LLM 时代，约三年扩展到 200+ 篇；Qwen3、DeepSeek-V4、Gemma 2、MiMo-V2-Flash 均以 OPD 为核心训练成分，横跨 MoE/dense 与 proprietary/open-weight；**DeepSeek-V4 更进一步，以 pure multi-teacher OPD 取代其 mixed RL stage** 做 model consolidation。

**四点贡献**：
1. **unified theoretical framework**：把 off-policy→on-policy 重铸为 sequential decision-making problem，证明核心 OPD 算法是 student-sampled trajectories 上 f-divergence minimization 的实例（§2），以 divergence choice、argument ordering、sampling mixture 澄清方法关系。
2. **design-centric taxonomy**：沿 objective function design（§4）、signal source architecture（§5）、training dynamics optimization（§6）三轴做 one-method-one-category 分类（§3），不按表面相似分组。
3. **theoretical account of failure**：整合经验记录为 flawed prefix trap、self-play saturation、diversity collapse、calibration-capability gap、multi-turn agentic failure modes，并给出 OPD 优于 off-policy 的形式化条件（§7）。
4. **cross-regime comparative analysis**：在同一框架内考察 white-box / black-box / self-distillation，用分节对比表（Tables 1, 3–9, 2）汇总，并记录 method selection factors（§3.3），把部署约束映射到方法与算力预算。

**对现有综述的批评**：Xu et al. (2024) 一类综述仍持 classical compression framing，把 off-policy 与 on-policy 当作可互换变体，而非理论性质与 failure modes 实质不同的两个 regime；作者称本文为**首个把 OPD 当独立范式**（而非 classical KD 变体）处理的综述。

**④范围/前提**：只收学生自生成训练数据的方法；纳入作用于学生生成数据 + teacher/verifier 监督的 imitation learning、online RL、preference optimization 工作；排除 generic off-policy KD、pruning/quantization 等正交压缩技术、以及不改权重的 inference-time 方法。

## §2 Background and Unified Math
**on-policy 定义**：训练数据在训练时采样自学生当前 policy $p_\theta$（而非固定语料 $\mathcal{D}$ 或 $p_T$）：
$$\min_\theta \mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim p_\theta(\cdot|x)}\big[\mathcal{L}(y,x;\theta,T)\big]$$
外层期望取自学生自身生成，故 $\theta$ 一动数据分布 $p_\theta$ 就漂移，优化地形非平稳、每步需 fresh rollout——这是 OPD 的核心系统级挑战（§6.3）。记号：小写 $p$ 为 token-level、大写 $P$ 为 sequence-level 分布；$\pi_{\text{mix}}$ 为 mixture policy，$\pi_{\text{teach}}$ 为 DPO-style 序列级教师 policy，$\pi_{\text{ref}}$ 为 reference policy，$R(x,y)$ 为 outcome reward，$|V|$ 为 vocabulary size。

### 2.1 Classical Knowledge Distillation
**①**Hinton et al. (2015) 令学生匹配温度软化分布 $p_T^{(\tau)}(y|x)=\frac{\exp(z_y/\tau)}{\sum_j \exp(z_j/\tau)}$，$z_y$ 为教师 logits，$\tau>1$ 暴露 "dark knowledge"，$\tau\to\infty$ 趋于 uniform 而丢失该结构。梯度
$$\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S}=\frac{1}{\tau}\big(p_i^S-p_i^T\big)$$
高温下 Taylor 展开 $\exp(z_i/\tau)\approx 1+z_i/\tau$ 配合零均值假设 $\sum_i z_i^S=\sum_i z_i^T=0$ 给出 $\frac{1}{|V|\tau^2}(z_i^S-z_i^T)$，即等价于 raw logits 的 MSE；$1/\tau^2$ 因子解释了 soft-target 项常被 upweight $\tau^2$。**实践中** LLM 蒸馏用 $\tau=1$ 或低到中温（Zhong et al., 2024）：大词表 $|V|>30{,}000$ 已产生丰富的非峰值概率，更高温度会放大教师**校准不佳**的尾部噪声。

**②**自回归 token-level 损失（Eq. 1）：
$$\mathcal{L}_{\text{Token-KD}}=\mathbb{E}_{x,y\sim\mathcal{D}}\Big[\sum_{t=1}^{|y|} D_{\text{KL}}\big(p_T(\cdot|x,y_{<t})\,\|\,p_\theta(\cdot|x,y_{<t})\big)\Big]$$
可算源于静态预计算 prefix，但优化 next-token accuracy 时**假设了无误差历史 $y_{<t}$**。Kim & Rush (2016) 的 Seq-KD 改为序列级 global KL $D_{\text{KL}}(P_T(y|x)\|P_\theta(y|x))=\sum_{y\in\mathcal{Y}}P_T(y|x)\log\frac{P_T(y|x)}{P_\theta(y|x)}$；因 $|\mathcal{Y}|=|V|^T$ 指数增长不可精确计算，以 beam-search 输出 $\hat{y}=\arg\max_y P_T(y|x)$ 处的 Dirac delta 近似，退化为 $\mathcal{L}_{\text{Seq-KD}}\approx-\log P_\theta(\hat{y}|x)$，丢弃教师 uncertainty 与 alternative modes。

**③历史 = 对四条经典假设的放松**：(1) shared vocabulary、(2) i.i.d. data、(3) static teacher、(4) off-policy training data。Seq-KD 放松 (2)，GKD（Agarwal et al., 2024）放松 (4)，DSKD（Zhang et al., 2025b）以 dual-space alignment 放松 (1)，G-OPD（Yang et al., 2026e）同时放松 (2)–(4) 并加 RL-augmented reward extrapolation 把学生推过教师 frontier。**④**不必要的放松会引入可避免的复杂度：vocabulary mismatch 对同族蒸馏无关、对跨族迁移严重；exposure bias 对短序列可忽略、对 multi-step reasoning 极成问题。

### 2.2 Off-Policy Exposure Bias
**②**off-policy 在数据集分布的 state 上做 token-level 匹配：
$$\mathcal{L}_{\text{Off-Policy}}=\mathbb{E}_{x,y\sim p_{\text{data}}}\Big[\sum_{t=1}^{|y|}D_{\text{KL}}\big(p_T(\cdot|x,y_{<t})\,\|\,p_\theta(\cdot|x,y_{<t})\big)\Big]$$
推理时学生按自身 policy 行动，诱导不同的 state visitation 分布 $d_{p_\theta}(s)$。**DAgger 定理**（Ross et al., 2011）：若 policy 在训练分布下以每步误差 $\epsilon$ 模仿 expert，则长度 $T$ 轨迹在 learner 自身 state visitation 下的期望总偏差为 $O(\epsilon T^2)$。

**③数字**：即便在独立误差假设下，需 10 个 reasoning steps、每步 95% 准确率的证明也只有 $(0.95)^{10}\approx 60\%$ 的 trajectory-level 正确率；DAgger bound 预测更差——off-policy 训练很少暴露学生于自身错误状态，每次失误把它推入下一步准确率更差的区域（distributional shift）。

**④限定**：原定理假设 **interactive expert** 在任意 state 给出 optimal action。white-box OPD 的"expert"只给出条件于学生 prefix 的 $p_T(y_t|\hat{y}_{<t})$；若学生幻觉出严重 OOD 的 prefix，教师条件分布**自身也校准不佳**（教师极少在该类输入上训练过），逼学生匹配该噪声分布违反 interactive imitation learning 核心假设，可能**使训练不稳而非恢复 $O(\epsilon T)$ 界**。Jeong (2026) 给出经验证据：naive OPD 在缺 stable teacher dynamics 时灾难性不稳定，无条件 token-level matching + 周期 hard-copy teacher reset 导致 abrupt output collapse，**单个 teacher-reset 事件上 KL divergence 从 2.637 降到 0.343**；这引出 §6.1 adaptive trust mechanisms 与 §7.2 stable teacher dynamics。

### 2.3 The Unified f-Divergence Framework
**②**广义 on-policy 目标把采样轨迹与局部匹配度量解耦：
$$\mathcal{L}_{\text{OPD}}(\theta)=\mathbb{E}_{y\sim\pi_{\text{mix}}}\Big[\sum_{t=1}^{|y|}D_f\big(p_T(\cdot|x,y_{<t}),\,p_\theta(\cdot|x,y_{<t})\big)\Big],\qquad D_f(P\|Q)=\mathbb{E}_{y\sim Q}\Big[f\Big(\tfrac{P(y)}{Q(y)}\Big)\Big]$$
$f:(0,\infty)\to\mathbb{R}$ 为凸 generator、$f(1)=0$；$f$ 决定 likelihood ratio $p_T(y)/p_\theta(y)$ 的隐式加权：

- **Forward KL**（$f(u)=u\log u$）：mode-covering（zero-avoiding），会在不同教师 mode 之间"桥接"并幻觉。
- **Reverse KL**（$f(u)=-\log u$）：mode-seeking（zero-forcing），把质量集中到教师 modes 的子集，可能忽略次要可接受输出。
- **JSD**（$f(u)=u\log u-(u+1)\log\frac{u+1}{2}$）：对称、有界，在两端间平滑插值。
- **$\alpha$-divergence**：在 Forward KL（$\alpha\to 1$）与 Reverse KL（$\alpha\to 0$）间连续插值。

任务几何：唯一正确答案的任务（mathematical proofs、code correctness）适合 Reverse KL 的 mode-seeking；多可接受输出的任务（creative writing、open-ended QA）适合 Forward KL 保多样性；JSD 是折衷但对两个极端都无理论最优性。**关键**：$D_f(P_T\|P_\theta)=\mathbb{E}_{y\sim p_\theta}[f(p_T(y)/p_\theta(y))]$，故可直接 on-policy 优化；采样分布 $p_\theta$ 就是被优化的 policy，梯度经 reparameterized samples 流动，**无需 importance weights 或 off-policy correction**，不额外引入方差。

**三方法映射**：
- **GKD**：显式插值 $\pi_{\text{mix}}=\lambda p_\theta+(1-\lambda)p_{\text{data}}$，对 $D_f$ 不敏感（实测 Forward KL / Reverse KL / JSD）；$\lambda\to 1$ 为纯 on-policy，$\lambda=0$ 退化为 off-policy KD，$\lambda\approx 0.5$ 兼顾 on-policy exposure 与 grounded prefix 的稳定性；实测 on-policy（$\lambda=1$）在各 divergence 下均优于 off-policy；JSD 在 WMT 翻译上最好（输出多样性居中，对称性自然平衡两种压力）。
- **MiniLLM**：$D_f=D_{\text{KL}}(p_\theta\|p_T)$（Reverse KL），$\pi_{\text{mix}}=(1-\alpha)p_\theta+\alpha p_T$、$\alpha=0.2$；因学生同时出现在期望与 log-ratio 中，改用 REINFORCE，把 $\log(p_T/p_\theta)$ 当 per-step reward。
- **DistiLLM**：在 student-generated prefixes 上算 token-level loss，用基于 validation loss 的 adaptive scheduler 保守采纳学生输出 + replay buffer（"adaptive off-policy"）；核心为 skew KLD：用 $\tilde{p}=\alpha p+(1-\alpha)q$ 计算 $D_{\text{KL}}(p\|\tilde{p})$（SKL）或 $D_{\text{KL}}(q\|\tilde{p})$（SRKL），$\tilde{p}(y)$ 界离零，避免 $p_\theta(y)\approx 0$ 或 $p_T(y)\approx 0$ 时的 zero-division 不稳定，且不用 policy gradient。

**③经验结论**：三种 divergence 在 summarization 与 instruction-following 上结果相当，**支持 exposure bias 假设**——任务几何不强烈偏向某一极端时，采样策略（on-policy vs. off-policy）比具体 divergence 选择更重要。**Tradeoff triangle**：GKD 取 generality（divergence-agnostic）+ simplicity（无 REINFORCE 方差），代价是 divergence 选择缺理论指引；MiniLLM 取 Reverse KL 的 mode-seeking 精度但继承 REINFORCE 高方差，需 reward baselines、length penalties 与精调 $\alpha$；DistiLLM 以构造上数值稳定的 mixture target 同时规避二者，代价是 replay buffer 与 adaptive scheduler 的额外超参。现代 OPD 因此是对三个交互维度的系统探索：trajectory sampling（$\pi_{\text{mix}}$）、divergence generator（$f$）、argument ordering。

### 2.4 Distillation Scaling Laws
与 pre-training 的 Chinchilla laws（Hoffmann et al., 2022）不同，蒸馏预算分配在 teacher inference、student rollouts、gradient updates 三轴之间，且有非平凡交互。Busbridge et al. (2025) 开创蒸馏专属 scaling 研究：teacher-student 关系服从与单纯 pre-training 定性不同的动力学；教师相对学生过强时出现 **capacity gap**——学生 loss 是 teacher cross-entropy 的 power law 并在两 regime 间转变：起初更强教师改善学生（更好校准的 next-token probabilities 提供更密梯度信号），超过临界 teacher capacity 后吸收能力**饱和**。该 gap 与 §2.3 的 divergence 选择直接相连：Forward KL 的 mode-covering 压力**加剧** gap（须把质量铺满超出自身容量的教师 modes），Reverse KL 的 mode-seeking 行为**部分缓解**（选择性匹配最高置信子集）。

**③④**尚无综合框架预测蒸馏质量如何随 teacher size $N_T$、student size $N_S$、data volume $D$、on-policy rollout budget $R$ 联合 scaling；Busbridge 式 scaling law 的自然扩展或形如
$$\text{Quality}\propto N_T^{\alpha}\cdot N_S^{\beta}\cdot D^{\gamma}\cdot R^{\delta}$$
但指数与 interaction terms 的确定仍开放（§9）。**rollout budget $R$ 是 on-policy 蒸馏独有的新 scaling 轴**（pre-training 与 off-policy 蒸馏都没有）：每个 rollout step 需学生的完整 forward pass，算力成本随 $R$ **线性增长**，而学习收益可能 **diminishing returns**。

---

## §3 Landscape and Method Selection

### 3.1 Method Landscape

**核心问题**：先给出全局三轴组织，把 OPD 设计空间拆成三个顺序设计决策，三轴各对应统一 OPD 目标的一个独立项：
1. **Objective function（§4）**：由 f-divergence 生成元 f 决定学生是 mode-seeking、mode-covering 还是 adaptive。
2. **Signal source（§5）**：由提供 pT(·|x, y<t) 的实体身份与访问级别决定。
3. **Training dynamics（§6）**：由采样策略及其调度决定，含插值系数 λ、prompt 难度课程、token 级加权。

信号源用**两级分类**：一级是 external teacher distillation（独立且通常更强的模型提供监督）vs. self-distillation（模型自产训练信号）；二级把 external teacher 按访问级别分 white-box（需教师完整输出分布）vs. black-box（只用生成文本或标量分数）。论文强调这是**方法学选择（教师身份）与部署约束（访问级别）分层**，不是并列同类。objective 轴与后两轴基本解耦（同一 divergence 可配多数调度与任一教师身份）；信号源与目标只部分解耦——black-box 抹掉 token 级分布，只能退化为 sequence-level 代理，故"三轴独立"**只用于组织设计空间，不暗示所有组合可实现**（如精确 token 级 forward KL 排除只有文本的 API 场景）。**时间线**：2023–2024 聚焦 objective 轴（forward KL / reverse KL / JSD 之争）→ 2025 年中转 signal architecture（self-distillation）→ 2025 末–2026 转 training dynamics（on-policy 采样独有的不稳定）；DeepSeek-V4、Qwen3 已三轴并用。

### Figure 1 分类树（按三轴组织的 OPD 方法学）

> ⚠️ 原文 Figure 1 每个 leaf 带"该小节方法总数 badge"，抽取文本中**未包含 badge 数字**。

- **§4 Objective Functions and Optimization**
  - 4.1 Fixed Divergence Objectives：GKD (Agarwal et al., 2024), MiniLLM (Gu et al., 2024), DistiLLM (Ko et al., 2024), DistiLLM-2 (Ko et al., 2025), KETCHUP (Fan et al., 2025), vOPD (Oh et al., 2026), AntiSD (Shen et al., 2026a), OPD+ (Zhao et al., 2026a), Bridging (Jiang & Ferraro, 2026), Decomposed-OPD (Yoon et al., 2026), Surgical-PT (Lin & Han, 2026), Distributional DAgger (Agrawal et al., 2026)
  - 4.2 Adaptive Divergence Objectives：EOPD (Jin et al., 2026b), AOPD (Jia et al., 2026), Position-Weighted OPSD (Liu et al., 2026d), Token-Teachability (Wang et al., 2026j), Lookahead Group Reward (Liu et al., 2026f), Trust-Region OPD (Xing et al., 2026)
  - 4.3 RL-Augmented Objectives：G-OPD (Yang et al., 2026e), RLKD (Xu et al., 2025b), KDRL (Xu et al., 2025a), RLAD (Zhang et al., 2026o), SuperCorrect (Yang et al., 2025b), SCoRe (Lyu et al., 2025), AlignDistil (Zhang et al., 2025a), X-KD (Cai & Yuan, 2026), TSD-KD (Kim & Baek, 2026), REOPOLD (Ko et al., 2026), CMDP-KD (Zimmer et al., 2025), CoDistill-GRPO (Kwon et al., 2026), dGRPO (Ramos et al., 2026), Sparse-to-Dense (Xu et al., 2026d), KEPO (Yang et al., 2026b), TGPO (Liu et al., 2026e), AMR-SD (Wei et al., 2026), OPPO (Li et al., 2026k), StepOPSD (Zhang et al., 2026j), Self-Evaluation (Zhang et al., 2026h)
- **§5 Signal Source and Teacher Architecture**
  - 5.1 External Teacher Distillation
    - 5.1.1 White-Box Logit Supervision
      - Same-Family：MAD-OPD (Wang et al., 2026c), MPD (Yang et al., 2026c), BRTS (Zhang et al., 2026e), Pair-In, Pair-Out (Tan et al., 2026)
      - Cross-Family：Veto (Jang et al., 2026), PromptKD (Kim et al., 2024), DSKD (Zhang et al., 2025b), CSD (Kim et al., 2026c), SimCT (Sun et al., 2026), DuDi (Payoungkhamdee et al., 2026)
    - 5.1.2 Black-Box and API-Constrained：Lion (Jiang et al., 2023), GAD (Ye et al., 2025), OVD (Xiong et al., 2026), LUFFY (Yan et al., 2025), DASD (Yan et al., 2026), DDT (Zhang et al., 2026f), PRISM (Wang et al., 2026g), ROPD (Fang et al., 2026), ORPO-Distill (Singh et al., 2025), OmniOPD (Zhou et al., 2026b)
  - 5.2 Self-Distillation
    - 5.2.1 Privileged Information：OPSD (Zhao et al., 2026b), GATES (Stein et al., 2026), π-Distill (Penaloza et al., 2026), OPCD (Ye et al., 2026b), OEL (Ye et al., 2026a), HDPO (Ding, 2026), CRISP (Sang et al., 2026), OPSDL (Zhang et al., 2026g), GUI-SD (Zhang et al., 2026i), MSD (Qin et al., 2026), PBSD (Yu et al., 2026c), TT-OPD (Jeong, 2026), VISD (Lin et al., 2026a), OPHSD (Zhao et al., 2026c), COPSD (Liu et al., 2026h), ATESD (Han et al., 2026b), TRACE (Wang et al., 2026e), AVSD (Nguyen et al., 2026), Skill-Cond. SD (Huang et al., 2026), Critique-Distill (Jin et al., 2026a), Constitutional Cross-SFT (Wen et al., 2026), World-Model PI (Zhou et al., 2026a), SD-PG (Liu et al., 2026g)
    - 5.2.2 Pure Self-Distillation：SDFT (Shenfeld et al., 2026), MTP-SD (Kirchenbauer et al., 2026), UniSD (Jin et al., 2026c), VPG (Chen et al., 2026e), Self-Sup OPD (Tan & Hong, 2026b), HINT-SD (Yeo et al., 2026), SD-Search (Ma et al., 2026), Vision-OPD (Yuan et al., 2026), It Takes Two (Park et al., 2026), TOD Proactivity (Zhang et al., 2026a), Search-E1 (Liang et al., 2026b), MAIGO (Zheng et al., 2026b), ROSD (Zhao et al., 2026d), Canonical-Ctx OPD (Lin et al., 2026b), COMAP (Liu et al., 2026i)
    - 5.2.3 External Feedback：SDPO (Hübötter et al., 2026), SD-ZERO (He et al., 2026), RLSD (Yang et al., 2026a), SRPO (Li et al., 2026c), RLTF (Song et al., 2026), CoPD (Gu et al., 2026), RLRT (Kim et al., 2026a), π-Play (Zhang et al., 2026l), PAINT (Tan & Hong, 2026a), CREDIT (Shen et al., 2026b), OGLS-SD (Yang et al., 2026f), VPD (Li et al., 2026i), RESD (Zhang et al., 2026n), SDAR (Lu et al., 2026), OPCT (Han et al., 2026a)
- **§6 Training Efficiency and Stabilization**
  - 6.1 Token and Sample Weighting：TIP (Xu et al., 2026c), SCOPE (Zheng et al., 2026a), SelecTKD (Huang et al., 2025), AdaSwitch (Peng et al., 2025), R-OPD (Fu et al., 2026b), SOD (Zhong et al., 2026a), MOPD (Yu et al., 2026b), EGRSD (Ke et al., 2026), Visual-Adv. OPD (Liu et al., 2026c), Filter-Then-Reweight (Li et al., 2026m), Less-is-More (Ziheng et al., 2026)
  - 6.2 Curriculum and Difficulty Adaptation：PACED (Xu et al., 2026b), Stable-OPD (Luo et al., 2026a), CaOPD (Zhang et al., 2026d), TCOD (Wang et al., 2026d), Uni-OPD (Hou et al., 2026), DeltaPrompts (Jung et al., 2026), f-OPD (Chen et al., 2026c), TR-Behav. Blend (Plyusov et al., 2026), Adaptive Refresh (Guo et al., 2026), CEI (Chen et al., 2026a)
  - 6.3 Compute Optimization：FOPD (Zhang et al., 2026b), Lightning-OPD (Wu et al., 2026a), SKD (Xu et al., 2025c), NPD (Rang et al., 2026), Prune-OPD (Yang et al., 2026g), EffOPD (Cai et al., 2026b), POPD/TOPD (Zhang et al., 2026k), SafeSteer (Li et al., 2026e)

> Figure 1 说明：leaf 只列代表性子集；方法按 **primary contribution** 归位，故"用 white-box logit 但在目标上创新"者（GKD、DistiLLM、EOPD、G-OPD）放 §4 而非 §5.1.1，§5.1.1 只收主要贡献即 white-box 信号接口本身者；对次要维度有贡献者（如 PAINT、PRISM 的隐式课程）只在相应小节正文讨论，不在树里重复。

### 3.2 Method Comparison Table

**Classification methodology**：每方法**只归入一个 primary category，判据是 core contribution dimension**——损失函数创新（G-OPD 的 KL-constrained RL 重构）→"Objective"；新信号架构（DSKD 的 dual-space projection）→"Signal"；训练稳定性/效率贡献（TIP 的 token 加权）→"Dynamics"。跨多维度者按**最独特贡献**归类，以降低单方法横跨多维度的归类歧义。

**Distributional observations**：截至 2026 年初，**self-distillation 是最大且增长最快的类别，其条目多数发表于 2025–2026**。这反映从奠基性的"哪个 KL 方向"转向 self-improvement 与 adaptive 替代方案，也与"模型自身 rollout 日益成为主要数据源（无论有无外部教师）"的大趋势一致。

Table 1 的 Category 列中，DistiLLM、MiniLLM 沿用上一行 "Objective (Fixed)"，RLKD 沿用 "Objective (RL)"，GAD 沿用 "Signal (Black-Box)"（原表为合并单元格）。

#### Table 1: Comparison of key OPD methods categorized by their primary contribution

| Method | Category | Core Contribution | Signal Source | Loss Granularity |
|---|---|---|---|---|
| GKD (Agarwal et al., 2024) | Objective (Fixed) | DAgger for LLMs, interpolation | White-Box Logits | Token |
| DistiLLM (Ko et al., 2024) | Objective (Fixed) | Skewed KL mixtures | White-Box Logits | Token |
| MiniLLM (Gu et al., 2024) | Objective (Fixed) | Sequence-level Reverse KL via RL | White-Box Logits | Sequence |
| EOPD (Jin et al., 2026b) | Objective (Adaptive) | Entropy-aware FKL+RKL blend | White-Box Logits | Token |
| G-OPD (Yang et al., 2026e) | Objective (RL) | OPD equivalence to KL-RL | White-Box Logits | Sequence/Token |
| RLKD (Xu et al., 2025b) | Objective (RL) | KD augmented with reward model | Reward Model | Sequence |
| DSKD (Zhang et al., 2025b) | Signal (White-Box) | Cross-architecture alignment | White-Box Logits | Token |
| Lion (Jiang et al., 2023) | Signal (Black-Box) | Verbal feedback curriculum | Black-Box API | Sequence |
| GAD (Ye et al., 2025) | Signal (Black-Box) | Adversarial distribution matching | Black-Box API | Sequence |
| OPSD (Zhao et al., 2026b) | Signal (Self) | Privileged Info (PI) conditioning | Self-Generated | Token |
| TIP (Xu et al., 2026c) | Dynamics (Weight) | Token importance via entropy & divergence | White-Box Logits | Token |
| PACED (Xu et al., 2026b) | Dynamics (Curriculum) | Beta-kernel curriculum pacing | White-Box Logits | Problem |

#### Table 2: Representative experimental configurations of on-policy distillation methods across different application domains

| Method | Teacher | Student | Training Signal | Key Benchmarks |
|---|---|---|---|---|
| **Mathematical Reasoning** | | | | |
| G-OPD (Yang et al., 2026e) | Qwen3-30B-A3B-Instruct | Qwen3-1.7B/4B | Dense logit + reward extrap. | AIME, HMMT, Code |
| OPSD (Zhao et al., 2026b) | Self (GT-conditioned) | Qwen3-1.7B/4B/8B-Inst | GT-PI self-distillation | AIME, HMMT |
| PACED (Xu et al., 2026b) | Qwen3-14B | Qwen3-8B | Beta-kernel curriculum | MATH-500, AIME |
| KDRL (Xu et al., 2025a) | Skywork-OR1-Math-7B | R1-Distill-Qwen-1.5B/DeepScaleR | Joint KD + RL | MATH500, Minerva, AIME24 |
| LUFFY (Yan et al., 2025) | DeepSeek-R1 (off-policy traces) | Qwen2.5-Math-7B/1.5B | Mixed-policy GRPO | AIME, AMC, MATH-500 |
| AOPD (Jia et al., 2026) | Qwen3-32B/8B (teacher) | Qwen3-8B/4B-Base | Asymmetric PG + FKL | AIME'24/'25, HMMT'25 |
| vOPD (Oh et al., 2026) | Qwen3-1.7B-Inst (Inst→Base) | Qwen3-1.7B/4B-Base, OLMo-3-7B | CV baseline + single-sample RKL | MATH-500, AIME'24/'25, Minerva |
| **Instruction Following & General** | | | | |
| GKD (Agarwal et al., 2024) | T5-XL (3B) | T5-Small/Base | On-policy FKL/RKL/JSD | XSum, WMT |
| DistiLLM (Ko et al., 2024) | GPT-2 XL (1.5B) | GPT-2 (124M) | Skewed KL | ROUGE-L, Dolly |
| AlignDistil (Zhang et al., 2025a) | Self (contrastive DPO reward) | Qwen2/2.5-1.5B-Inst | Token-level RLHF distillation | AlpacaEval, MT-Bench, Arena-Hard |
| **Industrial Scale** | | | | |
| Qwen3 (Yang et al., 2025a) | Qwen3-32B/235B-A22B | Qwen3-0.6B–14B | On-policy logit distillation | AIME, MATH, LiveCode |
| Gemma 2 (Gemma Team et al., 2024) | Gemma 2 27B | Gemma 2 9B / 2B | KD in pre-training | MMLU, HellaSwag |
| MiMo-V2-Flash (Xiaomi LLM-Core Team et al., 2026) | Domain specialists | MiMo-V2-Flash (309B/15B MoE) | Multi-teacher logit + reward | AIME, MATH, GPQA |
| KAT-Coder-V2 (Li et al., 2026b) | 5 domain specialists | Unified agentic coder | Specialize-then-Unify OPD | SWE-bench (79.6%) |
| Nemotron-Cascade 2 (Yang et al., 2026h) | Domain RL/SFT teachers | 30B MoE (3B active) | Cascade RL + domain OPD | IMO, IOI, ICPC |
| DeepSeek-V4 (DeepSeek-AI, 2026) | 10+ domain experts (1.6T) | DeepSeek-V4-Pro (1.6T MoE) | Full-vocab multi-teacher RKL | LiveCodeBench, HMMT, IMOAnswerBench |
| ORBIT (Liang et al., 2026a) | Multi-mode RL experts | DeepSeek-Distill-Qwen-1.5B, Qwen3-4B, Openmath-Nemotron-7B | Multi-teacher mode fusion | AIME, GPQA, MMLU-Pro |
| **Multimodal & Domain-Specific** | | | | |
| SCoRe (Lyu et al., 2025) | Qwen2.5-72B-Inst | Qwen2.5-7B/3B, Llama-3.1-8B | Earliest-error correction | 12 Agent benchmarks |
| VOLD (Bousselham et al., 2025) | Text-only teacher | VLM student | GRPO + token-level KL | MMMU-Pro, MathVista, LogicVista |
| CORD (Hu et al., 2026) | Text-mode LALM | Audio-mode LALM | Cross-modal RKL + GRPO | Audio QA/reasoning |
| SKD (Xu et al., 2025c) | Gemma-7B-IT, Qwen2-7B | Gemma-2B, Qwen2-0.5B | Block-verified on-policy KL | Translation, Dialogue |
| DP-OPD (Khadem et al., 2026) | GPT-2 Large 774M | DistilGPT-2 82M | GKD + DP-SGD (student only) | Yelp, BigPatent |
| Veto (Jang et al., 2026) | Qwen2-7B-IT | Qwen2-0.5B-IT | Geometric bridge KL | Reasoning, Generation |
| GUI-SD (Zhang et al., 2026i) | Self (visual PI) | Qwen3-VL-Inst-8B | Entropy-guided KL | ScreenSpot-v2/Pro, OSWorld |
| MAD-OPD (Wang et al., 2026c) | Multi-teacher debate | Qwen3/3.5 (1.7B–14B) | Confidence-weighted JSD/RKL | Agentic, Code |
| MSD (Qin et al., 2026) | Self (English CoT PI) | Qwen2.5-7B, Llama-3-8B | DPSW cross-lingual distill | Multilingual Safety |
| NPD (Rang et al., 2026) | Teacher (parallel prefill) | openPangu-Embedded-1B | Async CE+KD + Δ-IFD | GSM8K, MATH (8.1×speedup) |
| SimCT (Sun et al., 2026) | Qwen2.5-7B-Inst, Phi-4-mini | Phi-4-mini, Gemma-2-2B-IT | Multi-token continuation KL | Math, Code (cross-tokenizer) |
| Prune-OPD (Yang et al., 2026g) | R1-Distill-Qwen-7B, Qwen3-4B | R1-Distill-Qwen-1.5B, Qwen3-1.7B | Top-k overlap drift pruning | AMC, AIME, HMMT (37–68% time saved) |
| SOD (Zhong et al., 2026a) | Qwen3-4B | Qwen3-0.6B/1.7B | Step-level divergence reweighting | AIME'25 (26.13% @0.6B), SciCode |
| LiteGUI (Wu et al., 2026b) | Qwen3-VL-32B | 2B–3B scale agents | Guided OPD + dual-level GRPO | ScreenSpot-Pro, OS-World, Lite-Bench |
| VISD (Lin et al., 2026a) | Self (video-aware judge) | VideoLLM | Dir.-mag. decoupled advantage + RL | Open-o3-Video benchmarks |
| Uni-OPD (Hou et al., 2026) | Multi-teacher (LLM+MLLM) | Qwen3 (1.7B–4B), Qwen3-VL-4B | Dual-perspective OPD | Math, Code, MLLM (16 benchmarks) |
| TGPO (Liu et al., 2026e) | Teacher (directional PG) | NLP2CT/NEU students | Dense directional teacher RKL | NLP2CT, NEU translation |
| RWOPD (Zou et al., 2026) | CodeV-SVA teacher | Qwen2.5-Coder-7B | Verifier-reward-weighted FKL | NL2SVA |
| ProteinOPD (Zhang et al., 2026m) | Pretrained PLMs | Preference-specific students | OPD on student trajectories | Protein property design (8×speedup) |
| HyperEyes (Li et al., 2026d) | External teacher | HyperEyes-30B | Dual-grained efficiency RL | IMEB |
| DAgger-LLM (Li et al., 2026a) | GPT-4o | Qwen3-4B/8B | Turn-level interpolation SL | SWE-bench Verified |
| MOPD (Yu et al., 2026b) | Peer-conditioned teacher | Multi-rollout students | Success/failure group signals | Reasoning (CMU) |
| Local Collapse (Liu et al., 2026b) | Qwen3 family (strong) | Qwen3 family (weak) | BIC-truncated OPD | 5 in-domain + OOD benchmarks |
| **Self-Distillation** | | | | |
| CRISP (Sang et al., 2026) | Self (concise prompt) | Qwen3-8B/14B | Per-token RKL | AIME, MATH-500 |
| SD-ZERO (He et al., 2026) | Self (revision-conditioned) | Qwen3-4B-Inst, OLMo-3-7B | On-policy self-distill | AIME, MATH, Codeforces |
| π-Play (Zhang et al., 2026l) | Self (QCP-conditioned) | Qwen3-4B/8B | Multi-agent self-distill | NQ, TriviaQA, HotpotQA |
| UniSD (Jin et al., 2026c) | Self (EMA + multi-teacher agree.) | Qwen2.5-7B/0.5B–3B, Llama-3.1-8B, Gemma-3-4B | Agreement + EMA + contrastive + feat. match | ScienceQA, MBPP, CoS-E, ToolAlpaca |
| SDPO (Hübötter et al., 2026) | Self (textual feedback) | Qwen3-8B | Credit-assignment self-distill | Code, Math |
| SRPO (Li et al., 2026c) | Self (SDPO+GRPO) | Qwen3-8B | Sample-routed dual-objective | Chem, Phys, Bio, Mat, ToolUse |
| PBSD (Yu et al., 2026c) | Self (ctx-augmented ref.) | Qwen3-1.7B/4B/8B-Inst | Pref.-based reward-reg. KL | AIME'24/'25, HMMT'25, Tool Use |
| TT-OPD (Jeong, 2026) | Self (EMA + outcome hints) | Qwen3.5-9B | Turn-level truncated KL | Healthcare AI GYM (18 benchmarks) |
| Kim & Lee (2026) | Self (GT-conditioned) | Qwen3-8B, R1-Distill-7B, AceReason-7B | Correct-only OPSD | MATH-500, AIME'24/'25 |
| EGRSD (Ke et al., 2026) | Self (entropy-gated) | Qwen3-4B/8B (thinking mode) | Entropy-guided confidence gate | Math reasoning (accuracy-length frontier) |
| RESD (Zhang et al., 2026n) | Self (reflection-enhanced) | LLM agents | Failure reflection + playbook | Continual agentic learning |
| SDAR (Lu et al., 2026) | Self (gated RL+OPSD) | Qwen2.5/Qwen3 families | Sigmoid-gated OPSD auxiliary | ALFWorld, WebShop, SearchQA |
| GEAR (Li et al., 2026g) | Self (GT-conditioned) | LLM agents | Adaptive segment advantage | Agent tasks |
| **Black-Box** | | | | |
| GAD (Ye et al., 2025) | GPT-5-Chat (API only) | Qwen2.5-14B-Inst | Adversarial minimax | LMSYS-Chat, Dolly |
| Lion (Jiang et al., 2023) | ChatGPT (API only) | LLaMA-7B/13B | NL critique curriculum | BIG-Bench Hard, AGIEval |
| OVD (Xiong et al., 2026) | QwQ-32B / Env. Agent | Qwen2.5-3B, Llama-3.2-3B | Verbal scores (0–9) | Web QA, MATH |
| ROPD (Fang et al., 2026) | GPT-5.2, Qwen3-30B-A3B | Qwen3-4B, Gemma3-4B | Rubric-based semantic scoring | AIME'24/'25, HMMT'25, GPQA-D, IFEval |

### 3.3 Method Selection Considerations

最成功的部署在三个阶段都做**刻意选择**，而非只优化单一轴；三轴约束把可行配置空间大幅压缩。
- **Teacher access** 决定可行目标：white-box logit→精确 token 级 divergence（DistiLLM、G-OPD、跨 tokenizer 的 DSKD）；API-only→生成文本或标量分数（OVD、GAD、LUFFY）；无外部教师→privileged information（OPSD、GATES）或 verifier 引导自我改进（SD-ZERO）。
- **Task**：推理/数学用 mode-seeking（reverse KL 或 RL-augmented）把质量集中到正确解；开放生成用 mode-covering（forward KL 或 entropy-aware）保多样性；指令跟随居中，**适度 on-policy 混合（λ≥0.5）配 JSD** 给强质量–算力折衷（GKD）；多步 agentic 需 sequence-level 目标 + trajectory-aware credit assignment（SCoRe、RLAD）。
- **Compute**：受约束（< 500 GPU-hours）用前缀截断 rollout 与 offline 缓存；中等规模可完整 on-policy + adaptive divergence（EOPD、AOPD）；大规模（> 5000 GPU-hours）用多阶段流水线：off-policy 预热 → on-policy 精修 → 可选 RL 探索（Qwen3 即如此报告）。
- **Stability**：长度膨胀用 reference divergence 约束与 rollout 混合（Stable-OPD）；**flawed prefix trap** 用 token 级可靠性过滤（TIP、SCOPE）；teacher–student thinking-pattern mismatch 先 off-policy cold-start 再转 on-policy。

## §4 Objective Functions and Optimization

目标函数决定学生每步优化什么数学量，支配梯度几何、mode coverage 与性能折衷。演进：**fixed divergence（全位置统一单一度量）→ adaptive divergence（按 token 局部分布几何调制）→ RL-augmented objectives（引入教师输出分布之外的 reward）**。

### 4.1 Fixed Divergence Objectives

**核心问题与形式化**：GKD 借 DAgger 范式建立 OPD 规范框架——从 $\pi_{\text{mix}}=\lambda p_\theta+(1-\lambda)p_{\text{data}}$ 采样轨迹，每 token 施加标准 f-divergence（Forward KL / Reverse KL / JSD）；λ 从完全 off-policy（λ=0）插值到完全 on-policy（λ=1），指令跟随与摘要上**λ≥0.5 优于纯 off-policy**。数值不稳（学生采到教师近零概率 token 时 forward KL 无界、reverse KL 完全忽略该区形成盲点）由 **DistiLLM 的 Skewed KL (SKL)** 解决：把 KL 的 $q_\theta$ 换成 $\tilde p=\alpha p_T+(1-\alpha)p_\theta$，

$$L_{\text{SKL}}=\mathbb{E}_{(x,y)\sim D_{\text{mix}}}\left[\sum_{t=1}^{|y|}D_{\text{KL}}\left(p_T(\cdot|y_{<t})\,\Big\|\,\alpha p_T(\cdot|y_{<t})+(1-\alpha)p_\theta(\cdot|y_{<t})\right)\right]$$

偏斜使目标密度下界为 $(1-\alpha)p_\theta$，无需 RL 策略梯度即免梯度爆炸；另有 mode-seeking 用的 **SRKL**。**DistiLLM-2** 认为 teacher-generated 与 student-generated 数据信号性质不同，用**非对称**目标（教师数据 Forward SKL / 学生数据 Reverse SRKL），说明**合适 divergence 依赖数据来源**。

**根本局限（引出 §4.2）**：KL 方向之选是**对所有 token 位置统一的全局决策**，但教师序列内分布结构差异极大——数学算符 token 宜 mode-seeking，连词/填充词（数十个近等概率 token）宜 mode-covering；单一 divergence 必然一处浪费梯度、一处丢弃覆盖信息。Figure 2 用双峰教师分布刻画：forward KL 是 mode-covering/zero-avoiding（在模式间 hallucination 区投放大量概率），reverse KL 是 mode-seeking/zero-forcing（锐化到单峰、丢掉另一模式）。

**Sequence-level 目标**：MiniLLM 把 reverse KL 提升到整序列（$p_\theta$ 出现在采样期望内部），由策略梯度定理得（Gu et al. 2024, Eq. 2）：

$$\nabla_\theta L_{\text{MiniLLM}}=-\mathbb{E}_{y\sim p_\theta}\left[\sum_{t=1}^{|y|}(R_t-1)\nabla_\theta\log p_\theta(y_t|y_{<t})\right],\quad R_t=\sum_{t'=t}^{|y|}\log\frac{p_T(y_{t'}|y_{<t'})}{p_\theta(y_{t'}|y_{<t'})}$$

−1 来自 reverse-KL 的 $-\log p_\theta(y|x)$ 熵项；**最小化 sequence-level reverse KL 等价于以 $r_t=\log[p_T(y_t|y_{<t})/p_\theta(y_t|y_{<t})]$ 加熵诱导常数做策略梯度 RL**。**KETCHUP** 用 K-step Bellman return 替代累积回报 $G_t$，以 K 的幂速率降方差且保持 sequence-level 目标。

**两个关键 tradeoff / 定量结论**：①**token-level vs. sequence-level**：token-level（GKD、DistiLLM）每位置给 |V| 维梯度，低方差但教师 token 分布校准差时**有偏**（flawed prefix trap）；sequence-level（MiniLLM、KETCHUP）无偏但单样本 REINFORCE 方差高。故 sequence-level 蒸馏与 RL-augmented **不是不同范式，而是同一连续统的两端**，由 reward 来源参数化（纯教师 ←→ 纯任务 reward），G-OPD、RLKD 居中。②**full-vocabulary vs. sampled-token KL**：只在采样 token 上算 $\log p_\theta-\log p_T$ 轻量但高方差、丢弃非采样 token 信息；DeepSeek-V4 报告全词表精确 $D_{\text{KL}}(p_\theta(\cdot|y_{<t})\|p_T(\cdot|y_{<t}))$ 得**更稳梯度与更忠实迁移**（10+ 万亿参数教师、>100K 词表，靠缓存教师 hidden state 经预测头实时重建 logit 实现）；**Lightning-OPD** 在 teacher consistency 下预计算 SFT rollout 的教师 log-prob，基本省掉教师前向，**4.0× speedup**。

**其余固定族方法**（OPD+、Bridging、Decomposed-OPD、Surgical-PT、Distributional DAgger、AntiSD、vOPD）：归类与逐条主张见 Table 3。其中两个带强定量结果：**AntiSD** 以 **ascent** 翻转 per-token 符号 boost deliberation token，4B–30B 五模型上 **2–10× 更少训练步**达 GRPO 级、等预算数学推理**超 GRPO 至多 11.5 分**；**vOPD** 以 control variate 无偏基线（$V_{p_\theta}(c_t)=-D_{\text{KL}}(p_\theta(\cdot|c_t)\|p_T(\cdot|c_t))$，无需 critic）在 Qwen3-1.7B/4B 六基准上 **平均 +3%**（MATH500 至多 **+6.2%**），追平全词表 OPD 而**墙钟至多省 57.7%**。OPD+ 另用 gradient-faithful 权重 $w_f(u)=-f(u)+uf'(u)$ 补二阶曲率。

#### Table 3: OPD methods: Fixed divergence objectives

| Method | Year | Category | Divergence/Objective | Signal | Granularity | Key Innovation |
|---|---|---|---|---|---|---|
| GKD (Agarwal et al., 2024) | 2023 | Objective | FKL / RKL / JSD | White-box | Token | Unified OPD framework, DAgger for LLMs |
| MiniLLM (Gu et al., 2024) | 2023 | Objective | Reverse KL | White-box | Sequence | REINFORCE with teacher as reward |
| DistiLLM (Ko et al., 2024) | 2024 | Objective | Skew KL (SKL+SRKL) | White-box | Token | α-smoothing for KL stability |
| DistiLLM-2 (Ko et al., 2025) | 2025 | Objective | Contrastive SKL/SRKL | White-box | Token | Asymmetric loss per data source |
| KETCHUP (Fan et al., 2025) | 2025 | Objective | Reverse KL | White-box | Sequence | K-step Bellman return estimation |
| vOPD (Oh et al., 2026) | 2026 | Objective | Reverse KL + CV baseline | White-box | Token | Control variate variance reduction |
| AntiSD (Shen et al., 2026a) | 2026 | Objective | Divergence ascent | Self (PI) | Token | Deliberation-token boosting via PMI sign flip |
| OPD+ (Zhao et al., 2026a) | 2026 | Objective | f-div gradient-faithful | White-box | Token | Curvature-aware advantage $w_f$ |
| Bridging (Jiang & Ferraro, 2026) | 2026 | Objective | Reverse KL + OT prefix signal | White-box | Token/Span | Near-future trajectory alignment |
| Decomposed-OPD (Yoon et al., 2026) | 2026 | Objective | Reverse KL + VGS | White-box | Token | Visual-language gradient steering |
| Surgical-PT (Lin & Han, 2026) | 2026 | Objective | Proximal KL-constrained | Oracle | Token/Sequence | Retention via KL-constrained reward |
| Distributional DAgger (Agrawal et al., 2026) | 2026 | Objective | Forward CE + REINFORCE | Self (EMA, feedback) | Token | DAgger with teacher-distribution feedback |

### 4.2 Adaptive Divergence Objectives

**核心思想**：按**局部分布几何**选择/插值 divergence——教师低熵（确定）时用 mode-seeking reverse KL 保精度，高熵（不确定）时用 mode-covering forward KL 保多样性。**EOPD** 以教师 per-token 熵 $H(p_T(\cdot|y_{<t}))$ 门控，基础处处为 reverse KL、高熵 token 上**加性**激活 forward KL：

$$L_{EA}=\mathbb{E}_{y\sim p_\theta}\left[\sum_{t=1}^{|y|}D_{\text{KL}}(p_\theta\|p_T)+\mathbb{I}[H_t>\tau_H]\cdot D_{\text{KL}}(p_T\|p_\theta)\right]$$

在六个竞赛数学推理基准上优于 fixed 基线；因只条件于教师不确定度（边缘属性），路由信号**独立于学生质量**，适合学生分布尚不可靠的早期训练。**AOPD** 诊断非正 advantage 区（$A_t\le 0$）三类病理——无界负 advantage 的重尾方差、零 advantage 处梯度消失（浪费教师信号）、exploration black hole（压制错误 token 只按学生自身先验重分配质量，无法提升低概率正确项），对策是 **token 级切换目标类**：正 advantage 保留 policy gradient exploitation，非正 token 换为教师 top-K 支撑上的局部 forward-KL guidance；阈值 τ 默认 0（教师与学生概率差），**τ=−1 退化标准 OPD，τ=1 退化 GKD**；在 AIME'24/'25、HMMT 上较标准 on-policy 基线**平均 +4.09（强初始化）/ +8.34（弱初始化）**，弱初始化差距更大印证 exploration black hole 是学生远离教师时的主导失效模式。

**其余自适应方法**（Position-Weighted OPSD、Token-Teachability、Direction-Adaptive SD、Lookahead Group Reward、Trust-Region OPD、RAFT、GNDPO，以及属 self-distillation reward calibration 的 CREDIT / OGLS-SD）：归类与逐条主张见 Table 4 与 §5.2.3。

**结论**：领域已从"用哪个 divergence"转向**"在哪个位置用哪个 divergence"**；EOPD/AOPD 证据一致表明**无单一 fixed divergence 普遍占优**，最简单的自适应路由（硬熵阈值）常胜最优 fixed 方案。但自适应与 fixed **共享结构性局限：二者都在优化到教师分布的距离，任何 f-divergence 下的完美优化也只得到等于教师、几乎无法超越教师的学生**——教师本身出错或缺少学生可发现的推理路径时，divergence minimization 更像天花板而非地板。

#### Table 4: OPD methods: Adaptive divergence objectives

| Method | Year | Category | Divergence/Objective | Signal | Granularity | Key Innovation |
|---|---|---|---|---|---|---|
| EOPD (Jin et al., 2026b) | 2026 | Objective | Continuous FKL+RKL blend | White-box | Token | Entropy-aware on-policy distillation |
| AOPD (Jia et al., 2026) | 2026 | Objective | Asymmetric PG/FKL switch | White-box | Token | Advantage-sign token-level switching |
| CREDIT (Shen et al., 2026b) | 2026 | Self (EF) | Contrastive pMI decomposition | Self (contrastive) | Token | Input-specific credit for OPSD reward |
| OGLS-SD (Yang et al., 2026f) | 2026 | Self (EF) | Outcome-guided logit steering | Self (outcome) | Token | Outcome-based teacher calibration |
| Position-Weighted OPSD (Liu et al., 2026d) | 2026 | Objective | Position-aware reliability | Self (PI) | Token | Per-position reliability score |
| Direction-Adaptive SD (Zhang et al., 2026c) | 2026 | Objective | Entropy-routed polarity | Self (PI) | Token | High-entropy teacher reversal |
| Token-Teachability (Wang et al., 2026j) | 2026 | Objective | Teachability-gated KL | White-box | Token | Fixed-context learnability diagnostic |
| Lookahead Group Reward (Liu et al., 2026f) | 2026 | Objective | RKL + look-ahead group reward | White-box | Token | One-step-ahead teacher confidence |
| RAFT (Li et al., 2026l) | 2026 | Objective | Top-K temperature distill | Self | Token | Data refinement + EMA loss balance |
| Trust-Region OPD (Xing et al., 2026) | 2026 | Objective | Adaptive trust-region KL | White-box | Token | Speculative-decoding-ratio gating |
| GNDPO (Hao et al., 2026) | 2026 | Objective | Global batch-normalized KL | White-box | Token | Batch-relative KL normalization for MLLM stability |

### 4.3 RL-Augmented Objectives

**核心问题**：上一节的能力天花板催生此类目标——注入可把学生引向**教师自己不太可能产出**之解的外部性能信号。**OPD as KL-constrained RL**：G-OPD 形式化标准 OPD 与 dense KL-constrained RL 的等价（token 级 reward = 教师与参考策略的 log 概率比，并对参考策略加 KL 惩罚）：

$$\max_\theta\ \mathbb{E}_{y\sim p_\theta}\left[\sum_{t=1}^{|y|}\alpha\log\frac{p_T(y_t|y_{<t})}{p_{\text{ref}}(y_t|y_{<t})}-D_{\text{KL}}\left(p_\theta(\cdot|y_{<t})\|p_{\text{ref}}(\cdot|y_{<t})\right)\right]$$

α=1 退化为标准 reverse KL；**α>1 迫使学生 extrapolate 超出教师概率质量**，即 **Reward Extrapolation**——在 $p_T(y|x)$ 低但 outcome reward 仍高的区域发现未见过且结构合理的推理路径。多教师下 **ExOPD** 产出超越**所有同规模领域教师**的统一学生，使 OPD 成为**能力放大机制**而非有损压缩。据此，GKD 的 token 级 KL、MiniLLM 的 sequence-level reverse KL、G-OPD 的 reward-augmented 形式都是**不同 reward 定义与约束强度下 KL-constrained policy optimization 的特例**。

**代表方法与关键定量（逐条主张见 Table 5）**：**REOPOLD** 以师生 log-likelihood ratio 为 token reward，用 mixture-based reward clipping、熵基动态采样、统一 exploration-to-refinement 调度稳定训练，**样本效率比近期 RL 方法高 6.7–12×**，7B 学生视觉推理追平 32B 教师且**约 3.3× 推理加速**。**RLKD** 弃 token 级 KL，改用 **GSRM** 把师生推理路径解析为 meta-reasoning/solving 步骤并打结构对齐分，$R_{\text{total}}=\alpha R_{\text{GSRM}}+(1-\alpha)R_{\text{outcome}}$，**仅 0.1% 数据**在纯 RL 下即超标准 SFT-RL 流水线。**CoDistill-GRPO** 双向共蒸馏（小模型用大模型分布的 KD reward 作密集监督，大模型用小模型 rollout + importance reweighting 省算力），Qwen2.5-Math-1.5B 上 Minerva **32%（比 base +11.6、比标准 GRPO +6.0）**，**约 18% speedup**。**TGPO** 把教师 per-token likelihood ratio 当方向信号嵌入 policy gradient 内部，免去 joint KD+RL 的平衡超参，在 MATH-500 上同时优于 standalone OPD 与 GRPO。**dGRPO** 面向**至多 128K tokens** 长上下文，在 GRPO 上加加性 dense OPD 项并配 LongBlocks 数据。**Sparse-to-Dense** 提出 reward-density 原则（稀疏序列 reward 用于最强调模型的教师侧 RL 探索，密集 token reward 用于压缩进小模型），三阶段流水线（教师 GRPO → forward-KL 预热 + OPD 桥 → 可选学生侧 GRPO）下，RL 改进的 8B 教师经桥**优于直接学生侧 GRPO（MATH 79.3% vs. 75.9%，AIME 2024 25.2 vs. 19.8）**，桥还使后续学生侧 GRPO **把 MATH 从 75.4% 提到 78.5%**，该教师质量排序在 Llama-3.1-8B-Instruct + Llama-3.3-70B-Instruct 教师上复现。

**其余方法归四类，归类与逐条主张见 Table 5**：**joint vs. sequential**（KDRL 加 on-policy KL 正则、CMDP-KD 用 constrained MDP 硬 KL 预算、RLAD 的 TRRD 只在教师信号改善更新时跟随）、**偏好式隐式 RL**（AlignDistil、OVD、PBSD、RWOPD；DPO 闭式解 $\pi^*(y|x)\propto\pi_{\text{ref}}(y|x)\exp(r(y,x)/\beta)$ 把教师 log 概率当隐式 reward 即得 GKD λ-mixing 的几何混合分布）、**细粒度 credit**（SuperCorrect、SCoRe 只纠正最早那个错误、AMR-SD、OPPO、StepOPSD、Self-Eval OPD、RLCSD）、**环境重构**（X-KD 用 AVRIL 重构教师训练环境、TSD-KD）。

**总结（progression）**：fixed divergence 稳定、易理解但受教师能力天花板限制；adaptive divergence 靠匹配损失景观局部几何提效；RL-augmented 提供绕过天花板的路径，代价是额外方差与不稳定。指令跟随（教师质量高）用 fixed 或 adaptive 常足够；推理（期望学生超越教师）RL 增强有吸引力；教师训练过程可访问时 environment reconstruction 是更完整的知识转移形式。

#### Table 5: OPD methods: RL-augmented objectives

| Method | Year | Category | Divergence/Objective | Signal | Granularity | Key Innovation |
|---|---|---|---|---|---|---|
| G-OPD (Yang et al., 2026e) | 2026 | Objective | KL-Constrained RL | White-box | Token/Seq | OPD ≡ dense KL-RL, reward extrapolation |
| RLKD (Xu et al., 2025b) | 2025 | Objective | KD + Reward Model | Reward Model | Sequence | Generative Structure Reward Model (GSRM) |
| KDRL (Xu et al., 2025a) | 2025 | Objective | Joint KD + RL | White-box | Token/Seq | On-policy KL regularizer during RL |
| RLAD (Zhang et al., 2026o) | 2026 | Objective | Trust Region Ratio | White-box | Token/Seq | PPO-style selective teacher following |
| AlignDistil (Zhang et al., 2025a) | 2025 | Objective | DPO + Reverse DPO | White-box | Token | Token-level RLHF via contrastive DPO reward |
| SuperCorrect (Yang et al., 2025b) | 2024 | Objective | Template + Cross-DPO | White-box | Sequence | Hierarchical thought scaffolding |
| SCoRe (Lyu et al., 2025) | 2025 | Objective | Earliest-error correction | White-box | Sequence | Root-cause error targeting |
| CMDP-KD (Zimmer et al., 2025) | 2025 | Objective | Constrained KL | White-box | Sequence | Unaugmented CMDP with hard KL budget |
| X-KD (Cai & Yuan, 2026) | 2026 | Objective | Inverse RL + KL | White-box | Token/Seq | AVRIL-based reward recovery distillation |
| TSD-KD (Kim & Baek, 2026) | 2026 | Objective | Indirect + Direct KD | White-box | Sequence | Indirect re-ranking + selective direct KL |
| REOPOLD (Ko et al., 2026) | 2026 | Objective | Log-ratio token reward | White-box | Token | Mixture reward clipping + entropy-dynamic sampling |
| CoDistill-GRPO (Kwon et al., 2026) | 2026 | Objective | GRPO + KD reward | White-box | Sequence | Bidirectional co-distillation, 18% speedup |
| dGRPO (Ramos et al., 2026) | 2026 | Objective | GRPO + dense OPD | White-box | Token/Seq | Long-context reasoning with LongBlocks data |
| Sparse-to-Dense (Xu et al., 2026d) | 2026 | Objective | Staged RL→OPD→RL | White-box | Token/Seq | Reward-density allocation principle |
| TGPO (Liu et al., 2026e) | 2026 | Objective | Directional teacher PG | White-box | Token | Dense directional teacher guidance on RKL |
| RWOPD (Zou et al., 2026) | 2026 | Objective | Verifier-weighted FKL | White-box | Sequence | Verifier-reward-weighted OPD for NL2SVA |
| AMR-SD (Wei et al., 2026) | 2026 | Objective | Asymmetric meta-reflective | Self | Token | Meta-reflection self-teacher in RLVR |
| OPPO (Li et al., 2026k) | 2026 | Objective | Bayesian value recursion | Oracle (PI) | Token | Likelihood-ratio per-token credit |
| StepOPSD (Zhang et al., 2026j) | 2026 | Objective | Step-aware preference + KL | Self (hindsight) | Step/Token | Dual-granularity for multi-turn agents |
| Self-Eval OPD (Zhang et al., 2026h) | 2026 | Objective | Cyclic GRPO + masked judge KD | External judge | Token | Self-evaluation channel calibration |
| RLCSD (Pan et al., 2026) | 2026 | Objective | Contrastive RKL in GRPO | White-box | Token | Correct-hint teacher pull, wrong-hint push in two-path distillation |

---

## §5 Signal Source and Teacher Architecture（信号来源与教师架构）

目标函数固定后（§4），下一个设计轴是**教师信号从哪来**。本章两层组织：第一层分 **external teacher distillation**（独立更强模型监督）与 **self-distillation**（模型利用内部不对称性自产信号）；第二层在外部教师内分 **white-box**（可访问教师完整输出分布）与 **black-box**（仅有生成文本或 scalar score）。两层分别刻画教师**身份**（外部 vs 自身）与**访问级别**；二者不独立（访问级别约束可用目标函数），但分开可避免把部署约束混作方法论选择。演化上访问级别维度先动（GKD、DistiLLM 假定完整 white-box；GPT-4/Claude 等专有 API 迫使 black-box 从纯文本抽知识），身份维度后动（self-distillation 移除外部教师）；两者都是**自主性上升、信号密度下降**，且两个 trade-off 基本独立。

复杂推理是 OPD 增益最显著的场景，因推理 **path-dependent**（Agarwal et al., 2024）："assume for contradiction..." 与 "we proceed by induction..." 的后续 token 完全不同，off-policy 固定语料只覆盖有限解路径。故有

$$L_{\text{CoT-OPD}}(\theta)=\mathbb{E}_{x\sim D,\,r_S\sim P_\theta(\cdot|x)}\Big[\sum_{t=1}^{|r_S|} D_{\text{KL}}\big(p_T(\cdot|x,r_S^{<t})\,\|\,p_\theta(\cdot|x,r_S^{<t})\big)\Big]$$

**Forward KL** 鼓励 mode-covering，使学生在自己 prefix 上收到纠错反馈。rationale 迁移优于 answer-only 由 Hsieh et al. (2023) 在 off-policy 下确立，on-policy 生成进一步降低 exposure bias（Agarwal et al., 2024）。

### 5.1 External Teacher Distillation

假定有强于学生的独立模型在学生的 on-policy rollout 上给纠正信号；两小节按**信号密度**分：完整 logit（5.1.1）最丰富但需与教师共置，API 受限（5.1.2）以密度换部署灵活性。

#### 5.1.1 White-Box Logit Supervision

**核心问题**：white-box 给最密信号——每 decoding step 上 |V| 个 token 的**完整概率分布**，含错误 token 间的相对概率（Hinton 的 **"dark knowledge"**），可直接做 token-level divergence matching。white-box 是本文多数方法的隐含假设，本节只收**主要贡献在信号接口本身**者（cross-tokenizer alignment、full-vocabulary compression、teacher-rollout selection、confidence-head construction）。同族共享 tokenizer 时直接算 per-token KL，无对齐开销；教师 forward 开销可缓存整条 rollout 的 logits，同架构下约为 student-only RL 的 **1.5–2×**（Agarwal et al., 2024）。代表方法：

- **DSKD**（Zhang et al., 2025b）：dual-space KD，跨 vocabulary 投影 + 精确 token alignment：$L_{\text{DSKD}}=D_{\text{KL}}(P_{T\to S}\|P_S)+D_{\text{KL}}(P_T\|P_{S\to T})$。跨族时 vocabulary 不匹配下直接 KL 无定义。
- **SimCT**（Sun et al., 2026）：在两种 tokenizer 都能实现的短 multi-token continuation 单元上比较（放大粒度而非投影），三对异构模型上胜 shared-vocabulary OPD，**词汇分歧最大处增益最显著**；**CSD** 用 continuous score matching、**BTB** 学 token-to-token 映射桥、**OPRD** 加 hidden-state alignment、**DuDi** 用 cross-lingual verbalizer。
- **MPD**（Yang et al., 2026c）：mixed-policy，教师把学生轨迹改写成更简洁推理链，Qwen3-1.7B 上 token 用量最多降 **27.1%** 且多基准提升。
- **BRTS**（Zhang et al., 2026e）：Best-of-N teacher selection，按**先正确性后对齐度**择优，AIME 2024/2025、AMC 2023 上优于标准 OPD，难题增益更明显。
- **SelecTKD**（propose-and-verify，按师生一致度路由）、**AdaSwitch**（按 student prefix reliability 在 exploration/guidance 间切换）——两准则可加性，并与 entropy-based divergence routing（EOPD）正交。
- **Veto**（post-hoc，logit 空间 geometric bridge，β 作 Adaptive Gradient Veto + Decisiveness Knob）与 **PromptKD**（pre-hoc，前置 learnable prompt tokens，仅占教师参数 **0.0007%**）解决教师 **overconfidence**（对超出学生可达空间的 token 赋极端概率 → 目标"既精确又无用"），二者可组合；**Pair-In, Pair-Out** 用 full-backbone 教师校准 MTP confidence head。

#### 5.1.2 Black-Box and API-Constrained Distillation

只有 API 时学生只见教师 top-1 文本，信息瓶颈迫使方法转 **sequence level**，但纯文本观测仍可恢复教师相当部分知识。代表方法：

- **Distilling Step-by-Step**（Hsieh et al., 2023）：奠基性方法，抽教师的中间 rationales（技术上 off-policy）；多任务目标使 **770M 学生超过 540B 教师**，参数少 **700×** 以上，样本平均少 50%（部分任务最多减 85%）。
- **GAD**（Ye et al., 2025）：对抗分布匹配 $\max_G\min_D V(G,D)=\mathbb{E}_{(x,y_T)\sim T}[-\log\sigma(D(y_T)-D(G(x)))]$，判别器由学生初始化加 scalar head、用 Bradley-Terry 偏好模型，生成器以判别器分数为 evolving reward、用 **GRPO** 优化；继承对抗训练不稳（判别器可过拟合风格线索）。
- **Lion**（Jiang et al., 2023）：verbal feedback 三段循环（imitation / discrimination / generation），Lion-13B 仅 **70K** 样本在 BIG-Bench Hard、AGIEval 上竞争。
- **OVD**（Xiong et al., 2026）：用离散 **verbal scores（0–9）** 做 trajectory matching，免 token 对齐、免 logit 存储；单样本训练下 web QA **+12.9%** 绝对 EM、math **+25.7%**——on-policy exploration 已供大部分信号，教师更像**在学生轨迹中挑选**。
- **DASD**（每 prompt 多条教师轨迹做序列级对齐，仅 **448K** 样本）、**DDT**（RL+KD 理论框架，解释 RL 因 on-policy 数据生成而泛化更好）。
- **PRISM**（Wang et al., 2026g）：在 SFT 与 RLVR 间插入 OPD **pre-alignment**，用 **MoE discriminator**（perception/reasoning 双专家头）给解耦纠错信号，教师为 Gemini 3 Flash；Qwen3-VL 4B/8B 上预对齐即比 SFT→RLVR 高 **+4.4 / +6.0** 分。
- **ROPD**（Fang et al., 2026）：结构化 **rubric** 取代 logits，Rubricator 归纳 prompt 专属 rubric、Verifier 打分驱动 on-policy 优化，可离线生成；约 **10×** 样本效率；AIME 2025 thinking 下 Qwen3-4B 学生 **68.75%** 超其 GPT-5.2 教师 **67.08%**（学生可超教师）。
- **ORPO-Distill**（preference optimization 替代 token KL，mixed-policy 负采样 φ=0.5，跨架构 Llama→Qwen 无 logit）、**OmniOPD**（chunk 级 Monte Carlo 验证 + Dirichlet-Multinomial Bayesian smoother，被审计 chunk 用教师分布、其余用 trust-region KL anchor，以 black-box 查询恢复大部分 white-box 性能）、**LUFFY**（GRPO 扩为 mixed-policy 纳入 off-policy 轨迹，importance sampling 重加权避免只抄高概率 token，比 RLVR 平均 **+6.4**）。

**小结**：white-box 达到最紧对齐但须共置教师，black-box 换部署灵活性。未解张力是——**最强教师（专有前沿模型）通常只能经 API 触达**，恰是最难拿到 dense per-token 信号的场景。

### 5.2 Self-Distillation

完全移除外部教师，利用模型**自身内部不对称性**构造信号。三族按不对称性来源划分，也构成"信号从哪来"的分类骨架：**privileged information（PI）** 的信号来自"额外上下文条件化出的更强 self-teacher"（同模型、多输入）；**pure self-distillation** 的信号来自"同输入下多样输出"的**方差本身**；**external feedback** 的信号来自模型之外（verifier/环境协作演化），用于给自产信号定方向、防塌缩。**范围边界**：只收训练信号来自"同一模型不同 view 间分布对齐"（conditioned vs unconditioned、frozen vs updated、temperature-sampled vs greedy）者，排除博弈式 self-play **SPIN**、**IRIS**（核心是 checkpoint 间对抗竞争收敛到 Nash 均衡，属 §9）。

#### 5.2.1 Privileged Information

用训练时额外上下文条件化同一模型，得更强 policy 去监督无条件版本。OPSD（Zhao et al., 2026b）是 GT-as-PI 范式：

$$L_{\text{OPSD}}(\theta)=\mathbb{E}_{(x,y^\star)\sim S}\,\mathbb{E}_{\hat y\sim p_\theta(\cdot|x)}\Big[\sum_{n=1}^{|\hat y|} D\big(p_\theta(\cdot|x,y^\star,\hat y_{<n})\,\|\,p_\theta(\cdot|x,\hat y_{<n})\big)\Big]$$

各方法**主要差别只在"什么充当上下文"**（GT 答案、检索文档、spatial mask、结构化 judge 反馈、未来观测、弱外部 critique），机制始终是"PI-conditioned self-teacher 在学生 rollout 上的 token-level matching"。**关键区分**：可跨实例聚合的规则（system prompt、constitution、grading rubric）能内化进无 PI 的 policy，实例专属答案不能——这决定 PI 蒸馏何时成立。代表方法：

- **OPSD**：AIME 2024/2025、HMMT 上 4B/8B 匹配或超过 GRPO，**每问题只用 1 条 rollout（GRPO 用 8）**；8B **+0.9**、4B **+0.8**，1.7B **−0.5**（需足够容量），并有"问题难度天花板"。
- **HDPO**：针对所有 rollout 全失败的 **cliff prompt**（梯度消失），在 GRPO 上加 GT-conditioned rollout 的 **JSD** 自蒸馏项补非零梯度。
- **GATES**（source document 作 PI，consensus gating 在 tutor 不确定度高时抑制梯度）、**π-Distill**（统一理论框架 + 可证改进条件）、**RESD**（通过率 <5% 时把失败当**负向 PI**，reflection 写持久 playbook，样本效率约 **8×** 于 GRPO）。
- **OPCD / OPSDL / OEL**：行为型 PI——经验知识与 system prompt / 短上下文作 self-teacher / 部署后持续学习闭环。
- **CRISP**："be concise" 模型教自己冗长版，MATH-500 上 CoT token **降 57–59%** 且准确率**升 9–16 个百分点**。
- **GUI-SD**（目标框 **Gaussian soft mask** 作视觉空间 PI + entropy-guided token weighting，单 rollout 超 GRPO，ScreenSpot-v2/Pro、OSWorld 等 6 基准）；**TWI**（高分辨率 crop）、**PTD-PO**（空间 + 完整推理链 + **top-K JSD**）、**MGSD**、**Visual-SDPO** 同属视觉 PI 谱系。
- **OPHSD**：把 inference harness（draft-verify、plan-solve）当 PI 内化，HMMT 2025 比 OPSD **+10.83%**，且推理时重挂 harness **不再有收益**——harness 是被永久吸收的临时脚手架。
- **COPSD**（英文题面/参考解作跨语言 PI，17 种低资源非洲语言超 GRPO）、**MSD**（英文推理作 PI 做多语言安全对齐，DPSW 加权）。
- **ATESD**：受控暴露扫描显示"教师看完整参考推理"**非最优**、师生失配随暴露单调增长；Beta-policy controller 建模 reveal ratio，Qwen3-1.7B/4B/8B 上比 OPSD **+0.95 ~ +2.33**。
- **TRACE**：全 token KL 浪费梯度并放大 PI leakage；privileged annotator 标 critical span，正确 rollout 关键 span 用 FKL、错误 span 可选 RKL、其余交 GRPO + KL annealing；四数学基准 + GPQA-Diamond 比 GRPO 平均 **+2.76**，是唯一保住 OOD 性能者（在线自标注仍 +1.90）。
- **PBSD**：reward-regularized 目标 $\max_\pi \mathbb{E}[r]-\beta D_{\text{KL}}(\pi\|\pi_{\text{teach}})$，解析最优 $\pi^\star\propto\pi_{\text{teach}}(y|x)\exp(r(x,y)/\beta)$ 可证优于教师本身；用 DPO 式 pairwise loss，Qwen3-1.7B/4B/8B 上峰值准确率高于 OPSD 且避免 post-peak decline。
- **TT-OPD**：PI 降到 **turn 级**，gradient-free EMA teacher 接收依正确性插入、并在输出 log-prob 中剥除的提示，学生看不到 hint 却每轮得 outcome-aware KL；PI 编码**轨迹级正确性**而非问题级 GT。**VISD** 用结构化多维 judge 做 **direction-magnitude decoupling**（reward 定方向、师生差异定 token 幅度），收敛约 **2×** 于 RLVR。
- 更远端 PI：**AVSD**（多 view consensus + residual）、**On-Policy Critique Distillation**（弱 critic 教强学生，scalable oversight）、**World-Model PI**（future video）、**HERO**（未来环境观测，turn-level KL）、**When Ctx Returns**（no-context self-teacher 作 anchoring regularizer）。
- **PI + 验证/校准**：**Skill-Conditioned Gated SD**（极性开关 r∈{−1,+1}）、**Constitutional Cross-SFT**（Cross-SFT cold-start + RKL）、**SD-PG**（full-vocab RKL + REINFORCE，positive-advantage 门控）、**PBSD\***（Bayesian 把轨迹 reward 重分配到 turn）、**AR-OPD**（诊断 **hindsight leakage**，分解为 anchor + 阈值触发 residual oracle 修正，比 full oracle OPD **+2.3**、比 SFT **+7.9**）、**Rubric-Guided SD**（rubric 作 PI，JSD 目标，推理时免 verifier）。

**局限**：难度超出理解阈值时无有效监督（ceiling 随容量缩放）；cliff prompt 梯度消失；过度 PI leakage 致 entropy 上升、推理变短、OOD 退化；全暴露非最优。

#### 5.2.2 Pure Self-Distillation

不依赖辅助输入，从模型自身输出分布取信号。共通机制是 **rollout diversity**：非零 temperature 采样得到质量有差异的完成集合，**方差本身供选择/蒸馏信号**。方法位于 **staleness spectrum** 上（完全 on-policy ↔ 每 epoch 生成一次复用），以 staleness 换算力/分布对齐。代表方法：

- **SDFT**：用自身 **in-context learning** 构造 demonstration-conditioned teacher，对自身 rollout 做 reverse KL，数学上等价于带隐式 KL 正则的 **trust-region RL**，缓解灾难性遗忘。
- **MTP-SD**：frozen copy 作教师给自采 token span 打分，转成 multi-token predictor，解码快 **>3×**、精度降 **3–7%**；**AR-to-Diffusion**：AR LM 作 frozen self-teacher 经 on-policy 蒸馏训 diffusion LM，训练 token 效率高 **15×–7000×**。
- **UniSD**：把 LLM 自蒸馏分解为三轴（supervision reliability：multi-teacher agreement + token contrastive；representation alignment；training stability：**EMA** teacher + divergence clipping）；6 基准 6 模型 3 族验证核心配方是 on-policy 轨迹 + agreement + EMA；UniSD\* 在 Qwen2.5-7B 上比基座 **+5.4**、比 GKD **+2.8**，无外部教师。
- **Self-Supervised OPD**：复用 GRPO 同组 rollout，对比组内正确/错误轨迹定位区分性位置，只在该处做 KL，**零额外推理开销**；**ROSD**（reflector 抽 corrective idea + error quote 定位首错 span，只蒸后缀）；**VPG**（length-aware variational posterior 偏向更短正确答案）。
- 多轮/agentic：**HINT-SD**（failure-relevant turn + hindsight relabel）、**SD-Search**（step 级 search query）、**Search-E1**（外层 GRPO 与 offline 自蒸馏交替自演化）、**MAIGO**（history-cleaned teacher，Qwen2.5-7B-Instruct 上 sharded-to-full 准确率比 **66.5%→84.1%**）、**Canonical-Context OPD**（answer-masked reverse KL）、**COMAP**（agent 与 textual world model 的 EMA 自蒸馏协同演化）。
- 领域专用：**Vision-OPD**（regional-to-global reverse KL）、**TOD Proactivity**（user-concern 标注）、**It Takes Two**（互补上下文 split 互监督做隐私对齐）。
- **OPCoD**：两个领域专家模型互相 tutoring，self-teacher 同时条件化于自己验证正确的 rollout 与对端自然语言反馈，cognizance-based gate 只在对方更擅长时交换反馈；Qwen3-8B 上物理-化学、化学-材料等达到 mutual Pareto improvement。

**局限**：收益**上界受预训练分布中已有信息限制**；temperature 多样性不足以浮现高质量完成时，必须接外部验证或内部生成 PI。

#### 5.2.3 External Feedback

纯自蒸馏下模型**无法区分"自信但错误"与"确实正确"**，导致 §7.2 的 saturation problem。本节保留自蒸馏架构（无独立教师）但注入外部信号（verifier、环境、multi-agent co-evolution、partial-solution oracle）锚定循环。代表方法：

- **SDPO**（结构化文本反馈：runtime error、失败单测、LLM judge，作 self-teacher 做 token-level KL）、**RLTF**（自由文本 critique，单轮策略匹配自身 feedback-conditioned 的第二轮生成）、**SD-ZERO**（Generator-Reviser 双角色把 binary verifier 转 dense 自监督；Qwen3-4B-Instruct 八基准 avg@8 **68.3%** 超 GRPO **64.2%**）。
- **RESD**（失败转主动纠错源 + 全局 playbook，稀有成功场景比 GRPO 样本效率高 **8×**）；**SRPO**（正确样本走 GRPO、失败样本走 SDPO logit 纠正；Qwen3-8B 比 GRPO **+3.4%**、比 SDPO **+6.3%**）。
- **RLSD**：把梯度分解为 **magnitude**（自蒸馏定每 token 概率改多少）与 **direction**（RLVR 定升降方向）；Qwen3-VL-8B-Instruct 仅 **200** 步超 GRPO 的 **400** 步，**2×** 样本效率。**RLRT** **反转**信号——verifier 判成功时，教师本不会预测的学生选择反映自主推理，故强化这些 "rebellious" token。
- **SDAR**：把自蒸馏作**门控辅助目标**、RL 为主干，sigmoid gate 加强 teacher-endorsed positive-gap token、软衰减负向拒绝；ALFWorld/WebShop/SearchQA 比 GRPO **+9.4% / +10.2% / +7.0%**。
- **CREDIT**：标准 OPSD token reward 的轨迹和等于 response 与 feedback 的 **pointwise mutual information**，可被 input-generic shortcut 抬高；用 batch-contrastive baseline 隔离 input-specific 分量。**OGLS-SD**：诊断 self-reflected teacher 分布的 **reflection-induced bias**，用可验证 outcome reward steer teacher logits 使其校准。
- **π-Play**：三智能体自演化（examiner / conditioned teacher / student），question construction path 成为**零成本 PI byproduct**，KL penalty 让教师跟随学生形成自调课程；Qwen3 4B/8B 上 **data-free** 即超 Search-R1。
- **PAINT**：只暴露 **partial solution**，范围由 rollout 与参考解 overlap（boxed answer、formula、key number）自适应控制（近则藏多、远则露多），再按 energy-space interpolation 只在 entropy mismatch 超阈值处算 loss；AIME 2024/2025 + HMMT 2025 平均比 OPSD **+2.1**、比 GRPO **+2.9**——**部分暴露可优于全暴露与不暴露**。
- 协同演化：**CoPD**（RLVR 与双向 OPD 交错取代"先训专家再蒸馏"）、**VPD**（形式化为 **Variational EM**：E-step 用 adaptive trust-region 依轨迹结果精炼教师、把文本反馈转成动态目标分布，M-step 学生在自身 rollout 上内化）。
- 安全方向：**OPCT**（**冻结副本**作 self-teacher，在 canonical 与遭 sycophancy/jailbreak 扰动的 prompt 对间做 per-token reverse KL，无需 verifier/reward model）；**SG-OPD**（按学生梯度与 verifier 信号的 **sign consistency** 门控 token 级蒸馏 + phased teacher sampling；AIME 2024/2025 + HMMT 2025 比 GRPO **+1.98**（avg@32）、**+7.50**（pass@32））。

**饱和与失败**：Kim et al. (2026b) 把数学推理退化归因于 **epistemic verbalization** 被抑制——轨迹变短时 hedging 与不确定性标记被过度删除，导致更短但校准更差的推理，需外部 grounding 或 π-Play 式内部 PI 才能突破。

**全章总结**：设计空间由 **signal density vs autonomy** 支配。white-box 给最密监督（每 token 一个 |V| 维分布）但需共置教师；black-box 牺牲分布信息换跨组织可用（GAD、OVD 已能恢复有效 proxy）；self-distillation 免外部依赖，但除非辅以外部验证或内部生成 PI，否则受模型既有能力上界约束。成熟度上 white-box 一端较成熟（GKD、DistiLLM、DSKD），self-distillation 一端对复杂推理仍缺原则性收敛保证；近期工作集中在类别边界，如 LUFFY 模糊 off-policy 示范与 on-policy RL 的界线，π-Play 把 self-distillation 与外部可验证的协同演化接起来。

---

## §6 Training Efficiency and Stabilization

On-policy 生成有三类 off-policy SFT 没有的挑战：非平稳分布（旧 rollout 变 stale）、hard prompt 上 gradient SNR collapse、autoregressive rollout+teacher scoring 的计算开销。

### 6.1 Token and Sample Weighting
问题：flawed prefix trap——student 早段错误 token 使 teacher 后续条件分布 $p_T(\cdot|x,\hat y^{bad}_{<t})$ 建立在未训练的 OOD prefix 上而失准，朴素匹配会传播错误。

Token 级方法：
- **TIP**：按 student entropy $h_t$ × teacher-student divergence $\delta_t$ 分四象限：Q1 高熵高散（主校正）、Q2 高熵低散、Q3 低熵高散（overconfident errors）、Q4 低熵低散（已掌握）；理论证明任何纯熵加权（非减、$f(0)=0$）无法区分 Q3/Q4；DeepPlanning 上仅用 Q3（<20% tokens）即超过全 token OPD。
- **SCOPE**：rollout 级 dual-path，错误轨迹走 teacher-perplexity 加权 KL、正确轨迹走 student-perplexity 加权 MLE，抗 diversity collapse，六基准 Pass@32 相对 +7.3%。
- **EGRSD**：entropy-guided confidence gate 降权高熵 teacher 位置但保非零下界；CL-EGRSD 用 causal-lookahead 区分持续高熵段与瞬时高熵。Qwen3-4B/8B thinking mode。
- **SelecTKD**：token 级 propose-and-verify（greedy Top-k / non-greedy Spec-k），拒绝位 masked/down-weighted，隐含 curriculum 以 Token Acceptance Rate (TAR) 量度。
- **AdaSwitch**：滑窗散度均值 $\bar d_{i-1}=\frac1L\sum_{j=i-L}^{i-1}d_j$，当 $d_i>K\bar d_{i-1}$ 时切 off-policy teacher guidance。
- 粒度谱系 TIP→SCOPE→SelecTKD→AdaSwitch，体现 signal fidelity 与 optimization stability 的权衡。

Sample 级：hard prompt pass rate→0，rollout 全含早期错误，梯度预算浪费。
- **Fu et al. 2026b**：三失效（token 监督失衡、student prefix 上 teacher 指导不可靠、tokenizer 不匹配），用 teacher top-$K$ local support matching（截断 reverse-KL + top-$p$ 采样 + 特符掩码）较基线 +19.8%。
- **SOD**：TIR 中错误 tool call 污染观测使散度阶梯加速，step 级自适应加权；Qwen3-4B 教师、0.6B/1.7B 学生最高超基线 20.86%，0.6B 在 AIME 2025 达 26.13%（avg@32）。
- **MOPD**（peer 成功/失败组对比 credit）、**GEAR**（自适应变长 span）、**Visual-Advantage OPD**、**Filter-Then-Reweight**（轨迹过滤+软 token 重加权+PPO-style clip）、**Less-is-More**（按 Off-policy Teacher Decay 截断）。

### 6.2 Curriculum and Difficulty Adaptation
生成前选 prompt。梯度幅度随 pass rate $p$ 两端消失、峰在中段。
- **PACED**：Beta-kernel frontier sampling，$w(p)=p^{\alpha}(1-p)^{\beta}$，$\alpha=\beta=1$ 给 $p(1-p)$、峰 $p^*=0.5$，非对称峰 $p^*=\alpha/(\alpha+\beta)$；是 leading-order minimax-optimal；单次评测每 prompt 约 $K=8$ rollout 估 $\hat p_i$ 并按 $\hat p_i(1-\hat p_i)$ 加权。
- SNR 推论：off-policy 难度分布固定，学生变强后多数数据落入"太易"区；on-policy 因采自 $p_\theta$ 自动贴合能力边界。隐式 curriculum：π-Play、PAINT、PRISM。
- **Uni-OPD**：pass rate 混同能力缺口与"解错"；student 侧 difficulty+correctness 平衡，teacher 侧 outcome-guided margin calibration；5 域 16 基准。
- **TCOD**：多轮 "Trajectory-Level KL Instability"；F2B/B2F 两调度；ALFWorld/WebShop/ScienceWorld 最高 +18 分。
- **Off-policy cold start**（Li et al. 2026j）解 thinking-pattern mismatch；Qwen3 用两阶段；DeepSeek-R1 用 100% off-policy。
- 数据/刷新维：**DeltaPrompts**（zero-delta trap）、**f-OPD**（freshness-aware 控 policy lag）、**Trust-Region Behavior Blending**（闭式混合+annealing）、**Adaptive Teacher-Refresh**（isolation/reward-ratchet/length-tail 三 gate）、**Continual Experience Internalization**、**TRD**（prefix-failure 点插教师校正并拼接）。

### 6.3 Compute Optimization
单步白盒 OPD 成本三部分：① student rollout（$B\times K$，KV cache，memory-bound，主导 wall-clock）；② teacher scoring（全词表 logits，compute-bound）；③ student update（最便宜，定显存下界）。
- **FOPD**：前缀 truncation（on-policy 前缀 + off-policy KD 补尾），匹配全 OPD 而训练 FLOP 降 **2×–47×**；前缀每步增 $\Delta L=256$（Qwen3-8B 教师 / Qwen3-8B-Base 学生）。
- **Lightning-OPD**：离线 teacher caching（需 teacher consistency），与在线同最优，训练效率 **4.0×**。
- **DistillSpec/SKD**：$\mathbb{E}[S]=\frac{\tau(x)}{c\gamma+1}$，$c=\frac{\text{cost(draft fwd)}}{\text{cost(target fwd)}}$；DistillSpec 加速 +10%–45%，多尺度部署降解码延迟 **6–10×**。
- 显存：70B 教师+7B 学生需 ~140 GB 教师权重（BF16）、~84 GB 学生权重/optimizer 及全词表 logits，峰值易超 8×80 GB H100；缓解：teacher quantization（FP8/INT4，2–4×）、logit offloading（99% 概率质量集中 <100 token）、gradient checkpointing。
- 成本：70B→7B、8×H100；off-policy 1B tokens ≈**300 GPU-hours**，on-policy ≈**1,200–1,500 GPU-hours**，**4–5×** 开销。
- **NPD**：三阶段异步（vLLM rollout + top-$k$ logits 并行 prefill/packing + CE+KD），$\Delta$-IFD filtering；**8.1× throughput**，11 基准 vs SFT **+8.09%**；NPD→GRPO 使 openPangu-Embedded-1B 达 68.73%，超 Qwen3-1.7B 的 63.69%。
- **EffOPD**：parameter-dynamics foresight，**3× 加速**；**Prune-OPD**：top-$k$ overlap 检测 prefix-drift，训练时间降 **37.6–68.0%**；**POPD/TOPD**、**SafeSteer**（仅 safety token）、**KAT**（KL-agreement trap 入口截断）。
- 三层叠加可复合各自 2–4× 增益，把 4–5× 成本差压到接近持平。

## §7 Understanding OPD

### 7.1 Success Conditions
1. **推理模式兼容**：teacher/student top-$k$ 分布高重叠；模式错配（non-thinking teacher→thinking student）无法靠训练弥补。
2. **教师须提供新能力**：同数据同 recipe 各尺度收敛相近，reverse distillation 显示同族 1.5B/7B 教师从学生视角已不可区分。
3. **token 级**：在 high-probability overlap tokens 上渐进对齐；密集监督质量随轨迹深度系统性退化，不稳定自后段反向传播。
4. **可开发间隙原则**：gap 太小（同 recipe）无新信息、太大（模式错配）无法吸收，有效区间在中间。
5. **训练无关诊断**（Armandpour et al. 2026）：以 ideal per-node gradient 与蒸馏梯度的 cosine（gradient alignment score）度量，incorrect rollout 对齐度高于 correct；无普适配置。
6. **训练前诊断清单**：① Token overlap ratio（过低→先 cold start）；② 目标 prompt pass rate（全 0→梯度消失，用 curriculum/warm-up）；③ Teacher calibration；④ Length stability（突增→Stable-OPD 预校）。
7. **OPSD 是压缩非校正**（Kim & Lee 2026）：correct-only 保精度并缩短 trace，incorrect-only 降精度；推荐顺序 SFT→RLVR→OPSD（post-RL）。

### 7.2 Failure Modes
1. **Flawed prefix trap**｜症状：错误前缀被传播。机制：teacher 未见过 OOD prefix；token 级 OPD 相对序列级 Reverse KL 有偏、最坏方差界更紧。缓解：TIP/SCOPE 加权、teacher top-$K$ 匹配。
2. **Extrapolation cliff**（ListOPD）｜症状：reward extrapolation $\lambda>1$ 下格式由保留突变崩坏。机制：闭式阈值 $\lambda^\star(p,b,c)$（teacher modal prob、warm-start mass、IS clip），超阈外推不动点离开 clip-safe 区。缓解：$\lambda$ 贴 $\lambda^\star$ 之下；1.7B 学生以 1/5 参数达 8B-SFT 域内持平。
3. **Rock Tokens**（Jiang et al. 2026b）｜症状：饱和后多达 18% token 持续高损、占大量梯度范数。机制：connectives/格式/hedging 等结构残差，因果干预显示对推理无贡献。缓解：视作不可学、配合 TIP 加权。
4. **跨设定综合失效**（Zhu et al. 2026）｜症状：分布错配、top-$K$ reverse-KL 有偏致优化不稳、OPSD 学得 PI-free policy。机制：prefix 条件化+实现梯度偏差；PI 为 instance-specific 时不足、为共享隐式规则时有效。缓解：stop-gradient top-$K$、RLVR-adapted teachers、SFT-stabilized students。
5. **Gradient SNR collapse**｜症状：pass rate→0 梯度消失。机制：全 rollout 含早期错误、信号被噪声主导。缓解：PACED 边界课程、easy warm-up。
6. **Epistemic suppression**（Kim et al. 2026b）｜症状：trace 变短但校准变差、自信犯错。机制：删除 hedging/不确定性标记，形成自强化循环。缓解：不确定性正则、保留校准信号。
7. **Local teachability collapse**（Liu et al. 2026b）｜症状：强→弱 OPD 后段 KL 梯度分辨力趋零（教师仍未失准）。机制：teacher margin 收窄、局部对比消失。缓解：段级 margin 聚合、BIC-style 变点截断（优于 FOPD 固定位置）。
8. **Self-play saturation（Ouroboros）**｜症状：$p_\theta(y_{\text{flawed}}|x)\to1$，梯度消失、策略被困。机制：缺分布多样性、无外部信号惩罚，类似 GAN mode collapse。缓解：引入外部教师/RLVR、rollout mixing。
9. **Precision-recall tradeoff**（Cha & Cho 2025）｜症状：Pass@1 高但 Pass@k 崩。机制：激进 Reverse KL 塌缩到单一策略，由 teacher entropy 调控。缓解：on-policy 探索、temperature 采样（GKD/G-OPD）、SCOPE。
10. **Calibration-capability gap**（CaOPD）｜症状：精度升但严重过度自信。机制：教师监督形成于 privileged context，部署时只有部署上下文。缓解：解耦能力与校准、报告可靠性边界。
11. **Agentic collapse**（TT-OPD 多轮）｜(a) teacher-dynamics collapse：hard-copy 致 KL 骤降（2.637→0.343，$T=30$）→EMA 教师；(b) trajectory-structure erosion：turn 7.65→5.52（重置）/7.82→6.23（EMA）→结构正则+粒度匹配 credit；(c) reward-hint runaway：hint 无长度控制致爆炸，精度 54.5%→49.0%→长度正则。
12. **Off-metric capability erosion**（Wang 2026）｜症状：只报保留任务分、不报损失。机制：蒸馏=有损投影（丢校准、推理链质量、分布窄化）。缓解：Distillation Loss Statement。
13. **Exposure-bias/多教师/梯度**｜MOTAB 双 exposure bias→回溯最近安全态；Counteraction-Aware MOPD 多教师 per-token 冲突相消→counteraction-aware router；Physics-Guided SD 用互信息作粘度式步长乘子。

### 7.3 Unified Theoretical Perspectives
- **统一 f-divergence**（Wen et al. 2023）：Forward KL、Reverse KL、JSD、total variation 皆由单一凸生成函数 $f$ 参数化，且都可写为 student on-policy 分布下的期望，故 divergence 选择本质是正则化；Forward KL 上调学生低估处（coverage）、Reverse KL 上调高估处（precision）。Wu et al. 2025 在离散 LLM 上归结为 head-tail focus（Forward KL 居 head、Reverse KL 重 tail）；Cha & Cho 2025：KD 的 precision-recall 权衡由 teacher entropy 单参数调控（SmolLM2），低熵教师更尖锐但覆盖更少 mode。
- GRPO 与 self-distillation 处理互补失效模式，可用 sample routing 组合（Li et al. 2026c）。
- **参数空间**：OPD 处 relaxed off-principal regime，核心现象 **subspace locking**（累积更新进入狭窄低维通道，单此即可恢复性能，同约束却损 SFT）；更新小且坐标稀疏却数值满秩、集中 FFN、偏好源权重近零坐标，只训子网络近乎恢复全训性能，但 AdamW 仍胜 SGD（Shen et al. 2026c；Yu et al. 2026a）。
- **长度膨胀**：源于 reverse-KL advantage 反馈环；**Stable-OPD** 加 reference divergence + 衰减比例混入 off-policy gold demonstrations，平均 **+7.2%**（Luo et al. 2026a）。
- **等价/桥接**：标准 OPD 是 dense KL-constrained RL 的特例（G-OPD）；将教师 log-prob 代入 DPO 隐式奖励即得 GKD 的 $\lambda$-mixing，统一 token 级 KL 蒸馏、偏好优化、reward-regularized RL。Li et al. 2025 把 hybrid 梯度分为稳定 dense KD 项 + 高方差 Monte Carlo RL 项（KD 降方差、RL 防塌缩）。Song et al. 2026：文本反馈（Lion、OVD）比二值 outcome 更密。结论：OPD/RL/偏好优化是由监督密度、divergence、信号来源参数化的连续谱。

### 7.4 On-Policy vs. Off-Policy: A Decision Framework
- **DeepSeek-R1 反例**：纯 off-policy 用约 800K CoT 轨迹；AIME 2024 pass@1 R1-Distill-Qwen-7B **55.5%**、32B **72.6%**，超同规模 GRPO（Qwen2.5-32B-Base **47.0%**）；因 671B MoE 教师覆盖稠密 + 轨迹多样 + 学生相对较小。
- **off-policy ceiling**（DDT）：静态数据无法覆盖全部有效推理路径，gap 随任务复杂度、输出多样性、模型能力拉大。
- **可操作判据清单**：
  1. 教师-学生容量比 >10× 且推理深度有界（factual QA/翻译/摘要）→ 仅用 off-policy SFT。
  2. 任一成立则切 on-policy OPD：(a) 学生 >~7B 且开始探索静态轨迹未覆盖区域；(b) 多步推理、中间错误累积；(c) off-policy loss 平台但 on-policy held-out reward 仍升。
  3. 其余情形→混合流水线（off-policy warmup + on-policy refinement）。
  4. 容量维：≤7B 宜 dense logit 监督（GKD、DistiLLM），≥7B 有可靠 verifier 宜 reward-guided OPD（KDRL、G-OPD）；目标"追平教师"选蒸馏、"超越教师"选 RL。
- **成本式**：
  $$C_{\text{off}}\approx N(F_{\text{teacher}}+F_{\text{student}}+B_{\text{student}}),\quad C_{\text{on}}\approx N(G_{\text{student}}+\rho F_{\text{teacher}}+F_{\text{student}}+B_{\text{student}})$$
  $\rho\in(0,1]$ 为教师监督刷新率；因 $G_{\text{student}}\gg F_{\text{student}}$，on-policy 有显著乘子。通用知识迁移很少值得，reasoning/code/数学值得；实践为前置 off-policy、on-policy 留给最后冲刺。OPD 与 RLHF 仅信号源不同（教师分布 vs reward model preference），可复用 OpenRLHF 等基础设施。


---

## §8 Applications, Systems, and Emerging Domains

论文主张：OPD 方法选择多取决于实际约束（算力、teacher 访问层级、multi-teacher 协调、system 吞吐、跨模态/隐私难题），而非理论优雅性。工业应用按 **deployment pattern** 组织，新兴领域按新挑战组织。

### 8.1 Industrial Deployment（五种 pattern + 一条正交安全线）

**① Two-phase pipeline（off-policy cold-start + on-policy refinement）**
- **Qwen3**：strong-to-weak distillation，teacher 32B / 235B-A22B，在 student 自生成序列上给 **logit-level** 监督；student 分 thinking / non-thinking 模式，关键设计 **Thinking Mode Fusion**（从各模式 teacher 独立蒸馏，因"何时思考"本身是学到的 policy）。报告 on-policy distillation 以约 **1/10 GPU hours** 优于直接 RL，并在 AIME 上改善 pass@64（同起点 RL 做不到）。
- **Gemma 2**：把 off-policy KD 嵌入 pre-training，2B 与 9B 经 logit-matching KD 从更大 teacher 蒸馏（27B 从头训练），二者**并行**而非级联；论文明确其非严格 OPD。
- **MiMo-V2-Flash**（Xiaomi）：多领域专用 teacher 蒸馏 + **MTP** + hybrid attention，**309B MoE（15B active）**。

**② OPD as model consolidation**
- **DeepSeek-V4**：**完全取代** DeepSeek-V3 的 mixed RL，改纯 multi-teacher OPD，将 10+ 领域专家（math/coding/agent/instruction-following）合并为 **1.6T** 统一模型，用 **full-vocabulary Reverse KL**（全 $|V|$ 维 KL 而非 sampled-token 近似，保证合并多样专长梯度稳定）+ hidden-state caching / on-the-fly logit reconstruction。
- **KAT-Coder-V2**：agentic coding 分 5 专家域独立 SFT+RL，OPD 合并 → **SWE-bench Verified 79.6%**。
- **Nemotron-Cascade 2**：**30B MoE（仅 3B activated）**，在 **IMO / IOI / ICPC World Finals** 达 Gold Medal 级，参数比 DeepSeek-R1 少 **20×**。
- **CoPD**：指出 sequential expert-then-distill 的缺陷（完全训练 teacher 与未训练 student 行为鸿沟大，独立专家策略互不兼容）。解法 **co-evolution**：并行 RLVR + 交错**双向 OPD**，专家互为 teacher；得统一 text+image+video 推理模型，超 mixed RLVR 与 sequential MOPD。

**③ OPD for multi-budget reasoning — ORBIT**：在逐步收紧 budget（$L_{k+1}=L_k/2$）下用 RLVR 得分档专家，经 **mode-aware Reverse KL** 融合：
$$L_{\text{fusion}}(\phi)=\mathbb{E}_{k\sim U(1,K)}\mathbb{E}_{q\sim D,o\sim\pi_\phi(\cdot|q,p_k)}\Big[\sum_t\log\tfrac{\pi_\phi(o_t\mid q,p_k,o_{<t})}{\pi_{\theta_k}(o_t\mid q,p_k,o_{<t})}\Big]$$
以 model merging 初始化缓解 cold-start。在 **DeepSeek-Distill-Qwen-1.5B、Qwen3-4B、Openmath-Nemotron-7B** 上产出 4 档推理模式（budget **2K–32K tokens**，随 student 规模）平滑插值的模型。

**④ Agentic distillation（新挑战：error compounding）**，按粒度四种解法：
- **trajectory 级 — TCOD**：temporal curriculum 逐步扩展监督视界，对抗 "Trajectory-Level KL Instability"，在 **ALFWorld / WebShop / ScienceWorld** 上较 vanilla multi-turn OPD **+18 points**。
- **step 级 — MAD-OPD 的 OPAD**：每步为独立蒸馏单元，断跨步梯度；加 multi-teacher debate ensemble 治单 teacher 后期不可靠；task-adaptive divergence（agentic 用 **JSD**、code 用 **Reverse KL**），较 single-teacher OPD **+2.4%**。
- **skill 级 — OpenClaw-RL**：hindsight-guided OPD + PRM 做 step-level credit assignment，训 **Qwen3-4B** agent（personal/terminal/GUI/SWE）。**Skill-SD**：importance-weighted Reverse KL + dynamic teacher sync，**AppWorld +14.0%**、**Sokoban +10.9%**（较 GRPO）。
- **turn 级 — Li et al. (2026a)**：重访 **DAgger**，student/teacher policy 按轮插值再对完整轨迹监督学习；**SWE-bench Verified** 上 **Qwen3-4B / Qwen3-8B**（teacher **GPT-4o**）较纯 SFT **+3.9 pts**。
- 原则：蒸馏单元粒度须匹配错误复合的粒度；粗（整轨迹）欠佳（复合是局部的），细（token 级）浪费算力；新兴观点认为有效粒度是 **decision boundary**，类比 hierarchical RL 的 option boundaries。

**⑤ Unified safety infrastructure — Safactory**（Shanghai AI Lab）：闭环安全演化流水线，整合并行仿真、可信数据管理、结合 async RL 与 OPD 的自主演化；OPD 是**反复出现**的组件而非一次性训练步。

五种 pattern 把 OPD 定位为：质量精炼、架构简化、推理时控制（单模型多档 latency-accuracy）、薄弱领域使能器（agentic / multi-turn / multi-modal）、自主 agent 的安全合并机制。

**正交线 — safety-tax reduction**：**Fu et al. (2026a)** 针对 safety tax（安全微调带来的通用能力下降），用 per-token reverse-KL 从 **privileged-context teacher**（同模型 + constitution 式安全前置语）蒸馏到仅见 user prompt 的 student；借 OPSD 模板，更新只作用于安全相关 token，通用能力几乎不动。

### 8.2 Emerging Domains

**Multimodal**：**VOLD**（纯文本 teacher 无需看视觉即可用 OPD 引导 VLM student，文本推理结构为 modality-agnostic scaffold；**需 SFT cold-start** 否则不收敛；**MMMU-Pro** 提升）。两轴：#intra-model self-alignment 与 external cross-modal transfer。**CORD**（同一模型 text mode 教 audio mode，加权 Reverse KL + GRPO）：
$$L_{\text{CORD}}=\mathbb{E}_{x\sim D}\mathbb{E}_{\hat y\sim\pi_{\text{audio}}(\cdot|x)}\Big[\sum_t w_t D_{\text{KL}}\big(\pi_{\text{audio}}(\cdot|x,\hat y_{<t})\|\pi_{\text{text}}(\cdot|x,\hat y_{<t})\big)\Big]$$
**Video-OPD**（temporal video grounding）；**X-OPD**（text-only teacher → speech LLM student）；**KEPO**（初始 solve rate→0 时稀疏 RLVR 因 exploration collapse 失败，以 teacher knowledge 条件化 preference optimization 提供 bootstrapping；medical VQA）；**VISD**（视频推理 structured self-distillation，direction-magnitude decoupling，收敛比 RLVR 快 **~2×**）；**HyperEyes**（多模态搜索 agent，**TRACE** macro trajectory reward + micro OPD；**HyperEyes-30B** 较同规模最强开源 agent 准确率 **+9.9%**、工具调用轮数减少 **5.3×**）。完整弧线：KEPO（dense bootstrapping）→ VOLD/X-OPD（cross-modal transfer）→ CORD（intra-model self-alignment）→ VISD（temporal）→ HyperEyes（efficiency-aware agentic）。

**GUI agent — LiteGUI**（Wu et al., 2026b）：标准 GKD-style OPD 失效（student 输出常为非法 JSON、GUI 多解）。用 **Guided On-policy Distillation** 从动态检索的合法动作集给 teacher 提供 oracle 参考轨迹，加 **Multi-solution Dual-level GRPO** 联合 macro subtask planning 与 micro execution。teacher **Qwen3-VL-32B**，student **2B–3B**；**ScreenSpot-Pro / OS-World / Lite-Bench** 上强表现。

**Embodied / physical**（连续输出、错误不可逆）：**HY-Embodied-0.5**（Tencent Robotics X，RL→RFT→OPD，**32B expert VLM → 2B embodied MoT**，**16/22** benchmark 领先）；**OPD-AV**（GKD **5× 压缩**，nuScenes，**Qwen3-8B SFT → 1.7B**）；**GUI-SD**（privileged 自身含 bounding box + Gaussian soft mask 给 token-level KL，entropy-guided 按 digit positional significance 与 teacher confidence 加权，**6** 个 GUI grounding benchmark）；**VLA-OPD**（自生成机器人轨迹上 Reverse KL，VLA student；连续动作空间下失配表现为物理不安全轨迹；样本效率优于纯 RL、鲁棒性优于 offline SFT）。观察：任务从认知转向物理，on-policy 越重要。

**Medical / clinical**：**Healthcare AI Gym**（Jeong, 2026；**10** clinical domains、**3.6K+** tasks、**135** tools、**828K** medical passages）。暴露 **agentic-textual transfer gap**：**MMLU-Med 从 logprob 评测 83.8% 降至 agentic 评测 60–66%**。**TT-OPD** 在 **18 个 benchmark 中 10 个**最佳，较非 RL baseline 平均 **+3.9 pts**。

**Protein — ProteinOPD**（Zhang et al., 2026m）：扩展至 PLM，多目标（stability/activity/solubility）对齐且不毁 designability；每目标一个 preference-specific teacher，在 student 自身轨迹上蒸馏到加权 teacher 分布的归一化几何共识，梯度冲突下有界优化，**具冲突梯度下的形式化收敛保证**；较 RL 竞争方案 **快 8×**。

**Hardware — RWOPD**（Zou et al., 2026）：从自然语言规格生成 **SystemVerilog Assertions (SVA)**，property-equivalence verifier 打分并用 verifier reward 加权 forward-KL 蒸馏。模式：有廉价可靠 verifier 的领域（形式验证、编译器检查、单元测试）适合 **reward-weighted OPD**。

**Test-time OPD — TTS**（Kumar et al., 2026）：针对 speculative decoding 中顶级 speculator 的 acceptance length 在数千 token 内衰减到 1；当作 **online OPD**，生成中持续让 draft（student）适配 target（teacher），token 验证信号零额外成本。跨 **Qwen-3 / Qwen-3.5 / Llama-3.1**，acceptance length 最多 **+72%**（平均 **+41%**），增益随长度放大。相较 **DistillSpec**（离线 OPD），TTS 是推理时 OPD。

**Privacy — DP-OPD**（Khadem et al., 2026）：on-policy 训练中对 student 梯度注入 **DP-SGD** 噪声、teacher 冻结；on-policy querying 可在 teacher 训练数据的**形式化 DP 保证**下提取效用。

共同模式：on-policy 机制原样迁移，但"privileged information / reward / trajectory"的定义须按模态与任务重新协商；核心框架 modality-agnostic，领域特异性集中在 signal interface。

### 8.3 System-Level Integration

生产级 OPD 须编排三个**并发**负载：student rollout（自回归、memory-bound）、teacher scoring（compute-bound 前向）、student 梯度更新（compute-bound 反向）。**OpenRLHF / veRL** 拆进独立 process groups：teacher 用 tensor parallelism 跨 **4–8 GPUs**，student 生成用 pipeline parallelism，optimizer 用 **ZeRO-3**。核心挑战是**同步**——student policy 每步梯度后即变，pending rollout 变陈旧；异步架构容忍 staleness 换吞吐、代价是梯度更噪。**vLLM / TensorRT-LLM** 以 continuous batching 与 PagedAttention 服务生成。**通信**：white-box OPD 需暴露完整 logits；**70B teacher + 7B student、8×H100** 每 batch 约 **16 GB**（$[B,T,|V|]\times2$ bytes BF16，$B{=}16,T{=}4096,|V|{=}128\text{K}$），单节点 NVLink（900 GB/s）可行，多节点需 tensor slicing 或 top-k sparsification。

**DeepSeek-V4 系统细节**：不 materialize 每个 teacher 的 $|V|{>}100\text{K}$ 张量，只在集中 buffer 缓存**最后一层 hidden states**、训练时经 prediction head **on-the-fly 重建 logits**；mini-batch 内按 teacher identity 排序，任一时刻最多一个 prediction head 驻显存；权重加载/卸载全异步；OPD 阶段集成 **FP4 quantization-aware training**（量化 MoE expert 权重与 attention QK 激活），从起点适配部署精度。

**三项 OPD 独有系统需求**：**teacher co-hosting**（teacher 须全程驻留，显存随其规模膨胀）、**logit-tensor transfer**（传完整/top-k 分布而非标量 reward，跨节点带宽升数数量级）、**staleness tolerance**（teacher 评分须与当前 student policy 一致，收紧异步窗口）。故 OPD 主流框架（**OpenRLHF、veRL、SLIME、verl-pipeline**）均把生成 / 评分 / 更新拆为独立 process group 加显式同步屏障。新兴观点：万亿参数级 OPD 需框架级 hidden-state caching、logit reconstruction、teacher-aware batch scheduling，而非把 teacher 当被动数据服务。

**Compute budget allocation（三阶段）**：① **off-policy warm-up**（占大头）——teacher 数据 SFT；② **on-policy logit distillation**（on-policy 中最大份额）——token-level OPD + 自适应 divergence（GKD、DistiLLM、EOPD）；③ **reward-guided refinement**（最后较小份额）——RL 增强（G-OPD、KDRL、RLAD）推过 teacher 上限。**Qwen3** 四阶段（cold-start SFT → reasoning RL → thinking-mode fusion SFT → general RL）为同类模式，约 **1/10 GPU 成本**。on-policy 在 off-policy warm-up 把 student 拉近 teacher 后最有用。

## §9 Open Problems and Future Directions

方法沿"固定目标→自适应、外部 teacher→自生成、单轮→多轮"演进，问题按此组织：

1. **Distillation scaling laws**：pre-training 缩放律成熟，**蒸馏专属缩放**才起步。Busbridge et al. (2025) 首个 off-policy 蒸馏缩放律：**compute 与最优 teacher 规模非单调**——算力增则最优 teacher 先增至略超 student、后平台化、最终**下降**（大 teacher 推理成本吞预算）；最优 student 蒸馏 token 仍按幂律。DeepSeek-R1 补 student 轴证据（AIME 2024，1.5B→7B→14B→32B 为 **28.9%→55.5%→69.7%→72.6%**，14B 后急剧递减）。**难**：on-policy 的 rollout 生成成本引入 off-policy 没有的额外算力轴，无对应框架。论文给**推测性**（未验证）联合幂律 $L(N_S,N_T,D_{\text{on}})=E+\frac{A}{N_S^\alpha}+\frac{B}{N_T^\beta}+\frac{C}{D_{\text{on}}^\gamma}+f(N_S,N_T)$（$f$ 建模容量差干扰）；非单调 teacher 说明 $B/N_T^\beta$ 不足、$f$ 须含 teacher inference 压倒质量增益的区间，且须计 teacher diversity（Qwen3 表明多专用 teacher 优于等价总参数单一大 teacher）。关键待答：$C/D_{\text{on}}^\gamma$ 与 teacher 项是**协同还是拮抗**（更多 rollout 能否补偿弱 teacher，还是 teacher 质量设硬上限）。**方向**：受控网格独立变 $N_S,N_T,D_{\text{on}}$。
2. **Uncertainty-aware feedback**：teacher 只给点估计，自己不确定时 logits 信息量低却被平等对待。方向：纳入 **epistemic uncertainty**，在 teacher 靠猜处打折（用 PRM 逐步打分定位可靠 vs 投机）。关联 §7.2：**flawed prefix trap 本质是未被识别的 teacher 不确定性**，可在 teacher 熵趋最大处压制蒸馏梯度。**TIP** 支持此方向且表明**relative uncertainty（teacher 对 student 续写的惊讶度）比绝对置信度更重要**。下一前沿 **predictive uncertainty**（预估若干 token 后 teacher 的可靠性以避开其无能区）。**难**：如何量化并实时利用 teacher 不确定性而不付出完整前向成本。
3. **Agent-level, continual, lifelong distillation**：多轮需新 credit assignment（长视界工具使用、状态跟踪、错误恢复）；OpenClaw-RL / Skill-SD / TCOD / MAD-OPD 是早期步骤，但 **teacher 监督随轨迹深度系统性退化**（Li et al., 2026j）仍难 → 需 **temporal decomposition + progressive exposure**。三子问题：**environment non-stationarity**（环境随动作变化，须 teacher 基于 student 实际状态做**反事实评估**）；**tool-use combinatorics**（数十个带 schema 的工具，或须在 **tool-call level** 蒸馏并把成败归因到具体工具选择）；**safety-critical exploration**（一次错误写文件/误点击即可不可逆，须防灾难性危险动作）。另：teacher 自身演化（安全补丁、能力更新）时目标如何适应，**SDFT / OEL** 是初步答案，但 **lifelong distillation**（不灾难遗忘、不重训、不累积漂移）仍是难题，且**师生分布双双非平稳**。
4. **Efficiency frontiers**：FOPD / Lightning-OPD / SKD 有进展，但 **OPD 理论最低成本（达给定质量所需 on-policy 数据量的信息论下界）仍未知**。欠探索方向 **selective teacher inference**：MiniPLM 的 "Difference Sampling"（按 teacher-reference log-prob 差异选样本）在 off-policy KD 有效，移植到 on-policy 可跳过 student 已收敛的无信息 rollout 的昂贵 teacher 推理。
5. **Latent-space and cross-modal distillation**：当前 OPD 全在 **vocabulary space**，跨架构对齐须专门机制（DSKD dual-space projection、ULD probability-space transport）。更激进方案是在 **shared latent space** 蒸馏，绕开词表瓶颈。待答：多模态一致性下暴露偏差如何表现？teacher/student 处于不同模态空间时怎么选 divergence？统一 $f$-divergence 框架能否扩展到连续信号（图像、音频）与离散 token 并存？VOLD/X-OPD/CORD 是经验起点但理论基础未立。
6. **Privacy and evaluation methodology**：DP-OPD 给出 DP 蒸馏早期框架（非对称：只对 student 施加 DP-SGD）；on-policy 契合 DP（自生成前缀减少 DP 噪声的复合误差）。待答：扩展到推理型 LLM、与 self-distillation 结合。评测：Fu et al. (2026b) 表明 token-level OPD 可在标准 benchmark 成功却在分布偏移 prompt 失败；Cha & Cho (2025) 的 precision-recall 框架显示蒸馏在 sample precision 与 coverage 间有深层 trade-off，故评测须探查学到的是**因果机制**还是表层相关性。
7. **Diagnostic tools for failure modes**：§7.2 的失败模式（flawed prefix trap、epistemic suppression）需更系统诊断，领域缺统一框架。方向：实时 diagnostic probes 追踪 gradient signal-to-noise ratio、representation collapse、teacher-student divergence dynamics 而不付完整前向成本，自动提示 **Ouroboros self-distillation saturation** 或 agent 的 **trajectory-structure erosion**，把调试从 post-hoc 失败分析转为训练中主动干预。
8. **Cross-architecture scalability**：异构模型族映射摩擦大（§5.1.1）；DSKD / ULD 只在较小规模差验证。**容量差跨数量级时（如 400B MoE teacher → 1B dense student）可行性基本未证**，表征瓶颈或使线性投影/词表对齐失效；需**非线性、层次化对齐机制**。
9. **Unifying distillation and RL**：① **temporal scheduling**——G-OPD / KDRL / RLAD 证明稠密引导 + 稀疏 reward 有价值，但如何在生命周期内分配"模仿 vs 超越"仍是启发式；应条件化于 student 相对能力，即**能力差大时蒸馏、追平时转 RL** 的 dynamic policy。② **loop structure**——静态蒸馏耗尽 teacher 知识后饱和，纯 RL 稀疏 reward 失稳；KDRL / REOPOLD 用早期稠密 KD 求稳、晚期 reward 探索；**CoPD 最激进**，以 RLVR 与双向 OPD 同跑的 co-evolution 取代 train-then-distill。③ **limit**——**π-Play** 是自生成课程/自过滤数据的早期存在性证明，但饱和分析标明边界：**无外部验证或内部 privileged information，纯自蒸馏收敛到自身先验**。能否不靠更强 oracle 抬高天花板、实现 **capability creation 而非仅有 redistribution**，是最深问题，进展或取决于可规模化、无需人工标注的 **verification mechanisms**。
10. **Game-theoretic self-play**：**SPIN** 训练更新后模型区分上轮生成与人类参考（数据分布处不动点保证）；**IRIS** 用 interpolative Rényi divergence 推广，统一多个 self-play 变体。其目标（贴近参考分布）与 OPD 不同，但互补：Nash 收敛保证或可稳定迭代自蒸馏循环，OPD 的稠密监督可缓解对抗训练样本低效；**混合框架**是有前景方向。

## §10 Conclusion

沿三轴组织文献：**Objective function design**（固定 divergence→自适应路由→RL 增强以抬高 teacher 上限）、**Signal source**（稠密 white-box logits→black-box API→完全去 teacher 的 self-distillation）、**Training dynamics**（token weighting、curriculum、compute optimization）。

**Key Findings（三条反复模式）**：① 从 off-policy 转 on-policy（$y\sim p_{\text{data}}\to y\sim p_\theta$）在数学推理、代码、指令遵循上一致增益，归因于**暴露偏差降低**，推理链越长差距越大。② divergence 宜**按任务与 token 位置**自适应：**Reverse KL 擅推理**（防幻觉）、**Forward KL 保多样性**；**EOPD / AOPD** 是按局部几何自适应的前沿。③ **蒸馏与 RL 边界持续收窄**——G-OPD / RLAD / REOPOLD 表明稠密 teacher 监督 + 稀疏 reward 有效（KD 稳梯度、RL 允许超越 teacher），并产出共享工具箱。

**Practical Takeaways**：white-box 时 token-level（GKD、DistiLLM）质量-算力 trade-off 佳；推理密集任务偏好 sequence-level（MiniLLM）或混合（FOPD、PACED、Lightning-OPD），代价是 REINFORCE 式更高梯度方差。仅 API 时 **GAD**（对抗式）与 **OVD**（verbal）可支持 on-policy；self-distillation（**OPSD、SD-ZERO**）完全免 teacher，资源受限下吸引。开销可用 speculative decoding、prefix truncation、offline precomputation 降低。

**Looking Ahead**：从 off-policy 模仿到 on-policy 自纠正呼应机器学习的更广模式（把 learner 自身分布纳入目标）；**缺 distillation scaling laws、teacher uncertainty quantification 未解、无灾难遗忘的 lifelong adaptation** 是通往"模型持续生成-评估-更新自身"regime 的主要障碍。

**Limitations**：仅涵盖训练中含 student on-policy 生成的方法，排除一般 KD（feature-based、attention transfer）、正交压缩（pruning、quantization）、仅推理时用生成数据的方法（self-consistency、majority voting）；覆盖截至 **2026 年 6 月**，每月新增 10+ 篇 OPD 论文，部分同期工作或缺。

**Reproducibility**：方法在 base model、算力、benchmark 版本（MATH-500 vs 完整 MATH）、rollout 数上不一致，很少全控变量，故跨方法比较不可靠；Tables 3–9 与 2 报告原始论文结果而非受控复现。呼吁**标准化 OPD benchmark 协议**（固定 base model、算力预算、评测套件），类比 HELM。

**Broader impact**：① **可及性**——小模型跑消费级硬件，self-distillation 免外部 teacher 而提升能力。② **安全**——风险是降低误用门槛；缓解是 teacher 训练中保留监督角色。**MSD**（Qin et al., 2026）以 on-policy self-distillation 把安全对齐从高资源迁到低资源语言而**无需目标语言安全数据**，泛化到未见语言与更强 jailbreak；**Safactory** 表明 OPD 可作持续安全演化循环的反复组件。③ **可持续性**——OPD 算力开销（较 off-policy SFT **4–5×**）在工业规模引忧，但更小更强 student 降低下游推理成本，摊销后**全生命周期净碳足迹可能更低**。蒸馏、RL、self-play 收敛为更统一的 on-policy 训练图景，未来 LLM 或不再是一次性产物，而是经与用户/环境交互周期性精炼的系统。

---

## 11. 落地清单 【编者按，非原文】

以下是我从论文里抽出来、可以直接拿去用的一张检查表（每条的出处章节已标出）：

**上车前（§7.1 训练前诊断清单）**
1. 先量 **token overlap ratio**（师生 top-k 分布重叠）：过低说明师生推理模式不兼容，不要直接上 OPD，先做 off-policy cold-start。
2. 量**目标 prompt 的 pass rate**：若近乎全 0，梯度会消失（gradient SNR collapse），需要 easy warm-up 或课程（PACED）。
3. 查 **teacher calibration**：教师对"学生自己走出来的 OOD prefix"给出的条件分布可能本身就不准 —— 这是 flawed prefix trap 的根因。
4. 盯 **length stability**：训练中出现长度突增 = 长度膨胀，需 reference divergence 约束或混入 off-policy gold demonstration（Stable-OPD）。

**踩坑预警（§7.2 失败模式，13 条里最常撞的几条）**
- **flawed prefix trap**：学生早段错误 → 教师在该 OOD prefix 上的监督不可靠 → 错误被传播。缓解：TIP / SCOPE 式 token 加权、teacher top-K 匹配。
- **self-play saturation（Ouroboros）**：纯自蒸馏没有外部信号纠偏，会收敛到自身先验。缓解：引入 verifier / RLVR / 外部教师。
- **precision-recall tradeoff**：激进 reverse KL 会让 Pass@1 高但 Pass@k 崩（多样性塌缩）。
- **calibration-capability gap**：精度上升但严重过度自信 —— 部署前要单独报告可靠性边界。

**资源与成本（§6.3、§8.3）**
- 成本量级：off-policy 约 1B tokens ≈ **300 GPU-hours**；on-policy 同量 ≈ **1,200–1,500 GPU-hours**，即 **4–5×** 开销。
- 但这 4–5× 可以被压回来：FOPD（前缀截断，FLOPs 降 **2×–47×**）、Lightning-OPD（教师 log-prob 离线缓存，**4.0×**）、NPD（异步三阶段，**8.1× throughput**）、Prune-OPD（prefix drift 剪枝，省 **37.6–68%** 时间）。
- 显存：70B 教师 + 7B 学生需要约 **140 GB** 教师权重（BF16）+ 约 **84 GB** 学生权重/optimizer，再加全词表 logits，峰值很容易超过 8×80GB H100。缓解手段：teacher quantization（FP8/INT4）、logit offloading、gradient checkpointing；DeepSeek-V4 的做法是**只缓存最后一层 hidden state，训练时 on-the-fly 重建 logits**。

**小模型的红线**
- 容量是硬约束：**OPSD 在 1.7B 上是负收益（−0.5）**，4B/8B 才有正收益（+0.8 / +0.9）。要做 ≤1B 的蒸馏，先看论文里 0.6B 级别的做法（SOD 在 AIME'25 达 26.13% @0.6B 是少见的正面案例）。
- §5.2.1 给出的判据可以直接用：**可跨实例聚合的规则**（system prompt、constitution、grading rubric）能被内化进无 PI 的 policy；**实例专属的答案不能** —— 这决定了 PI 蒸馏在你这个场景是否成立。

**与 RL 的组合方式（§4.3、§7.4、§9.9）**
- 通行配方是**三阶段**：off-policy warm-up（占大头）→ on-policy logit distillation（on-policy 中最大份额）→ reward-guided refinement（较小份额，推过教师上限）。Qwen3 的四阶段流水线是同类模式，报告约 **1/10 GPU 成本**优于直接 RL。
- 判据（§7.4）：教师-学生容量比 >10× 且推理深度有界（factual QA / 翻译 / 摘要）→ 用 off-policy 就够；**学生 >~7B 且开始探索静态轨迹未覆盖的区域**、多步推理中间错误累积、或 off-policy loss 平台但 on-policy held-out reward 仍在涨 → 必须切 on-policy。

**评测陷阱（§9.6、§10 可复现性）** 【编者按 —— 这条对你尤其重要】
论文自己承认：各方法在 base model、算力、benchmark 版本（MATH-500 vs 完整 MATH）、rollout 数上都不一致，**很少控制变量，因此跨论文的方法比较不可靠**；正文表格报的是原论文结果，不是受控复现。所以看到"方法 A 比方法 B 高 X 分"时，要按"两个不同实验"处理，而不是"一次消融"。论文呼吁建立类似 HELM 的标准化 OPD benchmark 协议。

---

## 附录 B：术语对照表

| 英文 | 中文/说明 |
|---|---|
| on-policy / off-policy | 训练数据在训练时采自学生当前 policy / 采自固定语料 |
| exposure bias | 训练状态与部署状态之间的分布失配 |
| compounding error | 沿轨迹累积的误差；自回归下按序列长度二次放大，O(εT²) |
| DAgger | 交互式模仿学习：在 learner 自己访问的状态上查询 expert，界降到 O(εT) |
| f-divergence | 由凸生成函数 f 参数化的一族散度，$D_f(P\|Q)=\mathbb{E}_{y\sim Q}[f(P(y)/Q(y))]$ |
| forward KL | $f(u)=u\log u$，mode-covering / zero-avoiding，保多样性 |
| reverse KL | $f(u)=-\log u$，mode-seeking / zero-forcing，防幻觉、擅推理 |
| JSD | 对称有界的折衷散度，$f(u)=u\log u-(u+1)\log\frac{u+1}{2}$ |
| logit / dark knowledge | 教师未归一化的输出分数 / 错误 token 之间的相对概率结构 |
| white-box / black-box | 可访问教师完整输出分布 / 只有生成文本或标量分数（API 受限） |
| privileged information (PI) | 训练时额外上下文条件化出的"更强 self-teacher"，推理时不可得 |
| self-distillation | 不依赖外部教师，从模型自身内部不对称性取信号 |
| $\pi_{\text{mix}}$ | 采样混合策略，如 $\lambda p_\theta+(1-\lambda)p_{\text{data}}$ |
| reward extrapolation | G-OPD 中 α>1 迫使学生超出教师概率质量去发现新推理路径 |
| flawed prefix trap | 学生错误前缀使教师条件分布失准，错误被进一步传播 |
| mode collapse / saturation | 策略塌缩到少数模式 / 自蒸馏无外部锚点时收敛到自身先验 |
| staleness | rollout 相对当前 student policy 的陈旧程度（异步换吞吐的代价） |
| credit assignment | 把序列级成败归因到具体 step / token / turn |
| teacher co-hosting | 教师须全程驻留显存，是 OPD 独有的系统约束 |
| MoE / dense | 混合专家架构 / 稠密架构 |
| pass@k / avg@k | 采样 k 次至少一次正确 / 采样 k 次的平均正确率 |

---

## 附录 C：本摘要的生成方式、已核验与未核验

**生成方式**：先用 pypdf 从 `2604.00626v4.pdf` 抽取纯文本（89 页 / 41.2 万字符，存为 `/tmp/paper.txt`），再把正文按 10 章切成 5 块，由 5 个子代理并行精读、各自产出一份中文笔记，最后由我合并、统一标题层级、补写速览与附录。

**已核验（我逐条回原文比对）**
- 全部**定量断言做了 grep 复核**，均命中原文：`2.637→0.343`（§2.2 与 §7.2 两处，语境不同、均属实）、`55.5 / 72.6 / 47.0`（§7.4）、`57.7%`（vOPD 墙钟节省）、`11.5`（AntiSD）、`26.13% @0.6B`（SOD）、`79.6%`（KAT-Coder-V2 SWE-bench）、`8.1×`（NPD）、`68.75 / 67.08`（ROPD）、`84.1%`（MAIGO）、`27.1%`（MPD）。
- **Table 1 与 Table 2 由我本人逐行读过原文**再与笔记比对，一致。
- **Table 3/4/5 抽查了特征单元格**（`gradient-faithful`、`Curvature-aware`、`Advantage-sign`、`entropy-dynamic`、`reward-density`、`LongBlocks`、`contrastive pMI`、`Speculative-decoding-ratio`、`batch-normalized`、`cognizance`、`hindsight leakage`），均存在于原文。
- 一处抽取损坏已被修正：原文 JSD 生成元公式在 PDF 抽文本时分数被拉平，摘要按数学语义写作 $f(u)=u\log u-(u+1)\log\frac{u+1}{2}$。

**未核验（请按"待复核"对待）**
- 除上表方法外的**其余散文叙述未逐句回查原文**；子代理在长段落中有可能存在轻微压缩或措辞偏移。
- Table 3/4/5 的**逐格内容**只做了抽样核验，未逐格比对。
- 本摘要**未覆盖参考文献部分**（约 920 行），也未做引用完整性检查。

**原文自身声明的局限（§10）**
- 范围：只收"训练中含学生 on-policy 生成"的方法；排除一般 KD（feature-based、attention transfer）、正交压缩（pruning / quantization）、纯推理时方法（self-consistency、majority voting）。
- 时效：覆盖截至 **2026 年 6 月**，而该领域每月新增 10+ 篇。
- 可复现性：见 §11 最后一条 —— 跨方法比较不可靠，表格报的是原论文数字。

**本地文件**
- 论文原文：`2604.00626v4.pdf`
- 抽取的纯文本：`/tmp/paper.txt`（临时文件，重启后可能被清理）
- 分块笔记：`.qwen/tmp/parts/*.md`（中间产物，可删）
