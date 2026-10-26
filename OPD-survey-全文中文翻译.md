# A Survey of On-Policy Distillation for LLMs —— 全文中文翻译

> **原文**：Mingyang Song & Mao Zheng（Tencent, Large Language Model Department）. *A Survey of On-Policy Distillation for Large Language Models*. Preprint, arXiv:2604.00626v4 [cs.LG], 18 Jun 2026. 共 89 页（正文 §1–§10 = PDF 第 1–72 页；参考文献自 PDF 第 76 页起）。
>
> **译文说明**
>
> - 这是**正文 §1–§10 的逐段完整中文翻译**；参考文献按约定**不译**，以原始英文列在文末。
> - **术语、方法名、模型名、benchmark 名、作者年份引用一律保留英文**（OPD、f-divergence、forward KL、logit、πmix、GRPO、GKD、Qwen3、AIME、SWE-bench…），便于对照检索。
> - 公式以 **LaTeX 原文符号**保留；PDF 抽取时被拆行的分数/上下标已按数学语义还原。
> - 小节标题为「中文标题（English Title）」，编号与原文一致。
> - 形如 `<!-- p. 15 -->` 的注释是**原文页码标记**（渲染时不可见），便于与 PDF 对照。
> - 原文在页面切换处本就存在断句，译文保持原样、未作改写。
> - 抽取不清处标注为 `> ⚠️ 原文此处抽取不清：<原文片段>`。
>
> **生成方式**：PDF 经 pypdf 抽取文本后，按章节与页边界切成 30 块（并保证 9 张表格不被切断），由 5 个并行代理逐块翻译，随后按原文顺序拼接并修复接缝。**已知局限**：分块翻译可能在个别接缝处留下轻微语气不连贯；摘要性引用（如表格中的"一句话主张"）以原文为准。

---

<!-- p. 1 -->

**原文标题：A Survey of On-Policy Distillation for Large Language Models（大语言模型在线策略蒸馏综述）**

Mingyang Song & Mao Zheng

Large Language Model Department

Tencent, China

nickmysong@tencent.com

/gtbnick7nlp/Awesome-LLM-On-Policy-Distillation

## 摘要（Abstract）

随着 Large Language Models (LLMs) 在能力与成本两方面双双增长，知识蒸馏已成为把前沿能力迁移到更小、可部署的学生模型中的标准技术。当前主流方案——对教师生成文本的 off-policy 模仿——存在一个结构性弱点：学生在无瑕的教师前缀上训练，但在推理时却必须从自身不完美的输出出发生成，由此产生的 exposure bias 会沿轨迹不断累积。On-Policy Distillation (OPD) 通过让学生生成自己的序列、并让教师对这些序列给出反馈来应对这一失配，从而把蒸馏从单遍模仿转变为迭代式纠错过程。该方法已迅速从研究走向生产，但相关文献仍散落在知识蒸馏、强化学习和模仿学习等不同社群之中，缺乏统一的处理。本综述提供了这样的处理。我们把 OPD 形式化为在学生采样轨迹上的 f-divergence 最小化，沿三条设计轴（目标函数、信号来源与训练动力学）组织该领域，并综合成功条件、反复出现的失效模式，以及 OPD 与 KL 约束强化学习之间的形式化联系。最后我们给出待解问题，包括蒸馏 scaling law、不确定性感知的教师反馈以及智能体级蒸馏。

## 1 引言（Introduction）

Large Language Models (LLMs) 的能力在过去两年中快速提升，近期系统在推理、代码生成和多语言 benchmark 上达到了很强的表现（Yang et al., 2025a; DeepSeek-AI et al., 2025; Gemma Team et al., 2024）。这一进展伴随着只有少数组织才能承受的训练成本。因此，把由此获得的能力从大规模前沿模型迁移到更小、可部署的学生模型，已成为现代 LLM 流水线中的核心步骤。知识蒸馏最初由 Hinton et al. (2015) 形式化，其目的是让学生继承教师的软输出结构，在此期间其范围不断拓宽。它最初只是一种在同一架构内迁移知识的模型压缩技术，如今则成为跨模型规模与架构族迁移能力的通用机制。DeepSeek-R1（DeepSeek-AI et al., 2025）的发布让这一转变变得具体：它把一个 671B 的 mixture-of-experts 教师蒸馏为参数规模从 1.5B 到 70B 的稠密学生模型，同时基本完整地保留了长 chain-of-thought 推理能力。

然而，这些成功背后的主流方案存在一个已知的结构性弱点。在几乎所有大规模流水线中，蒸馏都是以 off-policy 方式进行的。学生被训练去在固定语料上匹配教师的下一 token 分布，该语料通常是原始预训练数据或教师事先生成的轨迹，而每一个梯度步都以无瑕的教师前缀为条件。在推理时，这一假设不再成立。学生要从自身的部分输出出发自回归地生成，而训练状态与其在部署时所访问状态之间的任何差异，都会沿轨迹转化为 compounding

arXiv:2604.00626v4  [cs.LG]  18 Jun 2026

<!-- p. 2 -->

error。这种分布失配对应交互式模仿学习中所识别的 compounding error（Ross et al., 2011）：自回归生成会使分布失配按序列长度平方级放大，在长度为 T 的时间跨度上产生 $O(\epsilon T^2)$ 的期望偏差。其后果在推理任务上最为突出，因为一个早期的失误就可能让整个证明或程序脱轨。Gudibande et al. (2023) 报告称，对专有 LLM 的 off-policy 模仿往往得到的学生只复刻了表面风格，而未获得底层推理能力。

解决这种失配，需要改变训练数据来自哪里，而不是改变要匹配什么。On-Policy Distillation (OPD) 的做法是把采样分布从静态语料转移到学生自身不断演化的策略上。学生提出轨迹，填充它部署时将要访问的状态，并在这些状态上——而不是在理想化的专家前缀上——接受教师反馈。视可获得的教师访问权限而定，这种反馈的范围从白盒设定下的完整 token 级分布（Agarwal et al., 2024; Gu et al., 2024），到仅有黑盒 API 可用情形下的标量奖励或成对偏好（Ye et al., 2025; Jiang et al., 2023），再到无教师设定下模型从自身输出中导出的对比信号（Zhao et al., 2026b）。其理论收益与经典的 DAgger 结果（Ross et al., 2011）相呼应：在学习者自己访问过的状态上向专家查询，可把 off-policy 模仿的 $O(\epsilon T^2)$ 复合误差降为 $O(\epsilon T)$。翻译到自回归语言建模中，这就把蒸馏从一个单遍匹配问题变成了一个迭代优化循环：训练分布与模型共同演化，使得训练与部署之间的分布差距可以在训练期间被修正，而不是被推迟到推理阶段。

由这一重新表述衍生出的工作体系，可以围绕少数几个反复出现的设计选择来组织。早期方法关注如何采样、以及在学生轨迹上施加哪一种 divergence（Agarwal et al., 2024; Gu et al., 2024; Ko et al., 2024），并确立了合适的 divergence 取决于在序列的哪个位置进行训练、以及教师能提供何种监督。一条并行的工作线揭示了所得目标在形式上等价于 KL 约束形式的强化学习（Yang et al., 2026e），从而连接起此前彼此分离的两个研究社群。第三条分支表明“教师”根本不必是外部的：OPSD（Zhao et al., 2026b）与 SD-ZERO（He et al., 2026）等方法显示，模型可以通过利用其输出分布在不同上下文中的不对称性，从自己的生成中自举式地获得提升。这些想法此后已从论文走进大规模训练流水线：Qwen3（Yang et al., 2025a）、DeepSeek-V4（DeepSeek-AI, 2026）、Gemma 2（Gemma Team et al., 2024）和 MiMo-V2-Flash（Xiaomi LLM-Core Team et al., 2026）都把 OPD 作为核心训练要素。这种采用跨越了架构界线，既覆盖 mixture-of-experts 也覆盖稠密系统，并横跨专有与开放权重两种发布策略，因为一旦推理链长到足以让复合误差占主导，所有这些系统都会面临同样的结构性问题。DeepSeek-V4 比多数系统走得更远，它用纯多教师 OPD 取代了其混合 RL 阶段，以完成模型整合。

GKD（Agarwal et al., 2024）与同期出现的 MiniLLM（Gu et al., 2024）在 2023 年年中把 on-policy distillation 带入 LLM 时代；在此后约三年间，文献已扩展到两百多篇论文，涵盖 divergence 设计、奖励引导的优化、self-play、多教师辩论、智能体轨迹蒸馏以及跨模态迁移。研究焦点也随之转移。早期工作问的是 on-policy 训练到底有没有用，当前前沿问的则是：如何让它高效到足以支撑万亿参数教师的推理，如何把它从单轮生成扩展到多步智能体轨迹，以及如何把它与日益共享其计算基础设施的强化学习流水线统一起来。这一轨迹与 LLM 发展中更广泛的转向平行。随着模型能力更强、推理链更长，off-policy 训练状态与部署时状态之间的差距会扩大，这使得 on-policy 修正对下一代具备推理能力的系统越来越有吸引力（Fu et al., 2026b; DeepSeek-AI, 2026）。

尽管采用如此迅速，关于 OPD 的研究文献尚未跟上实践。各种方法从知识

<!-- p. 3 -->

蒸馏、RLHF 与交互式模仿学习等相互分离的视角切入同一问题，各自带着来自其母社群的不同记号、benchmark 与失效分类法。现有的 LLM 蒸馏综述（Xu et al., 2024）总体上仍保留经典压缩框架，把 off-policy 与 on-policy 方法视为可互换的变体，而不是视作在理论性质与失效模式上存在实质差异的不同机制。据我们所知，目前还没有工作给出统一刻画把这些线索联系起来的 on-policy 动力学的数学描述，而研究人员与工程师如今在实践中必须相互权衡的白盒、黑盒与无教师三类实例之间的系统比较也仍然缺失。

本综述正是为了填补这一空白。据我们所知，它是首个把 on-policy distillation 视为一种独立范式、而非经典知识蒸馏之变体的工作，并通过四项贡献给出统一论述，这四项贡献也构成了本文其余部分的结构。

- **统一理论框架**。我们把从 off-policy 到 on-policy 蒸馏的转变重新表述为一个序贯决策问题，并表明核心 OPD 算法都是在学生采样轨迹上做 f-divergence 最小化的实例（第 2 节）。该框架为此前被孤立研究的方法提供了共同的分析词汇，并从 divergence 选择、参数顺序与采样混合的角度澄清了它们之间的关系。
- **以设计为中心的分类法**。我们不按表面相似性对方法分组，而是沿三条核心设计轴组织它们（第 3 节），即目标函数设计（第 4 节）、信号来源架构（第 5 节）与训练动力学优化（第 6 节）。由此得到的“一方法一类别”分类，从每篇论文所优化的具体设计选择出发，澄清了方法之间的关系。
- **对失效的理论论述**。我们把经验记录整合成关于 OPD 何时以及为何失效的连贯观点（第 7 节），涵盖有缺陷前缀陷阱（flawed prefix trap）、self-play 饱和、多样性坍缩、校准—能力差距以及新出现的多轮智能体失效模式，并描述 OPD 优于 off-policy 替代方案的若干形式化条件。
- **跨机制比较分析**。我们在这一共同框架内考察白盒、黑盒与自蒸馏机制，在各节的比较表中对它们加以总结（表 1、表 3–9 和表 2），并记录把部署约束映射到适用方法与算力预算的方法选择因素（第 3.3 节）。

本文的结构遵循同样的设计轴逻辑。第 2 节展开数学预备知识，把自回归生成形式化为序贯决策问题，并推导出促成 on-policy 训练的 exposure bias。第 3 节给出方法全景与选择考量。第 4–6 节随后依次走过三条设计轴，从“优化什么”到“信号来自哪里”再到“如何在实践中稳定训练”，每一节都建立在前一节所浮现出的选择之上。第 7 节整合各理论视角，表明第 4–6 节所观察到的模式可由少数统一原则推出。第 8 节讨论工业部署、系统级优化与新兴领域，第 9 节列出由本综述综合而出的待解问题。

我们的范围以 on-policy 机制为中心。我们涵盖学生在蒸馏过程中自己生成训练数据的方法，并纳入来自模仿学习、在线 RL 与偏好优化等相邻领域的工作——只要它们是在教师或验证器监督下、基于学生生成的数据运作。我们排除通用的 off-policy KD、与训练机制正交的压缩技术（如剪枝与量化），以及不改变模型权重的推理时方法。

## 2 背景与统一数学（Background and Unified Math）

为了理解为何 on-policy 训练往往更可取，我们形式化自回归生成的复合几何结构。我们先从经典 KD 及其局限开始，随后建立统一了现代 OPD 目标的 f-divergence 框架。 
**定义"on-policy"（Defining "on-policy"）。** 如果学生在训练时所使用的训练数据是在训练时从学生自身当前策略 pθ 采样得到的，而不是来自固定的外部语料 D 或教师的生成分布 pT，那么这个蒸馏方法就是 on-policy 的。形式上，on-policy 训练优化的目标是：

$$
\min_{\theta}\ \mathbb{E}_{x\sim D}\ \mathbb{E}_{y\sim p_{\theta}(\cdot|x)}\left[L(y,x;\theta,T)\right]
$$

其中 L 表示蒸馏损失（散度、奖励或混合形式）。关键区别在于，外层期望是对学生自身的生成取期望，而不是对一个静态数据集取期望。这就造成了一个非平稳的优化地形，因为随着 θ 更新，数据分布 pθ 会随之漂移，从而要求在每一个训练步都进行新的 rollout。这种重新生成所带来的计算开销是 OPD 的一个核心系统级挑战（第 6.3 节）。

**记号（Notation）。** 在本综述全文中，我们用小写 p 表示 token 级的条件分布（教师为 pT(·|x, y<t)，学生为 pθ(·|x, y<t)），用大写 P 表示序列级分布（PT(y|x)、Pθ(y|x)）。D 表示训练数据集，pdata 表示数据分布，|V| 表示词表规模。DKL 与 Df 分别表示 KL 散度和一般 f-divergence。在 RL 形式化中，π 表示策略（例如 πmix 表示混合策略，πteach 表示 DPO 式推导中的序列级教师策略，πref 表示参考策略），R(x, y) 表示结果奖励。在讨论各个具体方法时，凡清晰性要求之处，我们遵循每篇论文自己的记号约定。

### 2.1 经典知识蒸馏（Classical Knowledge Distillation）

最初的 KD 形式化（Hinton et al., 2015）训练学生去匹配教师经温度软化后的输出分布：

$$
p_T^{(\tau)}(y|x)=\frac{\exp(z_y/\tau)}{\sum_j \exp(z_j/\tau)}
$$

其中 zy 是教师的 logit，τ>1 是温度参数。适中的温度（τ>1）会暴露出教师关于类间相似性的"暗知识"（dark knowledge），而硬标签会把这些信息隐藏起来。在 τ→∞ 的极限下，分布趋近于均匀分布，这种结构也随之丢失。蒸馏损失对学生 logit zS_i 的梯度说明了温度为何重要：

$$
\frac{\partial L_{\mathrm{KD}}}{\partial z_i^S}=\frac{1}{\tau}\left(p_i^S-p_i^T\right)
$$

在高温极限下（τ→∞），泰勒展开 $\exp(z_i/\tau)\approx 1+z_i/\tau$ 连同零均值假设 $\sum_i z_i^S=\sum_i z_i^T=0$ 一起，把梯度化简为 $\frac{1}{|V|\tau^2}(z_i^S-z_i^T)$（Hinton et al., 2015），因此在此区间内经典 KD 等价于最小化原始 logit 之间的均方误差。这鼓励学生复制教师**完整**的 logit 结构，既迁移主要的预测模式，也迁移细微的次优概率质量（即"暗知识"）。因子 1/τ² 解释了为什么在组合式 KD 目标中，软目标项相对于硬标签交叉熵通常会被按 τ² 加权。在实践中，LLM 蒸馏工作在 τ= 1 或低到中等温度下（Zhong et al., 2024），因为巨大的词表（|V|> 30,000）本身就已经产生了结构丰富的非峰值概率，无需显式软化；而更高的温度可能会放大教师校准不佳的尾部分布中的噪声。

对于自回归 LLM，token 级蒸馏损失可分解为：

$$
L_{\text{Token-KD}}=\mathbb{E}_{x,y\sim D}\left[\sum_{t=1}^{|y|}D_{\mathrm{KL}}\left(p_T(\cdot|x,y_{<t})\,\|\,p_{\theta}(\cdot|x,y_{<t})\right)\right] \tag{1}
$$

这在计算上是可处理的，因为它依赖于数据集中静态、预先算好的前缀，但它是在假设历史 y<t 无错误的前提下优化下一 token 的准确率。Kim & Rush (2016) 将其扩展为序列级知识蒸馏（Sequence-Level Knowledge Distillation, Seq-KD），后者转而在整个序列空间上最小化全局 KL 散度：

$$
L_{\text{Seq-KD}}=D_{\mathrm{KL}}\left(P_T(y|x)\,\|\,P_{\theta}(y|x)\right)=\sum_{y\in Y}P_T(y|x)\log\frac{P_T(y|x)}{P_{\theta}(y|x)}
$$

<!-- p. 5 -->

由于 |Y|=|V|^T 随长度指数增长，精确计算是不可行的。Seq-KD 用在 beam-search 输出 $\hat{y}=\arg\max_y P_T(y|x)$ 处的 Dirac delta 来近似 PT(y|x)，从而退化为在教师生成的序列上做标准的负对数似然：

$$
L_{\text{Seq-KD}}\approx-\log P_{\theta}(\hat{y}|x)
$$

当教师高度自信时，这种近似表现良好，但它丢弃了关于教师不确定性和替代模式的大量信息。其实际后果有两方面。Seq-KD 把教师的分布丰富性坍缩为单一的点估计；并且它仍然在一个与学生的生成行为相脱节的**静态**数据集上训练。这两个局限（分布信号的丢失与 off-policy 训练）正是下文更丰富的目标的动机所在。

后续工作（Zhong et al., 2024）针对现代 LLM 重新审视了这些基础，并指出在跨异构模型家族做蒸馏时，经典假设（教师-学生差距小、共享词表、相似的架构归纳偏置）往往不成立。经典假设与现代现实之间的落差正是下文更一般的 f-divergence 框架的动机。

**从经典 KD 到 on-policy（From classical KD to on-policy）。** 从 Hinton 的经典 KD 到现代 OPD 的历史演进，本质上是对四个经典假设的一系列放松，即（1）共享词表，（2）i.i.d. 数据，（3）静态教师，（4）off-policy 训练数据。Seq-KD（Kim & Rush, 2016）通过在序列级别操作放松了假设（2），但保留了静态数据假设。GKD（Agarwal et al., 2024）通过在学生生成的序列上训练放松了假设（4）。DSKD（Zhang et al., 2025b）通过双空间对齐放松了假设（1）。G-OPD（Yang et al., 2026e）同时放松了假设（2）–（4），并增加了 RL 增强的奖励外推，把学生推向教师的前沿之外。每一次放松都针对一个随模型规模增大而加剧的特定失效模式。词表不匹配在同族蒸馏中往往无关紧要，但在跨族迁移中非常严重（放松 1）；而暴露偏差（exposure bias）在短序列上可忽略不计，在多步推理中却极具危害（放松 4）。这一演进过程不仅仅是历史叙述，而且具有信息价值：某一给定部署场景中的相关假设决定了需要哪一级别的放松，而不必要的放松会引入本可避免的复杂度。

### 2.2 Off-policy 暴露偏差（Off-Policy Exposure Bias）

式 (1) 中的 token 级损失以静态数据集提供的真实前缀 y<t 为条件。在推理时，学生生成自己的前缀，从而在训练与部署之间造成分布不匹配，且这种不匹配会随序列长度不断累积。下面我们形式化地说明这种不匹配为何是灾难性的。

标准知识蒸馏最小化在来自数据集分布 dD(s) 的状态上的 KL 散度：

$$
L_{\text{Off-Policy}}=\mathbb{E}_{x,y\sim p_{\text{data}}}\left[\sum_{t=1}^{|y|}D_{\mathrm{KL}}\left(p_T(\cdot|x,y_{<t})\,\|\,p_{\theta}(\cdot|x,y_{<t})\right)\right]
$$

在推理时，学生按照自己的策略行动，从而诱导出不同的状态访问分布 dpθ(s)。根据 DAgger 定理（Ross et al., 2011），如果一个策略在训练分布下以每步误差 ϵ 模仿专家，那么在学习者自身的状态访问下，长度为 T 的轨迹上的期望总偏差按 $O(\epsilon T^2)$ 规模增长。

这种二次累积对推理任务有严重的实际后果。即使在最简单的独立误差假设下，一个需要 10 个推理步骤、每步准确率 95% 的数学证明也只能得到 (0.95)^10≈60% 的轨迹级正确率。而 DAgger 界预测的结果严格更差。因为 off-policy 训练很少把学生暴露在自己的错误状态上，每一个错误都会把模型推入其下一步准确率进一步退化的区域（分布漂移），从而产生 $O(\epsilon T^2)$ 的总误差，而不是独立累积情形下的 $O(\epsilon T)$。On-policy 训练提供了一条缓解这种自我强化动态的途径，即在训练期间把学生暴露在它自己的错误状态上，使其能够学到恢复策略，从而阻止分布漂移把局部错误放大为轨迹级崩溃。

<!-- p. 6 -->

**关于 LLM 中 DAgger 界的注记（Remark on the DAgger bound in LLMs）。** 尽管理论上从 $O(\epsilon T^2)$ 降到 $O(\epsilon T)$ 的改进很吸引人，但把 DAgger 界应用于 LLM 需要谨慎的限定。原始定理假设存在一个**交互式专家**，它在任何状态下都会给出最优动作。在白盒 OPD 中，"专家"给出的下一 token 分布 pT(yt|ˆy<t) 是以学生的前缀 ˆy<t 为条件的。如果学生幻觉出一个严重分布外的前缀，教师的条件分布本身也可能变得校准不佳，因为教师很少在这样的输入上被训练过。强迫学生匹配这个带噪声的分布违反了交互式模仿学习的核心假设，可能使训练不稳定，而不是恢复到 $O(\epsilon T)$ 的界。Jeong (2026) 提供了直接的经验证据，表明不具备稳定教师动态的朴素 OPD 会遭遇灾难性的不稳定。无条件的 token 级匹配加上周期性的硬拷贝教师重置会导致输出突然崩溃（在单次教师重置事件中，KL 散度从 2.637 降到 0.343），这说明仅靠无条件的 token 级匹配不足以实现稳定的 OPD。这一观察正是第 6.1 节所综述的自适应信任机制以及第 7.2 节所讨论的稳定教师动态的动机。

### 2.3 统一的 f-divergence 框架（The Unified f-Divergence Framework）

OPD 通过把训练期望改为从学生自身的 rollout，或从一个混合策略 πmix 采样，来解决暴露偏差问题。广义的 on-policy 目标把采样轨迹与局部匹配度量解耦：

$$
L_{\text{OPD}}(\theta)=\mathbb{E}_{y\sim\pi_{\text{mix}}}\left[\sum_{t=1}^{|y|}D_f\left(p_T(\cdot|x,y_{<t}),p_{\theta}(\cdot|x,y_{<t})\right)\right]
$$

这里 Df 表示来自 f-divergence 族的一种散度（Wen et al., 2023）。形式上，给定两个分布 P 和 Q，f-divergence 定义为：

$$
D_f(P\|Q)=\mathbb{E}_{y\sim Q}\left[f\left(\frac{P(y)}{Q(y)}\right)\right]
$$

其中 f:(0, ∞)→R 是一个凸生成函数，且 f(1) = 0。f 的选择决定了对似然比 pT(y)/pθ(y) 的隐式加权：

- **Forward KL**（$f(u)=u\log u$）：覆盖模式（mode-covering，避免零概率，zero-avoiding）。学生被鼓励在教师分配概率质量的任何地方都分配概率质量，这往往会跨越教师的不同模式之间的空隙，并在模式之间的区域产生幻觉。
- **Reverse KL**（$f(u)=-\log u$）：寻找模式（mode-seeking，强制为零，zero-forcing）。学生把概率质量集中到教师模式的一个子集上，避开教师赋予接近零概率的区域，但可能忽略次要的可接受输出。
- **JSD**（$f(u)=u\log u-(u+1)\log\frac{u+1}{2}$）：对称、有界，并在覆盖模式与寻找模式的行为之间平滑插值。
- **α-divergence**：一个参数化族，在 Forward KL（α→ 1）与 Reverse KL（α→ 0）之间连续插值，允许对寻找模式/覆盖模式的权衡做细粒度控制。

具体而言，这些散度各自适合不同的任务几何。对于具有**唯一**正确答案的任务（数学证明、代码正确性），Reverse KL 的寻找模式行为会把学生集中到一条合适的求解路径上。对于存在**多种**可接受输出的任务（创意写作、开放式问答），Forward KL 的覆盖模式行为会保留输出的多样性。JSD 提供了一个折中，但对两种极端情形都缺乏理论上的最优性性质。以上所有散度都可以写成学生自身策略下的期望（因为 $D_f(P_T\|P_{\theta})=\mathbb{E}_{y\sim p_{\theta}}[f(p_T(y)/p_{\theta}(y))]$），因而可以直接用于 on-policy 优化。采样分布 pθ 正是被优化的策略，因此梯度可以直接通过重参数化后的样本流动，而不需要重要性权重或会额外引入方差的 off-policy 修正。 
该框架涵盖了许多现代 OPD 目标。f 的选择决定几何行为，而 πmix 的选择控制 on-policy 探索的程度。我们通过把三种基础方法映射进这一空间来加以说明。

GKD (Agarwal et al., 2024) 把 $\pi_{\text{mix}}=\lambda p_\theta+(1-\lambda)p_{\text{data}}$ 定义为显式插值，且对 D_f 不作假设，经验性地测试了 Forward KL、Reverse KL 与 JSD。令 λ→1 使 GKD 变为纯 on-policy，而 λ=0 则退化为标准的 off-policy KD。λ 参数提供了两个极端之间的平滑调节旋钮，中等取值（λ≈0.5）往往能把 on-policy 曝光的好处与有据可依、源自数据集的 prefix 所带来的稳定性结合起来。GKD 的实验报告称，在他们测试的各种散度选择下，on-policy 采样（λ=1）都优于 off-policy。JSD 在翻译任务（如 WMT）上表现最好，因为这类任务的输出空间具有中等多样性（既不是唯一正确答案，也不是完全开放式的生成），而 JSD 的对称性天然平衡了 mode-covering 与 mode-seeking 两种压力，恰好适合这一中间状态。

在经验上，三种散度在摘要与指令遵循任务上取得了可比的结果，这支持了 exposure bias 假说。这表明，当任务几何并不强烈偏向某一极端时，采样策略（on-policy 与 off-policy）比具体的散度选择更重要。

MiniLLM (Gu et al., 2024) 选择 Reverse KL 作为 D_f = D_KL(p_θ∥p_T)，并采用 $\pi_{\text{mix}}=(1-\alpha)p_\theta+\alpha p_T$，取 α=0.2 以保证稳定性。由于 Reverse KL 同时把学生放在期望与 log-ratio 之中，MiniLLM 通过 REINFORCE 重新表述优化，把 log(p_T/p_θ) 当作每步的 reward。

DistiLLM (Ko et al., 2024) 在学生生成的 prefix 上计算其 token 级损失，但使用一个自适应调度器，依据验证损失保守地使用学生生成的输出。它还维护一个 replay buffer 以提升样本效率（一种“自适应 off-policy”策略）。其核心贡献是一种偏斜 KLD 损失（skew KLD loss），使用混合分布 $\tilde{p}=\alpha p+(1-\alpha)q$，计算 $D_{KL}(p\|\tilde{p})$（SKL）或 $D_{KL}(q\|\tilde{p})$（SRKL）。当 pθ(y)≈0 或 pT(y)≈0 时，这避免了除零不稳定，因为 $\tilde{p}(y)$ 有下界远离零，从而在不使用 policy gradient 的情况下提升了可处理性与稳定性。

对比性综述（Comparative synthesis）。这三种方法暴露出 OPD 设计中的一个核心权衡三角。GKD 优先考虑通用性（对散度不作假设）与简洁性（没有 REINFORCE 的方差），代价是在散度选择上缺乏理论指导。MiniLLM 为获得 mode-seeking 的精确性而选定 Reverse KL，但继承了 REINFORCE 的高方差，需要 reward baseline、长度惩罚以及细致的 α 调参。DistiLLM 通过设计一个构造上数值稳定的混合目标，同时避开了散度选择问题与 MiniLLM 的方差问题，代价是引入 replay buffer（混合 on-policy 与 off-policy 数据）以及一个带来额外超参数的自适应调度器。因此，从 GKD 经 MiniLLM 到 DistiLLM 的这条线索，勾勒出从算法简洁性向计算与工程精巧性的转移，较晚的方法往往针对较早方法的具体局限。

从这一分解来看，现代 OPD 是对三个相互作用的设计选择的系统性探索：轨迹采样（πmix）、散度生成函数（f），以及所选散度内部的参数顺序。

### 2.4 蒸馏缩放定律（Distillation Scaling Laws）

统一的 f -divergence 框架刻画了要优化什么（what to optimize），但没有回答性能如何随计算投入而缩放（how performance scales）。与预训练不同——在预训练中 Hoffmann et al. (2022) 建立了计算最优缩放定律（Chinchilla 定律）——蒸馏产生的是一个性质不同的资源分配问题。预算通常被拆分给教师推理、学生 rollout 与梯度更新，而这些轴之间存在不平凡的相互作用。

Busbridge et al. (2025) 开启了针对蒸馏的缩放研究，并发现师生关系遵循与单独预训练性质不同的动力学。他们发现，当教师相对学生变得过于强大时，会出现容量差（capacity gap）。

<!-- p. 8 -->

学生的损失关于教师交叉熵遵循幂律，并在两种状态之间发生转变。起初，更强的教师会改善学生（来自校准更好的下一 token 概率的更密集梯度信号），但超过某个临界教师容量后，学生吸收该信号的能力会饱和。这一容量差现象与 2.3 节的散度选择直接相关：Forward KL 的 mode-covering 压力会加剧这一差距（学生必须把概率质量铺展到教师的所有 mode 上，包括超出其容量的那些），而 Reverse KL 的 mode-seeking 行为则部分缓解它（学生选择性地匹配其最高置信度的子集）。

目前尚无综合框架能预测蒸馏质量如何随教师规模 NT、学生规模 NS、数据量 D 以及 on-policy rollout 预算 R 联合缩放。对 Busbridge et al. (2025) 缩放定律的一个自然扩展可能形如 Quality∝N_T^α·N_S^β·D^γ·R^δ，但确定其中的指数与交互项对 on-policy 方法而言仍是开放问题（第 9 节）。rollout 预算 R 是 on-policy 蒸馏特有的，它增加了一个预训练与 off-policy 蒸馏都不具备的新缩放轴。每个 rollout 步骤都需要一次完整的学生前向传播，使计算成本随 R 线性增长，而学习收益很可能呈现递减回报。

## 3 领域全景与方法选择（Landscape and Method Selection）

在详细考察各个方法之前，本节对该领域作结构化概述，涵盖组织 taxonomy、代表性方法的对比表，以及对方法选择考量因素的讨论。

### 3.1 方法全景（Method Landscape）

现有的 OPD 方法可以沿着三个轴来组织，它们对应训练流程中依次做出的设计决策：(1) 要优化的目标函数，(2) 监督信号的来源，(3) 稳定训练动力学的机制。对于信号来源轴，我们采用两级分类。第一级区分外部教师蒸馏（external teacher distillation）——由另一个通常更强的模型提供监督——与自蒸馏（self-distillation）——模型自己生成训练信号。在外部教师蒸馏内部，第二级按访问级别区分方法：白盒方法需要教师的完整输出分布，而黑盒方法只依赖生成的文本或标量分数。这种嵌套结构把教师身份（外部 vs. 自身）与教师访问级别（白盒 vs. 黑盒）放在不同层级，而不是把它们当作同级类别，因为前者是方法论选择，而后者是部署约束。图 1 展示了这一组织方式。

这三个轴映射到统一 OPD 目标的独立项。f -divergence 生成函数 f 支配目标函数（轴 1，第 4 节），决定学生是 mode-seeking、mode-covering 还是自适应的。提供 pT(·|x, y<t) 的实体的身份与访问级别支配信号来源（轴 2，第 5 节）：是具有完整 logit 访问权限的外部教师、只有文本或分数输出的外部教师，还是模型自身条件性的 privileged 分布。采样策略及其调度支配训练动力学（轴 3，第 6 节），包括插值系数 λ、prompt 难度的课程安排以及 token 级加权。第一个轴在很大程度上与其他两个轴解耦，因为任一给定的散度都可以与大多数采样调度、以及任一种教师身份搭配。信号来源与目标函数只部分解耦，因为黑盒访问去掉了 token 级分布，从而把可行的散度限制为序列级替代目标。因此，把这三者当作独立的轴来处理，能够组织设计空间，但并不意味着每种组合都可实现。

为了说明该 taxonomy 如何映射到具体系统，GKD (Agarwal et al., 2024) 选择 forward KL（或 JSD）作为其散度目标，依赖白盒教师 logit 作为信号来源，并在学生生成序列与真值序列之间插值（πmix）以稳定 on-policy 训练动力学。DeepSeek-R1 蒸馏模型（DeepSeek-AI et al., 2025）虽然是 off-policy 的（在静态教师轨迹而非学生 rollout 上训练），

<!-- p. 9 -->

**图 1 的分类树（以下为图中节点标签；方法名保留英文）**

- **§4 目标函数与优化（Objective Functions and Optimization）**
  - **4.1 固定散度目标（Fixed Divergence Objectives）**：GKD (Agarwal et al., 2024), MiniLLM (Gu et al., 2024), DistiLLM (Ko et al., 2024), DistiLLM-2 (Ko et al., 2025), KETCHUP (Fan et al., 2025), vOPD (Oh et al., 2026), AntiSD (Shen et al., 2026a), OPD+ (Zhao et al., 2026a), Bridging (Jiang & Ferraro, 2026), Decomposed-OPD (Yoon et al., 2026), Surgical-PT (Lin & Han, 2026), Distributional DAgger (Agrawal et al., 2026)
  - **4.2 自适应散度目标（Adaptive Divergence Objectives）**：EOPD (Jin et al., 2026b), AOPD (Jia et al., 2026), Position-Weighted OPSD (Liu et al., 2026d), Token-Teachability (Wang et al., 2026j), Lookahead Group Reward (Liu et al., 2026f), Trust-Region OPD (Xing et al., 2026)
  - **4.3 RL 增强目标（RL-Augmented Objectives）**：G-OPD (Yang et al., 2026e), RLKD (Xu et al., 2025b), KDRL (Xu et al., 2025a), RLAD (Zhang et al., 2026o), SuperCorrect (Yang et al., 2025b), SCoRe (Lyu et al., 2025), AlignDistil (Zhang et al., 2025a), X-KD (Cai & Yuan, 2026), TSD-KD (Kim & Baek, 2026), REOPOLD (Ko et al., 2026), CMDP-KD (Zimmer et al., 2025), CoDistill-GRPO (Kwon et al., 2026), dGRPO (Ramos et al., 2026), Sparse-to-Dense (Xu et al., 2026d), KEPO (Yang et al., 2026b), TGPO (Liu et al., 2026e), AMR-SD (Wei et al., 2026), OPPO (Li et al., 2026k), StepOPSD (Zhang et al., 2026j), Self-Evaluation (Zhang et al., 2026h)
- **§5 信号来源与教师架构（Signal Source and Teacher Architecture）**
  - **§5.1 外部教师蒸馏（External Teacher Distillation）**
    - **§5.1.1 白盒 Logit 监督（White-Box Logit Supervision）**
      - **同族（Same-Family）**：MAD-OPD (Wang et al., 2026c), MPD (Yang et al., 2026c), BRTS (Zhang et al., 2026e), Pair-In, Pair-Out (Tan et al., 2026)
      - **跨族（Cross-Family）**：Veto (Jang et al., 2026), PromptKD (Kim et al., 2024), DSKD (Zhang et al., 2025b), CSD (Kim et al., 2026c), SimCT (Sun et al., 2026), DuDi (Payoungkhamdee et al., 2026)
    - **§5.1.2 黑盒与 API 受限（Black-Box and API-Constrained）**：Lion (Jiang et al., 2023), GAD (Ye et al., 2025), OVD (Xiong et al., 2026), LUFFY (Yan et al., 2025), DASD (Yan et al., 2026), DDT (Zhang et al., 2026f), PRISM (Wang et al., 2026g), ROPD (Fang et al., 2026), ORPO-Distill (Singh et al., 2025), OmniOPD (Zhou et al., 2026b)
  - **§5.2 自蒸馏（Self-Distillation）**
    - **5.2.1 特权信息（Privileged Information）**：OPSD (Zhao et al., 2026b), GATES (Stein et al., 2026), π-Distill (Penaloza et al., 2026), OPCD (Ye et al., 2026b), OEL (Ye et al., 2026a), HDPO (Ding, 2026), CRISP (Sang et al., 2026), OPSDL (Zhang et al., 2026g), GUI-SD (Zhang et al., 2026i), MSD (Qin et al., 2026), PBSD (Yu et al., 2026c), TT-OPD (Jeong, 2026), VISD (Lin et al., 2026a), OPHSD (Zhao et al., 2026c), COPSD (Liu et al., 2026h), ATESD (Han et al., 2026b), TRACE (Wang et al., 2026e), AVSD (Nguyen et al., 2026), Skill-Cond. SD (Huang et al., 2026), Critique-Distill (Jin et al., 2026a), Constitutional Cross-SFT (Wen et al., 2026), World-Model PI (Zhou et al., 2026a), SD-PG (Liu et al., 2026g)
    - **5.2.2 纯自蒸馏（Pure Self-Distillation）**：SDFT (Shenfeld et al., 2026), MTP-SD (Kirchenbauer et al., 2026), UniSD (Jin et al., 2026c), VPG (Chen et al., 2026e), Self-Sup OPD (Tan & Hong, 2026b), HINT-SD (Yeo et al., 2026), SD-Search (Ma et al., 2026), Vision-OPD (Yuan et al., 2026), It Takes Two (Park et al., 2026), TOD Proactivity (Zhang et al., 2026a), Search-E1 (Liang et al., 2026b), MAIGO (Zheng et al., 2026b), ROSD (Zhao et al., 2026d), Canonical-Ctx OPD (Lin et al., 2026b), COMAP (Liu et al., 2026i)
    - **5.2.3 外部反馈（External Feedback）**：SDPO (Hübötter et al., 2026), SD-ZERO (He et al., 2026), RLSD (Yang et al., 2026a), SRPO (Li et al., 2026c), RLTF (Song et al., 2026), CoPD (Gu et al., 2026), RLRT (Kim et al., 2026a), π-Play (Zhang et al., 2026l), PAINT (Tan & Hong, 2026a), CREDIT (Shen et al., 2026b), OGLS-SD (Yang et al., 2026f), VPD (Li et al., 2026i), RESD (Zhang et al., 2026n), SDAR (Lu et al., 2026), OPCT (Han et al., 2026a)
- **§6 训练效率与稳定性（Training Efficiency and Stabilization）**
  - **6.1 Token 与样本加权（Token and Sample Weighting）**：TIP (Xu et al., 2026c), SCOPE (Zheng et al., 2026a), SelecTKD (Huang et al., 2025), AdaSwitch (Peng et al., 2025), R-OPD (Fu et al., 2026b), SOD (Zhong et al., 2026a), MOPD (Yu et al., 2026b), EGRSD (Ke et al., 2026), Visual-Adv. OPD (Liu et al., 2026c), Filter-Then-Reweight (Li et al., 2026m), Less-is-More (Ziheng et al., 2026)
  - **6.2 课程与难度自适应（Curriculum and Difficulty Adaptation）**：PACED (Xu et al., 2026b), Stable-OPD (Luo et al., 2026a), CaOPD (Zhang et al., 2026d), TCOD (Wang et al., 2026d), Uni-OPD (Hou et al., 2026), DeltaPrompts (Jung et al., 2026), f-OPD (Chen et al., 2026c), TR-Behav. Blend (Plyusov et al., 2026), Adaptive Refresh (Guo et al., 2026), CEI (Chen et al., 2026a)
  - **6.3 计算优化（Compute Optimization）**：FOPD (Zhang et al., 2026b), Lightning-OPD (Wu et al., 2026a), SKD (Xu et al., 2025c), NPD (Rang et al., 2026), Prune-OPD (Yang et al., 2026g), EffOPD (Cai et al., 2026b), POPD/TOPD (Zhang et al., 2026k), SafeSteer (Li et al., 2026e)

**图 1：**On-Policy Distillation 方法的分类体系，沿三条设计轴组织：（1）目标函数设计（§4），（2）信号来源与教师架构（§5），（3）训练动态与效率（§6）。每个叶子节点列出该子方向的一个代表性方法子集，旁边的徽标给出该子章节的方法总数（涵盖所列方法以及对应子章节正文中讨论的额外论文）。方法被归入其**主要贡献**之下，因此一个使用白盒 logit 但在目标函数上有创新的方法（例如 GKD、DistiLLM、EOPD、G-OPD）出现在 §4 之下而非 §5.1.1 之下，而 §5.1.1 叶子节点只收集那些主要贡献就是白盒信号接口本身的方法。对次要维度有贡献的方法（例如 PAINT 与 PRISM 贡献了隐式课程）在相关章节正文中讨论，而不在树中重复列出。 
在工业规模上阐释了同样的三阶段分解，采用交叉熵作为目标、以从 R1 蒸馏得到的 800K 条精选样本（主要是推理轨迹）作为信号来源，并使用带难度感知课程（difficulty-aware curricula）的余弦学习率衰减来保证稳定性。

这三个阶段以既相互约束又相互强化的方式互动。有些组合彼此不兼容。精确 token 级形式的前向 KL（Forward KL）需要教师完整的输出分布，因此排除了只能获得生成文本的 API 受限场景。另一些组合则相互强化，因为 RL 增强型目标天然地与外部反馈来源（验证器、奖励模型）耦合，而固定散度（fixed divergence）则与允许精确梯度计算的白盒 logit 访问相匹配。因此，对目标函数的选择会约束可行的信号来源与动力学策略。

从时间顺序看，方法的分布反映了研究重点的转移。早期工作（2023–2024）集中于目标轴，讨论 forward KL、reverse KL 与 JSD（Agarwal et al., 2024; Gu et al., 2024）。到 2025 年中期，焦点转向信号架构，尤其是不依赖外部教师的自蒸馏（self-distillation）方法。最新的工作（2025 年末–2026 年）则针对训练动力学，特别是 on-policy 采样引入而静态数据集蒸馏所避免的不稳定性。诸如 DeepSeek-V4 与 Qwen3 这样的工业系统，正日益将三个轴结合起来，把自适应或 RL 增强型目标与多教师信号、以及基于课程的稳定化配对使用（Yang et al., 2026e; 2025a）。

### 3.2 方法对比表（Method Comparison Table）

**分类方法。** 每个方法都依据其核心贡献维度被恰好归入一个主类别。新颖性在于损失函数的论文（例如 G-OPD 的 KL 约束 RL 重构）归入 "Objective"。新的信号架构（例如 DSKD 的双空间投影）将方法归入 "Signal"。在训练稳定性或效率上的贡献（例如 TIP 的 token 加权）则映射到 "Dynamics"。对多个维度都有贡献的方法，按其最具辨识度的贡献来分类，从而减少当一个方法横跨多个维度时所产生的归类歧义（Xu et al., 2024）。

**分布层面的观察。** 截至 2026 年初，自蒸馏构成最大且增长最快的类别，其中大部分条目发表于 2025–2026 年。这一分布反映了从基础的 "哪个 KL 方向" 问题，逐步转向自我改进与自适应替代方案的转变，也与 LLM 研究中的一个更广泛趋势相符：无论是否存在外部教师，模型自身的 rollout 正日益成为主要数据来源。

**表 1（Table 1）：按主要贡献分类的关键 OPD 方法对比。**

| 方法（Method）             | 类别                  | 核心贡献                                  | 信号来源         | 损失粒度       |
| -------------------------- | --------------------- | ----------------------------------------- | ---------------- | -------------- |
| GKD (Agarwal et al., 2024) | Objective (Fixed)     | DAgger for LLMs, interpolation            | White-Box Logits | Token          |
| DistiLLM (Ko et al., 2024) | Objective (Fixed)     | Skewed KL mixtures                        | White-Box Logits | Token          |
| MiniLLM (Gu et al., 2024)  | Objective (Fixed)     | Sequence-level Reverse KL via RL          | White-Box Logits | Sequence       |
| EOPD (Jin et al., 2026b)   | Objective (Adaptive)  | Entropy-aware FKL+RKL blend               | White-Box Logits | Token          |
| G-OPD (Yang et al., 2026e) | Objective (RL)        | OPD equivalence to KL-RL                  | White-Box Logits | Sequence/Token |
| RLKD (Xu et al., 2025b)    | Objective (RL)        | KD augmented with reward model            | Reward Model     | Sequence       |
| DSKD (Zhang et al., 2025b) | Signal (White-Box)    | Cross-architecture alignment              | White-Box Logits | Token          |
| Lion (Jiang et al., 2023)  | Signal (Black-Box)    | Verbal feedback curriculum                | Black-Box API    | Sequence       |
| GAD (Ye et al., 2025)      | Signal (Black-Box)    | Adversarial distribution matching         | Black-Box API    | Sequence       |
| OPSD (Zhao et al., 2026b)  | Signal (Self)         | Privileged Info (PI) conditioning         | Self-Generated   | Token          |
| TIP (Xu et al., 2026c)     | Dynamics (Weight)     | Token importance via entropy & divergence | White-Box Logits | Token          |
| PACED (Xu et al., 2026b)   | Dynamics (Curriculum) | Beta-kernel curriculum pacing             | White-Box Logits | Problem        |

> 说明：原表 Category 列存在合并单元格，抽取后部分行该列为空。上表中空单元格已按上一行的值补全（DistiLLM、MiniLLM 补为 "Objective (Fixed)"；RLKD 补为 "Objective (RL)"；GAD 补为 "Signal (Black-Box)"）。

<!-- p. 11 -->

**表 2（Table 2）：不同应用领域中 on-policy 蒸馏方法的代表性实验配置。**

| 方法（Method）                                    | 教师                              | 学生                                                       | 训练信号                                    | 关键 benchmark                            |
| ------------------------------------------------- | --------------------------------- | ---------------------------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| **Mathematical Reasoning**                  |                                   |                                                            |                                             |                                           |
| G-OPD (Yang et al., 2026e)                        | Qwen3-30B-A3B-Instruct            | Qwen3-1.7B/4B                                              | Dense logit + reward extrap.                | AIME, HMMT, Code                          |
| OPSD (Zhao et al., 2026b)                         | Self (GT-conditioned)             | Qwen3-1.7B/4B/8B-Inst                                      | GT-PI self-distillation                     | AIME, HMMT                                |
| PACED (Xu et al., 2026b)                          | Qwen3-14B                         | Qwen3-8B                                                   | Beta-kernel curriculum                      | MATH-500, AIME                            |
| KDRL (Xu et al., 2025a)                           | Skywork-OR1-Math-7B               | R1-Distill-Qwen-1.5B/DeepScaleR                            | Joint KD + RL                               | MATH500, Minerva, AIME24                  |
| LUFFY (Yan et al., 2025)                          | DeepSeek-R1 (off-policy traces)   | Qwen2.5-Math-7B/1.5B                                       | Mixed-policy GRPO                           | AIME, AMC, MATH-500                       |
| AOPD (Jia et al., 2026)                           | Qwen3-32B/8B (teacher)            | Qwen3-8B/4B-Base                                           | Asymmetric PG + FKL                         | AIME'24/'25, HMMT'25                      |
| vOPD (Oh et al., 2026)                            | Qwen3-1.7B-Inst (Inst→Base)      | Qwen3-1.7B/4B-Base, OLMo-3-7B                              | CV baseline + single-sample RKL             | MATH-500, AIME'24/'25, Minerva            |
| **Instruction Following & General**         |                                   |                                                            |                                             |                                           |
| GKD (Agarwal et al., 2024)                        | T5-XL (3B)                        | T5-Small/Base                                              | On-policy FKL/RKL/JSD                       | XSum, WMT                                 |
| DistiLLM (Ko et al., 2024)                        | GPT-2 XL (1.5B)                   | GPT-2 (124M)                                               | Skewed KL                                   | ROUGE-L, Dolly                            |
| AlignDistil (Zhang et al., 2025a)                 | Self (contrastive DPO reward)     | Qwen2/2.5-1.5B-Inst                                        | Token-level RLHF distillation               | AlpacaEval, MT-Bench,=-- Arena-Hard       |
| **Industrial Scale**                        |                                   |                                                            |                                             |                                           |
| Qwen3 (Yang et al., 2025a)                        | Qwen3-32B/235B-A22B               | Qwen3-0.6B–14B                                            | On-policy logit distillation                | AIME, MATH, LiveCode                      |
| Gemma 2 (Gemma Team et al., 2024)                 | Gemma 2 27B                       | Gemma 2 9B / 2B                                            | KD in pre-training                          | MMLU, HellaSwag                           |
| MiMo-V2-Flash (Xiaomi LLM-Core Team et al., 2026) | Domain specialists                | MiMo-V2-Flash (309B/15B MoE)                               | Multi-teacher logit + reward                | AIME, MATH, GPQA                          |
| KAT-Coder-V2 (Li et al., 2026b)                   | 5 domain specialists              | Unified agentic coder                                      | Specialize-then-Unify OPD                   | SWE-bench (79.6%)                         |
| Nemotron-Cascade 2 (Yang et al., 2026h)           | Domain RL/SFT teachers            | 30B MoE (3B active)                                        | Cascade RL + domain OPD                     | IMO, IOI, ICPC                            |
| DeepSeek-V4 (DeepSeek-AI, 2026)                   | 10+ domain experts (1.6T)         | DeepSeek-V4-Pro (1.6T MoE)                                 | Full-vocab multi-teacher RKL                | LiveCodeBench, HMMT, IMOAnswerBench       |
| ORBIT (Liang et al., 2026a)                       | Multi-mode RL experts             | DeepSeek-Distill-Qwen-1.5B, Qwen3-4B, Openmath-Nemotron-7B | Multi-teacher mode fusion                   | AIME, GPQA, MMLU-Pro                      |
| **Multimodal & Domain-Specific**            |                                   |                                                            |                                             |                                           |
| SCoRe (Lyu et al., 2025)                          | Qwen2.5-72B-Inst                  | Qwen2.5-7B/3B, Llama-3.1-8B                                | Earliest-error correction                   | 12 Agent benchmarks                       |
| VOLD (Bousselham et al., 2025)                    | Text-only teacher                 | VLM student                                                | GRPO + token-level KL                       | MMMU-Pro, MathVista, LogicVista           |
| CORD (Hu et al., 2026)                            | Text-mode LALM                    | Audio-mode LALM                                            | Cross-modal RKL + GRPO                      | Audio QA/reasoning                        |
| SKD (Xu et al., 2025c)                            | Gemma-7B-IT, Qwen2-7B             | Gemma-2B, Qwen2-0.5B                                       | Block-verified on-policy KL                 | Translation, Dialogue                     |
| DP-OPD (Khadem et al., 2026)                      | GPT-2 Large 774M                  | DistilGPT-2 82M                                            | GKD + DP-SGD (student only)                 | Yelp, BigPatent                           |
| Veto (Jang et al., 2026)                          | Qwen2-7B-IT                       | Qwen2-0.5B-IT                                              | Geometric bridge KL                         | Reasoning, Generation                     |
| GUI-SD (Zhang et al., 2026i)                      | Self (visual PI)                  | Qwen3-VL-Inst-8B                                           | Entropy-guided KL                           | ScreenSpot-v2/Pro, OSWorld                |
| MAD-OPD (Wang et al., 2026c)                      | Multi-teacher debate              | Qwen3/3.5 (1.7B–14B)                                      | Confidence-weighted JSD/RKL                 | Agentic, Code                             |
| MSD (Qin et al., 2026)                            | Self (English CoT PI)             | Qwen2.5-7B, Llama-3-8B                                     | DPSW cross-lingual distill                  | Multilingual Safety                       |
| NPD (Rang et al., 2026)                           | Teacher (parallel prefill)        | openPangu-Embedded-1B                                      | Async CE+KD +∆-IFD                         | GSM8K, MATH (8.1×speedup)                |
| SimCT (Sun et al., 2026)                          | Qwen2.5-7B-Inst, Phi-4-mini       | Phi-4-mini, Gemma-2-2B-IT                                  | Multi-token continuation KL                 | Math, Code (cross-tokenizer)              |
| Prune-OPD (Yang et al., 2026g)                    | R1-Distill-Qwen-7B, Qwen3-4B      | R1-Distill-Qwen-1.5B, Qwen3-1.7B                           | Top-koverlap drift pruning                  | AMC, AIME, HMMT (37–68% time saved)      |
| SOD (Zhong et al., 2026a)                         | Qwen3-4B                          | Qwen3-0.6B/1.7B                                            | Step-level divergence reweighting           | AIME'25 (26.13% @0.6B), SciCode           |
| LiteGUI (Wu et al., 2026b)                        | Qwen3-VL-32B                      | 2B–3B scale agents                                        | Guided OPD + dual-level GRPO                | ScreenSpot-Pro, OS-World, Lite-Bench      |
| VISD (Lin et al., 2026a)                          | Self (video-aware judge)          | VideoLLM                                                   | Dir.-mag. decoupled advantage + RL          | Open-o3-Video benchmarks                  |
| Uni-OPD (Hou et al., 2026)                        | Multi-teacher (LLM+MLLM)          | Qwen3 (1.7B–4B), Qwen3-VL-4B                              | Dual-perspective OPD                        | Math, Code, MLLM (16 benchmarks)          |
| TGPO (Liu et al., 2026e)                          | Teacher (directional PG)          | NLP2CT/NEU students                                        | Dense directional teacher RKL               | NLP2CT, NEU translation                   |
| RWOPD (Zou et al., 2026)                          | CodeV-SVA teacher                 | Qwen2.5-Coder-7B                                           | Verifier-reward-weighted FKL                | NL2SVA                                    |
| ProteinOPD (Zhang et al., 2026m)                  | Pretrained PLMs                   | Preference-specific students                               | OPD on student trajectories                 | Protein property design (8×speedup)      |
| HyperEyes (Li et al., 2026d)                      | External teacher                  | HyperEyes-30B                                              | Dual-grained efficiency RL                  | IMEB                                      |
| DAgger-LLM (Li et al., 2026a)                     | GPT-4o                            | Qwen3-4B/8B                                                | Turn-level interpolation SL                 | SWE-bench Verified                        |
| MOPD (Yu et al., 2026b)                           | Peer-conditioned teacher          | Multi-rollout students                                     | Success/failure group signals               | Reasoning (CMU)                           |
| Local Collapse (Liu et al., 2026b)                | Qwen3 family (strong)             | Qwen3 family (weak)                                        | BIC-truncated OPD                           | 5 in-domain + OOD benchmarks              |
| **Self-Distillation**                       |                                   |                                                            |                                             |                                           |
| CRISP (Sang et al., 2026)                         | Self (concise prompt)             | Qwen3-8B/14B                                               | Per-token RKL                               | AIME, MATH-500                            |
| SD-ZERO (He et al., 2026)                         | Self (revision-conditioned)       | Qwen3-4B-Inst, OLMo-3-7B                                   | On-policy self-distill                      | AIME, MATH, Codeforces                    |
| π-Play (Zhang et al., 2026l)                     | Self (QCP-conditioned)            | Qwen3-4B/8B                                                | Multi-agent self-distill                    | NQ, TriviaQA, HotpotQA                    |
| UniSD (Jin et al., 2026c)                         | Self (EMA + multi-teacher agree.) | Qwen2.5-7B/0.5B–3B, Llama-3.1-8B, Gemma-3-4B              | Agreement + EMA + contrastive + feat. match | ScienceQA, MBPP , CoS-E, ToolAlpaca       |
| SDPO (H ¨ubotter et al., 2026)                   | Self (textual feedback)           | Qwen3-8B                                                   | Credit-assignment self-distill              | Code, Math                                |
| SRPO (Li et al., 2026c)                           | Self (SDPO+GRPO)                  | Qwen3-8B                                                   | Sample-routed dual-objective                | Chem, Phys, Bio, Mat, ToolUse             |
| PBSD (Yu et al., 2026c)                           | Self (ctx-augmented ref.)         | Qwen3-1.7B/4B/8B-Inst                                      | Pref.-based reward-reg. KL                  | AIME'24/'25, HMMT'25, Tool Use            |
| TT-OPD (Jeong, 2026)                              | Self (EMA + outcome hints)        | Qwen3.5-9B                                                 | Turn-level truncated KL                     | Healthcare AI GYM (18 benchmarks)         |
| Kim & Lee (2026)                                  | Self (GT-conditioned)             | Qwen3-8B, R1-Distill-7B, AceReason-7B                      | Correct-only OPSD                           | MATH-500, AIME'24/'25                     |
| EGRSD (Ke et al., 2026)                           | Self (entropy-gated)              | Qwen3-4B/8B (thinking mode)                                | Entropy-guided confidence gate              | Math reasoning (accuracy-length frontier) |
| RESD (Zhang et al., 2026n)                        | Self (reflection-enhanced)        | LLM agents                                                 | Failure reflection + playbook               | Continual agentic learning                |
| SDAR (Lu et al., 2026)                            | Self (gated RL+OPSD)              | Qwen2.5/Qwen3 families                                     | Sigmoid-gated OPSD auxiliary                | ALFWorld, WebShop, SearchQA               |
| GEAR (Li et al., 2026g)                           | Self (GT-conditioned)             | LLM agents                                                 | Adaptive segment advantage                  | Agent tasks                               |
| **Black-Box**                               |                                   |                                                            |                                             |                                           |
| GAD (Ye et al., 2025)                             | GPT-5-Chat (API only)             | Qwen2.5-14B-Inst                                           | Adversarial minimax                         | LMSYS-Chat, Dolly                         |
| Lion (Jiang et al., 2023)                         | ChatGPT (API only)                | LLaMA-7B/13B                                               | NL critique curriculum                      | BIG-Bench Hard, AGIEval                   |
| OVD (Xiong et al., 2026)                          | QwQ-32B / Env. Agent              | Qwen2.5-3B, Llama-3.2-3B                                   | Verbal scores (0–9)                        | Web QA, MATH                              |
| ROPD (Fang et al., 2026)                          | GPT-5.2, Qwen3-30B-A3B            | Qwen3-4B, Gemma3-4B                                        | Rubric-based semantic scoring               | AIME'24/'25, HMMT'25, GPQA-D, IFEval      |

> 说明：原表以分区标题行（Mathematical Reasoning 等）划分方法组，上表中以加粗行保留。

### 3.3 方法选择考量（Method Selection Considerations）

Table 2 表明，最成功的部署往往在三个阶段的每一处都做出深思熟虑的选择，而不是孤立地优化单一轴。这些轴之间的约束与相互作用似乎大幅缩减了可行配置的空间，我们在下面总结关键的选择因素。

**教师访问约束。** 可用的信号来源决定了哪些目标是可行的。白盒 logit 访问支持精确的 token 级散度计算，从而支持 DistiLLM（Ko et al., 2024）、G-OPD（Yang et al., 2026e）等方法，以及像 DSKD（Zhang et al., 2025b）这样的跨 tokenizer 方法。仅有 API 访问时，监督被限制为生成文本或标量分数，因而更适合为稀疏反馈设计的方法，如 OVD（Xiong et al., 2026）、GAD（Ye et al., 2025）与 LUFFY（Yan et al., 2025）。当没有外部教师可用时，自蒸馏方法要么通过 privileged information 运作（OPSD（Zhao et al., 2026b）、GATES（Stein et al., 2026）），要么通过验证器引导的自我改进运作（SD-ZERO（He et al., 2026））。

**任务特性。** 推理与数学任务受益于 mode-seeking 目标（reverse KL 或 RL 增强型变体），它们把概率质量集中到正确解上（Gu et al., 2024; Yang et al., 2026e; Xu et al., 2025a）。开放式生成任务则相反，需要 mode-covering 目标（forward KL 或自适应熵感知散度）以保持输出多样性。指令跟随任务处于中间地带，适度 on-policy 混合（λ≥ 0.5）配合 JSD 能提供很强的质量–算力折衷（Agarwal et al., 2024）。多步 agentic 任务通常需要序列级目标，并配以轨迹感知的 credit assignment（Lyu et al., 2025; Zhang et al., 2026o）。

**算力预算。** 在算力受限（<500 GPU-hours）时，前缀截断的 rollout（Zhang et al., 2026b）与离线缓存策略（Wu et al., 2026a）能大幅降低 on-policy 开销。中等规模下，配合自适应散度的完整 on-policy 训练（EOPD（Jin et al., 2026b）、AOPD（Jia et al., 2026））变得可行。大规模部署（>5000 GPU-hours）通常采用多阶段流水线，先进行 off-policy 预热，再过渡到 on-policy 精炼，并可选地加入基于 RL 的探索，正如 Qwen3 所报告的那样（Yang et al., 2025a）。

**稳定性要求。** 常见的失败模式促成了具体的稳定化选择。on-policy 训练中的长度膨胀（length inflation）可以通过参考散度约束与 rollout 混合来缓解（Luo et al., 2026a）。flawed prefix trap——即教师对学生所生成错误的反馈产生误导性梯度——则通过 token 级可靠性过滤来处理（Xu et al., 2026c; Zheng et al., 2026a）。教师与学生之间的思维模式不匹配，通常先通过一个 off-policy 冷启动阶段（Li et al., 2026j），再过渡到 on-policy 训练来解决。

<!-- p. 12 -->

**表 3（Table 3）：OPD 方法：固定散度目标。**

| 方法（Method）                               | 年份 | 类别      | 散度/目标                     | 信号                 | 粒度           | 关键创新                                      |
| -------------------------------------------- | ---- | --------- | ----------------------------- | -------------------- | -------------- | --------------------------------------------- |
| GKD (Agarwal et al., 2024)                   | 2023 | Objective | FKL / RKL / JSD               | White-box            | Token          | Unified OPD framework, DAgger for LLMs        |
| MiniLLM (Gu et al., 2024)                    | 2023 | Objective | Reverse KL                    | White-box            | Sequence       | REINFORCE with teacher as reward              |
| DistiLLM (Ko et al., 2024)                   | 2024 | Objective | Skew KL (SKL+SRKL)            | White-box            | Token          | α-smoothing for KL stability                 |
| DistiLLM-2 (Ko et al., 2025)                 | 2025 | Objective | Contrastive SKL/SRKL          | White-box            | Token          | Asymmetric loss per data source               |
| KETCHUP (Fan et al., 2025)                   | 2025 | Objective | Reverse KL                    | White-box            | Sequence       | K-step Bellman return estimation              |
| vOPD (Oh et al., 2026)                       | 2026 | Objective | Reverse KL + CV baseline      | White-box            | Token          | Control variate variance reduction            |
| AntiSD (Shen et al., 2026a)                  | 2026 | Objective | Divergence ascent             | Self (PI)            | Token          | Deliberation-token boosting via PMI sign flip |
| OPD+ (Zhao et al., 2026a)                    | 2026 | Objective | $f$-div gradient-faithful   | White-box            | Token          | Curvature-aware advantage$w_f$              |
| Bridging (Jiang & Ferraro, 2026)             | 2026 | Objective | Reverse KL + OT prefix signal | White-box            | Token/Span     | Near-future trajectory alignment              |
| Decomposed-OPD (Yoon et al., 2026)           | 2026 | Objective | Reverse KL + VGS              | White-box            | Token          | Visual-language gradient steering             |
| Surgical-PT (Lin & Han, 2026)                | 2026 | Objective | Proximal KL-constrained       | Oracle               | Token/Sequence | Retention via KL-constrained reward           |
| Distributional DAgger (Agrawal et al., 2026) | 2026 | Objective | Forward CE + REINFORCE        | Self (EMA, feedback) | Token          | DAgger with teacher-distribution feedback     |

## 4 目标函数与优化（Objective Functions and Optimization）

目标函数决定了学生在每个训练步优化的是什么数学量，它支配着梯度几何、mode coverage 以及由此产生的性能折衷。这一类方法已从对所有位置统一施加单一度量的固定散度，经由依据局部分布几何逐 token 调节度量的自适应散度，发展到纳入教师输出分布之外奖励信号的 RL 增强型目标。

### 4.1 固定散度目标（Fixed Divergence Objectives）

GKD（Agarwal et al., 2024）借鉴自回归语言模型的 on-policy 模仿学习范式（Ross et al., 2011），建立了经典的 OPD 框架。它从混合策略 $\pi_{\text{mix}} = \lambda p_\theta + (1-\lambda)p_{\text{data}}$ 中采样轨迹，并在每个 token 位置施加标准的 $f$-divergence（Forward KL、Reverse KL 或 JSD）。混合系数 $\lambda$ 在完全 off-policy（$\lambda=0$）与完全 on-policy（$\lambda=1$）之间插值，充当缓解 exposure bias 的单一控制旋钮。在指令跟随与摘要基准上，适度 on-policy 混合（$\lambda \geq 0.5$）优于纯 off-policy 训练，支持了分布对齐假设。

纯 KL 散度虽然优雅，但在 on-policy 探索过程中表现出数值不稳定性。当学生生成教师赋予接近零概率的 token 时，Forward KL 会产生无界梯度。Reverse KL 则完全忽略这些区域，在损失曲面上造成盲区。DistiLLM（Ko et al., 2024）通过 **Skewed KL**（SKL）来解决这一问题，它把标准 KL 中的 $q_\theta$ 替换为一个偏斜混合 $\tilde{p} = \alpha p_T + (1-\alpha)p_\theta$，其中 $\alpha \in (0,1]$。token 级 SKL 损失为

$$
L_{\mathrm{SKL}} = \mathbb{E}_{(x,y)\sim D_{\mathrm{mix}}}\left[\sum_{t=1}^{|y|} D_{\mathrm{KL}}\left(p_T(\cdot \mid y_{<t}) \,\|\, \alpha p_T(\cdot \mid y_{<t}) + (1-\alpha)p_\theta(\cdot \mid y_{<t})\right)\right]
$$

其中 $D_{\mathrm{mix}}$ 依据 DistiLLM 的自适应 off-policy 调度器，将真值序列与缓存的 student rollout 混合。偏斜保证了目标密度有 $(1-\alpha)p_\theta$ 作为下界，从而在无需 RL 策略梯度的情况下避免梯度爆炸。DistiLLM 进一步提出 **Skewed Reverse KL**（SRKL）用于 mode-seeking 应用，形成一对对称的稳定化散度。其后续工作 DistiLLM-2（Ko et al., 2025）认识到教师生成数据与学生生成数据承载着性质不同的学习信号，因而施加**非对称**目标：对教师生成数据使用 Forward SKL（此时 mode-covering 教学生探索教师的完整分布），对学生生成数据使用 Reverse SRKL（此时 mode-seeking 教学生锐化自己的最佳回答）。这种来源感知的非对称性，在指令跟随与摘要任务上相比对称的单散度基线带来了一致的改进。合适的散度取决于数据来源。

<!-- p. 13 -->

(a) 教师 $P_T$ —— 双峰参考 / Mode 1 / Mode 2 —— 两个同等有效的 mode（两个同样正确的答案）

(b) Forward KL —— Mode-covering（zero-avoiding）：$P(y)$ 轴上出现 "Hallucination"，$\min_{P_S} D_{\mathrm{KL}}(P_T \| P_S)$

(c) Reverse KL —— Mode-seeking（zero-forcing）：$P(y)$ 轴上出现 "Mode Dropped"，$\min_{P_S} D_{\mathrm{KL}}(P_S \| P_T)$

Figure 2: 用于将学生分布 $P_S$ 拟合到双峰教师 $P_T$ 的 Forward KL 与 Reverse KL 散度对比。**(a)** 教师在两个有效 mode 上放置相等的概率质量（例如两个正确答案或两种风格变体）。**(b)** Forward KL 最小化 $D_{\mathrm{KL}}(P_T \| P_S)$，属于 mode-covering（zero-avoiding）：学生把单个宽高斯铺在两个 mode 之上，并在 mode 之间的幻觉区（红色虚线边界）放置了可观的概率，这对应于两类有效答案混合后因而两者都不是的输出。**(c)** Reverse KL 最小化 $D_{\mathrm{KL}}(P_S \| P_T)$，属于 mode-seeking（zero-forcing）：学生尖锐地集中到单个峰（Mode 2），完全丢弃 Mode 1，但在所选 mode 内产生清晰、连贯的生成。这一折衷正是 §4 中自适应散度方法（EOPD、AOPD）的动机，它们基于逐 token 的教师熵或对数比统计量在两种机制之间插值。

DistiLLM-2 对不同数据源施加不同的散度，但其操作粒度较粗（按来源把整条序列分组）。固定散度这一族面临一个更根本的局限，它根植于**单条序列内部**自回归 token 分布的异质性。Forward KL（mode-covering）与 Reverse KL（mode-seeking）之间的选择是一个全局决定，被统一施加于所有 token 位置（Figure 2）。然而教师的分布结构在同一条序列内部变化相当大。在一个数学运算符 token 处，教师把概率质量集中在某一两个正确符号上，因此 mode-seeking 是合适的，因为学生必须确定地给出正确答案。在一个连词或填充词处，数十个 token 共享相近的概率，因此 mode-covering 是合适的，因为任何同义词都行。从而对两种位置施加单一散度，要么在一种机制下浪费梯度信号，要么在另一种机制下丢弃覆盖信息。这种损失几何与目标分布几何之间的逐位置错配，正是 §4.2 自适应方法的动机。

序列级目标提供了序列级的视角。MiniLLM（Gu et al., 2024）把 Reverse KL 提升到完整的序列级，此时学生 $p_\theta$ 出现在采样期望内部。核心挑战在于：当 $p_\theta$ 本身由 $\theta$ 参数化时，如何计算关于 $p_\theta$ 的期望的梯度。从 $D_{\mathrm{KL}}(p_\theta \| p_T) = \mathbb{E}_{y\sim p_\theta}[\log p_\theta(y|x) - \log p_T(y|x)]$ 出发并应用策略梯度定理，MiniLLM 推导出（Gu et al., 2024 的 Eq. 2）：

$$
\nabla_\theta L_{\mathrm{MiniLLM}} = -\mathbb{E}_{y\sim p_\theta}\left[\sum_{t=1}^{|y|}(R_t - 1)\nabla_\theta \log p_\theta(y_t \mid y_{<t})\right], \quad R_t = \sum_{t'=t}^{|y|} \log \frac{p_T(y_{t'} \mid y_{<t'})}{p_\theta(y_{t'} \mid y_{<t'})}
$$

其中 $-1$ 来自 reverse-KL 目标的 $-\log p_\theta(y|x)$ 熵项。因此，最小化序列级 Reverse KL 在数学上等价于策略梯度 
 RL，其 per-step reward 为 $r_t = \log[p_T(y_t|y_{<t})/p_\theta(y_t|y_{<t})]$，外加一个由熵引出的常数。MiniLLM 进一步把 $R_t$ 分解为一个单步质量项和一个未来回报项（Eq. 3 of Gu et al., 2024），并在词表上以闭式形式（而非蒙特卡洛采样）计算 $\mathbb{E}_{y_t \sim q_\theta}[r_t]$。这一分解通过把梯度噪声的主导来源（单步质量）与累积的未来回报隔离开来，降低了方差，并支持 token 级的信用分配。per-step return $R_t$ 把轨迹对教师的偏离归因到具体的 token 决策上，而不是把责任均匀摊到整个序列，从而把优化置于 RL 框架之内，以教师的 log-probability 作为稠密奖励。这种与策略优化的联系与 MiniLLM 在推理任务上的经验优势相吻合——在这些任务中，序列级连贯性比 token 级准确率更重要——代价是更高的训练算力，因为在组合式输出空间中的 REINFORCE 估计需要更多迭代以及谨慎的 baseline 相减才能收敛。KETCHUP（Fan et al., 2025）通过把完整的累积回报 $G_t$ 替换为源自最优性方程的 $K$-step Bellman 回报估计来应对这一方差问题，在保留序列级目标的同时以 $K$ 的幂次速率降低梯度方差（Theorem 1 of Fan et al., 2025）。

**token 级与序列级的权衡（The token-level vs. sequence-level tradeoff）**。token 级目标与序列级目标之间的选择构成一个核心的偏差—方差权衡，其解决方式预示了 §4.3 中的 RL 增强方法。token 级方法（GKD、DistiLLM）在每一个位置都产生 $|V|$ 维的梯度信号，方差低，但当教师的 token 级分布校准不佳时（第 6.1 节的 flawed prefix trap），梯度可能有偏。序列级方法（MiniLLM、KETCHUP）通过策略梯度定理得到无偏梯度，但由于在指数级庞大的序列空间上做单样本 REINFORCE 估计，会遭受高方差。实践上的解决取决于任务。对于 token 级教师质量较高的指令跟随，GKD 式方法往往占优。对于序列级连贯性更重要的推理任务，MiniLLM 式方法常被优先采用，尽管其方差较高，而 KETCHUP 的 $K$-step Bellman 回报在这一谱系上提供了一个居中的偏差—方差点。认识到 MiniLLM 实际上已经在执行策略梯度 RL（以教师的 log-probability 作为奖励），意味着序列级蒸馏与 RL 增强蒸馏并非彼此不同的范式，而是同一连续统的两个端点；该连续统由奖励来源参数化，一端是纯教师信号，另一端是纯任务奖励，而混合目标（G-OPD、RLKD）则在两者之间插值。

**全词表 KL 与采样 token KL（Full-vocabulary vs. sampled-token KL）**。与 token 级/序列级的选择正交，一个实现层面的决策决定了在每个位置使用多少教师分布。许多基于 RL 的 OPD 实现通过只在被采样的 token $y_t$ 上计算 $\log p_\theta(y_t|y_{<t})-\log p_T(y_t|y_{<t})$ 来近似每个位置的 KL，并在策略梯度框架中把这一个标量当作 per-token advantage（Lu & Thinking Machines Lab, 2025）。虽然计算上很轻（每个位置只需一次教师 log-probability），但这种单样本估计会带来较高的梯度方差，并丢弃未采样 token 上的分布信息。DeepSeek-V4（DeepSeek-AI, 2026）报告称，full-vocabulary logit distillation——即在每个位置对所有 $|V|$ 个 token 精确计算 $D_{\mathrm{KL}}(p_\theta(\cdot|y_{<t}) \parallel p_T(\cdot|y_{<t}))$——会产生更稳定的梯度和更忠实于教师的知识迁移，代价是要传输完整的教师 logit 向量。他们的工程解法（缓存教师隐状态，并通过预测头即时重建 logit）使这一点即使在 10+ 万亿参数教师、词表超过 100K token 的情况下也可行，从而把 full-vocabulary KL 定位为在基础设施允许时工业级 OPD 的一个有吸引力的选择。Lightning-OPD（Wu et al., 2026a）进一步报告称，在 teacher consistency 条件下（即 SFT 数据生成与蒸馏使用同一教师），通过在 SFT rollout 上预计算教师 log-probability，可以基本免除完整的教师前向传播，相对标准 OPD 录得 4.0× 加速。

**用于深思的 divergence 上升（Divergence ascent for deliberation）**。AntiSD（Shen et al., 2026a）把自蒸馏梯度整体反转。一项 pointwise mutual information 分析表明，以 privileged context 为条件会抬高教师在解已暗示的 token（结构性连接词、可验证的断言）上的置信度，同时压低在驱动多步搜索的 deliberation token（Wait, Let, Maybe）上的置信度。通过上升（ascending）学生—教师 divergence，而不是沿其下降： 
AntiSD 会翻转每 token 的符号，从而得到天然有界的 advantage，在单次更新步骤中提升 deliberation token。一个由熵触发的门控会在教师熵崩塌后关闭 anti-distillation 项，防止教师分布已经尖峰化时出现退化行为。在从 4B 到 30B 参数的五个模型上，AntiSD 以 2–10× 更少的训练步数达到 GRPO 级别的准确率；在同等训练预算下，在数学推理上比 GRPO 高出多达 11.5 个点，这表明当 privileged 信号系统性偏置时，标准的“向教师下降”假设可以放松。

控制变量方差缩减（Control variate variance reduction）。vOPD (Oh et al., 2026) 通过把 policy-gradient RL 中的控制变量基线（control variate baseline）引入 OPD 估计器，为这一方差-成本权衡提供了第三种解决方案。把单样本 OPD 梯度（Oh et al., 2026 的公式 3）表述为带每 token reward $r_t=\log p_T(y_t|c_t)-\log p_\theta(y_t|c_t)$ 的 REINFORCE 更新后，vOPD 在乘以 $\nabla_\theta\log p_\theta$ 之前从 $r_t$ 中减去一个基线 $b_t(c_t)$。该 reward 的自然价值函数具有闭式解，即学生与教师之间每 token 负 reverse KL：$V^{p_\theta}(c_t)=-D_{KL}(p_\theta(\cdot|c_t)\|p_T(\cdot|c_t))$，可直接由已经产生 OPD 目标的那次前向传播算出，无需辅助 critic 或额外 rollout。由于基线只依赖上下文 $c_t$ 而不依赖被采样的 token $y_t$，减去它保持了梯度的无偏性，同时降低了方差。该基线的 top-k 近似（把 KL 求和限制在学生的 k 个最可能 token 上）进一步降低计算量，且由于该近似仍与所采样的动作无关，无论 k 取何值无偏性依然成立。在 Qwen3-1.7B 与 4B 的六个推理基准上，vOPD 相对 vanilla 单样本 OPD 平均获得 +3% 绝对增益（在 MATH500 上最多 +6.2%），在达到全词表 OPD 准确率的同时把 wall-clock 时间最多减少 57.7%，并总体优于 top-k OPD——后者增加计算却引入偏差。这一结果把 vOPD 置于推理模型 on-policy 蒸馏的偏差-方差-计算 Pareto 前沿上的有利位置。

固定散度训练内的 advantage 与梯度重设计（Advantage and gradient redesigns within fixed-divergence training）。上述方差控制结果保留了传统的 advantage 与梯度定义，只对估计器进行调优。另一条研究线索则追问：一旦引入 on-policy rollout，advantage 本身是否被正确设定。OPD+ (Zhao et al., 2026a) 从 f -divergence 梯度分析重新推导每 token 权重，并观察到传统的 stop-gradient reward $r_t=\log[p_T(y_t|y_{<t})/p_\theta(y_t|y_{<t})]$ 对应 $f(u)=-\log u$，但忽略了二阶曲率；而忠实于梯度的权重 $w_f(u)=-f(u)+uf'(u)$ 使 OPD 更新与真实的 f -divergence 梯度对齐，并减少一种当学生远离教师时会增大的系统性偏差。OPD+ 修正的是每 token 的 advantage，而 Bridging Reasoning Trajectories (Jiang & Ferraro, 2026) 处理的是短时程轨迹信号：它通过短窗口最优传输距离（optimal-transport distance）检测轨迹级散度，并在学生的近未来续写偏离教师轨迹时注入基于 OT 的教师信号。低 OT 但高损失的 token 被降权以抑制误报，两阶段调度（off-policy 预热后接 on-policy 蒸馏）把 OT 修正限制在最有信息量的 on-policy 阶段。同样的逐位置失配问题在视觉-语言学生上再次出现：在那里 reverse-KL 梯度可能被语言先验支配，从而实际上绕开视觉 grounding。Decomposed-OPD (Yoon et al., 2026) 以 Visual Gradient Steering (VGS) 回应，这是一种将每 token 梯度分解为语言先验分量与视觉 grounding 分量、按梯度范数对二者做归一化、并用从 cross-attention 统计中读出的 Visual Dependency Score (VDS) 放大视觉分量的分解方法。一个语言保持正则项防止再平衡后的梯度损害纯文本能力。合起来看，这三种方法说明：同一个固定散度骨架可以通过作用于梯度信号的三个不同层级而得到改进——每 token 权重（OPD+）、短时程轨迹（Bridging）、模态分解（Decomposed-OPD）——而无需改变散度类型本身。

近端与 DAgger 式重构（Proximal and DAgger-style reformulations）。另有两种方法保留固定散度模板，但重新定义了被约束的对象。Surgical Post-Training (Lin & Han, 2026) 把推理注入视为在相对更新前策略的 KL 约束下的近端 on-policy 蒸馏，其分析与经验主张是：正是这种 KL 约束下

<!-- p. 16 -->

**表 4（Table 4）：OPD 方法：自适应散度目标（OPD methods: Adaptive divergence objectives）。**

| 方法（Method）                              | 年份 | 类别      | 散度/目标                     | 信号               | 粒度  | 关键创新                                           |
| ------------------------------------------- | ---- | --------- | ----------------------------- | ------------------ | ----- | -------------------------------------------------- |
| EOPD (Jin et al., 2026b)                    | 2026 | Objective | Continuous FKL+RKL blend      | White-box          | Token | Entropy-aware on-policy distillation               |
| AOPD (Jia et al., 2026)                     | 2026 | Objective | Asymmetric PG/FKL switch      | White-box          | Token | Advantage-sign token-level switching               |
| CREDIT (Shen et al., 2026b)                 | 2026 | Self (EF) | Contrastive pMI decomposition | Self (contrastive) | Token | Input-specific credit for OPSD reward              |
| OGLS-SD (Yang et al., 2026f)                | 2026 | Self (EF) | Outcome-guided logit steering | Self (outcome)     | Token | Outcome-based teacher calibration                  |
| Position-Weighted OPSD (Liu et al., 2026d)  | 2026 | Objective | Position-aware reliability    | Self (PI)          | Token | Per-position reliability score                     |
| Direction-Adaptive SD (Zhang et al., 2026c) | 2026 | Objective | Entropy-routed polarity       | Self (PI)          | Token | High-entropy teacher reversal                      |
| Token-Teachability (Wang et al., 2026j)     | 2026 | Objective | Teachability-gated KL         | White-box          | Token | Fixed-context learnability diagnostic              |
| Lookahead Group Reward (Liu et al., 2026f)  | 2026 | Objective | RKL + look-ahead group reward | White-box          | Token | One-step-ahead teacher confidence                  |
| RAFT (Li et al., 2026l)                     | 2026 | Objective | Top-K temperature distill     | Self               | Token | Data refinement + EMA loss balance                 |
| Trust-Region OPD (Xing et al., 2026)        | 2026 | Objective | Adaptive trust-region KL      | White-box          | Token | Speculative-decoding-ratio gating                  |
| GNDPO (Hao et al., 2026)                    | 2026 | Objective | Global batch-normalized KL    | White-box          | Token | Batch-relative KL normalization for MLLM stability |

的 reward 形式，支撑了推理专门化过程中对既有能力的保持。Distributional DAgger (Agrawal et al., 2026) 回到最初启发 GKD 的 DAgger (Ross et al., 2011) 蓝图，并把标量反馈推广为教师分布。学生在自己的策略下 rollout，反馈（例如 verifier 或丰富的偏好信号）对一个 EMA self-teacher 进行条件化，由此得到的 forward cross-entropy 与一个 REINFORCE 未来信用项配对，使 token 级更新在 PPO 式信任域内考虑下游后果。top-K 教师截断使每 token 成本保持有界。这两种重构把固定散度 OPD 框定为约束式策略改进的一个实例：KL 约束 reward 充当保持机制（Surgical Post-Training），而具备未来意识的 DAgger 更新充当信用分配机制（Distributional DAgger），二者都为 4.2 节的逐位置自适应散度做了铺垫。

### 4.2 自适应散度目标（Adaptive Divergence Objectives）

自适应方法通过基于局部分布几何来选择或插值散度，来处理这种逐位置失配（表 4）。当教师确定（低熵）时，mode-seeking 的 Reverse KL 保持精确性；当教师不确定（高熵）时，mode-covering 的 Forward KL 保持多样性。

EOPD (Jin et al., 2026b) 使用教师的每 token 熵 $H(p_T(\cdot|y_{<t}))$ 作为门控信号。基础目标在所有位置都是 Reverse KL，而在教师把概率分配给多个有效续写的高熵 token 处，以加性方式激活 Forward KL：

$$
\mathcal{L}_{EA}=\mathbb{E}_{y\sim p_\theta}\Big[\sum_{t=1}^{|y|}D_{KL}(p_\theta\|p_T)+\mathbb{I}[H_t>\tau_H]\cdot D_{KL}(p_T\|p_\theta)\Big]
$$

EOPD 在六个竞赛数学推理基准上优于固定散度基线，证实了合适的散度依赖位置。仅以教师不确定性（一个边缘性质）为条件，使 EOPD 获得不依赖学生质量的稳定路由信号，因而适合在学生的分布尚不可靠的训练早期使用。

自蒸馏 reward 校准（Self-distillation reward calibration）。两种近期方法通过注入对比式或基于结果的修正，来处理自蒸馏 reward 信号特有的病理。CREDIT (Shen et al., 2026b) 通过一个 batch 对比基线，把标准 OPSD 的 token reward 分解为输入特有（input-specific）与输入通用（input-generic）两个分量，从而抑制那些抬高 pointwise mutual information、却不贡献任务相关推理的通用捷径（5.2.3 节）。OGLS-SD (Yang et al., 2026f) 使用可验证的结果 reward 来校准自我反思得到的教师分布，对比成功与失败的 on-policy 轨迹，处理反思引入的偏差——该偏差会使教师 logit 偏离最优目标（5.2.3 节）。两种方法都在自蒸馏范式内运作，详见 5.2.3 节。

AOPD (Jia et al., 2026) 拓展了逐 token 自适应原则，诊断标准 OPD 梯度为何在特定 token 位置失效。在非正 advantage 区域（At≤0），标准 advantage 加权 policy gradient 表现出三种在弱初始化下会加剧的病理。来自无界负 advantage 的重尾方差会破坏更新的稳定性；在零 advantage 位置梯度消失，浪费教师信号；而“探索黑洞”会困住学习，因为压制一个错误 token 只会

<!-- p. 17 -->

按学生自身先验的比例重新分配概率质量，无法提升低概率的正确替代项。AOPD 通过在 token 级别切换目标来处理这一点。对于正 advantage 的 token（At>0），保留标准 policy gradient 的利用。对于非正 token，该框架在教师的 top-K 支撑上用局部化的 forward-KL 指导替代，形成一种非对称的利用-模仿机制。阈值参数 τ（默认设为 0，对应教师与学生之间的概率差）决定切换点：τ=−1 可恢复标准 OPD（干预很少触发，因为 PT−PS≤−1 几乎从不成立），τ=1 可恢复 GKD（处处干预，因为 PT−PS≤1 通常成立）。在竞赛级数学（AIME 2024/2025、HMMT）上，AOPD 相对标准 on-policy 基线在强初始化下平均增益 +4.09、在弱初始化下 +8.34；弱初始化下差距更大，说明当学生起点远离教师时，探索黑洞是主导的失效模式。AOPD 依据每个位置梯度结构的健康程度，调整目标类别本身（policy gradient 与散度最小化），这与 EOPD 中调整散度类型的基于熵的路由形成互补。

Token 可靠性与可教性（Token reliability and teachability）。另一条平行研究线索通过用对每个位置可信任程度或可学习程度的直接测量，替换隐式信号（教师熵、advantage 符号）来改进逐 token 路由。Position-Weighted OPSD (Liu et al., 2026d) 观察到仅凭教师熵不足以预测给定位置的监督质量，并且可靠性会随深入学生 rollout 而下降，因为较后位置条件于越来越 off-policy 的 prefix。该方法用在学生生成序列上的教师行为读出的位置感知可靠性分数来调制每 token 损失，在均匀加权会压平梯度的情形下恢复了准确率。Token-Teachability OPD (Wang et al., 2026j) 从学生一侧处理同一问题。一个固定上下文诊断测量学生在每个位置是否能合理地吸收教师偏好的 token，从而把真正可学习的 token 与仅仅是高分歧的 token 区分开。仅限可教 token 的选择性 OPD 在推理基准上优于基于熵与基于分歧的选择。合在一起，两种方法定位了教师的信号在哪里携带有用信息（Position-Weighted）以及学生在哪里处于能够利用它的位置（Token-Teachability），从而收紧了 EOPD 与 AOPD 在更粗的逐位置粒度上所应用的路由逻辑。Direction-Adaptive Self-Distillation (Zhang et al., 2026c) 把这种路由推进得更远：在高熵 token 处翻转教师的极性，其精神类似 AntiSD 的散度上升（4.1 节），但像 EOPD 一样以熵作为门控，从而在教师不确定性意味着探索价值而非指导价值的 deliberation 点上把学生推离教师。

前瞻与信任域自适应（Look-ahead and trust-region adaptation）。逐位置路由还可以通过把教师信号跨位置扩展、或按稳定性对更新做门控来改进。Lookahead Group Reward (Liu et al., 2026f) 识别出监督保真度衰减（Supervision Fidelity Decay），即在学生 rollout 上，每 token 的教师置信度会随轨迹推进、学生偏离 off-policy 而下降。该方法用一步前瞻 group reward 增强每 token 信号：在一次前瞻步后于 tree-attention 掩码下计算教师置信度，对所得分数做 group 归一化，并以局部熵门控其激活。Trust-Region OPD (Xing et al., 2026) 直接自适应调整每步信任域。speculative decoding 的接受率被重新用作学生-教师策略散度的廉价代理；一个离群值估计器裁剪 top-k FKL 的越界值；在学生 rollout 信息量最低的训练早期，用余弦退火混入一个 off-policy 教师 prefix。K1 reverse-KL 估计器为该信任域提供锚定，在轨迹层面重新获得了 EOPD 与 AOPD 的门控原则。RAFT (Li et al., 2026l) 把同一自适应原则耦合到数据侧：交替进行自我条件化重写（通过余弦相似度语义过滤与多专家融合来精炼含噪的领域 prompt）与答案条件化的 on-policy 蒸馏（针对原始 instruction-tuned 模型使用 top-K 温度蒸馏）。一个基于 EMA 的自适应损失平衡器防止蒸馏分量与 SFT 分量相互干扰，支持了更一般的观察：在领域专门化场景中，on-policy 蒸馏受益于同时调整数据分布与每 token 权重。

<!-- p. 18 -->

**表 5（Table 5）：OPD 方法：RL 增强目标（OPD methods: RL-augmented objectives）。**

| 方法（Method）                      | 年份 | 类别      | 散度/目标                     | 信号             | 粒度       | 关键创新                                                            |
| ----------------------------------- | ---- | --------- | ----------------------------- | ---------------- | ---------- | ------------------------------------------------------------------- |
| G-OPD (Yang et al., 2026e)          | 2026 | Objective | KL-Constrained RL             | White-box        | Token/Seq  | OPD≡dense KL-RL, reward extrapolation                              |
| RLKD (Xu et al., 2025b)             | 2025 | Objective | KD + Reward Model             | Reward Model     | Sequence   | Generative Structure Reward Model (GSRM)                            |
| KDRL (Xu et al., 2025a)             | 2025 | Objective | Joint KD + RL                 | White-box        | Token/Seq  | On-policy KL regularizer during RL                                  |
| RLAD (Zhang et al., 2026o)          | 2026 | Objective | Trust Region Ratio            | White-box        | Token/Seq  | PPO-style selective teacher following                               |
| AlignDistil (Zhang et al., 2025a)   | 2025 | Objective | DPO + Reverse DPO             | White-box        | Token      | Token-level RLHF via contrastive DPO reward                         |
| SuperCorrect (Yang et al., 2025b)   | 2024 | Objective | Template + Cross-DPO          | White-box        | Sequence   | Hierarchical thought scaffolding                                    |
| SCoRe (Lyu et al., 2025)            | 2025 | Objective | Earliest-error correction     | White-box        | Sequence   | Root-cause error targeting                                          |
| CMDP-KD (Zimmer et al., 2025)       | 2025 | Objective | Constrained KL                | White-box        | Sequence   | Unaugmented CMDP with hard KL budget                                |
| X-KD (Cai & Yuan, 2026)             | 2026 | Objective | Inverse RL + KL               | White-box        | Token/Seq  | AVRIL-based reward recovery distillation                            |
| TSD-KD (Kim & Baek, 2026)           | 2026 | Objective | Indirect + Direct KD          | White-box        | Sequence   | Indirect re-ranking + selective direct KL                           |
| REOPOLD (Ko et al., 2026)           | 2026 | Objective | Log-ratio token reward        | White-box        | Token      | Mixture reward clipping + entropy-dynamic sampling                  |
| CoDistill-GRPO (Kwon et al., 2026)  | 2026 | Objective | GRPO + KD reward              | White-box        | Sequence   | Bidirectional co-distillation, 18% speedup                          |
| dGRPO (Ramos et al., 2026)          | 2026 | Objective | GRPO + dense OPD              | White-box        | Token/Seq  | Long-context reasoning with LongBlocks data                         |
| Sparse-to-Dense (Xu et al., 2026d)  | 2026 | Objective | Staged RL→OPD→RL            | White-box        | Token/Seq  | Reward-density allocation principle                                 |
| TGPO (Liu et al., 2026e)            | 2026 | Objective | Directional teacher PG        | White-box        | Token      | Dense directional teacher guidance on RKL                           |
| RWOPD (Zou et al., 2026)            | 2026 | Objective | Verifier-weighted FKL         | White-box        | Sequence   | Verifier-reward-weighted OPD for NL2SVA                             |
| AMR-SD (Wei et al., 2026)           | 2026 | Objective | Asymmetric meta-reflective    | Self             | Token      | Meta-reflection self-teacher in RLVR                                |
| OPPO (Li et al., 2026k)             | 2026 | Objective | Bayesian value recursion      | Oracle (PI)      | Token      | Likelihood-ratio per-token credit                                   |
| StepOPSD (Zhang et al., 2026j)      | 2026 | Objective | Step-aware preference + KL    | Self (hindsight) | Step/Token | Dual-granularity for multi-turn agents                              |
| Self-Eval OPD (Zhang et al., 2026h) | 2026 | Objective | Cyclic GRPO + masked judge KD | External judge   | Token      | Self-evaluation channel calibration                                 |
| RLCSD (Pan et al., 2026)            | 2026 | Objective | Contrastive RKL in GRPO       | White-box        | Token      | Correct-hint teacher pull, wrong-hint push in two-path distillation |

用于稳定 on-policy 蒸馏的批归一化（Batch normalization for stable on-policy distillation）。GNDPO (Hao et al., 2026) 处理自适应 KL 损失应用于多模态 LLM（MLLM）时出现的不稳定。当 token 数分布变化很大时（例如不同分辨率的视觉输入），标准的逐样本 KL 归一化会在 mini-batch 之间产生大的梯度方差。GNDPO 相对一个全局参考分布在整个 batch 上归一化 KL 目标，为用 on-policy 蒸馏训练的 MLLM 产生更平滑的梯度估计与更稳定的训练动力学。

该领域已明显从“哪个散度？”转向“哪个散度、在哪里？”。来自 EOPD 与 AOPD 的证据在各个评测设置中指向一个一致发现：没有任何单一固定散度能普遍占优，而且即便最简单的自适应路由（一个硬熵阈值）也常常胜过调优最好的固定替代方案。然而，自适应散度无论其逐 token 路由多么精巧，都与固定散度共享一个结构性局限。它们优化的是到教师分布的距离。在任何 f -divergence 下做到完美优化，得到的学生也只能匹配教师，而很少能超越教师。当教师本身会犯错、或缺少一个得到良好引导的学生本可发现的推理路径时，散度最小化更像是一个上限而非下限。

### 4.3 RL 增强目标（RL-Augmented Objectives）

上一节末尾指出的能力上限，引出了一类不同的目标。RL 增强方法注入一个外部性能信号，可以把学生引向教师本身不太可能产生的解法。

把 OPD 视为 KL 约束 RL（OPD as KL-constrained RL）。G-OPD (Yang et al., 2026e) 形式化了标准 OPD 与密集 KL 约束强化学习之间的等价性。它把蒸馏表述为最大化一个 token 级 reward，该 reward 定义为教师与参考策略之间的 log 概率比，并以相对参考策略的 KL 散度作为惩罚：

$$
\max_\theta\ \mathbb{E}_{y\sim p_\theta}\Big[\sum_{t=1}^{|y|}\alpha\log\frac{p_T(y_t|y_{<t})}{p_{\text{ref}}(y_t|y_{<t})}-D_{KL}\big(p_\theta(\cdot|y_{<t})\|p_{\text{ref}}(\cdot|y_{<t})\big)\Big]
$$

当 α=1 时，这退化为标准 Reverse KL 蒸馏。当 α>1 时，该目标迫使学生外推到教师概率质量之外，允许一种被记录为 reward 外推（Reward Extrapolation）的现象：学生会发现此前未见、结构上合理的推理路径，这些路径位于教师生成概率 pT(y|x) 低但结果 reward 仍然高的区域。在多个领域特定 RL 专家共享同一基座模型的多教师设定中，reward 外推（ExOPD）产生一个统一的学生，超越其所有同尺寸的领域教师——这一结果把 OPD 定位为一种潜在的放大机制，而非有损压缩步骤。

这一等价性对整个小节还有更广泛的后果。它把蒸馏与 RL 两个社群连接起来：此处讨论的目标（GKD 的 token 级 KL、MiniLLM 的序列级 Reverse KL、G-OPD 的 reward 增强表述）都可以

<!-- p. 19 -->

被视为具有不同 reward 定义与约束强度的 KL 约束策略优化的特例。因此，KL 约束 RL 的进展（更好的信任域、自适应惩罚系数、方差缩减）很可能可以迁移到 OPD，反之亦然。

梯度分解与 KD+RL 的互补性（Gradient decomposition and the KD+RL complementarity）。Li et al. (2025) 的统一 KD+RL 框架把混合梯度分解为两个分量：用于 token 级模仿的密集 KD 梯度，以及用于 reward 最大化的 Monte Carlo RL 梯度。KD 分量抑制 policy gradient 估计固有的高方差，而 RL 分量防止学生坍缩到对下游任务次优的教师 mode 上。REOPOLD (Ko et al., 2026) 通过把 on-policy 蒸馏解释为策略优化（其中师生 log-likelihood 比充当 token reward）来实例化这一洞见。它通过三种机制稳定训练：基于混合的 reward 裁剪以防止对极端教师信号的过度信任，基于熵的动态采样在高不确定 token 处选择性加权教师反馈，以及一个统一的“从探索到精炼”调度。经验上，REOPOLD 达到比近期 RL 方法高 6.7–12× 的样本效率，并使 7B 学生能在视觉推理上匹敌 32B 教师，同时获得约 3.3× 的推理加速。

结构感知 reward 作为密集 KL 的替代（Structure-aware reward as an alternative to dense KL）。G-OPD 与 REOPOLD 保留 token 级 KL 作为主要蒸馏信号（只在如何增强或稳定它上不同），而 RLKD (Xu et al., 2025b) 完全脱离这一模板，用 Generative Structure Reward Model (GSRM) 取代 token 级 KL。GSRM 把教师与学生的推理路径都解析为 meta-reasoning 步骤与 solving 步骤的序列，然后为两个序列之间的结构对齐打分。这一结构 reward 与任务级结果 reward（例如数学准确率）结合，并通过 GRPO 优化：

$$
\mathcal{R}_{\text{total}}(x,y)=\alpha\mathcal{R}_{\text{GSRM}}(y,y_{\text{teach}})+(1-\alpha)\mathcal{R}_{\text{outcome}}(x,y)
$$

其主要思想是：推理蒸馏不必在 token 级进行。通过在步骤结构层面监督，RLKD 避免了基于 logit 的方法的词表耦合假设，同时仍然迁移教师的多分支推理策略。经验上，RLKD 在纯 RL 体制下只用 0.1% 的数据训练，就超过了标准的 SFT-RL 流水线，说明当教师的价值在于其推理组织结构而非 token 级分布时，结构 reward 可以替代密集 KL。从密集 token 级 KL（G-OPD）经方差缩减的 token reward（REOPOLD）到步骤级结构匹配（RLKD）的这条轨迹，勾勒出“教师知识”含义的抽象程度不断提升：从分布保真度走向行为组织。

双向协同蒸馏（Bidirectional co-distillation）。CoDistill-GRPO (Kwon et al., 2026) 处理 GRPO 在小模型上的常见失败（在困难任务上稀疏 reward 带来的学习信号可忽略），其协同蒸馏算法同时训练一个大模型与一个小模型，并使用精心设计的 GRPO 目标。小模型使用由大模型分布导出的 on-policy KD reward 作为密集监督，而大模型在小模型的 rollout 上训练并进行重要性重加权，从而降低 rollout 生成的计算开销。在 Qwen2.5-Math-1.5B 上，CoDistill-GRPO 在 Minerva 上达到 32%，比基座模型提升 11.6 个点、比标准 GRPO 高 6.0 个点。大模型（Qwen2.5-Math-7B）尽管在小模型 rollout 上训练，也几乎与标准 GRPO 持平，提供约 18% 的加速。这种双向设计避免了“必须存在一个预训练 oracle”的常见假设，而是同时产生教师信号与学生改进。

Reverse KL 上的方向性教师引导（Directional teacher guidance on reverse KL）。TGPO (Liu et al., 2026e) 脱离加性 KD+RL 模板，把教师引导嵌入 policy gradient 本身之内。TGPO 不把教师分布当作单独的正则项，而是用教师的每 token likelihood ratio 作为方向性信号，把 reverse-KL 梯度引向教师认为有前景的区域，将更新集中到教师与学生分歧最大的 token 上。这种密集的方向性耦合避免了联合 KD+RL 目标所需的平衡超参数（RLKD 中的 α、KDRL 中的 KL 惩罚），转而让教师的置信度在每个位置调制 RL 梯度。在数学推理基准（如 MATH-500）上，TGPO 同时优于单独的 OPD 与单独的 GRPO。

<!-- p. 20 -->

与单独的 OPD 和单独的 GRPO 相比，教师指导与策略优化之间更紧密的耦合可能比相加式组合更有效。

**面向长上下文推理的稠密指导（Dense guidance for long-context reasoning）。** dGRPO (Ramos et al., 2026) 针对与 CoDistill-GRPO 相同的稀疏奖励失效模式，但采用了不同的架构。dGRPO 不是协同训练两个模型，而是通过一个相加式的 OPD 项，从更强的教师那里为单个学生的 GRPO 训练补充稠密的 token 级指导。该方法专门针对长上下文推理场景（最长 128K token），在这一场景中稀疏奖励尤其不稳定，因为正确的推理路径在更长的序列中变得越来越稀少，有效学习信号随上下文长度而退化。配套的 LongBlocks 数据集提供了跨越多跳推理、上下文接地（contextual grounding）和长文本生成的合成长上下文任务。KD+RL 的组合目标比任一单独成分都提供更稳定的优化，解决了这一场景中纯 GRPO 样本效率低下（大多数长上下文 rollout 完全失败）而纯 OPD 缺乏直接奖励优化来发现此前未见的长程推理模式的问题。CoDistill-GRPO 在两个模型之间双向使用 KD 信号，而 dGRPO 则从一个固定教师单向施加该信号，用架构上的简洁性换取相互自举带来的收益。

**稀疏到稠密的奖励分配（Sparse-to-dense reward allocation）。** Xu et al. (2026d) 提出了一条奖励密度原则，把 GRPO 与 OPD 重新表述为两种不同的奖励密度机制，而不是相互竞争的配方。稀疏的序列级奖励在最强模型上最有成效，此时探索能够发现新解（教师侧 RL），而稠密的 token 级教师奖励对于把已学到的行为压缩进更小的模型最为有效（学生侧 OPD）。其操作层面的教训是一个三阶段流水线：先在教师上做 GRPO 以发现能力，然后做 forward-KL 预热，接着用 OPD 作为稠密桥梁，最后可选地在桥梁之后做学生侧的稀疏 RL。在固定的 Qwen3-1.7B 部署规模上，一个经 RL 改进的 8B 教师通过该稠密桥梁蒸馏后，优于直接的学生侧 GRPO（MATH 上 79.3% 对 75.9%，AIME 2024 上 25.2 对 19.8），而从同一个教师**在做 RL 之前**迁移则表现更差。该桥梁还使后续的学生侧 GRPO 在原先无效的情况下变得有效，把 MATH 从 75.4% 提升到 78.5%。教师质量的排序结论在 Llama-3.1-8B-Instruct 上、以 Llama-3.3-70B-Instruct 为教师时得到了复现。

**联合优化与顺序优化（Joint vs. sequential optimization）。** 鉴于 KD 与 RL 提供互补的梯度信号（稠密模仿与稀疏奖励最大化），一个长期存在的工程挑战是**如何**在单一训练流水线中把二者结合起来而不产生相互干扰。顺序式方法（先蒸馏后 RL，或先 RL 后蒸馏）往往遭受灾难性干扰，因为 RL 阶段会通过策略漂移破坏蒸馏得到的知识，而 KD 阶段会抑制探索过程中此前未发现的解。KDRL (Xu et al., 2025a) 通过在 **RL 训练期间**加入学生与教师之间的 on-policy KL 正则项来实现联合优化，在仍允许奖励驱动探索的同时缓解策略漂移。CMDP-KD (Zimmer et al., 2025) 扩展了这一正则化方案，通过约束式 MDP 形式化把软 KL 惩罚替换为**硬** KL 预算。其状态增强的 Saute 形式化可以为 LLM 简化，即完全去掉状态增强，因为完整历史已经决定了剩余预算，从而在不要求在推理时访问教师的情况下给出约束满足保证。RLAD (Zhang et al., 2026o) 进一步细化，指出无条件的 KL 正则化本身也是次优的。其 Trust Region Ratio Distillation（TRRD）用锚定到教师-旧策略混合分布的 PPO 式似然比目标替换了标准 KL 惩罚。这种选择性策略只在教师信号能改善策略更新时才跟随教师，从而防止盲目的 KL 最小化把学生在教师本身也无法解决的问题上拖向次优轨迹。从 KDRL 的均匀正则化到 RLAD 的选择性信任域的演进，与散度目标中从固定到自适应的转变（第 4.1→4.2 节）如出一辙，呼应了本综述中反复出现的一个模式：把学习信号条件化于局部几何往往比均匀施加它效果更好。

**作为隐式 RL 的基于偏好的目标（Preference-based objectives as implicit RL）。** 当无法获得 logit 访问，或序列级质量判断比 token 级分布更自然时，有若干方法把 KD+RL 问题映射到 Direct Preference Optimization（DPO）上。AlignDistil (Zhang et al., 2025a) 从对比式 DPO 与反向

<!-- p. 21 -->

DPO 模型导出 token 级奖励，把序列级 RLHF 转换为逐 token 的策略蒸馏，在 token 粒度上同时施加标准偏好目标与反向偏好目标。OVD (Xiong et al., 2026) 向教师查询关于学生生成回复的成对偏好，从而完全避免了连续概率的抽取。这些基于偏好的目标与 token 级 KL 方法有深层的联系。在 DPO 的闭式解下，最优策略满足 $\pi^*(y|x)\propto\pi_{\text{ref}}(y|x)\exp(r(y,x)/\beta)$，而把教师的 log 概率代入作为隐式奖励，就得到 GKD 用 λ-mixing 所瞄准的几何混合分布。其实际含义是：当构造偏好对比计算完整教师 logit 更便宜时，DPO 式方法往往最有用。PBSD (Yu et al., 2026c) 在自蒸馏场景中把这一联系推得更远，它针对上下文增强的教师而非静态参考策略，导出奖励正则化后的最优解 $\pi^*\propto\pi_{\text{teach}}\exp(r/\beta)$；只要奖励非退化，这个奖励倾斜的目标在同一目标函数下就优于教师本身（第 5.2.1 节）。这一谱系暴露出一个统一的梯度。token 级 KL（G-OPD、REOPOLD）、步级结构匹配（RLKD）与序列级偏好排序（AlignDistil、OVD）都是在不同抽象层级上实现 KL 约束的策略优化，而适当的层级由信号可得性以及教师专长所体现的粒度决定。

**验证器奖励加权的蒸馏（Verifier-reward-weighted distillation）。** 当某个领域专用验证器能够为学生 rollout 打正确性分数时，这些分数为蒸馏目标提供了自然的逐样本加权。RWOPD (Zou et al., 2026) 在 NL-to-SystemVerilog 生成中实例化了这一思想，其中一个开放的性质等价性验证器会针对形式化规约给学生生成的每个硬件描述打分。验证器奖励对 forward-KL 损失进行加权，使更接近功能正确性的 rollout 得到更强的教师监督，而明显错误的 rollout 被衰减而不是被完全丢弃（从而保留来自部分成功的梯度信号）。该方法在 NL2SVA 基准上创造了新的最佳结果，这一结果把验证器加权的 OPD 定位为非常适合形式化验证领域——在这些领域中二元或分级的正确性可以廉价计算，而标准 RL 下的奖励地形是稀疏的。

**细粒度信用分配（Fine-grained credit assignment）。** 上述所有方法，无论在 token、步还是序列层级上操作，都共享一个局限：它们在学生的整条轨迹上均匀施加监督。当学生在 10 步证明的第 3 步犯了一个错误时，整体式模仿会重新训练全部 10 步，从而稀释掉真正失败点上的纠正梯度。SuperCorrect (Yang et al., 2025b) 提出带跨模型 DPO 的分层思维模板来做错误定位，识别出哪个推理步发生了偏离，并把反馈集中到那里。SCoRe (Lyu et al., 2025) 采取更激进的做法，**只**纠正学生轨迹中**最早的错误**，把所有监督集中到纠正影响力最大的那一个点上。这利用了推理错误的级联特性：纠正根因可以消除下游后果。与 RLKD 的结构化方法相比，这体现了互补的设计选择。RLKD 奖励整体步骤组织与教师一致的轨迹，而 SCoRe 则精准地纠正第一个偏离点。二者是互补的，而把全局结构奖励与局部错误纠正结合起来仍未被探索。

**Token 级贝叶斯与元反思式信用（Token-level Bayesian and meta-reflective credit）。** 当教师信号本身不稳定时，信用分配问题会变得更加紧迫，例如在 RLVR 下的自蒸馏中，或在奖励只在长动作序列末尾才到达的多轮智能体 rollout 中。AMR-SD (Wei et al., 2026) 通过一个非对称的元反思式自教师，把逐 token 信号与策略更新分离开来，从而稳定 RLVR 自蒸馏。学生进行 rollout，一个元反思阶段把验证器结果、同伴 rollout 与参考反馈聚合为逐 token 监督，而非对称损失则在外在正则化之外防止学生-教师循环发散。OPPO (Li et al., 2026k) 在贝叶斯框架内形式化了同样的逐 token 信用问题，通过 oracle 条件下的似然比递归地估计逐 token 价值。这种似然比递归产生的逐 token 优势与轨迹级结果保持一致，从而在不需要单独 critic 网络的情况下给出细粒度奖励归因。AMR-SD 与 OPPO 在单轮推理轨迹上操作，而 StepOPSD (Zhang et al., 2026j) 把步感知的偏好蒸馏扩展到多轮智能体 RL，把智能体轨迹切分为 
因果交互步，并把步级教师打分与 token 级蒸馏联合施加。这种双粒度对齐把稀疏的环境奖励（在步边界到达）与稠密的逐 token 监督（填补每一步内部空隙）匹配起来，解决了限制 OPD 在长时程智能体任务上表现的信用分配错配问题。

**作为评判蒸馏的自评估（Self-evaluation as judge distillation）。** 上面大多数方法把评判者或奖励模型视为外部信号源。Zhang et al. (2026h) 转而追问：基础模型本身是否已经携带一个可以通过 on-policy distillation 引出的潜在评判者。其训练循环在以下两个阶段之间交替：一是与校准耦合的 GRPO 阶段（策略在按属性-分数分箱做分层轮转采样的样本上更新，并用一个非线性校准指数塑造奖励），二是掩码评判蒸馏阶段，它把外部评判者的实际分数蒸馏回 rollout 轨迹的自我评估 token 上，而答案 token 保持不动。这种掩码把 OPD 更新限制在自我评估通道内，从而把评判者视为专门服务于该通道的 on-policy 教师。该结果把评判者校准刻画为一个 OPD 问题，可以与策略更新在同一个 RL 循环内运行，这与 off-policy 的奖励模型微调形成对比。

**RL 内部的对比蒸馏（Contrastive distillation within RL）。** 一条相关的工作路线把对比式推理施加到教师信号本身，而不是评判者校准上。RLCSD (Pan et al., 2026) 通过一个双路径损失把对比式 reverse KL 集成到 GRPO 训练循环中。当教师以正确提示为条件时，学生被拉向教师的分布（模仿目标）。当教师以错误提示为条件时，学生被推离（对比目标）。这种正确/错误分解产生了一个有方向性的梯度，在强化正确推理策略的同时主动抑制误导性策略，且不需要单独的判别器或偏好对构造。该方法自然地用一个逐 token 蒸馏信号扩展了 GRPO 的组相对优势，而该信号的符号由提示的正确性而不是答案的结果决定。

**从信号匹配到环境重建（From signal matching to environment reconstruction）。** 一个更进一步的扩展追问：学生能否恢复教师**底层的优化地形**。X-KD (Cai & Yuan, 2026) 采用 AVRIL 框架，联合建模教师的隐式奖励函数并执行策略蒸馏，鼓励学生在一个重建的教师训练环境中优化，而不是模仿教师的输出。TSD-KD (Kim & Baek, 2026) 把间接蒸馏（学生生成候选项，教师通过偏好反馈对其重排序）与直接蒸馏（KL 匹配有选择地施加于师生差距最大的高熵 token 上）结合起来。这些方法超越了匹配教师的**行为**，走向恢复其**激励结构**。

在各类目标之间，可以看到一条大致的演进脉络。固定散度给出稳定、已被充分理解的优化，但往往受限于教师能力的天花板。自适应散度可以通过匹配损失地形的局部几何来提高效率。RL 增强目标提供了绕过教师天花板的路径，代价是额外的方差与不稳定性。恰当的选择取决于具体应用。对于教师质量很高的指令跟随场景，固定或自适应散度往往就足够了。对于期望学生超越教师的推理任务，RL 增强就变得有吸引力。对于教师的训练过程可以获取的任务，环境重建提供了更完整形式的知识迁移。

## 5 信号来源与教师架构（Signal Source and Teacher Architecture）

在确定目标函数（第 4 节）之后，下一个设计轴关注**教师信号来自哪里**。我们沿两个层级组织信号来源。第一层区分外部教师蒸馏（由一个单独的更强模型提供监督）与自蒸馏（模型利用内部不对称性自己生成训练信号）。在外部教师蒸馏内部，第二层区分可访问教师完整输出分布的白盒方法与只能利用生成文本或标量分数的黑盒方法。这两个层级追踪教师的两种不同属性：它的身份（外部还是模型自身），以及当它是外部模型时，其信号可被观测到的访问层级。二者并非完全独立，因为访问层级 
反过来又约束了哪些目标可用，但把它们放在不同层级可以避免把一项部署约束与一项方法论选择混为一谈。每一种组合都会对适用的目标与可达性能施加不同的约束。

从历史上看，信号源在这两个层级上都发生了演化。首先移动的是访问层级这一维度。早期 OPD 方法（GKD、DistiLLM）假定可以对模型内部进行完全白盒访问，这一要求把蒸馏限制在同一组织内部的部署；而专有 API-only 模型（GPT-4、Claude）的兴起，则使仅从文本输出中抽取知识的黑盒方法成为必需。教师身份这一维度随后发生移动：自蒸馏方法完全移除了外部教师，使得无需访问任何更强的模型也能持续改进。沿访问层级维度的移动——从完全访问到受限访问——以及沿身份维度的移动——从外部教师到无教师——都代表着以信号密度为代价换取自主性的提升，而这两个权衡在很大程度上是相互独立的。

复杂推理是 OPD 相对 off-policy 替代方案往往表现出显著收益的场景之一。其根本原因在于推理是路径依赖的（path-dependent）(Agarwal et al., 2024)。一个以"假设反证……"开头的证明，其后续 token 与以"我们用归纳法来推进……"开头的证明完全不同。如果学生在训练期间从未生成过归纳法这一思路，那么在推理时当该路径最优时，它也不太可能重新找到它。在教师推理轨迹的固定数据集上进行 off-policy 训练，只能覆盖有限的解路径集合，而学生需要在由部分解、死胡同与恢复策略构成的自己的地形中穿行。这一观察为推理迁移的同策略（on-policy）表述提供了动机。学生生成自己的中间推理步骤 $r_S \sim P_\theta(\cdot|x)$，教师评价这些学生生成的推理链，而不是指定自己的推理链：

$$
\mathcal{L}_{\text{CoT-OPD}}(\theta) = \mathbb{E}_{x\sim\mathcal{D},\, r_S\sim P_\theta(\cdot|x)}\left[\sum_{t=1}^{|r_S|} D_{\mathrm{KL}}\left(p_T(\cdot|x, r_S^{<t}) \,\|\, p_\theta(\cdot|x, r_S^{<t})\right)\right]
$$

Forward KL 方向鼓励 mode-covering，使得学生在自己生成的 prefix 上、在教师分配了概率的地方也分配概率，从而在自身的实际生成模式上获得纠正性反馈，而不是去记忆教师偏好的轨迹。这种自洽的学习动态——学生在专家监督下练习自己的推理风格——使 on-policy CoT 蒸馏成为把多步推理迁移到更小模型时的常见选择。"理据迁移优于仅答案监督"这一原则最早由 Hsieh et al. (2023) 在 off-policy 设定下确立，随后有工作表明，在 on-policy 下生成理据 (Agarwal et al., 2024) 可以通过降低 exposure bias 带来进一步的收益，这一效应在 DeepSeek-R1 (DeepSeek-AI et al., 2025) 及其后继模型的大规模实践中也被观察到。

### 5.1 外部教师蒸馏（External Teacher Distillation）

外部教师蒸馏假定可以访问一个独立模型，其在目标任务分布上的能力超过学生，而教师的作用是在学生的 on-policy rollout 上提供纠正性信号。下面两部分按该信号的密度对方法进行划分。完全 logit 访问（第 5.1.1 节）能给出最丰富的逐 token 监督，但要求与教师共同托管；而受 API 约束的设定（第 5.1.2 节）则以信号密度换取部署灵活性。

#### 5.1.1 白盒 Logit 监督（White-Box Logit Supervision）

白盒访问能给出尽可能稠密的信号，即在每个解码步上覆盖 $|V|$ 个 token 的完整概率分布（表 6）。这暴露了教师的隐式决策边界，包括错误 token 之间的相对概率（Hinton 的"暗知识"），这些概率携带了二值标签无法捕捉的相似性结构信息。

**本节范围。**白盒 logit 访问是本综述其他位置所编目的绝大多数方法的隐含假设，包括大多数固定散度（第 4.1 节）、

<!-- p. 24 -->

**表 6（Table 6）：OPD 方法：白盒 logit 监督方向的创新。**

| 方法                                 | 年份 | 类别   | 散度/目标                                         | 信号                  | 粒度       | 关键创新                                                                                   |
| ------------------------------------ | ---- | ------ | ------------------------------------------------- | --------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| DSKD (Zhang et al., 2025b)           | 2025 | Signal | 双空间 KL（Dual-Space KL）                        | 白盒                  | token      | 跨词表投影（Cross-vocabulary projection）                                                  |
| PromptKD (Kim et al., 2024)          | 2024 | Signal | 基于 prompt 的诱导（Prompt-based elicitation）    | 白盒                  | token      | 输入侧蒸馏（Input-side distillation）                                                      |
| MAD-OPD (Wang et al., 2026c)         | 2026 | Signal | 多智能体辩论集成（Multi-agent debate ensemble）   | 白盒                  | token      | 置信度加权的多教师（Confidence-weighted multi-teacher）                                    |
| Veto (Jang et al., 2026)             | 2026 | Signal | 几何桥 KL（Geometric bridge KL）                  | 白盒                  | token      | 自适应目标重构（Adaptive target reformulation）                                            |
| SimCT (Sun et al., 2026)             | 2026 | Signal | 多 token 续写 KL（Multi-token continuation KL）   | 白盒                  | token      | 跨 tokenizer 的监督恢复（Cross-tokenizer supervision recovery）                            |
| CSD (Kim et al., 2026c)              | 2025 | Signal | 连续 score matching（Continuous score matching）  | 白盒                  | token      | 绕过离散词表不匹配（Bypasses discrete vocabulary mismatch）                                |
| MPD (Yang et al., 2026c)             | 2026 | Signal | 混合策略 KL（Mixed-policy KL）                    | 白盒                  | token      | 教师将学生轨迹改写得更简洁（Teacher rewrites student traces concisely）                    |
| BRTS (Zhang et al., 2026e)           | 2026 | Signal | Best-of-N 教师选择（Best-of-N teacher selection） | 白盒                  | 序列       | 面向教师可靠性的优先级 rollout 选择（Priority rollout selection for teacher reliability）  |
| Pair-In, Pair-Out (Tan et al., 2026) | 2026 | Signal | MTP confidence-head KL（MTP confidence-head KL）  | 白盒（full backbone） | token      | OPD 监督的 latent-MTP 头（OPD-supervised latent-MTP head）                                 |
| DuDi (Payoungkhamdee et al., 2026)   | 2026 | Signal | 双信号跨语言 KD（Dual-signal cross-lingual KD）   | 白盒                  | token/序列 | 通过跨语言 verbalizer 的 SPIN + token KD（SPIN + token KD via cross-lingual verbalizer）   |
| OPRD (Yang et al., 2026d)            | 2026 | Signal | 表征空间 OPD（Representation-space OPD）          | 白盒                  | 层         | 隐状态对齐扩展了 logit 空间蒸馏（Hidden-state alignment extends logit-space distillation） |
| BTB (Niu et al., 2026)               | 2026 | Signal | 跨 tokenizer OPD（Cross-tokenizer OPD）           | 白盒                  | token      | token 映射桥接异构模型家族（Token-mapping bridges heterogeneous model families）           |

自适应散度（第 4.2 节）与 RL 增强（第 4.3 节）目标，第 5.2.1 节中以完整教师 logits 为条件的特权信息自蒸馏方法，以及第 6 节中的大多数加权与课程方法。为避免在多个类别下重复列出这些方法，第 5.1.1 节（以及图 1 的叶节点）只覆盖其中较小的一个子集，即**主要贡献**在于白盒信号接口本身创新的方法，具体包括跨 tokenizer 对齐、全词表压缩、教师 rollout 选择以及 confidence-head 构造。对完整白盒方法集合感兴趣的读者应查阅表 3–9 中的 Signal 列（"White-box"）。

**同家族蒸馏。**当教师与学生共享同一 tokenizer 时（例如在同一模型家族内），白盒 logit 迁移是直接的，方法可以直接计算逐 token 的 KL 散度，而无需对齐开销。

**标准 logit 蒸馏。**第 4 节综述的大多数方法都假定同家族白盒访问，并在每个学生生成的 token 位置上计算 $p_T(\cdot|x, y_{<t})$。计算开销主要由教师的前向传播决定，可以通过在整个 rollout 上缓存教师 logits 再计算学生损失来摊薄。当教师与学生共享同一架构时，这种缓存把每训练步的墙钟开销相对于仅学生的 RL 降至大约 1.5–2× (Agarwal et al., 2024)，这一开销推动了第 6.3 节讨论的系统级优化。

**混合策略推理压缩。**MPD (Yang et al., 2026c) 通过一种混合策略机制，把简洁推理行为从较大的教师迁移到较小的学生。给定一条学生采样的轨迹，教师把它改写为更简洁的推理轨迹，学生则在压缩后的轨迹上通过基于 KL 的对齐进行训练。这既保留了学生策略的探索（不同于 off-policy 蒸馏），又注入了教师引导的压缩（不同于标准 on-policy 蒸馏，后者是在学生的冗长轨迹上把学生对齐到教师分布）。在 Qwen3-1.7B 上，MPD 在多个推理 benchmark 上提升性能的同时将 token 用量最多减少 27.1%，利用的是这样一个观察：更大的推理模型对同样的问题会产生更简洁的轨迹。

**Best-of-N 教师 rollout 选择。**BRTS (Zhang et al., 2026e) 针对标准 OPD 中的一个方差问题。在含噪的学生生成上下文下计算出的教师监督，往往依赖于单次随机的教师 rollout，而这一次 rollout 可能是错误的，或与学生自身的推理行为匹配得很差。BRTS 采样一个小规模的教师轨迹池，并用一条优先级规则（先正确性、后学生对齐）选出最优者，同时对无条件教师采样失败的更难 prompt 采用一个以 ground truth 为条件的恢复步骤。被选中的轨迹在 OPD 循环内提供可靠的教师上下文监督。在 AIME 2024、AIME 2025 与 AMC 2023 上，BRTS 相对标准 OPD 取得提升，且在更难的数据集上增益更为显著。当问题接近学生能力边界时，教师信号质量更为重要。

**Token 级自适应监督。**在白盒设定内，若干方法细化了**哪些** token 接受监督，可以按"驱动选择决策的因素"这一轴来组织。SelecTKD (Huang et al., 2025) 基于学生—教师一致性进行路由，采用一种 propose-and-verify 机制：学生提出 token，教师进行验证，只有一致性成立的位置获得完整损失，而被拒绝的 token 则被 mask 或降权。AdaSwitch (Peng et al., 2025) 基于学生 prefix 的可靠性进行路由，维护一个持续更新的误差估计，并在探索（学生自由生成）与引导（教师提供稠密监督）之间切换，

<!-- p. 25 -->

使引导只在轨迹漂移得足够远、值得教师介入时才被激活。这些选择准则——一致性与 prefix 质量——是互补而非相互竞争的，各自捕捉了关于"监督在何处最有信息量"的不同信号。它们也自然地与基于熵的散度路由（EOPD）组合：后者选择在每个位置**应用哪种损失函数**，而 token 级选择决定**有多少梯度信号**传播。这一组合产生了一个正交的二维设计空间，当前方法才刚刚开始联合探索。这些 token 选择机制与第 6 节讨论的样本级和 token 级加权方案互为补充。

**带 OPD 监督 confidence head 的 latent 压缩。**白盒监督的另一项用途是把 OPD 与多 token 预测（MTP）结合起来，以提升推理时效率。Pair-In, Pair-Out (Tan et al., 2026) 把学生的输入序列压缩为 latent code，并用一个并行的 MTP 头替换自回归输出头，在 MTP 输出之上训练一个 confidence head，用以决定哪些位置可以投机性地提交。同一模型的完整 backbone 版本充当 OPD 教师，对压缩后的学生 rollout 打分并提供逐 token 信号来校准这一 confidence head，从而使 latent 压缩受教师自身的逐 token 偏好约束，而不是受一个无监督瓶颈约束。这一构造把 OPD 定位为架构压缩的校准机制，与上述逐 token logit 迁移方法互为补充。

**跨家族蒸馏。**上述方法都假定教师与学生共享同一 tokenizer，这在两者属于同一模型家族时成立（例如 Qwen3-30B→Qwen3-4B）。当教师与学生使用不同的 tokenizer 或词表规模时会出现一个关键挑战，这在跨家族迁移中很常见（例如 Llama 教师 → Qwen 学生）。在词表不匹配的情况下，直接的 KL 散度是无定义的。DSKD (Zhang et al., 2025b) 通过**双空间**知识蒸馏来处理这一问题，使用在教师与学生表征空间之间做映射的投影器：

$$
\mathcal{L}_{\text{DSKD}} = D_{\mathrm{KL}}(P_{T\to S} \| P_S) + D_{\mathrm{KL}}(P_T \| P_{S\to T})
$$

其中 $P_{T\to S}$ 表示教师隐状态被投影到学生空间、并通过学生的预测头解码后的结果。一种精确的 token 对齐算法通过识别不同 tokenizer 之间相同的 token 来处理词表不匹配。语义嵌入相似的词表 token 应当获得相近的概率质量，即便分词的方案不同，从而允许在没有任何词表重叠的不同架构之间进行蒸馏（例如 Llama 教师到 Qwen 学生）。CSD (Kim et al., 2026c) 提供了另一种解决方案：通过一个连续的 score 函数进行蒸馏，绕过离散词表不匹配的问题。

SimCT (Sun et al., 2026) 采取了另一种完全在 OPD 损失形式内部运作的途径。SimCT 不投影表征、也不搬运概率质量，而是通过比较两种 tokenizer 都能实现的短多 token 续写单元来扩大监督空间。这些单元代表了最细粒度的、可联合分词的监督接口，从而在词表不一致的位置上恢复了精确的共享 token 匹配所默默丢弃的教师信号。在数学推理与代码生成上，跨三组异构的教师—学生对（Qwen2.5-7B→Phi-4-mini、Qwen2.5-7B→Gemma-2-2B、Phi-4-mini→Gemma-2-2B），SimCT 同时优于共享词表的 OPD 和更粗粒度的跨 tokenizer 基线，且在词表分歧最大的位置上增益显著。这一结果表明，在共享 token 匹配下所记录的跨 tokenizer 监督损失中有相当一部分并非源自固有的不兼容，而是源自一种不必要的、过于严格的匹配粒度。

这些方法共同推动 OPD 变得更加与架构无关，使组织能够把某一模型家族中的教师（例如基于 Llama 的）蒸馏到另一家族的学生（例如基于 Qwen 的）中，而不再受历史上"必须共享词表"这一限制性要求的约束，并支持过去难以处理的异构教师—学生对。

<!-- p. 26 -->

**表 7（Table 7）：OPD 方法：黑盒与受 API 约束的蒸馏。**

| 方法                              | 年份 | 类别   | 散度/目标                                                      | 信号            | 粒度        | 关键创新                                                  |
| --------------------------------- | ---- | ------ | -------------------------------------------------------------- | --------------- | ----------- | --------------------------------------------------------- |
| GAD (Ye et al., 2025)             | 2025 | Signal | 对抗式匹配（Adversarial matching）                             | 黑盒            | 序列        | 判别器作为代理奖励（Discriminator as proxy reward）       |
| Lion (Jiang et al., 2023)         | 2023 | Signal | 言语反馈（Verbal feedback）                                    | 黑盒            | 序列        | 自然语言批评课程（NL critique curriculum）                |
| OVD (Xiong et al., 2026)          | 2026 | Signal | 言语分数 0–9（Verbal scores (0–9)）                          | 黑盒            | 序列        | 基于分数的轨迹匹配（Score-based trajectory matching）     |
| LUFFY (Yan et al., 2025)          | 2025 | Signal | 混合策略 GRPO（Mixed-policy GRPO）                             | Off-policy + RL | 序列        | 重要性加权的 off-policy（Importance-weighted off-policy） |
| PRISM (Wang et al., 2026g)        | 2026 | Signal | 黑盒预对齐（Black-box pre-alignment）                          | 黑盒（API）     | 序列        | RLVR 之前的多模态 OPD（Multimodal OPD before RLVR）       |
| ROPD (Fang et al., 2026)          | 2026 | Signal | 基于 rubric 的评分（Rubric-based scoring）                     | 黑盒（API）     | 序列        | 结构化语义 rubrics，10× sample efficiency                |
| DASD (Yan et al., 2026)           | 2026 | Signal | 序列级分布匹配（Sequence-level distribution match）            | 黑盒            | 序列        | 多轨迹分布对齐（Multi-trace distribution alignment）      |
| DDT (Zhang et al., 2026f)         | 2026 | Signal | 理论 RL+KD 框架（Theoretical RL+KD framework）                 | 黑盒            | 序列        | on-policy 泛化理论（On-policy generalization theory）     |
| ORPO-Distill (Singh et al., 2025) | 2025 | Signal | ORPO 对比偏好（ORPO contrastive preference）                   | 黑盒（API）     | 序列        | 混合策略负采样（Mixed-policy negative sampling）          |
| OmniOPD (Zhou et al., 2026b)      | 2026 | Signal | Chunk MC 验证 + KL anchor（Chunk MC verification + KL anchor） | 黑盒（API）     | chunk/token | 无 logit 的贝叶斯平滑（Logit-free Bayesian smoothing）    |

跨家族迁移中的另一个挑战是教师**过度自信**。当教师给学生的可达空间之外的 token 分配极端概率时，由此产生的优化目标既精确又无用。有两种解决方案从相反方向应对这一问题。Veto (Jang et al., 2026)（自适应目标重构）在事后（post-hoc）细化教师的输出，在 logit 空间构造一座几何桥，其中的参数 β 既是自适应梯度否决器（Adaptive Gradient Veto，抑制低置信度 token 上的病态梯度），又是果断性旋钮（Decisiveness Knob，在性能与多样性之间平衡），从而根据学生当前的能力动态重构教师的目标分布。PromptKD (Kim et al., 2024) 采取相反的做法，在事前（pre-hoc）调整教师，方法是前置可学习的 prompt token（仅增加教师参数的 0.0007%），使教师产生更"对学生友好"、更契合学生有限容量的分布。事后方法（Veto）作用于已经产生之后的监督信号，而事前方法（PromptKD）则修改教师自身的生成过程。两者都有效，且可以互不干扰地组合。

**通过共享 verbalizer 进行跨语言 logit 迁移。**上述跨 tokenizer 方法作用于单语推理数据。DuDi (Payoungkhamdee et al., 2026) 把同样的双空间原则适配到面向东南亚小语种模型的多语种设定中。一个跨语言 verbalizer 在教师与学生的词表之间映射语义对齐的内容，使 token 级蒸馏能够在共享的 verbalizer 空间上进行，即便跨语言的表面分词形式不同。训练信号由两个分量组成：一个在线序列级的 SPIN 式目标，用于改善学生在首选语言上的回答；以及一个 token 级 KD 项，把 off-policy 教师数据与 on-policy 学生 rollout 混合起来。on-policy 这一路为学生在低资源语言中自行生成的错误提供纠正性的逐 token 反馈，与在教师数据充足的语言上的 off-policy 示范互为补充。这一构造表明，只要有共享 verbalizer 可用，SimCT 那种"拓宽匹配粒度"的思想同样适用于跨语言场景。

**表征空间与跨 tokenizer 的扩展。**还有两种方法把白盒 OPD 推进到 logit 层之外。OPRD (Yang et al., 2026d) 通过标准 logit 蒸馏之外再增加一个隐状态对齐项，把 OPD 扩展到表征空间。on-policy 学生 rollout 同时触发通常的逐 token KL 损失和一个逐层表征匹配目标，从而在迁移教师输出分布之外，还迁移编码了中间推理结构的教师内部激活。BTB (Niu et al., 2026) 通过在不兼容的 tokenizer 之间构造一个显式的 token 到 token 映射桥来处理异构模型家族。BTB 不依赖共享词表子集或基于投影的对齐（如 DSKD 与 SimCT），而是学习一个轻量映射，把每个教师 token 映射到最接近的学生侧 token，并在这个桥接后的词表上应用标准的 on-policy KD。该设计把 OPD 扩展到诸如 Llama-to-Qwen 这样没有任何 tokenizer 重叠的跨家族对，用直接的 token 级映射补充了 SimCT 的多 token 续写思路。

#### 5.1.2 黑盒与受 API 约束的蒸馏（Black-Box and API-Constrained Distillation）

当只能获得 API 访问时，学生观察到的是教师的 top-1 文本输出（或最多是 top-k log 概率），而不是完整分布（表 7）。这一信息瓶颈排除了 token 级的散度匹配，迫使方法在序列层面运作。尽管存在这一限制，黑盒方法在实践中据报道表现良好，

<!-- p. 27 -->

即在所研究的设定中，教师知识中有相当一部分可以从纯文本观察中恢复出来。

**逐步蒸馏（Distilling Step-by-Step）。**Hsieh et al. (2023) 提出了这一奠基性的黑盒方法，不仅抽取教师的最终答案，还抽取其中间推理步骤（理据）作为额外监督。尽管从技术上讲这是 off-policy 的（学生是在静态的、由教师生成的理据上微调，而不是在自己的输出上微调），但这项工作是在策略 CoT 蒸馏的概念先驱，并确立了一条关键原则。学生以结合答案预测与理据生成的多任务目标进行训练，使得一个 770M 参数的模型能以超过 700× 更少的参数和显著更少的训练数据（平均少 50% 的样本，某些任务上最多减少 85%）超越一个 540B 参数的教师。这一结果表明，教师**如何**推理可能比它**得出什么**结论更有价值，这一原则在现代 OPD 中反复出现，并推动了下面这些 on-policy 扩展。

如果抽取文本理据能恢复教师推理的一部分，那么进一步的问题是：能否仅从黑盒输出中重建出教师完整分布的更丰富代理？下列方法尝试了越来越激进的还原方式，从对抗式判别，到言语打分，再到偏好排序。

**对抗式分布匹配。**GAD (Ye et al., 2025) 把蒸馏问题重新表述为一场生成对抗博弈：

$$
\max_G \min_D V(G, D) = \mathbb{E}_{(x, y_T)\sim T}\left[-\log\sigma\left(D(y_T) - D(G(x))\right)\right]
$$

判别器 $D$（由学生初始化并加一个标量预测头）通过 Bradley-Terry 偏好模型区分学生 rollout 与教师 API 输出，而生成器 $G$ 通过策略梯度（以 GRPO 实现）最大化 $V$，使用判别器分数作为一个不断演化的 on-policy 奖励信号。与匹配完整分布的白盒方法不同，GAD 通过一个标量质量信号抽取知识，以分布保真度换取 API 兼容性。它继承了对抗性目标的训练不稳定性，因为判别器可能过拟合到风格线索而非语义质量，但相对 SFT 基线能稳定地带来增益。

**言语反馈与课程。**Lion (Jiang et al., 2023) 利用教师作为自然语言批评者的能力，通过一个三阶段对抗循环来实现，包括：(i) **模仿**阶段，学生在教师生成的回答上微调；(ii) **判别**阶段，教师找出学生仍然表现不足的指令；以及 (iii) **生成**阶段，教师针对已识别出的学生弱点构造更难的指令。这一闭环课程稳步缩小教师—学生差距，且不需要从教师获得任何梯度信息，因此适用于任何遵循指令的 API。Lion-13B 仅用 70K 条训练样本就在 BIG-Bench Hard 和 AGIEval 上达到有竞争力的性能。

**分布对齐序列蒸馏（Distribution-Aligned Sequence Distillation, DASD）**(Yan et al., 2026) 重新审视了在教师生成的推理轨迹上做 SFT 这一主流做法，并指出三点局限，即：(i) 从单次 best-of-n 采样中无法充分表示教师的序列级分布；(ii) 教师输出分布与学生能力之间存在错配；(iii) 教师强制（teacher-forced）训练带来的 exposure bias。其分布对齐流水线通过为每个 prompt 生成多条多样化的教师轨迹、并通过序列级匹配把学生的生成分布对齐到教师分布，同时应对这三点，仅用 448K 条训练样本就达到很强的性能。DDT (Zhang et al., 2026f) 用一个理论框架补充了 DASD 的实证发现，解释了为什么 RL 在推理任务上比 SFT 泛化得更好。RL 的 on-policy 数据生成使训练与模型不断演化的分布保持一致，而不是与一个静态快照保持一致。

**言语分数蒸馏。**OVD (Xiong et al., 2026) 用来自教师模型的离散言语分数（0–9）进行轨迹匹配，取代 token 级概率匹配。这消除了词表之间 token 级对齐的要求，通过避免存储 logits 降低了内存消耗，并报告在单样本训练机制下 web QA 上最多 +12.9% 的绝对 EM 提升、数学 benchmark 上最多 +25.7% 的提升。这种粗粒度监督的有效性表明，on-policy 探索本身提供了 
构成学习信号中相当大的一部分，教师的角色更接近于在学生的轨迹之间进行选择，而不是纠正单个 token。

这些方法构成了一个"信号丰富度 vs. 访问要求"的谱系。对抗式方法能恢复教师分布的稠密代理，但需要复杂的训练动态（判别器协同训练、生成器—判别器不稳定性）。基于分数的方法更易训练，但会丢失分布信息，只能恢复一个标量质量信号。因此信号类型自然地映射到 API 访问层级：完全 logit 访问使 KL 方法成为可能（第 4 节），而无法获得 token 级 logits 则推动了对抗式或基于分数的方法。

**用于流水线分阶段 OPD 的 MoE 判别器。**PRISM (Wang et al., 2026g) 为黑盒蒸馏贡献了一项独特的架构创新：在 SFT 与 RLVR 之间插入一个基于 OPD 的**预对齐**阶段，针对的是这样一个观察——从 SFT checkpoint 直接进行 RLVR 会因 SFT 输出与奖励对齐行为之间的巨大分布差距而遭遇冷启动不稳定。PRISM 不像 GAD 那样使用单一整体的判别器，而是采用一个 Mixture-of-Experts 判别器，带有专门化的感知专家头与推理专家头，提供**解耦的**纠正信号，以区分视觉 grounding 错误与逻辑推理失败。整场对抗博弈完全在回答层面运作，不需要教师 logits。学生生成 on-policy 回答，一个黑盒教师（Gemini 3 Flash）生成参考示范，MoE 判别器学习区分二者，同时通过其专家路由给出差异化的反馈。这种解耦反馈与多模态模型的结构相匹配——在多模态模型中失败模式是异质的，因为一个模型可能正确地感知到图像但在其推理上出错，反之亦然。在 4B 与 8B 规模的 Qwen3-VL 上，仅预对齐阶段就比直接的 SFT-to-RLVR 流水线分别提升 +4.4 与 +6.0 分，并且随后的 RLVR 训练收敛得更快、更稳定。因此，黑盒 OPD 充当一种结构化的预热，为下游 RL 优化塑造策略地形。

**基于 rubric 的语义蒸馏。**ROPD (Fang et al., 2026) 挑战了"有效的 OPD 需要 token 级 logit 访问"这一假设，提出以结构化语义 rubrics 作为教师 logits 的可扩展替代。一个 Rubricator 通过对比教师与学生的 rollout 归纳出针对具体 prompt 的评测 rubric，一个 Verifier 则依据这些 rubric 为学生 rollout 打分，以驱动 on-policy 优化。教师模型同时承担这两个角色，而它与训练循环相互独立这一点使得离线 rubric 生成成为可能，从而大幅降低 GPU 需求。在多种 benchmark（AIME 24/25、HMMT 25、GPQA-Diamond、HealthBench、IFEval）与多种配置（Qwen3-4B 与 Gemma3-4B 学生，配 GPT-5.2 与 Qwen3-30B 教师）上，ROPD 在实现约 10× 样本效率的同时，达到或超过基于 logit 的 OPD 方法。在开启 thinking 模式的 AIME 2025 上，Qwen3-4B 学生（68.75%）超过了其 GPT-5.2 教师（67.08%）：rubric 增强的优化在有利配置下，即便没有梯度级访问，也能支撑"学生超过教师"的性能。ROPD 在黑盒谱系中占据一个独特位置：GAD 通过对抗判别恢复标量质量信号，OVD 通过言语分数恢复，而 ROPD 恢复的是**结构化的多维**评测准则，把质量分解为可解释的 rubric 维度，从而在保持完全黑盒兼容的同时给出比分数更丰富的反馈。

**跨架构的偏好蒸馏。**ORPO-Distill (Singh et al., 2025) 把跨架构的黑盒蒸馏重塑为一个带混合策略负样本的偏好优化问题。多样化的教师 CoT 轨迹（以 K=8、temperature 0.8 采样，并用 ROUGE-L 去重）作为正样本，学生生成的错误轨迹作为负样本，用 ORPO 对比损失（λ=1）取代 token 级 KL。一个 $\phi=0.5$ 的混合策略采样器在取自基座 checkpoint $\theta_0$ 的负样本与取自最新 checkpoint $\theta_{t-1}$ 的负样本之间交替，从而在缓解 exposure bias 与保持固定参考策略的稳定性之间取得平衡。在五个多选 QA benchmark 上，该构造优于纯 off-policy 与纯 on-policy 基线，且蒸馏可以跨架构迁移（例如把 Llama 家族教师迁移到 Qwen 家族学生），而两侧都不需要 logit 访问。 
**分块投机式验证（Chunk-level speculative verification）。** OmniOPD (Zhou et al., 2026b) 在不要求教师 logit 的情况下缩小了黑盒与白盒 OPD 之间的差距。一个峰值熵调度器在学生的 rollout 内部挑选分块边界，黑盒教师被以同一前缀为条件查询 $N$ 次蒙特卡洛续写，而一个 Dirichlet-Multinomial 贝叶斯平滑器把分块级语义相似度转换为逐分块的奖励分布。被审核分块内部的 token 接收来自已验证教师分布的监督，而未被审核 token 上的信任域 KL 锚点则防止分块外的梯度漂移。该方法仅用黑盒查询就恢复了白盒 OPD 性能的相当一部分，说明当教师只能通过 API 访问时，分块级蒙特卡洛验证可以替代稠密的 logit 信号。

**Off-policy 知识整合（Off-policy knowledge integration）。** 并非所有训练信号都必须源自 on-policy。把来自 off-policy 来源的知识（例如已有的教师生成数据集）纳入 on-policy 框架会带来它自身的困难。LUFFY (Yan et al., 2025) 通过把 GRPO 扩展为一个混合策略目标来处理这一问题，该目标在学生的 on-policy rollout 之外还纳入 off-policy 推理轨迹。关键的设计挑战是缓解表层模仿，因为如果不加修正，学生会倾向于复制 off-policy 轨迹中的高概率 token，而忽略低概率但重要的推理步骤。LUFFY 通过正则化重要性采样进行的策略塑形对梯度重新加权，以强调从不熟悉但有效的动作中学习，相比标准 RLVR 平均提升 +6.4 分。

在两个访问层级上，外部教师蒸馏提供了 OPD 中最稠密的监督，代价是受访问方式约束的限制。白盒方法通过全词表 KL 达到最紧密的分布对齐，但需要与教师共同托管；黑盒方法则用信号密度换取部署灵活性。尚未解决的张力在于：最强的教师，即专有的前沿模型，通常只能通过 API 访问，而这恰恰是 OPD 所依赖的稠密逐 token 信号最难获得的场景。下一节的自蒸馏方法完全去掉了外部教师，用这种依赖换取另一种局限，我们将在下文考察。

### 5.2 自蒸馏（Self-Distillation）

自蒸馏完全去除了对外部教师的需要（表 8）。取而代之的是，模型利用**自身内部**的不对称性来构造训练信号。目前已经出现三个方法族，它们通过这种内部不对称性的来源相区分，即：训练时可获得而推理时不可获得的特权信息；通过 rollout 多样性和架构式自训练的纯自蒸馏；以及来自验证器或环境、为自生成信号提供锚点的外部反馈。

**范围界定（Scope boundary）。** 本节涵盖那些训练信号来自同一模型不同视图之间分布对齐的方法（有条件下与无条件、冻结与更新、温度采样与贪心）。我们排除博弈论式的自博弈方法，例如 SPIN (Chen et al., 2024) 与 IRIS (Liao et al., 2026)，它们的核心机制是相继 checkpoint 之间的对抗竞争，目标是在一个参考分布上收敛到 Nash 均衡。那种框架在概念上不同于定义 on-policy distillation 的分布对齐。这些博弈论方法将在第 9 节作为相邻范式简要讨论。

#### 5.2.1 特权信息（Privileged Information）

第一个方法族通过**特权信息**（privileged information, PI）制造不对称性，即在训练时可以访问推理时不可获得的上下文。把同一个模型条件化于额外上下文会产生一个更强的策略，它可以监督无条件版本，从而把单个网络同时变成教师和学生。下文各方法的主要差异在于什么充当了该上下文的角色，其范围从真实答案，到检索到的文档、空间掩码、结构化评判反馈、未来观测，再到一个弱的外部批判。这种表面上的多样性在很大程度上与蒸馏机制本身正交，后者仍然是在学生 rollout 上做 PI 条件自教师的 token 级匹配。真正有后果的差异在于特权信号的**类型**。一条跨实例通用的共享潜在

<!-- p. 30 -->

**表 8（Table 8）：OPD 方法：自蒸馏。**

| 方法                                        | 年份 | 类别         | 散度/目标                     | 信号                   | 粒度       | 关键创新                                                   |
| ------------------------------------------- | ---- | ------------ | ----------------------------- | ---------------------- | ---------- | ---------------------------------------------------------- |
| OPSD (Zhao et al., 2026b)                   | 2026 | 自蒸馏（PI） | 以 GT 为条件的自身上的 KL     | 自身                   | Token      | 真实答案作为特权信息                                       |
| CRISP (Sang et al., 2026)                   | 2026 | 自蒸馏（PI） | 简洁性 PI                     | 自身                   | Token      | token 减少 57%，准确率 +9%                                 |
| GATES (Stein et al., 2026)                  | 2026 | 自蒸馏（PI） | 门控一致性 KL                 | 自身                   | Token      | 文档作为 PI，门控监督                                      |
| OPSDL (Zhang et al., 2026g)                 | 2026 | 自蒸馏（PI） | 短上下文→长上下文            | 自身                   | Token      | 长上下文自蒸馏                                             |
| SD-ZERO (He et al., 2026)                   | 2026 | 自蒸馏（EF） | 自修订偏好                    | 自身 + 验证器          | Sequence   | 二元奖励→稠密自监督                                       |
| π-Play (Zhang et al., 2026l)               | 2026 | 自蒸馏（EF） | 多智能体协同演化              | 自身                   | Token      | 由协同演化内部生成 PI                                      |
| UniSD (Jin et al., 2026c)                   | 2026 | 自蒸馏（SD） | 多轴统一 SD                   | 自身（EMA + 一致性）   | Token      | 系统化的组件分析                                           |
| RLRT (Kim et al., 2026a)                    | 2026 | 自蒸馏（EF） | 反转的教师信号                | 自身（反转）           | Token      | 强化自驱动的推理 token                                     |
| SDPO (Hübötter et al., 2026)              | 2026 | 自蒸馏（EF） | 文本反馈 + DPO                | 自身 + 验证器          | Sequence   | 由结果进行信用分配                                         |
| RLTF (Song et al., 2026)                    | 2026 | 自蒸馏（EF） | 以批判为条件                  | 自身 + 验证器          | Sequence   | 用于自蒸馏的文本反馈                                       |
| SRPO (Li et al., 2026c)                     | 2026 | 自蒸馏（EF） | GRPO + SDPO 路由              | 自身 + 验证器          | Sequence   | 样本路由的多目标                                           |
| CoPD (Gu et al., 2026)                      | 2026 | 自蒸馏（EF） | 协同演化的双向 OPD            | 互相（并行）           | Sequence   | 多能力整合                                                 |
| VPD (Li et al., 2026i)                      | 2026 | 自蒸馏（EF） | 变分 EM 协同演化              | 自身（协同演化）       | Token      | 由语言反馈驱动的信任域教师自适应                           |
| PAINT (Tan & Hong, 2026a)                   | 2026 | 自蒸馏（EF） | 部分解掩码                    | 自身（重新打分）       | Token      | 重叠度自适应自蒸馏                                         |
| RESD (Zhang et al., 2026n)                  | 2026 | 自蒸馏（EF） | 反思增强的自蒸馏              | 自身 + 环境反馈        | Token      | 失败反思 + 持久化 playbook                                 |
| MSD (Qin et al., 2026)                      | 2026 | 自蒸馏（PI） | 跨语言安全蒸馏                | 自身（英文 PI）        | Token      | 多语言安全对齐                                             |
| GUI-SD (Zhang et al., 2026i)                | 2026 | 自蒸馏（PI） | 用于 GUI 的视觉空间 PI        | 自身（高斯掩码）       | Token      | 多模态 GUI 定位                                            |
| PBSD (Yu et al., 2026c)                     | 2026 | 自蒸馏（PI） | 基于偏好的奖励正则 KL         | 自身（上下文增强）     | Sequence   | DPO 式损失，偏好间隔                                       |
| TT-OPD (Jeong, 2026)                        | 2026 | 自蒸馏（PI） | 轮次级截断 KL                 | 自身（EMA + 结果提示） | Turn       | 多轮智能体 OPD                                             |
| VISD (Lin et al., 2026a)                    | 2026 | 自蒸馏（PI） | 方向-幅度解耦                 | 自身（结构化评判者）   | Token      | 用于视频推理的结构化 PI                                    |
| π-Distill (Penaloza et al., 2026)          | 2026 | 自蒸馏（PI） | 统一 PI 框架                  | 自身                   | Token      | PI 自蒸馏的理论基础                                        |
| OPCD (Ye et al., 2026b)                     | 2026 | 自蒸馏（PI） | 上下文条件 RKL                | 自身（经验 PI）        | Token      | 经验性知识的内化                                           |
| OEL (Ye et al., 2026a)                      | 2026 | 自蒸馏（PI） | 在线经验学习                  | 自身（轨迹 PI）        | Token      | 部署后持续蒸馏循环                                         |
| HDPO (Ding, 2026)                           | 2026 | 自蒸馏（PI） | GRPO + JSD 自蒸馏             | 自身（GT 条件）        | Sequence   | cliff prompt 的非零梯度补丁                                |
| MTP-SD (Kirchenbauer et al., 2026)          | 2026 | 自蒸馏（SD） | 多 token 跨度打分             | 自身（冻结副本）       | Span       | 面向 >3× 解码的架构式自蒸馏                               |
| SDFT (Shenfeld et al., 2026)                | 2026 | 自蒸馏（SD） | 上下文示例教师 RKL            | 自身（ICL 条件）       | Token      | 通过信任域隐式 KL 抗遗忘                                   |
| RLSD (Yang et al., 2026a)                   | 2026 | 自蒸馏（EF） | RLVR + 自蒸馏分解             | 自身 + 验证器          | Token      | 幅度（自蒸馏）+ 方向（RLVR），2× 效率                     |
| SDAR (Lu et al., 2026)                      | 2026 | 自蒸馏（EF） | 门控 OPSD + RL                | 自身（PI）+ RL         | Token      | 面向智能体的 sigmoid 门控辅助自蒸馏                        |
| OPHSD (Zhao et al., 2026c)                  | 2026 | 自蒸馏（PI） | harness 增强的自 KL           | 自身（harness PI）     | Token      | 把推理脚手架内化为 PI                                      |
| COPSD (Liu et al., 2026h)                   | 2026 | 自蒸馏（PI） | 跨语言 token KL               | 自身（英文 PI）        | Token      | 通过跨语言 PI 覆盖 17 种非洲语言                           |
| ATESD (Han et al., 2026b)                   | 2026 | 自蒸馏（PI） | Beta 策略暴露控制             | 自身（自适应 PI）      | Token      | 可学习的教师暴露比例                                       |
| TRACE (Wang et al., 2026e)                  | 2026 | 自蒸馏（PI） | token 路由的 FKL/RKL          | 自身（标注者 PI）      | Span       | 关键跨度 OPD + 其余部分 GRPO                               |
| ListOPD (Li et al., 2026h)                  | 2026 | 理论         | 奖励外推的裁剪安全性          | 白盒                   | Token      | 针对格式坍塌的闭式$\lambda^\star$                        |
| Rock Tokens (Jiang et al., 2026b)           | 2026 | 理论         | 持续性高损失分析              | 白盒                   | Token      | 18% 结构上无信息量的 token                                 |
| Many Faces (Zhu et al., 2026)               | 2026 | 理论         | 全面的失败目录                | 白盒/自身              | Token      | 有偏的 top-K KL + PI 类型局限                              |
| Unmasking OPD (Armandpour et al., 2026)     | 2026 | 理论         | 逐 token 梯度对齐             | 白盒/自身              | Token      | 免训练的诊断框架                                           |
| Local Collapse (Liu et al., 2026b)          | 2026 | 理论         | BIC 式发布规则                | 白盒                   | Segment    | 后缀 OPD 中的局部可教性坍塌                                |
| Safactory (Chen et al., 2026d)              | 2026 | 应用         | 安全演化流水线                | 白盒                   | Trajectory | 面向智能体安全的闭环 OPD                                   |
| HyperEyes (Li et al., 2026d)                | 2026 | 应用         | 双粒度高效 RL                 | 白盒                   | Token/Seq  | 结合 OPD 的多模态搜索智能体                                |
| ProteinOPD (Zhang et al., 2026m)            | 2026 | 应用         | 多教师共识 OPD                | 白盒（多教师）         | Token      | 蛋白质 PLM 偏好对齐                                        |
| TTS (Kumar et al., 2026)                    | 2026 | 应用         | 在线测试时 OPD                | 白盒                   | Token      | 面向投机解码的 draft 模型自适应                            |
| DAgger-LLM (Li et al., 2026a)               | 2026 | 应用         | 轮次级学生-教师混合           | 白盒                   | Turn       | 面向多轮 LLM 智能体的 DAgger                               |
| AVSD (Nguyen et al., 2026)                  | 2026 | 自蒸馏（PI） | 多视图共识 + 残差             | 自身（多视图）         | Token      | 分解共识与视图特有部分                                     |
| Skill-Cond. Gated SD (Huang et al., 2026)   | 2026 | 自蒸馏（PI） | 结果验证的极性                | 自身（多教师）         | Token      | 结果门控的有界 SD                                          |
| Critique Distill (Jin et al., 2026a)        | 2026 | 自蒸馏（PI） | 弱批判者条件 KL               | 自身（弱批判者）       | Token      | 可扩展监督的反转                                           |
| Constitutional Cross-SFT (Wen et al., 2026) | 2026 | 自蒸馏（PI） | 跨 SFT 冷启动 + RKL           | 自身（宪法）           | Token      | 安全性与表达力的保持                                       |
| World-Model PI (Zhou et al., 2026a)         | 2026 | 自蒸馏（PI） | 优势加权的决策节点 KL         | 自身（未来视频 PI）    | Step       | 验证/调用/依赖世界模型 rollout                             |
| SD-PG (Liu et al., 2026g)                   | 2026 | 自蒸馏（PI） | 全词表 RKL + REINFORCE        | 自身                   | Token      | 带正优势门控的全词表 SD                                    |
| VPG (Chen et al., 2026e)                    | 2026 | 自蒸馏（SD） | 长度感知的变分后验            | 自身                   | Sequence   | 效率感知的推理自蒸馏                                       |
| Self-Sup OPD (Tan & Hong, 2026b)            | 2026 | 自蒸馏（SD） | 组内 KL                       | 自身（GRPO 组）        | Token      | 正确/错误 rollout 对比                                     |
| HINT-SD (Yeo et al., 2026)                  | 2026 | 自蒸馏（SD） | 定向的事后重标注              | 自身                   | Turn       | 与失败相关的轮次选择                                       |
| SD-Search (Ma et al., 2026)                 | 2026 | 自蒸馏（SD） | 步级事后重标注                | 自身                   | Step       | 搜索查询级监督                                             |
| Vision-OPD (Yuan et al., 2026)              | 2026 | 自蒸馏（SD） | 区域→全局 RKL                | 自身（区域 PI）        | Token      | 细粒度 MLLM 定位                                           |
| It Takes Two (Park et al., 2026)            | 2026 | 自蒸馏（SD） | 互补并行 SD                   | 自身（互相切分）       | Token      | 上下文完整性的隐私对齐                                     |
| TOD Proactivity (Zhang et al., 2026a)       | 2026 | 自蒸馏（SD） | 非对称用户关切 PI             | 自身（关切）           | Token      | 主动式任务导向对话                                         |
| Search-E1 (Liang et al., 2026b)             | 2026 | 自蒸馏（SD） | GRPO + 离线 SD 演化           | 自身（快照）           | Sequence   | 面向搜索推理的自演化循环                                   |
| MAIGO (Zheng et al., 2026b)                 | 2026 | 自蒸馏（SD） | 历史清理后的 RKL              | 自身（单轮 PI）        | Token      | 多轮对话中的信息丢失 gap                                   |
| ROSD (Zhao et al., 2026d)                   | 2026 | 自蒸馏（SD） | 反思 + 引文定位               | 自身（反思器）         | Token/Span | 针对错误后缀的 SD                                          |
| Canonical-Ctx OPD (Lin et al., 2026b)       | 2026 | 自蒸馏（SD） | 答案掩码 RKL                  | 自身（全上下文 PI）    | Token      | 多轮规范上下文对齐                                         |
| COMAP (Liu et al., 2026i)                   | 2026 | 自蒸馏（SD） | EMA SD + 反思门控             | 自身（世界模型）       | Trajectory | 协同演化智能体与世界模型                                   |
| OPCT (Han et al., 2026a)                    | 2026 | 自蒸馏（EF） | 冻结副本对比 RKL              | 自身（冻结副本）       | Token      | 谄媚/越狱一致性                                            |
| MGSD (Luo et al., 2026b)                    | 2026 | 自蒸馏（PI） | 模态差距感知的两阶段 SD       | 自身（模态 PI）        | Token      | 通过桥接模态差距进行空间规划                               |
| PTD-PO (Xiang et al., 2026)                 | 2026 | 自蒸馏（PI） | Top-K JSD 特权辅导            | 自身（空间+推理 PI）   | Token      | 面向多模态策略优化的空间+推理提示                          |
| TWI (Cai et al., 2026a)                     | 2026 | 自蒸馏（PI） | 裁剪图像放大 PI               | 自身（裁剪 PI）        | Token      | 特权放大推理内化视觉定位                                   |
| PBSD∗ (Tian et al., 2026)                  | 2026 | 自蒸馏（PI） | 贝叶斯轮次级 SD               | 自身（轨迹奖励）       | Turn       | 稀疏轨迹奖励→轮次级信用                                   |
| Visual-SDPO (Dong, 2026)                    | 2026 | 自蒸馏（PI） | 语句加权 KL + GRPO            | 自身（视觉产物 PI）    | Token      | 把渲染出的视觉产物作为特权反馈                             |
| AR-OPD (Zhang, 2026)                        | 2026 | 自蒸馏（PI） | 锚点+oracle 残差 OPD          | 自身（oracle PI）      | Token      | 防止事后泄漏；相比完整 OPD +2.3                            |
| HERO (Liu et al., 2026a)                    | 2026 | 自蒸馏（PI） | 带未来观测 PI 的轮次级 KL     | 自身（未来环境 PI）    | Turn       | 把未来环境观测作为特权上下文                               |
| When Ctx Returns (Wang et al., 2026h)       | 2026 | 自蒸馏（PI） | FKL 无上下文锚定              | 自身（无上下文 PI）    | Token      | 锚定正则项防止上下文导致的退化                             |
| Rubric-Guided SD (Rezaei et al., 2026)      | 2026 | 自蒸馏（PI） | 带评分量表 PI 的 JSD          | 自身（评分量表 PI）    | Token      | 通过评分量表条件的 SD 免除对外部验证器的需求               |
| AR-to-Diffusion (Su et al., 2026)           | 2026 | 自蒸馏（SD） | AR→diffusion 的 on-policy SD | 自身（冻结 AR）        | Token      | 把 AR LM 转换为 diffusion LM；训练 token 减少 15×–7000× |
| SG-OPD (Xu et al., 2026a)                   | 2026 | 自蒸馏（EF） | 符号一致性门控 SD             | 自身 + 验证器          | Token      | 分阶段教师采样；+1.98 avg@32 / +7.50 pass@32               |
| OPCoD (Byeon et al., 2026)                  | 2026 | 自蒸馏（EF） | 耦合协同蒸馏 + 同伴反馈       | 互相（同伴）           | Token      | 跨域互惠的 Pareto 改进                                     |

> 注：原表中表头"Signal Granularity"是跨两列的合并单元格，此处拆分为"信号"与"粒度"两列；表中"自蒸馏（PI）"即 Self (PI)，"自蒸馏（EF）"即 Self (EF)，"自蒸馏（SD）"即 Self (SD)。

规则（rule that applies across instances，例如一条系统提示、一部宪法、一个评分量表）可以被内化进一个不含 PI 的策略，而一个实例特定的答案则无法以同样的方式被聚合，这一区别界定了 PI 蒸馏何时能够成功，我们将在第 7.2 节的失败分析中回到这一点。因此我们按照与最初"以真实答案为 PI"这一构造距离递增的顺序来呈现各方法，并在每个方法处说明它引入了何种特权信号、解决了哪一局限。

**以真实答案为 PI（Ground-truth as PI）。** OPSD (Zhao et al., 2026b) 通过同时以问题 x 和真实答案 y⋆ 为条件来构造教师策略，而学生只以 x 为条件。学生生成 on-policy rollout，教师提供稠密的 token 级监督。

$$
L_{\text{OPSD}}(\theta)=\mathbb{E}_{(x,y^\star)\sim S}\ \mathbb{E}_{\hat{y}\sim p_{\theta}(\cdot|x)}\left[\sum_{n=1}^{|\hat{y}|}D\left(p_{\theta}(\cdot|x,y^\star,\hat{y}_{<n})\,\|\,p_{\theta}(\cdot|x,\hat{y}_{<n})\right)\right]
$$

在竞赛级数学基准（AIME 2024/2025、HMMT）上，OPSD 在 4B 和 8B 规模上达到或超过 GRPO，而每个问题只使用一次 rollout，相比之下 GRPO 需要八次，从而带来显著的计算优势。增益在 8B（相比 GRPO +0.9 平均分）与 4B（相比 GRPO +0.8）最大，但在 1.7B 上它不如 GRPO（-0.5），这可能是因为 OPSD 依赖自我合理化（self-rationalization），需要足够的模型容量才能产生有意义的稠密 token 级信号。作者指出一个理论局限：当问题超出模型的理解阈值时，即使以真实答案为条件的教师也无法提供有意义的监督，这是 PI 机制中存在一个随模型容量缩放的问题难度天花板的证据。

另一个单独的局限出现在给定提示的所有 rollout 都失败时（"cliff prompt"问题），此时策略梯度完全消失。这一失效模式直接 
**放宽 ground-truth 要求。**Ground-truth 标签并不总是可用。GATES (Stein et al., 2026) 表明，多种形式的训练期上下文都可以充当 PI：让单个模型既作为 tutor（以源文档为条件）又作为 student（仅根据问题作答）。由于源文档并不保证正确回答，GATES 提出一种基于共识的门控机制，在 tutor 不确定性较高时抑制梯度，防止模型强化不可靠的自监督。π-Distill (Penaloza et al., 2026) 是一项更早的形式化工作，把 OPSD 式的特权自蒸馏范式表述为一个统一的理论框架，并识别出特权自蒸馏能够可证明地优于标准训练的条件。RESD (Zhang et al., 2026n) 把 PI 范式扩展到成功 rollout 极为罕见（通过率低于 5%）的设定。RESD 不丢弃失败的轨迹，而是把它们视为一种负向特权信息：一个 reflection 模块从失败中蒸馏出结构化的诊断性反馈，写入一份持续存在的 playbook，用以条件化后续的 on-policy 生成，从而在低通过率任务上提供稠密监督，其样本效率比 GRPO 高出约 8×。

**行为型 PI：上下文蒸馏。**PI 范式不止于事实性 ground truth，还扩展到行为指令。OPCD (Ye et al., 2026b) 把累积的经验知识（历史解题轨迹）与优化后的 system prompts 视为特权上下文，在模型自己的 rollouts 上训练学生，同时最小化相对于一个上下文条件化教师的反向 KL，使模型能够内化那些在推理时否则需要昂贵 in-context 示范才能获得的行为。OPSDL (Zhang et al., 2026g) 把这一点扩展到长上下文设定：模型自身的短上下文能力充当 self-teacher，通过 token 级反向 KL 监督其 on-policy 的长上下文生成，从而在无需依赖高质量监督数据或稀疏序列级奖励的情况下增强长上下文能力。OEL (Ye et al., 2026a) 进一步扩展到持续的部署后学习，从交互轨迹中抽取可迁移的经验知识作为动态 PI，并以 on-policy 巩固迭代地进行知识抽取，形成一个在线学习闭环，闭合了 OPCD 所打开的环路。

**通过 PI 进行推理压缩。**CRISP (Sang et al., 2026) 发现，一个被提示"保持简洁"的模型可以充当其自身冗长版本的教师，在 MATH-500 上把 chain-of-thought 的 token 数减少 57–59%，同时把准确率**提升** 9–16 个百分点。蒸馏得到的推理模型中的大量冗长内容是可训练的低效，而不是胜任推理的必要特征。

**面向 GUI grounding 的视觉 PI。**GUI-SD (Zhang et al., 2026i) 把特权信息范式扩展到多模态 GUI grounding，其任务是在给定自然语言指令的情况下预测屏幕上的点击坐标。特权上下文是一个以目标元素 bounding box 为中心的高斯软 mask，使教师版本拥有显式的空间定位信息，而学生必须学会仅从原始截图中推断出这一信息。这代表了一种在性质上不同于文本 ground truth（OPSD）或文档上下文（GATES）的 PI 形式，它是**视觉的**、**空间的**，是 PI 机制可跨模态扩展的证据。除了这一新的 PI 类型，GUI-SD 还提出通过双准则进行熵引导的 token 加权：token 依据其数字显著性（坐标数字比填充 token 承载更多语义权重）以及教师在该位置上的置信度两者获得更高的损失权重。在每个实例仅用一次 on-policy rollout 的情况下，GUI-SD 在包括 ScreenSpot-v2、ScreenSpot-Pro 与 OSWorld 在内的六个 GUI grounding benchmark 上优于 GRPO（后者需要多次 rollout 来估计奖励）。单 rollout PI 蒸馏在视觉领域的成功与 OPSD 在数学推理中的发现相呼应，进一步表明特权信息能够提供足够稠密的梯度信号，从而减少对 RL 类替代方案所需的、依赖大量样本的奖励估计的依赖。

<!-- p. 32 -->

**把推理 harness 内化为 PI。**OPHSD (Zhao et al., 2026c) 把 PI 范式从事实性 ground truth 扩展到**推理时脚手架**。当 LLM 与外部的 harness 配对使用时（draft-verify 循环、plan-solve 工作流），收益来自流程而非模型。OPHSD 把 harness 增强后的模型用作教师进行自蒸馏，把特定任务的 harness 能力内化到学生中。在文本分类的 draft-verify 与数学推理的 plan-solve 上评测，OPHSD 优于基线（在 HMMT 2025 上比 OPSD 高 +10.83%），而在推理时重新挂上 harness 并不能带来额外收益，这证明复杂的 harness 可以作为临时训练支架，其收益被永久吸收。

**通过 PI 进行跨语言推理。**COPSD (Liu et al., 2026h) 把模型自身在高资源语言上的推理行为迁移到低资源语言。教师接受特权的跨语言上下文（问题的英文翻译与参考解），而学生只能看到低资源语言的问题。训练在学生的自身 rollouts 上最小化全分布的 token 级散度，在提供稠密监督的同时避免仅以结果为导向的 RL 所固有的稀疏性与不稳定性。在 17 种低资源非洲语言上，COPSD 提升了数学推理能力，并大幅优于 GRPO，且在监督差距最大的最低资源语言上增益最大。MSD (Qin et al., 2026)（见下文）是一项同期工作，独立地把基于跨语言特权信息的自蒸馏应用于安全对齐而非推理，证明"以英语作为 PI"在能力维度上具有泛化性。

**自适应教师曝光。**ATESD (Han et al., 2026b) 质疑几乎所有 PI 方法共享的一个假设：教师应当始终看到**完整**的参考推理。一次受控的曝光扫描表明，完整曝光并不总是最优的，而且随着教师看到更多特权上下文，学生—教师错配会单调增大。ATESD 用一个轻量的 Beta-policy 控制器对 reveal ratio 建模，该控制器以紧凑的训练状态统计量为条件，并用一个折扣的学习进度奖励来优化——该奖励按每次曝光决策对学生未来提升的影响、而不是按即时损失来打分。在 Qwen3-{1.7B, 4B, 8B} 上，ATESD 优于 OPSD（+0.95 至 +2.33 Average@12 分），把自适应教师曝光确立为推理自蒸馏的一个新维度。

**Token 路由的稀疏 self-OPD。**TRACE (Wang et al., 2026e) 认为，self-OPD 中的全 token KL 把梯度花在大多冗余的位置上，并放大特权信息泄漏，导致熵上升、推理变短以及分布外退化。TRACE 使用一个特权标注器标出每条 rollout 中的关键 span，对正确 rollout 的关键 span 施加 forward KL，并可选地对局部错误 span 施加 reverse KL，其余所有 token 则交给 GRPO。一个 KL annealing 调度使累计的特权梯度曝光保持有限。在四个数学 benchmark 加 GPQA-Diamond 上，TRACE 平均比 GRPO 提升 2.76 分，并且是其评测中唯一在 GRPO 与全 token self-OPD 基线退化的情况下仍保持 OOD 性能的训练方法。这些增益在在线自标注下依然存在（使用训练学生作为标注器时 +1.90 分），降低了对外部监督者的依赖。

**通过 PI 实现跨语言安全。**MSD (Qin et al., 2026) 与 COPSD (Liu et al., 2026h) 在不同领域独立探索了基于跨语言特权信息的自蒸馏，其中 MSD 聚焦多语种安全对齐，COPSD 聚焦跨语言数学推理（见上文）。在 MSD 中，同一模型的英文推理充当其低资源语言学生的特权信息，在无需目标语言安全数据的情况下迁移安全边界。这两项并行工作表明，跨语言 PI 是与能力无关的。其 Dual-Perspective Safety Weighting（DPSW）通过结合教师置信度与学生分歧，把蒸馏梯度集中在安全关键的 token 上。我们在第 9 节讨论 MSD 对安全感知 OPD 的更广泛意义。

**超越 KL 匹配：通过奖励正则化。**上文 PI 文献中反复出现的一个担忧是：直接匹配特权教师的分布可能使训练不稳定，并且在最坏情况下会随着训练进行而降低推理能力。PBSD (Yu et al., 2026c) 通过一个奖励正则化的视角诊断这一病态，用目标 $\max_\pi \mathbb{E}[r(x, y)] - \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{teach}})$ 取代纯粹的 KL 匹配，其解析最优解是一个奖励重加权后的教师 $\pi^* \propto \pi_{\text{teach}}(y|x)\exp(r(x,y)/\beta)$，只要奖励是有信息量的，它就可证明地优于 
教师本身，只要奖励是有信息量的。在实践中，PBSD 通过一个 DPO 式的成对损失实现这一最优解：偏好样本 $y^+$ 取自上下文增强后的教师，非偏好样本 $y^-$ 取自当前学生，从而在保留 on-policy 采样的同时避免对每一个教师 token 的强制模仿。在 Qwen3-1.7B/4B/8B 上，PBSD 在保持 OPSD 的 token 效率的同时超过了其峰值准确率，并避免了 Yu et al. (2026c) 图 2 中记录的后峰值下降：因此，基于偏好的目标可以比单独的课程或信号增广补丁更稳定地替代直接 KL 匹配。这一后峰值下降与 CaOPD (Zhang et al., 2026d) 归因于训练—部署上下文错配、以及 Kim et al. (2026b) 诊断为 epistemic suppression（第 7.2 节）的病态是同一种，因此 PBSD 也可以被解读为一种自蒸馏机制，它把这些诊断洞见转化为一个纠正性的优化目标，而不是一个事后重校准步骤。

**面向智能体推理的、turn 级结果条件化 PI。**上述 PI 方法都针对单轮生成。把 PI 扩展到多轮智能体设定会引出一个关于粒度的独立问题：特权信息应当在哪个时间层级上介入？已有的智能体 OPD 工作已经探索过**轨迹**粒度（TCOD 对 turn prefix 的时间课程，第 6.2 节）、**步骤**粒度（MAD-OPD 的步骤级分解，第 8 节）以及**技能**粒度（Skill-SD 的技能条件化更新，第 8 节），但都没有在 **turn** 这一层级本身提供 PI。TT-OPD (Jeong, 2026) 填补了这一空白。一个无需梯度的 EMA 教师接受依赖正确性的提示（对评为正确的轨迹使用强化式短语，对评为错误的轨迹使用纠正式短语），这些提示被插入到 prompt-response 边界处，随后从教师的输出 log 概率中剥离，因此学生从不直接观察到提示，但在每一轮对话中仍然获得一个对结果有感知的 KL 信号。有两个方面使这一实例区别于此前的 PI 机制。第一，特权的上下文编码的是轨迹级的正确性，而不是问题级的 ground truth；据我们所知，这是一种 PI 实现方式，其中教师所条件化的对象是学生自身 rollout 的一个派生属性（其结果的正确性），而不是直接以问题级 ground truth 为条件。这把 PI 范式从单轮回答条件化监督（GATES、OPSD、GUI-SD）推广到多轮行为正则化。第二，消融实验分离出了 turn 级 PI 在此处为何重要：仅靠 EMA 教师更新可以减少 KL-collapse 病态，但仍会让多轮结构被侵蚀，而缺少长度控制的结果提示会引发回答爆炸，因此单独的 EMA 或单独的提示都不足够。这些病态正是第 7.2 节所记录失败模式中的一部分，而教师动态与结果条件化 PI 之间的相互作用，似乎正是这一设定下智能体 OPD 得以保持稳定的原因。

把 TT-OPD 与 TCOD、Skill-SD 和 MAD-OPD 放在一起看，它为智能体蒸馏补全了一个粒度谱系，其中时间单元（轨迹、turn、步骤、技能）的选择要与目标任务中错误发生累积的层级相匹配。

**面向视频推理的结构化 PI。**VISD (Lin et al., 2026a) 把特权信息范式从通用二值信号（OPSD 的正确性条件化）扩展到面向 VideoLLM 推理的**结构化多维**反馈。一个具备视频感知能力的 judge 模型沿答案正确性、逻辑一致性以及时空 grounding 质量三条轴评价 rollout，产生结构化反馈，把推理错误分解为具有诊断意义的子类型。其核心机制是方向—幅度解耦。来自环境奖励的 rollout 级 advantage 决定**更新方向**（强化或惩罚哪些轨迹），而以 judge 反馈为条件的结构化教师—学生差异则调制 **token 级的更新幅度**。这既保证了 RL 稳定性，又允许进行细粒度的信用分配，使之与具体的推理错误类型对齐，而不是无差别的 token 级匹配。结合课程调度（蒸馏→RL 的过渡）与针对长视频序列的 EMA 教师稳定化，VISD 达到比 RLVR 基线约快 2× 的收敛速度，并在视频推理 benchmark 上取得一致的准确率提升。方向—幅度解耦与 Self-Distilled RLVR 的分解方式（第 5.2.3 节）相呼应——后者由自蒸馏决定幅度、由 RL 决定方向——但 VISD 的结构化 judge 通过区分一个 token **为何**重要（grounding 错误 vs. 逻辑不一致 vs. 事实错误）提供了更丰富的幅度信号。因此，VISD 把 PI 谱系从二值（OPSD）经空间

<!-- p. 34 -->

（GUI-SD）扩展到多维结构化反馈，其中领域特定的错误分类体系决定了 PI 编码什么。

**多视角、批评与未来观测 PI。**随着特权信号越来越远离字面答案，它开始检验"什么仍可作为可用的 self-teacher"这一边界。AVSD (Nguyen et al., 2026) 迈出温和的一步，把同一输入在训练期的多个视角视为并行的 PI 来源，把逐 token 信号分解为一个跨视角共享的共识分量和一个视角特定的残差，使学生在各视角不一致时不会塌缩到单一教师视角。更大的偏离来自 On-Policy Critique Distillation (Jin et al., 2026a)，其 PI 不来自输入，而来自一个**弱**外部批评者。强学生进行 rollout，一个弱模型批评这些 rollout，一个结果与 rubric 过滤器只保留有用的批评，随后一个以批评为条件的 self-teacher 通过 token 级 KL 监督强学生。这颠倒了传统的教师—学生能力顺序，并实例化了可扩展监督（scalable oversight）：一旦弱监督被集中到 rubric 标记为有信息量的位置上，它就是足够的。World-Model PI (Zhou et al., 2026a) 走得最远，进入多模态前瞻。其特权上下文是 ground-truth 的未来视频与最终答案，训练期评测者可以获得，但学生的感知中并不存在。学生在无特权的上下文下 rollout，而评测者在中间决策节点构造一个候选集合，其 advantage 加权的蒸馏教会学生何时调用 world-model rollout、何时验证它、以及何时信任其结果。从输入扰动，经弱批评，到未来观测，这一梯度表明 self-teacher 可以容忍越来越抽象的特权信号，其约束条件并不在于信号的形式，而在于它在推理时是否仍然缺席。

**把 PI 与结果验证和校准组合起来。**一个反复出现的担忧是：一个以 PI 为条件的 self-teacher，无论其上下文多么丰富，仍然可能把学生引向自信的错误，因此第二组方法把特权信号与一个否决或重加权它的纠正机制配对。Skill-Conditioned Gated SD (Huang et al., 2026) 把纠正当作极性开关。它维护一个技能条件化的多教师池，并使用一个二值正确性信号 $r \in \{-1, +1\}$，在 rollout 正确时朝教师方向蒸馏、在错误时背离教师方向蒸馏，并通过 masking、clipping 与 thresholding 保持这个有界门控目标的稳定。Constitutional Cross-SFT (Wen et al., 2026) 把纠正提前到教师构造阶段。为避免朴素安全 SFT 所记录的安全—表达力塌缩，它插入一个 Cross-SFT 冷启动，在 on-policy 蒸馏开始之前校准受宪法条件化的教师，然后在校准后的教师下优化一个 token 级 reverse-KL 奖励，并用一项几何泄漏分析界定了读者能从学生中恢复出多少宪法内容。Self-Distilled Policy Gradient (Liu et al., 2026g) 把纠正放在蒸馏项旁边，把全词表 reverse-KL 自蒸馏与一个作用于组相对 advantage 的 REINFORCE 目标组合起来，仅在正 advantage 上启用 SD 项，使其跟踪成功的 rollout，同时由策略梯度项驱动去发现教师众数之外的解。在这三者中，特权信号提供密度，验证信号提供方向；每当仅靠 PI 有强化"看似合理但错误"的续写的风险时，这种分工就会重现。

**多模态空间与视觉 PI。**视觉模态提供了它自己的一套特权信号清单，一组方法表明：一旦识别出视觉特有的不对称性，同样的 OPSD 构造就能沿用，而它们各自处于不同的视觉粒度层级。在表征层级，MGSD (Luo et al., 2026b) 直接面对模态鸿沟，通过一个两阶段流程桥接视觉与语言表征之间的差异：先由 gap-bridging 阶段对齐特征分布，再由基于 PI 的阶段在一个 gap-aware self-teacher 下于学生自己的 rollout 上训练学生。在推理内容层级，PTD-PO (Xiang et al., 2026) 让教师同时获得空间上下文与完整推理轨迹（对学生保密），然后把一个 top-K JSD 目标集中在每条 rollout 中最有信息量的位置上。在原始感知层级，Thinking Without Images (Cai et al., 2026a)（TWI）让教师看到目标区域的全分辨率裁剪图，而学生只看到原始图像，于是 on-policy KD 内化了该裁剪图所揭示的细粒度 grounding。Visual-SDPO (Dong, 2026) 又把环路闭合回符号结构，把由问题构造出的图示或表格渲染为训练期 PI 
 channel，并结合 statement-weighted KL 与 GRPO，使每个 token 的损失都追踪所渲染产物的相关性。合起来看，这四种方法把视觉 PI 定位在一条从感知到抽象的轴上，这与上文文本方法围绕答案与规则所做的组织方式是同一举措。

**未来观测与上下文锚定的 PI（Future-observation and context-anchoring PI）**。HERO（Liu et al., 2026a）把智能体式 PI 从 ground-truth 条件化（TT-OPD）扩展到未来的环境观测。训练时，教师不仅接收当前状态，还接收下一时刻的环境观测，从而产生一个能预判未来上下文的策略。学生通过 turn 级 KL 蒸馏这种前瞻能力，而在推理时不接触未来观测。HERO 注入的是未来上下文，而 When Context Returns（Wang et al., 2026h）处理的是一个相关问题：在推理时加入长上下文会损害一个此前未在该上下文下训练过的模型。该方法从同一模型在不带额外上下文的情况下运行得到的结果中构造出一个“no-context”自教师，并施加 forward KL 作为锚定正则项，防止学生在上下文引发的分布偏移下抛弃其无上下文能力。

**贝叶斯信用再分配与 oracle 残差 PI（Bayesian credit redistribution and oracle residual PI）**。PBSD$^*$（Tian et al., 2026）（Yang Tian et al.，不要与 PBSD（Yu et al., 2026c）混淆）通过贝叶斯自蒸馏来处理多轮智能体 OPD 中稀疏的轨迹级奖励。turn 级信用由一个贝叶斯更新规则导出，它利用自教师的 log-probability 比值把轨迹奖励再分配给各个 turn，从而实现细粒度的逐 turn 监督，而无需为每个 turn 单独设置奖励模型。AR-OPD（Zhang, 2026）把 hindsight leakage 识别为完全 oracle 条件化的一种失效模式：享有特权的教师看到了完整的 oracle 答案，可能“向后泄漏”，从而扭曲学生较早的推理 token。AR-OPD 通过把蒸馏目标分解为一个锚定项（不带 oracle 的标准 on-policy KL）和一个残差 oracle 修正来缓解这一问题，后者仅施加于锚定项不确定性超过阈值的那些 token 上，相比完全 oracle OPD 取得 +2.3 分、相比 SFT 取得 +7.9 分。Rubric-Guided SD（Rezaei et al., 2026）把 rubric 评分标准作为 PI 提供，使教师以一份结构化评分 rubric 为条件，而学生必须在推理时不接触该 rubric 的前提下学会满足它。这种带 rubric 条件化的基于 JSD 的目标在推理时免除了对外部验证器的调用，同时达到与验证器监督相当的表现，从而把 rubric 内化界定为一项 OPD 任务。

#### 5.2.2 纯自蒸馏（Pure Self-Distillation）

特权信息通过额外上下文制造不对称性，而本节的方法则在没有任何辅助输入的情况下，从模型自身的输出分布中提取训练信号。其共同机制是 rollout diversity：通过在非零温度下采样，模型产生一族质量各异的补全，而方差本身提供了选择或蒸馏信号。这些方法位于一条 staleness 谱系之上，从完全 on-policy（训练分布每个梯度步都重新生成）到轻度 stale（rollout 每 epoch 生成一次并复用），stale 程度在算力成本与分布对齐之间权衡。

**通过自蒸馏抵抗遗忘（Forgetting resistance through self-distillation）**。SDFT（Shenfeld et al., 2026）把 on-policy 自蒸馏定位为一种抵抗遗忘的机制。它利用模型自身的 in-context learning 能力，从自身构造出一个以演示为条件的教师，然后让学生在自己的 rollout 上通过 reverse KL 训练，这在数学上等价于带隐式 KL 正则的 trust-region RL。这一重新表述把从演示中进行的持续学习转化为一个 on-policy 蒸馏问题：trust-region 约束防止对已获得能力的灾难性遗忘，同时容纳新能力。该方法是完全 on-policy 的（每步都用新鲜 rollout），因此位于 staleness 谱系最“新鲜”的一端。

**架构式自蒸馏（Architectural self-distillation）**。MTP-SD（Kirchenbauer et al., 2026）针对的是架构层面的改进而非行为层面的改进：它以模型自身的一个冻结副本作为教师，对学生生成的 token 片段打分，从而把一个预训练的自回归模型转换为一个独立的多 token 预测器。这在推理时无需任何辅助模块的情况下带来 > 3× 的解码加速，准确率通常下降 3–7%（依模型而定），从而把自蒸馏可优化的范围从输出分布拓展到生成架构。

<!-- p. 36 -->

AR-to-Diffusion（Su et al., 2026）把同一原则扩展到不同生成范式之间：用一个自回归 LM 作为冻结的自教师，通过 on-policy token 级蒸馏来训练一个 diffusion LM。学生生成 diffusion 去噪轨迹，冻结的 AR 教师给出 token 级概率以引导去噪分布。通过把 AR LM 转换为 diffusion LM，该方法以比标准 diffusion-LM 预训练高 15×–7000× 的训练 token 效率达到其报告的质量，从而确立 on-policy 自蒸馏能够跨不同生成架构迁移知识。

**自蒸馏组件的系统性统一（Systematic unification of self-distillation components）**。上述方法各自考察单一机制（in-context 演示教师、多 token 片段打分），但这些设计选择之间的交互效应仍不清楚。UniSD（Jin et al., 2026c）通过提出一个统一框架来填补这一空白，该框架沿三条互补的轴系统性地分解 LLM 自蒸馏：**监督可靠性**（用多教师一致性识别可信的自生成信号，用 token 级对比学习区分正确选项与似是而非的错误选项）、**表示对齐**（通过特征匹配迁移输出分布之外的内部结构）以及**训练稳定性**（用 EMA 教师做时间平滑，用 divergence clipping 防止罕见的高 divergence token 主导更新）。在六个 benchmark、六个模型（跨越三个族：Qwen2.5、Llama-3.1、Gemma-3）上评估后，UniSD 发现 on-policy 轨迹结合多教师一致性与 EMA 稳定化，构成了有效自蒸馏的核心要素。集成变体 UniSD$^*$ 在 Qwen2.5-7B 上相比基座模型提升 +5.4 分，相比最强基线（GKD）提升 +2.8，且不依赖任何外部教师。UniSD 的三轴分解为未来的自蒸馏方法提供了一套设计词汇，凸显出哪些设计选择最为重要。

**自监督推理信号（Self-supervised reasoning signals）**。三种以推理为焦点的方法从模型自身能够打分的对比中提取学习信号。Self-Supervised OPD（Tan & Hong, 2026b）把 GRPO rollout 的 per-prompt 组重新用作自监督信号，在同一组内对比正确与错误的 rollout，找出区分二者的 per-token 位置，并仅在这些对比位置上施加 token 级 KL，从而复用 GRPO 已有的 rollout、不增加额外推理成本。ROSD（Zhao et al., 2026d）把 OPSD 中以解为条件的自教师替换为一个 reflector：给定一条错误 rollout，该 reflector 提取出一个解释该错误的纠正性想法，并通过 error quote 定位出第一个出错的片段。纠正性想法使自教师的分布偏向针对该错误的监督，而 error quote 把蒸馏限制在从错误处开始的后续序列上，从而保留有效的推理前缀。Variational Posterior Guidance（Chen et al., 2026e）加入了一个效率感知的变分后验，用长度感知的效用为每条候选 rollout 打分，使自蒸馏目标偏向更短的正确答案轨迹而非冗长的轨迹。这三种方法表明，自蒸馏所需的不对称性可以来自组内对比（Self-Supervised OPD）、反思式错误定位（ROSD）或后验加权效用（VPG），而无需任何辅助模型。

**多轮与智能体式自蒸馏（Multi-turn and agentic self-distillation）**。第二个集群把纯自蒸馏扩展到学生展开长轨迹、监督目标事后构造的设定。HINT-SD（Yeo et al., 2026）与 SD-Search（Ma et al., 2026）都采用 hindsight relabeling，但粒度不同。HINT-SD 从长时程智能体 rollout 中只挑出与失败相关的 turn，针对一个 hindsight 重标注过的成功续写做 token 级自蒸馏；而 SD-Search 把监督定位到搜索增强推理轨迹内部的 step 级搜索查询上，并把所得的 per-step 信号与外层 GRPO 结合。Search-E1（Liang et al., 2026b）交替进行外层 GRPO 与在 GRPO 改进后 checkpoint 上的离线自蒸馏，把每一轮蒸馏都视为一次自我演化步骤，用于在下一轮 RL 之前固化策略改进。MAIGO（Zheng et al., 2026b）针对纯多轮设定中出现的 lost-in-conversation 差距，构造出一个历史清理过的特权教师（即同一模型、以该任务的单轮改写为条件），并在原始含噪多轮对话上监督学生 rollout，在 Qwen2.5-7B-Instruct 上把 sharded-to-full 准确率比从 66.5% 提升到 84.1%，同时把全上下文准确率的差距保持在 2.3 分之内。Canonical-Context OPD（Lin et al., 2026b）从另一个角度追求同样的多轮对齐。学生从原始 sharded 
 对话历史，而一个冻结的同骨干教师则在规范的全上下文历史下回答同一提示，并通过一个答案掩码的 reverse KL，在相同前缀重标记（same-prefix relabeling）下把学生的行为与规范上下文教师对齐。COMAP（Liu et al., 2026i）把智能体与一个文本世界模型耦合到一个协同演化循环中：对世界模型施加带 EMA 教师的 on-policy 自蒸馏，而智能体则使用带动作门（action gate）的未来感知反思，并以一个世界状态门课程与基于后缀的精炼初始化来保持循环稳定。在这六种方法中，共同的模式是：监督目标从 rollout 本身重建而来（事后重标记、历史清理后的重新表述、规范上下文、世界模型预测的未来），从而把多轮结构转变为非对称性的来源，而不是一个需要被容忍的复杂因素。

**领域特定的自蒸馏。** 另外三种方法把纯自蒸馏专门化到非对称性源自某种领域原生切分的领域中。Vision-OPD（Yuan et al., 2026）为细粒度 MLLM 理解引入了一种区域到全局（regional-to-global）的切分，其中模型的区域条件变体充当全局条件学生的自教师，而后者在推理时缺乏区域标注，并在学生 rollout 上施加 token 级 reverse KL。TOD Proactivity（Zhang et al., 2026a）把 privileged 的用户关切标注（user-concern annotations）当作任务型对话中的非对称信号，训练一个以标注的用户关切为条件的非对称自教师，并蒸馏到一个只能观察到原始话语的学生上，从而使主动信息收集行为被内化，而无需在推理时访问该标注。It Takes Two（Park et al., 2026）为上下文完整性（contextual-integrity）隐私对齐追求互补式自蒸馏，其中在互补上下文切分下的两条并行 rollout 提供相互监督，把隐私决策锚定在上下文扰动之间，而不是针对任何单一的静态教师。这三种方法把纯自蒸馏的 rollout 多样性原则分别适配到多模态 grounding（Vision-OPD）、对话策略（TOD Proactivity）与隐私对齐（It Takes Two），证明只要识别出与领域相称的非对称性，这一架构模式就能跨模态迁移。

**跨领域协同蒸馏。** OPCoD（Byeon et al., 2026）把相互监督的思想从单个模型的互补 rollout 扩展到两个在不同领域之间互相辅导的专家模型。每个模型运行 on-policy 自蒸馏，其自教师以自身经验证正确的 rollout 与来自同伴的自然语言反馈消息为条件，因此两个自蒸馏循环相互耦合并在训练中协同演化。一个基于认知（cognizance-based）的门控仅在同伴在当前领域更有能力时才交换反馈，从而缓解在混合不同领域数据时通常出现的负迁移；一个反馈锚定项则把消息落到问题本身上，以防止它破坏本已正确的 rollout。在物理–化学、化学–材料等领域配对上的科学问答任务中，OPCoD 在 Qwen3-8B 上达到相互 Pareto 改进，每个模型都在不丧失原有专长的情况下获得域外能力。CoPD（Gu et al., 2026）在单一能力内部耦合两个模型，It Takes Two 切分单个模型的上下文，而 OPCoD 则针对多领域情形，此时的非对称性来自同伴不同的专长。

这些方法共同表明，纯自蒸馏无需外部监督即可在编码、推理与指令跟随任务上带来有意义的增益。然而，这些增益受限于模型预训练分布中已经存在的信息。当仅靠温度多样性不足以显现高质量完成结果时，跨越这一边界要么需要把模型连接到外部验证信号，要么需要在内部生成 privileged information，两者都构成了下文外部反馈机制的动机。

#### 5.2.3 外部反馈（External Feedback）

上述纯自蒸馏方法从模型自身的分布生成训练信号，但该信号受限于模型参数中已编码的信息。在无法访问真值或环境验证的情况下，模型无法区分"自信但错误"的输出与真正正确的输出，这一局限导致了 §7.2 所分析的饱和问题。本节的方法保留自蒸馏架构（没有单独的教师模型），同时注入外部信号——无论来自验证器、环境、多智能体协同演化，还是部分解 oracle——这些信号锚定训练循环，防止其坍缩到自我强化的错误之上。

<!-- p. 38 -->

**从二值奖励到稠密监督。** 一条研究脉络通过不同形式的反馈，把 RLVR 的稀疏结果奖励转化为稠密的 token 级自监督。SDPO（H ¨ubotter et al., 2026）是早期实例，它使用**结构化文本反馈**（运行时错误、失败的单元测试以及 LLM 评判评估）把成功或失败归因到具体的推理步骤，并以反馈条件化的模型充当 token 级 KL 蒸馏的自教师。RL from Text Feedback（RLTF）（Song et al., 2026）同期使用来自自动评判器的自由形式自然语言批评，训练模型的单轮策略去匹配其自身反馈条件化的第二轮生成。较新的实例 SD-ZERO（He et al., 2026）采取了不同角度，通过双角色 Generator-Reviser 架构把同样的二值验证器信号转化为稠密自监督：Reviser 以 Generator 的输出及其二值正确性信号为条件，产生一个改进版本，随后 Generator 从 Reviser 的 token 级分布中蒸馏。在 Qwen3-4B-Instruct 上，SD-ZERO 在八个基准上录得 68.3% 的平均准确率（avg@8），优于 GRPO（64.2%），表明在把稀疏奖励转化为学习信号方面，自我修订可能比标准 RL 更有效。

从 SD-ZERO 的二值信号，到 SDPO 的结构化错误，再到 RLTF 的自由形式批评，一条清晰的轨迹浮现出来。随着外部反馈变得更丰富，自蒸馏目标也变得更精准，使模型能够纠正具体的失败模式，而不是全局地调整其策略。RESD（Zhang et al., 2026n）把这条轨迹推得更远：它把原始的失败反馈转化为纠正性监督的主动来源，而不是把它当作被动的条件信号。当学生的 on-policy rollout 失败时，RESD 生成回顾性反思来诊断局部错误，并在训练步之间整理出一份持久的、可复用教训的全局 playbook。被丰富的上下文使自教师即便在没有任何成功示范的情况下也能提供可操作的 token 级监督，直接应对了 GRPO 与标准 OPSD 因奖励方差坍缩为零而失去信号的稀有成功情形。在持续 agentic 学习任务上，RESD 的样本效率比 GRPO 高出 8×（Amazon/UCSD），证明对失败的结构化反思可以替代大多数自蒸馏方法隐含所需的成功示范。Sample-Routed Policy Optimization（SRPO）（Li et al., 2026c）表明这些方法可以通过路由来组合。正确样本走 GRPO 的强化路径（奖励加权似然最大化），而失败样本走 SDPO 的定向 logit 级纠正。正确与错误样本承载着不同类型的学习信号，把它们同等对待会浪费各自携带的信息。在科学与工具使用任务上，SRPO 在 Qwen3-8B 上把五基准平均值相对 GRPO 提高 3.4%，相对单独使用 SDPO 提高 6.3%。一种有用的外部反馈策略，是把每个样本路由到对其最有信息量的目标上。

**分解幅度与方向。** Self-Distilled RLVR（RLSD）（Yang et al., 2026a）把梯度更新分解为两个在功能上不同的组成部分。自蒸馏决定逐 token 策略更新的**幅度**（每个 token 的概率该改变多少），而来自验证器的环境 RLVR 反馈则锚定**方向**（哪些 token 的概率应上升、哪些应下降）。仅用 200 个训练步，RLSD 就在 Qwen3-VL-8B-Instruct 上超过了训练 400 步的 GRPO，展现出双信号结构带来的 2× 样本效率。这一分解澄清了为什么外部 grounding 与自蒸馏是相加的而非竞争的。自蒸馏成分利用模型内部的分布几何来校准更新幅度，而外部验证器则确保这些更新指向正确的方向。

RLRT（Kim et al., 2026a）完全反转了这一自蒸馏范式。标准自蒸馏引导学生走向教师的预测，而 RLRT 观察到：在**成功**的 rollout 上（由外部验证器判定），那些教师本不会预测出学生所选 token 的位置，反映的是学生自我驱动的推理而非对教师的模仿。通过反向解读自蒸馏信号，并以 GRPO 增强来强化这些"叛逆的" token，RLRT 把探索框定为并非均匀多样性，而是植根于学生自身成功的有价值偏离。在 base、 
 instruction-tuned 与 thinking-tuned 的 Qwen3 checkpoint 上，RLRT 优于标准自蒸馏与基于探索的基线，把信息不对称确立为外部 grounded 自我改进的一条设计轴。

**面向多轮智能体的门控自蒸馏。** RLSD 把梯度分解为幅度（自蒸馏）与方向（RL），SDAR（Lu et al., 2026）则把这一分解带入多轮 agentic 场景，并识别出该情形特有的一个失败模式：多轮的不稳定性叠加会动摇自教师的监督，而技能条件化的 privileged 指导会产生负教师信号，这些信号源自不完美的技能检索，而非学生真正糟糕的行为。SDAR 把 on-policy 自蒸馏视为一个门控辅助目标，同时让 RL 保持为主要优化骨干，并把 detach 后的 token 级教师–学生对数比信号通过一个 sigmoid 门映射，从而在教师认可的 positive-gap token 上加强蒸馏，并柔和地衰减教师的负向拒绝。在 Qwen2.5 与 Qwen3 系列上，于 ALFWorld、WebShop 与 SearchQA 上，SDAR 相对 GRPO 分别提升 +9.4%、+10.2% 与 +7.0%。当自蒸馏从单轮迁移到多轮场景时，对正负教师信号的非对称处理似乎是一个关键因素。

更广泛地说，本节的外部反馈方法都共享一个把它们与纯自蒸馏区分开来的性质：编译器输出、单元测试结果与符号验证器提供无歧义的纠正信号，打破了自我强化的回音室，因为即便一个模型对自己的错误推演高度自信，也会从验证器那里收到 R= 0。

**通过逐点 MI 分解实现输入特定的 credit。** CREDIT（Shen et al., 2026b）处理自蒸馏奖励中一个更微妙的病态。在反馈条件化的后验兼容性解释下，标准的 OPSD token 奖励化约为一个贝叶斯滤波增量，其轨迹求和等于在给定输入下响应与反馈之间的逐点互信息（pointwise mutual information, pMI）。这一 pMI 可能被输入通用的捷径（在任何正向反馈下都通用地容易出现的响应）所抬高，而非来自输入特定的推理。CREDIT 通过一个批次对比基线分离出输入特定的成分，该基线惩罚在无关输入下仍然容易出现的响应。在序列层面，这起到一个针对对比式 pMI 目标的教师侧替代量（surrogate）的作用。在两个模型系列的编码、科学推理与工具使用基准上，CREDIT 以可忽略的额外算力交付了很强的总体性能。因此，抑制通用捷径是自蒸馏奖励一个广泛有用的修正。

**结果引导的 logit steering。** OGLS-SD（Yang et al., 2026f）识别出 OPSD 中一个此前自适应方法所忽视的教师–学生错配：教师响应中的反思引入偏差（reflection-induced bias）。自我反思后的教师分布可能被响应模板与解条件化的人工痕迹所偏移，产生失校准的 token 级监督。OGLS-SD 通过使用可验证的结果奖励来对比成功与失败的 on-policy 轨迹，利用结果信号把教师 logits 引向校准的目标。通过把结果层面的正确性与经由 logit steering 得到的稠密 token 级指导相结合，OGLS-SD 稳定了自蒸馏，并在多种基准上相对标准 OPSD 改善了推理性能。因此，结果信息可以作为自蒸馏目标的一个独立校准轴。

**多智能体协同演化与部分解 oracle。** 上述外部信号都来自环境侧的验证器。当模型通过多智能体交互或部分解披露（partial-solution disclosure）生成自己的验证信号时，就会出现一个独立的外部 grounding 来源。π-Play（Zhang et al., 2026l）把 privileged information 与 self-play 统一到一个多智能体自我演化框架中。三个协同演化的智能体（出题者 $\pi^E_\phi$、以问题构造路径为条件的教师 $\pi^T_\psi$、以及不具备该路径的学生 $\pi^S_\theta$）通过交替更新被联合优化。self-play 在任务生成过程中产生一个副产品：出题者智能体构造出问题构造路径（question construction path, QCP），从而以零额外成本提供高质量的 PI。随着学生变强，出题者生成更难的任务，而教师的行为则通过一个 KL 惩罚被柔和地约束以跟踪学生，形成一个无需外部数据的自适应课程。在 Qwen3（4B、8B）上，无数据的 π-Play 超过了完全监督的搜索智能体，如 Search-R1， 
并在多跳 benchmark 上取得显著增益——在这些任务中，教师的 token 级信用分配具有相当大的影响——其演化效率也明显高于常规 self-play（Dr.Zero）。

PAINT（Tan & Hong, 2026a）处在一个独特的位置：介于纯自蒸馏（无外部信息）与完全 PI（揭示完整答案）之间。它不像 OPSD 那样以完整的 ground-truth 解为条件，而是只暴露一个**部分解**，其暴露程度由学生 rollout 与参考解之间的重叠度自适应控制。它计算一个 recall 式的重叠分数，依据是参考锚点（boxed 答案、公式、关键数字）在学生 rollout 中出现的比例：当学生已经接近时，就隐藏更多参考（鼓励更广的泛化，因为 rollout 已抓住本质结构）；当偏离较远时，就揭示更多参考（在学生已经跑偏时提供更强的纠错引导）。这种随重叠度自适应的部分解遮蔽，在自蒸馏机制内部形成了一种天然的课程。PAINT 还采用 energy-space 插值，仅在教师—学生熵失配超过阈值的位置施加蒸馏损失，从而产生一个把梯度信号集中于最具信息量位置的稀疏损失。在竞赛级数学上（AIME 2024/2025 与 HMMT 2025 平均值），PAINT 相对 OPSD 基线录得 +2.1、相对 GRPO 录得 +2.9：对特权信息的部分揭示可以同时优于完整揭示与不揭示，这条信息揭示曲线的理论刻画仍待解决。

**共同演化式蒸馏（Co-evolutionary distillation）**。另一条线索完全放弃固定教师假设，联合训练教师与学生，使二者在蒸馏过程中都得到提升。CoPD（Gu et al., 2026）表明，顺序流水线（先训练专家、再蒸馏）可以被共同演化式训练取代：让 RLVR 与双向 OPD 交替进行，维持行为上的邻近性，从而使知识能够被持续吸收，而不是事后迁移。Variational Policy Distillation (VPD)（Li et al., 2026i）把这种共同演化形式化为一个变分期望最大化（Variational Expectation-Maximization）问题：E 步通过自适应的 trust-region 更新在轨迹结果上改进教师，把文本反馈转化为动态改进的目标分布；M 步则让学生在自己的 rollout 上内化这一分布指导。通过持续提升教师从诊断性批评中提取可执行信号的能力，VPD 缓解了被动的（固定教师）自蒸馏随学生变强而遇到的平台期。在科学推理与代码生成任务上，VPD 同时优于标准 RLVR 与现有的自蒸馏基线；而在数学推理与冷启动设定上的压力测试，则揭示出反馈驱动的自蒸馏相对于纯环境驱动 RL 的实际边界。

**饱和分析（Saturation analysis）**。Kim et al. (2026b) 把数学推理中的自蒸馏退化追溯到对 **epistemic verbalization**——即模型在推理过程中对不确定性的表达——的压制。当自蒸馏缩短推理轨迹时，它会不成比例地删除那些充当困难子问题隐式“注意标志”的含糊措辞与不确定性标记。结果就是推理更短但校准更差：模型自信地走上错误的步骤。这与第 7 节中的教师不确定性挑战相关联——在那里，过度自信的学生会产生不可靠的自蒸馏目标——而饱和问题往往需要外部接地或 π-Play 的内部生成 PI 机制才能克服。

**用于安全对齐的冻结副本一致性（Frozen-copy consistency for safety alignment）**。On-Policy Consistency Training（Han et al., 2026a）把外部反馈模板适配到安全对齐上，而无需验证器或奖励模型。更新前策略的一个冻结副本充当自教师，学生在对比式 prompt 对下 rollout（一个规范 prompt，一个被谄媚或 jailbreak 触发词扰动的 prompt），而冻结副本教师在规范 prompt 上的 per-token reverse KL 与学生在该扰动 prompt 上的 per-token reverse KL 相对齐，使学生对扰动 prompt 的回应与教师对规范 prompt 的回应一致。由于教师与学生共享骨干、仅条件 prompt 不同，这一构造提供了稠密的 per-token 一致性监督，同时把安全更新局限在扰动分支上。在谄媚、jailbreak 防御与安全意识 benchmark 上，该方法在能力退化极小的前提下提升了安全分数。Consistency-against

<!-- p. 41 -->

**表 9（Table 9）：OPD 方法：训练效率与稳定化（Table 9: OPD methods: Training efficiency and stabilization）。**

| 方法                                    | 年份 | 类别   | 散度/目标               | 信号              | 粒度         | 关键创新                              |
| --------------------------------------- | ---- | ------ | ----------------------- | ----------------- | ------------ | ------------------------------------- |
| TIP (Xu et al., 2026c)                  | 2026 | 动力学 | Soft-OR token 加权      | 白盒              | token 级     | 熵×散度四象限打分                    |
| SCOPE (Zheng et al., 2026a)             | 2026 | 动力学 | Rollout 路由            | 白盒              | token 级     | 在 flawed prefix 上重定向             |
| SelecTKD (Huang et al., 2025)           | 2025 | 动力学 | 置信度过滤              | 白盒              | token 级     | Top-1 概率阈值                        |
| AdaSwitch (Peng et al., 2025)           | 2025 | 动力学 | 探索/引导切换           | 白盒              | token 级     | 二值模式切换                          |
| PACED (Xu et al., 2026b)                | 2026 | 动力学 | Beta-kernel 课程        | 白盒              | 样本级       | 前沿难度采样                          |
| FOPD (Zhang et al., 2026b)              | 2026 | 动力学 | 前缀截断                | 白盒              | 混合         | 聚焦早期 token                        |
| Lightning-OPD (Wu et al., 2026a)        | 2026 | 动力学 | 离线缓存                | 白盒              | token 级     | 成本降低 4.0×                        |
| SKD (Xu et al., 2025c)                  | 2025 | 动力学 | Draft-verify            | 白盒              | 块级         | 摊销教师成本                          |
| NPD (Rang et al., 2026)                 | 2026 | 动力学 | 异步生成-训练解耦       | 白盒              | token 级     | 通过 ∆-IFD 过滤实现 8.1× 加速       |
| Prune-OPD (Yang et al., 2026g)          | 2026 | 动力学 | Top-k 重叠漂移检测      | 白盒              | token 级     | 动态轨迹剪枝，时间减少 37–68%        |
| R-OPD (Fu et al., 2026b)                | 2026 | 动力学 | Top-p rollout + masking | 白盒              | token 级     | 针对经验失效模式的修复                |
| SOD (Zhong et al., 2026a)               | 2026 | 动力学 | 步级散度重加权          | 白盒              | 步级         | tool call 级联衰减                    |
| Stable-OPD (Luo et al., 2026a)          | 2026 | 动力学 | RKL + 参考散度          | 白盒              | token 级     | 为稳定性做 rollout 混合               |
| TCOD (Wang et al., 2026d)               | 2026 | 动力学 | 时间维度课程            | 白盒              | 轨迹级       | 多轮 KL 稳定性                        |
| Uni-OPD (Hou et al., 2026)              | 2026 | 动力学 | 双视角数据平衡          | 白盒（多教师）    | token 级     | 统一 LLM+MLLM OPD                     |
| CaOPD (Zhang et al., 2026d)             | 2026 | 动力学 | 校准感知 OPD            | 白盒              | token 级     | 诊断错校准的 scaling law              |
| EGRSD (Ke et al., 2026)                 | 2026 | 动力学 | 熵门控自蒸馏            | 自蒸馏（PI）      | token 级     | 教师熵置信门 + 因果前瞻               |
| MOPD (Yu et al., 2026b)                 | 2026 | 动力学 | 同伴条件多 rollout      | 白盒              | 样本级       | 成功/失败 rollout 组加权              |
| GEAR (Li et al., 2026g)                 | 2026 | 动力学 | 自适应片段重加权        | 自蒸馏（PI）      | 段级         | 粒度自适应的 advantage 边界           |
| EffOPD (Cai et al., 2026b)              | 2026 | 动力学 | 参数动力学前瞻          | 白盒              | token 级     | 通过更新外推实现 3× 加速             |
| Visual-Adv. OPD (Liu et al., 2026c)     | 2026 | 动力学 | 视觉接地重加权          | 白盒              | token 级     | per-token 视觉 advantage 梯度         |
| Filter-Then-Reweight (Li et al., 2026m) | 2026 | 动力学 | 轨迹过滤 + token 重加权 | 白盒              | token/轨迹级 | 双粒度 OPD                            |
| Less-is-More (Ziheng et al., 2026)      | 2026 | 动力学 | 早停 rollout            | 白盒              | token 级     | off-policy teacher-decay 截断         |
| DeltaPrompts (Jung et al., 2026)        | 2026 | 动力学 | Prompt 散度课程         | 白盒（VLM）       | prompt 级    | zero-delta-trap 合成                  |
| f-OPD (Chen et al., 2026c)              | 2026 | 动力学 | 新鲜度感知控制          | 白盒              | 轨迹级       | Async-OPD 滞后界定                    |
| TR-Behav. Blend (Plyusov et al., 2026)  | 2026 | 动力学 | Trust-region 预热       | 白盒              | token 级     | 闭式几何混合                          |
| Adaptive Refresh (Guo et al., 2026)     | 2026 | 动力学 | 三重门控教师刷新        | 自蒸馏（PI）      | 轨迹级       | 隔离/奖励/长度尾部门控                |
| CEI (Chen et al., 2026a)                | 2026 | 动力学 | 迭代经验内化            | 自蒸馏            | 步级         | 多轮迭代智能体自我演化                |
| POPD/TOPD (Zhang et al., 2026k)         | 2026 | 动力学 | 视界受控 rollout        | 白盒              | token 级     | 渐进式/截断视界                       |
| SafeSteer (Li et al., 2026e)            | 2026 | 动力学 | 安全 token 局部化 RKL   | 自蒸馏（steered） | token 级     | 激活引导的安全教师                    |
| TRD (Jiang et al., 2026a)               | 2026 | 动力学 | 轨迹精化蒸馏            | 白盒              | 序列级       | 在 prefix-failure 处做教师修正        |
| KAT (Xin et al., 2026)                  | 2026 | 动力学 | KL 一致陷阱截断         | 白盒              | token 级     | 在 KL 一致陷阱区域做在线 rollout 截断 |

perturbation 因此可以替代外部安全奖励，只要这种不对称性能够仅凭 prompt 编码。

**符号一致性门控与分阶段教师采样（Sign-consistency gating with phased teacher sampling）**。SG-OPD（Xu et al., 2026a）引入一种验证器引导的门控机制，按学生 rollout 梯度与另行计算的验证器信号之间的符号一致性来过滤 token 级蒸馏信号。一个分阶段的教师采样时间表在训练早期用高质量教师回答预热，随后再过渡到 on-policy rollout，从而平滑弱学生从第一次迭代起就被迫匹配强教师时所出现的分布偏移。在竞赛级数学 benchmark 上（AIME 2024/2025 与 HMMT 2025），SG-OPD 在 per-sample 层面（avg@32）相对基础 GRPO 设定提升 +1.98，在 per-question 层面（pass@32）提升 +7.50；作者把增益归因于丢弃了那些本会强化自信但错误推理步骤的虚假 token 级梯度。

跨信号来源——从白盒 logit 到黑盒 API 再到自蒸馏——信号密度与自主性之间的根本权衡主导着整个设计空间。白盒方法提供最稠密的监督（每个 token 处 $|V|$ 维分布），但要求同组织部署以及教师与学生的共置。黑盒方法牺牲分布信息，却能跨组织边界运作，近期方法（GAD、OVD）恢复了有效的代理信号。自蒸馏消除了外部依赖，但若没有外部验证或内部生成的 PI 加以增强，就会受限于模型既有能力的边界。这一连续统也反映了该领域的成熟度梯度。白盒这一端相对成熟（GKD、DistiLLM、DSKD），而自蒸馏这一端对于复杂推理仍缺乏有原则的收敛保证。近期的活跃工作集中于各类别之间的边界：LUFFY 等方法模糊了 off-policy 演示与 on-policy RL 之间的界线，π-Play 则把自蒸馏与外部接地的共同演化连接起来。

## 6 训练效率与稳定化（Training Efficiency and Stabilization）

在给定目标函数（第 4 节）与信号来源（第 5 节）之后，最终的工程决策就是**如何**稳定并加速 on-policy 训练循环。on-policy 生成引入了标准监督微调中基本不存在的优化挑战，包括非平稳的数据分布（学生的策略在训练过程中变化，使较早的 rollout 变得 stale）、困难 prompt 上梯度信噪比（SNR）的坍缩（此时所有 rollout 都失败，产生不了有用的梯度），以及学生自回归 rollout 加教师打分所带来的计算开销（比 off-policy 训练昂贵数倍）。

<!-- p. 42 -->

这些挑战并非彼此独立。困难 prompt 上的 SNR 坍缩浪费算力，促使人们设计避开此类 prompt 的课程策略。非平稳性使缓存的教师信号失效，促使人们设计高效的在线打分。而自回归生成的计算开销造成系统级瓶颈，促使人们设计架构层面的解法。本节的方法通过三种互补机制应对这些挑战。

### 6.1 token 与样本加权（Token and Sample Weighting）

在 on-policy 设定下，并非所有教师监督都同样可靠。核心问题是 **flawed prefix trap**：学生在序列早期生成一个错误 token，会使后续所有教师预测都以这个分布外前缀为条件。由于教师从未在这样的前缀上训练过，其条件分布 $p_T(\cdot|x, \hat{y}^{\text{bad}}_{<t})$ 会变得嘈杂而不可靠。朴素地匹配这一被污染的信号，只会传播错误而非纠正错误。

**教师可靠性过滤（Teacher reliability filtering）**。TIP（Xu et al., 2026c）（Token Importance Profiling）提供了一个有原则的 token 加权框架：它沿两条轴组织重要性——学生熵 $h_t$ 与教师—学生散度 $\delta_t$。两轴交叉得到四个在学习角色上性质不同的象限。高熵、高散度的 token（Q1）承载主要的纠错信号；高熵、低散度的 token（Q2）表示在不确定性上的一致；低熵、高散度的 token（Q3）代表**过度自信的错误**，即学生很确定但错了；低熵、低散度的 token（Q4）则是真正已掌握的。一项理论结果确立：Q3 token 在结构上对任何仅依赖熵的加权方案都不可见，因为任何满足 $f(0)=0$ 的非降熵函数都无法把它们与 Q4 区分开。TIP 的无参数 token 选择在保持 Q1/Q2 敏感性的同时恢复了 Q3 的覆盖。在 DeepPlanning 这类长时程规划 benchmark 上，仅用 Q3 token（不足全部 token 的 20%）训练即可超过全 token OPD。

SCOPE（Zheng et al., 2026a）把这一分析从 token 提升到 **rollout** 层面。其双路径架构按正确性路由 rollout。错误轨迹接受以教师 perplexity 加权的 KL 蒸馏（对 flawed prefix 上不可靠的引导降权），而正确轨迹接受以学生 perplexity 加权的 MLE（把强化集中在能力边界处）。这对抗了 diversity collapse——即 Pass@1 提升却以 Pass@k 为代价的反直觉现象——在六个推理 benchmark 上相对有竞争力的基线取得 7.3% 的相对 Pass@32 增益。

EGRSD（Ke et al., 2026）通过引入一个熵引导的置信门把 token 级加权与自蒸馏设定统一起来：该门对高熵的教师位置降权，同时对每个 token 权重维持一个非零下界。其核心观察是：教师熵在一条 chain-of-thought 序列上变化很大，而均匀加权所有位置会迫使学生在真正不确定的推理转折处去匹配嘈杂的教师预测。EGRSD 进一步提出一个因果前瞻变体（CL-EGRSD），用于把持续的高熵片段（教师确实不确定、监督不可靠）与那些后续上下文迅速收敛到低熵的瞬时高熵位置（此处的短暂不确定反映的是分支点，而非教师的无知）区分开。这种时间维度上的区分，以熵在推理轨迹上如何演化的动态视角，补充了 TIP 的静态象限分析。在 thinking 模式下的 Qwen3-4B 与 Qwen3-8B 上，EGRSD 与 CL-EGRSD 在所比较的可训练方法中推进了准确率—长度的前沿；熵感知门控在不牺牲覆盖的前提下提升了自蒸馏效率。

SelecTKD（Huang et al., 2025）在 token 级应用一种 propose-and-verify 机制：学生提出 token，教师通过贪心 Top-k 或非贪心 Spec-k 匹配加以验证。被接受的 token 接受完整损失，被拒绝的 token 则被遮蔽或降权，从而诱导出一种由 Token Acceptance Rate (TAR) 量化的隐式课程。教师—学生一致性失败的位置，要么说明存在真正的歧义，要么说明存在分布散度；无论哪种情况，未过滤的梯度都不携带可靠信息。 
AdaSwitch（Peng et al., 2025）提出一种在纯探索与引导式蒸馏之间动态切换的机制。与只是简单衰减不可靠信号的静态过滤不同，AdaSwitch 维护一个近期 token 级散度（学生与教师 logit 之间的 KL 或 JSD）的滑动窗口，并在长度为 L 的窗口上计算运行平均 $\bar{d}_{i-1} = \frac{1}{L}\sum_{j=i-L}^{i-1} d_j$。当当前 token 的散度超过一个随上下文自适应的阈值 $\tau_i = K\cdot\bar{d}_{i-1}$（其中 K 是灵敏度乘子）时，该方法就从 on-policy 探索切换到 off-policy 教师引导。这种单次切换的设计既保持了语义连贯（避免了先前混合方法那种突兀的 token 级交替），又恰好在学生的生成漂移进高散度区域时提供高质量监督。阈值自适应在这里很重要，因为固定阈值往往要么过于激进（在微小散度无害的简单前缀上就切换），要么过于宽松（在学生进入真正有问题的状态时未能干预）。通过以近期散度历史为条件，AdaSwitch 把其干预阈值校准到局部序列难度上。

这四种方法覆盖了从细粒度到粗粒度干预的一个设计谱系。TIP 对每个 token 施加连续加权，SCOPE 把整条 rollout 路由到不同的损失路径，SelecTKD 对每个 token 做二值的接受/拒绝决策，而 AdaSwitch 则承诺一次轨迹级模式切换。这一排序揭示出信号保真度（更细的粒度保留更多信息）与优化稳定性（更粗的决策降低梯度方差）之间的底层权衡。这一权衡不只是理论上的。像 TIP 这样的细粒度方法需要可靠的 token 级教师信号，而恰恰在 flawed prefix 问题严重时，这种信号会退化。粗粒度方法牺牲 per-token 最优性，但对局部信号污染具有鲁棒性，因为单个坏 token 无法污染整条轨迹的梯度。在这一谱系上的选择与主要失效模式相关。当教师可靠性在一条序列内部变化剧烈时（例如长推理链中有些步骤容易、另一些需要关键洞见），token 级方法往往表现突出；而当首要担忧是整段生成过程中的前缀质量漂移时（例如早期错误会改变整场对话主题的多轮对话），轨迹级方法则更受青睐。

**样本级加权（Sample-level weighting）**。除 token 级过滤之外，若干方法根据信息量对整条样本加权。flawed prefix trap 在**困难** prompt 上表现得最为严重，此时学生的通过率接近零。在这些 prompt 上，几乎所有 rollout 都包含早期的灾难性错误，所得教师信号被噪声主导。均匀采样 prompt 的方法会把大部分梯度预算浪费在无信息的困难 prompt 或平凡正确的简单 prompt 上，而真正有产出的学习集中在处于学生能力边界上的 prompt 上。

**失效模式分析（Failure-mode analysis）**。Fu et al. (2026b) 对为什么 sampled-token OPD 在长时程设定下很脆弱做了系统的经验分析。他们识别出三种失效模式：(1) token 级监督不均衡，(2) 在学生生成的前缀上教师引导不可靠，(3) tokenizer 或 special token 不匹配。他们提出的修复方案是 teacher top-K local support matching，即一个截断的 reverse-KL 目标，配合 top-p rollout 采样与 special token 遮蔽；它在推理与智能体设定下产生更稳定的优化，相对标准 sampled-token OPD 基线取得 19.8% 的性能提升。

**面向工具集成推理的步级重加权（Step-level reweighting for tool-integrated reasoning）**。上述加权机制都在 token 或轨迹粒度上运作。SOD（Zhong et al., 2026a）识别出工具集成推理（TIR）智能体特有的一种失效模式，由此引出居中的**步级**粒度。在 TIR 设定中，错误的工具调用会把被污染的观测注入推理上下文，使学生—教师散度在随后的步骤上急剧加速，而不是像纯文本推理那样逐渐漂移。标准 token 级 OPD 无法检测这种不连续的状态跃迁，因为它对步骤之间的边界没有可见性。SOD 依据累积的步级散度，对每个推理步骤的蒸馏强度做自适应重加权，在高散度区域衰减可能具有误导性的教师信号，同时保留教师—学生仍保持对齐处的稠密引导。以 Qwen3-4B 为教师、Qwen3-0.6B/1.7B 为学生，SOD 在数学、科学与代码 benchmark 上比第二名基线高出最多 20.86%，其中 0.6B 学生在 AIME 2025 上达到 26.13%（average@32）。

<!-- p. 44 -->

步级粒度填补了 TIP 的 per-token 加权（对检测 tool call 级联来说过细）与 SCOPE 的轨迹级路由（对在部分受损轨迹中保留对齐良好步骤上的有用监督来说过粗）之间的空隙。

**同伴条件多 rollout 加权（Peer-conditioned multi-rollout weighting）**。MOPD（Yu et al., 2026b）指出上述加权方法共有的一个局限：它们把蒸馏信号条件化于每个位置上教师与学生的关系，而没有利用同一 prompt 下**其他学生 rollout** 的信息。通过把一批学生 rollout 划分为成功组与失败组，MOPD 构造出同伴条件的教师信号，对比教师在给定成功学生轨迹与失败学生轨迹时会做出的预测。这种同伴条件化提供了一种任何单条 rollout 都无法提供的对比式信用分配，因为教师在失败 rollout 上的信号可以分解为一个与成功 rollout 共享的成分（无信息量）和一个隔离出该失败特有散度的残差。该方法利用的信息来源（跨 rollout 结构）不同于 token 级熵方法（TIP、EGRSD）或步级散度方法（SOD），两者在原理上互补。

**粒度自适应的 advantage 重加权（Granularity-adaptive advantage reweighting）**。GEAR（Li et al., 2026g）依据 on-policy 学生与以 ground-truth 为条件的教师之间的散度，把每条轨迹自适应地切分为变长片段。它不施加统一的 token 级或预先确定的步级边界，而是检测学生—教师 advantage 轮廓发生偏移的片段边界，然后计算 per-segment advantage，把学习信号集中在对轨迹每一区域最具信息量的粒度上。这种自适应切分填补了 token 级方法（在个别位置上可能嘈杂）与轨迹级方法（在异质区域上做平均）之间的空隙。在 LLM 智能体任务上，GEAR 优于固定粒度基线。因此，最优加权粒度不仅随任务变化（如 SOD 的步级分析所暗示），而且在单条轨迹内部也会变化。

**多模态与视界感知重加权（Multimodal and horizon-aware reweighting）**。另外三种方法把加权原则扩展到多模态与视界感知设定，同时保持底层机制不变。Visual-Advantage OPD（Liu et al., 2026c）把 token 级重加权适配到视觉语言学生上：它计算一个 per-token 视觉 advantage，量化学生 rollout 的每个 token 在多大程度上依赖视觉证据而非语言先验，然后把 OPD 梯度集中到有视觉接地的 token 上。这一构造与 Decomposed-OPD 的梯度引导（第 4.1 节）平行，但位于加权层而非损失形式层。Filter-Then-Reweight（Li et al., 2026m）把粗粒度与细粒度组合起来。先做一个基于教师 log-probability 的轨迹级过滤，丢弃教师自身也无法锚定的 rollout；再用一个软的 per-token 重加权把教师置信度与学生困惑度混合起来，以保留幸存的轨迹中部分信息。对加权 advantage 施加 PPO 式裁剪目标，使每个 token 权重变化很大时更新仍然有界。Less-is-More（Ziheng et al., 2026）从视界角度攻击同一个教师可靠性问题，识别出 **Off-policy Teacher Decay** 现象：学生 rollout 中越靠后的位置对教师而言越 off-policy，监督质量随之下降。在学生 rollout 中教师可靠性开始恶化的位置处截断轨迹，被证明能以一小部分 rollout 长度换回大部分 OPD 收益。这三种方法占据粒度谱系上的不同点，即模态特定的 token 加权（Visual-Advantage）、轨迹加 token 的双粒度重加权（Filter-Then-Reweight）与视界层面的 rollout 截断（Less-is-More）。

上述 token 与样本加权方法都是**反应式**运作的：在学生已经生成了一条可能无信息量的 rollout **之后**再衰减损失信号。一种更主动的做法则是问：我们能否在生成**之前**就选择 prompt，以最大化每条 rollout 的期望信息增益？这就引出了下文的课程方法。

### 6.2 课程与难度自适应（Curriculum and Difficulty Adaptation）

样本加权问题促生了一个平行的解法。**课程设计**主动选择与学生当前能力水平相匹配的 prompt。加权是在 
生成**之后**对损失做的事，而课程设计则在生成**之前**介入，办法是选择呈现哪些提示。这一设计空间横跨多个轴，包括提示难度、多轮交互中的时间深度、监督的语义粒度以及初始化策略，每个轴针对朴素均匀采样的一种不同失效模式。

**能力边界采样（Competence-boundary sampling）。** PACED (Xu et al., 2026b) 使用一个以学生当前通过率为中心的 Beta 核分布来动态建模提示难度。随着学生进步，课程向更难的提示倾斜，同时避开通过率接近零（梯度 SNR 消失）或接近一（学生已经掌握该内容）的提示。这种"前沿采样"策略把梯度预算集中到学习往往最高效的那个狭窄提示带上。

其数学依据与梯度 SNR 分析相关。对于一个通过率为 p 的提示，二元奖励下的期望梯度幅度在两个极端（p→ 0 与 p→ 1）都会消失，并在中等难度处达到峰值。PACED 通过一个一般的 Beta 核权重族 $w(p)=p^{\alpha}(1-p)^{\beta}$ 把这一点形式化，其中对称情形 α=β= 1 给出熟悉的 $p(1-p)$，峰值在 $p^*=0.5$；而非对称选择（$\alpha\neq\beta$）把峰值移到 $p^*=\alpha/(\alpha+\beta)$，从而在关于通过率分布的先验知识支持这种倾斜时，把课程倾向更难的问题（α<β，峰值在 0.5 左侧）或更简单的问题（α>β，峰值在 0.5 右侧）。PACED 证明这一 Beta 族是在 SNR 模型有界误设下的领先阶 minimax 最优加权，并在经验上表明对称默认 $w(p)=p(1-p)$ 已经捕获了大部分可获得的增益。具体而言，PACED 通过一次性的评估阶段（通常每个提示 K=8 次 rollout）估计逐提示难度 $\hat{p}_i$，并以 $\hat{p}_i(1-\hat{p}_i)$ 为每个提示加权，把训练集中到有产出的学习区间，同时避开已经掌握或难以攻克的问题。单遍估计就已足够，因为 Beta 核对于过期的通过率具有 minimax 鲁棒性，不过迭代重算可以带来适度的额外增益。

这一 SNR 分析有一个更深层的含义，它超出了课程设计的范围，因为它解释了为什么 on-policy 训练在推理任务上往往优于 off-policy 方法，且无需显式的课程工程。在 off-policy 设定中，问题难度分布由数据集固定。随着学生变强，越来越大部分的训练数据落入梯度可忽略的"过易"区域，导致收益递减。相反，on-policy 训练会自动调整有效难度，因为学生的 rollout 反映了它当前的能力，自然会在其能力边界处生成序列。PACED 把这种自动调整显式化、可调化，但即便是朴素的 on-policy 训练，也仅凭从 pθ 而非 pdata 采样就获得了大部分收益。

**自调整课程（Self-adjusting curricula）。** 前面几节中的若干方法提供了隐式课程机制，而无需显式的难度估计。π-Play 的多智能体框架（第 5.2.3 节）会随着学生变强自动生成更难的任务；PAINT (Tan & Hong, 2026a)（第 5.2.3 节）根据师生重叠度调整其部分解披露；PRISM (Wang et al., 2026g)（第 5.1.2 节）通过一个结构化的多模态预热来塑造供后续 RL 使用的策略地形。这些方法以涌现式的难度自适应补充了 PACED 的显式难度估计。

**双视角课程（Dual-perspective curriculum）。** PACED 把学生视为一个黑盒函数，其通过率完全刻画提示难度；而 Uni-OPD (Hou et al., 2026) 认为仅凭通过率会混淆两种不同的失效模式：学生无法解决的问题（因能力缺口导致低通过率）与学生以微妙方式解错的问题（中等通过率但解的质量低）。要区分二者，不仅需要考察学生**是否**成功，还要考察它**如何**失败。从**学生**视角看，Uni-OPD 实现了难度感知与正确性感知的数据平衡，在 rollout 采样期间促进对有信息量状态的探索，把 PACED 的前沿采样洞见从单一的通过率指标扩展为同时考虑问题难度与 rollout 正确性的双准则。从**教师**视角看，它处理一个更微妙的失效模式。即使学生生成了有信息量的 rollout，聚合后的 token 级

<!-- p. 46 -->

监督也可能与结果级奖励信号不一致。一条获得更高 token 级教师分数的轨迹，实际上可能产生更差的最终答案。Uni-OPD 的结果引导间隔校准恢复了正确与错误轨迹之间的顺序一致性，使 token 级梯度方向与序列级质量排序对齐。这一双视角框架可泛化到 LLM 与 MLLM（5 个领域、16 个基准），并同时兼容单教师与多教师设定，这意味着课程问题与监督可靠性问题在很大程度上是正交的，可以独立求解。

**面向多轮蒸馏的时间课程（Temporal curriculum for multi-turn distillation）。** PACED、Uni-OPD 以及那些自调整方法都在**难度**轴上操作，选择训练**哪些提示**。TCOD (Wang et al., 2026d) 开辟了一条正交的轴，它问的不是哪些提示，而是在单个提示的轨迹中**多深**的位置施加监督，从而处理多轮智能体蒸馏独有的失效模式。在应用于多轮任务的标准 OPD 中，学生生成完整的交互轨迹并在每一轮都接受教师监督。TCOD 识别出"轨迹级 KL 不稳定性"（Trajectory-Level KL Instability），即随着对话深度增加，学生与教师分布之间的 KL 散度不断升高，原因是早期轮次中的复合误差改变了状态分布，使得后续的教师监督与学生实际的轨迹上下文越来越不匹配。所提出的时间课程逐步扩展教师监督的时域，而不是从一开始就在所有轮次上均匀施加。两种调度变体针对互补的失效模式。Forward-to-Backward（F2B）从早期轮次开始监督并逐步扩展到较晚轮次，让学生先建立可靠的早期轮次行为，再去应对复合误差的机制。Backward-to-Forward（B2F）用教师生成早期轮次（提供干净的状态分布）而只在较晚轮次施加监督，随着训练推进逐步要求学生独立处理更早的轮次。在 ALFWorld、WebShop 与 ScienceWorld 上，TCOD 相比朴素多轮 OPD 带来最高 +18 点的增益，其中 F2B 变体在早期轮次错误后果最严重的任务上表现尤其突出。时间课程的洞见与上文建立的梯度 SNR 框架直接相关。正如 PACED 指出通过率接近零的提示产生的梯度可忽略，TCOD 指出长轨迹中的较晚轮次遭受类似的梯度退化，因为学生的状态已经漂移得太远，教师的监督在那里已不再有意义。

**Off-policy 冷启动（Off-policy cold start）。** Li et al. (2026j) 指出，当学生的初始策略距离教师分布太远时（"思维模式不匹配"问题），OPD 可能灾难性地失败。他们提出的解决方案是一个 off-policy 冷启动阶段，即在切换到 on-policy 训练之前，用一小段时间在教师生成的数据上做标准 SFT，以缩小模式差距。这种两阶段做法，即 off-policy 预热后接 on-policy 精修，已经成为常见的工业模式（第 8 节）。

**SFT+OPD 混合流水线（The hybrid SFT+OPD pipeline）。** off-policy 冷启动（Li et al. (2026j)）与 on-policy 精修的结合已经成为一种常见的工业配方。从经验上看，从 off-policy 训练转向 on-policy 训练的一个有用切换点出现在学生的生成分布与教师的生成分布达到足够重叠之时。切换过早会把 on-policy 算力浪费在无信息量的 rollout 上（学生太弱，无法进行有意义的自生成）。切换过晚则往往让学生卡在 off-policy 的天花板附近。Qwen3 (Yang et al., 2025a) 遵循这一两阶段做法，先用 off-policy 蒸馏建立基本推理与模式切换能力，然后转向 on-policy 蒸馏，由学生生成自己的序列以做 logit 级对齐。DeepSeek-R1 (DeepSeek-AI et al., 2025) 使用 100% off-policy（不做切换），这在教师能力极强的条件下效果很好，但对那些学生本可以通过 on-policy 探索超越教师的推理任务，可能会损失部分性能。

**数据新鲜度与教师刷新课程（Data freshness and teacher-refresh curricula）。** 课程设计的推广超出了提示难度，涵盖数据、rollout 的新鲜度以及教师刷新计划。DeltaPrompts (Jung et al., 2026) 识别出 VLM 蒸馏中的**零差值陷阱**（zero-delta trap），即相当大一部分训练提示会引出教师与学生无法区分的输出，因而产生零 OPD 信号。在受控的难度带内合成能最大化师生分歧的高散度提示，从而把数据集转换成 
一个把算力分配给有信息量样本的主动 OPD 课程。f-OPD (Chen et al., 2026c) 处理课程中的新鲜度一侧。在长时程异步 OPD 流水线中（rollout 生成、教师打分与学生更新运行在不同的 worker 上），陈旧度会不断累积并使损失不稳定。一个感知新鲜度的控制框架为正在 rollout 的策略与正在更新的策略之间的滞后设定上界，按新鲜度预算而不是固定节拍来调度 rollout，从而在吞吐更低的情况下恢复纯同步流水线所提供的稳定性。Trust-Region Behavior Blending (Plyusov et al., 2026) 把同样的洞见带入预热阶段。一个以学生为中心的 KL 约束下的教师引导行为策略在训练早期以闭式形式混合教师与学生的 token 分布，而一个退火日程随着学生策略收敛把该混合退回纯 on-policy。这一构造在完全退火时恢复常规的 OPD 最优解，同时提供了实践者原本需要手动搭建的隐式课程。Adaptive Teacher-Refresh (Guo et al., 2026) 接着追问在 self-OPD 中教师究竟何时应该移动。三个门控（一个强制最小冻结期的隔离门、一个要求可测改进的奖励棘轮门，以及一个防止序列长度退化为爆炸的长度尾部门）为自教师的每一次硬刷新设置条件，为通常靠试错调节的一个超参数给出有原则的答案。Continual Experience Internalization (Chen et al., 2026a) 为自演化智能体场景闭合了这个循环。原则级经验抽取、逐步注入，以及带拒绝采样的 off-policy 上下文蒸馏被交错编织进一个迭代式自演化循环，使智能体在多次内化轮次中保持稳定。这五种方法合在一起，把课程设计从**哪些提示**（PACED、Uni-OPD）与**在轨迹中多深**（TCOD）扩展到**哪些数据、多新鲜、如何混合、何时刷新以及如何内化**，揭示出此前工作视为固定的 OPD 课程在时间与结构上的轴。

**通过前缀失败纠正的轨迹精修蒸馏（Trajectory-refined distillation via prefix-failure correction）。** TRD (Jiang et al., 2026a) 增加了又一条轴：它问的不仅是哪些提示、多深，而是**在一条受损的 rollout 中教师应在哪个点介入**。在 on-policy 生成过程中，TRD 监控学生轨迹中的前缀失败事件，即学生的累积 token 序列已经漂离教师流形太远、以至于继续施加教师监督将是在一个分布外前缀之上进行的位置。TRD 不是丢弃整条 rollout，也不是均匀施加监督，而是在检测到的失败点插入教师生成的纠正，把纠正后的后缀拼接回 rollout，并在纠正后的轨迹上施加蒸馏。这一轨迹精修步骤把原本只会贡献噪声或零梯度信号的低质量 rollout 转换成可用的训练样本，在不需要超出纠正推理调用之外的额外教师查询的情况下，同时提升样本效率与训练稳定性。

### 6.3 计算优化（Compute Optimization）

token 加权（第 6.1 节）减少了浪费在无信息量 token 上的有效梯度预算，而课程设计（第 6.2 节）则完全避免生成无信息量的 rollout。二者合起来提升了**样本效率**，让每次 rollout 提取出更多学习。然而二者都没有消除底层的**系统级**瓶颈。OPD 要求在每一个训练步都生成学生 rollout **并**用教师对其打分，因此无论每条 rollout 被利用得多么高效，相对于 off-policy SFT 都会产生可观的额外计算开销。下文的方法攻击的是这一正交的开销维度。

一个典型白盒 OPD 步骤的成本构成包括：

1. **学生 rollout**：自回归生成 B×K 条补全（批大小 B，每个提示 K 条 rollout），因 KV cache 增长而受内存限制。实践中这一部分通常主导墙钟时间，因为生成是串行的，无法在 token 之间轻易并行化。
2. **教师打分**：对教师做一次前向传播，以在学生生成的序列上获得全词表 logit，受算力限制。当教师明显大于学生时，这一遍的开销可能与学生的反向传播相当甚至更高。

<!-- p. 48 -->

3. **学生更新**：计算蒸馏损失梯度的标准反向传播，主要受优化器状态内存支配。这一部分通常是三者中最便宜的，但它设定了内存使用的下界。

每个组成部分都允许独立优化，下文的方法分别攻击不同的瓶颈。

**前缀截断（Prefix truncation）。** FOPD (Zhang et al., 2026b) 观察到有用的蒸馏信号（以逐 token reverse-KL 损失衡量）集中在学生生成序列的前缀部分，并推测学生在早期 token 所承载的高层规划决策上最为薄弱。随着生成越来越受前缀上下文约束，靠后的 token 贡献的信号呈边际递减。利用这一观察，FOPD 把 on-policy rollout 截断到长度为 k 的前缀，其余部分退回到 off-policy KD。On-policy 前缀蒸馏（FOPD）在把训练 FLOP 减少 2×–47× 的同时达到完整 OPD 的性能。最优截断点 k 取决于任务的信息密度剖面。以 Qwen3-8B 为教师、Qwen3-8B-Base 为学生时，一种每个梯度步按固定增量（∆L= 256）从 1 个 token 开始逐步增加训练前缀的前缀调度策略，在降低计算量的同时接近完整 OPD 的质量。

**离线教师缓存（Offline teacher caching）。** Lightning-OPD (Wu et al., 2026a) 通过在固定的 SFT rollout 上一次性预先算好教师 log 概率，把学生优化与教师推理彻底解耦，从而在训练期间不再需要实时教师服务器。其使能条件是**教师一致性**（teacher consistency），即要求 SFT 阶段（生成训练轨迹）与 OPD 阶段（提供参考分布）使用同一个教师模型。在教师一致性下，Lightning-OPD 可证明与标准在线 OPD 共享同一个最优解，并具有有界的梯度偏差以及一种隐式正则化效应，后者在没有显式 KL 惩罚的情况下防止策略漂移。该框架相比标准 OPD 获得 4.0× 更高的训练效率，大幅降低了学术研究的门槛。

这两种方法揭示出一个更深层的结构性要点。"on-policy"这一要求并非二元的，而是容许不同程度的近似。FOPD 表明部分 rollout 保留了大部分分布对齐收益。Lightning-OPD 表明当教师一致性成立、且隐式正则化约束住策略漂移时，固定（过期）的教师信号依然有效。二者合起来定义了一条**保真度-效率前沿**（fidelity-efficiency frontier），沿这条前沿 on-policy 近似的程度与计算成本相互权衡：一端是完全 on-policy 训练（最大对齐、最大成本），另一端是完全离线 SFT（最小成本、暴露偏差上限）。

**投机知识蒸馏（Speculative knowledge distillation）。** DistillSpec (Zhou et al., 2024) 首次把 on-policy KD 应用于改进**推理时**投机解码所使用的 **draft** 模型，训练 draft 更好地匹配目标模型的分布，从而提高投机提议 token 的接受率。SKD (Xu et al., 2025c) 随后把投机解码机制适配到**训练侧**，由学生生成候选 token，教师验证并依据自身分布有选择地替换它们，从而形成一种在监督式 KD 与 on-policy KD 之间动态插值的数据生成策略。当学生样本持续被拒绝时，SKD 退化为监督式 KD。当它们持续被接受时，则退化为 on-policy KD。投机解码的期望墙钟加速为

$$
E[S]=\frac{\tau(x)}{c\gamma+1}
$$

其中 $c=\dfrac{\text{cost(draft forward)}}{\text{cost(target forward)}}$，τ(x) 表示块效率（每个验证步期望生成的 token 数），γ 表示块大小。DistillSpec 通过把 draft 模型的分布与目标模型的分布对齐来改进 τ(x)，相比标准投机解码获得 10%–45% 的加速提升。结合在贪心与非贪心采样下的有损投机解码，DistillSpec 还使得对延迟-性能权衡的细粒度控制成为可能。在实际的多尺度部署中，先蒸馏出一个更强的目标模型，再应用 DistillSpec 训练一个对齐良好的 draft 模型，可以把解码延迟降低 6–10×，而相对于不做蒸馏的标准解码仅有极小的性能下降。

<!-- p. 49 -->

**GPU 内存管理（GPU memory management）。** 除了 FLOPs，白盒 OPD 最严峻的约束是 GPU 内存。同时持有一个 70B 教师和一个 7B 学生，需要教师的权重（BF16 下约 140 GB）、学生的权重与优化器状态（约 84 GB），以及用于蒸馏的全词表 logit 张量 [B, T,|V|]。峰值内存很容易超出一个 8× 80 GB 的 H100 节点。实践者通过以下方式缓解：（1）**教师量化**，以 FP8 或 INT4 服务教师，在精度损失可忽略的情况下获得 2–4× 的内存下降；（2）**logit 卸载**，立即把 logit 流式写入 CPU 内存，或只保留 top-k 项（因为 99% 的概率质量通常集中在不到 100 个 token 上）；以及（3）**激进的梯度检查点**，以在反向传播期间最小化学生的激活内存。在标准集群上实现可行的白盒 OPD 通常需要把这些技术组合起来。

**具体成本示例（Concrete cost example）。** 为把这些公式落到实处，考虑在 8×H100 GPU 上把一个 70B 教师蒸馏进一个 7B 学生（基于已报告基准做缩放得到的代表性估计）。在 1B token 上做 off-policy 涉及三个阶段。教师离线生成数据集（约 200 GPU 小时），学生训练约 100 GPU 小时，合计约 300 GPU 小时。在 1B token 上做 on-policy 时，每个步都需要（1）学生生成（由于自回归解码，约为前向传播成本的 3×）、（2）教师打分（一次 70B 前向传播）与（3）学生反向传播，总计约 1,200–1,500 GPU 小时，即 4–5× 的额外开销，这与更小规模上的经验测量一致（Wu et al., 2026a）。对于推理任务，on-policy 方法往往优于 off-policy SFT，而且 off-policy 的性能天花板更低，因为静态数据集无法覆盖有效推理路径的组合空间。这种"天花板差距"随任务复杂度而扩大，为 on-policy 方法提供了强有力的成本-收益依据。第 7.4 节把这一成本-质量权衡形式化为一个决策框架。

**异步生成-训练解耦（Asynchronous generation-training decoupling）。** NPD (Rang et al., 2026) 攻击 OPD 的一个主要计算瓶颈，即学生生成与梯度更新之间的同步耦合，它迫使 GPU 在受内存限制的解码与受算力限制的反向传播之间交替。NPD 把 on-policy distillation 重新表述为一个三阶段异步流水线。学生 rollout 通过 vLLM 批量生成，教师 top-k logit 通过带序列打包的并行 prefill 计算，学生则在打包序列上以 CE+KD 的复合损失训练。异步 OPD 的主要挑战是策略滞后（policy lag），即生成策略在异步迭代之间偏离训练策略，可能退化为 off-policy 学习。NPD 用 ∆-IFD 过滤来应对，它通过 Instruction-Following Difficulty 指标估计师生差异，并拒绝那些会把更新推出"安全近端学习区"的极端分布外样本。由此得到的框架相比同步 on-policy 基线记录了 8.1× 的吞吐加速，同时在覆盖通用、数学、推理与代码任务的 11 个评测基准上平均超出 SFT +8.09%。NPD 还表明 on-policy distillation 为后续 RL 产生更优的初始化，收窄了 GRPO 的探索空间。NPD→GRPO 流水线使 openPangu-Embedded-1B 达到 68.73% 的平均准确率，超过了更大的 Qwen3-1.7B（63.69%）。NPD 与其他计算方法之间的关系是范围正交的关系。FOPD 缩短每条 rollout 的**长度**，Lightning-OPD 消除了对教师信号**新鲜度**的要求，而 NPD 消除了生成与训练之间的**同步**要求。三者都可以组合在同一个系统内。

**参数动态的前瞻性（Parameter-dynamics foresight）。** EffOPD (Cai et al., 2026b) 从另一个角度看待效率，它在参数层面追问**为什么** OPD 比 RL 收敛更快。分析表明 OPD 的效率源自一种"前瞻"：它在训练早期就建立起一条通向最终模型的稳定更新轨迹。在模块分配层面，OPD 识别出边际效用低的区域，并把更新集中到与推理最相关的模块上。在更新方向层面，OPD 表现出更强的低秩集中性，其主要子空间在训练早期就与最终更新子空间紧密对齐。基于这些发现，EffOPD 提出一种即插即用的加速方法，它自适应地选择外推步长并沿当前更新方向移动，在不需要额外可训练模块或复杂超参数调节的情况下实现 3× 的训练加速，同时保持可比的最终性能。 
**漂移感知的动态截断（Drift-aware dynamic truncation）**。Prune-OPD（Yang et al., 2026g）针对一种更隐蔽、且专门出现在长时程推理中的算力浪费来源。随着学生生成更长的序列，其前缀会逐步偏离教师的推理路径，使后续的教师奖励越来越不可靠。Prune-OPD 不施加固定截断长度（如 FOPD 那样），而是通过 per-position top-k 重叠比对局部学生—教师兼容性做持续监测。当该指标跌破阈值时，就记录一次 **prefix-drift event**，随后对 OPD 奖励做单调降权。一旦累积漂移超过预算，整条 rollout 就被完全截断，把算力严格重新分配到教师监督可靠的区域。在多种教师—学生组合上（DeepSeek-R1-Distill-Qwen-7B→R1-Distill-Qwen-1.5B、JustRL-DeepSeek-1.5B→R1-Distill-Qwen-1.5B、Qwen3-4B→Qwen3-1.7B/4B-Base），Prune-OPD 把训练时间减少 37.6–68.0%，同时在 AMC、AIME 和 HMMT 上保持或提升性能。其自适应行为是关键：当学生—教师兼容性保持高位时，Prune-OPD 会通过扩大训练窗口自动保留长上下文监督，避免固定截断带来的信息损失。因此 FOPD 与 Prune-OPD 代表两种互补的截断哲学。FOPD 使用一个**按计划**推进的前缀，它不依赖兼容性而统一增长；Prune-OPD 则使用一个**反应式**预算，它依据实时漂移检测收缩或扩张，更适合漂移发生时点在不同实例间变化的长时程任务。

**rollout 截断与局部化蒸馏（Rollout truncation and localized distillation）**。另外两项算力侧的创新针对 rollout 成本本身，而不是系统级的内存墙。POPD/TOPD（Zhang et al., 2026k）考察是否真的需要完整 rollout。Progressive Horizon Expansion（POPD）从较短的 rollout 视界开始并逐步增长，使早期训练集中于信号密度最高的前缀（与 FOPD 的前缀截断发现一致），后期训练则在学生稳定后扩展覆盖范围。Truncated Rollout Distillation（TOPD）则始终在单个较短的 log-ratio 视界上训练，以序列覆盖换取 per-step 成本的下降。SafeSteer（Li et al., 2026e）从另一个方向攻击成本：把 OPD 更新限制在一小组与安全相关的 token 上。一个激活引导的安全教师只对经对比式 log-probability 投票标记为安全关键的 token 提供 per-token reverse-KL 信号，rollout 的其余部分不受影响。结果是一种局部化的蒸馏流程，其算力成本随安全 token 占比而非完整 rollout 长度增长，说明任务特有的结构可以恢复前缀截断方法通过通用视界控制所取得的效率。KAT（Xin et al., 2026）处理另一种失效模式，识别出 **KL-agreement trap** 区域，即学生与教师的 top-1 预测一致、但完整分布差异很大的连续 token 片段，它们制造出一种具有欺骗性的梯度地形，把算力浪费在低价值的更新上。一个在线 rollout 监视器实时检测这些陷阱区域，并在陷阱入口处截断当前 rollout，把生成预算重新分配到学生—教师散度真实且富有信息的新鲜 rollout 上。实时检测与在线截断的组合不需要额外前向传播，并能直接整合进现有的 OPD 训练循环，以基于一致陷阱的准则补充 Prune-OPD 基于漂移的预算，从而针对一种结构上不同的算力浪费来源。

**训练效率栈（The training efficiency stack）**。上述三个小节构成一个分层的效率栈。token 加权（第 6.1 节）在梯度层面降低噪声，使每条 rollout 对学习的贡献更有效。课程设计（第 6.2 节）在采样层面减少浪费，使每条 rollout 都瞄准学生的能力前沿。算力优化（本节）在系统层面降低开销，减少生产与打分每条 rollout 的成本。这些层次是互补而非替代关系。PACED 的前沿采样生成信息量最大的 prompt，TIP 的 token 加权从由此产生的 rollout 中提取最大信号，而 FOPD 的前缀截断最小化这些 rollout 的生成成本。同时部署这三层可以把它们各自 2–4× 的效率增益叠加起来，有可能把 on-policy 与 off-policy 训练之间的有效成本差距从 4–5× 缩小到接近持平，同时保留质量上限上的优势。

<!-- p. 51 -->

## 7 理解 OPD：理论、失效模式与成本（Understanding OPD: Theory, Failure Modes, and Cost）

前面各节讨论了 on-policy distillation 的机制，涵盖优化什么（第 4 节）、信号来自哪里（第 5 节）以及如何稳定训练（第 6 节）。本节转向同样重要的**为什么**与**何时**：OPD 在什么条件下成功，它如何失效，有哪些理论框架统一了上文考察的方法，以及何时 OPD 优于更简单的 off-policy 流水线？

这四个理解维度并非彼此独立，而是构成一个层层递进的论证。识别 OPD **何时**有效（第 7.1 节）确立了理论框架必须解释的前提条件。罗列它**如何**失效（第 7.2 节）为具体的设计选择——自适应散度、课程加权与混合 RL 目标——提供了动机，而第 7.3 节为这些选择给出理论依据。成功条件与失效模式随后共同支撑第 7.4 节的实践成本—收益分析，后者把这些理论洞见提炼为工程准则。解决这些问题，会把 OPD 从一种基本靠经验的实践推进为一门更系统的工程学科（Zhang et al., 2026f; Li et al., 2026j）。

### 7.1 成功条件（Success Conditions）

Li et al. (2026j) 对 OPD 训练动力学做了系统研究，指出有效蒸馏的两个必要条件。第一，学生与教师必须共享兼容的推理模式（操作化为二者 top-k token 分布的高重叠）。即使教师更优，如果其思考风格与学生在结构上不匹配（例如一个非 thinking 教师被蒸馏进一个 thinking 学生），也无法迁移知识，因为初始分布重叠过低无法仅靠训练恢复。第二，教师必须提供学生尚未获得的新能力。当两个模型在同一数据与配方上训练时，它们会在各自规模上收敛到相似的分布，使教师几乎没有可迁移的信号，正如 reverse distillation 实验所示：从学生的视角看，同族的 1.5B 与 7B 教师变得在分布上无法区分。

在 token 层面，成功的 OPD 以在**高概率重叠 token**——即集中了大部分概率质量的那个小的共享 token 集合——上的渐进对齐为特征。当这些条件不满足时，可用的恢复策略包括 off-policy 冷启动（通过 SFT 初始化缩小模式差距）以及与教师对齐的 prompt 选择。一个持续的挑战是：OPD 稠密监督的质量会随轨迹深度系统性退化，不稳定性从较靠后的 token 起源并向后传播，这为 OPD 在长时程推理上的适用性带来了悬而未决的问题。

这些发现指向一条一般原则。OPD 的收益随教师与学生之间**可开发差距**的大小而缩放。当这一差距太小（同配方模型）时，教师提供的新信息很少。当差距太大（思考模式不匹配）时，学生无法吸收监督。有产出的机制位于二者之间，而识别它仍是一个关键的诊断挑战。

**免训练 per-token 诊断（Training-free per-token diagnostics）**。Armandpour et al. (2026) 提出一个互补的诊断框架，它以最高分辨率运作：per token、per question、per teacher。他们导出一个理想的 per-node 梯度，定义为能最大程度提升学生成功概率的参数更新，并开发出一个可扩展的 targeted-rollout 算法来高效估计它，即使对很长的中间思考链也适用。gradient alignment score（该理想梯度与给定蒸馏梯度之间的余弦相似度）量化了某个教师配置在多大程度上逼近理想信号。在自蒸馏设定与外部教师模型中，蒸馏引导在错误 rollout 上表现出的与理想梯度的对齐都明显高于正确 rollout——在正确 rollout 上学生本已表现良好，教师信号往往变得嘈杂。最优的蒸馏上下文同时取决于学生模型的容量与目标任务，没有出现单一普遍有效的配置。这些发现为上文成功条件所引出的 per-task 诊断分析提供了一个有原则、免训练的基础。

<!-- p. 52 -->

**诊断清单（Diagnostic checklist）**。基于所考察的文献，有四项预训练诊断可以作为 OPD 收益的预测因子：

1. Token 重叠率：在留出集上计算教师与学生之间的 top-k 重叠。如果过低，先施加 off-policy 冷启动。
2. 目标 prompt 上的通过率：如果学生在目标领域一道题也解不出，OPD 梯度将会消失。使用课程节奏控制（Xu et al., 2026b）或更容易的预热任务。
3. 教师校准：验证教师的置信度在学生生成的前缀上是否与正确性相关（Zhang et al., 2026d）。校准不佳的教师会注入噪声。
4. 长度稳定性：在训练早期监测输出长度。长度突然膨胀表明出现了 Luo et al. (2026a) 所识别的自我强化重复循环。应预先施加 Stable-OPD 修正。

**OPSD 是压缩而非纠错（OPSD as compression, not correction）**。Kim & Lee (2026) 更清晰地刻画了 on-policy 自蒸馏在支持 thinking 的数学推理中实际完成的事情。通过把 OPSD 分别应用于仅包含正确 rollout 与仅包含错误 rollout 的组，他们分离出两种可能的机制。仅正确样本的 OPSD 在保持准确率的同时大幅缩短推理轨迹（压缩），而仅错误样本的 OPSD 则使准确率下降（纠错失败）。三种替代解释（教师上下文更丰富、轨迹中途反馈重注入、训练时间更长）都被检验并排除，支持这样一解释：以 hindsight 为条件的自教师能够识别长思考轨迹中的冗余，但不能可靠地提供更好的替代推理步骤。这一发现细化了上述成功条件。OPSD 的有产出机制似乎是“让模型更高效地表达已知解法”，而不是“让模型解决更难的问题”。其实践含义是更精细的流水线排序：SFT 建立格式合规性，RLVR 扩展可达的正确轨迹集合，OPD 则把它们压缩以降低推理成本。颠倒最后两个阶段（在探索之前先压缩）往往会移除探索本会产生的原材料，这提示了为何后置 RL 的位置更受青睐。在 Qwen3-8B 与 AceReason-Nemotron-7B 上（并以 DeepSeek-R1-Distill-Qwen-7B 作为补充性的 pre-RL 参照），他们的实验中 post-RLVR + 仅正确样本 OPSD 的配置在准确率—长度平面上达到最有利的位置，这与 OPSD 的机制性角色相符，而不是取决于实现细节。

### 7.2 失效模式（Failure Modes）

作为成功条件的补充，越来越多工作系统梳理了即使结构性前提成立、OPD 仍会失效的具体方式，这些失效是 on-policy 训练动力学自身的产物。理解这些失效模式有助于调试 OPD 流水线，而每一种模式都有上文各方法章节中所开发的相应缓解策略。

我们按**根因**而非症状来组织失效模式，因为多种失效模式可以通过不同机制产生相似的表面退化（例如准确率坍缩）。

**flawed prefix trap（有缺陷前缀陷阱）**。如第 6.1 节所述，当学生生成一个错误前缀时，教师对后续 token 的条件分布校准不佳，因为教师从未在此类分布外输入上训练过。Fu et al. (2026b) 形式化刻画了三种具体失效模式，包括 (1) 不均衡的单 token 信号，它把分布匹配退化为单一样本；(2) 在分布外学生前缀上教师引导不可靠；(3) tokenizer 不匹配造成的扭曲。该理论分析识别出一个偏差—方差权衡：token 级 OPD 相对于序列级 Reverse KL 是有偏的，但享有更紧的最坏情况方差界。

**结构化输出中的外推悬崖（Extrapolation cliff in structured outputs）**。ListOPD（Li et al., 2026h）识别出结构化输出任务上奖励外推式 OPD（$\lambda > 1$）特有的一条陡峭失效边界。在单位置 Bernoulli 归约中，作者导出一个闭式的、相对于基线的裁剪安全阈值 $\lambda^\star(p, b, c)$，它由三个可测量量决定，即教师 
modal 概率、warm-start 质量以及 importance-sampling 裁剪强度。超过 λ⋆ 后，外推得到的不动点会离开裁剪安全区域，使训练从保持格式切换到格式崩塌。在校准过的 listwise JSON 任务（Amazon Fashion）上，三项预注册测试在锁定的预测窗口内印证了所预测的断崖区间。在略低于 λ⋆ 处运行，ListOPD 以五分之一的参数让 1.7B Qwen3 学生在域内达到与 8B-SFT 基线持平的水平，而增益主要由格式遵循而非排序质量驱动。这一外推断崖与 G-OPD 的 reward 外推机制（4.3 节）直接相关，为“对教师的外推何时从有益变为有害”提供了形式化刻画。

持续高损失 token 作为结构性残差（Persistent high-loss tokens as structural residuals）。Jiang et al. (2026b) 研究了一个相关的失效现象。即便 OPD 训练达到表面饱和之后，仍有相当一部分 token（最多 18%）持续表现出高损失。这些“Rock Tokens”抵抗教师驱动的修正，尽管由于其高出现频率而贡献了不成比例的大份额梯度范数。因果干预表明，Rock Tokens 对实际推理性能的功能贡献可忽略，说明它们代表的是学生模型无法或无需从教师那里内化的结构性、篇章性残差（连接词、格式、含糊限定语）。这一发现意味着，OPD 优化带宽中有相当一部分花在既不能提升也不能告知学生推理能力的 token 上；它与 TIP 的 token 加权框架（6.1 节）互补，刻画了哪些 token 是结构性无信息，而不仅仅是当前无信息。

跨设置的综合失效分析（Comprehensive failure analysis across settings）。Zhu et al. (2026) 提出了一项横跨 OPD 与 OPSD 的系统性经验研究，逐项记录各方法何时有效、何时失效。三种失效机制浮现出来：(1) 由条件于学生生成 prefix 引起的师生分布失配（即上文的有缺陷 prefix 陷阱，现在在多种设置下得到测量），(2) 来自标准实现悄然引入的有偏 top-K reverse-KL 梯度的优化不稳定，以及 (3) 一个 OPSD 特有的局限：学生学到一个不含 PI 的策略，它聚合了以 PI 为条件的教师；当 PI 是实例特定的（例如真值解）时这不充分，但当 PI 表示一个共享的潜在规则（例如 system prompt）时则有效。stop-gradient 的 top-K 目标、经 RLVR 适配的教师，以及经 SFT 稳定的学生，都可缓解这些失效。OPD 有效性取决于 privileged information 类型（共享规则 vs. 实例特定答案）这一发现，细化了 5.2.1 节的 PI taxonomy，暴露出一个课程或信号增强补丁可能无法完全解决的结构性局限。

两种更隐蔽的失效模式加剧了 prefix 陷阱。梯度 SNR 崩塌（Gradient SNR collapse）发生在学生在一个 prompt 上的通过率趋近于零时。在这种状态下，所有 rollout 都包含早期错误，教师信号被噪声主导，梯度 SNR 消失。这有助于解释为什么在标准 OPD 中困难 prompt 不产生学习信号，并促成了 PACED 的能力边界课程（6.2 节）。认知压制（Epistemic suppression，Kim et al., 2026b）表现在自蒸馏中。当该过程缩短推理轨迹时，它会不成比例地删除那些充当困难子问题隐式注意力标记的含糊限定语与不确定性标记。这产生更短但校准更差的推理，模型会自信地投入有缺陷的步骤。这种校准退化直接关联到教师不确定性挑战：过度自信的学生产生不可靠的自蒸馏目标，形成一个每迭代都不断加剧的自我强化循环。

局部可教性崩塌（Local teachability collapse）。Liu et al. (2026b) 识别出强到弱 OPD 特有的、比有缺陷 prefix 陷阱更细粒度的失效模式。即便教师的反馈在学生生成的轨迹中名义上一直可用，较后的片段也会逐渐失去那种使密集监督对学习有效的局部对比（local contrast）。教师相对学生 top-K 候选集的间隔收窄到 token 级 KL 梯度几乎不携带判别信息的程度，作者将这一状态称为局部可教性崩塌。它与有缺陷 prefix 陷阱（教师分布被分布外 prefix 破坏）的不同之处在于：教师仍然校准良好，但其信号变得信息量不足。所提出的释放规则（release rule）在句子片段上聚合教师的判别间隔，并在 BIC 式下行变化点处截断 OPD 监督，把学习集中在教师反馈仍然可操作的轨迹区域。

<!-- p. 54 -->

在使用 Qwen3 模型族的强到弱蒸馏任务中，这种截断在五个域内基准上优于标准的全轨迹 OPD，并更好地保持域外能力。这一发现细化了 FOPD 的 prefix 截断策略（6.3 节）——后者基于固定的位置启发式而聚焦早期 token。局部可教性崩塌表明，最优截断点是轨迹特定的，应由教师的局部判别能力而非仅由 token 位置决定。

自博弈饱和（Ouroboros 问题，Self-play saturation）。在自蒸馏设置中，学生针对一个共享其自身归纳偏置与架构局限的目标进行优化，导致假设空间逐渐坍缩。如果模型发现一个句法上的“hack”或一个高置信度但有缺陷的推理启发式，没有任何外部信号会惩罚它。模型自我强化这条有缺陷的轨迹，驱使 pθ(yflawed|x)→1。一旦对自己的幻觉完全确信，梯度消失、探索停止，策略被困住。这与 exposure bias（源于 off-policy 设置中的训练-测试分布失配）不同。饱和源于完全 on-policy 自蒸馏中分布多样性的缺失，在结构上类似 GAN 的模式崩塌——一旦判别器信号变得无信息，生成器就会锁定在一个狭窄的输出流形上。

精确率-召回率权衡（The precision-recall tradeoff）。Cha & Cho (2025) 形式化了蒸馏中的一个核心张力。当学生把概率质量集中到高质量输出上（提高精确率）时，它必然减少对教师完整输出分布的覆盖（降低召回率）。在 OPD 设置中，这表现为“多样性崩塌”问题。激进的 Reverse KL 蒸馏产生一个在 Pass@1（精确率）上得分很高、但在 Pass@k（召回率）上灾难性表现的学生，因为学生已坍缩到每个 prompt 只有单一推理策略。on-policy 生成可以通过允许学生探索自己的分布、而不是被约束在教师演示过的路径上，来移动这一权衡，有望在不牺牲精确率的情况下提升召回率。如 GKD 与 G-OPD 所用，在 on-policy 生成期间使用温度缩放采样提供了一个直接的控制旋钮：更高的采样温度鼓励分布探索（提升召回率），代价是更嘈杂的梯度（可能降低精确率）。

校准-能力差距（The calibration-capability gap）。CaOPD (Zhang et al., 2026d) 识别出一条“误校准的缩放定律”（“Scaling Law of Miscalibration”）。尽管 OPD 能可靠提升任务准确率，作者报告称它倾向于让模型处于严重过度自信的状态。他们把根因归结为信息失配：教师监督是在训练时可获得的 privileged 上下文（完整推理轨迹、多次采样）下形成的，而部署时的模型必须仅用其部署期上下文来报告置信度。通过把能力提升与校准退化解耦，CaOPD 论证蒸馏后的模型变得更有能力，但对自己的不确定性边界了解更少。这一发现对部署有潜在影响：一个在 benchmark 上得分更高、却不能可靠指出自己何时可能出错的蒸馏模型，可以说比一个更弱但校准更好的基线更不安全。“确定性的错觉”（illusion of certainty）与上文的饱和分析相关联。两者都表明，OPD 可能产生在任务级性能提升的同时丧失元认知能力（不确定性意识、认知表述）的学生。

多轮 OPD 中的智能体崩塌（Agentic collapse in multi-turn OPD）。前面的失效模式都假设单轮回答。把 OPD 扩展到多轮工具使用智能体会浮现出一类性质不同的失效家族：其中教师动力学与轨迹结构之间的相互作用产生单轮设置中见不到的病理。TT-OPD (Jeong, 2026) 在 Healthcare AI Gym 上诊断出三种此类病理，每一种都映射到一种独特的结构性失配，而其他智能体 OPD 方法也从互补角度面对这一问题。第一，教师动力学崩塌（teacher-dynamics collapse）。周期性硬拷贝教师重置导致 KL 崩塌（散度从其累积值骤然降至接近零，例如在周期 T=30 的单次拷贝事件中从 2.637→0.343），抹掉蒸馏梯度并驱使准确率单调下降。这是 TCOD 的“轨迹级 KL 不稳定”（6.2 节）的多轮对应物，但出现在教师更新的粒度而非轮深粒度上，说明稳定的教师动力学（EMA 而非硬拷贝）是任何其他结构性修复的前提。第二，轨迹结构侵蚀（trajectory-structure erosion）。即便使用能防止 KL 崩塌的 EMA 教师，多轮结构本身也会侵蚀，因为学生学到 
冗长的单轮独白比协同的工具使用序列更容易最大化稀疏的终端 reward（在周期性重置下，每个 episode 的轮数从 7.65 崩塌到 5.52；即便使用 EMA 教师，也会从 7.82 侵蚀到 6.23）。这映照出促使 Skill-SD 做技能级分解、MAD-OPD 做步骤级采样的 reward 粒度失配（第 8 节），说明仅靠稀疏终端 reward 无法维持多轮结构，无论粒度如何。第三，reward 提示失控（reward-hint runaway）。加入以结果为条件的教师提示而不加长度控制会触发回复爆炸：正向提示强化越来越长的推理，直到回复填满上下文上限、准确率从 54.5% 崩塌到 49.0%。这是 Kim et al. (2026b) 记录的认知压制现象的多轮延伸，因为两者都发生在“平均而言有帮助”的监督信号被应用、却缺少对学生回复形态的正则约束之时。

综合来看，这些病理为智能体 OPD 暗示了一条一般原则。基于当前证据，稳定的教师动力学、对轨迹结构的显式正则项、以及与粒度匹配的信用分配似乎共同是必要的，而每个已有方法（TCOD、Skill-SD、MAD-OPD、TT-OPD）都针对这三者的不同子集。

度量外能力侵蚀（Off-metric capability erosion）。除了任务特定的失效模式之外，Wang (2026) 认为蒸馏评估本身在结构上是不完整的。当前实践只报告学生保留了什么（任务分数），却没有计入他们失去了什么（那些使这些分数可靠的、度量之外的教师能力）。他们把蒸馏重新框定为有损投影，并将现有证据综合成一个度量外损失 taxonomy，包括校准退化（学生得分高但不知道自己何时出错）、推理轨迹质量侵蚀（更短的轨迹丢失中间验证步骤），以及分布收窄（pass@1 指标看不见的模式崩塌）。这一 taxonomy 横切上文中编目的具体失效模式，提供了一种统一的评估视角。所提出的蒸馏损失声明（Distillation Loss Statement）（主要指标、关键的度量外能力、保持目标、压力分布、观察到的侵蚀）为报告蒸馏结果提供了一个超越头条准确率的结构化模板，与 7.1 节的诊断清单互补，提供一套面向部署的评估协议。

exposure bias 恢复、多教师干扰与梯度稳定性（Exposure-bias recovery, multi-teacher interference, and gradient stability）。另外三项失效模式分析关联到上文的 prefix 陷阱与饱和病理，但指出不同的结构性成因。MOTAB (Wang et al., 2026a) 编目了推理蒸馏中的双重 exposure bias：传统的训练-推理 prefix 失配，以及在学生滚入教师很少访问状态时出现的高熵 token exposure bias。所提出的补救措施监控学生的 on-policy 轨迹，一旦超过散度阈值就回溯到最近的安全状态，之后教师恢复纠正性监督。这一构造把 exposure bias 的缓解框定为一种可恢复操作，而非回避性约束，与 FOPD 和 Prune-OPD 的 prefix 截断哲学互补（6.3 节）。Counteraction-Aware MOPD (Chen et al., 2026b) 暴露出领域专门化过程中灾难性遗忘问题的多教师对应物。当 prompt 分布与各教师的专长相不匹配时，vanilla 多教师 OPD 会悄然退化通用能力，因为来自能力区域互不重叠的教师的冲突 token 级信号会相互抵消而不是复合。一个反作用感知路由器检测这些冲突并分配监督以避免破坏性干扰，在保持领域性能的同时恢复通用能力。Physics-Guided Self-Distillation (Wang et al., 2026f) 从失效景观的梯度稳定性一侧入手，用信息论信号调制每步更新幅度。学生预测与反馈条件化教师之间的互信息估计充当类似粘性的乘子，并由一个指数上界约束，在低 MI 区域（教师信号无信息）收缩步长、在高 MI 区域放松步长。这一构造把 OPD 更新解释为一个耗散动力系统，为固定步长或仅靠学习率的调度提供了有原则的替代方案。每种方法处理失效堆栈的不同层，即轨迹回溯（MOTAB）、多教师路由（Counteraction MOPD）与步长调制（Physics-Guided SD）。

<!-- p. 56 -->

### 7.3 统一理论视角（Unified Theoretical Perspectives）

上文的成功条件与失效模式共享着比其多样表面表现所暗示的更深的数学根源。

Wen et al. (2023) 的统一 f -divergence 框架把 Forward KL、Reverse KL、JSD 与 total variation distance 作为特例纳入其中，由一个凸生成函数 f 参数化。由于所有这些散度都可以表示为学生 on-policy 分布下的期望，散度的选择最终是一个正则化决策。生成函数 f 决定 likelihood ratio pT(y)/pθ(y) 的隐式加权：Forward KL 对学生低估教师的区域（覆盖）加权更高，Reverse KL 对高估的区域（精确）加权更高。Wu et al. (2025) 为离散 LLM 分布细化了这一图景，表明在实际 epoch 预算下，mode-seeking/mode-covering 二分法更好地理解为头尾聚焦（head-tail focus）。Forward KL 把学习集中在教师分布的头部，而 Reverse KL 强调尾部，这解释了为何在测试设置中，基于局部几何在散度之间切换的自适应方法往往优于任何固定选择。在更基础的层面，Cha & Cho (2025) 表明生成模型中的 KD 会引发一个由单一参数——教师熵——调制的精确率-召回率权衡。当教师分布熵更低（更尖峰）时，蒸馏后的学生把概率质量集中在高似然区域，提升样本精确率，代价是分布覆盖（召回率）。这一权衡在大规模 LLM 实验（SmolLM2 家族）中得到验证：熵更低的教师产生的学生生成更锐利、更流畅的文本，但覆盖的输出空间模式更少。这与 Li et al. (2026j) 经验性识别出的“可利用差距”（exploitable gap）直接相关。当教师的熵显著低于学生仅靠数据所能达到的熵时，蒸馏带来的精确率增益最大——而这恰恰是教师提供了仅靠数据训练无法高效复制的、对输出空间的有信息压缩之时。另一个平行统一来自 Li et al. (2026c)：他们表明 GRPO 与自蒸馏处理互补的失效模式，并可通过样本路由组合进单一框架。正确样本走 GRPO 与 reward 对齐的强化路径，而失败样本从反馈条件化的自我教师那里获得密集的 logit 级修正；这证明两种范式针对的是不同的学习信号，而不是对同一目标的冗余做法。

超越散度层面的视角，另一条平行分析追问 OPD 如何在参数空间中重塑模型，以及其动力学是否不同于 SFT 与 RLVR 的动力学。Shen et al. (2026c) 通过一套参数空间诊断刻画 OPD 的更新轨迹，并把它置于一种放松的 off-principal 体制中：其更新触及的权重比 SFT 更少、更强地避开主奇异方向，同时受到的约束不如 RLVR 那样紧密。他们的核心观察是子空间锁定（subspace locking）：累积更新迅速进入一个狭窄的低维通道，该通道本身就足以复现 OPD 的性能，而对 SFT 施加同样的约束却会造成退化。Yu et al. (2026a) 在若干语言与视觉-语言模型对上得到一致的图景，发现 OPD 更新幅度小、坐标稀疏，但在数值上满秩，集中在 FFN 模块中，且不成比例地落在源权重接近零的坐标上。只训练所发现的子网络几乎可以复现全量训练的性能，尽管这一稀疏支撑并没有消除对自适应优化的需要，因为 AdamW 相对 SGD 仍有优势，而密集教师信号似乎需要这一优势。合起来看，这些分析表明密集教师监督并不会把 OPD 坍缩为普通的 SFT 式参数重写，而且 on-policy 采样留下的几何特征更接近 RLVR，而非仅由密集目标所能预测的。

长度膨胀（length inflation，OPD 学生的输出在训练中逐渐变长）是一个持续的实际关切。Luo et al. (2026a) 把根因归为 reverse-KL advantage 中的一个反馈回路。一旦学生漂向冗长重复的轨迹，教师在这些轨迹上的 log 概率上升得比学生更快，产生大的正 advantage，进一步鼓励退化的续写。他们的 Stable-OPD 框架加入一个参考散度项，把学生分布锚定到蒸馏前的基线，并结合 rollout 混合，把 on-policy 学生 
与 off-policy 黄金示范以衰减的比例混合。这些稳定化机制通过打破自我放大循环，平均比 vanilla OPD 提升 +7.2%。

G-OPD 的等价性推导（第 4.3 节）支持这样一种观点：标准 OPD 可以被表述为稠密 KL 约束 RL 的一个特例，从而在蒸馏与 RL 两个社区之间架起桥梁，使任一领域的进展更有可能迁移到另一领域。这一联系也延伸到基于 DPO 的方法。把教师的 log 概率代入为 DPO 的隐式奖励（第 4.3 节），会产生与 GKD 的 λ-mixing 所瞄准的相同的几何混合，从而把 token 级 KL 蒸馏、偏好优化与奖励正则化 RL 关联为通往同一族目标分布的三条优化轨迹，而该族分布由散度选择与监督密度参数化。

Li et al. (2025) 的统一 KD+RL 框架把这种互补性说得更加精确：把混合梯度分解为一个稠密的 KD 分量（稳定、可解析计算）与一个 Monte Carlo RL 分量（高方差、由奖励驱动）。KD 抑制了策略梯度估计中的方差，而 RL 则防止塌缩到次优的教师众数上。Song et al. (2026) 进一步表明，来自教师的文本反馈可以把 RL 的有效能力扩展到仅用结果奖励所能达到的范围之外，为第 5.1.2 节综述的言语反馈方法（Lion、OVD）提供了理论依据。文本反馈携带的奖励信号比二值结果更稠密，使学生能够从二值验证会丢弃的部分成功中学习。

这些视角提示了一个统一的认识。On-policy 蒸馏、强化学习与偏好优化占据的是一个由监督密度、散度选择以及训练信号来源共同参数化的连续谱，而不是若干彼此分离的训练范式。有了这一统一视角，剩下的实践问题就是：on-policy 方法在何时值得其相对于更廉价的 off-policy 替代方案所付出的额外算力成本。

### 7.4 同策略 vs. 异策略：一个决策框架（On-Policy vs. Off-Policy: A Decision Framework）

上述理论框架表明，在某些机制下，on-policy 训练相对 off-policy 替代方案具有系统性优势。但仅凭理论无法解决这一工程问题：这种优势在何时值得额外的基础设施成本。本节把累积的证据提炼为一个实用的决策框架。

**DeepSeek-R1 的异常。**DeepSeek-R1 (DeepSeek-AI et al., 2025) 是一个突出的 off-policy 蒸馏成功案例，用纯 off-policy SFT 在约 800K 条 chain-of-thought 轨迹上训练了从 1.5B 到 70B 参数的学生。在 AIME 2024（pass@1）上，R1-Distill-Qwen-7B 达到 55.5%，R1-Distill-Qwen-32B 达到 72.6%，优于同等规模下的直接 RL（在 Qwen2.5-32B-Base 上运行 GRPO 仅得 47.0%）。有三个因素可以合理地解释这一异常。一个异常强大的 671B MoE 教师稠密地覆盖了问题空间；多样化的推理轨迹包含了回溯与验证策略；而相对较小的学生模型在 off-policy 学习下保持可控的有效条件复杂度。其含义是：当一个极其强大的教师为相对较小的学生生成足够多样的轨迹时，off-policy 蒸馏往往就足够了。随着学生容量增长、并与静态训练分布产生偏离，on-policy 变得更具吸引力。

**off-policy 的天花板。**DDT (Zhang et al., 2026f) 为 off-policy 方法的泛化局限提供了形式化证据。驱动 on-policy 泛化优势的一个关键因素是：从当前策略生成训练数据能使学习与模型不断演化的分布保持一致。Off-policy 的性能受制于随着学生策略偏离静态训练数据而累积的分布错配，因为固定数据集不太可能覆盖有效推理路径的组合空间。这一差距会随任务复杂度（更多推理步骤会产生更多需要覆盖的路径）、输出多样性要求（创造性任务 vs. 数学）以及模型能力（更强的学生会探索到离训练数据更远的地方）而扩大。

<!-- p. 58 -->

**蒸馏 vs. 直接 RL。**在知识蒸馏与直接 RL 之间的选择取决于三个以非平凡方式相互作用的因素。教师相对奖励信号的质量决定了稠密的逐 token 监督是否胜过稀疏的结果奖励。学生规模会调制这一权衡：较小的学生（≤7B）更能从稠密的教师监督中获益，因为其有限容量无法高效地探索奖励地形；而较大的学生（≥32B）则有足够容量进行 RL 探索。目标天花板决定了目标是匹配教师（受蒸馏约束）还是超越教师（由 RL 实现）。第 4.3 节综述的混合方法（G-OPD、KDRL、RLAD）通过把教师 logits 作为稠密的降方差基线、与驱动探索超越教师天花板的稀疏结果奖励结合起来，化解了这一张力。在经验上，在迄今测试过的设定中，这种组合往往优于单独使用任一方法 (Yang et al., 2026e; Xu et al., 2025a; Zhang et al., 2026o)。

**混合流水线作为标准配方。**Qwen3 (Yang et al., 2025a)、DeepSeek-V4 (DeepSeek-AI, 2026) 与 MiMo-V2-Flash (Xiaomi LLM-Core Team et al., 2026) 都收敛到同一个两阶段模式：由 off-policy SFT 建立强大的分布基础，再由 on-policy OPD 或 RL 弥合学生生成分布与最优策略之间的剩余差距。过渡点的选择取决于学生的初始能力。较弱的学生受益于更长的 off-policy 预热，以缩小教师—学生模式差距 (Li et al., 2026j)；而较强的学生可以更早过渡，因为它们的 on-policy rollout 已经位于教师分布的高密度区域。

**算力—质量权衡。**每种机制下 $N$ 个 token 的期望算力成本为：

$$
C_{\text{off}} \approx N \times (F_{\text{teacher}} + F_{\text{student}} + B_{\text{student}})
$$

$$
C_{\text{on}} \approx N \times (G_{\text{student}} + \rho F_{\text{teacher}} + F_{\text{student}} + B_{\text{student}})
$$

其中 $F$、$B$ 表示前向与反向 FLOPs，$G$ 是自回归生成成本，$\rho \in (0, 1]$ 是教师监督刷新率（需要重新做一次教师前向传播的步骤占比）。由于 $G_{\text{student}} \gg F_{\text{student}}$，on-policy 训练会带来可观的倍数开销。对于 off-policy 天花板较高的通用知识迁移，这一开销很少被证明合理。而对于推理、代码生成以及复杂数学——其 off-policy 天花板低、on-policy 的边际收益高——这一投入通常是划算的。在实践中，最有效的工业流水线往往把廉价的 off-policy 学习前置，把昂贵的 on-policy 算力留给最后的质量冲刺（第 8.3 节）。

**总结决策规则。**当教师—学生容量比超过 10× 且任务只容许有界的推理深度时（事实性 QA、翻译、摘要），仅使用 off-policy SFT。当下列三个条件中任一成立时，切换到 on-policy OPD：(1) 学生超过约 7B 参数，并开始探索静态教师轨迹未覆盖的区域；(2) 目标任务涉及多步推理，中间错误会累积；或 (3) off-policy 损失进入平台期，而在 on-policy rollout 下留出集奖励仍在改善。在所有其他情形下，混合流水线（off-policy 预热后接 on-policy 精修）往往能给出有利的成本—质量权衡。

**从理论到操作指南。**上述决策框架建立在理论论证与受控实验之上。把它转化为部署实践还会引入来自基础设施与学生规模交互的进一步约束。

一个关键的操作层面细化涉及在 on-policy 机制内部选择 logit 级还是奖励级监督。较小的学生（<7B）往往受益于稠密的 logit 监督（GKD、DistiLLM）以启动能力，而配有可靠验证器的较大学生（≥7B）则倾向于受益于奖励引导的 OPD（KDRL (Xu et al., 2025a)、G-OPD (Yang et al., 2026e)），后者允许发现超越教师天花板。把 off-policy SFT 预热与 on-policy 精修相结合的混合方法在本综述所考察的文献中表现良好（Qwen3 (Yang et al., 2025a)、MiMo-V2-Flash (Xiaomi LLM-Core Team et al., 2026)）。在更深层次上，OPD 与 RLHF 共享其大部分计算结构，包括 on-policy rollout、逐轨迹信号以及策略梯度更新。主要区别在于信号来源（教师分布 vs. 奖励模型偏好）。G-OPD 与 REOPOLD (Ko et al., 2026) 这类方法形式化了这一关系，并具有实际后果。对于奖励引导的 OPD，现有的 RLHF 基础设施大部分可以直接套用（rollout-score-update 循环是相同的），这有助于解释为什么像

<!-- p. 59 -->

OpenRLHF (Hu et al., 2024) 这样的框架同时服务于这两个社区。白盒 logit 级 OPD 还需要教师共同托管与全词表 logit 迁移，而现有框架通过其教师推理进程组（teacher-inference process groups）支持这一扩展。

## 8 应用、系统与新兴领域（Applications, Systems, and Emerging Domains）

前面各节把 OPD 从理论经由方法发展到统一理解。本节考察这些算法进展如何转化为实践。从研究方法到工业部署的转变表明，OPD 方法的选择往往较少由理论优雅性决定，而更多由实际约束决定：可用的算力、教师访问层级、多教师协调需求、系统级吞吐瓶颈，以及诸如跨模态迁移与隐私等特定领域的挑战。我们按**部署模式**来组织工业应用，按新兴领域所引入的**新挑战**来组织这些领域。

### 8.1 工业部署（Industrial Deployment）

近期的基础模型越来越多地把 OPD 纳入其后训练流水线，有四种不同的部署模式，与上文综述的理论进展相对应。

**两阶段蒸馏流水线。**最常见的工业模式把 off-policy 冷启动与 on-policy 精修结合起来。Qwen3 (Yang et al., 2025a) 采用强到弱蒸馏，由更大的模型（32B、235B-A22B）在学生生成的 on-policy 序列上提供 logit 级监督，学生可以以 thinking 模式或 non-thinking 模式生成回答。一个关键的架构选择是"Thinking Mode Fusion"（思维模式融合），学生通过分别从运行于各模式的教师处蒸馏，学会既产生简短直接的回答（non-thinking 模式），也产生扩展的 chain-of-thought 推理（thinking 模式）。on-policy 生成在这里发挥重要作用，因为学生的模式选择行为（何时思考、何时直接作答）本身就是一种习得的策略，难以用静态数据集捕捉。Qwen3 报告称 on-policy 蒸馏以大约十分之一的 GPU 小时优于直接强化学习，并在 AIME benchmark 上进一步改善 pass@64，而从同一 off-policy checkpoint 出发的强化学习做不到这一点。Gemma 2 (Gemma Team et al., 2024) 把 off-policy 知识蒸馏直接嵌入预训练，以弥合其模型家族内部的性能差距。2B 与 9B 模型通过来自更大教师（27B 模型本身从头训练）的 logit-matching KD 训练，且两个小模型是并行蒸馏的，而不是走 27B→9B→2B 的级联。虽然这严格来说并不是 on-policy 蒸馏（学生是在固定的预训练数据上训练，而非在自生成序列上训练），但它代表了一种激进的整合：蒸馏从一开始就是核心训练成分，预示了后来的模型所采用的后训练 OPD 模式。MiMo-V2-Flash (Xiaomi LLM-Core Team et al., 2026) 把来自多个领域专门化教师的蒸馏与 Multi-Token Prediction (MTP) 和混合注意力结合起来，用一个 309B MoE 模型（15B 激活参数）在推理与智能体任务上取得强劲性能。

**OPD 作为模型整合。**第二种模式使用 on-policy 蒸馏把多个专家模型**合并**为一个可部署的模型。DeepSeek-V4 (DeepSeek-AI, 2026) 完全用纯多教师 OPD 替换了其前代（DeepSeek-V3）的混合 RL 阶段，通过全词表 Reverse KL 蒸馏把十多个领域专门化专家（数学、编程、agent、指令遵循）整合为一个统一的 1.6T 参数模型。在每个位置上计算完整的 $|V|$ 维 KL（而不是采样 token 近似）能在合并具有多样专长的专家时保证更稳定的梯度，而带即时 logit 重构的隐状态缓存使这在万亿参数级教师上变得可行（第 4.1 节）。KAT-Coder-V2 (Li et al., 2026b) 把智能体编程分解为五个专家领域，每个领域独立进行 SFT 与 RL 训练，再通过 on-policy 蒸馏把它们整合起来，在 SWE-bench Verified 上得分 79.6%。Nemotron-Cascade 2 (Yang et al., 2026h) 把同一原则应用到一个仅有 3B 激活参数的 30B MoE 模型上，达到金牌（Gold Medal）级性能，在 
IMO、IOI 与 ICPC 世界总决赛，而参数量比 DeepSeek-R1 少 20×。CoPD（Gu et al., 2026）指出顺序式"先专家、后蒸馏"流水线的一个关键局限：完全训练好的教师与未训练的学生之间巨大的行为模式差距，会阻碍能力的有效吸收。根本原因在于，孤立训练出的专家会发展出各自独特（idiosyncratic）的推理策略，在合并时彼此不兼容。CoPD 的解决方案是**协同演化（co-evolution）**：并行运行 RLVR 专家训练与交替的双向 OPD，使专家在自身训练过程中就互为教师，而不是在训练完成之后。这在保留互补知识的同时，维持了专家之间一致的行为模式。其结果是得到一个整合文本、图像与视频推理能力的单一模型，且优于混合 RLVR 与顺序式多教师 OPD（MOPD）。更深的洞见是架构性的。协同演化对教师多样性施加了一种软约束，从而防止了那种困扰"从独立训练专家做事后蒸馏"的专长化–固化（specialization-consolidation）差距。

**面向多预算推理的 OPD。** ORBIT（Liang et al., 2026a）针对可变推理深度的部署挑战，即用户需要根据查询难度使用不同的算力预算。它通过在逐步收紧的上下文预算（$L_{k+1} = L_k/2$）下进行 RLVR，产生各阶段的专家策略，然后通过 mode-aware Reverse KL 把它们融合为单一模型：

$$
L_{\mathrm{fusion}}(\phi) = \mathbb{E}_{k\sim U(1,K)}\mathbb{E}_{q\sim D, o\sim \pi_\phi(\cdot \mid q, p_k)}\left[\sum_t \log \frac{\pi_\phi(o_t \mid q, p_k, o_{<t})}{\pi_{\theta_k}(o_t \mid q, p_k, o_{<t})}\right]
$$

其中 $p_k$ 是模式特定的提示，$\{\pi_{\theta_k}\}$ 是各路前沿专家。模型合并被用来初始化学生，以缓解 Jang et al.（2026）所指出的冷启动问题。如果初始化不在专家流形附近，学生的 on-policy rollout 很少能到达教师的高密度区域，使蒸馏目标得不到有用的信号。在 DeepSeek-Distill-Qwen-1.5B、Qwen3-4B 与 Openmath-Nemotron-7B 上，ORBIT 得到的模型能在四种推理模式之间平滑插值（预算档位依据学生规模从 2K 到 32K token），在每个预算层级上都有具备竞争力的性能，在最高预算下匹配初始模型的准确率，同时在较低预算下带来显著 token 节省。这与上面提到的固化模式相关联。KAT-Coder-V2 与 Nemotron-Cascade 2 合并的是按**领域**专长化的专家，而 ORBIT 合并的是按**算力预算**专长化的专家，这表明 on-policy 蒸馏正在成为一个跨越多种模型专长化轴的通用多教师融合工具。

**Agentic 蒸馏。** 把 OPD 从单轮生成扩展到多轮 agentic 任务会引入一个独特的挑战：**误差累积（error compounding）**。在单轮场景下，一个有缺陷的 token 只影响同一序列中其后的 token。在多轮智能体场景下，第 $t$ 步的一个有缺陷动作会改变整个环境状态，导致其后所有教师监督都在与教师自身训练时所遇到分布相距甚远的分布上运作。这一误差累积问题催生了三种互补的解决方案，它们在不同粒度上对其发起攻击。

在**轨迹层面**，TCOD（Wang et al., 2026d）通过控制教师监督的时间深度来防止累积。它的时间课程逐步扩展视野，起初只监督较短的交互前缀（此时累积最小），并随着学生早期轮次可靠性的提升，逐步扩展到完整轨迹。这直接对抗了 TCOD 从经验上识别出的"轨迹级 KL 不稳定性"（Trajectory-Level KL Instability），在 ALFWorld、WebShop 与 ScienceWorld 上相对普通多轮 OPD 取得最高 +18 点的提升。

在**步骤层面**，MAD-OPD 的 OPAD 变体（Wang et al., 2026c）提供了一种正交的分解。它不是限制**哪些**步骤接受监督，而是把每个 agentic 步骤当作独立的蒸馏单元，通过阻止梯度跨越步骤边界流动来打断误差传播链。这种步骤级隔离与多教师辩论集成（multi-teacher debate ensemble）相辅相成，后者针对第二个 agentic 失败模式：在轨迹的后期步骤上单个教师不可靠。当个体教师在复杂的顺序决策上出错时，辩论教师的集体智能能给出比任何单一教师都更可靠的监督。结合任务自适应散度选择（agentic 稳定性用 JSD，代码用 Reverse KL），OPAD 相对单教师 OPD 把 agentic 平均值提升 +2.4%。

<!-- p. 61 -->

在**技能层面**，OpenClaw-RL（Wang et al., 2026i）使用带 Process Reward Model 的事后引导 OPD，提供步骤级 credit assignment，在个人助理、终端、GUI 与 SWE 环境中训练一个 Qwen3-4B 智能体。Skill-SD（Wang et al., 2026b）对技能条件化学习施加带动态教师同步的 importance-weighted Reverse KL，在 AppWorld 上相对 GRPO 提升 +14.0%、在 Sokoban 上提升 +10.9%。

在**轮次层面**，Li et al.（2026a）重新审视面向多轮 LLM 智能体的经典 DAgger 算法（Ross et al., 2011），并表明：在学生策略与教师策略之间进行轮次级插值——即学生生成部分轮次、教师填补其余轮次，随后学生通过监督学习在完整轨迹上训练——有效地把 on-policy 训练的分布对齐与模仿学习的稠密监督结合起来。由此得到的方法以不同机制应对了驱动 TCOD（来自学生生成前缀的协变量偏移）与 MAD-OPD（后期步骤上单教师不可靠）的同一个两难问题：在轮次粒度上混合策略来源，而不是在生成后过滤或重加权信号。在使用 GPT-4o 引导的 Qwen3-4B 与 Qwen3-8B 学生、于 SWE-bench Verified 上，DAgger 相对纯 SFT 提升 +3.9 个百分点，说明当交互单元是一个对话轮次而非单个 token 时，经典交互式模仿学习框架能富有成效地迁移到 LLM 智能体上。

从轨迹级控制（TCOD）、经步骤级隔离（MAD-OPD/OPAD）到技能级分解（Skill-SD）的阶梯，揭示出一条普遍的设计原则。有效的 agentic 蒸馏要求蒸馏单元的粒度与误差累积的粒度相匹配。粗粒度方法（全轨迹 OPD）往往表现不佳，因为累积是局部的。细粒度方法（token 级）则在智能体行为已经正确的位置上浪费算力。一种新出现的观点认为，一个有效的粒度是**决策边界（decision boundary）**，即智能体做出一个其后果难以逆转的动作的时刻。这与分层强化学习中 option 边界的作用相呼应——在那里 credit assignment 的自然单元是子策略而非单个动作——并提示未来的 agentic OPD 方法可能受益于自动化的决策边界检测，而非手工指定的分解。

**统一的安全基础设施。** Safactory（Chen et al., 2026d）代表第五种部署模式：把 OPD 作为闭环安全演化流水线的一个组件。上海 AI Lab 的这一平台整合了用于轨迹生成的并行仿真、用于经验提取的可信数据管理，以及一个把异步 RL 与 on-policy 蒸馏相结合的自主演化模块。其中专门的 OPD 阶段（其报告中的 §5.3）把安全对齐行为从专门的教师智能体蒸馏到一个统一的学生中，得到的模型能在长时程 agentic 任务上保持可信决策。Safactory 值得注意之处在于，它是早期公开记录的基础设施之一，把 OPD 不是作为一次性的训练步骤，而是作为智能体安全持续改进循环中反复出现的组件。

这五种模式反映了业界对待 OPD 方式的一个更广泛转变，即把它当作：质量精炼工具（改进现有流水线的输出）、架构简化工具（把多专家系统坍缩为一个可部署的单一模型）、推理时控制机制（允许单一模型服务多个延迟–准确率工作点）、在传统训练力所不及的能力领域（agentic、多轮、多模态）的赋能者，以及面向自主智能体流水线的安全固化机制。所有五种模式都依赖 on-policy 生成而非静态数据集蒸馏，这与上文综述的分布对齐理论优势相一致。

**通过 privileged 上下文自蒸馏降低 safety tax。** 与上述五种部署模式正交，一条平行的安全对齐工作线针对 **safety tax**，即安全微调可靠地伴随而来的通用能力经验性下降。Fu et al.（2026a）通过逐 token 的 reverse-KL 蒸馏来应对它：从一个 privileged 上下文教师（同一模型，以宪法式安全前言为条件）蒸馏到一个只观察用户提示的无条件学生。该构造借用了 OPSD 的 privileged-information 模板（§5.2.1），并把它应用于安全领域，此时 privileged 上下文是一份固定的宪法，而非问题特定的 oracle。由于学生在规范提示分布下基于自身 rollout 训练，同时从宪法条件化的教师获得稠密的逐 token 监督，安全更新保持在安全相关 token 的局部，基本不触及通用能力表面。所报告的 safety tax 下降与更普遍的模式相符：当 privileged 上下文把目标能力隔离开来时，on-policy 蒸馏能够把目标能力与其附带影响解耦。

### 8.2 新兴领域（Emerging Domains）

**多模态 OPD。** 视觉–语言模型面临不同的 exposure bias 挑战，因为多模态交错生成必须在各模态之间保持连贯性，而高质量的视觉推理数据相比文本十分稀缺。这引出一个问题：on-policy 蒸馏能否**跨**模态边界迁移知识；如果可以，又有哪些结构条件使其可行？

VOLD（Bousselham et al., 2025）提供了证据，表明一个纯文本教师可以在从不观察视觉输入的情况下通过 on-policy 蒸馏引导一个 VLM 学生。教师在学生生成的视觉推理轨迹上提供推理监督，把基于文本的推理结构用作模态无关的脚手架。作者报告，一个 SFT 冷启动阶段对跨模态分布对齐很重要。没有它，学生在视觉推理空间中的 on-policy rollout 与教师基于文本的分布相距太远，蒸馏无法收敛。在 MMMU-Pro 上，VOLD 相对基线模型取得显著提升，这与如下假设相符：推理结构本身（而非感知 grounding）承载了大部分可迁移知识。

这一跨模态原则沿两个轴体现：模型内模态自对齐，以及外部教师的跨模态迁移。CORD（Hu et al., 2026）是模型内自蒸馏的早期实例，其中**同一**模型较强的文本模态充当较弱音频模态的教师。CORD 通过单一模型音频模态与文本模态之间的加权 Reverse KL 将其形式化：

$$
L_{\mathrm{CORD}} = \mathbb{E}_{x\sim D}\mathbb{E}_{\hat{y}\sim \pi_{\mathrm{audio}}(\cdot \mid x)}\left[\sum_t w_t \cdot D_{\mathrm{KL}}\left(\pi_{\mathrm{audio}}(\cdot \mid x, \hat{y}_{<t}) \,\|\, \pi_{\mathrm{text}}(\cdot \mid x, \hat{y}_{<t})\right)\right]
$$

其中音频模态生成 on-policy rollout，文本模态提供 token 级目标，并与 GRPO 结合以实现结果奖励最大化。CORD 报告了很强的语音推理性能，表明跨模态自蒸馏可以在单一模型内把较弱的模态对齐到较强的模态。沿第二个轴，Video-OPD（Li et al., 2026f）处理时序视频定位，即学生必须在视频序列中定位事件，这增加了一个静态图像推理所不具备的时间维度。X-OPD（Cao et al., 2026）通过生成 on-policy rollout 并借助 token 级反馈把它们与教师的评估对齐，把一个单独的纯文本教师蒸馏进一个语音 LLM 学生，从而有效地把教师的能力蒸馏到学生的多模态表征中。

KEPO（Yang et al., 2026b）识别出这些多模态 OPD 方法共有的一个前提条件。当初始解出率接近零时（在医学 VQA 等专门视觉领域中很常见），稀疏 RLVR 往往在任何有意义的 on-policy 数据得以生成之前就因探索坍缩而失败。通过把偏好优化条件化在教师知识之上，KEPO 提供了稠密的 bootstrapping 信号，使后续的 on-policy 训练得以收敛。VISD（Lin et al., 2026a）通过结构化自蒸馏把该范式扩展到视频推理，其中视频感知评判器（video-aware judge）把推理质量分解为多个诊断维度，而一个方向–幅度解耦机制把 RL 的稳定性与细粒度的 OPD credit assignment 结合起来，录得比 RLVR 基线约快 2× 的收敛速度（细节见 §5.2.1）。HyperEyes（Li et al., 2026d）把 OPD 应用于多模态搜索智能体，其中并行工具 
对视觉实体的大量调用带来了一个信用分配挑战，单靠稀疏的结果奖励无法解决。一个双粒度的、效率感知的 RL 框架把宏观层面的轨迹奖励（TRACE，它在训练过程中单调地收紧一个参考工具调用预算）与微观层面的 OPD 结合起来，后者在失败的 rollouts 上注入来自外部教师的稠密 token 级纠正。HyperEyes-30B 在准确率上超过同等规模最强的开源多模态搜索智能体 9.9%，同时平均把工具调用轮数减少 5.3×，这证明 OPD 与面向效率的 RL 可以在多模态智能体设定中协同优化。因此，从 VOLD 经 CORD 到 KEPO、VISD 与 HyperEyes 的这条弧线勾勒出一条完整的流水线：从稠密 bootstrap（KEPO）→ 从纯文本教师进行跨模态迁移（VOLD、X-OPD）→ 模型内自对齐（CORD）→ 面向时序推理的结构化自蒸馏（VISD）→ 面向多模态搜索的效率感知智能体 OPD（HyperEyes）。

**GUI 智能体蒸馏。**虽然 GUI-SD（第 5.2.1 节）通过视觉 PI 处理面向 GUI grounding 的自蒸馏，但为了完成完整的 GUI 任务而从更大的教师蒸馏出紧凑智能体则带来了不同的挑战。GKD 式的标准 OPD 在这一领域举步维艰，因为学生的输出往往是非法的 JSON 结构，教师无法对其做出有意义的评估，而 GUI 任务的多解性质（一条指令对应多条有效动作序列）又使单参考监督产生损失。LiteGUI (Wu et al., 2026b) 通过 Guided On-policy Distillation 同时应对这两点：它向教师提供来自动态检索到的有效动作集的 oracle 参考轨迹，减少教师监督中的幻觉，同时允许教师在上下文中评价学生的输出。在此基础之上，一个 Multi-solution Dual-level GRPO 框架把宏观层面的子任务规划与微观层面的执行匹配联合对齐，改善了长时程 GUI 智能体场景中的探索。使用 Qwen3-VL-32B 作为教师、2B–3B 规模的学生，LiteGUI 在 ScreenSpot-Pro、OS-World 以及新引入的 Lite-Bench 上取得了轻量级 GUI 智能体中报道的强劲性能，同时与远更大的模型保持竞争力。LiteGUI 代表了 on-policy 蒸馏在 GUI 智能体领域的一次系统性整合，用完整的教师—学生迁移范式补充了 GUI-SD 的自蒸馏思路。

**具身智能与物理推理。**走出纯语言领域，OPD 正越来越多地被应用于输出空间连续、且错误在物理上不可逆的设定。这些应用构成一个从高层空间推理、经规划、到低层运动控制的谱系，每一环都要求越来越严格的分布对齐。

在这一谱系的推理一端，HY-Embodied-0.5 (Tencent Robotics X et al., 2026) 把 on-policy 蒸馏用作多阶段训练流水线（RL→RFT→OPD）的最后阶段，把一个 32B 的专家 VLM 压缩为面向空间推理的 2B 具身 MoT 模型，在 22 个 benchmark 中的 16 个上优于同等规模的模型。这里的任务是认知性的（理解空间关系），而非物理性的，OPD 的作用主要是模型压缩，分布错配极小。

走向物理规划，OPD-AV (Afsharrad et al., 2026) 把 GKD 与 5× 压缩应用于 nuScenes 上的自动驾驶，把一个 Qwen3-8B SFT 模型蒸馏为 1.7B 的学生。挑战加剧，因为规划错误会随时间累积。在 $t$ 时刻一个次优的变道决策会约束此后所有的机动，这使得 on-policy 训练很重要：它让学生暴露在自己不完美决策的后果之下。GUI-SD (Zhang et al., 2026i) 处理图形界面中一个类似的累积问题：学生生成点击轨迹，而一个拥有特权信息的自身版本（可访问 bounding box 与高斯软 mask）提供 token 级 KL 监督。它的熵引导蒸馏依据数字位置显著性与教师置信度选择性地对 token 加权，把优化集中在最有影响、最可靠的坐标位置上，同时过滤掉不可靠的监督，并在 6 个 GUI grounding benchmark 上取得强劲结果。

在运动控制一端，VLA-OPD (Zhong et al., 2026b) 在自生成的机器人轨迹上应用 Reverse KL，用一个强教师模型为一个 Vision-Language-Action 学生提供稠密的 token 级监督。由于动作空间是连续的而非离散的，分布错配表现为物理上不安全的轨迹

<!-- p. 64 -->

（碰撞、掉落），而不是单纯的质量下降。学生必须从教师在其自身轨迹上给出的稠密 token 级监督中学会精确的运动控制，其评测报告称样本效率优于纯 RL，鲁棒性优于 offline SFT。

贯穿这一谱系的一个反复出现的观察是：随着任务从认知领域转向物理领域，on-policy 生成似乎变得**更加**重要。有效输出的多样性（多条正确的抓取轨迹、多个有效的时间定位、多种可接受的空间描述）使 off-policy 数据覆盖难以实现，而分布错配的后果也从更低的 benchmark 分数升级为物理上的失败。这一梯度表明，具身 OPD 将越来越需要与环境模拟器更紧密地集成，以提供安全策略迁移所需的稠密 rollout 数据。

**医疗智能体与工具增强的临床推理。**作为新兴领域，临床决策带来了一组不同的挑战。任务在本质上是多轮的（问诊、开检查、解读结果、选择治疗），工具库庞大且异质（医学知识库、影像解读器、药物相互作用检查器），而相对于 episode 长度而言终端奖励是稀疏的。Healthcare AI Gym (Jeong, 2026) 在这一规模上提供了一个兼容 gymnasium 的环境，覆盖 10 个临床领域、3.6K+ 个任务、135 个领域特定工具以及 828K 条医学文献段落，并用它来系统研究 OPD 在工具增强的临床推理中表现如何。除 benchmark 之外，该研究还暴露了一个智能体—文本迁移差距。多轮智能体评测在知识回忆类 benchmark 上引入了系统性开销（MMLU-Med 在 logprob 评测下为 83.8%，在智能体评测下降至 60–66%），表明多轮工具使用是以参数化精度换取检索增强的推理。所提出的 TT-OPD 方法（在第 5.2.1 与 7.2 节讨论过）在 18 个 benchmark 中的 10 个上取得最佳成绩，相对非 RL 基线平均提升 +3.9 个百分点，同时在整个训练过程中维持稳定的多轮结构。综合来看，Healthcare AI Gym 表明高风险的专业领域并不仅仅是**放大版**的智能体环境，而是需要那些能把程序性能力与参数化回忆显式解耦的蒸馏目标，而这一区分在当前单领域智能体 benchmark 中是缺失的。

**蛋白质序列设计。**ProteinOPD (Zhang et al., 2026m) 把 on-policy 蒸馏扩展到蛋白质语言模型（PLMs），其目标是让生成的蛋白质序列与多个相互竞争的性质目标（稳定性、活性、可溶性）对齐，同时不破坏预训练所得的可设计性（designability）。基于标准 RL 的对齐会灾难性地遗忘基本的蛋白质结构，而 SFT 的 mode-covering 目标又限制了新颖性。ProteinOPD 把一个预训练 PLM 适配为偏好特定的教师（每个目标一个），并通过在学生自身轨迹上的 token 级 OPD 把它们的知识蒸馏到一个共享学生中，对齐到一个加权的教师分布归一化几何共识，并在梯度冲突下进行有界优化。虽然多教师 OPD 已在 LLM 设定中被探索过（DeepSeek-V4 的领域专家整合、MAD-OPD 的辩论集成、ORBIT 的模式融合），但 ProteinOPD 的贡献在于冲突教师梯度下的形式化收敛保证；多目标生物优化能从这一性质中获益，因为这里的教师不仅仅是对同一任务的不同观点，而代表真正相互拮抗的目标。ProteinOPD 在目标偏好指标上取得明显增益，同时不损害可设计性，训练速度比基于 RL 的竞争者快 8×。OPD 的 mode-seeking 性质可迁移到生物序列生成。

**硬件描述生成。**RWOPD (Zou et al., 2026) 把 OPD 扩展到形式化硬件验证，其中学生从自然语言规格生成 SystemVerilog Assertions (SVA)，一个开放的性质等价性验证器为每次生成的功能正确性打分。验证器奖励对 forward-KL 蒸馏目标加权，使更接近形式等价的 rollout 获得更强的教师监督，从而形成一个领域——其中奖励信号的精确性（二值的形式化正确性）使验证器反馈与稠密蒸馏得以干净地整合。这一应用体现了一个更广泛的模式。拥有廉价、可靠验证器的领域（形式化验证、编译器检查、单元测试）很适合奖励加权的 OPD，因为该奖励信号 
在很大程度上是确定性的，而这些领域 RL 中常见的稀疏 reward 问题可以通过在高 reward rollout 上的密集教师监督得到缓解。

用于 speculative decoding 的测试时 on-policy 蒸馏（Test-time on-policy distillation for speculative decoding）。Test-Time Speculation (TTS) (Kumar et al., 2026) 处理 speculative decoding 中的一种退化现象：即便最先进的 speculator，其接受长度也会在数千个输出 token 内衰减到 1（即无加速），因为 draft 模型是在短序列上离线训练的，却必须在推理时匹配长得多输出上的目标模型。TTS 把它当作一个在线 on-policy 蒸馏问题，在生成过程中持续让 draft 模型（学生）适配目标模型（教师）。由于 token 验证步骤已经为每个 draft token 调用目标模型，训练信号无需额外成本即可获得。在经过多轮推测后，TTS 在实际生成上下文上收紧 draft 与目标的对齐。在 Qwen-3、Qwen-3.5 与 Llama-3.1 家族上，TTS 相对最先进的 speculator 把接受长度最多提升 72%（平均 41%），增益随生成长度增长。DistillSpec (Zhou et al., 2024) 开创了用离线 OPD 改进 speculative decoding 的 draft 模型（6.3 节），而 TTS 更进一步，在生成过程中在线进行蒸馏，适配实际的推理时分布而非静态训练语料。这代表了在测试时——而非训练时——应用 on-policy 蒸馏：OPD 的分布对齐收益自然延伸到推理时自适应，在那里师生差距持续演化。

隐私保护 OPD（Privacy-preserving OPD）。DP-OPD (Khadem et al., 2026) 在 on-policy 训练期间向学生的梯度注入差分隐私噪声（DP-SGD），同时保持教师冻结。研究发现，on-policy 查询可以提取有用的效用，同时为教师的训练数据提供形式化差分隐私保证，这回应了一个日益增长的关切——随着从专有模型蒸馏成为标准做法。没有这类保证时，on-policy 查询存在记忆并复现教师私有训练样本的风险。

在这些新兴领域中，一个共同的模式反复出现。on-policy 机制本身原封不动地迁移，但“privileged information”、“reward”与“trajectory”的定义必须针对每种模态与任务结构重新协商。这些成功适配的多样性表明，OPD 核心框架是模态无关的，领域特定的设计选择集中在信号接口而非优化循环中。

### 8.3 系统级集成（System-Level Integration）

6.3 节综述的计算优化方法针对的是算法效率，但工业规模的 OPD 还需要跨越整个硬件-软件栈的系统级方案。

生产级 OPD 流水线必须编排三种并发工作负载，即学生 rollout 生成（自回归、内存受限）、教师打分（算力受限的前向传播）与学生梯度更新（算力受限的反向传播）。OpenRLHF (Hu et al., 2024) 与 veRL (Sheng et al., 2024) 之类的框架把它们分解为具有不同并行策略的独立进程组。教师推理使用跨 4–8 个 GPU 的张量并行以获得低延迟，学生生成使用流水线并行以获得吞吐，优化器使用 ZeRO-3 以提高内存效率。主要的工程挑战是同步。由于学生的策略在每个梯度步后都会变化，所有待处理的 rollout 都会变得陈旧。容忍轻微策略陈旧性的异步架构获得更高吞吐，代价是更嘈杂的梯度。高吞吐服务引擎（vLLM (Kwon et al., 2023)、TensorRT-LLM (NVIDIA, 2024)）服务生成阶段，其连续批处理与 PagedAttention 允许在大型 prompt 池上高效地进行 rollout 生成。白盒 OPD 还要求暴露完整词表 logit，而不仅仅是采样到的 token。通信成本相当可观。在 8×H100 上，70B 教师配 7B 学生每批必须传输约 16 GB 的 logit 数据（BF16 下 [B, T,|V|]×2 字节，例如 B=16、T=4096、|V|=128K），在单节点内通过 NVLink (900 GB/s) 是可行的，但多节点设置需要细致的张量切片或 top-k 稀疏化。

<!-- p. 66 -->

DeepSeek-V4 (DeepSeek-AI, 2026) 对万亿参数规模下全词表 OPD 的系统工程给出了详细公开说明，处理从 10 个以上、每个可能包含数万亿参数的教师进行蒸馏的具体挑战。该框架没有为每个教师实体化完整的 |V|>100K logit 张量（在同时从 10 个以上教师蒸馏时这是不可承受的），而是只在中心化缓冲区中缓存最后一层隐状态。它在训练时通过相应的预测头即时重建 logit，实际上消除了内存压力，而重计算开销可忽略。训练样本在每个 mini-batch 内按教师身份排序，因此在任一时刻最多只有一个教师的预测头驻留 GPU 内存。所有权重加载与卸载都异步进行，不阻塞关键路径。该框架在 OPD 阶段本身就集成了 FP4 量化感知训练，量化 MoE 专家权重与注意力 QK 激活，使蒸馏得到的模型从一开始就原生适配部署时的精度，而无需事后量化。

OPD 特有的基础设施需求（OPD-specific infrastructure demands）。与纯 SFT 或 RL 流水线相比，OPD 额外施加三项系统要求，而当前的 RLHF 框架只部分满足它们，即教师共驻（teacher co-hosting）、logit 张量传输（logit-tensor transfer）与陈旧性容忍（staleness tolerance）。教师共驻要求教师在整个训练过程中常驻，而不是用于一次性数据生成，这会使 GPU 内存需求按随教师规模缩放的倍数膨胀。logit 张量传输之所以出现，是因为白盒 OPD 需要传输完整或 top-k 词表分布而非标量 reward，使节点间带宽压力增加数个量级。陈旧性容忍之所以重要，是因为教师对学生 rollout 的打分必须与当前学生策略保持一致，这约束了异步窗口——而基于 reward model 的 RLHF 更容易容忍这一窗口。这些约束解释了为什么被最广泛采用的 OPD 框架（OpenRLHF、veRL、SLIME、verl-pipeline）共享一种架构模式：把 rollout 生成、教师打分与梯度更新分离到带有显式同步栅栏的独立进程组中。一种新兴观点是，万亿参数规模的 OPD 需要框架级支持隐状态缓存、logit 重建与教师感知的批调度，而不是把教师当作训练循环之外的被动数据生成服务。

计算预算分配（Compute budget allocation）。除基础设施架构之外，计算在各训练阶段之间的分配本身就是一个系统级决策。基于 6.3 节的计算分析与近期工业报告（DeepSeek-AI et al., 2025; Yang et al., 2025a）所述的分阶段流水线，可以在计算预算分配中观察到一个反复出现的模式。若干著名工业流水线近似一个三阶段结构：

1. **Off-policy 预热（off-policy warm-up，占用预算的大部分）**：在教师生成的数据上做标准 SFT，提供廉价、高带宽的学习，建立学生的基本能力，并把其分布对齐到足够接近教师，以便进行有效的 on-policy 训练。
2. **On-policy logit 蒸馏（on-policy logit distillation，占 on-policy 份额最大）**：带自适应散度（GKD、DistiLLM、EOPD）的 token 级 OPD 在学生自己的生成轨迹上弥合分布差距，在最需要的地方缓解 exposure bias。
3. **Reward 引导的精炼（reward-guided refinement，最后较小份额）**：RL 增强目标（G-OPD、KDRL、RLAD）在推理任务上把学生推过教师的能力上限，用稀疏结果 reward 发现新的解题路径。

这种分阶段做法把廉价、高带宽的学习前置，并把昂贵的 on-policy 计算留给最后的质量冲刺。Qwen3 (Yang et al., 2025a) 例示了一个相关模式：其四阶段后训练流水线（cold-start SFT → 推理 RL → thinking-mode 融合 SFT → 通用 RL）把监督阶段与 on-policy 阶段交错起来，而从更大模型（32B、235B-A22B）的蒸馏则充当较小变体的主要迁移机制，以大约 1/10

<!-- p. 67 -->

的 GPU 成本达到与 RL 相当的性能。on-policy 训练往往在 off-policy 预热把学生带到接近教师分布之后最为有用，因为这减少了师生差距过大时发生的浪费性 rollout（7.1 节）。

## 9 开放问题与未来方向（Open Problems and Future Directions）

上文综述的方法遵循一条可辨识的轨迹：从固定目标到自适应目标，从外部教师到自生成信号，从单轮生成到多轮轨迹。下面的开放问题正是沿这一方向组织的。

蒸馏缩放定律（Distillation scaling laws）。尽管预训练的参数缩放定律已经确立（Kaplan et al., 2020; Hoffmann et al., 2022），针对蒸馏的缩放研究才刚刚起步。Busbridge et al. (2025) 在 off-policy 设置下提出首批蒸馏缩放定律，揭示出计算预算与最优教师规模之间的非单调关系。随着计算增加，最优教师先增大直到略微超过学生，随后趋于平台，最终反而减小，因为大教师带来的推理成本主导了计算预算。与此同时，最优的学生蒸馏 token 数继续按幂律缩放。DeepSeek-R1 (DeepSeek-AI et al., 2025) 在学生规模轴上提供了互补的 off-policy 证据（7.4 节）：在 AIME 2024 上，学生规模 1.5B→7B→14B→32B 对应性能 28.9%→55.5%→69.7%→72.6%，其中 1.5B 到 7B 之间增益最陡，超过 14B 后回报急剧递减。即便在 off-policy 蒸馏下，学生容量也遵循幂律饱和模式。

然而对于 on-policy 蒸馏，尚不存在等价框架，因为学生 rollout 的生成成本引入了一个 off-policy 设置所没有的额外计算轴。类比这些 off-policy 发现，on-policy 蒸馏损失的一个可能被猜测的函数形式是联合幂律：

$$
L(N_S,N_T,D_{\text{on}})=E+\frac{A}{N_S^\alpha}+\frac{B}{N_T^\beta}+\frac{C}{D_{\text{on}}^\gamma}+f(N_S,N_T)
$$

其中 E 是不可约的任务熵，前三项刻画学生规模、教师规模与 on-policy rollout 预算各自的独立缩放，f(NS, NT) 建模学生与教师之间的容量差干扰。这一形式是推测性的，尚未在 on-policy 设置下得到经验验证。一个带加性交互项的可分离幂律是否刻画了真实的缩放行为仍是开放问题，实际关系可能复杂得多。尽管如此，即便仅作为启发式脚手架，这一猜测形式也有助于组织现有 off-policy 证据所施加的约束。教师规模的非单调发现意味着仅靠 B/N_T^β 是不充分的，交互项 f 必须刻画教师推理成本压倒质量收益的状态。Qwen3 (Yang et al., 2025a) 进一步表明，从多个专门化教师蒸馏比从单个总参数量相当的一体化教师蒸馏有更好的缩放，说明 f 还必须计入教师多样性。一个关键开放问题是：on-policy rollout 项 C/D_on^γ 与教师规模项之间是协同还是拮抗，即更多 on-policy 数据能否补偿更弱的教师，还是教师质量无论 rollout 量多大都构成硬上限。未来工作可以开展控制性网格搜索，在 on-policy 训练下独立变化 NS、NT 与 Don 以厘清这些效应，回答诸如“给定固定的 GPU 预算，投资更大的教师还是更多的学生 rollout 更有效？”这类分配问题。建立此类缩放定律可以使 OPD 的计算分配更接近预训练缩放所拥有的可预测基础。

不确定性感知反馈（Uncertainty-aware feedback）。教师目前提供的是点估计概率。当教师自己对某个 token 不确定时，其 logit 几乎不携带有用信息，但学生却同等对待所有教师信号。一个有前景的下一步是让 OPD 框架纳入教师的认知不确定性（epistemic uncertainty），使学生能够在教师只是猜测的查询上对反馈打折。Process reward model（Lightman et al., 2024）提供了一条潜在路径：对每个推理步骤独立打分，从而精确识别教师监督在哪里可靠、在哪里只是推测。 
与 §7.2 失败模式分析的联系是直接的。flawed prefix trap 本质上是一个"未被识别的教师不确定性"问题，对这份不确定性进行显式建模将给出针对性的解决方案。其思路是：在教师输出熵接近其最大值的位置抑制蒸馏梯度，从而有效地把这些位置视为无信息量。TIP（Xu et al., 2026c）为这一方向提供了经验支持，表明基于教师–学生散度的过滤优于仅基于熵的方案，**相对**不确定性（教师对学生的续写有多惊讶）比教师的绝对置信度更重要。

下一个前沿是**预测性**不确定性。目标不是测量教师在学生当前前缀上的熵，而是估计教师在若干个 token 之后会有多可靠，从而让学生避免投入那些将通往教师无能区域的推理路径。在学生耗费算力去探索这些路径之前就系统性地避开它们，可能带来有意义的效率提升。

**智能体级、持续式与终身蒸馏。** 扩展到多轮 agentic 场景需要新的 credit assignment 机制，因为学生必须在整条交互轨迹上学习长时程工具使用、环境状态跟踪与错误恢复。OpenClaw-RL（Wang et al., 2026i）、Skill-SD（Wang et al., 2026b）、TCOD（Wang et al., 2026d）以及 MAD-OPD 的 OPAD 变体（Wang et al., 2026c）代表了早期的步骤，但教师监督随轨迹深度而系统性退化（Li et al., 2026j）仍是一个具有挑战性的开放问题。MAD-OPD 的步骤级采样策略通过把每个 agentic 步骤当作独立的蒸馏单元、而不是把误差在整条轨迹上传播，提供了部分缓解；而 TCOD 的时间课程则控制每个训练阶段有多少轨迹接受教师监督。这些方法共同表明，多轮蒸馏前沿同时受益于**时间分解**（把轨迹切成可处理的单元）与**渐进式暴露**（逐步提高视野复杂度）。

第一，**环境非平稳性**。agentic 环境会随动作而改变，使得直接的轨迹比较失去意义，除非教师提供以学生实际状态为条件的反事实评估。第二，**工具使用的组合爆炸**。现代智能体框架暴露出数十个带有结构化参数 schema 的工具，这提示蒸馏可能需要在工具调用层面而非单个 token 层面运作，并由 credit assignment 机制把轨迹成功归因到具体的工具选择上。第三，**安全关键型探索**。编码智能体中一次错误的文件写入，或网页浏览智能体中一次误导性的点击，都可能触发不可逆的动作，因此需要安全约束来阻止探索灾难性危险的动程序列。

在另一条轴上，生产模型会经历反复的微调周期（安全补丁、能力更新），这提出一个问题：当教师本身也在演化时，蒸馏目标应如何适应。SDFT（Shenfeld et al., 2026）与 OEL（Ye et al., 2026a）给出了初步答案，但**终身蒸馏**——在不重训练、不灾难性遗忘、也不累积分布漂移的前提下达成重新对齐——联系到持续学习，并多了一层复杂性：教师与学生的分布都是非平稳的。

**效率前沿。** on-policy 训练可观的算力开销仍然是一个障碍。尽管 FOPD（Zhang et al., 2026b）、Lightning-OPD（Wu et al., 2026a）与 SKD（Xu et al., 2025c）已取得进展，on-policy 蒸馏的理论最小成本（即达到给定质量水平所需 on-policy 数据量的信息论下界）仍然未知。一个有前景但探索不足的方向是**选择性教师推理**。MiniPLM（Gu et al., 2025）表明，在 off-policy KD 下，基于教师与参考模型之间的对数概率差异来选择训练实例（"Difference Sampling"）能显著提升数据效率。把这一原则移植到 on-policy rollout 上——用教师–参考散度来确定哪些学生生成的序列值得获得稠密的教师反馈——可以通过跳过对已经收敛的无信息量 rollout 进行昂贵的教师推理，带来显著的算力节省。

<!-- p. 69 -->

**潜空间与跨模态蒸馏。** 当前的 OPD 完全在词表空间中运作，跨架构对齐需要专门的机制（DSKD 的双空间投影、ULD 的概率空间传输）。一种更激进的做法是在**共享潜空间**中蒸馏，完全绕开词表瓶颈，并允许在没有任何结构重叠的、架构上不相似的模型之间进行蒸馏。扩展到多模态场景还带来额外的开放问题。当多个模态必须保持连贯时，exposure bias 如何表现？当教师与学生运作在不同模态空间时，什么样的散度选择是合适的？统一的 $f$-divergence 框架能否扩展以同时处理连续信号空间（图像、音频）与离散 token？VOLD（Bousselham et al., 2025）、X-OPD（Cao et al., 2026）与 CORD（Hu et al., 2026）提供了经验的起点，但跨模态 OPD 的理论基础仍有待建立。

**隐私与评估方法学。** DP-OPD（Khadem et al., 2026）为差分隐私蒸馏提供了一个早期有原则的框架，其非对称设计仅对学生施加 DP-SGD，而保持教师冻结。on-policy 形式很适合 DP 场景，因为在自生成前缀上训练可以减少 DP 噪声带来的累积误差。开放问题包括把 DP-OPD 扩展到具备推理能力的 LLM，以及把 DP 与自蒸馏结合起来。在评估方面，Fu et al.（2026b）证明 token 级 OPD 可能在标准基准上看似成功，却在分布偏移的提示上失败，凸显出基准性能与真正泛化之间的差距。Cha & Cho（2025）的 precision-recall 框架表明蒸馏会在样本精度与分布覆盖之间引入一个深层折衷，因此评估协议需要探查学生学到的是否是底层的因果机制，而不是表面的相关性。

**失败模式的诊断工具。** 随着 OPD 的规模与复杂度上升，§7.2 所诊断的失败模式需要更系统化的诊断工具。当前文献识别出了孤立的陷阱，例如 flawed prefix trap 与 epistemic suppression（Kim et al., 2026b），但该领域缺乏一个在训练过程中检测并量化这些病态的统一框架。能够实时追踪梯度信噪比、表征坍缩与教师–学生散度动态、且不引入完整前向传播成本的诊断探针，将填补这一空白。借鉴预训练中的梯度剖析（gradient profiling）思路，这类工具可以自动显现学生何时进入 Ouroboros 自蒸馏饱和，或何时一个多轮智能体开始遭受轨迹结构侵蚀（trajectory-structure erosion）（Jeong, 2026）。这样的探针将把 OPD 调试从"事后的基准失败分析"转变为"训练中途的主动干预"。

**跨架构可扩展性。** 如 §5.1.1 所探讨的，在异构模型族之间建立映射会引入显著的摩擦。尽管 DSKD（Zhang et al., 2025b）与 ULD（Boizard et al., 2025）通过双空间投影与最优传输给出了初步解决方案，这些方法主要在相对较小的规模差异上得到验证。当容量差距跨越多个数量级时——例如把 400B 的 MoE 教师蒸馏进 1B 的稠密学生——这些映射的可行性仍大体未经验证。在如此极端的情形下，表征瓶颈可能使简单的线性投影或词表级对齐无法奏效。开发能够在不过度增加算力开销的前提下跨越巨大架构鸿沟的非线性、分层对齐机制，仍是一个具有重要实际意义的开放挑战。

**统一蒸馏与强化学习。** 蒸馏与 RL 的融合有三个侧面，最好放在一起理解。第一个是**时间调度**。诸如 G-OPD（Yang et al., 2026e）、KDRL（Xu et al., 2025a）与 RLAD（Zhang et al., 2026o）这样的联合目标确立了把稠密教师指导与稀疏结果奖励结合起来的价值，然而在整个训练生命周期中如何在"模仿教师"与"超越教师去探索"之间分配算力，目前仍是启发式的。对这份分配给出严格的刻画，其条件应以学生的相对能力为依据，因此合适的调度很可能是一个动态策略——当能力差距大时偏向蒸馏，当学生追平教师时转向 RL——而不是一个静态超参数。第二个侧面是**循环本身的结构**。静态蒸馏一旦学生穷尽教师的知识就会饱和，而纯 RL 在奖励稀疏时不稳定，KDRL（Xu et al., 2025a）与 REOPOLD（Ko et al., 2026）通过把稠密 KD 梯度 
用于早期阶段稳定性、用奖励信号用于后期探索。CoPD（Gu et al., 2026）走得最远，用协同演化训练取代"先训练、后蒸馏"的流水线，在其中 RLVR 与双向 OPD 一起运行（§8.1），维持行为上的邻近性，使知识能够被持续吸收而非事后转移。

第三个侧面是这一过程所趋近的**极限**。π-Play（Zhang et al., 2026l）提供了一个早期存在性证明：一个通过自生成课程与自过滤数据不断改进的系统是可能的；但 §7.2 的饱和分析标出了边界，即纯自蒸馏会收敛到模型自身的先验，除非由外部验证或内部生成的 privileged information 加以 grounded。这一天花板能否在没有更强 oracle 的情况下被抬高——实现能力的**创造**而不仅仅是重新分配——是蒸馏、RL 与学习理论交汇处最深的开放问题，其进展很可能取决于那些无需人工标注即可扩展的验证机制。

**与博弈论式 self-play 的联系。** SPIN（Chen et al., 2024）训练更新后的模型去区分其上一轮迭代的生成与人类参考，并在数据分布处具有不动点保证；IRIS（Liao et al., 2026）通过插值式 Rényi 散度（interpolative Rényi divergence）将其推广，把若干 self-play 变体统一到一个由散度参数化的框架下。这些方法以收敛到参考分布上的 Nash 均衡为目标，这一目标在结构上不同于 on-policy 蒸馏核心的分布对齐。然而它们的博弈论结构提供了互补的优势。伴随 Nash 均衡目标而来的收敛保证，有可能稳定上文讨论的迭代式自蒸馏循环，而 OPD 的稠密 token 级监督则可以缓解对抗训练的样本低效。研究把博弈论收敛性质与分布式蒸馏信号相结合的混合框架，是一个有前景的方向。

## 10 结论（Conclusion）

本综述系统性地考察了面向大语言模型的 on-policy 蒸馏，综合了基础形式化、算法创新、理论分析、失败模式诊断与工业部署方面的贡献。我们沿三条核心设计轴而非表层类别来组织文献。**目标函数**设计从固定散度，经自适应路由，发展到抬高教师天花板的 RL 增强型目标。**信号来源**架构从稠密的白盒 logits，经黑盒 API 访问，延伸到完全去除外部教师的自蒸馏。**训练动力学**涵盖 token 加权、课程适应与算力优化。

**关键发现。** 在所综述的方法中，有三种模式反复出现。第一，从 off-policy 到 on-policy 训练分布的转变——从 $y\sim p_{\text{data}}$ 到 $y\sim p_\theta$——在数学推理、代码生成与指令跟随上一直与一致的准确率提升相关联（Tables 3–9 与 2）。这一改进通常被归因于 exposure bias 的降低，因为学生学会从自身错误中恢复，而不是记住理想化的轨迹；对于误差累积更严重的长推理链，性能差距进一步扩大（Gu et al., 2024; Agarwal et al., 2024）。第二，散度选择似乎受益于针对任务**与** token 位置做自适应，而不是全局固定。Reverse KL 往往在 mode-seeking 能防止幻觉的推理任务上表现突出，而 Forward KL 则保留开放式生成常需的多样性。诸如 EOPD（Jin et al., 2026b）与 AOPD（Jia et al., 2026）这类依据局部分布几何来选择散度的自适应方法，代表了当前的前沿。第三，蒸馏与强化学习之间的边界持续收窄。G-OPD（Yang et al., 2026e）、RLAD（Zhang et al., 2026o）与 REOPOLD（Ko et al., 2026）表明，把稠密教师监督与稀疏结果奖励结合起来是有效的，其中 KD 提供梯度稳定性，而 RL 允许超越教师能力的探索。这种融合不仅是概念上的，还产生了一套共享的工具箱（信赖域、优势估计、策略梯度方差削减），供两个社区共同使用。

<!-- p. 71 -->

**实践要点。** 在白盒教师访问下，token 级方法（GKD（Agarwal et al., 2024）、DistiLLM（Ko et al., 2024））在通用指令跟随上往往提供有利的质量–算力折衷。对于推理密集型任务，序列级方法（MiniLLM（Gu et al., 2024））或混合方法（FOPD（Zhang et al., 2026b）、PACED（Xu et al., 2026b）、Lightning-OPD（Wu et al., 2026a））通常更受青睐，尽管 REINFORCE 式估计器天然具有较高的梯度方差（Gu et al., 2024）。在仅有 API 访问时，GAD（Ye et al., 2025）与 OVD（Xiong et al., 2026）分别可以通过对抗式与言语式（verbal）形式化支持 on-policy 学习。自蒸馏方法（OPSD（Zhao et al., 2026b）、SD-ZERO（He et al., 2026））完全免除了对单独教师的需求，使其在资源受限时颇具吸引力。on-policy 生成的可观算力开销通常可以通过投机解码（speculative decoding）（Xu et al., 2025c）、前缀截断（Zhang et al., 2026b）或离线预计算（Wu et al., 2026a）来降低。

**展望。** 从 off-policy 模仿向 on-policy 自我纠正的转变，呼应了机器学习中一个更广泛的模式。经典监督学习假设一个固定的目标分布，而 on-policy 蒸馏把学习者自身的分布当作训练目标的一部分。本文综述的方法逐步放宽了蒸馏最初作为压缩技术所依据的假设。固定目标被自适应目标取代，外部教师被自生成信号取代，单轮优化被多轮轨迹学习取代。这些趋势一个合理的延续是形成这样一种训练范式：模型更持续地生成、评估并更新自身行为，而不仅仅是在一次性的训练运行中完成。蒸馏 scaling law 的缺失（§9）、教师不确定性量化的开放问题，以及无灾难性遗忘的终身适配挑战（Kirkpatrick et al., 2017），似乎是当前最先进水平与这一范式之间的主要障碍之一。随着 LLM 从静态文本生成器演化为更具交互性的推理智能体，本文综述的 on-policy 对齐原则很可能成为 LLM 训练技术栈的标准组成部分。

**局限。** 本综述仅聚焦于训练过程中涉及学生 on-policy 生成的方法。我们有意排除了通用知识蒸馏（例如基于特征的、注意力迁移）、与训练范式正交的模型压缩技术（剪枝、量化），以及仅在推理时使用学生生成数据的方法（self-consistency、多数投票）。我们的覆盖范围截至 2026 年 6 月。鉴于该领域进展迅速（2026 年中期之前每月新增十余篇 OPD 论文），一些同期工作可能未被收录。§3.3 中的方法选择考量反映的是当前的实验证据与硬件约束。随着算力成本下降与训练框架成熟，成本–收益的计算将向更激进的 on-policy 方法倾斜。

**可复现性挑战。** OPD 文献中一个重大隐忧是公平比较的困难。各方法在不同基座模型、不同训练算力预算、不同基准版本（例如 MATH-500 与完整 MATH）、以及每个提示不同 rollout 数量上被评估。很少有论文同时控制所有这些变量，使得仅凭已发表数字进行跨方法比较并不可靠。我们的 Tables 3–9 与 2 报告的是原始论文的结果而非受控复现，读者在解读跨行比较时应保持适当的审慎。我们倡导标准化的 OPD 基准协议，固定基座模型、算力预算与评估套件，类似于 HELM（Liang et al., 2023）为通用 LLM 评估所提供的做法。

**更广泛的影响。** OPD 对 AI 可及性、安全与环境可持续性都有直接影响。

就可及性而言，有效的蒸馏使更小、更高效的模型得以在消费级硬件上运行，从而把原本需要昂贵的前沿模型 API 访问才能获得的先进推理能力民主化。本文综述的自蒸馏方法（OPSD、SD-ZERO）尤为相关，因为它们允许在没有任何外部教师的情况下改进能力，使资源受限的研究者与开发者能够快速迭代。

<!-- p. 72 -->

就安全而言，OPD 既带来风险也带来缓解手段。风险在于，对前沿能力的有效蒸馏降低了滥用的门槛。缓解在于，教师在训练期间保留监督角色，为注入安全约束、对齐目标与能力边界提供了一个天然的切入点。MSD（Qin et al., 2026）具体地实例化了安全感知的 OPD，通过 on-policy 自蒸馏把安全对齐从高资源语言迁移到低资源语言，而无需目标语言的安全数据。通过在教师置信度与学生分歧都高的位置自适应地提高安全关键 token 的权重，MSD 泛化到未见过的语言与更具挑战性的 jailbreak 攻击，表明 OPD 可以充当多语言安全对齐的可扩展机制。在系统层面，Safactory（Chen et al., 2026d）提示 OPD 可以作为持续安全演化循环中反复出现的组件，而非一次性的训练步骤，把蒸馏嵌入到仿真、经验提取与自主演化的闭环流水线之中。随着该领域成熟，我们预期安全感知的 OPD——即教师在交互式训练循环中不仅迁移知识，还强制执行对齐属性——将成为一个重要的研究方向。

就环境可持续性而言，OPD 的算力开销（相对 off-policy SFT 的 4–5×，§6.3）在工业规模上引发担忧。然而由此得到的更小、更有能力的学生可以大幅降低下游推理成本，在分摊到数十亿次推理调用之后，模型生命周期内的净碳足迹可能更低。蒸馏、强化学习与 self-play 更广泛地汇聚为一幅更统一的 on-policy 训练图景，这提示未来的 LLM 可能不再被当作一次成型的产物来训练，而更像通过持续与用户和环境的交互而周期性精炼的系统。

---

## References（原文，未翻译）

References
Amirhossein Afsharrad, Amirhesam Abedsoltan, Ahmadreza Moradipari, and Sanjay Lall.
On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning.
arXiv preprint arXiv:2604.07944, 2026. URLhttps://arxiv.org/abs/2604.07944.
Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu
Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-
Generated Mistakes.Proceedings of ICLR, 2024. URL https://arxiv.org/abs/2306.13649.
Rishabh Agrawal, Jacob Fein-Ashley, and Paria Rashidinejad. Reinforcement Learning from
Rich Feedback with Distributional DAgger.arXiv preprint arXiv:2606.05152, 2026. URL
https://arxiv.org/abs/2606.05152.
Mohammadreza Armandpour, Fatih Ilhan, David Harrison, Ajay Jaiswal, Duc N. M Hoang,
Fartash Faghri, Yizhe Zhang, Minsik Cho, and Mehrdad Farajtabar. Unmasking on-policy
distillation: Where it helps, where it hurts, and why.arXiv preprint arXiv:2605.10889v1,
2026.
Nicolas Boizard, Kevin El Haddad, C´eline Hudelot, and Pierre Colombo. Towards Cross-
Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs.Transactions on
Machine Learning Research, 2025. URLhttps://arxiv.org/abs/2402.12030.
Walid Bousselham, Hilde Kuehne, and Cordelia Schmid. VOLD: Reasoning Transfer
from LLMs to Vision-Language Models via On-Policy Distillation.arXiv preprint
arXiv:2510.23497, 2025. URLhttps://arxiv.org/abs/2510.23497.
Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russ
Webb. Distillation Scaling Laws. InProceedings of the International Conference on Machine
Learning (ICML), 2025. URLhttps://arxiv.org/abs/2502.08606.
Woohyeon Byeon, Jiwon Jeon, Jeonghye Kim, and Youngchul Sung. Be My Tutor: On-
Policy Co-Distillation for Mutual LLM Improvement via Peer Feedback.arXiv preprint
arXiv:2606.14368, 2026. URLhttps://arxiv.org/abs/2606.14368.
72

<!-- p. 73 -->

Yishuo Cai, Jiahui Liu, Yuanxin Liu, Haobo Deng, Linli Yao, Yuhao Zheng, Kun Ouyang,
Zhimo Li, Ziyue Wang, Xu Sun, Haoli Bai, and Xiaohui Li. Thinking Without Im-
ages: Internalizing Visual Manipulation with On-Policy Self-Distillation.arXiv preprint
arXiv:2606.08719, 2026a. URLhttps://arxiv.org/abs/2606.08719.
Yuang Cai and Yuyu Yuan.X -KD: General Experiential Knowledge Distillation for Large
Language Models.arXiv preprint arXiv:2602.12674, 2026. URL https://arxiv.org/abs/
2602.12674.
Yuchen Cai, Ding Cao, Liang Lin, Chunxi Luo, Xin Xu, Kai Yang, Weijie Liu, Saiyong
Yang, Tianxiang Zhao, Guangzhong Sun, Guiquan Liu, and Junfeng Fang. Learning
to foresee: Unveiling the unlocking efficiency of on-policy distillation.arXiv preprint
arXiv:2605.11739v1, 2026b.
Di Cao, Dongjie Fu, Hai Yu, Siqi Zheng, Xu Tan, and Tao Jin. X-OPD: Cross-Modal On-Policy
Distillation for Capability Alignment in Speech LLMs.arXiv preprint arXiv:2603.24596,
2026. URLhttps://arxiv.org/abs/2603.24596.
Sungmin Cha and Kyunghyun Cho. Why Knowledge Distillation Works in Generative
Models: A Minimal Working Explanation.Advances in Neural Information Processing
Systems (NeurIPS), 2025. URLhttps://arxiv.org/abs/2505.13111.
Jingwen Chen, Wenkai Yang, Shengda Fan, Wenbo Nie, Chenxing Sun, Shaodong Zheng,
Yangen Hu, Lu Pan, Ke Zeng, and Yankai Lin. Rethinking Continual Experience Inter-
nalization for Self-Evolving LLM Agents.arXiv preprint arXiv:2606.04703, 2026a. URL
https://arxiv.org/abs/2606.04703.
Tianlei Chen, Jiao Ou, Ziyuan Liu, Ruiming Tang, Jian Liang, and Han Li. Counteraction-
Aware Multi-Teacher On-Policy Distillation for General Capability Recovery with Domain
Preservation.arXiv preprint arXiv:2605.27115, 2026b. URL https://arxiv.org/abs/2605.
27115.
Xianwei Chen, Shimin Zhang, and Jibin Wu. f -OPD: Stabilizing Long-Horizon On-Policy
Distillation with Freshness-Aware Control.arXiv preprint arXiv:2605.17862, 2026c. URL
https://arxiv.org/abs/2605.17862.
Xinquan Chen, Zhenyun Yin, Shan He, Bin Huang, Shanzhe Lei, Pengcheng Shi, Kun Cai,
Bei Chen, Bangwei Liu, Zeyu Kang, Chao Huang, Yang Zhang, Wenjie Li, Ruijun Ge,
Yajie Wang, Tianshun Fang, Tianyang Xu, Yiwen Cong, Meng Jin, Gaolei Li, Xuansheng
Wu, Linhan Liu, Zijing He, An Li, Yan Teng, Xin Tan, Dongrui Liu, Jing Shao, ChaoChao
Lu, Ji He, Jie Li, Chunfeng Song, Jinya Xu, Fan Song, Shujie Wang, Jianmin Qian, Jie
Hou, Xuhong Wang, Yingchun Wang, Hui Wang, and Xia Hu. Safactory: A scalable
agentic infrastructure for training trustworthy autonomous intelligence.arXiv preprint
arXiv:2605.06230v2, 2026d. URLhttps://arxiv.org/abs/2605.06230.
Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-Play Fine-
Tuning Converts Weak Language Models to Strong Language Models.Proceedings of
ICML, 2024. URLhttps://arxiv.org/abs/2401.01335.
Zizhao Chen, Yuying Li, Siting Lin, and Lianxi Wang. Efficient LLM Reasoning via Vari-
ational Posterior Guidance with Efficiency Awareness.arXiv preprint arXiv:2605.11019,
2026e. URLhttps://arxiv.org/abs/2605.11019.
DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelli-
gence. Technical report, DeepSeek, 2026. URL https://huggingface.co/deepseek-ai/
DeepSeek-V4-Pro/blob/main/DeepSeek V4.pdf.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao
Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan
Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu
Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong
73

<!-- p. 74 -->

Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu,
Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong
Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L.
Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin
Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang,
Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun
Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu
Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu
Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu,
Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao,
Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An,
Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie,
Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin,
Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou,
Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao
Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong,
Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo,
Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui
Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren,
Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao,
Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie,
Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang.
DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
arXiv preprint arXiv:2501.12948, 2025. URLhttps://arxiv.org/abs/2501.12948.
Ken Ding. HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation.
arXiv preprint arXiv:2603.23871, 2026. URLhttps://arxiv.org/abs/2603.23871.
Haoyu Dong. Self-Distillation Policy Optimization via Visual Feedback: Bridging Code
and Visual Artifacts.arXiv preprint arXiv:2606.10334, 2026. URL https://arxiv.org/abs/
2606.10334.
Jiabin Fan, Guoqing Luo, Michael Bowling, and Lili Mou. KETCHUP: K-Step Return
Estimation for Sequential Knowledge Distillation.arXiv preprint arXiv:2504.19024, 2025.
URLhttps://arxiv.org/abs/2504.19024.
Junfeng Fang, Zhepei Hong, Mao Zheng, Mingyang Song, Gengsheng Li, Houcheng Jiang,
Dan Zhang, Haiyun Guo, Xiang Wang, and Tat-Seng Chua. Rubric-based On-policy
Distillation.arXiv preprint arXiv:2605.07396, 2026.
Yu Fu, Longxuan Yu, Haz Sameen Shahgir, Zhipeng Wei, Hui Liu, N. Benjamin Erichson,
and Yue Dong. Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-
Distillation.arXiv preprint arXiv:2605.15239, 2026a. URL https://arxiv.org/abs/2605.
15239.
Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and
Dongbin Zhao. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple
Fixes.arXiv preprint arXiv:2603.25562, 2026b. URLhttps://arxiv.org/abs/2603.25562.
Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin,
Surya Bhupatiraju, L ´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre
Ram´e, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos,
Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Pi-
otr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-
Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn,
Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen,
Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot,
Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christo-
pher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar,
74

<!-- p. 75 -->

Dominika Rogozi´nska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica
Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn
Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Pluci ´nska, Harleen Batra, Harsh
Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan,
Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van
Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed,
Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir
Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann,
Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Lu-
ciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin G¨orner, Mat Velloso,
Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal,
Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi
Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta
Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham,
Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona
Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan
Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, S ´ebastien M. R.
Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom,
Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee
Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley,
Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong,
Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran,
Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley,
Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray
Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel,
Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. Gemma 2: Im-
proving Open Language Models at a Practical Size.arXiv preprint arXiv:2408.00118, 2024.
URLhttps://arxiv.org/abs/2408.00118.
Naibin Gu, Chenxu Yang, Qingyi Si, Chuanyu Qin, Dingyu Yao, Peng Fu, Zheng Lin,
Weiping Wang, Nan Duan, and Jiaqi Wang. Co-Evolving Policy Distillation.arXiv preprint
arXiv:2604.27083, 2026. URLhttps://arxiv.org/abs/2604.27083.
Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge Distillation of
Large Language Models.Proceedings of ICLR, 2024. URL https://arxiv.org/abs/2306.
08543.
Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. MiniPLM: Knowledge
Distillation for Pre-Training Language Models.Proceedings of ICLR, 2025. URL https:
//arxiv.org/abs/2410.17215.
Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel,
Sergey Levine, and Dawn Song. The False Promise of Imitating Proprietary LLMs.arXiv
preprint arXiv:2305.15717, 2023. URLhttps://arxiv.org/abs/2305.15717.
Haowei Guo, Baolong Bi, Ruicheng Zhang, Bingqian Sun, and Wentao Zhang. When Should
the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation.arXiv
preprint arXiv:2606.03532, 2026. URLhttps://arxiv.org/abs/2606.03532.
Andy Han, Kristina Fujimoto, Avidan Shah, Kiet Nguyen, Kai Xu, Chen Yueh-Han, Ilia
Sucholutsky, and Rico Angell. On-Policy Consistency Training Improves LLM Safety
with Minimal Capability Degradation.arXiv preprint arXiv:2605.21834, 2026a. URL
https://arxiv.org/abs/2605.21834.
Zihao Han, Tiangang Zhang, Huaibin Wang, and Yilun Sun. Adaptive teacher exposure for
self-distillation in llm reasoning.arXiv preprint arXiv:2605.11458v1, 2026b.
Dongze Hao, Zhiwei Jin, Chen Chen, and Haonan Lu. Stabilizing On-Policy Distillation for
MLLM Reasoning with Global Normalization.arXiv preprint arXiv:2606.09091, 2026. URL
https://arxiv.org/abs/2606.09091.
75

<!-- p. 76 -->

Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam
Fowl, Abhishek Panigrahi, Danqi Chen, and Sanjeev Arora. Self-Distillation Zero: Self-
Revision Turns Binary Rewards into Dense Supervision.arXiv preprint arXiv:2604.12002,
2026. URLhttps://arxiv.org/abs/2604.12002.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.
arXiv preprint arXiv:1503.02531, 2015.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai,
Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark,
Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc,
Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals,
and Laurent Sifre. Training Compute-Optimal Large Language Models. InAdvances in
Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203.
15556.
Wenjin Hou, Shangpin Peng, Weinong Wang, Zheng Ruan, Yue Zhang, Zhenglin Zhou,
Mingqi Gao, Yifei Chen, Kaiqi Wang, Hongming Yang, Chengquan Zhang, Zhuotao Tian,
Han Hu, Yi Yang, Fei Wu, and Hehe Fan. Uni-opd: Unifying on-policy distillation with a
dual-perspective recipe.arXiv preprint arXiv:2605.03677, 2026.
Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexan-
der Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling Step-by-Step!
Outperforming Larger Language Models with Less Training Data and Smaller Model
Sizes.Findings of ACL, 2023. URLhttps://arxiv.org/abs/2305.02301.
Jian Hu, Xibin Tao, Weixun Hou, Quanzhi Liu, Zhengyu Feng, Yang Xu, Qing Ye, Qi Li,
Zenan Li, and Shuai Zhang. OpenRLHF: An easy-to-use, scalable and high-performance
RLHF framework.arXiv preprint arXiv:2405.11143, 2024.
Jing Hu, Danxiang Zhu, Xianlong Luo, Dan Zhang, Shuwei He, Yishu Lei, Haitao Zheng,
Shikun Feng, Jingzhou He, Yu Sun, Hua Wu, and Haifeng Wang. CORD: Bridging
the Audio-Text Reasoning Gap via Weighted On-policy Cross-modal Distillation.arXiv
preprint arXiv:2601.16547, 2026. URLhttps://arxiv.org/abs/2601.16547.
Haiduo Huang, Jiangcheng Song, Yadong Zhang, and Pengju Ren. SelecTKD: Selective
Token-Weighted Knowledge Distillation for LLMs.arXiv preprint arXiv:2510.24021, 2025.
URLhttps://arxiv.org/abs/2510.24021.
Jiazhen Huang, Xiao Chen, Xiao Luo, Yong Dai, Senkang Hu, and Yuzhi Zhao. Skill-
Conditioned Gated Self-Distillation for LLM Reasoning.arXiv preprint arXiv:2605.28791,
2026. URLhttps://arxiv.org/abs/2605.28791.
Jonas H ¨ubotter, Frederike L¨ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel
Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas
Krause. Reinforcement Learning via Self-Distillation.arXiv preprint arXiv:2601.20802,
2026. URLhttps://arxiv.org/abs/2601.20802.
Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggu Lim, and Taesup Kim. Stable On-Policy
Distillation through Adaptive Target Reformulation. InFindings of ACL, 2026. URL
https://arxiv.org/abs/2601.07155.
Minbyul Jeong. Healthcare AI GYM for medical agents.arXiv preprint arXiv:2605.02943,
2026.
Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng,
Xunliang Cai, and Zequn Sun. Asymmetric On-Policy Distillation: Bridging Exploitation
and Imitation at the Token Level.arXiv preprint arXiv:2605.06387, 2026. URL https:
//arxiv.org/abs/2605.06387.
Li Jiang, Haoran Xu, Yichuan Ding, and Amy Zhang. Trajectory-Refined Distillation.arXiv
preprint arXiv:2606.08432, 2026a. URLhttps://arxiv.org/abs/2606.08432.
76

<!-- p. 77 -->

Yuxin Jiang, Chunkit Chan, Mingyang Chen, and Wei Wang. Lion: Adversarial Distillation
of Proprietary Large Language Models.Proceedings of EMNLP, 2023. URL https://arxiv.
org/abs/2305.12870.
Yuxuan Jiang and Francis Ferraro. Bridging Reasoning Trajectories in On-Policy Distillation
via Near-Future Guidance.arXiv preprint arXiv:2606.00305, 2026. URL https://arxiv.
org/abs/2606.00305.
Yuxuan Jiang, Runchao Li, Shubhashis Roy Dipta, Dawei Li, and Zhao Yang. Cornerstones
or stumbling blocks? deciphering the rock tokens in on-policy distillation.arXiv preprint
arXiv:2605.09253v1, 2026b.
Can Jin, Jiakang Li, Rui Wu, Eddy Zhang, and Dimitris N. Metaxas. Weak Critics Make
Strong Learners: On-Policy Critique Distillation for Scalable Oversight.arXiv preprint
arXiv:2606.00424, 2026a. URLhttps://arxiv.org/abs/2606.00424.
Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei,
Nathalie Baracaldo, and Kimin Lee. Entropy-Aware On-Policy Distillation of Language
Models.arXiv preprint arXiv:2603.07079, 2026b. URL https://arxiv.org/abs/2603.07079.
Yiqiao Jin, Yiyang Wang, Lucheng Fu, Yijia Xiao, Yinyi Luo, Haoxin Liu, B. Aditya Prakash,
Josiah Hester, Jindong Wang, and Srijan Kumar. UniSD: Towards a Unified Self-Distillation
Framework for Large Language Models.arXiv preprint arXiv:2605.06597, 2026c.
Jaehun Jung, Hyunwoo Kim, Brandon Cui, Ximing Lu, David Acuna, Prithviraj Am-
manabrolu, and Yejin Choi. DeltaPrompts: Escaping the Zero-Delta Trap in Multimodal
Distillation.arXiv preprint arXiv:2605.15532, 2026. URL https://arxiv.org/abs/2605.
15532.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon
Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural
language models.arXiv preprint arXiv:2001.08361, 2020.
Junlong Ke, Yuxin Huang, Jingdi Lei, Zefan Cai, Xiangyu Peng, and Baobao Chang. Re-
specting self-uncertainty in on-policy self-distillation for efficient LLM reasoning.arXiv
preprint arXiv:2605.13255, 2026.
Fatemeh Khadem, Sajad Mousavi, Yi Fang, and Yuhong Liu. DP-OPD: Differentially Private
On-Policy Distillation for Language Models.arXiv preprint arXiv:2604.04461, 2026. URL
https://arxiv.org/abs/2604.04461.
Gyeongman Kim, Doohyuk Jang, and Eunho Yang. PromptKD: Distilling Student-Friendly
Knowledge for Generative Language Models via Prompt Tuning. InFindings of EMNLP,
2024. URLhttps://arxiv.org/abs/2402.12842.
Jaehoon Kim and Dongha Lee. OPSD Compresses What RLVR Teaches: A Post-RL
Compaction Stage for Reasoning Models.arXiv preprint arXiv:2605.06188, 2026. URL
https://arxiv.org/abs/2605.06188.
Jeonghye Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Rebellious Student: Revers-
ing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR.arXiv preprint
arXiv:2605.10781, 2026a.
Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon,
Dongsheng Li, and Yuqing Yang. Why Does Self-Distillation (Sometimes) Degrade the
Reasoning Capability of LLMs?arXiv preprint arXiv:2603.24472, 2026b. URL https:
//arxiv.org/abs/2603.24472.
Minsang Kim and Seung Jun Baek. Explain in Your Own Words: Improving Reasoning
via Token-Selective Dual Knowledge Distillation.Proceedings of ICLR, 2026. URL https:
//arxiv.org/abs/2603.13260.
77

<!-- p. 78 -->

Yeongmin Kim, Donghyeok Shin, Mina Kang, Byeonghu Na, and Il-Chul Moon. Distillation
of Large Language Models via Concrete Score Matching. InInternational Conference on
Learning Representations (ICLR), 2026c. URLhttps://arxiv.org/abs/2509.25837.
Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. InProceedings
of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1317–1327,
2016.
John Kirchenbauer, Abhimanyu Hans, Brian Bartoldson, Micah Goldblum, Ashwinee
Panda, and Tom Goldstein. Multi-Token Prediction via Self-Distillation.arXiv preprint
arXiv:2602.06019, 2026. URLhttps://arxiv.org/abs/2602.06019.
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins,
Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska,
et al. Overcoming catastrophic forgetting in neural networks.Proceedings of the National
Academy of Sciences, 114(13):3521–3526, 2017.
Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards
Streamlined Distillation for Large Language Models.Proceedings of ICML, 2024. URL
https://arxiv.org/abs/2402.03898.
Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and
Se-Young Yun. DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs. In
Proceedings of ICML, 2025. URLhttps://arxiv.org/abs/2503.07067.
Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling
Reasoning Efficiently via Relaxed On-Policy Distillation.arXiv preprint arXiv:2603.11137,
2026. URLhttps://arxiv.org/abs/2603.11137.
Avinash Kumar, Sujay Sanghavi, and Poulami Das. Test-time speculation.arXiv preprint
arXiv:2605.09329v1, 2026. URLhttps://arxiv.org/abs/2605.09329.
Soo Min Kwon, Ziteng Sun, Ananda Theertha Suresh, Himanshu Jain, and Sanjiv Kumar.
Codistill-grpo: A co-distillation recipe for efficient group relative policy optimization.
arXiv preprint arXiv:2605.08873v1, 2026.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large
language model serving with PagedAttention. InProceedings of the ACM SIGOPS 29th
Symposium on Operating Systems Principles (SOSP), 2023.
Changhao Li, Bohao Yang, Zexian Li, Chenyang Zhao, Jiahao Li, Hongjin Su, Qian Liu, and
Tao Yu. Revisiting DAgger in the era of LLM-agents.arXiv preprint arXiv:2605.12913,
2026a.
Fengxiang Li, Han Zhang, Haoyang Huang, Jinghui Wang, Jinhua Hao, Kun Yuan, Meng-
tong Li, Minglei Zhang, Pengcheng Xu, Wenhao Zhuang, Yizhen Shao, Zongxian Feng,
Can Tang, Chao Wang, Chengxiao Tong, Fan Yang, Gang Xiong, Haixuan Gao, Han
Gao, Hao Wang, Haochen Liu, Hongliang Sun, Jiabao Li, Jingwen Chang, Jun Du,
Junyi Peng, Leizhen Cui, Meimei Jing, Mingqi Wu, Shangpeng Yan, Shaotong Qi, Suzhe
Xu, Wenxuan Zhao, Xianda Sun, Xuan Xie, Yanbo Wang, Yao Xia, Yinghan Cui, Ying-
peng Chen, Yong Wang, Yuze Shi, Zhiwei Shen, Ziyu Wang, Ming Sun, Lin Ye, and Bin
Chen. KAT-Coder-V2 Technical Report.arXiv preprint arXiv:2603.27703, 2026b. URL
https://arxiv.org/abs/2603.27703.
Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan
Zhang, Jinqiao Wang, and Tat-Seng Chua. Unifying Group-Relative and Self-Distillation
Policy Optimization via Sample Routing.arXiv preprint arXiv:2604.02288, 2026c. URL
https://arxiv.org/abs/2604.02288.
Guankai Li, Jiabin Chen, Yi Xu, Xichen Zhang, and Yuan Lu. Hypereyes: Dual-grained
efficiency-aware reinforcement learning for parallel multimodal search agents.arXiv
preprint arXiv:2605.07177v2, 2026d. URLhttps://arxiv.org/abs/2605.07177.
78

<!-- p. 79 -->

Hao Li, Jingkun An, Zijun Song, Pengyu Zhu, Rui Li, Hao Wang, Wendi Feng, Yesheng Liu,
Lijun Li, Jin-Ge Yao, and Lei Sha. SafeSteer: Localized On-Policy Distillation for Efficient
Safety Alignment.arXiv preprint arXiv:2606.02530, 2026e. URL https://arxiv.org/abs/
2606.02530.
Jiaze Li, Hao Yin, Haoran Xu, Boshen Xu, Wenhui Tan, Zewen He, Jianzhong Ju, Zhenbo Luo,
and Jian Luan. Video-OPD: Efficient Post-Training of Multimodal Large Language Models
for Temporal Video Grounding via On-Policy Distillation.arXiv preprint arXiv:2602.02994,
2026f. URLhttps://arxiv.org/abs/2602.02994.
Sijia Li, Jiazuo Yu, Zhoujun Li, Buyun Zhang, Reza Azad, Salman Khan, and Fahad Shahbaz
Khan. GEAR: Granularity-adaptive advantage reweighting for LLM agents via self-
distillation.arXiv preprint arXiv:2605.11853, 2026g.
Xin Li, Hao Jiang, Annan Wang, Yichi Zhang, and Chau Yuen. The extrapolation
cliff in on-policy distillation of near-deterministic structured outputs.arXiv preprint
arXiv:2605.08737v1, 2026h.
Yang Li, Erik Nijkamp, Semih Yavuz, and Shafiq Rayhan Joty. Learning from language
feedback via variational policy distillation.arXiv preprint arXiv:2605.15113, 2026i. URL
https://arxiv.org/abs/2605.15113.
Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu
Yu, Huan ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking On-Policy
Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.arXiv
preprint arXiv:2604.13016, 2026j. URLhttps://arxiv.org/abs/2604.13016.
Yingru Li, Ziniu Li, and Jiacai Liu. A Note on Hybrid Online Reinforcement and Imitation
Learning for LLMs: Formulations and Algorithms.arXiv preprint arXiv:2512.23097, 2025.
URLhttps://arxiv.org/abs/2512.23097.
Yu Li, Rui Miao, Tian Lan, and Zhengling Qi. OPPO: Bayesian Value Recursion for Token-
Level Credit Assignment in LLM Reasoning.arXiv preprint arXiv:2605.21851, 2026k. URL
https://arxiv.org/abs/2605.21851.
Yuduo Li, Xiaofeng Shi, Qian Kou, Longbin Yu, and Hua Zhou. RAFT: Data Refinement and
Adaptive Distillation for Domain Fine-Tuning with Alleviated Forgetting.arXiv preprint
arXiv:2606.00147, 2026l. URLhttps://arxiv.org/abs/2606.00147.
Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Hangjie
Yuan, and Tao Feng. Filter, Then Reweight: Rethinking Optimization Granularity in
On-Policy Distillation.arXiv preprint arXiv:2606.02684, 2026m. URL https://arxiv.org/
abs/2606.02684.
Kun Liang, Clive Bai, Xin Xu, Chenming Tang, Sanwoo Lee, Weijie Liu, Saiyong Yang, and
Yunfang Wu. ORBIT: On-policy Exploration-Exploitation for Controllable Multi-Budget
Reasoning.arXiv preprint arXiv:2601.08310, 2026a. URL https://arxiv.org/abs/2601.
08310.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Ya-
sunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman,
Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning,
Christopher R´e, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus,
Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam,
Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha,
Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael
Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang,
Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Ko-
reeda. Holistic evaluation of language models.Transactions on Machine Learning Research,
2023.
79

<!-- p. 80 -->

Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Xuxin Zhang, Huangyu Dai, and Lingtao
Mao. Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning.
arXiv preprint arXiv:2605.22511, 2026b. URLhttps://arxiv.org/abs/2605.22511.
Wenjie Liao, Like Wu, Liangjie Zhao, Shihui Xu, and Shigeru Fujimura. IRIS: Interpola-
tive R ´enyi Iterative Self-play for Large Language Model Fine-Tuning.arXiv preprint
arXiv:2604.20933, 2026. URLhttps://arxiv.org/abs/2604.20933.
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee,
Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step.
Proceedings of ICLR, 2024. URLhttps://arxiv.org/abs/2305.20050.
Hao Lin, Kunyang Lv, Xu Jiang, Jingqi Tian, Zhongjing Du, Jiayu Ding, Qiaoman Zhang,
and Hongbo Jin. VISD: Enhancing Video Reasoning via Structured Self-Distillation.arXiv
preprint arXiv:2605.06094, 2026a. URLhttps://arxiv.org/abs/2605.06094.
Wenye Lin and Kai Han. Surgical Post-Training: Proximal On-Policy Distillation for
Reasoning with Knowledge Retention.arXiv preprint arXiv:2603.01683, 2026. URL
https://arxiv.org/abs/2603.01683.
Zizhuo Lin, Quanling Liu, Jinsheng Quan, Chao Zhang, Yifan Zhu, Xing Shi, Jingtao Xu,
Zhihui Li, and Yawei Luo. Same Evidence, Different Answers: Canonical-Context On-
Policy Distillation for Multi-Turn Language Models.arXiv preprint arXiv:2605.30251,
2026b. URLhttps://arxiv.org/abs/2605.30251.
Haoran Liu, Yuwei Zhang, Xiyao Li, Bohan Lyu, and Jingbo Shang. HERO: Hindsight-
Enhanced Reflection from Environment Observations for Agentic Self-Distillation.arXiv
preprint arXiv:2606.11559, 2026a. URLhttps://arxiv.org/abs/2606.11559.
Kaiyuan Liu, Ruijie Xu, Yifan Zhang, Silu Mao, Lizhen Qu, Shuming Shi, and Zhi-Hong
Deng. Prefix teach, suffix fade: Local teachability collapse in strong-to-weak on-policy
distillation.arXiv preprint arXiv:2605.13643, 2026b.
Ruiqi Liu, Xiaolei Lv, Gengsheng Li, Ximo Zhu, Zhiheng Wang, Zhengbo Zhang, Junkai
Chen, Zhiheng Li, Bo Li, Jun Gao, and Shu Wu. Visual-Advantage On-Policy Distillation
for Vision-Language Models.arXiv preprint arXiv:2605.21924, 2026c. URL https://arxiv.
org/abs/2605.21924.
Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, and Chaowei Xiao. When Are
Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning.
arXiv preprint arXiv:2605.21606, 2026d. URLhttps://arxiv.org/abs/2605.21606.
Xinyu Liu, Kechen Jiao, Chunyang Xiao, Runsong Zhao, Junhao Ruan, Bei Li, Jiahao Liu,
Qifan Wang, Xin Chen, Jingang Wang, Tong Xiao, and JingBo Zhu. Teacher-guided policy
optimization for LLM distillation.arXiv preprint arXiv:2605.13230, 2026e.
Yanjiang Liu, Jie Lou, Xinyan Guan, Yuqiu Ji, Hongyu Lin, Ben He, Xianpei Han, Le Sun,
Xing Yu, and Yaojie Lu. Your Teacher Can’t Help You Here: Combating Supervision
Fidelity Decay in On-Policy Distillation.arXiv preprint arXiv:2605.30833, 2026f. URL
https://arxiv.org/abs/2605.30833.
Yifeng Liu, Shiyuan Zhang, Yifan Zhang, and Quanquan Gu. Self-Distilled Policy Gradient.
arXiv preprint arXiv:2606.04036, 2026g. URLhttps://arxiv.org/abs/2606.04036.
Yihong Liu, Raoyuan Zhao, Michael A. Hedderich, and Hinrich Sch ¨utze. Crosslingual
on-policy self-distillation for multilingual reasoning.arXiv preprint arXiv:2605.09548v1,
2026h.
Youwei Liu, Jian Wang, Hanlin Wang, and Wenjie Li. COMAP: Co-Evolving World Models
and Agent Policies for LLM Agents.arXiv preprint arXiv:2606.02372, 2026i. URL https:
//arxiv.org/abs/2606.02372.
80

<!-- p. 81 -->

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Ma-
chines Lab: Connectionism, 2025. URL https://www.thinkingmachines.ai/blog/
on-policy-distillation. doi: 10.64434/tml.20251026.
Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang
Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Self-distilled agentic
reinforcement learning.arXiv preprint arXiv:2605.15155, 2026.
Feng Luo, Yu-Neng Chuang, Guanchu Wang, Zicheng Xu, Xiaotian Han, Tianyi Zhang, and
Vladimir Braverman. Demystifying OPD: Length Inflation and Stabilization Strategies
for Large Language Models.arXiv preprint arXiv:2604.08527, 2026a. URL https://arxiv.
org/abs/2604.08527.
Haocheng Luo, Jiahui Liu, Ruicheng Zhang, Zhizhou Zhong, Jiaqi Huang, Zunnan Xu,
Quan Shi, Jun Zhou, and Xiu Li. Learning Visual Spatial Planning from Symbolic State
via Modality-Gap-Aware Self-Distillation.arXiv preprint arXiv:2606.06076, 2026b. URL
https://arxiv.org/abs/2606.06076.
Yuanjie Lyu, Chengyu Wang, Jun Huang, and Tong Xu. From Correction to Mastery:
Reinforced Distillation of Large Language Model Agents.arXiv preprint arXiv:2509.14257,
2025. URLhttps://arxiv.org/abs/2509.14257.
Yufei Ma, Zihan Liang, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin
Zhang, Chenyi Lei, and Wenwu Ou. SD-Search: On-Policy Hindsight Self-Distillation
for Search-Augmented Reasoning.arXiv preprint arXiv:2605.18299, 2026. URL https:
//arxiv.org/abs/2605.18299.
Duy Nguyen, Hanqi Xiao, Archiki Prasad, Zaid Khan, Anirban Das, Austin Zhang, Sambit
Sahu, Hyunji Lee, Elias Stengel-Eskin, and Mohit Bansal. AVSD: Adaptive-View Self-
Distillation by Balancing Consensus and Teacher-Specific Privileged Signals.arXiv preprint
arXiv:2605.20643, 2026. URLhttps://arxiv.org/abs/2605.20643.
Yifan Niu, Han Xiao, Dongyi Liu, Zelong Wang, Dihong Gong, Yasheng Wang, and Jia
Li. Breaking the Tokenizer Barrier: On-Policy Distillation across Model Families.arXiv
preprint arXiv:2606.09456, 2026. URLhttps://arxiv.org/abs/2606.09456.
NVIDIA. TensorRT-LLM: High-performance inference framework for large language models.
https://github.com/NVIDIA/TensorRT-LLM, 2024.
Minjae Oh, Sangjun Song, Gyubin Choi, Yunho Choi, and Yohan Jo. KL for a KL: On-Policy
Distillation with Control Variate Baseline.arXiv preprint arXiv:2605.07865, 2026.
Leyi Pan, Shuchang Tao, Yunpeng Zhai, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei
Liu, and Lijie Wen. RLCSD: Reinforcement Learning with Contrastive On-Policy Self-
Distillation.arXiv preprint arXiv:2606.11709, 2026. URL https://arxiv.org/abs/2606.
11709.
Sangwoo Park, Woongyeong Yeo, Seanie Lee, Yumin Choi, Hyomin Lee, Kangsan Kim,
Jinheon Baek, Seong Joon Oh, and Sung Ju Hwang. It Takes Two: Complementary Self-
Distillation for Contextual Integrity in LLMs.arXiv preprint arXiv:2605.20258, 2026. URL
https://arxiv.org/abs/2605.20258.
Patomporn Payoungkhamdee, Tinnakit Udsa, Jian Gang Ngui, Sarana Nutanong, Al-
ham Fikri Aji, and Peerat Limkonchotiwat. DuDi: Dual-Signal Distillation with Cross-
Lingual Verbalizer.arXiv preprint arXiv:2606.04694, 2026. URL https://arxiv.org/abs/
2606.04694.
Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Char-
lin, and Massimo Caccia. Privileged Information Distillation for Language Models.arXiv
preprint arXiv:2602.04942, 2026. URLhttps://arxiv.org/abs/2602.04942.
81

<!-- p. 82 -->

Jingyu Peng, Maolin Wang, Hengyi Cai, Yuchen Li, Kai Zhang, Shuaiqiang Wang, Dawei
Yin, and Xiangyu Zhao. AdaSwitch: Balancing Exploration and Guidance in Knowledge
Distillation via Adaptive Switching.arXiv preprint arXiv:2510.07842, 2025. URL https:
//arxiv.org/abs/2510.07842.
Daniil Plyusov, Alexey Gorbatovski, Alexey Malakhov, Nikita Balagansky, Boris Sha-
poshnikov, Daria Korotyshova, and Daniil Gavrilov. Trust-Region Behavior Blend-
ing for On-Policy Distillation.arXiv preprint arXiv:2605.31159, 2026. URL https:
//arxiv.org/abs/2605.31159.
Ruiyang Qin, Qingzhuo Wang, Dongrui Liu, Qiang Li, Zhihua Wei, and Wen Shen. Multi-
lingual Safety Alignment via Self-Distillation.arXiv preprint arXiv:2605.02971, 2026. URL
https://arxiv.org/abs/2605.02971.
Miguel Moura Ramos, Duarte M. Alves, and Andr ´e F. T. Martins. Combining on-policy
optimization and distillation for long-context reasoning in large language models.arXiv
preprint arXiv:2605.12227v1, 2026.
Miao Rang, Zhenni Bi, Hang Zhou, Kai Han, Xuechun Wang, An Xiao, Xinghao Chen,
Yunhe Wang, and Hanting Chen. Near-Policy: Accelerating On-Policy Distillation via
Asynchronous Generation and Selective Packing.arXiv preprint arXiv:2605.05940, 2026.
URLhttps://arxiv.org/abs/2605.05940.
MohammadHossein Rezaei, Anas Mahmoud, Zihao Wang, Utkarsh Tyagi, Advait Gosai,
Razvan-Gabriel Dumitru, Aakash Sabharwal, Bing Liu, and Yunzhong He. Rubric-Guided
Self-Distillation: Post-Training Without Rubric Verifiers.arXiv preprint arXiv:2606.12507,
2026. URLhttps://arxiv.org/abs/2606.12507.
St´ephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation
learning and structured prediction to no-regret online learning. InProceedings of the 14th
International Conference on Artificial Intelligence and Statistics, pp. 627–635, 2011.
Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun.
CRISP: Compressed Reasoning via Iterative Self-Policy Distillation.arXiv preprint
arXiv:2603.05433, 2026. URLhttps://arxiv.org/abs/2603.05433.
Guobin Shen, Xiang Cheng, Chenxiao Zhao, Lei Huang, Jindong Li, Dongcheng Zhao, and
Xing Yu. Anti-self-distillation for reasoning rl via pointwise mutual information.arXiv
preprint arXiv:2605.11609v1, 2026a.
Guobin Shen, Lei Huang, Xiang Cheng, Chenxiao Zhao, Jindong Li, Dongcheng Zhao, and
Xing Yu. From generic correlation to input-specific credit in on-policy self distillation.
arXiv preprint arXiv:2605.11613v1, 2026b.
Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo
Han, Sunbowen Lee, and Yi R. Fung. On the Geometry of On-Policy Distillation.arXiv
preprint arXiv:2606.07082, 2026c. URLhttps://arxiv.org/abs/2606.07082.
Idan Shenfeld, Mehul Damani, Jonas H¨ubotter, and Pulkit Agrawal. Self-Distillation Enables
Continual Learning.arXiv preprint arXiv:2601.19897, 2026. URL https://arxiv.org/abs/
2601.19897.
Guangming Sheng, Chi Cao, Zilingfeng Gao, et al. HybridFlow: A flexible and efficient
RLHF framework.arXiv preprint arXiv:2409.19256, 2024.
Aasheesh Singh, Vishal Vaddina, and Dagnachew Birru. ORPO-Distill: Mixed-Policy
Preference Optimization for Cross-Architecture LLM Distillation.arXiv preprint
arXiv:2509.25100, 2025. URLhttps://arxiv.org/abs/2509.25100.
Yuda Song, Lili Chen, Fahim Tajwar, Remi Munos, Deepak Pathak, J. Andrew Bagnell, Aarti
Singh, and Andrea Zanette. Expanding the Capabilities of Reinforcement Learning via
Text Feedback.arXiv preprint arXiv:2602.02482, 2026. URL https://arxiv.org/abs/2602.
02482.
82

<!-- p. 83 -->

Alex Stein, Furong Huang, and Tom Goldstein. GATES: Self-Distillation under Privileged
Context with Consensus Gating.arXiv preprint arXiv:2602.20574, 2026. URL https:
//arxiv.org/abs/2602.20574.
Xingyu Su, Jacob Helwig, Shubham Parashar, Atharv Chagi, Lakshmi Jotsna, Degui Zhi,
James Caverlee, Dileep Kalathil, and Shuiwang Ji. Data-Efficient Autoregressive-to-
Diffusion Language Models via On-Policy Distillation.arXiv preprint arXiv:2606.06712,
2026. URLhttps://arxiv.org/abs/2606.06712.
Jie Sun, Mao Zheng, Mingyang Song, Qiyong Zhong, Yilin Cheng, Bichuan Feng, Pengfei Liu,
Junfeng Fang, and Xiang Wang. SimCT: Recovering Lost Supervision for Cross-Tokenizer
On-Policy Distillation.arXiv preprint arXiv:2605.07711, 2026.
Wenhui Tan, Minghao Li, Xiaoqian Ma, Siqi Fan, Xiusheng Huang, Liujie Zhang, Ruihua
Song, and Weihang Chen. Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient
LLMs.arXiv preprint arXiv:2605.27255, 2026. URLhttps://arxiv.org/abs/2605.27255.
Zhiquan Tan and Yinrong Hong. PAINT: Partial-Solution Adaptive Interpolated Training
for Self-Distilled Reasoners.arXiv preprint arXiv:2604.26573, 2026a. URL https://arxiv.
org/abs/2604.26573.
Zhiquan Tan and Yinrong Hong. Self-Supervised On-Policy Distillation for Reasoning
Language Models.arXiv preprint arXiv:2605.17497, 2026b. URL https://arxiv.org/abs/
2605.17497.
Tencent Robotics X, HY Vision Team, Xumin Yu, Zuyan Liu, Ziyi Wang, He Zhang,
Yongming Rao, Fangfu Liu, Yani Zhang, Ruowen Zhao, Oran Wang, Yves Liang,
Haitao Lin, Minghui Wang, Yubo Dong, Kevin Cheng, Bolin Ni, Rui Huang, Han
Hu, Zhengyou Zhang, Linus, and Shunyu Yao. HY-Embodied-0.5: Embodied Foun-
dation Models for Real-World Agents.arXiv preprint arXiv:2604.07430, 2026. URL
https://arxiv.org/abs/2604.07430.
Yang Tian, Rui Wang, Xumeng Wen, Junjie Li, Shizhao Sun, Lei Song, Jiang Bian, and
Bo Zhao. PBSD: Privileged Bayesian Self-Distillation for Long-Horizon Credit Assignment.
arXiv preprint arXiv:2606.09348, 2026. URLhttps://arxiv.org/abs/2606.09348.
Bing Wang, Shaotian Yan, Chen Shen, kaiyuan liu, Sinan Fan, Ximing Li, Rui Miao, Xiaosong
Yuan, Zhanming Shen, and Jieping Ye. Backtracking When It Strays: Mitigating Dual
Exposure Biases in LLM Reasoning Distillation.arXiv preprint arXiv:2605.19433, 2026a.
URLhttps://arxiv.org/abs/2605.19433.
Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei
Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. Skill-SD: Skill-Conditioned Self-
Distillation for Multi-turn LLM Agents.arXiv preprint arXiv:2604.10674, 2026b. URL
https://arxiv.org/abs/2604.10674.
Jianze Wang, Ying Liu, Jinlong Chen, Xuchun Hu, Qilong Zhang, Yu Cao, Jun Wang,
Hua Yang, Yong Xie, and Qianglong Chen. MAD-OPD: Breaking the Ceiling in On-
Policy Distillation via Multi-Agent Debate.arXiv preprint arXiv:2605.01347, 2026c. URL
https://arxiv.org/abs/2605.01347.
Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, and James Cheng. TCOD: Exploring
Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents.arXiv
preprint arXiv:2604.24005, 2026d. URLhttps://arxiv.org/abs/2604.24005.
Jiaxuan Wang, Xuan Ouyang, Zhiyu Chen, Yulan Hu, Zheng Pan, Xin Li, and Lan-Zhe Guo.
Trace: Distilling where it matters via token-routed self on-policy alignment.arXiv preprint
arXiv:2605.10194v1, 2026e. URLhttps://arxiv.org/abs/2605.10194.
Ke Wang, Yuning Wu, Haoran Liu, Chaoqun Jia, Devin Chen, and Kai Wei. Physics-Guided
Policy Optimization with Self-Distillation.arXiv preprint arXiv:2606.03620, 2026f. URL
https://arxiv.org/abs/2606.03620.
83

<!-- p. 84 -->

Sudong Wang, Weiquan Huang, Xiaomin Yu, Zuhao Yang, Hehai Lin, Keming Wu, Chaojun
Xiao, Chen Chen, Wenxuan Wang, Beier Zhu, Yunjian Zhang, and Chengwei Qin. Beyond
SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL.arXiv
preprint arXiv:2604.28123, 2026g. URLhttps://arxiv.org/abs/2604.28123.
Wenshuo Wang. Knowledge Distillation Must Account for What It Loses.arXiv preprint
arXiv:2604.25110, 2026.
Xun Wang, Ruishuo Chen, Zhuoran Li, Yu Chen, and Longbo Huang. When Context Returns:
Toward Robust Internalization in On-Policy Distillation.arXiv preprint arXiv:2606.11627,
2026h. URLhttps://arxiv.org/abs/2606.11627.
Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. OpenClaw-
RL: Train Any Agent Simply by Talking.arXiv preprint arXiv:2603.10165, 2026i. URL
https://arxiv.org/abs/2603.10165.
Yuanyi Wang, Su Lu, Yanggan Gu, Pengkai Wang, Yifan Yang, Zhaoyi Yan, Congkai Xie,
Jianmin Wu, and Hongxia Yang. Not All Disagreement Is Learnable: Token Teachability
in On-Policy Distillation.arXiv preprint arXiv:2605.26844, 2026j. URL https://arxiv.org/
abs/2605.26844.
Zhenlin Wei, Pu Jian, Yingzhuo Deng, Xiaohan Wang, Jiajun Chai, Zhexin Hu, Wei Lin,
Shanbin Zhang, and Guojun Yin. AMR-SD: Asymmetric Meta-Reflective Self-Distillation
for Token-Level Credit Assignment.arXiv preprint arXiv:2605.18529, 2026. URL https:
//arxiv.org/abs/2605.18529.
Ming Wen, Yuxuan Liu, Kun Yang, Yunhao Feng, Zhuoer Xu, Yuhao Sun, Shiwen Cui, Xiang
Zheng, Xingjun Ma, and Yu-Gang Jiang. Constitutional On-Policy Safe Distillation.arXiv
preprint arXiv:2606.03089, 2026. URLhttps://arxiv.org/abs/2606.03089.
Yuqiao Wen, Zichao Li, Wenyu Du, and Lili Mou. f-Divergence Minimization for Sequence-
Level Knowledge Distillation.Proceedings of ACL, 2023. URL https://arxiv.org/abs/
2307.15190.
Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, and Ngai Wong.
Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language
Models.Proceedings of COLING, 2025. URLhttps://arxiv.org/abs/2404.02657.
Yecheng Wu, Song Han, and Han Cai. Lightning OPD: Efficient Post-Training for Large
Reasoning Models with Offline On-Policy Distillation.arXiv preprint arXiv:2604.13010,
2026a. URLhttps://arxiv.org/abs/2604.13010.
Yubin Wu, Zicheng Cai, Liping Ning, Hua Wang, Zhi Chen, Yaohua Tang, and Hao Chen.
LiteGUI: Distilling Compact GUI Agents with Reinforcement Learning.arXiv preprint
arXiv:2605.07505, 2026b.
Shizhe Xiang, Ke An, Wenlong Yu, Yue Liu, Jian Luan, Pei Fu, and Qilong Wang. Teaching the
Way, Not the Answer: Privileged Tutoring Distillation for Multimodal Policy Optimization.
arXiv preprint arXiv:2606.07000, 2026. URLhttps://arxiv.org/abs/2606.07000.
Xiaomi LLM-Core Team, Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen,
Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, Gang Xie, Hailin
Zhang, Hanglong Lv, Hanyu Li, Heyu Chen, Hongshen Xu, Houbin Zhang, Huaqiu Liu,
Jiangshan Duo, Jianyu Wei, Jiebao Xiao, Jinhao Dong, Jun Shi, Junhao Hu, Kainan Bao,
Kang Zhou, Lei Li, Liang Zhao, Linghao Zhang, Peidian Li, Qianli Chen, Shaohui Liu,
Shihua Yu, Shijie Cao, Shimao Chen, Shouqiu Yu, Shuo Liu, Tianling Zhou, Weijiang Su,
Weikun Wang, Wenhan Ma, Xiangwei Deng, Bohan Mao, Bowen Ye, Can Cai, Chenghua
Wang, Chengxuan Zhu, Chong Ma, Chun Chen, Chunan Li, Dawei Zhu, Deshan Xiao,
Dong Zhang, Duo Zhang, Fangyue Liu, Feiyu Yang, Fengyuan Shi, Guoan Wang, Hao
Tian, Hao Wu, Heng Qu, Hongfei Yi, Hongxu An, Hongyi Guan, Xing Zhang, Yifan Song,
Yihan Yan, Yihao Zhao, Yingchun Lai, Yizhao Gao, Yu Cheng, Yuanyuan Tian, Yudong
Wang, Zhen Tang, Zhengju Tang, Zhengtao Wen, Zhichao Song, Zhixian Zheng, Zihan
84

<!-- p. 85 -->

Jiang, Jian Wen, Jiarui Sun, Jiawei Li, Jinlong Xue, Jun Xia, Kai Fang, Menghang Zhu, Nuo
Chen, Qian Tu, Qihao Zhang, Qiying Wang, Rang Li, Rui Ma, Shaolei Zhang, Shengfan
Wang, Shicheng Li, Shuhao Gu, Shuhuai Ren, Sirui Deng, Tao Guo, Tianyang Lu, Weiji
Zhuang, Weikang Zhang, Weimin Xiong, Wenshan Huang, Wenyu Yang, Xin Zhang, Xing
Yong, Xu Wang, Xueyang Xie, Yilin Jiang, Yixin Yang, Yongzhe He, Yu Tu, Yuanliang Dong,
Yuchen Liu, Yue Ma, Yue Yu, Yuxing Xiang, Zhaojun Huang, Zhenru Lin, Zhipeng Xu,
Zhiyang Chen, Zhonghua Deng, Zihan Zhang, and Zihao Yue. MiMo-V2-Flash Technical
Report.arXiv preprint arXiv:2601.02780, 2026. URLhttps://arxiv.org/abs/2601.02780.
Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, and Hui Xiong. Escaping the KL
Agreement Trap in On-Policy Distillation.arXiv preprint arXiv:2606.09471, 2026. URL
https://arxiv.org/abs/2606.09471.
Xingrun Xing, Haoqing Wang, Boyan Gao, Ziheng Li, and Yehui Tang. Trust Region On-
Policy Distillation.arXiv preprint arXiv:2606.01249, 2026. URL https://arxiv.org/abs/
2606.01249.
Jing Xiong, Hui Shen, Shansan Gong, Yuxin Cheng, Jianghan Shen, Chaofan Tao, Haochen
Tan, Haoli Bai, Lifeng Shang, and Ngai Wong. OVD: On-policy Verbal Distillation.arXiv
preprint arXiv:2601.21968, 2026. URLhttps://arxiv.org/abs/2601.21968.
Haoran Xu, Hongyu Wang, Yifei Gao, Jiaze Li, Xiaofeng Zhang, and Xiaosong Yuan. SG-
OPD: Sign-Gated On-Policy Distillation via Sign-Consistency Gating and Phased Teacher
Sampling.arXiv preprint arXiv:2606.09304, 2026a. URL https://arxiv.org/abs/2606.
09304.
Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang,
Ruifeng Xu, and Fei Mi. KDRL: Post-Training Reasoning LLMs via Unified Knowledge
Distillation and Reinforcement Learning.arXiv preprint arXiv:2506.02208, 2025a. URL
https://arxiv.org/abs/2506.02208.
Shicheng Xu, Liang Pang, Yunchang Zhu, Jia Gu, Zihao Wei, Jingcheng Deng, Feiyang Pan,
Huawei Shen, and Xueqi Cheng. RLKD: Distilling LLMs’ Reasoning via Reinforcement
Learning.arXiv preprint arXiv:2505.16142, 2025b. URL https://arxiv.org/abs/2505.
16142.
Wenda Xu, Rujun Han, Zifeng Wang, Long T. Le, Dhruv Madeka, Lei Li, William Yang Wang,
Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. Speculative Knowledge Distillation:
Bridging the Teacher-Student Gap Through Interleaved Sampling.Proceedings of ICLR,
2025c. URLhttps://arxiv.org/abs/2410.11325.
Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu,
Dacheng Tao, and Tianyi Zhou. A Survey on Knowledge Distillation of Large Language
Models.arXiv preprint arXiv:2402.13116, 2024. URL https://arxiv.org/abs/2402.13116.
Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. PACED: Distillation
and On-Policy Self-Distillation at the Frontier of Student Competence.arXiv preprint
arXiv:2603.11178, 2026b. URLhttps://arxiv.org/abs/2603.11178.
Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard.
TIP: Token Importance in On-Policy Distillation.arXiv preprint arXiv:2604.14084, 2026c.
URLhttps://arxiv.org/abs/2604.14084.
Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard.
Beyond grpo and on-policy distillation: An empirical sparse-to-dense reward principle
for language-model post-training.arXiv preprint arXiv:2605.12483v1, 2026d.
Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue
Zhang. Learning to Reason under Off-Policy Guidance.arXiv preprint arXiv:2504.14945,
2025. URLhttps://arxiv.org/abs/2504.14945.
85

<!-- p. 86 -->

Shaotian Yan, Kaiyuan Liu, Chen Shen, Bing Wang, Sinan Fan, Jun Zhang, Yue Wu, Zheng
Wang, and Jieping Ye. Distribution-Aligned Sequence Distillation for Superior Long-CoT
Reasoning.arXiv preprint arXiv:2601.09088, 2026. URL https://arxiv.org/abs/2601.
09088.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei
Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang,
Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei
Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li,
Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren,
Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang,
Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 Technical Report.arXiv
preprint arXiv:2505.09388, 2025a. URLhttps://arxiv.org/abs/2505.09388.
Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng
Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-Distilled RLVR.arXiv preprint
arXiv:2604.03128, 2026a. URLhttps://arxiv.org/abs/2604.03128.
Fan Yang, Rui Meng, Trudi Di Qi, Ali Ezzati, and Yuxin Wen. KEPO: Knowledge-Enhanced
Preference Optimization for Multimodal Reasoning with Applications to Medical VQA.
arXiv preprint arXiv:2602.00400, 2026b. URLhttps://arxiv.org/abs/2602.00400.
Han Yang, Mingyan Wu, Bailan He, Zeyu Cao, Sikuan Yan, Kevin Qinghong Lin, and
Zifeng Ding. Reasoning compression with mixed-policy distillation.arXiv preprint
arXiv:2605.08776, 2026c.
Ling Yang, Zhaochen Yu, Tianjun Zhang, Minkai Xu, Joseph E. Gonzalez, Bin Cui, and
Shuicheng Yan. SuperCorrect: Advancing Small LLM Reasoning with Thought Template
Distillation and Self-Correction.Proceedings of ICLR, 2025b. URL https://arxiv.org/
abs/2410.09008.
Shenzhi Yang, Guangcheng Zhu, Bowen Song, Haobo Wang, Mingxuan Xia, Xing Zheng,
Yingfan Ma, Zhongqi Chen, Weiqiang Wang, and Gang Chen. OPRD: On-Policy Repre-
sentation Distillation.arXiv preprint arXiv:2606.06021, 2026d. URL https://arxiv.org/
abs/2606.06021.
Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning
beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation.arXiv
preprint arXiv:2602.12125, 2026e. URLhttps://arxiv.org/abs/2602.12125.
Yuxiao Yang, Xiaoyun Wang, and Weitong Zhang. Ogls-sd: On-policy self-distillation with
outcome-guided logit steering for llm reasoning.arXiv preprint arXiv:2605.12400v1, 2026f.
Zhicheng Yang, Zhijiang Guo, Yifan Song, Minrui Xu, Yongxin Wang, Yiwei Wang, Xiaodan
Liang, and Jing Tang. Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-
Horizon Reasoning.arXiv preprint arXiv:2605.07804, 2026g.
Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu
Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander
Bukharin, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nemotron-Cascade 2:
Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation.arXiv
preprint arXiv:2603.19220, 2026h. URLhttps://arxiv.org/abs/2603.19220.
Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-Box
On-Policy Distillation of Large Language Models.arXiv preprint arXiv:2511.10643, 2025.
URLhttps://arxiv.org/abs/2511.10643.
Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Online
Experiential Learning for Language Models.arXiv preprint arXiv:2603.16856, 2026a. URL
https://arxiv.org/abs/2603.16856.
86

<!-- p. 87 -->

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-Policy Context Distillation
for Language Models.arXiv preprint arXiv:2602.12275, 2026b. URL https://arxiv.org/
abs/2602.12275.
Woongyeng Yeo, Yumin Choi, Taekyung Ki, and Sung Ju Hwang. HINT-SD: Targeted
Hindsight Self-Distillation for Long-Horizon Agents.arXiv preprint arXiv:2605.17873,
2026. URLhttps://arxiv.org/abs/2605.17873.
Hee Suk Yoon, Eunseop Yoon, Jaehyun Jang, SooHwan Eom, Ji Woo Hong, Mark Hasegawa-
Johnson, Qi Dai, Chong Luo, and Chang D. Yoo. Decomposed On-Policy Distillation
for Vision-Language Reasoning: Steering Gradients for Visual Grounding.arXiv preprint
arXiv:2606.00564, 2026. URLhttps://arxiv.org/abs/2606.00564.
Guo Yu, Wenlin Liu, Yulan Hu, Hao-Xuan Ma, Jun-Peng Jiang, and Han-Jia Ye. Dense
Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation.
arXiv preprint arXiv:2606.13657, 2026a. URLhttps://arxiv.org/abs/2606.13657.
Weichen Yu, Xiaomin Li, Yizhou Zhao, Xiaoze Liu, Ruowang Zhang, Haixin Wang, Yinyi
Luo, Chen Henry Wu, Gaurav Mittal, Matt Fredrikson, and Yu Hu. Multi-rollout on-policy
distillation via peer successes and failures.arXiv preprint arXiv:2605.12652, 2026b.
Xin Yu, Liuchen Liao, Yiwen Zhang, Yingchen Yu, Lingzhou Xue, and Qinzhen Guo.
Preference-based self-distillation: Beyond KL matching via reward regularization.arXiv
preprint arXiv:2605.05040, 2026c.
Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-
OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation.
arXiv preprint arXiv:2605.18740, 2026. URLhttps://arxiv.org/abs/2605.18740.
Azure Zhang, Ning Gao, Yuqin Dai, Ruiyuan Wu, Jinpeng Wang, Rena Wei Gao, Bingdong
Tan, Shuzheng Gao, Zongjie Li, and Chaozheng Wang. Unlocking Proactivity in Task-
Oriented Dialogue.arXiv preprint arXiv:2605.22240, 2026a. URL https://arxiv.org/abs/
2605.22240.
Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler, Qian Qian,
Gregory D. Lyng, Sanjit Singh Batra, and Robert E. Tillman. Fast and Effective On-
policy Distillation from Reasoning Prefixes.arXiv preprint arXiv:2602.15260, 2026b. URL
https://arxiv.org/abs/2602.15260.
Hongbin Zhang, Chaozheng Wang, Kehai Chen, Youcheng Pan, Yang Xiang, Jinpeng Wang,
and Min Zhang. Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation for
LLM Reasoning.arXiv preprint arXiv:2605.22263, 2026c. URL https://arxiv.org/abs/
2605.22263.
Jiaxin Zhang, Xiangyu Peng, Qinglin Chen, Qinyuan Ye, Caiming Xiong, and Chien-Sheng
Wu. The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy
Distillation.arXiv preprint arXiv:2604.16830, 2026d. URL https://arxiv.org/abs/2604.
16830.
Ke Zhang, Yunjie Tian, DongDi Zhao, Yijiang Li, Yuanye Liu, Vishal M Patel, and Di Fu. On-
policy distillation with best-of-n teacher rollout selection.arXiv preprint arXiv:2605.09725,
2026e.
Miaosen Zhang, Yishan Liu, Shuxia Lin, Xu Yang, Qi Dai, Chong Luo, Weihao Jiang, Peng
Hou, Anxiang Zeng, Xin Geng, and Baining Guo. Towards On-Policy SFT: Distribution
Discriminant Theory and its Applications in LLM Training.arXiv preprint arXiv:2602.12222,
2026f. URLhttps://arxiv.org/abs/2602.12222.
Songming Zhang, Xue Zhang, Tong Zhang, Bojie Hu, Yufeng Chen, and Jinan Xu. AlignDis-
til: Token-Level Language Model Alignment as Adaptive Policy Distillation. InProceedings
of ACL, 2025a. URLhttps://arxiv.org/abs/2503.02832.
87

<!-- p. 88 -->

Wenhao Zhang. Beyond Absolute Imitation: Anchored Residual Guidance for Privileged
On-Policy Distillation.arXiv preprint arXiv:2606.10385, 2026. URL https://arxiv.org/
abs/2606.10385.
Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, and Jingnan
Gu. OPSDL: On-Policy Self-Distillation for Long-Context Language Models.arXiv preprint
arXiv:2604.17535, 2026g. URLhttps://arxiv.org/abs/2604.17535.
XiuYu Zhang, Yi Shan, Junfeng Fang, and Zhenkai Liang. Self-Evaluation Is Already
There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data.arXiv preprint
arXiv:2606.05122, 2026h. URLhttps://arxiv.org/abs/2606.05122.
Xue Zhang, Songming Zhang, Yunlong Liang, Fandong Meng, Yufeng Chen, Jinan Xu, and
Jie Zhou. A Dual-Space Framework for General Knowledge Distillation of Large Language
Models.arXiv preprint arXiv:2504.11426, 2025b. URL https://arxiv.org/abs/2504.11426.
Yan Zhang, Daiqing Wu, Huawen Shen, Yu Zhou, and Can Ma. Learn where to click from
yourself: On-policy self-distillation for gui grounding.arXiv preprint arXiv:2605.00642,
2026i.
Yanfei Zhang, Xu Lin, and Chenglin Wu. StepOPSD: Step-Aware Online Preference Dis-
tillation for Agent Reinforcement Learning.arXiv preprint arXiv:2605.27140, 2026j. URL
https://arxiv.org/abs/2605.27140.
Yaocheng Zhang, Jiajun Chai, Yuqian Fu, Songjun Tu, Xiaohan Wang, Wei Lin, Guojun
Yin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. Are Full Rollouts Necessary for
On-Policy Distillation?arXiv preprint arXiv:2605.31490, 2026k. URL https://arxiv.org/
abs/2605.31490.
Yaocheng Zhang, Yuanheng Zhu, Wenyue Chong, Songjun Tu, Qichao Zhang, Jiajun Chai,
Xiaohan Wang, Wei Lin, Guojun Yin, and Dongbin Zhao. π-Play: Multi-Agent Self-Play
via Privileged Self-Distillation without External Data.arXiv preprint arXiv:2604.14054,
2026l. URLhttps://arxiv.org/abs/2604.14054.
Yulin Zhang, He Cao, Zihao Jiang, Chenyi Zi, Zhipeng Zhou, Zijing Liu, Yu Li, Jia Li, and
Ziqi Gao. Proteinopd: Towards effective and efficient preference alignment for protein
design.arXiv preprint arXiv:2605.10189v1, 2026m. URL https://arxiv.org/abs/2605.
10189.
Yuwei Zhang, Sha Li, Changlong Yu, Qin Lu, Shuowei Jin, Chengyu Dong, Haoran Liu, Ilgee
Hong, Xintong Li, Zhenyu Shi, Bing Yin, and Jingbo Shang. Learning with rare success
but rich feedback via reflection-enhanced self-distillation.arXiv preprint arXiv:2605.12741,
2026n.
Zhaoyang Zhang, Shuli Jiang, Yantao Shen, Yuting Zhang, Dhananjay Ram, Shuo Yang,
Zhuowen Tu, Wei Xia, and Stefano Soatto. Reinforcement-aware Knowledge Distillation
for LLM Reasoning.arXiv preprint arXiv:2602.22495, 2026o. URL https://arxiv.org/
abs/2602.22495.
Hanyang Zhao, Haoxian Chen, Han Lin, Genta Indra Winata, David Yao, and Wenpin
Tang. OPD+: Rethinking the Advantage Design for On-Policy Distillation.arXiv preprint
arXiv:2606.01039, 2026a. URLhttps://arxiv.org/abs/2606.01039.
Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya
Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models.
arXiv preprint arXiv:2601.18734, 2026b. URLhttps://arxiv.org/abs/2601.18734.
Zhengyang Zhao, Lu Ma, and Wentao Zhang. Training with harnesses: On-policy harness
self-distillation for complex reasoning.arXiv preprint arXiv:2605.08741v1, 2026c.
Ziqi Zhao, Xinyu Ma, Liu Yang, Yujie Feng, Daiting Shi, Jingzhou He, Xin Xin, Zhaochun
Ren, and Xiao-Ming Wu. ROSD: Reflective On-Policy Self-Distillation for Language
Model Reasoning across Domains.arXiv preprint arXiv:2605.28014, 2026d. URL https:
//arxiv.org/abs/2605.28014.
88

<!-- p. 89 -->

Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang
Zhu, Ke Zeng, and Xunliang Cai. SCOPE: Signal-Calibrated On-Policy Distillation En-
hancement with Dual-Path Adaptive Weighting.arXiv preprint arXiv:2604.10688, 2026a.
URLhttps://arxiv.org/abs/2604.10688.
Haoyu Zheng, Yun Zhu, Shu Yuan, Shangming Chen, Qing Wang, Wenqiao Zhang, Jun
Xiao, and Yueting Zhuang. MAIGO: Mitigating Lost-in-Conversation with History-
Cleaned On-Policy Self-Distillation.arXiv preprint arXiv:2605.27186, 2026b. URL https:
//arxiv.org/abs/2605.27186.
Qihuang Zhong, Liang Ding, Li Shen, Juhua Liu, Bo Du, and Dacheng Tao. Revisiting
Knowledge Distillation for Autoregressive Language Models.Proceedings of ACL, 2024.
URLhttps://arxiv.org/abs/2402.11890.
Qiyong Zhong, Mao Zheng, Mingyang Song, Xin Lin, Jie Sun, Houcheng Jiang, Xiang Wang,
and Junfeng Fang. SOD: Step-wise On-policy Distillation for Small Language Model
Agents.arXiv preprint arXiv:2605.07725, 2026a.
Zhide Zhong, Haodong Yan, Junfeng Li, Junjie He, Tianran Zhang, and Haoang Li. VLA-
OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-
Policy Distillation.arXiv preprint arXiv:2603.26666, 2026b. URL https://arxiv.org/abs/
2603.26666.
Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Ros-
tamizadeh, Sanjiv Kumar, Jean-Franc ¸ois Kagy, and Rishabh Agarwal. DistillSpec: Im-
proving Speculative Decoding via Knowledge Distillation.Proceedings of ICLR, 2024. URL
https://arxiv.org/abs/2310.08461.
Yucheng Zhou, Wei Tao, Yiwen Guo, and Jianbing Shen. World Models Meet Language
Models: On the Complementarity of Concrete and Abstract Reasoning.arXiv preprint
arXiv:2606.03603, 2026a. URLhttps://arxiv.org/abs/2606.03603.
Yuhang Zhou, Lizhu Zhang, Yifan Wu, Mingyi Wang, Peng Bo, Jiayi Liu, Xiangjun Fan, and
Zhuokai Zhao. OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification.
arXiv preprint arXiv:2606.01476, 2026b. URLhttps://arxiv.org/abs/2606.01476.
Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, and Ge Liu. The many faces of on-policy
distillation: Pitfalls, mechanisms, and fixes.arXiv preprint arXiv:2605.11182v1, 2026.
Zhou Ziheng, Jiaqi Li, Huacong Tang, Ying Nian Wu, and Demetri Terzopoulos. Less is
More: Early Stopping Rollout for On-Policy Distillation.arXiv preprint arXiv:2605.27028,
2026. URLhttps://arxiv.org/abs/2605.27028.
Matthieu Zimmer, Xiaotong Ji, Tu Nguyen, and Haitham Bou Ammar. Rethinking Large
Language Model Distillation: A Constrained Markov Decision Process Perspective.arXiv
preprint arXiv:2509.22921, 2025. URLhttps://arxiv.org/abs/2509.22921.
Qingyun Zou, Yifan Zhang, Bin Lin, and Wei Chen. Reward-weighted on-policy distillation
with an open property-equivalence verifier for NL-to-SVA generation.arXiv preprint
arXiv:2605.13501, 2026.
89
