# 1. 传统强化学习与 SAC 的动因

在深度强化学习中，算法通常分为两大派系：

| 派系                     | 代表算法       | 核心优势                                         | 致命缺陷                                                     |
| :----------------------- | :------------- | :----------------------------------------------- | :----------------------------------------------------------- |
| **On-Policy（同策略）**  | TRPO, PPO, A3C | 训练非常稳定，对超参数不敏感                     | **样本效率极低**。旧数据更新一次就扔，训练复杂的机器人往往需要上千万步交互。 |
| **Off-Policy（异策略）** | DQN, DDPG      | 拥有经验回放池（，数据可重复利用，**样本效率高** | **极其脆弱不稳定**。DDPG 对超参和随机种子极度敏感，在复杂高维连续动作空间极难收敛。 |

## 为什么 DDPG 如此脆弱？
1. **确定性策略的探索缺陷**：DDPG 输出一个确定性的动作 $a = \mu_\theta(s)$，只能靠在动作上强行加一点外挂的高斯噪声或 OU 噪声来探索。这是一种“盲目”的局部抖动，一旦环境奖励稀疏，智能体很容易陷入局部最优死锁。
2. **Q 值严重高估**：基于贝尔曼方程的 $\max$ 操作会导致价值估计被系统性推高，导致“自欺欺人”，产生策略崩溃。

> **SAC 的核心使命**：能否构建一个既具有 **Off-Policy 的极高样本利用率**，又拥有超越 PPO 的**极致稳定性**，还能**自主智能探索**的连续控制算法？
> 答案就是：**将“最大熵”数学原理与“异策略 Actor-Critic”完美融合**。

---

# 2. 最大熵强化学习数学框架

## 2.1 信息熵（Entropy）
在信息论中，一个随机变量 $X$ 的概率分布为 $P(X)$，其**香农熵**定义为：
$$
\mathcal{H}(P) = \mathbb{E}_{x \sim P} [-\log P(x)] = -\int P(x) \log P(x) dx
$$

- 熵衡量的是分布的**不确定性（随机性/混乱度）**。
- 如果分布极其集中（如狄拉克脉冲函数，确定性选择某个动作），熵为极小值；
- 如果分布极其均匀平坦（探索所有可能性），熵达到极大值。

---

## 2.2 有限时间步的最大熵目标函数
传统的 RL 只追求累积奖励期望最大化：$\max_\pi \sum_t \mathbb{E}[r_t]$。SAC 引入了**最大熵强化学习**目标：
$$
J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right] \tag{1}
$$

* **$r(s_t, a_t)$**：环境给予的即时奖励（Exploitation）。
* **$\mathcal{H}(\pi(\cdot | s_t)) = \mathbb{E}_{a_t \sim \pi}[-\log \pi(a_t|s_t)]$**：策略在当前状态下动作分布的熵（Exploration）。
* **$\alpha$（Temperature Parameter)**：控制熵与奖励之间的平衡。
  - 当 $\alpha \to 0$ 时，退化为传统强化学习；
  - $\alpha$ 越大，智能体越喜欢尝试探索各种未知的不同动作。

1. **防止策略过早收敛**：神经网络不会轻易变成确定性的，方差不会骤降，永远保持探索能力。
2. **多模态最优行为捕获**：如果去往终点有“左路”和“右路”两条一模一样通畅的路径，传统 RL 会随机强行收敛到其中一条；而最大熵 RL 会以各 $50\%$ 的概率同时掌握这两种通关方案！
3. **超强的鲁棒性**：面对外界意外扰动时，因为智能体平时探索过多种动作，它能更从容地应对意外状态。

---

## 2.3 无限时间步折现最大熵目标
当时间步 $T \to \infty$ 时，需要引入折现因子 $\gamma \in [0, 1)$。由于策略梯度算法通常不折现状态访问分布 $\rho_\pi$，论文附录 A 给出了严格的无限时间步折现目标定义：

$$
J(\pi) = \sum_{t=0}^\infty \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ \sum_{l=t}^\infty \gamma^{l-t} \mathbb{E}_{s_l \sim p, a_l \sim \pi} \left[ r(s_l, a_l) + \alpha \mathcal{H}(\pi(\cdot | s_l)) \;\Big|\; s_t, a_t \right] \right] \tag{14}
$$
该式明确了从任意状态-动作对 $(s_t, a_t)$ 出发，未来的**未来折现奖励和未来折现熵的总和**都被纳入优化。为了行文简洁，后续推导中作者将 $\alpha$ 吸收到奖励比例中（令 $r \leftarrow r/\alpha$），相当于暂设 $\alpha=1$。

---

# 3. 软策略迭代

在将算法用神经网络拟合之前，作者首先在**表格型**环境下证明：在最大熵框架下，交替进行“评估”与“改进”必然能收敛到全局最优策略。这被称为**软策略迭代**。

## 3.1 软状态价值与软贝尔曼算子

传统贝尔曼方程中，$V(s) = \mathbb{E}_{a}[Q(s, a)]$。而在最大熵框架下，价值中必须加上熵：

$$
V(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ Q(s_t, a_t) - \log \pi(a_t | s_t) \right] \tag{3}
$$

> **注**：因为 $-\log \pi(a_t|s_t)$ 的期望就是熵 $\mathcal{H}$，所以这里实际上就是 $\mathbb{E}[Q] + \mathcal{H}$。由于带上了熵，它被称为**软状态价值函数**。

由此定义**软贝尔曼期望算子（ $\mathcal{T}^\pi$）**：

$$
\mathcal{T}^\pi Q(s_t, a_t) \triangleq r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} \left[ V(s_{t+1}) \right] \tag{2}
$$

---

## 3.2 软策略评估的收敛性

> **Soft Policy Evaluation**：从任意初始函数 $Q^0: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 开始，反复应用算子 $Q^{k+1} = \mathcal{T}^\pi Q^k$，当 $k \to \infty$ 时，序列 $Q^k$ 必收敛到策略 $\pi$ 的真实软 Q 值。

将公式 (3) 代入公式 (2)，定义修正后的“广义奖励”：
$$
r_\pi(s_t, a_t) \triangleq r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} \left[ \mathcal{H}(\pi(\cdot | s_{t+1})) \right]
$$
此时软贝尔曼更新可以改写为标准贝尔曼形式：
$$
Q(s_t, a_t) \leftarrow r_\pi(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p, a_{t+1} \sim \pi} \left[ Q(s_{t+1}, a_{t+1}) \right] \tag{15}
$$
只要动作空间有界（$|\mathcal{A}| < \infty$），熵项就是有界的，修正后的广义奖励 $r_\pi$ 也是有界的。根据标准动态规划理论，由于 $\gamma < 1$，该算子在无穷范数下是一个**压缩映射**，根据巴拿赫不动点定理，必然收敛到唯一不动点 $Q^\pi$。 

---

## 3.3 软策略提升

知道了当前的 $Q^{\pi_{old}}$，如何更新策略产生新的更好策略 $\pi_{new}$？在无约束情况下，最大化期望回报加熵的最优解服从**玻尔兹曼分布（吉布斯分布）**：
$$
\pi(a|s) \propto \exp\left( \frac{1}{\alpha} Q(s, a) \right)
$$
但在连续控制中，我们通常将策略限制在某个容易采样的分布族 $\Pi$ 中（例如高斯分布族）。为了让新策略尽可能逼近理想的能量分布，作者使用 **KL 散度** 来定义策略提升：
$$
\pi_{new} = \arg\min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot | s_t) \;\Big\|\; \frac{\exp\left( Q^{\pi_{old}}(s_t, \cdot) \right)}{Z^{\pi_{old}}(s_t)} \right) \tag{4}
$$


其中：
- $Z^{\pi_{old}}(s_t) = \int \exp\left( Q^{\pi_{old}}(s_t, a) \right) da$ 是**配分函数**，起归一化常数的作用。
- $D_{\text{KL}}(P \parallel Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$ 衡量两个分布的差异。

> [!IMPORTANT]
>
> ### 一、 为什么要让策略 $\pi$ 和 $Q$（的指数）做 $D_{KL}$ 距离？
>
> 你可能会觉得奇怪：$\pi(a|s)$ 是一个概率分布，而 $Q(s, a)$ 是一个价值打分，**为什么把 $Q$ 扔进指数里变成 $\frac{\exp(Q)}{Z}$ 之后，要用 KL 散度去逼近它？**
>
> **这不是硬凑出来的，而是“最大化期望奖励 + 最大化熵”在数学上严格推导出来的必然结果！**
>
> 在最大熵强化学习中，我们在当前状态 $s_t$ 下想要达到的目标是：
> $$
> \max_{\pi} \left[ \mathbb{E}_{a_t \sim \pi}[Q(s_t, a_t)] + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]
> $$
> 为了推导清晰，我们令 $\alpha = 1$（其他值等价于缩放）。根据香农熵的数学定义：$\mathcal{H}(\pi) = \mathbb{E}_{a_t \sim \pi}[-\log \pi(a_t | s_t)]$，原目标变为：
> $$
> \max_{\pi} \mathbb{E}_{a_t \sim \pi} \big[ Q(s_t, a_t) - \log \pi(a_t | s_t) \big]
> $$
> 在优化理论中，**最大化目标**等价于**最小化其相反数**：
> $$
> \min_{\pi} \mathbb{E}_{a_t \sim \pi} \big[ \log \pi(a_t | s_t) - Q(s_t, a_t) \big] \tag {A}
> $$
> 现在，让我们凭空写出 **$\pi$ 与玻尔兹曼分布 $P^*(a|s) = \frac{\exp(Q(s_t, a))}{Z(s_t)}$ 之间的 KL 散度**定义式：
> $$
> D_{\text{KL}}\left( \pi(\cdot | s_t) \;\Big\|\; \frac{\exp(Q(s_t, \cdot))}{Z(s_t)} \right) = \mathbb{E}_{a_t \sim \pi} \left[ \log \left( \frac{\pi(a_t | s_t)}{\frac{\exp(Q(s_t, a_t))}{Z(s_t)}} \right) \right]
> $$
> 利用对数的除法展开：
> $$
> \begin{aligned}
> &= \mathbb{E}_{a_t \sim \pi} \left[ \log \pi(a_t | s_t) - \log \left( \exp(Q(s_t, a_t)) \right) + \log Z(s_t) \right] \\
> &= \mathbb{E}_{a_t \sim \pi} \big[ \log \pi(a_t | s_t) - Q(s_t, a_t) \big] + \mathbb{E}_{a_t \sim \pi} \big[ \log Z(s_t) \big]
> \end{aligned}
> $$
> 因为配分函数 $Z(s_t) = \int \exp(Q(s_t, a)) da$ 只是对动作空间积分后的一个常数，**没有变量 $a_t$**！因此外层的期望 $\mathbb{E}_{a_t \sim \pi}$ 对它没有任何影响，可以直接拿掉：
> $$
> D_{\text{KL}}\left( \pi \;\Big\|\; \frac{\exp(Q)}{Z} \right) = \underbrace{\mathbb{E}_{a_t \sim \pi} \big[ \log \pi(a_t | s_t) - Q(s_t, a_t) \big]}_{\text{这正好就是刚才的 (式 A)！}} + \; \log Z(s_t)
> $$
> 因此
> $$
> \arg\min_\pi D_{\text{KL}}\left( \pi \;\Big\|\; \frac{\exp(Q)}{Z} \right) \;\equiv\; \arg\max_\pi \Big( \mathbb{E}[Q] + \mathcal{H} \Big)
> $$
> **两者在数学上是 $100\%$ 等价的！**之所以写成 KL 散度的形式，是因为它为算法提供了极其优美的**统计物理与信息几何解释**：“提升策略的过程，本质上就是寻找一个策略 $\pi$，使其向理想的物理能量分布（玻尔兹曼分布）做**信息投影**。”
>
> ---
>
> ### 二、 这个大写的 $\Pi$ 是什么意思？
>
> $\Pi$（读作 Capital Pi）在强化学习和泛函分析中，表示**“候选策略族 / 策略假设空间（Policy Class / Policy Family）”**。
>
> #### 为什么要写 $\pi' \in \Pi$？
> 1. **如果不加约束（$\Pi$ 是所有可能的概率分布）**：
>    那么由 KL 散度的非负性质可知，当且仅当两个分布完全一致时，KL 散度达到最小值 $0$。此时最优解就是纯粹的吉布斯分布：
>    $$
>    \pi^*(a|s) = \frac{\exp(Q(s, a))}{Z(s)}
>    $$
>    **但是在连续控制的神经网络中，我们必须限制策略的形式！**在连续空间中，你不可能让神经网络输出任意奇形怪状的无穷维自由分布。我们通常**强行规定**策略必须属于某种可以用有限参数表达的分布族（比如高斯分布族）：
>    $$
>    \Pi = \left\{ \pi_\phi(a|s) = \mathcal{N}\big(\mu_\phi(s), \, \Sigma_\phi(s)\big) \;\Big|\; \phi \in \mathbb{R}^d \right\}
>    $$
> 2. **“投影”的几何含义**：既然理想的最优解 $\frac{\exp(Q)}{Z}$ 可能是一个形状极其不规则的复杂分布，而我们手头只有高斯分布这把“尺子”，那我们该选哪一个高斯分布呢？**公式 (4) 的含义就是：在所有你能用高斯分布表达的策略集合 $\Pi$ 中，挑选出一个跟理想分布在 KL 散度意义下距离最近的高斯分布！** 这在统计学中被称为**信息投影**。
>
> ---
>
> ### 三、 理论上服从吉布斯分布好，但实际上我们用高斯分布？
>
> #### 1. 为什么“理论上服从吉布斯分布更好”？
> 因为吉布斯分布 $\pi(a|s) \propto \exp(Q(s, a))$ 具有**天然的多模态（Multimodal）表达能力**。
> * **举个具象的科研例子**：
>   假设机器人在走廊上跑，正前方有一根柱子。
>   * 最优方案有两个：“往左绕（动作 $a = -1$）”或者“往右绕（动作 $a = +1$）”，这两个动作的 $Q$ 值都很高（比如 $Q = 100$）。
>   * 而“直直往前撞（动作 $a = 0$）”是个灾难，$Q$ 极低（比如 $Q = -100$）。
> * **吉布斯分布的表现**：
>   因为概率正比于 $\exp(Q)$，它在 $a = -1$ 和 $a = +1$ 处会各升起一个高高的概率尖峰，而在中间 $a=0$ 处概率几乎为 0。它完美保留了两种可能的通关解。
> * **高斯分布的致命困境（单模态）**：
>   标准高斯分布只有一个峰值（均值 $\mu$）。如果它想同时兼顾 $-1$ 和 $+1$，它计算出的均值可能刚好落在中间的 $\mu = 0$。结果就是：**高斯策略让机器人以最高的概率直接撞在柱子上！**
>
> #### 2. 那为什么实际工程中，SAC 绝大多数时候还是妥协使用“高斯分布”？
> 既然吉布斯分布这么好，为什么作者在公式 (11) 落地时，依然选用了高斯分布（加一个 $\tanh$ 激活）？因为工程实现的代价不可逾越：
>
> | 维度                              | 理想吉布斯分布 $\propto \exp(Q)$                             | 高斯分布 $\mathcal{N}(\mu, \sigma)$                          |
> | :-------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
> | **如何采样动作 $a$？**            | **极其困难（甚至不可能）**。连续空间无法直接采样，必须用 MCMC、吉布斯采样或 SVGD（朗之万动力学），每走一步要迭代循环几十次，慢得无法接受。 | **极快**。直接利用重参数化：$a = \mu + \sigma \odot \epsilon$，毫秒级完成。 |
> | **计算对数概率 $\log \pi(a\|s)$** | **算不出**。因为分母配分函数 $Z(s) = \int \exp(Q)da$ 在连续空间不可积。算不出概率，你就无法估算具体的熵值。 | **有完美的闭式解析解**，直接代入高斯公式即可求出精确对数概率。 |
> | **端到端求导**                    | 无法直接做低方差的端到端反向传播。                           | 完美契合**重参数化技巧（公式 13）**，梯度流平滑顺畅。        |
>
> 作者在 2017 年的前作 **Soft Q-Learning ** 中，就尝试过使用复杂的采样网络去强行拟合多模态的吉布斯分布，但结果发现算法极度复杂且容易崩溃。因此在 2018 年这篇 SAC 中，作者退后一步：**“承认高斯分布是局部单峰的，但通过高斯加 Tanh，结合双 Q 截断，换取绝对的数值稳定性与闪电般的训练速度”**。
>
> > **最新科研脉络**：
> > 直到 2023 年以后的机器人模仿学习与强化学习中，学术界用 **扩散模型（Diffusion Policy）** 重新替代了高斯策略。扩散模型正是一个既能输出复杂的任意多模态分布（类似吉布斯分布），又具有稳定数学采样的工具，成功圆了当年 SAC 无法使用复杂多峰策略的学术构想！

---

## 3.4 软策略单调提升证明

> 设 $\pi_{new}$ 是公式 (4) 的最优化解，则对所有状态-动作对，必然有：
> $$
> Q^{\pi_{new}}(s_t, a_t) \ge Q^{\pi_{old}}(s_t, a_t)
> $$

根据公式 (4)，新策略的目标定义为：
$$
\pi_{new}(\cdot | s_t) = \arg\min_{\pi' \in \Pi} D_{\text{KL}}\left( \pi'(\cdot | s_t) \;\Big\|\; \exp\left(Q^{\pi_{old}}(s_t, \cdot) - \log Z^{\pi_{old}}(s_t)\right) \right) \triangleq \arg\min_{\pi' \in \Pi} J_{\pi_{old}}(\pi'(\cdot | s_t)) \tag{16}
$$
因为 $\pi_{new}$ 是使上述目标最小化的最优解，所以它算出的目标值必定小于等于选旧策略 $\pi_{old}$ 时的目标值：
$$
J_{\pi_{old}}(\pi_{new}(\cdot | s_t)) \le J_{\pi_{old}}(\pi_{old}(\cdot | s_t))
$$
将 KL 散度展开写出（根据期望定义）：
$$
\mathbb{E}_{a_t \sim \pi_{new}} \left[ \log \pi_{new}(a_t | s_t) - Q^{\pi_{old}}(s_t, a_t) + \log Z^{\pi_{old}}(s_t) \right] \le \mathbb{E}_{a_t \sim \pi_{old}} \left[ \log \pi_{old}(a_t | s_t) - Q^{\pi_{old}}(s_t, a_t) + \log Z^{\pi_{old}}(s_t) \right] \tag{17}
$$
注意！$\log Z^{\pi_{old}}(s_t)$ **只依赖于状态 $s_t$，不依赖于动作 $a_t$**！因此不等式两边的 $\log Z$ 完全相同，可以直接消去，移项整理得到：
$$
\mathbb{E}_{a_t \sim \pi_{new}} \left[ Q^{\pi_{old}}(s_t, a_t) - \log \pi_{new}(a_t | s_t) \right] \ge \mathbb{E}_{a_t \sim \pi_{old}} \left[ Q^{\pi_{old}}(s_t, a_t) - \log \pi_{old}(a_t | s_t) \right] = V^{\pi_{old}}(s_t) \tag{18}
$$
公式 (18) 说明：**在旧的 Q 函数评价下，新策略的综合软价值已经全面超越了旧的软状态价值 $V^{\pi_{old}}$！**最后一步，利用软贝尔曼方程反复展开：

$$
\begin{aligned}
Q^{\pi_{old}}(s_t, a_t) &= r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}} \left[ V^{\pi_{old}}(s_{t+1}) \right] \\
&\le r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}} \left[ \mathbb{E}_{a_{t+1} \sim \pi_{new}} \left[ Q^{\pi_{old}}(s_{t+1}, a_{t+1}) - \log \pi_{new}(a_{t+1} | s_{t+1}) \right] \right] \quad (\text{代入公式 18}) \\
&\le \dots \quad (\text{无限递推代入下去}) \\
&\le Q^{\pi_{new}}(s_t, a_t) 
\end{aligned} \tag{19}
$$
至此，严格证明了 $Q^{\pi_{new}} \ge Q^{\pi_{old}}$ 单调递增。

---

# 4. Soft Actor-Critic 体系架构
理论证明了在表格型环境下可以收敛，但面对连续高维状态空间，必须用**深度神经网络**对函数进行参数化拟合。

## 4.1 系统的 5 个神经网络定义
在SAC 中，维护了 **5 套神经网络权重**：
1. **$V_\psi(s)$**：状态价值网络（State-Value Network），参数为 $\psi$；
2. **$V_{\bar{\psi}}(s)$**：目标状态价值网络（Target Value Network），参数为 $\bar{\psi}$（延迟平滑更新，用于稳定贝尔曼目标）；
3. **$Q_{\theta_1}(s, a)$**：第一个软 Q 网络，参数为 $\theta_1$；
4. **$Q_{\theta_2}(s, a)$**：第二个软 Q 网络，参数为 $\theta_2$（Twin Q 机制，用于缓解高估）；
5. **$\pi_\phi(a|s)$**：策略网络（Actor），参数为 $\phi$。

---

## 4.2 软状态价值网络 $V_\psi$ 的训练

理论上，由公式 (3) 可知 $V(s) = \mathbb{E}_a[Q(s, a) - \log \pi(a|s)]$，实际可以直接用 Q 和 $\pi$ 估算。但在实践中，单独设一个网络 $V_\psi$ 能极大平滑训练动态。目标是让 $V_\psi(s_t)$ 逼近右侧的软价值，构建**均方误差损失**：

$$
J_V(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \frac{1}{2} \left( V_\psi(s_t) - \mathbb{E}_{a_t \sim \pi_\phi} \left[ Q_\theta(s_t, a_t) - \log \pi_\phi(a_t | s_t) \right] \right)^2 \right] \tag{5}
$$

* $\mathcal{D}$ 是**经验回放池（Replay Buffer）**。注意：状态 $s_t$ 来自离线数据池 $\mathcal{D}$，但动作 $a_t$ 是**由当前策略 $\pi_\phi$ 最新采样出来的**！对其求梯度，得到无偏梯度估计器：

$$
\hat{\nabla}_\psi J_V(\psi) = \nabla_\psi V_\psi(s_t) \left( V_\psi(s_t) - Q_\theta(s_t, a_t) + \log \pi_\phi(a_t | s_t) \right) \tag{6}
$$

---

## 4.3 软 Q 网络 $Q_\theta$ 的训练

Q 网络的训练基于标准的**时序差分（TD）贝尔曼残差最小化**：

$$
J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_\theta(s_t, a_t) - \hat{Q}(s_t, a_t) \right)^2 \right] \tag{7}
$$
其中目标 Q 值（Target Value）$\hat{Q}$ 采用**目标价值网络 $V_{\bar{\psi}}$** 计算：

$$
\hat{Q}(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} \left[ V_{\bar{\psi}}(s_{t+1}) \right] \tag{8}
$$
其随机梯度更新式为：

$$
\hat{\nabla}_\theta J_Q(\theta) = \nabla_\theta Q_\theta(s_t, a_t) \left( Q_\theta(s_t, a_t) - r(s_t, a_t) - \gamma V_{\bar{\psi}}(s_{t+1}) \right) \tag{9}
$$
受到 TD3 论文的启发，作者同时训练了两个独立的 Q 网络 $Q_{\theta_1}$ 和 $Q_{\theta_2}$。在计算公式 (6) 的 $V$ 网络更新和后续策略提升时，**永远取两个 Q 网络的最小值**：
$$
Q_\theta(s_t, a_t) \triangleq \min\left( Q_{\theta_1}(s_t, a_t), \; Q_{\theta_2}(s_t, a_t) \right)
$$
这一“悲观保守”设计彻底解决了 Q 学习中的价值虚高问题。

---

# 5. 策略网络优化

## 5.1 策略损失目标
根据公式 (4)，我们希望策略逼近玻尔兹曼分布，在采样状态分布下的期望损失为：

$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ D_{\text{KL}}\left( \pi_\phi(\cdot | s_t) \;\Big\|\; \frac{\exp(Q_\theta(s_t, \cdot))}{Z_\theta(s_t)} \right) \right] \tag{10}
$$
我们尝试将 KL 散度在数学上完整展开：
$$
\begin{aligned}
D_{\text{KL}}\left( \pi_\phi \;\Big\|\; \frac{\exp(Q_\theta)}{Z_\theta} \right) &= \mathbb{E}_{a_t \sim \pi_\phi} \left[ \log \pi_\phi(a_t | s_t) - \log \left( \frac{\exp(Q_\theta(s_t, a_t))}{Z_\theta(s_t)} \right) \right] \\
&= \mathbb{E}_{a_t \sim \pi_\phi} \left[ \log \pi_\phi(a_t | s_t) - Q_\theta(s_t, a_t) + \log Z_\theta(s_t) \right]
\end{aligned}
$$
配分函数 $Z_\theta(s_t)$ 与策略参数 $\phi$ 毫无关系！所以对 $\phi$ 求导时，$\nabla_\phi \log Z_\theta(s_t) = 0$。因此 $Z$ 可以直接扔掉！

---

## 5.2 重参数化技巧
观察目标项：
$$
\mathbb{E}_{a_t \sim \pi_\phi} \left[ \log \pi_\phi(a_t | s_t) - Q_\theta(s_t, a_t) \right]
$$
注意：**期望操作下标依赖于待求导参数 $\phi$（$a_t \sim \pi_\phi$）！**

- 如果直接用策略梯度定理：$\nabla_\phi \mathbb{E}_{a \sim \pi_\phi}[f(a)] = \mathbb{E}[\nabla_\phi \log \pi_\phi \cdot f(a)]$，其**方差极其巨大**，训练极其缓慢。
- **重参数化技巧（Pathwise Derivative）**：将随机采样与网络参数**解耦**！引入一个独立于网络参数的基础随机噪声 $\epsilon_t \sim \mathcal{N}(0, I)$，让动作变成确定性神经网络变换：

$$
a_t = f_\phi(\epsilon_t; s_t) \tag{11}
$$

例如，对于高斯策略，网络输出均值 $\mu_\phi(s_t)$ 和标准差 $\sigma_\phi(s_t)$，则动作表示为：
$$
a_t = \mu_\phi(s_t) + \sigma_\phi(s_t) \odot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$
将公式 (11) 代入公式 (10) 并丢弃 $Z$，期望的随机性完全转移到了外层的无参数噪声分布 $\epsilon_t$ 上：

$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}, \, \epsilon_t \sim \mathcal{N}} \left[ \log \pi_\phi\left( f_\phi(\epsilon_t; s_t) \,\big|\, s_t \right) - Q_\theta\left( s_t, f_\phi(\epsilon_t; s_t) \right) \right] \tag{12}
$$
现在，我们可以直接把求导符号 $\nabla_\phi$ 放入期望内部！我们要求导的目标函数形式为：
$$
g(\phi) = \log \pi_\phi(a_t | s_t) - Q_\theta(s_t, a_t), \quad \text{其中 } a_t = f_\phi(\epsilon_t; s_t)
$$
注意，**$\phi$ 出现在了两个地方**：

1. $\log \pi_\phi$ 函数本身的显式参数 $\phi$；
2. 动作 $a_t = f_\phi(\epsilon_t; s_t)$ 内部隐式依赖 $\phi$。

根据多元微积分的**全导数公式**：
$$
\frac{d}{d\phi} \Big[ \log \pi_\phi(a_t | s_t) \Big] = \underbrace{\nabla_\phi \log \pi_\phi(a_t | s_t)}_{\text{对网络显式参数求偏导}} + \underbrace{\nabla_{a_t} \log \pi_\phi(a_t | s_t) \cdot \nabla_\phi f_\phi(\epsilon_t; s_t)}_{\text{通过动作输出层传导的链式偏导}}
$$
而第二项对 $Q$ 求导，由于 $Q_\theta$ 不显式依赖 $\phi$，导数完全通过动作 $a_t$ 链式传导：
$$
\frac{d}{d\phi} \Big[ -Q_\theta(s_t, a_t) \Big] = -\nabla_{a_t} Q(s_t, a_t) \cdot \nabla_\phi f_\phi(\epsilon_t; s_t)
$$
将上述结果合并提取公因式 $\nabla_\phi f_\phi(\epsilon_t; s_t)$，便得到了论文中的公式 (13)：

$$
\hat{\nabla}_\phi J_\pi(\phi) = \nabla_\phi \log \pi_\phi(a_t | s_t) + \left( \nabla_{a_t} \log \pi_\phi(a_t | s_t) - \nabla_{a_t} Q(s_t, a_t) \right) \nabla_\phi f_\phi(\epsilon_t; s_t) \tag{13}
$$

* 这个梯度估计是**完全无偏且低方差的**！
* 它把 DDPG 式的确定性策略梯度（通过 $\nabla_a Q$ 寻找 Q 值最大方向）优雅地泛化到了任意可导的随机分布上。

> [!IMPORTANT]
>
> ### 三、 为什么说 SAC 把 DDPG 的“确定性策略梯度”优雅泛化到了“任意随机分布”上？
>
> DDPG只能用于**确定性策略**。策略网络直接输出一个确定的动作向量：
> $$
> a = \mu_\phi(s)
> $$
> DDPG 的目标是最大化 $Q(s, \mu_\phi(s))$。对参数 $\phi$ 求导时，直接使用标准的微积分链式法则：
> $$
> \nabla_\phi J_{\text{DDPG}}(\phi) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_a Q(s, a)\big|_{a=\mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s) \right]
> $$
>
> - **结构极简**：**$(\text{Q对动作的导数}) \times (\text{动作对网络参数的导数})$**。
> - **但 DDPG 无法处理随机策略**：一旦策略是随机分布，动作就有了概率不确定性，DDPG 的这套链式法则在数学上就断裂了，无法直接穿透概率采样。
>
> SAC 想要一个**随机策略**（为了最大熵和探索），输出的是均值和方差。为了打破随机采样不可导的死局，SAC 引入了重参数化：把随机性剥离给外界噪声 $\epsilon \sim \mathcal{N}$，动作变成确定性变换 $a = f_\phi(\epsilon; s)$。此时，我们来看公式 (13)（暂时忽略熵正则项，只看 Q 值部分）：
> $$
> \hat{\nabla}_\phi J_{\text{SAC}}(\phi) \approx \mathbb{E}_{\epsilon} \left[ \nabla_a Q(s, a) \cdot \nabla_\phi f_\phi(\epsilon; s) \right]
> $$
> 它长得和 DDPG 几乎一模一样：
> * **DDPG**：让动作输出 $\mu_\phi(s)$ 沿着 Q 增大的方向走；
> * **SAC**：让采样映射函数 $f_\phi(\epsilon; s)$ 沿着 Q 增大的方向走！
>
> 1. **形式上的完全包容（特例与泛化）**：
>    如果我们在 SAC 中强行让方差 $\sigma \to 0$（不再有随机噪声，$\epsilon$ 失效），并且温度 $\alpha \to 0$（去掉熵项），那么 $f_\phi(\epsilon; s)$ 就彻底退化成了确定性函数 $\mu_\phi(s)$。**此时 SAC 的梯度公式瞬间 $100\%$ 精确还原为 DDPG 的确定性策略梯度！**
> 2. **突破了确定性限制**：
>    DDPG 必须要求策略是单一的确定性点，导致它探索脆弱；而 SAC 借助重参数化，**既完整保留了 DDPG 这种直接利用 $\nabla_a Q$ 的“低方差、高效率”传导优势，又将其成功赋能给了一个完整的连续概率分布族（带有均值、方差和熵）**！
>
> ---
>
> ### 设定一个问题
>
> * **智能体参数**：$\phi$（我们想要更新的网络权重）。
> * **策略均值**：$\mu_\phi = 2\phi$（网络输出的均值）。
> * **真实环境的评价网络（Critic）**：
>   $$
>   Q(a) = -(a - 3)^2
>   $$
>   *显然，当动作 $a = 3$ 时，评判得分 $Q$ 达到最大值 $0$。*
> * 假设当前网络参数 **$\phi = 1$**，此时策略计算出的均值 $\mu_\phi = 2$。
>
> 我们的目标是：**通过梯度上升法更新 $\phi$，让 $Q$ 值最大化。**
>
> ---
>
> ### DDPG（确定性策略）
>
> 在 DDPG 中，策略是**确定性的（没有随机采样）**。网络算出来的动作就是确定值：
> $$
> a = \mu_\phi = 2\phi
> $$
> 现在我们来看计算图（Computation Graph）的链条：
> $$\phi \xrightarrow{\text{网络前向}} a = 2\phi \xrightarrow{\text{Critic打分}} Q(a) = -(a - 3)^2$$
>
> 在这个链条里，每一步都是确定性的微积分函数，微积分的**链式法则（Chain Rule）**可以毫无阻碍地一路反向传导：
> 1. **最后一环**：Critic 对动作求导：
>    $$
>    \frac{\partial Q}{\partial a} = -2(a - 3)
>    $$
>    在当前点 $a = 2$ 时，$\frac{\partial Q}{\partial a} = -2(2 - 3) = \mathbf{+2}$ （意思是：动作 $a$ 变大，$Q$ 会变大）。
> 2. **第一环**：动作对参数 $\phi$ 求导：
>    $$
>    \frac{\partial a}{\partial \phi} = \frac{d(2\phi)}{d\phi} = \mathbf{2}
>    $$
> 3. **DDPG 链式连乘**：
>    $$
>    \frac{dQ}{d\phi} = \frac{\partial Q}{\partial a} \cdot \frac{\partial a}{\partial \phi} = (+2) \times 2 = \mathbf{+4}
>    $$
>
> **结果**：梯度是 $+4$！优化器直接把 $\phi$ 往大的方向推（$\phi \leftarrow \phi + \eta \times 4$），动作 $a$ 就会从 $2$ 走向 $3$。**一切都完美且顺畅，因为 $a$ 是 $\phi$ 的显式确定性数学函数。**现在，我们希望策略具备探索能力，把它改成**随机策略**（正态分布，方差设为固定的 1）：
> $$
> a \sim \pi_\phi(a) = \mathcal{N}(\mu_\phi, \, 1^2) = \mathcal{N}(2\phi, \, 1)
> $$
> 代码是长这样的：
> ```python
> # 1. 神经网络前向
> mu = 2 * phi
> 
> # 2. 随机采样动作 (掷骰子)
> a = torch.normal(mean=mu, std=1.0)  # 假设骰子掷出了一个具体数值：2.4
> 
> # 3. 计算 Q 值
> Q = -(a - 3) ** 2
> ```
>
> 现在，你试着在代码里执行 `Q.backward()`，**灾难发生了，代码报错或者梯度直接断掉！**尝试套用 DDPG 的链式法则公式：
> $$
> \frac{dQ}{d\phi} \stackrel{?}{=} \frac{\partial Q}{\partial a} \cdot \frac{\partial a}{\partial \phi}
> $$
> **$\frac{\partial a}{\partial \phi}$ 等于多少？**
>
> 1. **从代码和物理意义上看**：
>    $a$ 是系统调用随机数生成器（`random`）**凭空抛硬币抛出来的一个固定孤立数字**（比如 `2.4`）。当你把网络参数 $\phi$ 改变一点点（比如从 $1.000$ 变成 $1.001$）时，这个已经被抛出来的历史数字 `2.4` 会随之连续改变吗？**根本不会！**随机采样是一个**离散/随机跳转事件**，它不是一个关于 $\phi$ 处处连续可微的代数表达式！在计算图中，**梯度到这个“随机采样黑盒”就撞墙截断了，反向传播根本穿不过去！**
>    
> 2. **从高等数学的严格积分上看**，我们真正想最大化的是期望：
>    $$
>    J(\phi) = \mathbb{E}_{a \sim \mathcal{N}(2\phi, 1)} [Q(a)] = \int_{-\infty}^{+\infty} Q(a) \cdot p_\phi(a) \, da
>    $$
>    我们要对参数 $\phi$ 求导：
>    $$
>    \nabla_\phi J(\phi) = \int_{-\infty}^{+\infty} Q(a) \cdot \mathbf{\nabla_\phi p_\phi(a)} \, da
>    $$
>    **数学现实非常残酷：求导符号 $\nabla_\phi$ 根本没有落在动作 $a$ 身上，而是落在了概率密度函数 $p_\phi(a)$ 身上！**在数学上，式子里根本不存在一个叫“$\frac{\partial a}{\partial \phi}$”的导数！这就是所谓的**“DDPG 链式法则在数学上彻底断裂”**。
>
> 传统的强化学习（PPO、A3C 前身）为了绕过这个断裂，在式子里强行乘上 $\frac{p_\phi(a)}{p_\phi(a)}$：
> $$
> \nabla_\phi J(\phi) = \int Q(a) \frac{\nabla_\phi p_\phi(a)}{p_\phi(a)} p_\phi(a) da = \mathbb{E}_{a \sim \pi_\phi} \Big[ Q(a) \cdot \nabla_\phi \log p_\phi(a) \Big]
> $$
>
> - **它怎么更新？**
>   随机掷骰子采出一个 $a=2.4$。算出这个动作的得分 $Q(2.4) = -(2.4 - 3)^2 = -0.36$。它**完全不用** $\nabla_a Q$（根本不看 Critic 的斜率导数），只把 $-0.36$ 当成一个黑盒权重，去乘以概率密度的导数。
> - **为什么方差极大？**
>   如果下一次掷骰子运气不好，采出了 $a=-10$（一个极差的动作），得分变成 $-169$。这个巨大的负数会把整个梯度狠狠地往反方向砸！它必须采样成千上万次，才能把这个随机投掷的巨大波动给平均掉。
>
> ---
>
> ### SAC 的“重参数化技巧”
>
> **“采样操作不可导，是因为我们把‘参数 $\phi$’和‘随机掷骰子’混在了一个盒子里！”**SAC 做了一个**变量分离**：让随机性完全归外界管（采一个与 $\phi$ 毫无关系的纯粹标准噪声 $\epsilon \sim \mathcal{N}(0, 1)$）。然后把动作写成一个**完全合法的代数公式**：
> $$
> a = \mu_\phi + 1.0 \times \epsilon = 2\phi + \epsilon
> $$
> 现在再来看看代码和计算图：
> ```python
> # 1. 掷纯粹的噪声骰子 (完全不依赖 phi, 是一个独立的常数输入)
> eps = torch.randn(1)  # 假设抽到了 0.4
> 
> # 2. 动作用确定性算式连接！
> a = 2 * phi + eps  # 算式是平滑代数式！
> 
> # 3. 计算 Q 值
> Q = -(a - 3) ** 2
> ```
>
> 现在的计算图变成了：
> $$
> \phi \xrightarrow{\times 2} (2\phi) \xrightarrow{+ \epsilon} a \xrightarrow{\text{Critic}} Q(a)
> $$
> **从 $\phi$ 到 $Q$，没有任何一个黑盒阻断！全部都是普通的加法和乘法！**我们重新对 $\phi$ 求导：
>
> 1. $\frac{\partial Q}{\partial a} = -2(a - 3)$ （Critic 提供精准斜率）
> 2. $\frac{\partial a}{\partial \phi} = \frac{\partial(2\phi + \epsilon)}{\partial \phi} = \mathbf{2}$ （因为 $\epsilon$ 对 $\phi$ 求导是 0！）
> 3. 链式法则瞬间**复活**：
>    $$
>    \frac{dQ}{d\phi} = \frac{\partial Q}{\partial a} \cdot \frac{\partial a}{\partial \phi} = -2(a - 3) \times 2
>    $$
>
> 代入刚才抽到的 $a = 2\phi + \epsilon = 2(1) + 0.4 = 2.4$：
> $$\frac{dQ}{d\phi} = -2(2.4 - 3) \times 2 = +1.2 \times 2 = \mathbf{+2.4}$$
>
> > 在概率论中，怎样从 $\mathcal{N}(\mu, \sigma^2)$ 里产生一个具体数值 $a$ 呢？根据**高斯分布的高等数学性质**，以下两种方式在概率意义上**严格 $100\%$ 等价**：
> >
> > * **方式 A（黑盒方式）**：调用封装好的系统库函数，直接生成随机数：
> >   $$
> >   a \sim \text{sample\_from\_Normal}(\mu, \sigma)
> >   $$
> >
> > * **方式 B（代数展开方式）**：从零均值、单位方差的标准正态分布里抽一个纯噪声 $\epsilon \sim \mathcal{N}(0, 1)$，然后做尺度平移：
> >   $$
> >   a = \mu + \sigma \cdot \epsilon
> >   $$
> >
> > **在传统方法中，不可导的 `sample` 用在了包含网络参数的 $\pi_\phi(a|s)$ 身上，导致反向传播被当场切断；**
> > **而在 SAC 中，这个不可导的 `sample` 被移花接木，用在了一个完全没有任何网络参数的标准正态分布 $\epsilon \sim \mathcal{N}(0, 1)$ 身上！**
>

## 5.3 与高斯分布熵的关系

现在掌握了重参数化技巧，能够通过 $a = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon$ 对网络求导了。此时，如果我们**不加熵**，只像传统 RL 那样单纯最大化 $Q$ 值：
$$
\max_\phi \mathbb{E}_{\epsilon} \left[ Q\big(s, \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon\big) \right]
$$
请你跟着我推演一下，**神经网络会怎么耍小聪明？**

1. 假设 Critic 发现动作 $a = 3$ 是目前找到的最好动作（$Q$ 值最高）。
2. 网络会立刻把均值 $\mu_\phi(s)$ 调整到 $3$。
3. **关键问题来了：网络会怎么调整方差 $\sigma_\phi(s)$？**
   - 如果 $\sigma > 0$，加上噪声 $\epsilon$ 之后，采样出来的动作可能是 $2.7$ 或者 $3.3$。只要偏离了最高点 $3$，$Q$ 值就会下降！
   - 优化器为了让期望 $Q$ 值最大化，会毫不犹豫地做出一个极其理性的数学选择：**把方差 $\sigma_\phi(s)$ 一路降到 0！**
4. **灾难发生了**：
   在没有熵惩罚的情况下，训练仅仅跑几个回合，网络的**方差 $\sigma$ 就会彻底坍塌归零**！
   策略再次变成了死板的单点输出，重参数化彻底失效，**算法又一秒退化回了那个不会探索、极其脆弱的 DDPG！**

回忆高斯分布的微分熵公式：
$$
\mathcal{H}(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2} \log\left(2\pi e \sigma^2\right)
$$
**高斯分布的熵，只跟方差 $\sigma$ 有关！方差越大，熵越大；方差趋于 0，熵直接暴跌到负无穷！**当 SAC 把熵作为奖励加入目标函数后（论文公式 12）：
$$
\min_\phi \mathbb{E} \Big[ \underbrace{-Q(s, a)}_{\text{追求最大回报}} + \underbrace{\alpha \log \pi_\phi(a|s)}_{\text{追求最大熵（抑制方差坍塌）}} \Big]
$$

* **$Q$ 值项在使劲把方差往下拉**：“别乱动了！选当前最好的动作，不要冒险！”
* **熵项在使劲把方差往上拉**：“不行！方差太小我就扣你分！你必须保持足够大的随机混乱度，继续探索！”

> **重参数化赋予了网络“调整方差 $\sigma$ 的能力”；而最大熵赋予了网络“维持方差 $\sigma$ 不归零的动机”！两者缺一不可！**

# 6. 动作空间约束

在机器人控制中，关节力矩都有物理上下界，例如动作范围必须严格限制在 $[-1, 1]$ 之间。传统方法常采用高斯分布采样后直接做硬截断：$a = \text{clip}(u, -1, 1)$。
**数学大忌！** 硬截断会导致 $[-1, 1]$ 边界处的概率密度发生奇点突变（脉冲），导数无法计算，且破坏了连续概率密度的良定义。

## 6.1 双曲正切函数（Tanh）压缩
SAC 的做法是：让神经网络先输出一个无界高斯分布随机变量 $u \sim \mu(u|s) = \mathcal{N}(\mu(s), \sigma^2(s))$，然后通过一个平滑可导的非线性双曲正切映射：
$$
a = \tanh(u)
$$
将实数轴 $(-\infty, +\infty)$ 平滑无缝压缩到有限开区间 $(-1, 1)$。

```
无界高斯采样 u ∈ (-∞, +∞) ───[ a = tanh(u) ]───> 有界物理动作 a ∈ (-1, 1)
```

---

## 6.2 概率测度变换定理

当一个随机变量经历非线性变换 $a = g(u) = \tanh(u)$ 时，**概率密度函数不能直接等同代入**，必须乘以微元体积缩放的修正因子，即**概率论中的变量替换公式**：

$$
P(a) |da| = P(u) |du| \implies \pi(a|s) = \mu(u|s) \left| \det\left( \frac{da}{du} \right) \right|^{-1} \tag{20}
$$
我们来求一维或独立多维情况下 $\tanh$ 的雅可比矩阵：
已知标量求导公式：
$$
\frac{d}{du}\tanh(u) = 1 - \tanh^2(u)
$$
对于 $D$ 维动作向量，由于每一维是独立变换的，其雅可比矩阵是一个对角矩阵：
$$
\frac{da}{du} = \text{diag}\left( 1 - \tanh^2(u_1), \; 1 - \tanh^2(u_2), \; \dots, \; 1 - \tanh^2(u_D) \right)
$$
对角矩阵的行列式等于对角线元素的乘积：
$$
\det\left( \frac{da}{du} \right) = \prod_{i=1}^D \left( 1 - \tanh^2(u_i) \right)
$$
将行列式代入公式 (20)，并在两边取对数：

$$
\begin{aligned}
\log \pi(a|s) &= \log \left( \mu(u|s) \cdot \prod_{i=1}^D \left( 1 - \tanh^2(u_i) \right)^{-1} \right) \\
&= \log \mu(u|s) - \sum_{i=1}^D \log\left( 1 - \tanh^2(u_i) \right) 
\end{aligned} \tag{21}
$$

> [!IMPORTANT]
>
> ### 为什么硬截断（Clip）会导致奇点、破坏定义、无法求导？
>
> 假设机械臂的关节力矩物理上限是 $1$，下限是 $-1$。网络通过高斯分布采样出了一个动作原始值 $u \sim \mathcal{N}(0, 1^2)$。因为高斯分布的取值范围是 $(-\infty, +\infty)$，它很可能会采样出 $u = 2.5$ 这种超标的数值。**硬截断**：
> $$
> a = \text{clip}(u, -1, 1) = \begin{cases} 1, & u > 1 \\ u, & -1 \le u \le 1 \\ -1, & u < -1 \end{cases}
> $$
> 表面上看，这个代码运行得挺好，但**在数学底层，它犯了三大错误**：
>
> #### 错误 1：破坏了“连续随机变量”的良定义
>
> 在概率论中：
>
> * **对于真正的连续型随机变量**：取到任何**单点**的概率必须严格为 0，即 $P(X = 1.0) = 0$。连续变量只有“在某个区间内的概率密度（PDF）”，面积才代表概率。
> * 但是你看硬截断之后发生了什么：
>   所有大于 $1$ 的数（比如 $1.1, 2.5, 100$）全部被强行拍扁，变成了**完完全全精确等于 $1.0$ 这一个单点**！
> * 在标准正态分布中，$u > 1$ 的概率大约是 **$15.87\%$**。
>   这意味着，截断之后的随机变量 $a$，**在单点 $a = 1.0$ 处的离散概率高达 $P(a = 1.0) = 0.1587 > 0$！**
> * **这就是所谓的“奇点突变（脉冲）”**：
>   在连续概率密度图上，在 $a = 1.0$ 这个坐标点上，突然耸立起一根**高度为正无穷（$+\infty$）的狄拉克脉冲函数（Dirac Delta $\delta(a-1)$）**！
>   此时这个随机变量既不是纯连续的，也不是纯离散的，成了不伦不类的“混合型随机变量”。
>
> #### 错误 2：无法计算对数概率 $\log \pi(a)$，熵直接崩溃
>
> SAC 的优化目标里需要计算策略的对数概率和熵：
> $$
> \mathcal{H} = \mathbb{E}[-\log \pi(a|s)]
> $$
> 如果智能体采出了 $a = 1.0$：
>
> * 这个点的概率密度是**正无穷大（$\pi(1.0) = +\infty$）**。
> * 那么 $\log \pi(1.0) = \log(+\infty) = +\infty$。
> * 熵 $\mathcal{H} = -\infty$！
>   **你的损失函数里瞬间出现了无穷大或者 NaN，计算机程序当场崩溃！**
>
> #### 错误 3：导数死区
>
> 看硬截断函数的导数：当 $u > 1$ 时，$a = 1$ 是一个恒定常数。常数对自变量的导数等于多少？**$\frac{da}{du} = 0$！**
> 如果网络算出的动作太大（$u=3.0$），Critic 即使在大声呼喊“动作太大了，快减小！”，反向传播传到这里乘上导数 $0$，**梯度直接清零！** 网络根本无法通过梯度把超标的动作拉回来。

