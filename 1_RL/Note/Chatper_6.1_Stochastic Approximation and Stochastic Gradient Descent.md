# 均值估计

---

* **Sutton 书的学习体验**：Sutton 在第 6 章直接给出了 TD(0) 误差 $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 并写出更新公式 $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$。初学者往往会产生疑问：**凭什么用一个包含“猜测”的带噪声目标去更新当前估计，它在数学上就必然能收敛到真实值？**
* **赵书的逻辑补全**：第 4 章需要环境模型转移概率；第 5 章虽然 Model-Free，但是是**非增量式**的，必须等到整个 Episode 结束算出完整回报后再取经验平均；而第 7 章的 TD 方法是**按步在线增量式**的。
* **第 6 章的使命**：建立严格的数学桥梁——证明 TD 学习、Q-learning 等算法本质上都是**求解贝尔曼方程根的 Robbins-Monro 随机近似算法**。

---

考虑一个随机变量 $X$（取值属于集合 $\mathcal{X}$），假设我们有一组独立同分布（i.i.d.）的采样序列 $\{x_i\}_{i=1}^n$。我们的目标是估计数学期望 $\mathbb{E}[X]$。根据大数定律，期望值可以通过样本算术平均近似：

$$
\mathbb{E}[X] \approx \bar{x} := \frac{1}{n} \sum_{i=1}^n x_i \xrightarrow{n \to \infty} \mathbb{E}[X]
$$

强化学习中几乎所有核心指标都被定义为某种**数学期望**：
1. **状态价值函数**：$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$
2. **动作价值函数**：$q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$
3. **贝尔曼期望方程**：$v_\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]$
4. **策略梯度定理**：$\nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \ln \pi(A_t \mid S_t, \theta) Q_\pi(S_t, A_t)]$

---

计算平均值 $\bar{x}$ 通常有两种方式：

| 方式                   | 计算机制                                                     | 核心缺陷                                                     |
| :--------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **非增量式（批处理）** | 收集全部 $n$ 个样本后，一次性求和取平均：$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ | 必须等待所有样本采集完毕；若样本流式到达，无法实时获得当前估计；存储复杂度 $O(n)$。 |
| **增量式（迭代式）**   | 每来一个新样本 $x_k$，立即在旧估计 $w_k$ 基础上更新为新估计 $w_{k+1}$ | 初始阶段因样本量少估计不够精确。                             |

$$
\underbrace{w_{k+1}}_{\text{新估计}} = \underbrace{w_k}_{\text{旧估计}} + \frac{1}{k} \underbrace{(x_k - w_k)}_{\text{误差 (Error) = [目标 - 旧估计]}}
$$
与 Sutton 第 2 章 2.4 节给出的通用更新范式完全一致：
$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \times \big[ \text{Target} - \text{OldEstimate} \big]
$$
* **目标**：$x_k$（当前时刻观测到的随机样本）
* **误差**：$x_k - w_k$，即新信息与当前估计的偏差
* **步长**：$\frac{1}{k}$，新样本在整体估计中所占的权重随着历史样本数的增加而衰减

如果将衰减系数 $\frac{1}{k}$ 替换为一个更一般的正数序列 $\alpha_k > 0$：
$$
\mathbf{w_{k+1} = w_k - \alpha_k (w_k - x_k) = (1 - \alpha_k) w_k + \alpha_k x_k}
$$

1. **显式表达式失效**：
   当系数为 $\frac{1}{k}$ 时，我们可以显式证明 $w_{k+1} = \frac{1}{k}\sum_{i=1}^k x_i$；但当系数为任意序列 $\{\alpha_k\}$ 时，我们**无法写出简单的算术平均展开式**。
2. **收敛性疑问**：
   这样更新得到的 $w_k$，在 $k \to \infty$ 时**是否依然能收敛到真实数学期望 $\mathbb{E}[X]$？**
   * 如果 $\alpha_k$ 衰减太慢（如保持常数 $\alpha$），系统会一直震荡吗？
   * 如果 $\alpha_k$ 衰减太快（如 $\alpha_k = \frac{1}{k^2}$），算法会不会在还没走到真实均值前就“冻结”停滞了？
   * 需要对序列 $\{\alpha_k\}$ 施加什么样的温和条件，才能保证以概率 1 收敛？
3. **更深层次的数学统一**：
   * 这个均值迭代式，实际上是在用 **Robbins-Monro 算法**解方程 $g(w) = w - \mathbb{E}[X] = 0$ 的根；
   * 它同时也是在用 **随机梯度下降（SGD）** 最小化二次损失函数 $J(w) = \frac{1}{2}\mathbb{E}[(w - X)^2]$；
   * 在第 7 章中，Temporal-Difference只不过是将这里的 $x_k$ 替换为贝尔曼目标 $R + \gamma V(S')$

# Robbins-Monro 算法

如果说蒙特卡洛依靠“大数定律”用均值暴力逼近期望，那么时序差分（TD）与 Q-learning 的理论基石则是 **Robbins-Monro算法**。RM 算法是随机近似领域的奠基性工作，它彻底解决了**在“函数表达式未知”且“观测值带有随机噪声”的情况下求解方程根的难题**。

---

考虑一维方程的求根问题：
$$
g(w) = 0
$$
其中 $w \in \mathbb{R}$ 为待求解未知变量，$g: \mathbb{R} \to \mathbb{R}$ 为单调函数，真实根记为 $w^*$，即满足 $g(w^*) = 0$。

许多工程与机器学习问题都可以转化为求根问题：
* **无约束优化问题**：$\min_w J(w)$，其一阶必要条件为求梯度为零的点，即令 $g(w) := \nabla_w J(w) = 0$；
* **非零目标方程**：$f(w) = c$，可令 $g(w) := f(w) - c = 0$。

如果 $g(w)$ 解析式已知，我们可以使用牛顿法或一阶梯度下降等经典数值分析算法。但在强化学习与数据驱动场景中：
1. **函数解析式未知**：系统是一个黑箱，我们无法获得 $g(w)$ 的代数表达式，更无法直接求导 $\nabla g(w)$；
2. **观测受到噪声污染**：我们向黑箱输入 $w$，系统返回的不是精确的 $g(w)$，而是一个受到随机误差 $\eta$ 污染的带噪观测值：
   $$
   \tilde{g}(w, \eta) = g(w) + \eta
   $$
   其中 $\eta$ 为随机观测误差。

Robbins-Monro 算法采用如下极其简洁的一阶增量迭代形式：
$$
\mathbf{w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)} \quad (k = 1, 2, 3, \dots)
$$
* $w_k$：第 $k$ 步对待求根 $w^*$ 的估计值；
* $a_k > 0$：第 $k$ 步的步长（学习率）；
* $\tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k$：在当前估计点 $w_k$ 处采样获得的带噪输出。

---

## Robbins-Monro 定理

在带随机观测误差 $\eta$ 的情况下，确定 $w_k$ 是否收敛到 $w^*$ 是非平凡的。对于 RM 算法 $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$，若以下三个条件同时满足：

1. **函数增长性条件**：存在常数 $c_1, c_2 > 0$，使得对所有 $w$ 满足：
   $$
   0 < c_1 \le \nabla_w g(w) \le c_2
   $$
2. **步长衰减条件**：
   $$
   \sum_{k=1}^\infty a_k = \infty \quad \text{且} \quad \sum_{k=1}^\infty a_k^2 < \infty
   $$
3. **噪声统计特性条件**：定义历史信息集 $H_k = \{w_k, w_{k-1}, \dots, \eta_{k-1}, \dots\}$，观测误差满足：
   $$
   \mathbb{E}[\eta_k \mid H_k] = 0 \quad \text{且} \quad \mathbb{E}[\eta_k^2 \mid H_k] < \infty
   $$
   则序列 $\{w_k\}$ **几乎必然收敛**到满足 $g(w^*) = 0$ 的真根 $w^*$：
$$
\lim_{k \to \infty} w_k = w^* \quad \text{a.s.}
$$

---

### 物理直觉与数学内涵

**条件 1 剖析：$0 < c_1 \le \nabla_w g(w) \le c_2$**

* **严格下界 $0 < c_1 \le \nabla_w g(w)$**：
  * 说明 $g(w)$ 是**严格单调递增**的，它保证了方程 $g(w) = 0$ 的根**不仅存在，而且全局唯一**；
  * 在优化问题 $g(w) = \nabla_w J(w) = 0$ 中，该条件等价于目标函数 $J(w)$ 是**严格强凸**的，杜绝了驻点是局部极大值或鞍点的可能。
* **严格上界 $\nabla_w g(w) \le c_2$**：
  * 说明函数具有全局连续性，其变化速率不会无休止爆炸。这防止了单步更新因为函数陡峭而产生不可控的飞跃（例如 $g(w) = \tanh(w-1)$ 满足该条件，但 $g(w) = w^3 - 5$ 在全局范围内斜率无上界，只能保证局部收敛）。

---

**条件 2 剖析：$\sum_{k=1}^\infty a_k^2 < \infty$ 与 $\sum_{k=1}^\infty a_k = \infty$**

| 步长约束                                             | 核心物理目的                               | 为什么缺失会导致失败？                                       |
| :--------------------------------------------------- | :----------------------------------------- | :----------------------------------------------------------- |
| **$\sum_{k=1}^\infty a_k^2 < \infty$**<br>(平方可和) | **抑制随机噪声累计，保证步长最终收敛至 0** | 由级数收敛必要条件必有 $a_k \to 0$。<br>因为 $w_{k+1} - w_k = -a_k \tilde{g}(w_k, \eta_k) = -a_k [g(w_k) + \eta_k]$。当 $w_k \to w^*$ 时，$g(w_k) \to 0$，迭代步完全由随机噪声 $\eta_k$ 主导。若步长 $a_k$ 不趋于 0，即使已经抵达真实根附近，算法也依然会被噪声无休止地踢来踢去，无法稳定收敛。此外，平方可和保证了随机噪声的累积方差有限。 |
| **$\sum_{k=1}^\infty a_k = \infty$**<br>(发散不可和) | **赋予算法无限的“行走能量”，防止过早冻结** | 将每一步展开相加：<br>$(w_2 - w_1) + (w_3 - w_2) + \dots = w_\infty - w_1 = -\sum_{k=1}^\infty a_k \tilde{g}(w_k, \eta_k)$。<br>若 $\sum a_k < \infty$（如 $a_k = 1/k^2$），则从起点 $w_1$ 能走过的最大总路程受限：$|w_\infty - w_1| \le b < \infty$。一旦初始猜测 $w_1$ 选得较远（满足 $|w_1 - w^*| > b$），算法即使步步朝着目标走，还没走到真根 $w^*$ 步长就已经衰减殆尽，算法在半路“冻僵”！因此 $\sum a_k = \infty$ 保证了无论初值多么远离真根，算法都有足够的能量抵达目的地。 |

在工程与深度强化学习中，人们几乎总是采用**常数学习率**。
* 此时 $\sum \alpha^2 = \infty$，严格意义上违反了收敛条件，算法无法精确收敛到单点 $w^*$，而是在以 $w^*$ 为中心的一个方差为 $O(\alpha)$ 的微小邻域内持续波动；
* 但常数学习率换来了**追踪非平稳环境**的能力（由于强化学习中策略在不断改变，价值函数的真值处于动态漂移中，永不归零的步长能赋予智能体持续适应新环境的能力）。

---

**条件 3 剖析：$\mathbb{E}[\eta_k \mid H_k] = 0$ 且 $\mathbb{E}[\eta_k^2 \mid H_k] < \infty$**

* **条件零均值**：说明观测误差属于**无偏噪声**。在任意时刻，噪声既不会系统性偏大，也不会系统性偏小；

* **方差有界**：噪声的能量是有限的，杜绝了极端病态异常值；

  若 $\{\eta_k\}$ 为独立同分布随机序列，只要自身均值为 0、方差有限，该条件即自动成立。

---

## 应用于均值估计

为什么通用更新率算法 $w_{k+1} = w_k - \alpha_k(w_k - x_k)$ 能收敛到真实数学期望 $\mathbb{E}[X]$？设待求解目标为 $w^* = \mathbb{E}[X]$。构造目标函数为：

$$
g(w) := w - \mathbb{E}[X]
$$
显然，求 $g(w^*) = 0$ 的根，等价于求 $w^* = \mathbb{E}[X]$。我们无法直接得知期望值 $\mathbb{E}[X]$，每次只能观测到随机变量的一个独立样本 $x_k$。定义观测值为：

$$
\tilde{g}(w_k, x_k) := w_k - x_k
$$
将其恒等变换展开：
$$
\begin{aligned}
\tilde{g}(w_k, x_k) &= w_k - x_k + \mathbb{E}[X] - \mathbb{E}[X] \\
&= \underbrace{(w_k - \mathbb{E}[X])}_{g(w_k)} + \underbrace{(\mathbb{E}[X] - x_k)}_{\text{记为噪声 } \eta_k} \\
&= g(w_k) + \eta_k
\end{aligned}
$$
其中随机噪声项明确定义为：
$$
\eta_k := \mathbb{E}[X] - x_k
$$

代入标准 RM 算法：
$$
w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k) \\
w_{k+1} = w_k - \alpha_k(w_k - x_k) = (1 - \alpha_k)w_k + \alpha_k x_k
$$
### 验证 Robbins-Monro 定理三大条件
1. **函数增长性条件**：
   $$
   \nabla_w g(w) = \frac{d}{dw}(w - \mathbb{E}[X]) = 1
   $$
   常数取 $c_1 = c_2 = 1 > 0$，条件 1 处处严格满足。
2. **学习率条件**：
   选择满足 $\sum_{k=1}^\infty \alpha_k = \infty$ 且 $\sum_{k=1}^\infty \alpha_k^2 < \infty$ 的步长序列（例如 $\alpha_k = \frac{1}{k}$）。
3. **噪声统计特性**：
   * 条件均值：$\mathbb{E}[\eta_k \mid H_k] = \mathbb{E}[\mathbb{E}[X] - x_k] = \mathbb{E}[X] - \mathbb{E}[X] = 0$；
   * 条件方差：$\mathbb{E}[\eta_k^2 \mid H_k] = \mathbb{E}[(x_k - \mathbb{E}[X])^2] = \text{Var}(X) < \infty$（只要样本方差有限）。

# Stochastic Gradient Descent

考虑如下期望最小化问题：
$$
\min_w J(w) := \mathbb{E}[f(w, X)]
$$
* $w$：待优化的参数；
* $X$：连续或离散的随机变量，其概率分布未知或无法显式解析积分；
* $f(w, X)$：参数化目标函数，关于 $w$ 可微；
* $J(w)$：期望目标函数。

为了最小化 $J(w)$，我们需要计算目标函数的真实梯度：
$$
\nabla_w J(w) = \nabla_w \mathbb{E}[f(w, X)] = \mathbb{E}[\nabla_w f(w, X)]
$$
## 三种梯度下降方法

### Gradient Descent,
$$
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k) = w_k - \alpha_k \mathbb{E}[\nabla_w f(w_k, X)]
$$
* **核心缺陷**：计算解析期望 $\mathbb{E}[\cdot]$ 必须预先获知随机变量 $X$ 的精确概率分布 $p(x)$。在真实数据流中，环境的物理分布是未知的黑箱。

### Batch Gradient Descent
若收集到大样本量集合 $\{x_i\}_{i=1}^n$（$x_i \stackrel{\text{i.i.d.}}{\sim} X$），利用蒙特卡洛经验平均逼近期望：
$$
\mathbb{E}[\nabla_w f(w_k, X)] \approx \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i)
$$
代入得更新式：
$$
w_{k+1} = w_k - \alpha_k \left( \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i) \right)
$$
* **核心缺陷**：**计算复杂度与存储代价极高**。每更新一次 $w_k$，必把全部 $n$ 个样本扫描并求导一遍；如果数据是源源不断到来的在线数据流，BGD 无法实时响应。

### Stochastic Gradient Descent
既然求 $n$ 个样本平均太慢，SGD 采取了极端而激进的策略——**只用第 $k$ 步采集到的单一样本 $x_k$，直接用单样本梯度代替整体真实梯度期望**：
$$
\mathbf{w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)}
$$
* **随机**：更新方向完全依赖于每一步随机抽样得到的样本 $x_k$。

---

### 随机梯度到底是真实梯度的什么？
虽然单样本梯度 $\nabla_w f(w_k, x_k) \neq \mathbb{E}[\nabla_w f(w_k, X)]$，但将其恒等变形：
$$
\nabla_w f(w_k, x_k) = \underbrace{\mathbb{E}[\nabla_w f(w_k, X)]}_{\text{真实梯度 (True Gradient)}} + \underbrace{\Big( \nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)] \Big)}_{\text{随机扰动噪声 } \eta_k}
$$
代入 SGD 更新式：
$$
w_{k+1} = w_k - \alpha_k \mathbb{E}[\nabla_w f(w_k, X)] - \alpha_k \eta_k
$$
* 因为样本 $x_k$ 是独立同分布采样的，对噪声项取期望：
  $$
  \mathbb{E}_{x_k}[\eta_k] = \mathbb{E}_{x_k}[\nabla_w f(w_k, x_k)] - \mathbb{E}_X[\nabla_w f(w_k, X)] = 0
  $$
* SGD 本质上就是**带有零均值随机扰动的标准梯度下降**！在统计平均意义下，扰动不会系统性带偏搜索方向。

---

## 均值估计的优化视角

考虑最小化如下二次均方误差损失：
$$
\min_w J(w) = \mathbb{E}\left[ \frac{1}{2} \|w - X\|^2 \right] \doteq \mathbb{E}[f(w, X)]
$$
其中 $f(w, X) = \frac{1}{2} \|w - X\|^2$。

* **验证最优点**：令导数等于零：
  $$
  \nabla_w J(w) = \mathbb{E}\left[ \nabla_w \left(\frac{1}{2} \|w - X\|^2\right) \right] = \mathbb{E}[w - X] = w - \mathbb{E}[X] = 0 \implies \mathbf{w^* = \mathbb{E}[X]}
  $$
  这严格证明了：**求随机变量的数学期望，完全等价于求解上述二次损失函数的全局极小点**。

**GD**:
$$
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k) = w_k - \alpha_k \mathbb{E}[w_k - X]
$$
该式在实际中不可行，因为右侧包含待求解的目标 $\mathbb{E}[X]$。

**SGD**:对于单样本 $x_k$，其瞬时单样本损失函数为 $f(w, x_k) = \frac{1}{2}\|w - x_k\|^2$，其随机梯度为：
$$
\nabla_w f(w_k, x_k) = w_k - x_k
$$
代入 SGD 迭代公式：
$$
\mathbf{w_{k+1} = w_k - \alpha_k (w_k - x_k)}
$$

---

## 收敛性分析

对于 SGD 算法 $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$，若满足：
1. **强凸性与曲率有界条件**：存在常数 $c_1, c_2 > 0$，使得：
   $$
   0 < c_1 \le \nabla_w^2 f(w, X) \le c_2 \quad (\forall w, X)
   $$
   *(高维向量时，对应 Hessian 矩阵 $\nabla_w^2 f$ 的特征值一致介于 $c_1$ 与 $c_2$ 之间)*；
2. **学习率 RM 条件**：$\sum_{k=1}^\infty \alpha_k = \infty \quad \text{且} \quad \sum_{k=1}^\infty \alpha_k^2 < \infty$；
3. **样本独立同分布**：$\{x_k\}_{k=1}^\infty$ 为 i.i.d. 采样序列。则估计量 $w_k$ 几乎必然收敛到使真实梯度为 0 的唯一极小点 $w^*$：
$$
\lim_{k \to \infty} w_k = w^* \quad \text{a.s.}
$$

---

#### SGD 是 Robbins-Monro 算法特例
* **第一步：将优化极小化转化为求根问题**，令求根函数定义为真实梯度函数：
  $$
  g(w) := \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)]
  $$
  最小化 $J(w)$ 等价于求解根方程 $g(w) = 0$。
  
* **第二步：构造带噪观测值**,在第 $k$ 步，我们能计算的样本量为随机梯度 $\tilde{g} := \nabla_w f(w, x_k)$。改写为：
  $$
  \begin{aligned}
  \tilde{g}(w_k, \eta_k) &= \nabla_w f(w_k, x_k) \\
  &= \underbrace{\mathbb{E}[\nabla_w f(w_k, X)]}_{g(w_k)} + \underbrace{\Big( \nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)] \Big)}_{\eta_k} \\
  &= g(w_k) + \eta_k
  \end{aligned}
  $$
  代入标准 RM 算法式 $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$：
  $$
  w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)
  $$
  这在形式上与 SGD 算法完全吻合。**因此，SGD 是 RM 算法的一个具体实例！**

---

## 收敛模式分析

* 当初值远离最优解时，虽然每一次更新只用一个随机样本，但估值轨迹**几乎笔直、极其快速地冲向最优解邻域**；
* 当估值已经非常接近最优解时，轨迹开始**剧烈抖动、随机游走，收敛变得迟缓**。

### 相对误差 $\delta_k$ 的严谨数学推导
定义随机梯度相对于真实梯度的相对误差：
$$
\delta_k := \frac{\| \nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)] \|}{\| \mathbb{E}[\nabla_w f(w_k, X)] \|}
$$
对于最优解 $w^*$，必定满足真实驻点条件 $\mathbb{E}[\nabla_w f(w^*, X)] = 0$。因此分母可改写为与最优点差值：
$$
\| \mathbb{E}[\nabla_w f(w_k, X)] \| = \| \mathbb{E}[\nabla_w f(w_k, X)] - \mathbb{E}[\nabla_w f(w^*, X)] \|
$$
由**微分中值定理**，存在 $\tilde{w}_k \in [w_k, w^*]$ 使得：
$$
\mathbb{E}[\nabla_w f(w_k, X)] - \mathbb{E}[\nabla_w f(w^*, X)] = \mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)](w_k - w^*)
$$
结合强凸性假设 $\nabla_w^2 f \ge c > 0$：
$$
\| \mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)](w_k - w^*) \| \ge c \| w_k - w^* \|
$$
将该下界代入相对误差的分母中，得到极其深刻的**相对误差不等式**：
$$
\mathbf{\delta_k \le \frac{\| \nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)] \|}{c \cdot \| w_k - w^* \|}}
$$

该不等式清晰揭示出：**相对误差 $\delta_k$ 与到最优点的欧氏距离 $\|w_k - w^*\|$ 成严格反比！**

| 搜索阶段                                 | 几何与数值状态                             | 相对误差 $\delta_k$                       | 轨迹特征与物理机制                                           |
| :--------------------------------------- | :----------------------------------------- | :---------------------------------------- | :----------------------------------------------------------- |
| **远离最优点**<br>($\|w_k - w^*\|$ 很大) | 分母极大，<br>真实梯度模长占据绝对统治地位 | **$\delta_k \to 0$**<br>(相对误差极小)    | **行为高度贴近确定性 GD**。<br>单样本的随机扰动相对而言微不足道，梯度下降以极高信噪比直插最优点。 |
| **逼近最优点**<br>($\|w_k - w^*\|$ 很小) | 分母极小，<br>真实梯度模长已趋于 0         | **$\delta_k$ 剧烈放大**<br>(相对误差极大) | **行为表现出高度随机性与抖动**。<br>此时真实梯度几乎消失，残余更新量全被单样本噪声主导，导致轨迹在最优解附近徘徊。必须依靠步长 $\alpha_k \to 0$ 最终将噪声按平。 |

---

## A Deterministic Formulation

在许多机器学习场景（如分类、回归）中，问题往往以离散数据集形式给出，而不显式提及概率分布。

给定一个包含 $n$ 个离散实数的静态数据集 $\{x_i\}_{i=1}^n$（注意：这里的 $x_i$ 只是确定的静态数值，未必来自某个已知随机变量）。优化目标为最小化样本经验平均：
$$
\min_w J(w) = \frac{1}{n} \sum_{i=1}^n f(w, x_i)
$$
标准梯度下降为：
$$
w_{k+1} = w_k - \alpha_k \left( \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i) \right)
$$
若样本集规模 $n$ 极大，工程上每一步只能抽取一个样本计算：
$$
w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)
$$

1. **这里完全没有随机变量和期望符号，这个算法还能叫“随机”梯度下降（SGD）吗？**
2. **在实际读取这 $n$ 个样本时，我们应该排好序一个一个按顺序取，还是随机抽样？**

赵书给出了一个极其精炼的解答：**我们可以通过人为引入一个离散均匀分布随机变量，将确定性经验风险严格转化为随机期望优化！**

* 定义一个定义在集合 $\{x_i\}_{i=1}^n$ 上的离散随机变量 $X$，其满足**均匀分布**：
  $$
  p(X = x_i) = \frac{1}{n}, \quad \forall i = 1, \dots, n
  $$
* 则确定性优化目标可严格恒等变形为：
  $$
  J(w) = \frac{1}{n} \sum_{i=1}^n f(w, x_i) \equiv \sum_{i=1}^n p(X = x_i) f(w, x_i) \equiv \mathbf{\mathbb{E}[f(w, X)]}
  $$

* 既然 $J(w) \equiv \mathbb{E}[f(w, X)]$，那么为了满足 i.i.d. 采样前提：每一步的样本 $x_k$ 必须从数据集 $\{x_i\}_{i=1}^n$ 中独立且均匀地随机抽取。

# BGD、MBGD、SGD

## BGD
在第 $k$ 步迭代时，**使用数据集中的全部 $n$ 个样本**计算梯度的算术平均值：
$$
\mathbf{w_{k+1} = w_k - \alpha_k \left( \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i) \right)}
$$
* **特点**：当样本量 $n$ 足够大时，$\frac{1}{n}\sum_{i=1}^n \nabla_w f(w_k, x_i) \approx \mathbb{E}[\nabla_w f(w_k, X)]$，更新方向极其平滑精准，逼近真实的确定性梯度下降。

## MBGD
在第 $k$ 步迭代时，**从全集中随机抽取一个规模为 $m$ 的子集**（$1 < m < n$），记为索引子集 $I_k \subset \{1, \dots, n\}$，其中 $|I_k| = m$：
$$
\mathbf{w_{k+1} = w_k - \alpha_k \left( \frac{1}{m} \sum_{j \in I_k} \nabla_w f(w_k, x_j) \right)}
$$
* **特点**：它是 BGD 与 SGD 之间的折中与泛化。利用 $m$ 个样本的平均来抵消大部分单样本噪声。

## SGD
在第 $k$ 步迭代时，**仅独立随机抽取 1 个单一样本 $x_k$**：
$$
\mathbf{w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)}
$$
* **特点**：极端激进，单步计算复杂度降至最低，实时响应单步流式数据。

| 维度                  | 批量梯度下降 (BGD)         | 小批量梯度下降 (MBGD)                 | 随机梯度下降 (SGD)                     |
| :-------------------- | :------------------------- | :------------------------------------ | :------------------------------------- |
| **单步样本消耗**      | $n$（全部样本）            | $m$（小批量，$1 < m < n$）            | $1$（单个样本）                        |
| **单步计算复杂度**    | $O(n \cdot d)$（昂贵）     | $O(m \cdot d)$（适中）                | $O(d)$（极快）                         |
| **单步梯度方差**      | 极低（接近 0）             | 适中（与 $\frac{1}{m}$ 成正比）       | 极大（由单样本分布决定）               |
| **收敛轨迹特征**      | 平滑、笔直奔向极值点       | 伴随微小波动，稳定前进                | 前期快速冲向邻域，后期剧烈震荡         |
| **硬件并行度**        | 计算量过大，常受显存限制   | **高度契合现代 GPU 矩阵张量并行加速** | 无法利用 GPU 向量化 SIMD 并行能力      |
| **摆脱鞍点/局部极小** | 极易陷入鞍点或次优局部极小 | 适度的随机抖动有助于跳出不良鞍点      | 随机性极强，具备最强的跳出局部极值能力 |

当 $m = n$ 时，MBGD 是否严格等价于 BGD？

* **在严格的概率统计定义下，并不等价！**
  * **原因**：根据随机近似与 SGD 的独立同分布理论假设，MBGD 的 $m$ 个样本是从全集 $\{x_i\}_{i=1}^n$ 中**独立有放回随机抽样**得到的；
  * 如果 $m = n$，MBGD 随机抽取了 $n$ 次，由于是有放回抽样，**某些样本可能被重复抽到多次，而另一些样本可能一次都没被抽中**；
  * 而 BGD 是**确定性地、无重复地完整遍历全部 $n$ 个样本**。
  * 只有在工程实践中采用“每个 Epoch 随机打乱且无放回扫描”时，两者在满批次下才形式重合。

---

## 均值估计问题

目标为估计样本均值 $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$，构造二次损失函数：
$$
\min_w J(w) = \frac{1}{2n} \sum_{i=1}^n \|w - x_i\|^2
$$
显然最优解为 $w^* = \bar{x}$。求导可知对单个样本梯度为 $\nabla_w f(w, x_i) = w - x_i$。代入三种算法：

* **BGD**：
  $$
  w_{k+1} = w_k - \alpha_k \left( \frac{1}{n} \sum_{i=1}^n (w_k - x_i) \right) = w_k - \alpha_k (w_k - \bar{x})
  $$
* **MBGD**：定义小批量样本均值为 $\bar{x}_k^{(m)} := \frac{1}{m} \sum_{j \in I_k} x_j$，则：
  $$
  w_{k+1} = w_k - \alpha_k \left( \frac{1}{m} \sum_{j \in I_k} (w_k - x_j) \right) = w_k - \alpha_k \left( w_k - \bar{x}_k^{(m)} \right)
  $$
* **SGD**：
  $$
  w_{k+1} = w_k - \alpha_k (w_k - x_k)
  $$

当步长取经典衰减率 $\alpha_k = \frac{1}{k}$ 时的展开式，三者可以显式展开为：

* **BGD**：
  $$
  \mathbf{w_{k+1} = \frac{1}{k} \sum_{j=1}^k \bar{x} \equiv \bar{x}}
  $$
  > BGD 在第 1 步更新后，估计值直接与全集真均值 $\bar{x}$ 完全重合，一步归真！
* **MBGD**：
  $$
  \mathbf{w_{k+1} = \frac{1}{k} \sum_{j=1}^k \bar{x}_j^{(m)}}
  $$
  > 每一步吸收的是一个**已经经过 $m$ 重平均的低方差估计量 $\bar{x}_j^{(m)}$**。其单步方差仅为单样本方差的 $\frac{1}{m}$。
* **SGD**：
  $$
  \mathbf{w_{k+1} = \frac{1}{k} \sum_{j=1}^k x_j}
  $$
  > 必须完全依靠外层的 $\frac{1}{k}$ 慢速平摊单个原始样本 $x_j$ 的离散方差。
  
  > [!IMPORTANT]
  >
  > 首先观察均值优化问题：$\min_w \frac{1}{2n}\sum_{i=1}^n \|w - x_i\|^2$。当学习率取 $\alpha_k = \frac{1}{k}$ 时，这三个算法的更新公式具有**完全一模一样的数学结构**：
  >
  > $$
  > \begin{aligned}
  > \text{BGD:} \quad & w_{k+1} = w_k - \frac{1}{k}(w_k - \bar{x}) \\
  > \text{MBGD:} \quad & w_{k+1} = w_k - \frac{1}{k}\Big(w_k - \bar{x}_k^{(m)}\Big) \\
  > \text{SGD:} \quad & w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)
  > \end{aligned}
  > $$
  >
  > 我们令 $y_k$ 表示**第 $k$ 步算法所接收到的“目标观测值”**：
  >
  > * 对于 **BGD**：$y_k \equiv \bar{x}$（每一步都是全集平均值，是一个固定的常数）；
  > * 对于 **MBGD**：$y_k = \bar{x}_k^{(m)} = \frac{1}{m}\sum_{j \in I_k} x_j$（第 $k$ 步抽取的小批量子集的平均值）；
  > * 对于 **SGD**：$y_k = x_k$（第 $k$ 步随机抽取的单个样本）。
  >
  > 因此，三个算法可以**完全抽象为同一个主递推式**：
  > $$
  > \mathbf{w_{k+1} = w_k - \frac{1}{k}(w_k - y_k) = \left(1 - \frac{1}{k}\right)w_k + \frac{1}{k}y_k = \frac{k-1}{k}w_k + \frac{1}{k}y_k} \quad (k = 1, 2, 3, \dots)
  > $$
  >
  > 只要我们能证明该通式的解为：
  > $$
  > \mathbf{w_{k+1} = \frac{1}{k} \sum_{j=1}^k y_j}
  > $$
  > 那么把各自的 $y_j$ 代入，三个公式就全部水落石出！
  >
  > ---
  >
  > ### 推导1：逐步代入展开法
  >
  > 设初始估计为任意给定的初值 $w_1$。
  >
  > #### 1. 第 $k = 1$ 步：
  >
  > $$
  > w_2 = \frac{1 - 1}{1} w_1 + \frac{1}{1} y_1 = 0 \cdot w_1 + y_1 = \mathbf{y_1}
  > $$
  >
  > > **极其关键的细节**：注意看系数 $\frac{k-1}{k}$，当 $k=1$ 时它等于 $\frac{0}{1} = 0$！这意味着**无论初始值 $w_1$ 选得多么离谱、多么随意，在第 1 步更新后，它的影响被乘以 0 瞬间彻底抹平，算法强行将 $w_2$ 重置为第一个样本值 $y_1$**。
  >
  > #### 2. 第 $k = 2$ 步：
  >
  > 将 $w_2 = y_1$ 代入递推式：
  > $$
  > w_3 = \frac{2 - 1}{2} w_2 + \frac{1}{2} y_2 = \frac{1}{2} (y_1) + \frac{1}{2} y_2 = \mathbf{\frac{1}{2}(y_1 + y_2)}
  > $$
  >
  > #### 3. 第 $k = 3$ 步：
  >
  > 将 $w_3 = \frac{1}{2}(y_1 + y_2)$ 代入递推式：
  > $$
  > \begin{aligned}
  > w_4 &= \frac{3 - 1}{3} w_3 + \frac{1}{3} y_3 \\
  > &= \frac{2}{3} \cdot \left[ \frac{1}{2}(y_1 + y_2) \right] + \frac{1}{3} y_3
  > \end{aligned}
  > $$
  > 注意这里的关键代数抵消：$\frac{2}{3} \times \frac{1}{2} = \frac{1}{3}$！于是上式变为：
  > $$
  > w_4 = \frac{1}{3}(y_1 + y_2) + \frac{1}{3} y_3 = \mathbf{\frac{1}{3}(y_1 + y_2 + y_3)}
  > $$
  >
  > #### 4. 第 $k = 4$ 步：
  >
  > 将 $w_4 = \frac{1}{3}(y_1 + y_2 + y_3)$ 代入递推式：
  > $$
  > \begin{aligned}
  > w_5 &= \frac{4 - 1}{4} w_4 + \frac{1}{4} y_4 \\
  > &= \frac{3}{4} \cdot \left[ \frac{1}{3}(y_1 + y_2 + y_3) \right] + \frac{1}{4} y_4 \\
  > &= \frac{1}{4}(y_1 + y_2 + y_3) + \frac{1}{4} y_4 \\
  > &= \mathbf{\frac{1}{4}(y_1 + y_2 + y_3 + y_4)}
  > \end{aligned}
  > $$
  >
  > 对于任意的步数 $k$，分子分母交叉相消，最终必定有：
  > $$
  > w_{k+1} = \frac{1}{k} \sum_{j=1}^k y_j
  > $$
  >
  > ---
  >
  > ### 推导2：裂项差分法
  >
  > 我们可以对通式两边同乘以 $k$：
  > $$
  > w_{k+1} = \frac{k-1}{k}w_k + \frac{1}{k}y_k \iff \mathbf{k \cdot w_{k+1} = (k-1) \cdot w_k + y_k}
  > $$
  >
  > 定义辅助累加序列：$S_k := k \cdot w_{k+1}$（对应地，$S_{k-1} = (k-1) \cdot w_k$）。上式立即变成一个极其清爽的一阶差分方程：
  > $$
  > S_k = S_{k-1} + y_k \iff \mathbf{S_k - S_{k-1} = y_k}
  > $$
  >
  > 把 $j = 1, 2, \dots, k$ 的这 $k$ 个等式排成一列：
  > $$
  > \begin{aligned}
  > S_1 - S_0 &= y_1 \\
  > S_2 - S_1 &= y_2 \\
  > S_3 - S_2 &= y_3 \\
  > &\;\;\vdots \\
  > S_k - S_{k-1} &= y_k
  > \end{aligned}
  > $$
  > 全部等式**左侧与右侧分别相加**：
  > $$
  > (S_k - S_{k-1}) + \dots + (S_2 - S_1) + (S_1 - S_0) = \sum_{j=1}^k y_j
  > $$
  > 左侧抵消后只剩下首尾两项：
  > $$
  > S_k - S_0 = \sum_{j=1}^k y_j
  > $$
  > 由定义，$S_0 = 0 \cdot w_1 = 0$。因此：
  > $$
  > S_k = \sum_{j=1}^k y_j
  > $$
  > 将 $S_k = k \cdot w_{k+1}$ 代回：
  > $$
  > k \cdot w_{k+1} = \sum_{j=1}^k y_j \implies \mathbf{w_{k+1} = \frac{1}{k} \sum_{j=1}^k y_j}
  > $$
  
  
