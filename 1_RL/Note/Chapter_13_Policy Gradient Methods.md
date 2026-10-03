* **动作价值方法**：通过学习动作的价值（如 $q(s,a)$），进而依据估计出的动作价值来选择动作；如果没有动作价值的估计，策略甚至根本无法存在。
* 本章引入全新范式——**参数化策略方法**：直接学习一个参数化的策略，在选择动作时无需向价值函数进行查询。
* 价值函数在此类方法中并非必需，但仍可用于**辅助策略参数的学习过程**；动作的选择与决策执行则完全脱离价值函数。

* **决策机制解耦**：
  * **传统动作价值方法（隐式策略）**：决策机制是“评估驱动”的，动作的选择严重依赖 $\arg\max_{a} \hat{q}(s,a)$ 或 $\epsilon$-贪婪规则。价值函数是策略存在的先决条件。
  * **参数化策略方法（显式策略）**：决策机制是“策略独立”的。策略本身被直接建模为一个函数，输入当前状态即可直接输出动作的概率分布。
* **价值函数的职能重定位**：价值函数不再作为动作决策的过滤器，仅在参数更新时充当评估器。

| 符号                                 | 数学空间                                  | 物理与理论含义                                               |
| :----------------------------------- | :---------------------------------------- | :----------------------------------------------------------- |
| $\boldsymbol{\theta}$                | $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ | **策略参数矢量**，维度为 $d'$。                              |
| $\pi(a \mid s, \boldsymbol{\theta})$ | $[0, 1]$                                  | 在参数矢量为 $\boldsymbol{\theta}$ 时，状态 $s$ 下采取动作 $a$ 的条件概率分布。 |
| $\mathbf{w}$                         | $\mathbf{w} \in \mathbb{R}^d$             | **价值函数权重矢量**，维度为 $d$。                           |
| $\hat{v}(s, \mathbf{w})$             | $\mathbb{R}$                              | 若算法结合了价值函数，则以 $\mathbf{w}$ 参数化的状态价值近似函数。 |

参数化策略的条件概率定义式：

$$
\pi(a \mid s, \boldsymbol{\theta}) \doteq \Pr\{A_t = a \mid S_t = s, \boldsymbol{\theta}_t = \boldsymbol{\theta}\}
$$

* 策略参数空间维度记为 $d'$，而价值参数空间维度记为 $d$。表明策略模型与价值模型属于两个相互独立、通常维度并不相同的参数化子空间。

---

* 学习策略参数的目标是最大化某个**标量性能指标** $J(\boldsymbol{\theta})$。
* 算法通过逼近关于指标 $J$ 的**梯度上升**来更新策略参数：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \widehat{\nabla J(\boldsymbol{\theta}_t)}
$$

* **优化目标**：
  $$
  \max_{\boldsymbol{\theta} \in \mathbb{R}^{d'}} J(\boldsymbol{\theta})
  $$
  
  * $\boldsymbol{\theta}_t \in \mathbb{R}^{d'}$：当前第 $t$ 步迭代时的策略参数矢量。
  * $\alpha \in \mathbb{R}^{+}$：优化更新的步长参数。
  * $\widehat{\nabla J(\boldsymbol{\theta}_t)} \in \mathbb{R}^{d'}$：性能指标梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)$ 的**随机估计量**。
    文本明确指出，$\widehat{\nabla J(\boldsymbol{\theta}_t)}$ 的数学期望必须近似或等于性能指标关于参数 $\boldsymbol{\theta}_t$ 的真实梯度：
    $$
    \mathbb{E}\left[ \widehat{\nabla J(\boldsymbol{\theta}_t)} \right] \approx \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)
    $$

---

1. **策略梯度方法**：
   * **定义**：所有遵循公式这一基本更新模式的方法，统称为策略梯度方法。
   * **范围判定**：**无论该方法是否同时学习一个近似价值函数**，只要是基于性能梯度的参数化策略优化，均属于策略梯度方法。
2. **演员-评论家方法**：
   * **定义**：同时学习策略近似与价值函数近似的方法，称为演员-评论家方法。
   * **职责划分**：
     * **Actor**：指代被学习的参数化策略 $\pi(a \mid s, \boldsymbol{\theta})$，职责是产生动作并与环境交互；
     * **Critic**：指代被学习的价值函数，通常为状态价值函数 $\hat{v}(s, \mathbf{w})$，职责是评估当前策略表现并提供学习信号。

---

针对不同类型的任务环境，给出了标量性能指标 $J(\boldsymbol{\theta})$ 的两类具体定义方式：

**分幕式任务**

* **定义准则**：性能指标定义为**在参数化策略下起始状态的价值**。
* **数学表达**：即衡量从初始状态开始直至该幕结束所获得的累积期望回报。
  $$
  J(\boldsymbol{\theta}) \doteq v_{\pi_{\boldsymbol{\theta}}}(s_0)
  $$

**持续性任务**

* **定义准则**：性能指标定义为**平均奖励率**。
* **数学表达**：即衡量系统长期运行每一步所获得的稳态期望即时奖励
  $$
  J(\boldsymbol{\theta}) \doteq r(\pi)
  $$

# 13.1 Policy Approximation and its Advantages