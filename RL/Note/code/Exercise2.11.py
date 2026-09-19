import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# 非平稳多臂老虎机 (严格遵循 Exercise 2.5: q* 全部从 0 开始随机游走)
class NonStationaryBandit:
    def __init__(self, runs, k=10, walk_std=0.01):
        self.runs = runs
        self.k = k
        self.walk_std = walk_std
        # Exercise 2.5 要求初始价值全相等 (设为 0)
        self.q_true = np.zeros((runs, k))
        
    def step(self, actions):
        rewards = np.random.normal(self.q_true[np.arange(self.runs), actions], 1.0)
        self.q_true += np.random.normal(0, self.walk_std, (self.runs, self.k))
        return rewards

# 1. 常数步长 ε-greedy (α = 0.1)
class EpsilonGreedyConst:
    def __init__(self, runs, k=10, alpha=0.1, epsilon=0.1):
        self.runs = runs
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((runs, k))
        
    def select_action(self):
        explore = np.random.rand(self.runs) < self.epsilon
        random_actions = np.random.randint(0, self.k, self.runs)
        greedy_actions = np.argmax(self.Q + np.random.rand(self.runs, self.k) * 1e-9, axis=1)
        return np.where(explore, random_actions, greedy_actions)
            
    def update(self, actions, rewards):
        idx = np.arange(self.runs)
        self.Q[idx, actions] += self.alpha * (rewards - self.Q[idx, actions])

# 2. 样本平均 ε-greedy (原版 Figure 2.6 中的对照项)
class EpsilonGreedySampleAvg:
    def __init__(self, runs, k=10, epsilon=0.1):
        self.runs = runs
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros((runs, k))
        self.N = np.zeros((runs, k))
        
    def select_action(self):
        explore = np.random.rand(self.runs) < self.epsilon
        random_actions = np.random.randint(0, self.k, self.runs)
        greedy_actions = np.argmax(self.Q + np.random.rand(self.runs, self.k) * 1e-9, axis=1)
        return np.where(explore, random_actions, greedy_actions)
            
    def update(self, actions, rewards):
        idx = np.arange(self.runs)
        self.N[idx, actions] += 1
        self.Q[idx, actions] += (rewards - self.Q[idx, actions]) / self.N[idx, actions]

# 3. 原版 UCB (Figure 2.6 标准算法: 基于样本平均)
class UCB:
    def __init__(self, runs, k=10, c=1.0):
        self.runs = runs
        self.k = k
        self.c = c
        self.Q = np.zeros((runs, k))
        self.N = np.zeros((runs, k))
        self.t = 0
        
    def select_action(self):
        self.t += 1
        safe_N = np.maximum(self.N, 1)
        bonus = self.c * np.sqrt(np.log(self.t) / safe_N)
        bonus[self.N == 0] = np.inf
        return np.argmax(self.Q + bonus + np.random.rand(self.runs, self.k) * 1e-9, axis=1)
        
    def update(self, actions, rewards):
        idx = np.arange(self.runs)
        self.N[idx, actions] += 1
        self.Q[idx, actions] += (rewards - self.Q[idx, actions]) / self.N[idx, actions]

# 4. 梯度 Bandit (Gradient Bandit)
class GradientBandit:
    def __init__(self, runs, k=10, alpha=0.1):
        self.runs = runs
        self.k = k
        self.alpha = alpha
        self.H = np.zeros((runs, k))
        self.R_bar = np.zeros(runs)
        self.t = 0
        
    def select_action(self):
        self.t += 1
        exp_H = np.exp(self.H - np.max(self.H, axis=1, keepdims=True)) 
        self.pi = exp_H / np.sum(exp_H, axis=1, keepdims=True)
        cum_pi = np.cumsum(self.pi, axis=1)
        rand_vals = np.random.rand(self.runs, 1)
        # 稳妥的采样，避免浮点累积概率越界
        return np.clip(np.sum(cum_pi < rand_vals, axis=1), 0, self.k - 1)
        
    def update(self, actions, rewards):
        if self.t == 1:
            self.R_bar = rewards.copy()
        else:
            self.R_bar += (1.0 / self.t) * (rewards - self.R_bar)
            
        idx = np.arange(self.runs)
        action_mask = np.zeros((self.runs, self.k))
        action_mask[idx, actions] = 1.0
        delta = self.alpha * (rewards - self.R_bar)
        self.H += delta[:, None] * (action_mask - self.pi)

# 5. 乐观初始化贪婪 (Optimistic Greedy, α = 0.1)
class OptimisticGreedy:
    def __init__(self, runs, k=10, alpha=0.1, Q0=5.0):
        self.runs = runs
        self.k = k
        self.alpha = alpha
        self.Q = np.ones((runs, k)) * Q0 
        
    def select_action(self):
        return np.argmax(self.Q + np.random.rand(self.runs, self.k) * 1e-9, axis=1)
        
    def update(self, actions, rewards):
        idx = np.arange(self.runs)
        self.Q[idx, actions] += self.alpha * (rewards - self.Q[idx, actions])


def run_experiment(algo_name, param, steps=200000, last_steps=100000, runs=200):
    env = NonStationaryBandit(runs)
    
    if algo_name == 'epsilon_const':
        agent = EpsilonGreedyConst(runs, alpha=0.1, epsilon=param)
    elif algo_name == 'epsilon_sa':
        agent = EpsilonGreedySampleAvg(runs, epsilon=param)
    elif algo_name == 'ucb':
        agent = UCB(runs, c=param)
    elif algo_name == 'gradient':
        agent = GradientBandit(runs, alpha=param)
    elif algo_name == 'optimistic':
        agent = OptimisticGreedy(runs, alpha=0.1, Q0=param)
        
    last_rewards = 0.0
    measure_from = steps - last_steps
    for t in range(steps):
        actions = agent.select_action()
        rewards = env.step(actions)
        agent.update(actions, rewards)
        if t >= measure_from:
            last_rewards += np.sum(rewards)
            
    return last_rewards / (last_steps * runs)


if __name__ == "__main__":
    # 参数设置 (与 Figure 2.6 对齐)
    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
    ucb_cs = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    grad_alphas = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    q0s = [1/4, 1/2, 1, 2, 4]
    
    RUNS = 2000
    STEPS = 200000
    LAST_STEPS = 100000

    print("Running experiments in parallel...")
    
    # 1. 常数步长 ε-greedy (α=0.1)
    res_eps_const = Parallel(n_jobs=-1)(
        delayed(run_experiment)('epsilon_const', eps, STEPS, LAST_STEPS, RUNS) for eps in tqdm(epsilons, desc="ε-const")
    )
    
    # 2. 样本平均 ε-greedy (对照)
    res_eps_sa = Parallel(n_jobs=-1)(
        delayed(run_experiment)('epsilon_sa', eps, STEPS, LAST_STEPS, RUNS) for eps in tqdm(epsilons, desc="ε-sa")
    )
        
    # 3. UCB
    res_ucb = Parallel(n_jobs=-1)(
        delayed(run_experiment)('ucb', c, STEPS, LAST_STEPS, RUNS) for c in tqdm(ucb_cs, desc="UCB")
    )
        
    # 4. Gradient Bandit
    res_gradient = Parallel(n_jobs=-1)(
        delayed(run_experiment)('gradient', alpha, STEPS, LAST_STEPS, RUNS) for alpha in tqdm(grad_alphas, desc="Gradient")
    )
        
    # 5. Optimistic Greedy
    res_optimistic = Parallel(n_jobs=-1)(
        delayed(run_experiment)('optimistic', q0, STEPS, LAST_STEPS, RUNS) for q0 in tqdm(q0s, desc="Optimistic")
    )

    # 绘图：精确复现 Figure 2.6 风格
    fig, ax = plt.subplots(figsize=(11, 6))
    
    ax.plot(epsilons, res_eps_const, 'r-', linewidth=2, label=r'$\epsilon$-greedy ($\alpha = 0.1$)')
    ax.plot(epsilons, res_eps_sa, 'r--', linewidth=1.5, label=r'$\epsilon$-greedy (sample-avg)')
    ax.plot(ucb_cs, res_ucb, 'b-', linewidth=2, label='UCB')
    ax.plot(grad_alphas, res_gradient, 'g-', linewidth=2, label='gradient bandit')
    ax.plot(q0s, res_optimistic, 'k-', linewidth=2, label=r'greedy with opt. init. ($\alpha = 0.1$)')
    
    # 对数坐标与刻度标签设置
    ax.set_xscale('log', base=2)
    ticks = [2**i for i in range(-7, 3)]
    tick_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=11)
    
    ax.set_xlabel(r'Parameter ($\epsilon$, $\alpha$, $c$, $Q_0$)', fontsize=13)
    ax.set_ylabel('Average reward over last 100,000 steps', fontsize=13)
    ax.set_title('Exercise 2.11: Nonstationary 10-armed Bandit (Figure 2.6 Analogous)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    # plt.savefig('exercise_2_11.png', dpi=300)
    plt.show()