import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_experiment(runs=2000, steps=10000, eps=0.1, alpha=0.1):
    k = 10
    
    avg_rewards = np.zeros((2, steps))
    optimal_action_pct = np.zeros((2, steps))
    
    q_star = np.zeros((runs, k))
    Q = np.zeros((runs, 2, k))
    N = np.zeros((runs, k), dtype=int)
    
    for t in tqdm(range(steps), desc="Running steps", unit="step"):
        # 所有 run 同时更新真实动作价值
        q_star += np.random.normal(0, 0.01, size=(runs, k))
        optimal_action = np.argmax(q_star, axis=1)  # (runs,)
        
        for method in range(2):
            # epsilon-greedy
            rand = np.random.rand(runs)
            random_actions = np.random.randint(0, k, size=runs)
            
            # 贪婪动作
            max_q = np.max(Q[:, method, :], axis=1, keepdims=True)
            is_best = (Q[:, method, :] == max_q)
            rand_vals = np.random.rand(runs, k)
            masked_vals = np.where(is_best, rand_vals, -np.inf)
            greedy_actions = np.argmax(masked_vals, axis=1)
            
            actions = np.where(rand < eps, random_actions, greedy_actions)
            
            rewards = np.random.normal(q_star[np.arange(runs), actions], 1.0)
            
            # 指标
            avg_rewards[method, t] += np.sum(rewards)
            optimal_action_pct[method, t] += np.sum(actions == optimal_action)
            
            # 更新 Q
            if method == 0:
                # 样本平均法
                N[np.arange(runs), actions] += 1
                Q[np.arange(runs), 0, actions] += (
                    (1.0 / N[np.arange(runs), actions]) *
                    (rewards - Q[np.arange(runs), 0, actions])
                )
            else:
                # 固定步长法
                Q[np.arange(runs), 1, actions] += alpha * (
                    rewards - Q[np.arange(runs), 1, actions]
                )
    
    avg_rewards /= runs
    optimal_action_pct = (optimal_action_pct / runs) * 100
    return avg_rewards, optimal_action_pct


steps = 10000
runs = 2000
avg_rewards, optimal_action_pct = run_experiment(runs=runs, steps=steps)


plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(avg_rewards[0], label=r'Sample-Average ($\alpha = 1/n$)', color='red', alpha=0.8)
plt.plot(avg_rewards[1], label=r'Constant Step-size ($\alpha = 0.1$)', color='blue', alpha=0.8)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Nonstationary 10-Armed Bandit: Average Reward')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 1, 2)
plt.plot(optimal_action_pct[0], label=r'Sample-Average ($\alpha = 1/n$)', color='red', alpha=0.8)
plt.plot(optimal_action_pct[1], label=r'Constant Step-size ($\alpha = 0.1$)', color='blue', alpha=0.8)
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Nonstationary 10-Armed Bandit: % Optimal Action')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()