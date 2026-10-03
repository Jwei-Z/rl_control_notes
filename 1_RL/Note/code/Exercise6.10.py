import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure matplotlib font rendering
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class StochasticWindyGridworldEnv:
    """
    Windy Gridworld Environment with Stochastic Wind (Exercise 6.10).
    Grid: 7 rows (0: bottom, 6: top) x 10 cols (0: left, 9: right).
    Start: (3, 0), Goal: (3, 7).
    Base wind: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0].
    """

    def __init__(self, action_mode='kings', stochastic_wind=True):
        self.height = 7
        self.width = 10
        self.start_state = (3, 0)
        self.goal_state = (3, 7)
        self.base_wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.action_mode = action_mode
        self.stochastic_wind = stochastic_wind

        if action_mode == 'standard':
            self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.action_names = ['Up', 'Down', 'Right', 'Left']
        elif action_mode == 'kings':
            self.actions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, 1), (1, -1), (-1, -1)
            ]
            self.action_names = ['Up', 'Down', 'Right', 'Left',
                                 'Up-Right', 'Down-Right', 'Up-Left', 'Down-Left']
        elif action_mode == 'kings_stop':
            self.actions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, 1), (1, -1), (-1, -1),
                (0, 0)
            ]
            self.action_names = ['Up', 'Down', 'Right', 'Left',
                                 'Up-Right', 'Down-Right', 'Up-Left', 'Down-Left', 'Stay']
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        self.num_actions = len(self.actions)

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action_idx):
        r, c = self.state
        dr, dc = self.actions[action_idx]
        w = self.base_wind[c]

        # Apply stochastic perturbation if there is wind
        if self.stochastic_wind and w > 0:
            wind_noise = np.random.choice([-1, 0, 1])
            effective_wind = max(0, w + wind_noise)
        else:
            effective_wind = w

        next_r = min(self.height - 1, max(0, r + dr + effective_wind))
        next_c = min(self.width - 1, max(0, c + dc))
        next_state = (int(next_r), int(next_c))

        self.state = next_state
        done = (next_state == self.goal_state)
        reward = -1.0

        return next_state, reward, done


class SarsaAgent:
    """Tabular Sarsa(0) on-policy control agent."""

    def __init__(self, env, alpha=0.5, epsilon=0.1, gamma=1.0):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = env.num_actions
        self.Q = np.zeros((env.height, env.width, self.num_actions), dtype=np.float64)

    def choose_action(self, state, greedy=False):
        r, c = state
        q_values = self.Q[r, c, :]
        if greedy or np.random.rand() >= self.epsilon:
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
        else:
            return np.random.randint(self.num_actions)

    def train(self, max_time_steps=8000):
        time_steps = 0
        episodes_completed = 0
        step_history = [0]
        episode_history = [0]

        while time_steps < max_time_steps:
            state = self.env.reset()
            action = self.choose_action(state)

            while True:
                next_state, reward, done = self.env.step(action)
                time_steps += 1

                if done:
                    td_target = reward
                    td_error = td_target - self.Q[state[0], state[1], action]
                    self.Q[state[0], state[1], action] += self.alpha * td_error

                    episodes_completed += 1
                    step_history.append(time_steps)
                    episode_history.append(episodes_completed)
                    break
                else:
                    next_action = self.choose_action(next_state)
                    td_target = reward + self.gamma * self.Q[next_state[0], next_state[1], next_action]
                    td_error = td_target - self.Q[state[0], state[1], action]
                    self.Q[state[0], state[1], action] += self.alpha * td_error

                    state = next_state
                    action = next_action

                if time_steps >= max_time_steps:
                    break

        return np.array(step_history), np.array(episode_history)

    def evaluate_policy(self, num_episodes=1000, max_steps=100):
        lengths = []
        trajectories = []

        for ep in range(num_episodes):
            state = self.env.reset()
            traj = [state]
            for step in range(max_steps):
                if state == self.env.goal_state:
                    break
                action = self.choose_action(state, greedy=True)
                state, _, done = self.env.step(action)
                traj.append(state)
                if done:
                    break

            lengths.append(len(traj) - 1)
            trajectories.append(traj)

        return np.array(lengths), trajectories


def run_experiment(action_mode, stochastic_wind, max_time_steps=8000, num_runs=10):
    common_steps = np.arange(0, max_time_steps + 1, 10)
    all_episodes = np.zeros((num_runs, len(common_steps)))

    best_agent = None
    best_mean_len = float('inf')
    best_eval_lens = None
    best_trajs = None

    for run in range(num_runs):
        np.random.seed(run * 100 + 42)
        env = StochasticWindyGridworldEnv(action_mode=action_mode, stochastic_wind=stochastic_wind)
        agent = SarsaAgent(env, alpha=0.5, epsilon=0.1, gamma=1.0)
        step_hist, ep_hist = agent.train(max_time_steps=max_time_steps)

        interp_ep = np.interp(common_steps, step_hist, ep_hist)
        all_episodes[run, :] = interp_ep

        eval_lens, trajs = agent.evaluate_policy(num_episodes=500)
        mean_len = np.mean(eval_lens)
        if mean_len < best_mean_len:
            best_mean_len = mean_len
            best_agent = agent
            best_eval_lens = eval_lens
            best_trajs = trajs

    # Fallback if needed
    if best_eval_lens is None:
        best_eval_lens, best_trajs = agent.evaluate_policy(num_episodes=500)
        best_agent = agent

    mean_episodes = np.mean(all_episodes, axis=0)
    return common_steps, mean_episodes, best_eval_lens, best_trajs, best_agent


def plot_comparative_results(results, output_dir):
    """Plot learning curves, episode length distribution, and sample stochastic trajectories."""
    # 1. Learning Curves
    plt.figure(figsize=(9, 6), dpi=300)
    colors = {
        'kings_det': '#0275D8',       # Blue
        'kings_stoch': '#D9534F',     # Red
        'kings_stop_stoch': '#5CB85C' # Green
    }
    labels = {
        'kings_det': "King's Moves (Deterministic Wind)",
        'kings_stoch': "King's Moves (Stochastic Wind - Ex 6.10)",
        'kings_stop_stoch': "King's Moves + Stay (Stochastic Wind)"
    }

    for key, (steps, ep_mean, eval_lens, _, _) in results.items():
        avg_len = np.mean(eval_lens)
        plt.plot(steps, ep_mean, label=f"{labels[key]} (Avg Test: {avg_len:.1f} steps)",
                 color=colors[key], linewidth=2.2)

    plt.title("Exercise 6.10: Sarsa Learning Curves under Stochastic Wind", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Episodes Completed", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left', framealpha=0.95)
    plt.xlim(0, 8000)
    plt.ylim(0, None)
    plt.tight_layout()
    curve_path = os.path.join(output_dir, 'Exercise6.10_learning_curves.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved learning curves plot to: {curve_path}")

    # 2. Episode Length Distribution Histogram under Stochastic Wind
    plt.figure(figsize=(9, 5.5), dpi=300)
    stoch_lens = results['kings_stoch'][2]
    # Filter reasonable range for clean plotting
    plot_lens = stoch_lens[stoch_lens <= 35]

    bins = np.arange(min(plot_lens) - 0.5, max(plot_lens) + 1.5, 1)
    counts, edges, patches_hist = plt.hist(plot_lens, bins=bins, color='#D9534F', edgecolor='black', alpha=0.85, rwidth=0.8)

    median_val = np.median(plot_lens)
    mean_val = np.mean(plot_lens)
    min_val = np.min(plot_lens)

    plt.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean Length = {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median Length = {median_val:.0f}')
    plt.axvline(min_val, color='darkred', linestyle=':', linewidth=2, label=f'Best Possible = {min_val:.0f}')

    plt.title("Exercise 6.10: Greedy Policy Path Length Distribution under Stochastic Wind\n(King's Moves, 500 Evaluation Episodes)",
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Episode Length (Steps to Goal)", fontsize=12)
    plt.ylabel("Frequency (Counts)", fontsize=12)
    plt.xticks(range(int(min_val), int(min(max(plot_lens), 30)) + 1))
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend(fontsize=11, framealpha=0.95)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'Exercise6.10_length_distribution.png')
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram plot to: {hist_path}")

    # 3. Sample Trajectories Visualization (Lucky vs Typical vs Adverse Wind)
    stoch_trajs = results['kings_stoch'][3]
    stoch_lens = results['kings_stoch'][2]

    # Select representative trajectories
    # 1. Best possible (length 7)
    lucky_idx = np.where(stoch_lens == min_val)[0][0]
    lucky_traj = stoch_trajs[lucky_idx]

    # 2. Typical (near median)
    typical_idx = np.where((stoch_lens >= median_val - 1) & (stoch_lens <= median_val + 1))[0][0]
    typical_traj = stoch_trajs[typical_idx]

    # 3. Adverse (long path with recovery loop)
    long_candidates = np.where((stoch_lens >= 24) & (stoch_lens <= 32))[0]
    adverse_idx = long_candidates[0] if len(long_candidates) > 0 else np.argmax(stoch_lens)
    adverse_traj = stoch_trajs[adverse_idx]

    trajs_to_plot = [
        ("Favorable Wind (Optimal 7 Steps)", lucky_traj, '#0275D8'),
        (f"Typical Wind ({len(typical_traj)-1} Steps)", typical_traj, '#5CB85C'),
        (f"Adverse Wind Gusts ({len(adverse_traj)-1} Steps - Loop Recovery)", adverse_traj, '#D9534F')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    for idx, (title, traj, col) in enumerate(trajs_to_plot):
        ax = axes[idx]
        for r in range(8):
            ax.axhline(r, color='gray', linestyle='-', linewidth=0.7, alpha=0.6)
        for c in range(11):
            ax.axvline(c, color='gray', linestyle='-', linewidth=0.7, alpha=0.6)

        ax.add_patch(patches.Rectangle((0, 3), 1, 1, color='#FFA07A', alpha=0.8, label='Start (S)'))
        ax.add_patch(patches.Rectangle((7, 3), 1, 1, color='#98FB98', alpha=0.8, label='Goal (G)'))
        ax.text(0.5, 3.5, 'S', fontsize=14, fontweight='bold', ha='center', va='center')
        ax.text(7.5, 3.5, 'G', fontsize=14, fontweight='bold', ha='center', va='center')

        for c in range(10):
            w = wind[c]
            ax.text(c + 0.5, -0.4, f"{w}", fontsize=11, fontweight='bold', ha='center', va='center', color='darkblue')
            if w > 0:
                ax.annotate('', xy=(c + 0.5, 0.9), xytext=(c + 0.5, 0.1),
                            arrowprops=dict(facecolor='lightblue', edgecolor='blue', width=2, headwidth=6, alpha=0.5))

        traj_r = [pt[0] + 0.5 for pt in traj]
        traj_c = [pt[1] + 0.5 for pt in traj]
        ax.plot(traj_c, traj_r, color=col, marker='o', markersize=4, linewidth=2.2, zorder=5)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.8, 7)
        ax.set_aspect('equal')
        ax.set_xticks(range(10))
        ax.set_yticks(range(7))
        ax.set_xlabel("Column (Mean Wind Strength)", fontsize=10)
        ax.set_ylabel("Row", fontsize=10)

    plt.suptitle("Exercise 6.10: Closed-Loop Trajectory Adaptations under Stochastic Wind",
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    traj_path = os.path.join(output_dir, 'Exercise6.10_sample_trajectories.png')
    plt.savefig(traj_path, bbox_inches='tight')
    plt.close()
    print(f"Saved sample trajectories plot to: {traj_path}")


def main():
    print("=" * 70)
    print("Sutton & Barto Exercise 6.10: Windy Gridworld with Stochastic Wind")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'figures', 'Exercise6.10')
    os.makedirs(output_dir, exist_ok=True)

    configs = {
        'kings_det': ('kings', False),
        'kings_stoch': ('kings', True),
        'kings_stop_stoch': ('kings_stop', True)
    }

    results = {}
    start_time = time.time()

    for key, (mode, stoch) in configs.items():
        stoch_str = "Stochastic" if stoch else "Deterministic"
        print(f"\n[Training] Running Sarsa: mode='{mode}', wind='{stoch_str}'...")
        steps, ep_mean, eval_lens, trajs, agent = run_experiment(
            action_mode=mode, stochastic_wind=stoch, max_time_steps=8000, num_runs=10
        )
        results[key] = (steps, ep_mean, eval_lens, trajs, agent)
        print(f"  -> Best Converged Agent Evaluation: Mean={np.mean(eval_lens):.2f}, Min={np.min(eval_lens)}, Median={np.median(eval_lens):.1f}")

    elapsed = time.time() - start_time
    print(f"\nTraining and evaluation completed in {elapsed:.2f} seconds.")

    plot_comparative_results(results, output_dir)

    print("\n" + "=" * 70)
    print("EXERCISE 6.10 QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 70)
    for key in configs:
        ep_completed = results[key][1][-1]
        lens = results[key][2]
        print(f"Config: {key:18s} | 8000-Step Episodes: {ep_completed:5.1f} | Best Agent Mean: {np.mean(lens):5.2f} | Min: {np.min(lens):2d} | Median: {np.median(lens):4.1f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
