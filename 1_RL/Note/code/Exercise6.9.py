"""
Sutton & Barto - Reinforcement Learning: An Introduction (2nd Edition)
Chapter 6: Temporal-Difference Learning
Exercise 6.9: Windy Gridworld with King's Moves (Programming)

This script:
1. Implements the Windy Gridworld environment (Example 6.5).
2. Supports three action sets:
   - Standard 4 actions (Up, Down, Left, Right)
   - King's moves (8 actions, including diagonals)
   - King's moves with a 9th action (Stay / No movement, subject only to wind)
3. Implements on-policy Sarsa(0) control with epsilon-greedy exploration.
4. Trains agents across 8,000+ time steps (and multiple independent runs for statistical stability).
5. Extracts and validates the optimal deterministic greedy paths.
6. Generates publication-quality figures:
   - Comparative learning curves (Episodes vs. Time steps)
   - Grid visualization of optimal trajectories under all three action sets.
7. Saves figures to figures/Exercise6.9/.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure matplotlib font rendering
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class WindyGridworldEnv:
    """
    Windy Gridworld Environment according to Sutton & Barto Example 6.5 & Exercise 6.9.
    
    Grid dimension:
        Height = 7 (rows: 0 to 6, where 0 is bottom and 6 is top)
        Width  = 10 (cols: 0 to 9, where 0 is left and 9 is right)
    Start state: (3, 0)
    Goal state:  (3, 7)
    Wind strength across columns 0 to 9:
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    """

    def __init__(self, action_mode='standard'):
        self.height = 7
        self.width = 10
        self.start_state = (3, 0)
        self.goal_state = (3, 7)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.action_mode = action_mode

        # Define action offsets: (delta_row, delta_col)
        # delta_row > 0 means Up, delta_row < 0 means Down
        # delta_col > 0 means Right, delta_col < 0 means Left
        if action_mode == 'standard':
            # 4 actions
            self.actions = [
                (1, 0),   # 0: Up
                (-1, 0),  # 1: Down
                (0, 1),   # 2: Right
                (0, -1)   # 3: Left
            ]
            self.action_names = ['Up', 'Down', 'Right', 'Left']
        elif action_mode == 'kings':
            # 8 actions (King's moves)
            self.actions = [
                (1, 0),    # 0: Up
                (-1, 0),   # 1: Down
                (0, 1),    # 2: Right
                (0, -1),   # 3: Left
                (1, 1),    # 4: Up-Right
                (-1, 1),   # 5: Down-Right
                (1, -1),   # 6: Up-Left
                (-1, -1)   # 7: Down-Left
            ]
            self.action_names = ['Up', 'Down', 'Right', 'Left',
                                 'Up-Right', 'Down-Right', 'Up-Left', 'Down-Left']
        elif action_mode == 'kings_stop':
            # 9 actions (King's moves + Stay still)
            self.actions = [
                (1, 0),    # 0: Up
                (-1, 0),   # 1: Down
                (0, 1),    # 2: Right
                (0, -1),   # 3: Left
                (1, 1),    # 4: Up-Right
                (-1, 1),   # 5: Down-Right
                (1, -1),   # 6: Up-Left
                (-1, -1),  # 7: Down-Left
                (0, 0)     # 8: Stay (No movement except wind)
            ]
            self.action_names = ['Up', 'Down', 'Right', 'Left',
                                 'Up-Right', 'Down-Right', 'Up-Left', 'Down-Left', 'Stay']
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        self.num_actions = len(self.actions)

    def reset(self):
        """Reset agent to the start state."""
        self.state = self.start_state
        return self.state

    def step(self, action_idx):
        """
        Transition function according to Example 6.5.
        Next state = clip(state + action + (wind, 0)).
        Returns: (next_state, reward, done)
        """
        r, c = self.state
        dr, dc = self.actions[action_idx]
        w = self.wind[c]

        # Apply action and vertical wind, then clip to grid boundaries
        next_r = min(self.height - 1, max(0, r + dr + w))
        next_c = min(self.width - 1, max(0, c + dc))
        next_state = (next_r, next_c)

        self.state = next_state
        done = (next_state == self.goal_state)
        reward = -1.0

        return next_state, reward, done


class SarsaAgent:
    """
    On-policy Sarsa control agent with tabular Q-function.
    """

    def __init__(self, env, alpha=0.5, epsilon=0.1, gamma=1.0):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = env.num_actions
        
        # Initialize Q-table: (height, width, num_actions)
        self.Q = np.zeros((env.height, env.width, self.num_actions), dtype=np.float64)

    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action selection with random tie-breaking."""
        r, c = state
        q_values = self.Q[r, c, :]
        if greedy or np.random.rand() >= self.epsilon:
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
        else:
            return np.random.randint(self.num_actions)

    def train(self, max_time_steps=8000):
        """
        Train using Sarsa(0) for a fixed budget of total environment steps.
        Tracks episodes completed over cumulative time steps.
        """
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
                    # Q(terminal, ·) = 0
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

    def extract_optimal_trajectory(self, max_steps=50):
        """Run pure greedy policy from start state to inspect path and length."""
        state = self.env.reset()
        trajectory = [state]
        actions_taken = []

        for _ in range(max_steps):
            if state == self.env.goal_state:
                break
            action = self.choose_action(state, greedy=True)
            actions_taken.append(action)
            state, _, done = self.env.step(action)
            trajectory.append(state)
            if done:
                break

        return trajectory, actions_taken


def run_experiment(action_mode, max_time_steps=8000, num_runs=10):
    """Run multiple independent Sarsa trials and average the step-episode curves."""
    common_steps = np.arange(0, max_time_steps + 1, 10)
    all_episodes = np.zeros((num_runs, len(common_steps)))

    best_trajectory = None
    best_actions = None
    best_len = float('inf')
    best_agent = None

    for run in range(num_runs):
        np.random.seed(run * 100 + 42)
        env = WindyGridworldEnv(action_mode=action_mode)
        agent = SarsaAgent(env, alpha=0.5, epsilon=0.1, gamma=1.0)
        step_hist, ep_hist = agent.train(max_time_steps=max_time_steps)

        # Interpolate episodes over uniform time steps
        interp_ep = np.interp(common_steps, step_hist, ep_hist)
        all_episodes[run, :] = interp_ep

        # Extract greedy path from this run
        traj, acts = agent.extract_optimal_trajectory(max_steps=50)
        cur_len = len(traj) - 1
        if traj[-1] == env.goal_state and cur_len < best_len:
            best_len = cur_len
            best_trajectory = traj
            best_actions = acts
            best_agent = agent

    # Fallback if no run reached goal under pure greedy evaluation
    if best_trajectory is None:
        best_trajectory, best_actions = agent.extract_optimal_trajectory(max_steps=50)
        best_agent = agent

    mean_episodes = np.mean(all_episodes, axis=0)
    return common_steps, mean_episodes, best_trajectory, best_actions, best_agent


def plot_learning_curves(results, save_path):
    """Plot comparative learning curves: Episodes vs. Time steps."""
    plt.figure(figsize=(9, 6), dpi=300)
    colors = {
        'standard': '#D9534F',     # Red
        'kings': '#0275D8',        # Blue
        'kings_stop': '#5CB85C'    # Green
    }
    labels = {
        'standard': 'Standard 4 Actions (Baseline)',
        'kings': "King's Moves (8 Actions)",
        'kings_stop': "King's Moves + Stay (9 Actions)"
    }

    for mode, (steps, episodes, traj, actions, _) in results.items():
        opt_len = len(traj) - 1
        plt.plot(steps, episodes, label=f"{labels[mode]} (Opt Path: {opt_len} steps)",
                 color=colors[mode], linewidth=2.2)

    plt.title("Exercise 6.9: Sarsa on Windy Gridworld with Action Variations", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Episodes Completed", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left', framealpha=0.95)
    plt.xlim(0, 8000)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves plot to: {save_path}")


def plot_trajectories(results, save_path):
    """Plot grid map and optimal trajectories side-by-side for all action modes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    modes = ['standard', 'kings', 'kings_stop']
    titles = [
        "Standard 4 Actions\nOptimal Steps: {0}",
        "King's Moves (8 Actions)\nOptimal Steps: {0}",
        "King's Moves + Stay (9 Actions)\nOptimal Steps: {0}"
    ]
    colors = ['#D9534F', '#0275D8', '#5CB85C']

    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        traj = results[mode][2]
        opt_steps = len(traj) - 1

        # Draw grid cells
        for r in range(8):
            ax.axhline(r, color='gray', linestyle='-', linewidth=0.7, alpha=0.6)
        for c in range(11):
            ax.axvline(c, color='gray', linestyle='-', linewidth=0.7, alpha=0.6)

        # Highlight start (3,0) and goal (3,7)
        ax.add_patch(patches.Rectangle((0, 3), 1, 1, color='#FFA07A', alpha=0.8, label='Start (S)'))
        ax.add_patch(patches.Rectangle((7, 3), 1, 1, color='#98FB98', alpha=0.8, label='Goal (G)'))
        ax.text(0.5, 3.5, 'S', fontsize=14, fontweight='bold', ha='center', va='center')
        ax.text(7.5, 3.5, 'G', fontsize=14, fontweight='bold', ha='center', va='center')

        # Draw wind labels and upward indicators
        for c in range(10):
            w = wind[c]
            ax.text(c + 0.5, -0.4, f"{w}", fontsize=11, fontweight='bold', ha='center', va='center', color='darkblue')
            if w > 0:
                ax.annotate('', xy=(c + 0.5, 0.9), xytext=(c + 0.5, 0.1),
                            arrowprops=dict(facecolor='lightblue', edgecolor='blue', width=2, headwidth=6, alpha=0.5))

        # Plot agent trajectory
        traj_r = [pt[0] + 0.5 for pt in traj]
        traj_c = [pt[1] + 0.5 for pt in traj]
        ax.plot(traj_c, traj_r, color=colors[idx], marker='o', markersize=5, linewidth=2.4, zorder=5)

        ax.set_title(titles[idx].format(opt_steps), fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.8, 7)
        ax.set_aspect('equal')
        ax.set_xticks(range(10))
        ax.set_yticks(range(7))
        ax.set_xlabel("Column (Wind Strength Below)", fontsize=10)
        ax.set_ylabel("Row", fontsize=10)

    plt.suptitle("Exercise 6.9: Comparison of Optimal Trajectories across Action Spaces", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectories plot to: {save_path}")


def main():
    print("=" * 70)
    print("Sutton & Barto Exercise 6.9: Windy Gridworld with King's Moves")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'figures', 'Exercise6.9')
    os.makedirs(output_dir, exist_ok=True)

    action_modes = ['standard', 'kings', 'kings_stop']
    results = {}

    start_time = time.time()
    for mode in action_modes:
        print(f"\n[Training] Running Sarsa on action_mode='{mode}'...")
        steps, ep_mean, traj, actions, agent = run_experiment(mode, max_time_steps=8000, num_runs=10)
        results[mode] = (steps, ep_mean, traj, actions, agent)
        opt_len = len(traj) - 1
        print(f"  -> Optimal trajectory length: {opt_len} steps")
        print(f"  -> Path coordinates: {traj}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # Save plots
    learning_curve_path = os.path.join(output_dir, 'Exercise6.9_learning_curves.png')
    plot_learning_curves(results, learning_curve_path)

    trajectories_path = os.path.join(output_dir, 'Exercise6.9_trajectories.png')
    plot_trajectories(results, trajectories_path)

    # Detailed textual summary
    print("\n" + "=" * 70)
    print("EXERCISE 6.9 QUANTITATIVE RESULTS SUMMARY")
    print("=" * 70)
    for mode in action_modes:
        traj = results[mode][2]
        opt_len = len(traj) - 1
        ep_completed = results[mode][1][-1]
        print(f"Mode: {mode:12s} | Optimal Steps: {opt_len:2d} | Avg Episodes at 8000 steps: {ep_completed:6.1f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
