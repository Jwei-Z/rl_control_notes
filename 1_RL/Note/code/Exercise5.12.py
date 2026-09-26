"""
Sutton & Barto - Reinforcement Learning: An Introduction (2nd Edition)
Chapter 5: Monte Carlo Methods
Exercise 5.12: Racetrack (Programming)

This script:
1. Implements the discrete Racetrack environment (Figure 5.5) for both Track 1 (left) and Track 2 (right).
2. Implements On-policy First-Visit Monte Carlo Control with epsilon-greedy exploration.
3. Incorporates stochastic velocity noise (p=0.1 of 0 acceleration).
4. Accurately checks collision and finish line crossing along projected continuous paths.
5. Trains on Track 1 and Track 2, logs training progress, and extracts optimal policies.
6. Visualizes and saves publication-quality trajectory plots (with noise turned off) to the figures directory.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Configure matplotlib font for clean rendering
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class RacetrackEnv:
    """
    Racetrack Environment according to Sutton & Barto Exercise 5.12.
    Cell types:
        0: Out-of-bounds (wall / grass)
        1: Track
        2: Starting line (red in Figure 5.5)
        3: Finish line (green in Figure 5.5)
    """

    def __init__(self, track_id=1, noise=0.1):
        self.track_id = track_id
        self.noise = noise
        self.grid, self.height, self.width, self.start_cells, self.finish_cells = self._build_track(track_id)
        
        # 9 discrete acceleration actions: (ay, ax) in {-1, 0, 1} x {-1, 0, 1}
        self.actions = [(ay, ax) for ay in [-1, 0, 1] for ax in [-1, 0, 1]]
        self.num_actions = len(self.actions)
        
        # Precompute valid action indices for each velocity pair (vy, vx)
        # Velocities are constrained to 0 <= vy < 5, 0 <= vx < 5, and not both 0 (except at start line)
        self.valid_actions = {}
        for vy in range(5):
            for vx in range(5):
                valid = []
                for a_idx, (ay, ax) in enumerate(self.actions):
                    nvy = vy + ay
                    nvx = vx + ax
                    if 0 <= nvy < 5 and 0 <= nvx < 5:
                        if nvy > 0 or nvx > 0:
                            valid.append(a_idx)
                self.valid_actions[(vy, vx)] = valid

    def _build_track(self, track_id):
        if track_id == 1:
            # Track 1 (Left in Figure 5.5): Height 32, Width 17
            height, width = 32, 17
            grid = np.ones((height, width), dtype=np.int32)
            
            # Off-track boundaries
            grid[0:26, 9:17] = 0
            grid[25, 9] = 1
            grid[0:18, 0] = 0
            grid[0:10, 1] = 0
            grid[0:4, 2] = 0
            
            # Starting line (Row 0, Columns 3 to 8)
            start_cells = []
            for j in range(3, 9):
                grid[0, j] = 2
                start_cells.append((0, j))
                
            # Finish line (Rows 26 to 31, Column 16)
            finish_cells = set()
            for i in range(26, 32):
                grid[i, 16] = 3
                finish_cells.add((i, 16))
                
        else:
            # Track 2 (Right in Figure 5.5): Height 30, Width 32
            height, width = 30, 32
            grid = np.ones((height, width), dtype=np.int32)
            
            # Off-track boundaries
            for i in range(0, 17):
                for j in range(23, width):
                    grid[i, j] = 0
            grid[17, 24:width] = 0
            grid[18, 26:width] = 0
            grid[19, 27:width] = 0
            grid[20, 30:width] = 0
            for i in range(3, 16):
                for j in range(0, i - 2):
                    grid[i, j] = 0
            grid[16:21, 0:14] = 0
            grid[21, 0:13] = 0
            grid[22, 0:12] = 0
            grid[23:27, 0:11] = 0
            grid[27, 0:12] = 0
            grid[28, 0:13] = 0
            grid[29, 0:16] = 0
            
            # Starting line (Row 0, Columns 0 to 22)
            start_cells = []
            for j in range(0, 23):
                grid[0, j] = 2
                start_cells.append((0, j))
                
            # Finish line (Rows 21 to 29, Column 31)
            finish_cells = set()
            for i in range(21, height):
                grid[i, width - 1] = 3
                finish_cells.add((i, width - 1))
                
        return grid, height, width, start_cells, finish_cells

    def reset(self, start_pos=None):
        """Reset car to a random (or specified) position on the starting line with 0 velocity."""
        if start_pos is None:
            idx = np.random.randint(len(self.start_cells))
            pos = self.start_cells[idx]
        else:
            pos = start_pos
        return pos[0], pos[1], 0, 0

    def _get_projected_cells(self, y, x, vy, vx):
        """Samples the line segment from (y, x) to (y + vy, x + vx) to check intersection."""
        steps = max(abs(vy), abs(vx))
        if steps == 0:
            return [(y, x)]
        cells = []
        num_samples = steps * 4
        visited = set()
        for s in range(1, num_samples + 1):
            t = s / num_samples
            cy = int(round(y + t * vy))
            cx = int(round(x + t * vx))
            if (cy, cx) not in visited:
                visited.add((cy, cx))
                cells.append((cy, cx))
        return cells

    def step(self, y, x, vy, vx, action_idx, noise=None):
        """
        Executes one step in the environment.
        Returns:
            next_y, next_x, next_vy, next_vx, reward, done, crashed
        """
        if noise is None:
            noise = self.noise
            
        ay, ax = self.actions[action_idx]
        # Stochastic velocity noise: 10% chance acceleration is (0, 0)
        if np.random.rand() < noise:
            ay, ax = 0, 0
            
        nvy = max(0, min(4, vy + ay))
        nvx = max(0, min(4, vx + ax))
        
        # Velocity components cannot both be zero unless resetting at start line
        if nvy == 0 and nvx == 0:
            nvy = max(nvy, vy)
            nvx = max(nvx, vx)
            if nvy == 0 and nvx == 0:
                nvy = 1

        path = self._get_projected_cells(y, x, nvy, nvx)
        
        # Check intersection along path: finish line has precedence over boundary
        for cy, cx in path:
            if (cy, cx) in self.finish_cells:
                # Crossed finish line! Episode ends
                return cy, cx, 0, 0, -1, True, False
            if cy < 0 or cy >= self.height or cx < 0 or cx >= self.width or self.grid[cy, cx] == 0:
                # Crash into boundary: reset to starting line with zero velocity
                idx = np.random.randint(len(self.start_cells))
                sy, sx = self.start_cells[idx]
                return sy, sx, 0, 0, -1, False, True
                
        # Arrived safely on track
        return y + nvy, x + nvx, nvy, nvx, -1, False, False


class MonteCarloRacetrack:
    """On-policy First-Visit MC Control Agent with epsilon-greedy exploration."""

    def __init__(self, env):
        self.env = env
        # Q(y, x, vy, vx, a): initialized to -100.0
        self.Q = np.full((env.height, env.width, 5, 5, env.num_actions), -100.0, dtype=np.float32)
        self.N = np.zeros((env.height, env.width, 5, 5, env.num_actions), dtype=np.int32)

    def select_action(self, y, x, vy, vx, epsilon):
        valid_acts = self.env.valid_actions[(vy, vx)]
        if len(valid_acts) == 0:
            return 0
        if np.random.rand() < epsilon:
            return np.random.choice(valid_acts)
        
        q_vals = [self.Q[y, x, vy, vx, a] for a in valid_acts]
        max_q = max(q_vals)
        best_acts = [a for a, q in zip(valid_acts, q_vals) if q == max_q]
        return np.random.choice(best_acts)

    def train(self, num_episodes=35000, eps_start=0.3, eps_end=0.02, max_steps_per_episode=400):
        print(f"\n--- Training on Track {self.env.track_id} ({num_episodes} episodes) ---")
        t0 = time.time()
        returns_history = []
        lengths_history = []

        for ep in range(1, num_episodes + 1):
            eps = max(eps_end, eps_start * (1.0 - ep / num_episodes))
            
            y, x, vy, vx = self.env.reset()
            episode = []
            
            for step_count in range(max_steps_per_episode):
                a_idx = self.select_action(y, x, vy, vx, eps)
                ny, nx, nvy, nvx, r, done, crashed = self.env.step(y, x, vy, vx, a_idx)
                episode.append((y, x, vy, vx, a_idx, r))
                if done:
                    break
                y, x, vy, vx = ny, nx, nvy, nvx

            # Record episode statistics
            ep_return = sum(item[5] for item in episode)
            returns_history.append(ep_return)
            lengths_history.append(len(episode))

            # First-visit MC Policy Evaluation and Improvement
            G = 0.0
            visited = set()
            for t in reversed(range(len(episode))):
                sy, sx, svy, svx, sa, sr = episode[t]
                G += sr
                state_action = (sy, sx, svy, svx, sa)
                if state_action not in visited:
                    visited.add(state_action)
                    self.N[sy, sx, svy, svx, sa] += 1
                    n = self.N[sy, sx, svy, svx, sa]
                    # Incremental update rule: Q <- Q + (1/n) * (G - Q)
                    self.Q[sy, sx, svy, svx, sa] += (G - self.Q[sy, sx, svy, svx, sa]) / n

            if ep % (num_episodes // 5) == 0 or ep == num_episodes:
                avg_len = np.mean(lengths_history[-500:])
                print(f"Episode {ep:6d}/{num_episodes} | Epsilon: {eps:.3f} | Recent Avg Steps: {avg_len:5.1f} | Elapsed: {time.time()-t0:.1f}s")

        print(f"Training completed in {time.time()-t0:.2f} seconds.")
        return returns_history, lengths_history

    def generate_optimal_trajectory(self, start_pos, max_steps=50):
        """Generates deterministic trajectory following optimal greedy policy (noise=0)."""
        y, x, vy, vx = self.env.reset(start_pos)
        trajectory = [(y, x, vy, vx)]
        
        for _ in range(max_steps):
            valid_acts = self.env.valid_actions[(vy, vx)]
            if len(valid_acts) == 0:
                break
            q_vals = [self.Q[y, x, vy, vx, a] for a in valid_acts]
            max_q = max(q_vals)
            best_acts = [a for a, q in zip(valid_acts, q_vals) if q == max_q]
            best_a = best_acts[0]
            
            ny, nx, nvy, nvx, r, done, crashed = self.env.step(y, x, vy, vx, best_a, noise=0.0)
            if crashed:
                return trajectory, False
            trajectory.append((ny, nx, nvy, nvx))
            if done:
                return trajectory, True
            y, x, vy, vx = ny, nx, nvy, nvx
                
        return trajectory, False


def plot_racetrack_and_trajectories(env, trajectories, track_id, save_path):
    """Visualizes the racetrack with optimal trajectories."""
    fig, ax = plt.subplots(figsize=(8, 12) if track_id == 1 else (12, 10), dpi=300)

    # 0: dark (wall), 1: light (track), 2: salmon (start), 3: lightgreen (finish)
    cmap = ListedColormap(['#2c3e50', '#ecf0f1', '#e74c3c', '#2ecc71'])
    
    # Plot grid (origin='lower', row 0 at bottom, matching book coordinates)
    ax.imshow(env.grid, origin='lower', cmap=cmap, aspect='equal')

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='#bdc3c7', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    # Color palette for different start trajectories
    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#f1c40f', '#e91e63']

    for idx, (traj, finished) in enumerate(trajectories):
        if not finished or len(traj) < 2:
            continue
        xs = [pt[1] for pt in traj]
        ys = [pt[0] for pt in traj]
        c = colors[idx % len(colors)]
        
        # Plot trajectory line and markers
        ax.plot(xs, ys, color=c, marker='o', markersize=4, linewidth=2)
        
        # Draw arrows along the path
        for i in range(len(xs) - 1):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            if dx != 0 or dy != 0:
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.5, shrinkA=3, shrinkB=3))

    ax.set_title(f"Sutton & Barto Exercise 5.12: Racetrack {track_id} Optimal Trajectories (Noise Off)", 
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Horizontal Position (X)", fontsize=11)
    ax.set_ylabel("Vertical Position (Y)", fontsize=11)

    # Legend patches
    wall_patch = mpatches.Patch(color='#2c3e50', label='Boundary / Wall')
    track_patch = mpatches.Patch(color='#ecf0f1', label='Track')
    start_patch = mpatches.Patch(color='#e74c3c', label='Starting Line')
    finish_patch = mpatches.Patch(color='#2ecc71', label='Finish Line')
    
    handles = [wall_patch, track_patch, start_patch, finish_patch]
    for idx, (traj, finished) in enumerate(trajectories):
        if finished:
            c = colors[idx % len(colors)]
            handles.append(mpatches.Patch(color=c, label=f'Trajectory from X={traj[0][1]} ({len(traj)-1} steps)'))

    ax.legend(handles=handles, loc='upper left' if track_id == 1 else 'center left', 
              bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot to: {save_path}")


def plot_learning_curves(lengths_t1, lengths_t2, save_path):
    """Plots training performance (running average steps to finish)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    def moving_average(data, window=300):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    ma_t1 = moving_average(lengths_t1, window=300)
    ma_t2 = moving_average(lengths_t2, window=300)

    ax.plot(ma_t1, label="Track 1 (Left - Narrow)", color="#2980b9", lw=1.8)
    ax.plot(ma_t2, label="Track 2 (Right - Wide)", color="#e74c3c", lw=1.8)

    ax.set_title("Monte Carlo Control Learning Curves: Racetrack Task", fontsize=13, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Running Avg Steps to Finish (Window=300)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves plot to: {save_path}")


def main():
    # Set random seed for reproducible optimal trajectories
    np.random.seed(42)

    # Ensure dedicated figures output directory exists: figures/Exercise5.12
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures", "Exercise5.12")
    os.makedirs(figures_dir, exist_ok=True)


    # 1. Train on Track 1
    env1 = RacetrackEnv(track_id=1, noise=0.1)
    agent1 = MonteCarloRacetrack(env1)
    _, lengths_t1 = agent1.train(num_episodes=35000, eps_start=0.3, eps_end=0.02)

    # Generate optimal trajectories for Track 1 from distinct start positions
    trajectories_t1 = []
    print("\n[Track 1] Evaluating optimal policy from starting line (noise off):")
    for start_cell in env1.start_cells:
        traj, finished = agent1.generate_optimal_trajectory(start_cell)
        trajectories_t1.append((traj, finished))
        status = f"SUCCESS in {len(traj)-1} steps" if finished else "CRASH / OUT OF BOUNDS"
        print(f"  Start at (y={start_cell[0]}, x={start_cell[1]}): {status}")

    plot_path_t1 = os.path.join(figures_dir, "Exercise5.12_track1_trajectories.png")
    plot_racetrack_and_trajectories(env1, trajectories_t1, track_id=1, save_path=plot_path_t1)

    # 2. Train on Track 2
    env2 = RacetrackEnv(track_id=2, noise=0.1)
    agent2 = MonteCarloRacetrack(env2)
    _, lengths_t2 = agent2.train(num_episodes=35000, eps_start=0.3, eps_end=0.02)

    # Generate optimal trajectories for Track 2 from sampled start positions
    trajectories_t2 = []
    print("\n[Track 2] Evaluating optimal policy from starting line (noise off):")
    sample_starts = [env2.start_cells[i] for i in np.linspace(0, len(env2.start_cells)-1, 6, dtype=int)]
    for start_cell in sample_starts:
        traj, finished = agent2.generate_optimal_trajectory(start_cell)
        trajectories_t2.append((traj, finished))
        status = f"SUCCESS in {len(traj)-1} steps" if finished else "CRASH / OUT OF BOUNDS"
        print(f"  Start at (y={start_cell[0]}, x={start_cell[1]}): {status}")

    plot_path_t2 = os.path.join(figures_dir, "Exercise5.12_track2_trajectories.png")
    plot_racetrack_and_trajectories(env2, trajectories_t2, track_id=2, save_path=plot_path_t2)

    # 3. Save combined learning curve
    learning_curve_path = os.path.join(figures_dir, "Exercise5.12_learning_curve.png")
    plot_learning_curves(lengths_t1, lengths_t2, learning_curve_path)

    print("\nAll tasks completed successfully!")
    print(f"Generated figures in: {figures_dir}")


if __name__ == "__main__":
    main()
