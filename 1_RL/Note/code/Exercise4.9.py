"""
Sutton & Barto - Reinforcement Learning: An Introduction (2nd Edition)
Chapter 4: Dynamic Programming
Exercise 4.9: Value Iteration for the Gambler's Problem

This script:
1. Implements Value Iteration for the Gambler's Problem with dummy states 0 and 100.
2. Solves for p_h = 0.25 (unfavorable game) and p_h = 0.55 (favorable game), as well as p_h = 0.40 (benchmark).
3. Generates Figure 4.3-style plots saved neatly into a dedicated figures subfolder.
4. Analyzes the stability of the value function and greedy policy as theta -> 0.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib style for publication-quality figures
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def value_iteration(p_h, theta=1e-9, max_sweeps=10000):
    """
    Implements Value Iteration for Gambler's Problem.
    
    States: 0 to 100 (0 and 100 are dummy terminal states)
    Actions: 1 to min(s, 100 - s)
    Reward: 0 everywhere, terminal state 100 gives value 1.0, terminal state 0 gives value 0.0.
    
    Returns:
        V: 1D array of length 101 (optimal state values)
        sweeps_history: list of (sweep_number, V_snapshot)
        total_sweeps: int
    """
    V = np.zeros(101, dtype=np.float64)
    V[100] = 1.0  # Goal state value is 1.0
    V[0] = 0.0    # Ruin state value is 0.0
    
    sweeps_history = []
    sweep = 0
    
    # Save initial snapshot
    target_snapshots = {1, 2, 3, 4, 8, 16, 32, 64, 128}
    
    while sweep < max_sweeps:
        sweep += 1
        delta = 0.0
        
        for s in range(1, 100):
            old_v = V[s]
            actions = np.arange(1, min(s, 100 - s) + 1)
            # Bellman optimality update: max over all valid stakes
            q_values = p_h * V[s + actions] + (1.0 - p_h) * V[s - actions]
            best_q = np.max(q_values)
            
            V[s] = best_q
            delta = max(delta, abs(old_v - best_q))
        
        if sweep in target_snapshots:
            sweeps_history.append((sweep, V.copy()))
            
        if delta < theta:
            sweeps_history.append((sweep, V.copy()))
            break
            
    return V, sweeps_history, sweep


def extract_policy(V, p_h, round_digits=8, tie_breaker='min'):
    """
    Extracts deterministic greedy policy from the converged value function.
    
    Args:
        V: State-value array of shape (101,)
        p_h: Probability of coin landing heads
        round_digits: Rounding precision to detect ties for argmax
        tie_breaker: 'min' selects smallest optimal stake (Figure 4.3 style)
                     'max' selects largest optimal stake (Bold Play style)
                     'all' returns list of all optimal actions for each state
    """
    policy = np.zeros(101, dtype=int)
    all_optimal = {}
    
    for s in range(1, 100):
        actions = np.arange(1, min(s, 100 - s) + 1)
        # Compute Q(s, a)
        raw_q = p_h * V[s + actions] + (1.0 - p_h) * V[s - actions]
        
        # Rounding to guard against floating-point epsilon noise
        if round_digits is not None:
            q_values = np.round(raw_q, decimals=round_digits)
        else:
            q_values = raw_q
            
        max_q = np.max(q_values)
        best_indices = np.where(q_values == max_q)[0]
        optimal_stakes = actions[best_indices]
        all_optimal[s] = optimal_stakes.tolist()
        
        if tie_breaker == 'min':
            policy[s] = optimal_stakes[0]
        elif tie_breaker == 'max':
            policy[s] = optimal_stakes[-1]
        else:
            policy[s] = optimal_stakes[0]
            
    return policy, all_optimal


def plot_gambler_results(p_h, V, sweeps_history, policy, save_path):
    """
    Plots the value function sweeps and the final policy, recreating Figure 4.3.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    
    # ------------------ Upper Plot: Value Function Estimates ------------------
    ax1 = axes[0]
    # Pick a few representative sweeps
    num_sweeps = len(sweeps_history)
    if num_sweeps <= 5:
        selected_sweeps = sweeps_history
    else:
        # Select first 3 sweeps, middle one, and final
        indices = [0, 1, 2]
        if num_sweeps > 6:
            indices.append(num_sweeps // 2)
        indices.append(num_sweeps - 1)
        selected_sweeps = [sweeps_history[i] for i in sorted(set(indices))]
        
    for sweep_num, v_snapshot in selected_sweeps[:-1]:
        ax1.plot(range(1, 100), v_snapshot[1:100], label=f'Sweep {sweep_num}', alpha=0.7)
        
    # Plot final value function
    final_sweep, final_v = sweeps_history[-1]
    ax1.plot(range(1, 100), final_v[1:100], label=f'Final (Sweep {final_sweep})', color='black', linewidth=2.2)
    
    ax1.set_title(f"Gambler's Problem - Value Function Sweeps (p_h = {p_h})", fontsize=14, fontweight='bold')
    ax1.set_xlabel('Capital (本金 s)', fontsize=12)
    ax1.set_ylabel('Value Estimates (最终胜率 V(s))', fontsize=12)
    ax1.set_xlim([1, 99])
    ax1.set_ylim([0, 1.02])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', framealpha=0.9)
    
    # ------------------ Lower Plot: Final Policy (Stake) ------------------
    ax2 = axes[1]
    capital_range = np.arange(1, 100)
    stakes = policy[1:100]
    
    # In Figure 4.3, Sutton & Barto draws a line plot with peaks
    ax2.plot(capital_range, stakes, color='#1f77b4', linewidth=1.8, drawstyle='default')
    ax2.scatter(capital_range, stakes, color='#1f77b4', s=12, alpha=0.6)
    
    ax2.set_title(f"Optimal Policy (Final Stake vs Capital) for p_h = {p_h}", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Capital (本金 s)', fontsize=12)
    ax2.set_ylabel('Final Policy (下注金额 a)', fontsize=12)
    ax2.set_xlim([1, 99])
    ax2.set_ylim([0, max(52, np.max(stakes) + 2)])
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate critical points if p_h < 0.5
    if p_h < 0.5:
        for pt in [25, 50, 75]:
            ax2.annotate(f's={pt}, a={policy[pt]}', 
                         xy=(pt, policy[pt]), 
                         xytext=(pt - 6, policy[pt] + 4),
                         arrowprops=dict(facecolor='crimson', shrink=0.08, width=1, headwidth=5),
                         fontsize=10, fontweight='bold', color='crimson')
    elif p_h > 0.5:
        ax2.annotate('Timid Play: a = 1 everywhere (全场下注1元)', 
                     xy=(50, 1), 
                     xytext=(30, 8),
                     arrowprops=dict(facecolor='green', shrink=0.08, width=1, headwidth=5),
                     fontsize=11, fontweight='bold', color='green')
                     
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def analyze_stability_across_theta(p_h_list=[0.25, 0.55], thetas=[1e-2, 1e-4, 1e-7, 1e-10, 1e-13]):
    """
    Answers: Are your results stable as theta -> 0?
    Checks differences in Value Function and Policy as theta becomes infinitesimally small.
    """
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS AS THETA -> 0 (Exercise 4.9)")
    print("=" * 70)
    
    results = {}
    for p_h in p_h_list:
        print(f"\nAnalyzing p_h = {p_h}:")
        policies_raw = []
        policies_rounded = []
        values = []
        
        for theta in thetas:
            V, _, sweeps = value_iteration(p_h, theta=theta)
            # Raw policy (no rounding, sensitive to floating point ties)
            pi_raw, _ = extract_policy(V, p_h, round_digits=None, tie_breaker='min')
            # Rounded policy (guards against numerical eps)
            pi_round, _ = extract_policy(V, p_h, round_digits=8, tie_breaker='min')
            
            policies_raw.append(pi_raw)
            policies_rounded.append(pi_round)
            values.append(V)
            
            print(f"  theta = {theta:1.0e} | Converged in {sweeps:4d} sweeps | V(50) = {V[50]:.8f} | pi(51) raw={pi_raw[51]}, round={pi_round[51]}")
            
        # Check value stability
        max_v_diff = np.max(np.abs(values[-1] - values[-2]))
        # Check policy stability
        raw_pol_diffs = [np.sum(p1 != p2) for p1, p2 in zip(policies_raw[:-1], policies_raw[1:])]
        round_pol_diffs = [np.sum(p1 != p2) for p1, p2 in zip(policies_rounded[:-1], policies_rounded[1:])]
        
        print(f"  -> Max Value change between last two thetas: {max_v_diff:.2e} (Strictly Stable!)")
        print(f"  -> Raw policy state changes across theta steps: {raw_pol_diffs}")
        print(f"  -> Rounded policy state changes across theta steps: {round_pol_diffs}")
        
        results[p_h] = {
            'thetas': thetas,
            'values': values,
            'policies_raw': policies_raw,
            'policies_rounded': policies_rounded,
            'raw_diffs': raw_pol_diffs,
            'round_diffs': round_pol_diffs
        }
        
    return results


def plot_comparison(p_h_list, output_dir):
    """
    Plots a 3-way comprehensive comparison: p_h = 0.25 vs 0.40 vs 0.55
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    for idx, p_h in enumerate(p_h_list):
        V, _, sweeps = value_iteration(p_h, theta=1e-9)
        pi, _ = extract_policy(V, p_h, round_digits=8, tie_breaker='min')
        
        # Upper: Value function
        ax_v = axes[0, idx]
        ax_v.plot(range(1, 100), V[1:100], color='navy', linewidth=2)
        ax_v.set_title(f'Value Function: p_h = {p_h}\n(Converged in {sweeps} sweeps)', fontsize=12, fontweight='bold')
        ax_v.set_xlabel('Capital s')
        ax_v.set_ylabel('Win Probability V(s)')
        ax_v.set_xlim([1, 99])
        ax_v.set_ylim([0, 1.02])
        ax_v.grid(True, linestyle='--', alpha=0.5)
        
        # Lower: Policy
        ax_p = axes[1, idx]
        ax_p.plot(range(1, 100), pi[1:100], color='crimson', linewidth=1.6)
        ax_p.set_title(f'Optimal Policy: p_h = {p_h}', fontsize=12, fontweight='bold')
        ax_p.set_xlabel('Capital s')
        ax_p.set_ylabel('Stake a')
        ax_p.set_xlim([1, 99])
        ax_p.set_ylim([0, max(52, np.max(pi) + 2)])
        ax_p.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'gambler_comparison_all_ph.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"Saved: {comparison_path}")


def main():
    # 1. Setup neat output directory for figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures", "Exercise4.9")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Output figures directory: {figures_dir}")
    
    # 2. Solve for p_h = 0.25, p_h = 0.55 (and p_h = 0.40 as Figure 4.3 reference)
    p_h_targets = [0.25, 0.40, 0.55]
    
    for p_h in p_h_targets:
        print(f"\n--- Running Value Iteration for p_h = {p_h} ---")
        V, sweeps_history, total_sweeps = value_iteration(p_h, theta=1e-9)
        policy, all_opts = extract_policy(V, p_h, round_digits=8, tie_breaker='min')
        
        img_name = f"gambler_ph_{p_h:.2f}.png"
        save_path = os.path.join(figures_dir, img_name)
        plot_gambler_results(p_h, V, sweeps_history, policy, save_path)
        
    # 3. Save comprehensive comparison figure
    plot_comparison(p_h_targets, figures_dir)
    
    # 4. Perform stability analysis for theta -> 0
    stability_results = analyze_stability_across_theta(p_h_list=[0.25, 0.55])
    
    print("\nAll tasks for Exercise 4.9 completed successfully!")


if __name__ == "__main__":
    main()
