"""
Exercise 4.7 (programming) - Jack's Car Rental Problem with Non-linearities
Book: Reinforcement Learning: An Introduction (Sutton & Barto, 2nd Edition)

This program implements Policy Iteration to solve:
1. The original Jack's Car Rental problem (replicating Figure 4.2).
2. The modified problem with realistic non-linearities:
   - One car can be shuttled from Location 1 to Location 2 for free by an employee.
   - A parking cost of $4 is incurred if more than 10 cars are kept overnight at either location.
"""

import math
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Problem Parameters
MAX_CARS = 20
MAX_MOVE = 5
GAMMA = 0.9
RENTAL_REWARD = 10.0
MOVE_COST = 2.0
PARKING_THRESHOLD = 10
PARKING_COST = 4.0

# Poisson parameter expectations
LAMBDA_RENTAL_1 = 3
LAMBDA_RENTAL_2 = 4
LAMBDA_RETURN_1 = 3
LAMBDA_RETURN_2 = 2

# Truncate Poisson distribution to limit computation while preserving 99.9%+ probability
POISSON_UPPER_BOUND = 11

# Cache Poisson probabilities
_poisson_cache = {}


def poisson_prob(n, lam):
    key = (n, lam)
    if key not in _poisson_cache:
        _poisson_cache[key] = (lam**n / math.factorial(n)) * math.exp(-lam)
    return _poisson_cache[key]


def precompute_location(lam_req, lam_ret):
    """
    Precomputes expected rental reward and state transition probabilities
    for a single location given the morning car count n_prime (0 to 20).
    
    Returns:
        exp_reward: shape (21,), expected daytime rental revenue for n_prime cars.
        transitions: shape (21, 21), P(n_end | n_prime) probability distribution.
    """
    exp_reward = np.zeros(MAX_CARS + 1)
    transitions = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    for n_prime in range(MAX_CARS + 1):
        for req in range(POISSON_UPPER_BOUND):
            p_req = poisson_prob(req, lam_req)
            rented = min(n_prime, req)
            exp_reward[n_prime] += p_req * RENTAL_REWARD * rented

            for ret in range(POISSON_UPPER_BOUND):
                p_ret = poisson_prob(ret, lam_ret)
                p = p_req * p_ret
                # Car count at the end of the day cannot exceed MAX_CARS
                n_end = min(n_prime - rented + ret, MAX_CARS)
                transitions[n_prime, n_end] += p

        # Normalize to account for truncated Poisson tail
        transitions[n_prime] /= np.sum(transitions[n_prime])

    return exp_reward, transitions


def get_action_cost(n1_prime, n2_prime, a, is_modified=False):
    """
    Computes overnight moving cost and parking cost.
    a: number of cars moved from Loc 1 to Loc 2 (-5 to +5).
    n1_prime, n2_prime: car counts after overnight moves.
    """
    if not is_modified:
        # Original problem: each moved car costs $2
        return MOVE_COST * abs(a)
    else:
        # Exercise 4.7 modifications:
        # 1. Employee shuttles 1 car from Loc 1 to Loc 2 for free
        if a > 0:
            move_cost = MOVE_COST * (a - 1)
        else:
            move_cost = MOVE_COST * abs(a)

        # 2. Parking cost: if more than 10 cars are kept overnight at a location
        park_cost = 0.0
        if n1_prime > PARKING_THRESHOLD:
            park_cost += PARKING_COST
        if n2_prime > PARKING_THRESHOLD:
            park_cost += PARKING_COST

        return move_cost + park_cost


def policy_iteration(r1, t1, r2, t2, is_modified=False, theta=1e-4):
    """
    Standard Policy Iteration algorithm for Jack's Car Rental.
    Uses matrix-vector precomputations for blazingly fast Policy Evaluation.
    """
    V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
    policy_history = [pi.copy()]

    iteration = 0
    start_time = time.time()

    while True:
        iteration += 1
        eval_sweeps = 0

        # --- 1. Policy Evaluation ---
        while True:
            eval_sweeps += 1
            # Vectorized expected future value: EV_all[n1', n2'] = t1[n1', :] @ V @ t2[n2', :].T
            EV_all = t1 @ V @ t2.T

            new_V = np.zeros_like(V)
            delta = 0.0

            for n1 in range(MAX_CARS + 1):
                for n2 in range(MAX_CARS + 1):
                    a = pi[n1, n2]
                    n1_p = min(n1 - a, MAX_CARS)
                    n2_p = min(n2 + a, MAX_CARS)

                    cost = get_action_cost(n1_p, n2_p, a, is_modified)
                    reward = -cost + r1[n1_p] + r2[n2_p]

                    v_new = reward + GAMMA * EV_all[n1_p, n2_p]
                    delta = max(delta, abs(v_new - V[n1, n2]))
                    new_V[n1, n2] = v_new

            V = new_V
            if delta < theta:
                break

        # --- 2. Policy Improvement ---
        EV_all = t1 @ V @ t2.T
        policy_stable = True
        new_pi = np.zeros_like(pi)

        for n1 in range(MAX_CARS + 1):
            for n2 in range(MAX_CARS + 1):
                old_action = pi[n1, n2]

                # Feasible action range: cannot move more cars than available
                min_a = max(-MAX_MOVE, -n2)
                max_a = min(MAX_MOVE, n1)

                best_q = -1e9
                best_a = old_action

                for a in range(min_a, max_a + 1):
                    n1_p = min(n1 - a, MAX_CARS)
                    n2_p = min(n2 + a, MAX_CARS)

                    cost = get_action_cost(n1_p, n2_p, a, is_modified)
                    reward = -cost + r1[n1_p] + r2[n2_p]
                    q = reward + GAMMA * EV_all[n1_p, n2_p]

                    # Break ties in favor of old_action (Exercise 4.4 fix)
                    if q > best_q + 1e-7:
                        best_q = q
                        best_a = a
                    elif abs(q - best_q) <= 1e-7 and a == old_action:
                        best_a = old_action

                new_pi[n1, n2] = best_a
                if best_a != old_action:
                    policy_stable = False

        pi = new_pi
        policy_history.append(pi.copy())
        print(f"  Iteration {iteration}: eval sweeps = {eval_sweeps:3d}, policy_stable = {policy_stable}")

        if policy_stable:
            break

    total_time = time.time() - start_time
    print(f"  Converged in {iteration} iterations ({total_time:.2f}s).\n")
    return policy_history, V


def plot_results(policy_history, V, title_prefix, filename):
    """
    Visualizes the sequence of policies and the final 3D value function,
    faithfully replicating the layout of Figure 4.2.
    """
    n_policies = len(policy_history)
    # Arrange subplots: e.g. 2 rows x 3 cols or 2 rows x 4 cols
    cols = min(n_policies + 1, 3)
    rows = math.ceil((n_policies + 1) / cols)

    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    plt.suptitle(f"Jack's Car Rental - {title_prefix}", fontsize=16, y=0.98)

    # Plot each policy as a contour / filled contour plot
    for i, pi in enumerate(policy_history):
        ax = fig.add_subplot(rows, cols, i + 1)
        # X: Cars at Loc 2, Y: Cars at Loc 1
        X, Y = np.meshgrid(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1))

        # Discrete levels from -5 to 5
        levels = np.arange(-5.5, 6.5, 1)
        cs = ax.contourf(X, Y, pi, levels=levels, cmap="coolwarm", alpha=0.85)
        contours = ax.contour(X, Y, pi, levels=np.arange(-5, 6, 1), colors="black", linewidths=0.75)
        ax.clabel(contours, inline=True, fontsize=9, fmt="%d")

        ax.set_title(r"$\pi_{%d}$%s" % (i, " (Optimal)" if i == n_policies - 1 else ""), fontsize=12)
        ax.set_xlabel("# Cars at second location")
        ax.set_ylabel("# Cars at first location")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_aspect("equal")

    # Plot final value function as 3D surface
    ax_3d = fig.add_subplot(rows, cols, n_policies + 1, projection="3d")
    X, Y = np.meshgrid(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1))
    surf = ax_3d.plot_surface(X, Y, V, cmap="viridis", edgecolor="none", alpha=0.9)
    ax_3d.set_title(r"$v_{\pi_*}$ (Optimal Value)", fontsize=12)
    ax_3d.set_xlabel("Loc 2")
    ax_3d.set_ylabel("Loc 1")
    ax_3d.set_zlabel("Value ($)")
    fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    plt.savefig(output_path, dpi=300)
    print(f"  Figure saved to: {output_path}")
    plt.close()


def plot_comparison(pi_orig, pi_mod, v_orig, v_mod, filename="comparison_jacks_car_rental.png"):
    """
    Direct side-by-side comparison of original vs modified optimal policy and value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle("Exercise 4.7 Comparison: Original vs Modified Problem", fontsize=15)

    X, Y = np.meshgrid(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1))
    levels = np.arange(-5.5, 6.5, 1)

    # 1. Original Optimal Policy
    cs0 = axes[0].contourf(X, Y, pi_orig, levels=levels, cmap="coolwarm", alpha=0.85)
    c0 = axes[0].contour(X, Y, pi_orig, levels=np.arange(-5, 6, 1), colors="black", linewidths=0.75)
    axes[0].clabel(c0, inline=True, fontsize=9, fmt="%d")
    axes[0].set_title("Original Optimal Policy $\pi_*$ (Ex 4.2)")
    axes[0].set_xlabel("# Cars at second location")
    axes[0].set_ylabel("# Cars at first location")
    axes[0].set_aspect("equal")

    # 2. Modified Optimal Policy
    cs1 = axes[1].contourf(X, Y, pi_mod, levels=levels, cmap="coolwarm", alpha=0.85)
    c1 = axes[1].contour(X, Y, pi_mod, levels=np.arange(-5, 6, 1), colors="black", linewidths=0.75)
    axes[1].clabel(c1, inline=True, fontsize=9, fmt="%d")
    axes[1].set_title("Modified Optimal Policy $\pi_*$ (Ex 4.7)")
    axes[1].set_xlabel("# Cars at second location")
    axes[1].set_ylabel("# Cars at first location")
    axes[1].set_aspect("equal")

    # 3. Policy Difference (Modified - Original)
    diff = pi_mod - pi_orig
    diff_levels = np.arange(diff.min() - 0.5, diff.max() + 1.5, 1)
    cs2 = axes[2].contourf(X, Y, diff, levels=diff_levels, cmap="PuOr", alpha=0.85)
    c2 = axes[2].contour(X, Y, diff, levels=np.arange(diff.min(), diff.max() + 1, 1), colors="black", linewidths=0.75)
    axes[2].clabel(c2, inline=True, fontsize=9, fmt="%d")
    axes[2].set_title("Action Difference ($\pi_{mod} - \pi_{orig}$)")
    axes[2].set_xlabel("# Cars at second location")
    axes[2].set_ylabel("# Cars at first location")
    axes[2].set_aspect("equal")
    fig.colorbar(cs2, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    plt.savefig(output_path, dpi=300)
    print(f"  Comparison figure saved to: {output_path}")
    plt.close()


def main():
    print("=" * 65)
    print("  Sutton & Barto RL Chapter 4 - Exercise 4.7 (Programming)")
    print("  Jack's Car Rental Problem & Policy Iteration")
    print("=" * 65)

    print("\n[Step 1] Precomputing daytime rental rewards and transition models...")
    t_start = time.time()
    r1, t1 = precompute_location(LAMBDA_RENTAL_1, LAMBDA_RETURN_1)
    r2, t2 = precompute_location(LAMBDA_RENTAL_2, LAMBDA_RETURN_2)
    print(f"  Precomputation finished in {time.time() - t_start:.2f}s.")

    print("\n[Step 2] Solving Original Problem (Example 4.2)...")
    policies_orig, v_orig = policy_iteration(r1, t1, r2, t2, is_modified=False)
    plot_results(policies_orig, v_orig, "Original Problem (Example 4.2)", "original_jacks_car_rental.png")

    print("\n[Step 3] Solving Modified Problem (Exercise 4.7 with Non-linearities)...")
    policies_mod, v_mod = policy_iteration(r1, t1, r2, t2, is_modified=True)
    plot_results(policies_mod, v_mod, "Modified Problem (Exercise 4.7)", "modified_jacks_car_rental.png")

    print("\n[Step 4] Plotting comparison between Original and Modified policies...")
    plot_comparison(policies_orig[-1], policies_mod[-1], v_orig, v_mod, "comparison_jacks_car_rental.png")

    print("\n[Analysis of Differences]")
    print("-" * 65)
    print("1. Free shuttle effect (Loc 1 -> Loc 2):")
    print("   Moving 1 car from Loc 1 to Loc 2 is now free. This shifts the 'move 1 car' region")
    print("   and makes Jack more willing to send at least 1 car to Location 2, even when")
    print("   the car differential is small.")
    print("2. Parking space constraint (Overnight cars > 10 costs $4 extra):")
    print("   When either location has more than 10 cars overnight, an extra $4 fee is incurred.")
    print("   This creates a sharp threshold around n = 10, encouraging Jack to move cars away")
    print("   from a location that exceeds 10 cars to avoid the $4 penalty.")
    print("=" * 65)
    print("  All tasks completed successfully!")


if __name__ == "__main__":
    main()
