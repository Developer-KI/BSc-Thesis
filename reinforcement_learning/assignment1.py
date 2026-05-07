"""
Reinforcement Learning - Assignment 1
=====================================

This script produces all the figures and numerical results for Assignment 1.

Exercise 1: Extension of the 10-armed bandit testbed (Figure 2.2 of Sutton & Barto).
            Adds optimistic initial values and UCB action selection, including a
            hyperparameter search.

Exercise 2: Implementation of value iteration on the gridworld of Seminar 4 and a
            speed comparison with policy iteration as a function of the size of the
            state space.

Exercise 3: Extension of off-policy Monte Carlo control on the Sutton & Barto
            Example 8.1 gridworld. Replaces the equiprobable behavior policy by an
            epsilon-soft policy that improves together with the action-value
            estimate.

Run:
    python assignment1.py            # runs everything (~2-4 minutes)
    python assignment1.py --quick    # smaller versions (faster, for testing)
    python assignment1.py --ex 1     # only run exercise 1
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import GridWorld as GW
import RL_utils

# Where figures will be saved
FIG_DIR = os.environ.get("FIG_DIR", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Exercise 1: 10-armed bandit testbed (vectorized)
# -----------------------------------------------------------------------------

def run_bandit_vectorized(n_runs, n_steps, method, **kwargs):
    """
    Vectorized 10-armed bandit testbed.

    For each run a fresh set of 10 true action values q*(a) ~ N(0,1) is drawn,
    and for n_steps the agent picks an arm and receives a noisy reward
    R_t ~ N(q*(A_t), 1).

    Parameters
    ----------
    n_runs : int
        Number of independent runs (rows of the vectorized state).
    n_steps : int
        Number of time steps per run.
    method : str
        One of {'epsilon_greedy', 'optimistic', 'ucb'}.
    kwargs : dict
        Method-specific hyperparameters:
          - epsilon_greedy: epsilon (float), alpha (float or None for sample average)
          - optimistic:     Q1 (float, initial value), alpha (float, default 0.1),
                            epsilon (float, default 0.0)
          - ucb:            c (float), alpha (float or None for sample average)

    Returns
    -------
    rewards : (n_steps,) ndarray
        Average reward across runs, per time step.
    optimal_action : (n_steps,) ndarray
        Fraction of runs taking the (true) optimal action, per time step.
    """
    K = 10  # number of arms
    rng = np.random.default_rng()

    # True action values per run, shape (n_runs, K)
    q_star = rng.normal(0.0, 1.0, size=(n_runs, K))
    optimal_arm = np.argmax(q_star, axis=1)  # (n_runs,)

    # Initial estimates and counts
    if method == 'optimistic':
        Q = np.full((n_runs, K), float(kwargs.get('Q1', 5.0)))
    else:
        Q = np.zeros((n_runs, K))
    N = np.zeros((n_runs, K), dtype=int)

    # Defaults per method
    if method == 'epsilon_greedy':
        epsilon = kwargs['epsilon']
        alpha = kwargs.get('alpha', None)  # None -> sample average
    elif method == 'optimistic':
        epsilon = kwargs.get('epsilon', 0.0)
        alpha = kwargs.get('alpha', 0.1)
    elif method == 'ucb':
        c = kwargs['c']
        alpha = kwargs.get('alpha', None)
    else:
        raise ValueError(f"Unknown method {method}")

    rewards = np.zeros(n_steps)
    opt_action = np.zeros(n_steps)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        # Action selection
        if method == 'ucb':
            # UCB: A = argmax [Q(a) + c * sqrt(ln(t+1) / N(a))]
            # Untried arms are picked greedily (treated as having infinite bonus).
            with np.errstate(divide='ignore', invalid='ignore'):
                bonus = c * np.sqrt(np.log(t + 1) / N)
            bonus[N == 0] = np.inf
            A = np.argmax(Q + bonus, axis=1)
        else:
            # epsilon-greedy (covers both 'epsilon_greedy' and 'optimistic')
            greedy = np.argmax(Q, axis=1)
            random_arm = rng.integers(0, K, size=n_runs)
            explore = rng.random(n_runs) < epsilon
            A = np.where(explore, random_arm, greedy)

        # Reward
        R = rng.normal(q_star[run_idx, A], 1.0)

        # Update counts and estimates
        N[run_idx, A] += 1
        if alpha is None:
            # Sample-average update
            step = 1.0 / N[run_idx, A]
        else:
            step = alpha
        Q[run_idx, A] += step * (R - Q[run_idx, A])

        rewards[t] = R.mean()
        opt_action[t] = (A == optimal_arm).mean()

    return rewards, opt_action


def hyperparameter_search(n_runs, n_steps, candidates, method, fixed_kwargs):
    """
    Sweep one hyperparameter for `method` across `candidates`. Returns the
    average reward over the second half of training (a stable performance
    measure), the full reward curves, and the optimal-action curves, for each
    candidate value.
    """
    avg_reward = np.zeros(len(candidates))
    curves = np.zeros((len(candidates), n_steps))
    opt_curves = np.zeros((len(candidates), n_steps))
    for i, val in enumerate(candidates):
        kw = dict(fixed_kwargs)
        if method == 'optimistic':
            kw['Q1'] = val
        elif method == 'ucb':
            kw['c'] = val
        elif method == 'epsilon_greedy':
            kw['epsilon'] = val
        rewards, opt = run_bandit_vectorized(n_runs, n_steps, method, **kw)
        curves[i] = rewards
        opt_curves[i] = opt
        # Average reward over the second half of training
        avg_reward[i] = rewards[n_steps // 2:].mean()
        print(f"  {method} param={val:>6}: avg reward (2nd half) = {avg_reward[i]:.4f}")
    return avg_reward, curves, opt_curves


def exercise1(n_runs=2000, n_steps=1000):
    """Exercise 1: extended Figure 2.2 with optimistic and UCB methods."""

    print("\n" + "=" * 70)
    print("Exercise 1: 10-armed bandit testbed")
    print("=" * 70)
    print(f"Settings: n_runs={n_runs}, n_steps={n_steps}")

    # --- Hyperparameter search for optimistic initial values ---
    print("\n[Hyperparameter search] Optimistic initial values "
          "(epsilon=0, alpha=0.1):")
    Q1_candidates = [0.5, 1.0, 2.0, 5.0, 10.0]
    opt_avg_reward, opt_curves, _ = hyperparameter_search(
        n_runs, n_steps, Q1_candidates, 'optimistic',
        fixed_kwargs={'epsilon': 0.0, 'alpha': 0.1}
    )
    best_Q1_idx = int(np.argmax(opt_avg_reward))
    best_Q1 = Q1_candidates[best_Q1_idx]
    print(f"  -> Best Q1 = {best_Q1}")

    # --- Hyperparameter search for UCB ---
    print("\n[Hyperparameter search] UCB (sample-average updates):")
    c_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    ucb_avg_reward, ucb_curves, _ = hyperparameter_search(
        n_runs, n_steps, c_candidates, 'ucb',
        fixed_kwargs={}
    )
    best_c_idx = int(np.argmax(ucb_avg_reward))
    best_c = c_candidates[best_c_idx]
    print(f"  -> Best c = {best_c}")

    # --- Hyperparameter sweep figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(Q1_candidates, opt_avg_reward, 'o-', color='C2')
    axes[0].axvline(best_Q1, color='gray', ls='--', alpha=0.5,
                    label=f"best Q1 = {best_Q1}")
    axes[0].set_xlabel(r"Initial value $Q_1$")
    axes[0].set_ylabel("Avg reward (2nd half of training)")
    axes[0].set_title("Optimistic initial values\n(eps=0, alpha=0.1)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(c_candidates, ucb_avg_reward, 'o-', color='C3')
    axes[1].axvline(best_c, color='gray', ls='--', alpha=0.5,
                    label=f"best c = {best_c}")
    axes[1].set_xlabel(r"Exploration constant $c$")
    axes[1].set_ylabel("Avg reward (2nd half of training)")
    axes[1].set_title("UCB action selection\n(sample-average updates)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ex1_hyperparam_search.png"), dpi=140)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR}/ex1_hyperparam_search.png")

    # --- Final figure: extended Figure 2.2 ---
    print("\n[Final runs] reproducing Figure 2.2 + optimistic + UCB:")

    # epsilon-greedy with sample-average updates (Figure 2.2 baselines)
    print("  eps=0   (greedy)")
    r0, a0 = run_bandit_vectorized(n_runs, n_steps, 'epsilon_greedy',
                                   epsilon=0.0)
    print("  eps=0.01")
    r1, a1 = run_bandit_vectorized(n_runs, n_steps, 'epsilon_greedy',
                                   epsilon=0.01)
    print("  eps=0.1")
    r2, a2 = run_bandit_vectorized(n_runs, n_steps, 'epsilon_greedy',
                                   epsilon=0.1)
    # Optimistic with best Q1
    print(f"  optimistic Q1={best_Q1}, eps=0, alpha=0.1")
    r_opt, a_opt = run_bandit_vectorized(n_runs, n_steps, 'optimistic',
                                         Q1=best_Q1, alpha=0.1, epsilon=0.0)
    # UCB with best c
    print(f"  UCB c={best_c}")
    r_ucb, a_ucb = run_bandit_vectorized(n_runs, n_steps, 'ucb', c=best_c)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(r0,   label=r"$\varepsilon=0$ (greedy)", color='C7')
    axes[0].plot(r1,   label=r"$\varepsilon=0.01$",       color='C0')
    axes[0].plot(r2,   label=r"$\varepsilon=0.1$",        color='C1')
    axes[0].plot(r_opt, label=fr"Optimistic, $Q_1={best_Q1}$, $\alpha=0.1$",
                 color='C2')
    axes[0].plot(r_ucb, label=fr"UCB, $c={best_c}$", color='C3')
    axes[0].set_ylabel("Average reward")
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"10-armed bandit testbed "
                      f"({n_runs} runs, {n_steps} steps)")

    axes[1].plot(100 * a0,   color='C7')
    axes[1].plot(100 * a1,   color='C0')
    axes[1].plot(100 * a2,   color='C1')
    axes[1].plot(100 * a_opt, color='C2')
    axes[1].plot(100 * a_ucb, color='C3')
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("% Optimal action")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ex1_extended_figure22.png"), dpi=140)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR}/ex1_extended_figure22.png")


# -----------------------------------------------------------------------------
# Exercise 2: Value iteration vs Policy iteration
# -----------------------------------------------------------------------------

def iterative_policy_evaluation(gridworld, policy, gamma=0.95, theta=1e-8,
                                v_initial=None, max_iter=100000):
    """
    In-place iterative policy evaluation (Sutton & Barto p. 75).
    """
    if v_initial is None:
        v_ = np.zeros(gridworld.nstates)
    else:
        v_ = v_initial.copy()
    k = 0
    while True:
        delta = 0.0
        for s in gridworld.states:
            v_old = v_[s]
            new_v = 0.0
            for i, a in enumerate(gridworld.actions):
                pi = policy[s, i]
                if pi == 0.0:
                    continue
                inner = 0.0
                for (s_prime, r), p in gridworld.p[(s, a)].items():
                    inner += p * (r + gamma * v_[s_prime])
                new_v += pi * inner
            v_[s] = new_v
            delta = max(delta, abs(v_old - new_v))
        k += 1
        if delta < theta or k >= max_iter:
            break
    return v_, k


def policy_iteration(gridworld, gamma=0.95, theta=1e-8, seed=0,
                     max_iter=10000):
    """
    Policy iteration with iterative policy evaluation (Sutton & Barto p. 80).
    Uses argmax (deterministic tie-breaking) and a policy-based stopping
    criterion.

    Returns
    -------
    policy, v : the converged greedy policy and its value function.
    iterations : number of outer (policy-improvement) iterations.
    total_sweeps : total number of Bellman sweeps (outer * inner sweeps).
    """
    policy = RL_utils.generate_random_policy(gridworld, deterministic=True,
                                             seed=seed)
    v = np.zeros(gridworld.nstates)
    iterations = 0
    total_sweeps = 0
    while iterations < max_iter:
        # 1. Policy evaluation
        v, k_inner = iterative_policy_evaluation(gridworld, policy,
                                                 gamma=gamma, theta=theta,
                                                 v_initial=v)
        total_sweeps += k_inner
        # 2. Policy improvement
        new_policy = RL_utils.greedy_policy(gridworld, v, gamma=gamma,
                                            use_argmax=True)
        iterations += 1
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, v, iterations, total_sweeps


def value_iteration(gridworld, gamma=0.95, theta=1e-8, max_iter=100000):
    """
    Value iteration (Sutton & Barto p. 83).

    Repeatedly applies the Bellman optimality update
        v(s) <- max_a sum_{s', r} p(s', r | s, a) [ r + gamma * v(s') ]
    until the largest change in v across one sweep is below theta. After
    convergence, the greedy policy with respect to v is returned.
    """
    v = np.zeros(gridworld.nstates)
    history = [v.copy()]
    k = 0
    while True:
        delta = 0.0
        for s in gridworld.states:
            v_old = v[s]
            best = -np.inf
            for a in gridworld.actions:
                inner = 0.0
                for (s_prime, r), p in gridworld.p[(s, a)].items():
                    inner += p * (r + gamma * v[s_prime])
                if inner > best:
                    best = inner
            v[s] = best
            delta = max(delta, abs(v_old - best))
        k += 1
        history.append(v.copy())
        if delta < theta or k >= max_iter:
            break
    policy = RL_utils.greedy_policy(gridworld, v, gamma=gamma, use_argmax=True)
    return policy, v, k, history


def make_gridworld(n):
    """
    Build an n x n WaterFireGridWorld with two terminal states.

    The two terminal states are placed in the interior of the grid (one cell
    away from the borders). This avoids a subtle issue in
    ``GridWorld.update_p`` where actions in a terminal state located on the
    grid boundary may incorrectly receive ``reward_to`` instead of
    ``reward_from`` (because for a boundary cell ``_default_s_new(s, a)``
    sometimes returns ``s`` itself, which is a special state). Placing the
    terminals in the interior side-steps this corner case entirely.
    """
    nrows, ncols = n, n
    water_rc = (nrows - 2, ncols - 2)        # one in from bottom-right
    fire_rc = (nrows // 2, ncols // 2)       # roughly centred
    water_state = water_rc[0] * ncols + water_rc[1]
    fire_state = fire_rc[0] * ncols + fire_rc[1]
    if fire_state == water_state:
        fire_state -= 1
    special_states = [
        GW.WaterState(location=water_state),
        GW.FireState(location=fire_state),
    ]
    return GW.GridWorld(grid_size=(n, n), default_reward=-1.0,
                        special_states=special_states)


def exercise2(grid_sizes=(5, 8, 12, 16, 20, 25), n_trials=3, gamma=0.95,
              theta=1e-6):
    """Exercise 2: value iteration vs policy iteration speed comparison."""

    print("\n" + "=" * 70)
    print("Exercise 2: Value iteration vs Policy iteration")
    print("=" * 70)

    # --- 2a: Show that value iteration is learning ---
    print("\n[2a] Demonstrate value iteration learning on a 5x5 gridworld.")
    gw = make_gridworld(5)
    policy_vi, v_vi, k_vi, history = value_iteration(gw, gamma=gamma,
                                                     theta=1e-10)
    print(f"  Value iteration converged in k = {k_vi} sweeps.")

    # Plot: convergence of v as a function of sweeps + final v + final policy.
    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    history_arr = np.array(history)  # (k+1, nstates)
    # Plot only non-terminal-state value trajectories
    for s in range(gw.nstates):
        if s in gw.special_states and gw.special_states[s].terminal:
            continue
        ax1.plot(history_arr[:, s], alpha=0.7)
    ax1.set_xlabel("Sweep")
    ax1.set_ylabel("V(s)")
    ax1.set_title(f"Value iteration: v(s) per sweep\n"
                  f"(converged in {k_vi} sweeps)")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 3, 2)
    gw.plot_v(ax=ax2, v=v_vi, title="Optimal v* (value iteration)")
    ax3 = fig.add_subplot(1, 3, 3)
    gw.plot_policy(ax=ax3, policy=policy_vi, title="Greedy policy w.r.t. v*")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ex2a_value_iteration.png"), dpi=140)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR}/ex2a_value_iteration.png")

    # --- 2b: Speed comparison as a function of grid size ---
    print("\n[2b] Speed comparison vs grid size (policy iteration"
          " vs value iteration).")
    pi_times, vi_times = [], []
    pi_outer, pi_sweeps, vi_sweeps = [], [], []
    for n in grid_sizes:
        gw = make_gridworld(n)
        # Policy iteration
        t_pi, outer, sweeps = [], [], []
        for trial in range(n_trials):
            t0 = time.perf_counter()
            _, _, k_outer, k_total = policy_iteration(gw, gamma=gamma,
                                                      theta=theta, seed=trial)
            t_pi.append(time.perf_counter() - t0)
            outer.append(k_outer)
            sweeps.append(k_total)
        # Value iteration (deterministic, run once is enough but average for
        # consistency)
        t_vi, vsweeps = [], []
        for trial in range(n_trials):
            t0 = time.perf_counter()
            _, _, k, _ = value_iteration(gw, gamma=gamma, theta=theta)
            t_vi.append(time.perf_counter() - t0)
            vsweeps.append(k)
        pi_times.append(np.mean(t_pi))
        vi_times.append(np.mean(t_vi))
        pi_outer.append(np.mean(outer))
        pi_sweeps.append(np.mean(sweeps))
        vi_sweeps.append(np.mean(vsweeps))
        print(f"  n={n:2d} (|S|={n*n:4d}): "
              f"PI {pi_times[-1]:.3f}s "
              f"({pi_outer[-1]:.1f} outer iter, "
              f"{pi_sweeps[-1]:.0f} total sweeps), "
              f"VI {vi_times[-1]:.3f}s ({vi_sweeps[-1]:.0f} sweeps)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    state_counts = [n * n for n in grid_sizes]
    axes[0].plot(state_counts, pi_times, 'o-', label="Policy iteration",
                 color='C0')
    axes[0].plot(state_counts, vi_times, 's-', label="Value iteration",
                 color='C3')
    axes[0].set_xlabel(r"Number of states $|S| = n^2$")
    axes[0].set_ylabel("Wall-clock time per run (s)")
    axes[0].set_title(f"Speed comparison (avg of {n_trials} trials)")
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].plot(state_counts, pi_sweeps, 'o-',
                 label="Policy iter (total Bellman sweeps)", color='C0')
    axes[1].plot(state_counts, pi_outer, '^--',
                 label="Policy iter (outer iterations)", color='C0',
                 alpha=0.5)
    axes[1].plot(state_counts, vi_sweeps, 's-',
                 label="Value iter (Bellman sweeps)", color='C3')
    axes[1].set_xlabel(r"Number of states $|S| = n^2$")
    axes[1].set_ylabel("Number of iterations")
    axes[1].set_title("Number of Bellman sweeps until convergence")
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ex2b_speed_comparison.png"), dpi=140)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR}/ex2b_speed_comparison.png")


# -----------------------------------------------------------------------------
# Exercise 3: Off-policy MC with a better behavior policy
# -----------------------------------------------------------------------------

def epsilon_soft_policy(Q, epsilon):
    """
    Build an epsilon-soft policy from an action-value function Q.

    Each non-greedy action gets probability ``epsilon/|A|``; the remaining
    probability ``1 - epsilon`` is distributed uniformly across all actions
    tied for the maximum Q-value (so when Q is initialised to zeros the
    behavior policy is just uniform). This avoids the pathological case where
    np.argmax always picks the same action when several actions have the same
    value.
    """
    nstates, nactions = Q.shape
    pi = np.full_like(Q, epsilon / nactions)
    max_q = Q.max(axis=1, keepdims=True)
    is_max = (Q == max_q)
    n_max = is_max.sum(axis=1, keepdims=True)
    pi += (1.0 - epsilon) * is_max / n_max
    return pi


def off_policy_mc_control(gridworld, n_episodes=500, gamma=0.95,
                          behavior='equiprobable', epsilon=0.1,
                          max_steps_per_episode=100000, rng=None):
    """
    Off-policy Monte Carlo control with weighted importance sampling
    (Sutton & Barto p. 111).

    Parameters
    ----------
    behavior : {'equiprobable', 'epsilon_soft'}
        - 'equiprobable': uniform random behavior policy (the default of the
          original notebook).
        - 'epsilon_soft': at the start of each episode the behavior policy is
          set to be epsilon-soft with respect to the *current* action-value
          estimate Q. This both makes the trajectories shorter and reduces the
          importance-sampling weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    nactions = len(gridworld.actions)
    Q = np.zeros((gridworld.nstates, nactions))
    C = np.zeros_like(Q)
    # Target policy is greedy w.r.t. Q (deterministic).
    target_action = np.argmax(Q, axis=1)

    # We track the length of each generated episode.
    steps_per_episode = np.zeros(n_episodes, dtype=int)

    for ep in range(n_episodes):
        # Choose behavior policy for this episode
        if behavior == 'equiprobable':
            b = np.full((gridworld.nstates, nactions), 1.0 / nactions)
        elif behavior == 'epsilon_soft':
            b = epsilon_soft_policy(Q, epsilon)
        else:
            raise ValueError(f"Unknown behavior '{behavior}'")

        # Generate an episode using b
        s = gridworld.initial_state
        states, actions, rewards = [], [], []
        for _ in range(max_steps_per_episode):
            if s == gridworld.terminal_state:
                break
            a_idx = rng.choice(nactions, p=b[s])
            a = gridworld.actions[a_idx]
            s_prime, r = gridworld.interact(s, a)
            states.append(s)
            actions.append(a_idx)
            rewards.append(r)
            s = s_prime
        T = len(rewards)
        steps_per_episode[ep] = T

        # Update Q and target policy by walking backward through the episode.
        G = 0.0
        W = 1.0
        for t in range(T - 1, -1, -1):
            G = gamma * G + rewards[t]
            s_t, a_t = states[t], actions[t]
            C[s_t, a_t] += W
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            target_action[s_t] = int(np.argmax(Q[s_t]))
            if a_t != target_action[s_t]:
                break
            W /= b[s_t, a_t]

    # Final greedy policy (one-hot).
    policy = np.zeros_like(Q)
    policy[np.arange(gridworld.nstates), np.argmax(Q, axis=1)] = 1.0
    return Q, policy, steps_per_episode


def exercise3(n_runs=30, n_episodes=300, gamma=0.95, epsilon=0.1, seed=0,
              max_steps_per_episode=20000):
    """
    Exercise 3: compare equiprobable behavior policy with an epsilon-soft
    behavior policy that adapts to the current Q estimate.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Off-policy MC control - choice of behavior policy")
    print("=" * 70)
    print(f"Settings: n_runs={n_runs}, n_episodes={n_episodes}, "
          f"gamma={gamma}, epsilon={epsilon}")
    print("Environment: Sutton & Barto Example 8.1 (6x9 maze, default reward 0,"
          " terminal reward +1).")

    # Build the maze once for plotting; for each run we re-instantiate so the
    # internal RNG of `interact` is freshly seeded via numpy's global RNG.
    gw_show = GW.Sutton_Barto_Example8_1()
    fig, ax = plt.subplots(figsize=(7, 4))
    gw_show.plot_gridworld(ax=ax,
                           title="Sutton & Barto Example 8.1 maze (6x9)",
                           print_states=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ex3_maze.png"), dpi=140)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR}/ex3_maze.png")

    # Run both methods n_runs times and average the steps-per-episode curves.
    results = {}
    for label, kwargs in [
        ('equiprobable', dict(behavior='equiprobable')),
        ('epsilon_soft', dict(behavior='epsilon_soft', epsilon=epsilon)),
    ]:
        print(f"\nRunning behavior='{label}' ({n_runs} runs of "
              f"{n_episodes} episodes)...")
        steps_all = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            rng = np.random.default_rng(seed + run)
            # Seed numpy's global RNG used inside Sutton_Barto_Example8_1.interact
            np.random.seed(seed + run)
            gw = GW.Sutton_Barto_Example8_1()
            _, _, steps = off_policy_mc_control(
                gw, n_episodes=n_episodes, gamma=gamma,
                max_steps_per_episode=max_steps_per_episode, rng=rng,
                **kwargs)
            steps_all[run] = steps
            if (run + 1) % max(1, n_runs // 5) == 0:
                print(f"  run {run + 1:3d}/{n_runs}: "
                      f"avg steps over last 50 episodes = "
                      f"{steps_all[run, -50:].mean():.1f}")
        results[label] = steps_all

    # Plot averaged curves with standard-error band on log scale
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'equiprobable': 'C0', 'epsilon_soft': 'C3'}
    nice_labels = {
        'equiprobable': 'Equiprobable behavior policy',
        'epsilon_soft': fr'$\varepsilon$-soft behavior policy '
                        fr'(eps={epsilon}, derived from $Q$)',
    }

    for ax, ylabel, log in [(axes[0], "Steps per episode (linear)", False),
                            (axes[1], "Steps per episode (log)",    True)]:
        for label, steps_all in results.items():
            mean = steps_all.mean(axis=0)
            sem = steps_all.std(axis=0) / np.sqrt(n_runs)
            ax.plot(mean, label=nice_labels[label], color=colors[label])
            ax.fill_between(np.arange(n_episodes), mean - sem, mean + sem,
                            color=colors[label], alpha=0.2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3, which='both')
        if log:
            ax.set_yscale('log')

    fig.suptitle(f"Off-policy MC control on Sutton-Barto Example 8.1 "
                 f"(averaged over {n_runs} runs)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ex3_steps_per_episode.png"), dpi=140)
    plt.close(fig)
    print(f"  -> saved {FIG_DIR}/ex3_steps_per_episode.png")

    # Print a concise summary
    print("\nFinal performance (mean steps over last 50 episodes):")
    for label, steps_all in results.items():
        last = steps_all[:, -50:].mean()
        print(f"  {label:14s}: {last:8.1f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex", type=str, default="all",
                        help="Which exercise(s) to run: '1', '2', '3', or 'all'.")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller settings for a quick smoke-test.")
    args = parser.parse_args()

    if args.quick:
        ex1_kwargs = dict(n_runs=400, n_steps=500)
        ex2_kwargs = dict(grid_sizes=(5, 8, 12, 16), n_trials=2)
        ex3_kwargs = dict(n_runs=10, n_episodes=150,
                          max_steps_per_episode=5000)
    else:
        ex1_kwargs = dict(n_runs=2000, n_steps=1000)
        ex2_kwargs = dict(grid_sizes=(5, 8, 12, 16, 20, 25), n_trials=3)
        ex3_kwargs = dict(n_runs=20, n_episodes=300,
                          max_steps_per_episode=5000)

    t0 = time.perf_counter()
    if args.ex in ("all", "1"):
        exercise1(**ex1_kwargs)
    if args.ex in ("all", "2"):
        exercise2(**ex2_kwargs)
    if args.ex in ("all", "3"):
        exercise3(**ex3_kwargs)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f} s")
    print(f"All figures saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()