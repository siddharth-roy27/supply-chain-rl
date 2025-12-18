#!/usr/bin/env python3
"""
Phase 4: Compare RL policy vs greedy baseline on SupplyChainEnv.
"""

import argparse
import os
import sys
from typing import Optional

# Add project root to path
if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO  # noqa: E402

from env.supply_chain_env import SupplyChainEnv  # noqa: E402
from simulation.simulator import SimConfig, SupplyChainSimulator  # noqa: E402


def greedy_baseline_action(env: SupplyChainEnv) -> tuple[int, int]:
    """
    Greedy baseline: choose warehouse+vehicle minimizing travel time.
    Similar to Phase 1 simulator's greedy dispatcher.
    """
    if env.current_order is None:
        # No order to assign, return dummy action
        return 0, 0

    order = env.current_order.order
    best_wh_idx = 0
    best_v_idx = 0
    best_time = float("inf")

    for wh_idx, wh in enumerate(env.warehouses):
        if not wh.can_fulfill(order.quantity):
            continue
        for v_idx, v in enumerate(env.vehicles):
            if v.capacity < order.quantity:
                continue
            # Compute travel time
            path1 = env.network.shortest_time_path(v.node_id, wh.node_id)
            path2 = env.network.shortest_time_path(wh.node_id, order.customer_node)
            t1 = env.network.total_path_travel_time(path1)
            t2 = env.network.total_path_travel_time(path2)
            total_t = t1 + t2
            if total_t < best_time:
                best_time = total_t
                best_wh_idx = wh_idx
                best_v_idx = v_idx

    return best_wh_idx, best_v_idx


def run_episode(
    env: SupplyChainEnv,
    policy_fn,
    policy_name: str,
    seed: Optional[int] = None,
):
    """Run a single episode with given policy."""
    obs, info = env.reset(seed=seed)
    done = False
    episode_reward = 0.0
    step_count = 0

    while not done:
        if policy_name == "greedy":
            action = greedy_baseline_action(env)
        else:
            # RL policy
            action, _ = policy_fn(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        if done or truncated:
            break

    return {
        "reward": episode_reward,
        "delivered": info.get("n_delivered", 0),
        "stockouts": info.get("n_stockouts", 0),
        "steps": step_count,
    }


def compare_policies(
    model_path: Optional[str],
    n_episodes: int = 10,
    seed: Optional[int] = None,
    horizon: float = 200.0,
    n_warehouses: int = 3,
    n_customers: int = 12,
    n_vehicles: int = 5,
):
    """Compare RL policy vs greedy baseline."""
    print("=" * 60)
    print("Phase 4: Policy Comparison (RL vs Greedy Baseline)")
    print("=" * 60)

    # Load RL model if provided
    rl_policy = None
    if model_path:
        # Try both with and without .zip extension (PPO.load handles both)
        model_file = model_path
        if not os.path.exists(model_file) and os.path.exists(model_file + ".zip"):
            model_file = model_file + ".zip"
        
        if os.path.exists(model_file):
            try:
                model = PPO.load(model_file)
                rl_policy = lambda obs, deterministic=True: model.predict(obs, deterministic=deterministic)
                print(f"✅ Loaded RL model: {model_file}")
            except Exception as e:
                print(f"⚠️  Could not load RL model: {e}")
                print("   Running greedy baseline only")
                rl_policy = None
        else:
            print(f"⚠️  Model file not found: {model_path} (or {model_path}.zip)")
            print("   Running greedy baseline only")
            rl_policy = None
    else:
        print("⚠️  No RL model provided, running greedy baseline only")

    # Create config
    config = SimConfig(
        seed=seed or 42,
        sim_horizon=horizon,
        n_warehouses=n_warehouses,
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        initial_inventory=40,
        vehicle_capacity=10,
        interarrival_mean=6.0,
    )

    # Run greedy baseline
    print("\n" + "-" * 60)
    print("Running Greedy Baseline...")
    print("-" * 60)
    greedy_results = []
    for episode in range(n_episodes):
        env = SupplyChainEnv(config=config, step_dt=5.0, reward_scale=0.01)
        result = run_episode(env, None, "greedy", seed=seed + episode if seed else None)
        greedy_results.append(result)
        print(f"Episode {episode + 1}: reward={result['reward']:.2f}, delivered={result['delivered']}, stockouts={result['stockouts']}")

    # Run RL policy if available
    rl_results = []
    if rl_policy:
        print("\n" + "-" * 60)
        print("Running RL Policy...")
        print("-" * 60)
        for episode in range(n_episodes):
            env = SupplyChainEnv(config=config, step_dt=5.0, reward_scale=0.01)
            result = run_episode(env, rl_policy, "rl", seed=seed + episode if seed else None)
            rl_results.append(result)
            print(f"Episode {episode + 1}: reward={result['reward']:.2f}, delivered={result['delivered']}, stockouts={result['stockouts']}")

    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    def print_stats(name: str, results: list):
        rewards = [r["reward"] for r in results]
        delivered = [r["delivered"] for r in results]
        stockouts = [r["stockouts"] for r in results]
        print(f"\n{name}:")
        print(f"  Reward:    {sum(rewards)/len(rewards):.2f} ± {(sum((r-sum(rewards)/len(rewards))**2 for r in rewards)/len(rewards))**0.5:.2f}")
        print(f"  Delivered: {sum(delivered)/len(delivered):.1f} ± {(sum((d-sum(delivered)/len(delivered))**2 for d in delivered)/len(delivered))**0.5:.1f}")
        print(f"  Stockouts: {sum(stockouts)/len(stockouts):.1f} ± {(sum((s-sum(stockouts)/len(stockouts))**2 for s in stockouts)/len(stockouts))**0.5:.1f}")

    print_stats("Greedy Baseline", greedy_results)
    if rl_results:
        print_stats("RL Policy", rl_results)
        print("\n" + "-" * 60)
        print("Improvement (RL vs Greedy):")
        greedy_avg_reward = sum(r["reward"] for r in greedy_results) / len(greedy_results)
        rl_avg_reward = sum(r["reward"] for r in rl_results) / len(rl_results)
        improvement = ((rl_avg_reward - greedy_avg_reward) / abs(greedy_avg_reward)) * 100 if greedy_avg_reward != 0 else 0
        print(f"  Reward: {improvement:+.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare RL policy vs greedy baseline")
    parser.add_argument("--model", type=str, default=None, help="Path to trained RL model (optional)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per policy")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--horizon", type=float, default=200.0, help="Simulation horizon")
    parser.add_argument("--n-warehouses", type=int, default=3, help="Number of warehouses")
    parser.add_argument("--n-customers", type=int, default=12, help="Number of customers")
    parser.add_argument("--n-vehicles", type=int, default=5, help="Number of vehicles")
    args = parser.parse_args()

    compare_policies(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        n_warehouses=args.n_warehouses,
        n_customers=args.n_customers,
        n_vehicles=args.n_vehicles,
    )


if __name__ == "__main__":
    main()

