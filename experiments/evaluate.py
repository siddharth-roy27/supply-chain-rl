#!/usr/bin/env python3
"""
Phase 4: Evaluate trained PPO model on SupplyChainEnv.
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
from simulation.simulator import SimConfig  # noqa: E402


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    seed: Optional[int] = None,
    horizon: float = 200.0,
    n_warehouses: int = 3,
    n_customers: int = 12,
    n_vehicles: int = 5,
    render: bool = False,
):
    """Evaluate a trained PPO model."""
    print("=" * 60)
    print("Phase 4: Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("=" * 60)

    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create environment
    config = SimConfig(
        seed=seed or 999,
        sim_horizon=horizon,
        n_warehouses=n_warehouses,
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        initial_inventory=40,
        vehicle_capacity=10,
        interarrival_mean=6.0,
    )
    env = SupplyChainEnv(config=config, step_dt=5.0, reward_scale=0.01, render_mode="human" if render else None)

    # Evaluation metrics
    episode_rewards = []
    episode_delivered = []
    episode_stockouts = []
    episode_distances = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode if seed else None)
        done = False
        episode_reward = 0.0
        episode_distance = 0.0

        step_count = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if render:
                env.render()

            if done or truncated:
                break

        # Collect metrics
        episode_rewards.append(episode_reward)
        episode_delivered.append(info.get("n_delivered", 0))
        episode_stockouts.append(info.get("n_stockouts", 0))

        # Calculate average distance from completed orders
        if env.completed_orders:
            avg_dist = sum(o.travel_distance for o in env.completed_orders if o.delivered) / len(
                [o for o in env.completed_orders if o.delivered]
            )
            episode_distances.append(avg_dist)
        else:
            episode_distances.append(0.0)

        print(f"\nEpisode {episode + 1}/{n_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Delivered: {info.get('n_delivered', 0)}")
        print(f"  Stockouts: {info.get('n_stockouts', 0)}")
        print(f"  Steps: {step_count}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"\nRewards:")
    print(f"  Mean: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"  Std:  {(sum((r - sum(episode_rewards)/len(episode_rewards))**2 for r in episode_rewards) / len(episode_rewards))**0.5:.2f}")
    print(f"  Min:  {min(episode_rewards):.2f}")
    print(f"  Max:  {max(episode_rewards):.2f}")
    print(f"\nDeliveries:")
    print(f"  Mean: {sum(episode_delivered) / len(episode_delivered):.1f}")
    print(f"  Total: {sum(episode_delivered)}")
    print(f"\nStockouts:")
    print(f"  Mean: {sum(episode_stockouts) / len(episode_stockouts):.1f}")
    print(f"  Total: {sum(episode_stockouts)}")
    print(f"\nDistance:")
    print(f"  Mean: {sum(episode_distances) / len(episode_distances):.2f}")
    print("=" * 60)

    return {
        "rewards": episode_rewards,
        "delivered": episode_delivered,
        "stockouts": episode_stockouts,
        "distances": episode_distances,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--horizon", type=float, default=200.0, help="Simulation horizon")
    parser.add_argument("--n-warehouses", type=int, default=3, help="Number of warehouses")
    parser.add_argument("--n-customers", type=int, default=12, help="Number of customers")
    parser.add_argument("--n-vehicles", type=int, default=5, help="Number of vehicles")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        n_warehouses=args.n_warehouses,
        n_customers=args.n_customers,
        n_vehicles=args.n_vehicles,
        render=args.render,
    )


if __name__ == "__main__":
    main()

