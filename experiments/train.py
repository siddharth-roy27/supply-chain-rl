#!/usr/bin/env python3
"""
Phase 4: Train PPO agent on SupplyChainEnv using Stable-Baselines3.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

# Add project root to path
if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from env.supply_chain_env import SupplyChainEnv  # noqa: E402
from simulation.simulator import SimConfig  # noqa: E402


def make_env(config: SimConfig, rank: int = 0, seed: int = 0):
    """Create and wrap environment for training."""
    def _init():
        env = SupplyChainEnv(config=config, step_dt=5.0, reward_scale=0.01)
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on SupplyChainEnv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--save-freq", type=int, default=10000, help="Model checkpoint frequency")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--log-dir", type=str, default="logs/training", help="Logging directory")
    parser.add_argument("--horizon", type=float, default=200.0, help="Simulation horizon")
    parser.add_argument("--n-warehouses", type=int, default=3, help="Number of warehouses")
    parser.add_argument("--n-customers", type=int, default=12, help="Number of customers")
    parser.add_argument("--n-vehicles", type=int, default=5, help="Number of vehicles")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create environment config
    config = SimConfig(
        seed=args.seed,
        sim_horizon=args.horizon,
        n_warehouses=args.n_warehouses,
        n_customers=args.n_customers,
        n_vehicles=args.n_vehicles,
        initial_inventory=40,
        vehicle_capacity=10,
        interarrival_mean=6.0,
    )

    print("=" * 60)
    print("Phase 4: PPO Training")
    print("=" * 60)
    print(f"Config: {args.n_warehouses} warehouses, {args.n_customers} customers, {args.n_vehicles} vehicles")
    print(f"Training for {args.total_timesteps} timesteps")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Create training environment
    train_env = DummyVecEnv([make_env(config, rank=0, seed=args.seed)])
    
    # Create evaluation environment
    eval_config = SimConfig(
        seed=args.seed + 1000,  # Different seed for eval
        sim_horizon=args.horizon,
        n_warehouses=args.n_warehouses,
        n_customers=args.n_customers,
        n_vehicles=args.n_vehicles,
        initial_inventory=40,
        vehicle_capacity=10,
        interarrival_mean=6.0,
    )
    eval_env = DummyVecEnv([make_env(eval_config, rank=0, seed=args.seed + 1000)])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.output_dir,
        name_prefix="ppo_supply_chain",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.output_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except ImportError:
        # Fallback if progress bar dependencies not installed
        print("Note: progress bar not available, training without it...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False,
        )

    # Save final model
    final_model_path = os.path.join(args.output_dir, "ppo_supply_chain_final")
    model.save(final_model_path)
    print(f"\n✅ Training complete! Final model saved to: {final_model_path}")

    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Tensorboard logs: {args.log_dir}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print("\nTo evaluate the model:")
    print(f"  python3 experiments/evaluate.py --model {final_model_path}")


if __name__ == "__main__":
    main()

