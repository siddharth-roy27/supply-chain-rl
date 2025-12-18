#!/usr/bin/env python3
"""
Phase 3: Quick test script for SupplyChainEnv.
"""

import os
import sys

# Add project root to path
if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from simulation.simulator import SimConfig  # noqa: E402
from env.supply_chain_env import SupplyChainEnv  # noqa: E402


def main():
    """Test the environment with random actions."""
    print("Testing SupplyChainEnv...")
    
    config = SimConfig(
        seed=42,
        sim_horizon=100.0,
        n_warehouses=3,
        n_customers=8,
        n_vehicles=4,
        initial_inventory=30,
    )
    
    env = SupplyChainEnv(config=config, step_dt=5.0, render_mode="human")
    
    # Reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    # Run a few steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {step}: reward={reward:.2f}, done={done}")
        if info.get("n_delivered", 0) > 0:
            print(f"  Delivered: {info['n_delivered']}, Stockouts: {info['n_stockouts']}")
        
        if done:
            print("Episode finished!")
            break
    
    env.render()
    print(f"\nFinal episode reward: {info.get('episode_reward', 0):.2f}")
    print("✅ Environment test passed!")


if __name__ == "__main__":
    main()

