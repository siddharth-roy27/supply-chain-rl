#!/usr/bin/env python3
"""
All-in-one script to train, evaluate, compare, and visualize RL supply chain system.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "=" * 70)
    print(f"🔹 {description}")
    print("=" * 70)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_ROOT)
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        return False
    print(f"✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="All-in-one script: train, evaluate, compare, and visualize RL supply chain"
    )
    parser.add_argument("--train", action="store_true", help="Train RL model")
    parser.add_argument("--train-timesteps", type=int, default=50000, help="Training timesteps (default: 50000)")
    parser.add_argument("--model-path", type=str, default="models/ppo_quick/ppo_supply_chain_final", help="Path to model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--compare", action="store_true", help="Compare RL vs baseline")
    parser.add_argument("--visualize", action="store_true", help="Visualize simulator logs")
    parser.add_argument("--run-sim", action="store_true", help="Run Phase 1 simulator")
    parser.add_argument("--all", action="store_true", help="Run everything (simulator, train, evaluate, compare)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for eval/comparison")
    parser.add_argument("--horizon", type=float, default=200.0, help="Simulation horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.train, args.evaluate, args.compare, args.visualize, args.run_sim, args.all]):
        args.all = True

    if args.all:
        args.run_sim = True
        args.train = True
        args.evaluate = True
        args.compare = True
        args.visualize = True

    print("\n" + "=" * 70)
    print("🚚 Supply Chain RL - All-in-One Pipeline")
    print("=" * 70)

    success = True

    # Step 1: Run Phase 1 simulator
    if args.run_sim:
        cmd = [
            "python3",
            "simulation/simulator.py",
            "--horizon", str(args.horizon),
            "--seed", str(args.seed),
        ]
        success = run_command(cmd, "Phase 1: Running Simulator") and success

        # Get latest log file for visualization
        log_dir = Path(_ROOT) / "data" / "logs"
        if log_dir.exists():
            csv_files = sorted(log_dir.glob("phase1_orders_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if csv_files:
                args.latest_log = str(csv_files[0])
            else:
                args.latest_log = None
        else:
            args.latest_log = None

    # Step 2: Train RL model
    if args.train:
        output_dir = "models/ppo_quick"
        cmd = [
            "python3",
            "experiments/train.py",
            "--total-timesteps", str(args.train_timesteps),
            "--output-dir", output_dir,
            "--seed", str(args.seed),
            "--horizon", str(args.horizon),
        ]
        success = run_command(cmd, f"Phase 4: Training RL Model ({args.train_timesteps} timesteps)") and success
        args.model_path = f"{output_dir}/ppo_supply_chain_final"

    # Step 3: Evaluate model
    if args.evaluate:
        if os.path.exists(f"{args.model_path}.zip") or os.path.exists(args.model_path):
            cmd = [
                "python3",
                "experiments/evaluate.py",
                "--model", args.model_path,
                "--episodes", str(args.episodes),
                "--seed", str(args.seed),
                "--horizon", str(args.horizon),
            ]
            success = run_command(cmd, f"Phase 4: Evaluating Model ({args.episodes} episodes)") and success
        else:
            print(f"⚠️  Model not found: {args.model_path}, skipping evaluation")

    # Step 4: Compare RL vs baseline
    if args.compare:
        if os.path.exists(f"{args.model_path}.zip") or os.path.exists(args.model_path):
            cmd = [
                "python3",
                "experiments/compare_baseline.py",
                "--model", args.model_path,
                "--episodes", str(args.episodes),
                "--seed", str(args.seed),
                "--horizon", str(args.horizon),
            ]
            success = run_command(cmd, f"Phase 4: Comparing RL vs Baseline ({args.episodes} episodes)") and success
        else:
            print(f"⚠️  Model not found: {args.model_path}, skipping comparison")

    # Step 5: Visualize logs
    if args.visualize:
        if hasattr(args, 'latest_log') and args.latest_log:
            cmd = [
                "python3",
                "dashboard/visualize_logs.py",
                "--log", args.latest_log,
            ]
            success = run_command(cmd, "Phase 2: Visualizing Logs") and success
        else:
            # Try to find latest log
            log_dir = Path(_ROOT) / "data" / "logs"
            if log_dir.exists():
                csv_files = sorted(log_dir.glob("phase1_orders_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                if csv_files:
                    cmd = [
                        "python3",
                        "dashboard/visualize_logs.py",
                        "--log", str(csv_files[0]),
                    ]
                    success = run_command(cmd, "Phase 2: Visualizing Latest Logs") and success
                else:
                    print("⚠️  No log files found, skipping visualization")
            else:
                print("⚠️  Log directory not found, skipping visualization")

    # Final summary
    print("\n" + "=" * 70)
    if success:
        print("✅ All operations completed successfully!")
        print("\nNext steps:")
        print("  • View TensorBoard: python3 -m tensorboard --logdir logs/training")
        print(f"  • Open visualizations: data/logs/*.html")
        print(f"  • Evaluate model: python3 experiments/evaluate.py --model {args.model_path}")
    else:
        print("⚠️  Some operations had errors. Check output above.")
    print("=" * 70)


if __name__ == "__main__":
    main()

