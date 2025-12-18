# 🚀 Quick Start Guide

## One-Command Setup 🎯

Run everything at once:

```bash
# Install dependencies first
pip install -r requirements.txt

# Run entire pipeline (simulator → train → evaluate → compare → visualize)
python3 experiments/run_all.py --all
```

## Individual Commands

### Phase 1: Run Simulator

```bash
python3 simulation/simulator.py --horizon 200 --seed 5
```

### Phase 2: Visualize Results

```bash
python3 dashboard/visualize_logs.py --log "$(ls -t data/logs/phase1_orders_*.csv | head -n 1)"
```

### Phase 3: Test RL Environment

```bash
python3 experiments/test_env.py
```

### Phase 4: Train RL Agent

```bash
# Quick (50k timesteps, ~5-10 min)
python3 experiments/train.py --total-timesteps 50000 --output-dir models/ppo_quick

# Full (100k timesteps, ~15-20 min)
python3 experiments/train.py --total-timesteps 100000 --output-dir models/ppo_full
```

### Phase 4: Evaluate & Compare

```bash
# Evaluate
python3 experiments/evaluate.py --model models/ppo_quick/ppo_supply_chain_final --episodes 10

# Compare vs baseline
python3 experiments/compare_baseline.py --model models/ppo_quick/ppo_supply_chain_final --episodes 10
```

### Monitor Training

```bash
~/.local/bin/tensorboard --logdir logs/training
# Open http://localhost:6006
```

---

## Expected Times

- **50k timesteps**: ~5-10 minutes (CPU)
- **100k timesteps**: ~15-20 minutes (CPU)
- **Full pipeline**: ~20-30 minutes

GPU support: PyTorch auto-detects if available.

