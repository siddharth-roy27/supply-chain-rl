## Supply Chain RL (Phase 1 Simulator)

Phase 1 provides a **discrete-event supply chain simulator** (no ML yet) with:
- Warehouses with inventory
- Vehicles with capacity and travel time
- Stochastic customer orders
- Greedy baseline dispatcher (controller-side policy)
- Per-order CSV logs in `data/logs/`

### Setup

```bash
python3 -m pip install -r requirements.txt
```

### Run Phase 1

```bash
python3 simulation/simulator.py --horizon 500 --seed 7
```

Outputs:
- A console summary (orders, distance, cost, avg delay)
- A CSV file like `data/logs/phase1_orders_<timestamp>.csv`


