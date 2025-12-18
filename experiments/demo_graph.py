#!/usr/bin/env python3
"""
Quick demo: Visualize the supply chain graph structure.
"""

import os
import sys

# Add project root to path
if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from dataclasses import dataclass
from simulation.network import build_supply_chain_graph  # noqa: E402
from experiments.visualize import plot_supply_chain  # noqa: E402


@dataclass
class W:
    id: int
    pos: tuple[float, float]
    inventory: int


@dataclass
class O:
    order_id: int
    pos: tuple[float, float]
    quantity: int
    time: float


def main():
    # Create sample warehouses and customers
    warehouses = [
        W(1, (0, 0), 50),
        W(2, (10, 0), 40),
        W(3, (5, 10), 60),
    ]
    customers = [O(i, (i * 2, i * 1), 5, 0.0) for i in range(5)]

    # Build graph
    G = build_supply_chain_graph(warehouses, customers)
    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Visualize
    plot_supply_chain(G)
    print("Graph visualization opened!")


if __name__ == "__main__":
    main()

