from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import simpy


@dataclass(frozen=True)
class Order:
    order_id: int
    created_time: float
    customer_node: str
    quantity: int


class OrderGenerator:
    """
    SimPy process that generates stochastic customer orders.
    """

    def __init__(
        self,
        *,
        env: simpy.Environment,
        customer_nodes: list[str],
        interarrival_mean: float,
        qty_min: int,
        qty_max: int,
        seed: int,
    ):
        self.env = env
        self.customer_nodes = customer_nodes
        self.interarrival_mean = float(interarrival_mean)
        self.qty_min = int(qty_min)
        self.qty_max = int(qty_max)
        self.rng = np.random.default_rng(seed)
        self._next_id = 0

    def sample_interarrival(self) -> float:
        # Exponential interarrival (Poisson process)
        return float(self.rng.exponential(self.interarrival_mean))

    def sample_quantity(self) -> int:
        return int(self.rng.integers(self.qty_min, self.qty_max + 1))

    def sample_customer(self) -> str:
        idx = int(self.rng.integers(0, len(self.customer_nodes)))
        return self.customer_nodes[idx]

    def run(self, on_order: Callable[[Order], None]) -> simpy.events.Event:
        def _proc() -> simpy.events.Event:
            while True:
                yield self.env.timeout(self.sample_interarrival())
                order = Order(
                    order_id=self._next_id,
                    created_time=float(self.env.now),
                    customer_node=self.sample_customer(),
                    quantity=self.sample_quantity(),
                )
                self._next_id += 1
                on_order(order)

        return self.env.process(_proc())

