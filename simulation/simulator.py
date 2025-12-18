from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import simpy

# Allow running as a script: `python simulation/simulator.py`
if __package__ is None:  # pragma: no cover
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from simulation.network import SupplyChainNetwork  # noqa: E402
from simulation.order_generator import Order, OrderGenerator  # noqa: E402
from simulation.vehicle import Vehicle  # noqa: E402
from simulation.warehouse import Warehouse  # noqa: E402


@dataclass
class SimConfig:
    seed: int = 7
    sim_horizon: float = 500.0

    # Network
    n_warehouses: int = 3
    n_customers: int = 12
    extra_edge_prob: float = 0.25
    min_edge_dist: float = 1.0
    max_edge_dist: float = 25.0

    # Warehouses / Vehicles
    initial_inventory: int = 40
    n_vehicles: int = 5
    vehicle_capacity: int = 10

    # Demand
    interarrival_mean: float = 6.0
    qty_min: int = 1
    qty_max: int = 6

    # Time/cost model
    speed: float = 1.0  # distance units per time unit
    fixed_service_time: float = 0.25
    cost_per_distance: float = 1.0
    stockout_penalty: float = 50.0

    # Logging
    log_dir: str = "data/logs"


@dataclass
class OrderLogRow:
    order_id: int
    created_time: float
    delivered_time: Optional[float]
    customer_node: str
    quantity: int
    warehouse_id: Optional[str]
    warehouse_node: Optional[str]
    vehicle_id: Optional[str]
    travel_distance: float
    travel_time: float
    delay: Optional[float]
    stockout: int
    cost: float


class SupplyChainSimulator:
    """
    Phase 1 simulator/controller:
    - Generates orders
    - Applies a greedy baseline policy (inventory + nearest travel-time)
    - Simulates vehicle travel with SimPy
    - Logs per-order metrics
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.network = SupplyChainNetwork.random_connected(
            seed=cfg.seed,
            n_warehouses=cfg.n_warehouses,
            n_customers=cfg.n_customers,
            extra_edge_prob=cfg.extra_edge_prob,
            min_dist=cfg.min_edge_dist,
            max_dist=cfg.max_edge_dist,
        )

        warehouse_nodes = self.network.nodes_of_kind("warehouse")
        customer_nodes = self.network.nodes_of_kind("customer")

        self.warehouses: list[Warehouse] = [
            Warehouse(warehouse_id=f"W{i}", node_id=warehouse_nodes[i], inventory=cfg.initial_inventory)
            for i in range(len(warehouse_nodes))
        ]

        # Vehicles start at random warehouse nodes (deterministic under seed)
        # Using a simple mapping rather than RNG to avoid cross-module coupling.
        self.vehicles: list[Vehicle] = []
        for i in range(cfg.n_vehicles):
            start_node = warehouse_nodes[i % len(warehouse_nodes)]
            self.vehicles.append(
                Vehicle(
                    vehicle_id=f"V{i}",
                    capacity=cfg.vehicle_capacity,
                    start_node=start_node,
                    env=self.env,
                )
            )

        self.order_gen = OrderGenerator(
            env=self.env,
            customer_nodes=customer_nodes,
            interarrival_mean=cfg.interarrival_mean,
            qty_min=cfg.qty_min,
            qty_max=cfg.qty_max,
            seed=cfg.seed + 1,
        )

        self.order_logs: list[OrderLogRow] = []

        # Summary counters
        self.total_distance = 0.0
        self.total_cost = 0.0
        self.n_orders = 0
        self.n_stockouts = 0

    def travel_time(self, distance: float) -> float:
        return float(distance / self.cfg.speed) if self.cfg.speed > 0 else float("inf")

    def _choose_greedy_assignment(self, order: Order) -> tuple[Optional[Warehouse], Optional[Vehicle], float, float, float]:
        """
        Greedy baseline:
        - choose (warehouse, vehicle) minimizing total travel time:
            vehicle->warehouse + warehouse->customer
        - subject to warehouse having enough inventory and vehicle capacity >= order qty
        Returns (warehouse, vehicle, dist_to_pickup, dist_to_dropoff, total_distance)
        """
        best = None
        for wh in self.warehouses:
            if not wh.can_fulfill(order.quantity):
                continue
            for v in self.vehicles:
                if v.capacity < order.quantity:
                    continue
                d1 = self.network.shortest_distance(v.node_id, wh.node_id)
                d2 = self.network.shortest_distance(wh.node_id, order.customer_node)
                total_d = d1 + d2
                total_t = self.travel_time(total_d)
                cand = (total_t, total_d, wh, v, d1, d2)
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            return None, None, 0.0, 0.0, 0.0
        _, _, wh, v, d1, d2 = best
        return wh, v, d1, d2, d1 + d2

    def on_order(self, order: Order) -> None:
        self.n_orders += 1
        wh, v, d_pick, d_drop, total_d = self._choose_greedy_assignment(order)

        if wh is None or v is None:
            # Stockout: nothing can fulfill right now (inventory constraint dominates in Phase 1)
            cost = float(self.cfg.stockout_penalty)
            self.n_stockouts += 1
            self.total_cost += cost
            self.order_logs.append(
                OrderLogRow(
                    order_id=order.order_id,
                    created_time=order.created_time,
                    delivered_time=None,
                    customer_node=order.customer_node,
                    quantity=order.quantity,
                    warehouse_id=None,
                    warehouse_node=None,
                    vehicle_id=None,
                    travel_distance=0.0,
                    travel_time=0.0,
                    delay=None,
                    stockout=1,
                    cost=cost,
                )
            )
            return

        # Allocate inventory immediately (reservation) to keep Phase 1 simple/deterministic.
        allocated = wh.allocate(order.quantity)
        if not allocated:
            # Rare due to check in _choose_greedy_assignment, but keep safe.
            cost = float(self.cfg.stockout_penalty)
            self.n_stockouts += 1
            self.total_cost += cost
            self.order_logs.append(
                OrderLogRow(
                    order_id=order.order_id,
                    created_time=order.created_time,
                    delivered_time=None,
                    customer_node=order.customer_node,
                    quantity=order.quantity,
                    warehouse_id=wh.warehouse_id,
                    warehouse_node=wh.node_id,
                    vehicle_id=None,
                    travel_distance=0.0,
                    travel_time=0.0,
                    delay=None,
                    stockout=1,
                    cost=cost,
                )
            )
            return

        t_pick = self.travel_time(d_pick)
        t_drop = self.travel_time(d_drop)
        t_total = t_pick + t_drop + float(self.cfg.fixed_service_time)
        cost = float(total_d * self.cfg.cost_per_distance)

        self.total_distance += float(total_d)
        self.total_cost += cost

        # Log after delivery completes (need delivered_time)
        self.env.process(
            self._delivery_process(
                order=order,
                wh=wh,
                v=v,
                d_total=total_d,
                t_total=t_total,
                cost=cost,
                t_pick=t_pick,
                t_drop=t_drop,
            )
        )

    def _delivery_process(
        self,
        *,
        order: Order,
        wh: Warehouse,
        v: Vehicle,
        d_total: float,
        t_total: float,
        cost: float,
        t_pick: float,
        t_drop: float,
    ):
        start_time = float(self.env.now)
        yield v.deliver(
            pickup_node=wh.node_id,
            dropoff_node=order.customer_node,
            travel_time_to_pickup=t_pick,
            travel_time_to_dropoff=t_drop,
            service_time=float(self.cfg.fixed_service_time),
        )
        delivered_time = float(self.env.now)
        delay = delivered_time - order.created_time
        self.order_logs.append(
            OrderLogRow(
                order_id=order.order_id,
                created_time=order.created_time,
                delivered_time=delivered_time,
                customer_node=order.customer_node,
                quantity=order.quantity,
                warehouse_id=wh.warehouse_id,
                warehouse_node=wh.node_id,
                vehicle_id=v.vehicle_id,
                travel_distance=float(d_total),
                travel_time=float(t_total),
                delay=float(delay),
                stockout=0,
                cost=float(cost),
            )
        )

    def run(self) -> None:
        self.order_gen.run(self.on_order)
        self.env.run(until=float(self.cfg.sim_horizon))

    def write_logs(self) -> str:
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(self.cfg.log_dir, f"phase1_orders_{ts}.csv")

        # Stable column order
        fieldnames = list(asdict(self.order_logs[0]).keys()) if self.order_logs else list(OrderLogRow.__annotations__.keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.order_logs:
                w.writerow(asdict(row))
        return out_path

    def print_summary(self) -> None:
        delivered = sum(1 for r in self.order_logs if r.stockout == 0)
        avg_delay = (
            sum(r.delay for r in self.order_logs if r.delay is not None) / delivered
            if delivered > 0
            else 0.0
        )
        print("=== Phase 1 Simulator Summary ===")
        print(f"sim_horizon={self.cfg.sim_horizon}")
        print(f"orders={self.n_orders} delivered={delivered} stockouts={self.n_stockouts}")
        print(f"total_distance={self.total_distance:.2f} total_cost={self.total_cost:.2f}")
        print(f"avg_delay={avg_delay:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: SimPy supply chain simulator (greedy baseline).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--horizon", type=float, default=500.0)
    p.add_argument("--log-dir", type=str, default="data/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(seed=args.seed, sim_horizon=args.horizon, log_dir=args.log_dir)
    sim = SupplyChainSimulator(cfg)
    sim.run()
    sim.print_summary()
    out_path = sim.write_logs()
    print(f"wrote_logs={out_path}")


if __name__ == "__main__":
    main()


