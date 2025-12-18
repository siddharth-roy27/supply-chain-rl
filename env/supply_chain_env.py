from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy

# Allow running as a script
if __package__ is None:  # pragma: no cover
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from env.reward import compute_reward, RewardComponents  # noqa: E402
from env.state_encoder import StateSnapshot  # noqa: E402
from simulation.network import SupplyChainNetwork  # noqa: E402
from simulation.order_generator import Order, OrderGenerator  # noqa: E402
from simulation.simulator import SimConfig  # noqa: E402
from simulation.vehicle import Vehicle  # noqa: E402
from simulation.warehouse import Warehouse  # noqa: E402


@dataclass
class PendingOrder:
    """Order waiting for agent action."""

    order: Order
    assigned_warehouse_idx: Optional[int] = None
    assigned_vehicle_idx: Optional[int] = None
    delivered: bool = False
    travel_distance: float = 0.0
    travel_time: float = 0.0
    delay: Optional[float] = None
    reward_components: Optional[RewardComponents] = None


class SupplyChainEnv(gym.Env):
    """
    Phase 3: Gymnasium-compatible RL environment for supply chain optimization.

    The environment:
    - Steps forward in time using SimPy
    - When orders arrive, pauses for agent action (warehouse_idx, vehicle_idx)
    - Executes the action and computes reward
    - Returns state, reward, done, info

    State space: [warehouse_inventory, customer_demand, vehicle_positions, pending_order_features]
    Action space: MultiDiscrete([n_warehouses, n_vehicles]) for order assignment
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        reward_scale: float = 1.0,
        step_dt: float = 1.0,
        max_pending_orders: int = 10,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or SimConfig()
        self.reward_scale = reward_scale
        self.step_dt = step_dt  # time step per env.step() call
        self.max_pending_orders = max_pending_orders
        self.render_mode = render_mode

        # Initialize simulator components
        self.env = simpy.Environment()
        self.network = SupplyChainNetwork.random_connected(
            seed=self.config.seed,
            n_warehouses=self.config.n_warehouses,
            n_customers=self.config.n_customers,
            extra_edge_prob=self.config.extra_edge_prob,
            min_dist=self.config.min_edge_dist,
            max_dist=self.config.max_edge_dist,
            speed=self.config.speed,
        )

        warehouse_nodes = self.network.nodes_of_kind("warehouse")
        customer_nodes = self.network.nodes_of_kind("customer")

        self.warehouses: list[Warehouse] = [
            Warehouse(
                warehouse_id=f"W{i}",
                node_id=warehouse_nodes[i],
                inventory=self.config.initial_inventory,
            )
            for i in range(len(warehouse_nodes))
        ]
        for wh in self.warehouses:
            self.network.set_inventory(wh.node_id, wh.inventory)

        self.vehicles: list[Vehicle] = []
        for i in range(self.config.n_vehicles):
            start_node = warehouse_nodes[i % len(warehouse_nodes)]
            self.vehicles.append(
                Vehicle(
                    vehicle_id=f"V{i}",
                    capacity=self.config.vehicle_capacity,
                    start_node=start_node,
                    env=self.env,
                )
            )

        self.order_gen = OrderGenerator(
            env=self.env,
            customer_nodes=customer_nodes,
            interarrival_mean=self.config.interarrival_mean,
            qty_min=self.config.qty_min,
            qty_max=self.config.qty_max,
            seed=self.config.seed + 1,
        )

        # Order queue
        self.pending_orders: deque[PendingOrder] = deque(maxlen=max_pending_orders)
        self.current_order: Optional[PendingOrder] = None  # order awaiting action
        self.completed_orders: list[PendingOrder] = []

        # Stats
        self.episode_reward = 0.0
        self.n_delivered = 0
        self.n_stockouts = 0

        # Define observation space: state snapshot + pending order features
        state_snap = self._encode_state_direct()
        n_warehouses = len(self.warehouses)
        n_vehicles = len(self.vehicles)
        n_customers = len(customer_nodes)
        # State: inventory (n_wh) + demand (n_cust) + vehicle_pos (n_veh) + current_order_features (3: qty, customer_node_idx, created_time_normalized)
        obs_dim = n_warehouses + n_customers + n_vehicles + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space: [warehouse_idx, vehicle_idx]
        self.action_space = spaces.MultiDiscrete([n_warehouses, n_vehicles])

        # Start order generation
        self.order_gen.run(self._on_order_callback)

    def _encode_state_direct(self) -> StateSnapshot:
        """Directly encode state without needing full simulator object."""
        g = self.network.g
        node_list = list(g.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        warehouse_nodes = self.network.nodes_of_kind("warehouse")
        customer_nodes = self.network.nodes_of_kind("customer")

        wh_inv = np.array([int(g.nodes[n].get("inventory", 0)) for n in warehouse_nodes], dtype=np.int32)
        cust_dem = np.array([int(g.nodes[n].get("demand", 0)) for n in customer_nodes], dtype=np.int32)
        veh_pos = np.array([node_to_idx[v.node_id] for v in self.vehicles], dtype=np.int32)

        return StateSnapshot(
            warehouse_inventory=wh_inv,
            customer_demand=cust_dem,
            vehicle_node_idx=veh_pos,
        )

    def _on_order_callback(self, order: Order) -> None:
        """Callback when order arrives (adds to pending queue)."""
        if len(self.pending_orders) < self.max_pending_orders:
            self.pending_orders.append(PendingOrder(order=order))

    def _get_observation(self) -> np.ndarray:
        """Encodes current state into observation vector."""
        state_snap = self._encode_state_direct()
        n_warehouses = len(self.warehouses)
        n_customers = len(self.network.nodes_of_kind("customer"))
        n_vehicles = len(self.vehicles)

        # Base state: inventory + demand + vehicle positions (as node indices)
        obs_list = [
            state_snap.warehouse_inventory.astype(np.float32),
            state_snap.customer_demand.astype(np.float32),
            state_snap.vehicle_node_idx.astype(np.float32),
        ]

        # Add current order features if exists, else zeros
        if self.current_order:
            order = self.current_order.order
            # Find customer node index
            customer_nodes = list(self.network.nodes_of_kind("customer"))
            try:
                cust_idx = customer_nodes.index(order.customer_node)
            except ValueError:
                cust_idx = 0
            # Normalize time (simple: divide by horizon)
            time_norm = float(order.created_time / self.config.sim_horizon) if self.config.sim_horizon > 0 else 0.0
            obs_list.append(np.array([float(order.quantity), float(cust_idx), time_norm], dtype=np.float32))
        else:
            obs_list.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        return np.concatenate(obs_list, axis=0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets environment to initial state."""
        if seed is not None:
            self.config.seed = seed

        # Reset SimPy environment
        self.env = simpy.Environment()

        # Reinitialize network (deterministic with seed)
        self.network = SupplyChainNetwork.random_connected(
            seed=self.config.seed,
            n_warehouses=self.config.n_warehouses,
            n_customers=self.config.n_customers,
            extra_edge_prob=self.config.extra_edge_prob,
            min_dist=self.config.min_edge_dist,
            max_dist=self.config.max_edge_dist,
            speed=self.config.speed,
        )

        warehouse_nodes = self.network.nodes_of_kind("warehouse")

        # Reset warehouses
        for i, wh in enumerate(self.warehouses):
            wh.inventory = self.config.initial_inventory
            wh.node_id = warehouse_nodes[i]
            self.network.set_inventory(wh.node_id, wh.inventory)

        # Reset vehicles
        for i, v in enumerate(self.vehicles):
            start_node = warehouse_nodes[i % len(warehouse_nodes)]
            v.env = self.env
            v.node_id = start_node
            v.busy = simpy.Resource(self.env, capacity=1)
            v.last_available_time = 0.0

        # Reset order generator
        self.order_gen = OrderGenerator(
            env=self.env,
            customer_nodes=self.network.nodes_of_kind("customer"),
            interarrival_mean=self.config.interarrival_mean,
            qty_min=self.config.qty_min,
            qty_max=self.config.qty_max,
            seed=self.config.seed + 1,
        )

        # Reset queues
        self.pending_orders.clear()
        self.current_order = None
        self.completed_orders.clear()

        # Reset stats
        self.episode_reward = 0.0
        self.n_delivered = 0
        self.n_stockouts = 0

        # Restart order generation
        self.order_gen.run(self._on_order_callback)

        obs = self._get_observation()
        info = {"time": float(self.env.now)}
        return obs, info

    def step(
        self, action: np.ndarray | tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Steps the environment.

        Action: [warehouse_idx, vehicle_idx] to assign to current order.

        Returns: (observation, reward, terminated, truncated, info)
        """
        wh_idx, v_idx = int(action[0]), int(action[1])

        reward = 0.0

        # If we have a pending order, assign it
        if self.current_order is None and len(self.pending_orders) > 0:
            self.current_order = self.pending_orders.popleft()

        if self.current_order is not None:
            # Execute assignment
            order = self.current_order.order
            wh = self.warehouses[wh_idx % len(self.warehouses)]
            v = self.vehicles[v_idx % len(self.vehicles)]

            # Check feasibility
            if wh.can_fulfill(order.quantity) and v.capacity >= order.quantity:
                # Allocate inventory
                allocated = wh.allocate(order.quantity)
                self.network.set_inventory(wh.node_id, wh.inventory)

                if allocated:
                    # Compute paths and travel times
                    path1 = self.network.shortest_time_path(v.node_id, wh.node_id)
                    path2 = self.network.shortest_time_path(wh.node_id, order.customer_node)
                    d1 = self.network.total_path_distance(path1)
                    d2 = self.network.total_path_distance(path2)
                    total_d = d1 + d2
                    t1 = self.network.total_path_travel_time(path1)
                    t2 = self.network.total_path_travel_time(path2)
                    total_t = t1 + t2 + self.config.fixed_service_time

                    # Execute delivery (simulate travel)
                    delivery_time = float(self.env.now) + total_t
                    delay = delivery_time - order.created_time

                    # Update vehicle position (immediate for simplicity)
                    v.node_id = order.customer_node
                    v.last_available_time = delivery_time

                    # Update network demand
                    self.network.serve_demand(order.customer_node, order.quantity)

                    # Compute reward
                    reward, components = compute_reward(
                        delivered=True,
                        travel_distance=total_d,
                        travel_time=total_t,
                        delay=delay,
                        cost_per_distance=self.config.cost_per_distance,
                        stockout_penalty=self.config.stockout_penalty,
                    )
                    reward = reward * self.reward_scale

                    # Update order record
                    self.current_order.assigned_warehouse_idx = wh_idx
                    self.current_order.assigned_vehicle_idx = v_idx
                    self.current_order.delivered = True
                    self.current_order.travel_distance = total_d
                    self.current_order.travel_time = total_t
                    self.current_order.delay = delay
                    self.current_order.reward_components = components

                    self.n_delivered += 1
                else:
                    # Allocation failed (rare)
                    reward, components = compute_reward(
                        delivered=False,
                        travel_distance=0.0,
                        travel_time=0.0,
                        delay=None,
                        stockout_penalty=self.config.stockout_penalty,
                    )
                    reward = reward * self.reward_scale
                    self.current_order.reward_components = components
                    self.n_stockouts += 1
            else:
                # Infeasible assignment: stockout penalty
                reward, components = compute_reward(
                    delivered=False,
                    travel_distance=0.0,
                    travel_time=0.0,
                    delay=None,
                    stockout_penalty=self.config.stockout_penalty,
                )
                reward = reward * self.reward_scale
                self.current_order.reward_components = components
                self.n_stockouts += 1

            # Move to completed
            self.completed_orders.append(self.current_order)
            self.current_order = None

        # Advance time
        self.env.run(until=min(float(self.env.now) + self.step_dt, self.config.sim_horizon))

        # Check if done
        done = float(self.env.now) >= self.config.sim_horizon

        self.episode_reward += reward

        obs = self._get_observation()
        info = {
            "time": float(self.env.now),
            "n_delivered": self.n_delivered,
            "n_stockouts": self.n_stockouts,
            "episode_reward": self.episode_reward,
        }

        return obs, reward, done, False, info

    def render(self) -> Optional[str]:
        """Renders the environment."""
        if self.render_mode == "human":
            print(f"\n=== Supply Chain Env (t={self.env.now:.2f}) ===")
            print(f"Warehouses:")
            for wh in self.warehouses:
                print(f"  {wh.warehouse_id}: inventory={wh.inventory}")
            print(f"Vehicles:")
            for v in self.vehicles:
                print(f"  {v.vehicle_id}: node={v.node_id}, idle={v.is_idle()}")
            print(f"Pending orders: {len(self.pending_orders)}")
            if self.current_order:
                print(f"Current order: {self.current_order.order.order_id} (qty={self.current_order.order.quantity})")
            print(f"Stats: delivered={self.n_delivered}, stockouts={self.n_stockouts}, reward={self.episode_reward:.2f}")
            return None
        elif self.render_mode == "ansi":
            lines = [
                f"t={self.env.now:.2f}",
                f"delivered={self.n_delivered} stockouts={self.n_stockouts} reward={self.episode_reward:.2f}",
            ]
            return "\n".join(lines)
        return None

