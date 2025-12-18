from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.simulator import SupplyChainSimulator


@dataclass(frozen=True)
class StateSnapshot:
    """
    Phase 2: a simple RL-ready state snapshot.

    This is intentionally lightweight; Phase 3 will formalize this into Gym spaces.
    """

    # Shapes:
    # - warehouse_inventory: (n_warehouses,)
    # - customer_demand: (n_customers,)
    # - vehicle_node_idx: (n_vehicles,)
    warehouse_inventory: np.ndarray
    customer_demand: np.ndarray
    vehicle_node_idx: np.ndarray


def encode_state(sim: SupplyChainSimulator) -> StateSnapshot:
    """
    Encodes the current simulator state into fixed-size numeric arrays.

    Node indexing is derived from the network node list order (stable per run).
    """
    g = sim.network.g
    node_list = list(g.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    warehouse_nodes = sim.network.nodes_of_kind("warehouse")
    customer_nodes = sim.network.nodes_of_kind("customer")

    wh_inv = np.array([int(g.nodes[n].get("inventory", 0)) for n in warehouse_nodes], dtype=np.int32)
    cust_dem = np.array([int(g.nodes[n].get("demand", 0)) for n in customer_nodes], dtype=np.int32)
    veh_pos = np.array([node_to_idx[v.node_id] for v in sim.vehicles], dtype=np.int32)

    return StateSnapshot(
        warehouse_inventory=wh_inv,
        customer_demand=cust_dem,
        vehicle_node_idx=veh_pos,
    )


