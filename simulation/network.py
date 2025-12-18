from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: str  # "warehouse" | "customer"


def build_supply_chain_graph(warehouses: list[Any], customers: list[Any]) -> nx.Graph:
    """
    Convenience builder for visualization / notebooks (Phase 2).

    Builds a NetworkX graph with:
    - node attribute 'type' in {'warehouse', 'customer'}
    - node attribute 'pos' as (x, y) for plotting

    This intentionally uses duck-typing to accept many object shapes:
    - Warehouses: expects either .node_id or .warehouse_id or .id, and either .pos or .location
    - Customers/Orders: expects either .customer_node or .order_id or .id, and either .pos or .location

    NOTE: This is separate from `SupplyChainNetwork.random_connected()` which is the simulator's
    default topology builder.
    """

    def _get_id(obj: Any, fallbacks: Tuple[str, ...]) -> str:
        for k in fallbacks:
            if hasattr(obj, k):
                return str(getattr(obj, k))
        # final fallback: repr-based stable-ish id
        return str(obj)

    def _get_pos(obj: Any) -> tuple[float, float]:
        for k in ("pos", "location", "coord", "coords"):
            if hasattr(obj, k):
                v = getattr(obj, k)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    return float(v[0]), float(v[1])
        raise ValueError(f"Object {obj!r} is missing a 2D position (expected .pos or .location).")

    G = nx.Graph()

    for w in warehouses:
        wid = f"W{_get_id(w, ('node_id', 'warehouse_id', 'id'))}"
        G.add_node(wid, type="warehouse", pos=_get_pos(w))

    for c in customers:
        cid = f"C{_get_id(c, ('customer_node', 'order_id', 'id'))}"
        G.add_node(cid, type="customer", pos=_get_pos(c))

    # Optional: connect every customer to every warehouse (distance as Euclidean) for a readable view.
    # Users can replace this with road-network edges if desired.
    for w in warehouses:
        wid = f"W{_get_id(w, ('node_id', 'warehouse_id', 'id'))}"
        wx, wy = G.nodes[wid]["pos"]
        for c in customers:
            cid = f"C{_get_id(c, ('customer_node', 'order_id', 'id'))}"
            cx, cy = G.nodes[cid]["pos"]
            dist = float(((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5)
            G.add_edge(wid, cid, distance=dist)

    return G


class SupplyChainNetwork:
    """
    Thin wrapper around a NetworkX graph with convenience helpers used by the simulator.

    Nodes are strings; edges carry a 'distance' float attribute.
    """

    def __init__(self, graph: nx.Graph):
        self.g = graph

    @staticmethod
    def random_connected(
        *,
        seed: int,
        n_warehouses: int,
        n_customers: int,
        extra_edge_prob: float = 0.25,
        min_dist: float = 1.0,
        max_dist: float = 20.0,
        speed: float = 1.0,
    ) -> "SupplyChainNetwork":
        """
        Builds a simple connected undirected graph:
        - Start with a random spanning tree for connectivity
        - Add extra random edges
        """
        rng = np.random.default_rng(seed)

        nodes: list[NodeSpec] = []
        for i in range(n_warehouses):
            nodes.append(NodeSpec(node_id=f"W{i}", kind="warehouse"))
        for i in range(n_customers):
            nodes.append(NodeSpec(node_id=f"C{i}", kind="customer"))

        g = nx.Graph()
        for n in nodes:
            # Node features are dynamic and will be updated by the simulator/controller.
            # We initialize them to sensible defaults to keep the graph self-contained.
            g.add_node(
                n.node_id,
                kind=n.kind,
                inventory=0,  # only meaningful for warehouses
                demand=0,  # only meaningful for customers (pending demand qty)
                served=0,  # cumulative served qty (optional metric)
            )

        # Random spanning tree (connect all nodes)
        order = [n.node_id for n in nodes]
        rng.shuffle(order)
        for i in range(1, len(order)):
            u = order[i]
            v = order[rng.integers(0, i)]
            dist = float(rng.uniform(min_dist, max_dist))
            g.add_edge(u, v, distance=dist)

        # Extra edges
        all_nodes = [n.node_id for n in nodes]
        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                if g.has_edge(all_nodes[i], all_nodes[j]):
                    continue
                if float(rng.random()) < extra_edge_prob:
                    dist = float(rng.uniform(min_dist, max_dist))
                    g.add_edge(all_nodes[i], all_nodes[j], distance=dist)

        net = SupplyChainNetwork(g)
        net.set_speed(speed)
        return net

    def nodes_of_kind(self, kind: str) -> list[str]:
        return [n for n, data in self.g.nodes(data=True) if data.get("kind") == kind]

    def set_speed(self, speed: float) -> None:
        """
        Sets/updates a derived 'travel_time' edge attribute using:
            travel_time = distance / speed
        """
        speed = float(speed)
        if speed <= 0:
            raise ValueError("speed must be > 0")
        for u, v, data in self.g.edges(data=True):
            dist = float(data.get("distance", 0.0))
            data["travel_time"] = float(dist / speed)

    def shortest_distance(self, src: str, dst: str) -> float:
        return float(
            nx.shortest_path_length(self.g, source=src, target=dst, weight="distance")
        )

    def shortest_travel_time(self, src: str, dst: str) -> float:
        return float(
            nx.shortest_path_length(self.g, source=src, target=dst, weight="travel_time")
        )

    def shortest_path(self, src: str, dst: str) -> list[str]:
        return nx.shortest_path(self.g, source=src, target=dst, weight="distance")

    def shortest_time_path(self, src: str, dst: str) -> list[str]:
        return nx.shortest_path(self.g, source=src, target=dst, weight="travel_time")

    def edge_distance(self, u: str, v: str) -> float:
        return float(self.g.edges[u, v]["distance"])

    def edge_travel_time(self, u: str, v: str) -> float:
        return float(self.g.edges[u, v]["travel_time"])

    def total_path_distance(self, path: Iterable[str]) -> float:
        it = iter(path)
        try:
            prev = next(it)
        except StopIteration:
            return 0.0
        total = 0.0
        for cur in it:
            total += self.edge_distance(prev, cur)
            prev = cur
        return float(total)

    def total_path_travel_time(self, path: Iterable[str]) -> float:
        it = iter(path)
        try:
            prev = next(it)
        except StopIteration:
            return 0.0
        total = 0.0
        for cur in it:
            total += self.edge_travel_time(prev, cur)
            prev = cur
        return float(total)

    # --- dynamic node feature helpers (used by simulator and later RL env) ---

    def set_inventory(self, warehouse_node: str, inventory: int) -> None:
        self.g.nodes[warehouse_node]["inventory"] = int(inventory)

    def add_demand(self, customer_node: str, qty: int) -> None:
        self.g.nodes[customer_node]["demand"] = int(self.g.nodes[customer_node].get("demand", 0) + int(qty))

    def serve_demand(self, customer_node: str, qty: int) -> None:
        cur = int(self.g.nodes[customer_node].get("demand", 0))
        self.g.nodes[customer_node]["demand"] = max(0, cur - int(qty))
        self.g.nodes[customer_node]["served"] = int(self.g.nodes[customer_node].get("served", 0) + int(qty))

