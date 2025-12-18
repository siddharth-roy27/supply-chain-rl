from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: str  # "warehouse" | "customer"


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
            g.add_node(n.node_id, kind=n.kind)

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

        return SupplyChainNetwork(g)

    def nodes_of_kind(self, kind: str) -> list[str]:
        return [n for n, data in self.g.nodes(data=True) if data.get("kind") == kind]

    def shortest_distance(self, src: str, dst: str) -> float:
        return float(
            nx.shortest_path_length(self.g, source=src, target=dst, weight="distance")
        )

    def shortest_path(self, src: str, dst: str) -> list[str]:
        return nx.shortest_path(self.g, source=src, target=dst, weight="distance")

    def edge_distance(self, u: str, v: str) -> float:
        return float(self.g.edges[u, v]["distance"])

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

