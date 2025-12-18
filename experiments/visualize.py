from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx


def plot_supply_chain(G: nx.Graph) -> None:
    pos = nx.get_node_attributes(G, "pos")
    types = nx.get_node_attributes(G, "type")

    plt.figure(figsize=(8, 6))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n, t in types.items() if t == "warehouse"],
        node_color="blue",
        node_size=500,
        label="Warehouse",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n, t in types.items() if t == "customer"],
        node_color="green",
        node_size=300,
        label="Customer",
    )

    # Draw edges + labels
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.legend()
    plt.title("Supply Chain Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


