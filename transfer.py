"""
Transfer principle test for the network inequity metric (xi_a).

Applies a fixed Pigou-Dalton-style resource transfer to pairs of nodes on
small toy graphs (Path-4, Star-4, Cycle-4) and reports how xi_a responds.
Total resource R is held constant across each transfer; only its
distribution across nodes changes.
"""

import networkx as nx
from helper import xi_a  

ALPHA = 1.5
N = 4
DELTA = 0.2  # amount of resource moved in each transfer


def path4():
    return nx.path_graph(N)  # 0-1-2-3


def star4():
    return nx.star_graph(N - 1)  # center 0, leaves 1,2,3


def cycle4():
    return nx.cycle_graph(N)  # 0-1-2-3-0


def uniform_R(total=1.0):
    return {i: total / N for i in range(N)}


def transfer(R, a, b, delta):
    """Move `delta` units of resource from node a to node b, holding sum(R) constant."""
    R2 = dict(R)
    R2[a] -= delta
    R2[b] += delta
    return R2


# (graph label, graph, transfer description, source node, target node)
SCENARIOS = [
    ("Path-4",  "path4",  "Center -> End",         1, 0),
    ("Star-4",  "star4",  "Center -> Periphery",   0, 1),
    ("Path-4",  "path4",  "End -> opposite End",   0, 3),
    ("Cycle-4", "cycle4", "Node -> opposite Node", 0, 2),
    ("Star-4",  "star4",  "Periphery -> Center",   1, 0),
    ("Path-4",  "path4",  "End -> Center",         0, 1),
]

GRAPH_BUILDERS = {"path4": path4, "star4": star4, "cycle4": cycle4}


def run():
    rows = []
    for label, graph_key, desc, a, b in SCENARIOS:
        G = GRAPH_BUILDERS[graph_key]()
        R0 = uniform_R()
        xi_before, _ = xi_a(G, R0, alpha=ALPHA)

        R1 = transfer(R0, a, b, DELTA)
        xi_after, _ = xi_a(G, R1, alpha=ALPHA)

        if abs(xi_after - xi_before) < 1e-9:
            arrow = "="
        elif xi_after > xi_before:
            arrow = "up"
        else:
            arrow = "down"

        rows.append((label, desc, arrow, xi_before, xi_after))
    return rows


if __name__ == "__main__":
    print(f"{'Graph':<8} {'Transfer':<24} {'Delta xi':<8} {'xi_t -> xi_t+1'}")
    for label, desc, arrow, xi_before, xi_after in run():
        print(f"{label:<8} {desc:<24} {arrow:<8} {xi_before:.2f} -> {xi_after:.2f}")
