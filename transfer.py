"""
Transfer principle test for the network inequity metric (xi_a).

Applies a fixed Pigou-Dalton-style resource transfer on a 3-node path graph
(End - Center - End). Total resource R is held constant; only its
distribution across nodes changes. All three transfers are genuine
Pigou-Dalton transfers (resource moves from a strictly richer node to a
strictly poorer one, and the two never swap rank).
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

from helper import xi_a

ALPHA = 1.5
N = 3  # End(0) - Center(1) - End(2)
DELTA = 0.15  # amount of resource moved in each transfer

BASELINE_R = {0: 0.7, 1: 0.3, 2: 0}  # sum(R) = 1.0

POS = {0: (0, 0), 1: (1, 0), 2: (2, 0)}  # fixed layout: same 3 nodes throughout

_BLUE = "#4682b4"  # high access
_GRAY = "#808080"  # low access
_CMAP = mcolors.LinearSegmentedColormap.from_list("access", [_GRAY, _BLUE])

FONTSIZE = 12
NODE_SIZE = 700


def path3():
    return nx.path_graph(N)  # 0-1-2


def transfer(R, a, b, delta):
    """Move `delta` units of resource from node a to node b, holding sum(R) constant."""
    R2 = dict(R)
    R2[a] -= delta
    R2[b] += delta
    assert R2[a] >= R2[b], (
        f"transfer reverses rank between {a} and {b}: "
        f"not a valid Pigou-Dalton transfer"
    )
    return R2


# (label, source node, target node)
SCENARIOS = [
    ("0 > 1", 0, 1),    
    ("1 > 2", 1, 2),    
    ("0 > 2", 0, 2),       
]


def run():

    G = path3()
    xi0, A0 = xi_a(G, BASELINE_R, alpha=ALPHA)
    states = [("Baseline", BASELINE_R, xi0, A0, None, None)]

    for label, a, b in SCENARIOS:
        R = transfer(BASELINE_R, a, b, DELTA)
        xi, A = xi_a(G, R, alpha=ALPHA)
        states.append((label, R, xi, A, a, b))

    return G, states


def plot(G, states, out_path="results/paper/transfer_plot"):
    n_cols = len(states)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols, 2.9))

    # Normalize colors against the full spread of A values seen across all
    # panels, so shading stays comparable panel-to-panel and isn't washed
    # out by the fixed [0, 1] range A rarely reaches with only 3 nodes.
    all_A = [v for _, _, _, A, _, _ in states for v in A.values()]
    vmin, vmax = min(all_A), max(all_A)

    def color_for(value):
        return _CMAP((value - vmin) / (vmax - vmin))

    for ax, (label, R, xi, A, a, b) in zip(axes, states):
        node_colors = [color_for(A[node]) for node in G.nodes()]
        nx.draw_networkx(
            G, pos=POS, ax=ax,
            node_color=node_colors,
            node_size=NODE_SIZE,
            width=1.2,
            edge_color="#999999",
            font_color="white",
            font_size=FONTSIZE - 1,
            with_labels=True,
        )
        for node, (x, y) in POS.items():
            # R_i (endowment, the thing the transfer actually moves) above
            # A_i (resulting access score); both printed in plain black,
            # since node fill color already encodes A.
            ax.text(x, y - 0.28, f"$R_{{{node}}}$ = {R[node]:.2f}",
                    fontsize=FONTSIZE - 3, ha="center", va="top", color="#333333")
            ax.text(x, y - 0.44, f"$A_{{{node}}}$ = {A[node]:.2f}",
                    fontsize=FONTSIZE - 3, ha="center", va="top", color="#333333")

        if a is None:
            title = f"{label}\n$\\xi_a$ = {xi:.2f}"
        else:
            transfer_eq = (
                f"$R_{{{a}}} \\leftarrow R_{{{a}}} - \\delta,"
                f"\\ R_{{{b}}} \\leftarrow R_{{{b}}} + \\delta$"
            )
            title = f"{transfer_eq}\n$\\xi_a$ = {xi:.2f}"
        ax.set_title(title, fontsize=FONTSIZE - 1)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.75, 0.6)
        ax.axis("off")

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = f"{out_path}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Plot saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    G, states = run()
    plot(G, states)
