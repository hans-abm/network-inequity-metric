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
_ARROW_COLOR = "#D55E00"  # the transfer itself, kept distinct from the access palette

FONTSIZE = 12
NODE_SIZE = 700
_ARROW_RAD = 0.35  # arc curvature; also makes the 0->2 transfer visibly arc over node 1


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
    """Baseline xi_a plus the post-transfer state for each scenario.

    No separate baseline panel is plotted -- each scenario panel carries the
    baseline implicitly, spelled out as the exact arithmetic in its title
    (R_a before -> after, R_b before -> after) plus the resulting xi_a
    transition (baseline -> after). The plotted network/colors are always the
    post-transfer state.
    """
    G = path3()
    xi0, _ = xi_a(G, BASELINE_R, alpha=ALPHA)

    scenarios = []
    for label, a, b in SCENARIOS:
        R = transfer(BASELINE_R, a, b, DELTA)
        xi, A = xi_a(G, R, alpha=ALPHA)
        scenarios.append((label, a, b, R, xi, A))

    return G, xi0, scenarios


def plot(G, xi0, scenarios, out_path="results/paper/transfer_plot"):
    n_cols = len(scenarios)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.7 * n_cols, 2.7))

    # Normalize colors against the spread of A values across the plotted
    # (post-transfer) panels, so shading stays comparable panel-to-panel and
    # isn't washed out by the fixed [0, 1] range A rarely reaches with only
    # 3 nodes.
    all_A = [v for _, _, _, _, _, A in scenarios for v in A.values()]
    vmin, vmax = min(all_A), max(all_A)

    def color_for(value):
        return _CMAP((value - vmin) / (vmax - vmin))

    for ax, (_label, a, b, R, xi, A) in zip(axes, scenarios):
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

        # Arc above the row from donor (a) to recipient (b), so the two
        # nodes actually exchanging resource are visually obvious -- and for
        # the 0->2 transfer, the arc visibly bows over node 1, showing it's
        # bypassed rather than involved.
        ax.annotate(
            "", xy=POS[b], xytext=POS[a],
            arrowprops=dict(
                arrowstyle="-|>", color=_ARROW_COLOR, lw=1.8,
                shrinkA=16, shrinkB=16, mutation_scale=16,
                connectionstyle=f"arc3,rad={_ARROW_RAD}",
            ),
        )

        for node, (x, y) in POS.items():
            # R_i (endowment, the thing the transfer actually moves) above
            # A_i (resulting access score); both printed in plain black,
            # since node fill color already encodes A. These are the
            # post-transfer values -- the state actually drawn.
            ax.text(x, y - 0.28, f"$R_{{{node}}}$ = {R[node]:.2f}",
                    fontsize=FONTSIZE - 3, ha="center", va="top", color="#333333")
            ax.text(x, y - 0.39, f"$A_{{{node}}}$ = {A[node]:.2f}",
                    fontsize=FONTSIZE - 3, ha="center", va="top", color="#333333")

        # Title spells out the exact transfer -- baseline value, delta, and
        # result -- for both nodes it touches. This is what carries
        # "baseline" now that its own panel is gone.
        donor_eq = fr"$R_{{{a}}} = {BASELINE_R[a]:.2f} - {DELTA:.2f} = {R[a]:.2f}$"
        recip_eq = fr"$R_{{{b}}} = {BASELINE_R[b]:.2f} + {DELTA:.2f} = {R[b]:.2f}$"
        title = "\n".join([donor_eq, recip_eq])
        ax.set_title(title, fontsize=FONTSIZE - 1, pad=8, linespacing=1.9)

        # xi_a's before -> after sits under the three nodes as a panel
        # caption, same size as the title equations above.
        xi_eq = fr"$\xi_a\!: {xi0:.2f} \to {xi:.2f}$"
        ax.text(1, -0.58, xi_eq, fontsize=FONTSIZE - 1, ha="center", va="top", color="black")

        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.82, 0.12)
        ax.axis("off")

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = f"{out_path}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Plot saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    G, xi0, scenarios = run()
    plot(G, xi0, scenarios)
