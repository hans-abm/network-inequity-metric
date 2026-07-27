import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

from equity_reach import (
    BETA, K, M, N,
    get_weights,
    make_network,
)

SEED = 42
KINDS = ["random", "scale_free", "small_world"]
WEIGHT_TYPES = ["uniform", "degree", "inv_degree"]
WEIGHT_LABELS = {"uniform": "Equal", "degree": "Central", "inv_degree": "Peripheral"}
KIND_LABELS = {"random": "Random", "scale_free": "Scale-free", "small_world": "Small-world"}

_HIGH = "#DD8452"
_LOW  = "#4C72B0"

FONTSIZE = 12
NODE_SIZE = 40


_CMAP = mcolors.LinearSegmentedColormap.from_list("wt", [_LOW, _HIGH])


def weight_to_color(w: float):
    return _CMAP(w)


def main():
    n_rows = len(KINDS)
    n_cols = len(WEIGHT_TYPES)

    fig = plt.figure(figsize=(7.5, 7))
    # Leave right margin for colorbar
    gs = fig.add_gridspec(n_rows, n_cols, left=0.08, right=0.88,
                          top=0.93, bottom=0.04, hspace=0.08, wspace=0.05)

    for row, kind in enumerate(KINDS):
        G = make_network(n=N, kind=kind, rng_seed=SEED, p=0.05, m=M, k=K, beta=BETA)
        pos = nx.spring_layout(G, seed=SEED)

        for col, wt in enumerate(WEIGHT_TYPES):
            ax = fig.add_subplot(gs[row, col])
            w = get_weights(G, wt)
            node_colors = [weight_to_color(w[node]) for node in G.nodes()]

            nx.draw_networkx(
                G, pos=pos, ax=ax,
                node_color=node_colors,
                node_size=NODE_SIZE,
                width=0.4,
                edge_color="#cccccc",
                with_labels=False,
                arrows=False,
            )
            ax.axis("off")

            if row == 0:
                ax.set_title(WEIGHT_LABELS[wt], fontsize=FONTSIZE)

        # Vertical row label on the left
        fig.text(
            0.01,
            gs[row, 0].get_position(fig).y0 + gs[row, 0].get_position(fig).height / 2,
            KIND_LABELS[kind],
            fontsize=FONTSIZE,
            va="center", ha="left",
            rotation=90,
        )

    # Colorbar in its own axes to the right of the grid
    cbar_ax = fig.add_axes([0.90, 0.30, 0.02, 0.35])
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Node weight", fontsize=FONTSIZE)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    out = "models/diffusion/weights_plot.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
