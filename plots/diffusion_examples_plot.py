import random

import matplotlib.pyplot as plt
import networkx as nx

from models.diffusion.chapter.equality_reach import (
    diffuse,
    make_network,
    sample_thresholds_normal,
    seed_by_degree,
)

# Parameters (match your simulation)
N = 100
P = 0.05
M = 3
K = 6
BETA = 0.1
SEED = 39

SEED_FRAC = 0.01
THETA_MEAN = 0.30
THETA_SD = 0.10
MAX_STEPS = 50

kinds = ["random", "small_world", "scale_free"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, kind in zip(axes, kinds):
    G = make_network(N, kind, SEED, P, M, K, BETA)

    seeds = seed_by_degree(G, SEED_FRAC)  # single initial adopter (SEED_FRAC * N = 1)
    rng = random.Random(SEED)
    theta_i = sample_thresholds_normal(G, rng=rng, mean=THETA_MEAN, sd=THETA_SD)
    history = diffuse(G, seeds, theta_i=theta_i, max_steps=MAX_STEPS)
    adopted = history[-1]

    node_colors = [
        "orange" if node in seeds
        else "steelblue" if adopted[node]
        else "gray"
        for node in G.nodes()
    ]

    # Same layout style for visual comparability
    pos = nx.spring_layout(G, seed=SEED)

    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color=node_colors)

    ax.set_title(kind.replace("_", " ").title(), fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig("models/diffusion/diffusion_examples.png", dpi=600)
plt.show()
