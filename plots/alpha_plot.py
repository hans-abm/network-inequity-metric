import numpy as np
import matplotlib.pyplot as plt

d = np.linspace(0, 6, 300)
alphas = [0.75, 1.5, 2.25]
labels = [r"$\alpha = 0.75$", r"$\alpha = 1.50$", r"$\alpha = 2.25$"]

COLORS = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for alpha, label, color in zip(alphas, labels, COLORS):
    ax.plot(d, np.exp(-alpha * d), label=label, color=color)

ax.set_xlabel("Degrees of separation $d$", fontsize=12)
ax.set_ylabel(r"$A_i$ ($e^{-\alpha d}$)", fontsize=12)
ax.legend(fontsize=12, frameon=False)
ax.tick_params(labelsize=12)
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("models/diffusion/alpha_curves.png", dpi=150)
plt.show()
