# Network inequity metric

This directory develops and validates a pair of metrics for how equitably a
resource is distributed relative to network structure, then uses a diffusion
model as the testbed for studying them.

## Metrics (`helper.py`)

- **`xi_a`** — continuous, distance-attenuated access. Each node's access score
  aggregates every other node's resource level `R[j]`, discounted by
  `exp(-alpha * distance)`, so nearby resources count more than distant ones.
- **`xi_d`** — binary, hop-cutoff reachability. A node has access (1) if it is
  within `d` hops of any resource holder, else 0 — no partial credit.

Both take a social network `G`, per-node resource map `R` and optional priority weights `w` (e.g.
favoring central or peripheral nodes), and return a single inequity score in
`[0, 1]`: 0 is perfect equity (everyone has access), 1 is maximal inequity
(no one does). `xi_a` and `xi_d` represent a modeling choice — smooth
attenuation vs. a hard cutoff — and are designed to be fed the same `R` so the
two can be compared directly on the same run. `tutorial.ipynb` / `tutorial.Rmd`
give the full derivation and worked examples in Python and R.

## Diffusion model

To see how the metric behaves under a realistic adoption process, `main.py`
and `sensitivity.py` run a synchronous network threshold model:

- **Networks**: Erdős–Rényi (random), Barabási–Albert (scale-free),
  Watts–Strogatz (small-world).
- **Diffusion**: a node adopts once the fraction of adopting neighbors exceeds
  its individual threshold `theta_i` (sampled from a clipped normal
  distribution).
- **Seeding**: initial adopters chosen by degree or at random.

Each run's final adoption pattern becomes `R`, letting `xi_a`/`xi_d` measure how
equitably that diffusion reached the network under different topologies,
seeding strategies, and node weightings.

## Files

- `helper.py` — `xi_a` / `xi_d` metric implementations.
- `transfer.py` — Pigou–Dalton transfer-principle check for `xi_a` on a 3-node
  path graph (End–Center–End); plots `xi_a` before/after each transfer.
- `main.py` — runs the batch diffusion simulation (network kind × weighting ×
  targeting × repeats) and writes results/plots to `results/`.
- `slopes.py` - analyses the results from `main.py` and calculates Average Marginal Effects
- `sensitivity.py` — sweeps `theta_mean` × `theta_sd` and plots heatmaps of mean
  adoption/inequity by network kind.
- `tutorial.ipynb` / `tutorial.Rmd` — walkthrough of the metrics in Python and R.
- `/plots` - contains several scripts for plots.

## Running

```bash
python models/diffusion/main.py
python models/diffusion/sensitivity.py
python models/diffusion/transfer.py
```

Parameters (network size, thresholds, seed fraction, number of runs, etc.) are
edited directly at the top of `main.py`.
