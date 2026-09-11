"""
Implemented versions of the network inequity metrics. 
For more general descriptions of the two network inequity metrics, 
please see tutorial.ipynb or tutorial.Rmd for implementations in Python and R, respectively.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, Sequence

def xi_a(G: nx.Graph,
         R: Dict[int, float],
         w: Optional[Dict[int, float]] = None,
         alpha: float = 1.0,
         edge_weight: Optional[str] = None,
         normalize_A: bool = True) -> Tuple[float, Dict[int, float]]:

    """
    Compute the attenuated network equity metric (xi_a).

    This function measures how equitably resources are accessible across 
    the nodes of a network. Accessibility is determined by the
    relational distance between nodes, with attenuation applied so that 
    closer resources contribute more strongly than distant ones.

    The metric is defined as:
      - ξ = 0 indicates perfect equity (all nodes have full access to resources).
      - ξ = 1 indicates maximal inequity (nodes have no access to resources).

    Parameters
    ----------
    G : networkx.Graph
        The input graph representing the network topology.
    R : dict
        Node → resource value mapping (values in [0,1]).
        Represents how much resource each node holds.
    w : dict, optional
        Node → priority weight mapping (values in [0,1]).
        Assigns relative importance to nodes when aggregating inequity.
        If None, all nodes are equally weighted.
    alpha : float, default=1.0
        Distance decay parameter. Larger values increase the penalty 
        for distance, making nearby resources disproportionately more valuable.
    edge_weight : str, optional
        Name of the edge attribute in G to use as tie strength (values in [0,1]).
        If provided, stronger ties imply shorter effective distances 
        (distance = 1 / edge_weight). If None, graph is treated as unweighted.
    normalize_A : bool, default=True
        Whether to clip access scores A[i] to the interval [0,1].
        Prevents nodes from accumulating "super-accessibility" 
        when many nearby resources exist.

    Returns
    -------
    xi : float
        The network equity metric value in [0,1].
        Smaller values indicate more equitable access.
    A : dict
        Node → access score mapping (values in [0,1]).
        Higher A[i] means node i has better access to resources.
    """

    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return 1.0, {}
    
    # Default equal node weights
    if w is None:
        w = {node: 1.0 for node in nodes}
    
    # Compute distances with edge weight heterogeneity
    if edge_weight is not None:
        # Convert edge weights eij ∈ [0,1] to distances (inverse relationship)
        G_dist = nx.Graph()
        G_dist.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            edge_strength = data.get(edge_weight, 1.0)
            if edge_strength <= 0:
                edge_strength = 1e-6  # Avoid division by zero
            # Distance is inverse of edge weight: stronger connections = shorter distances
            G_dist.add_edge(u, v, weight=1.0 / edge_strength)
        
        try:
            dist_dict = dict(nx.all_pairs_dijkstra_path_length(G_dist, weight='weight'))
        except:
            # Fallback to unweighted if weighted computation fails
            dist_dict = dict(nx.all_pairs_shortest_path_length(G))
    else:
        # Unweighted shortest path distances
        dist_dict = dict(nx.all_pairs_shortest_path_length(G))
    
    # Compute access scores with distance attenuation
    A = {}
    for i in nodes:
        total_access = 0.0
        for j in nodes:
            d_ij = dist_dict[i].get(j, np.inf)
            if np.isfinite(d_ij):
                resource_j = R.get(j, 0.0)
                total_access += resource_j * np.exp(-alpha * d_ij)
        
        if normalize_A:
            total_access = min(total_access, 1.0)
        A[i] = total_access
    
    # Compute weighted network inequity metric
    total_weight = sum(w.values())
    if total_weight == 0:
        return 1.0, A
    
    xi = sum(w[i] * (1 - A[i]) for i in nodes) / total_weight
    return xi, A


def xi_a_multi(G: nx.Graph,
               R: Dict[int, float],
               w: Optional[Dict[int, float]] = None,
               alphas: Sequence[float] = (1.0,),
               edge_weight: Optional[str] = None,
               normalize_A: bool = True) -> Dict[float, Tuple[float, float]]:

    """
    Evaluate xi_a at several decay parameters in one pass.

    Equivalent to calling `xi_a` once per alpha, but the all-pairs distances
    (the expensive part) are computed once and shared, and the access sums are
    vectorised over numpy rather than looped in Python. A sweep over k alphas
    therefore costs roughly the same as a single `xi_a` call, not k times as much.
    This matters because alpha never affects the diffusion itself, only the
    post-hoc metric, so sweeping it is pure post-processing on fixed runs.

    Parameters
    ----------
    G, R, w, edge_weight, normalize_A
        As in `xi_a`.
    alphas : sequence of float
        Distance decay parameters to evaluate.

    Returns
    -------
    dict
        alpha → (xi, sat_frac), where `xi` matches what `xi_a` would return for
        that alpha and `sat_frac` is the fraction of nodes whose access score hit
        the `normalize_A` cap of 1.0. A high `sat_frac` means alpha is too small
        to discriminate: every node reads as fully served and xi collapses to 0.
        `sat_frac` is 0.0 when `normalize_A` is False.
    """

    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return {float(a): (1.0, 0.0) for a in alphas}

    if w is None:
        w = {node: 1.0 for node in nodes}

    # Same distance semantics as xi_a: optional edge strengths become inverse
    # distances, otherwise plain unweighted hop counts.
    if edge_weight is not None:
        G_dist = nx.Graph()
        G_dist.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            edge_strength = data.get(edge_weight, 1.0)
            if edge_strength <= 0:
                edge_strength = 1e-6
            G_dist.add_edge(u, v, weight=1.0 / edge_strength)

        try:
            dist_dict = dict(nx.all_pairs_dijkstra_path_length(G_dist, weight='weight'))
        except Exception:
            dist_dict = dict(nx.all_pairs_shortest_path_length(G))
    else:
        dist_dict = dict(nx.all_pairs_shortest_path_length(G))

    index = {node: i for i, node in enumerate(nodes)}
    D = np.full((n, n), np.inf)
    for i, targets in dist_dict.items():
        row = index[i]
        for j, d_ij in targets.items():
            D[row, index[j]] = d_ij

    r = np.array([R.get(node, 0.0) for node in nodes], dtype=float)
    wv = np.array([w[node] for node in nodes], dtype=float)
    total_weight = wv.sum()

    results: Dict[float, Tuple[float, float]] = {}
    for alpha in alphas:
        if total_weight == 0:
            results[float(alpha)] = (1.0, 0.0)
            continue

        # exp(-alpha * inf) underflows to 0, which is exactly the "unreachable
        # contributes nothing" branch in xi_a; errstate keeps numpy quiet about it.
        with np.errstate(over="ignore", invalid="ignore"):
            A = (np.exp(-alpha * D) * r).sum(axis=1)

        if normalize_A:
            sat_frac = float((A >= 1.0 - 1e-9).mean())
            A = np.minimum(A, 1.0)
        else:
            sat_frac = 0.0

        xi = float((wv * (1.0 - A)).sum() / total_weight)
        results[float(alpha)] = (xi, sat_frac)

    return results


# -----------------------------
# Discrete xi metric (binary reachability + node need weights)
# xi_d = weighted fraction of nodes unreachable from any resource holder within d hops.
# -----------------------------
def xi_d(G: nx.Graph,
         R: Dict[int, float],
         w: Optional[Dict[int, float]] = None,
         d: Optional[int] = None) -> Tuple[float, Dict[int, float]]:

    """
    Compute the discrete network equity metric (xi_d).

    This function measures how equitably a resource has spread across
    the nodes of a network, using binary (holds it / doesn't)
    reachability rather than the continuous attenuation used by xi_a.
    A node counts as having access if it lies within `d` hops of at
    least one resource holder; otherwise it counts as fully excluded.

    R plays the same conceptual role here as it does in xi_a: both
    mark which nodes currently hold the resource. xi_a treats R as a
    continuous source strength that decays smoothly with distance
    (`R[j] * exp(-alpha * d_ij)`), so partial, graded access is
    possible. xi_d only checks whether R is zero or not (`R[j] > 0`
    marks a holder), then applies a hard yes/no threshold: a node is
    either within `d` hops of a holder (access = 1) or it is not
    (access = 0), with no partial credit for being "almost" reachable.
    Feeding the same R to both is what makes differences between the
    two metrics on the same run reflect the attenuation-vs-cutoff
    modeling choice, not different underlying data.

    The metric is defined as:
      - ξ = 0 indicates perfect equity (every node is within d hops
        of a holder).
      - ξ = 1 indicates maximal inequity (no node is reachable, e.g.
        there are no holders at all).

    Parameters
    ----------
    G : networkx.Graph
        The input graph representing the network topology.
    R : dict
        Node → resource value mapping. Read as binary: any node with
        `R[node] > 0` is treated as a holder, regardless of magnitude.
    w : dict, optional
        Node → priority weight mapping (values in [0,1]).
        Assigns relative importance to nodes when aggregating inequity.
        If None, all nodes are equally weighted.
    d : int, optional
        Maximum hop distance at which a node is still considered to
        have access to a holder. `None` means no distance limit
        (a node has access if it can reach any holder at all, i.e.
        lies in the same connected component).

    Returns
    -------
    xi : float
        The network equity metric value in [0,1].
        Smaller values indicate more equitable access.
    A : dict
        Node → access score mapping (1.0 if within d hops of a
        holder, else 0.0).
    """

    nodes = list(G.nodes())
    if not nodes:
        return 1.0, {}

    if w is None:
        w = {node: 1.0 for node in nodes}

    total_weight = sum(w.values())
    if total_weight == 0:
        return 1.0, {}

    holders = [i for i, v in R.items() if v > 0]
    reachable: set = set()
    for h in holders:
        reachable.update(nx.single_source_shortest_path_length(G, h, cutoff=d))

    A = {i: (1.0 if i in reachable else 0.0) for i in nodes}

    xi = sum(w[i] * (1 - A[i]) for i in nodes) / total_weight
    return xi, A