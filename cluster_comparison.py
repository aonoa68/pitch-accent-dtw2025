#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare clustering methods on the saved 68x68 FastDTW distance matrix.

Methods compared:
  - Ward linkage on raw DTW matrix (paper's current choice; technically
    requires Euclidean inputs)
  - Average linkage on raw DTW matrix (defensible for arbitrary
    dissimilarities)
  - PAM (k-medoids) on raw DTW matrix (no metric assumption)
  - Ward linkage on MDS-Euclidean coordinates (theoretically clean Ward)

Reports cluster sizes, silhouette, and Adjusted Rand Index between
methods for each k in {2,3,4,5,6}.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import MDS
# PAM (k-medoids) implemented inline to avoid sklearn-extra dependency.


def cluster_hier(D, method, k):
    cond = squareform(D, checks=False)
    Z = linkage(cond, method=method)
    return fcluster(Z, k, criterion="maxclust") - 1


def cluster_pam(D, k, seed=42, max_iter=300):
    """Minimal PAM (BUILD + SWAP) on a precomputed dissimilarity matrix."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]

    # BUILD: greedy initial medoids
    medoids = [int(np.argmin(D.sum(axis=1)))]
    while len(medoids) < k:
        not_m = [i for i in range(n) if i not in medoids]
        best, best_gain = None, -1.0
        cur = D[:, medoids].min(axis=1)
        for cand in not_m:
            new = np.minimum(cur, D[:, cand])
            gain = (cur - new).sum()
            if gain > best_gain:
                best_gain, best = gain, cand
        medoids.append(best)

    # SWAP
    for _ in range(max_iter):
        cur = D[:, medoids].min(axis=1)
        cur_cost = cur.sum()
        best_swap, best_delta = None, 0.0
        for mi, m in enumerate(medoids):
            others = [x for x in medoids if x != m]
            for cand in range(n):
                if cand in medoids:
                    continue
                new_med = others + [cand]
                new_cost = D[:, new_med].min(axis=1).sum()
                delta = cur_cost - new_cost
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (mi, cand)
        if best_swap is None:
            break
        mi, cand = best_swap
        medoids[mi] = cand

    labels = D[:, medoids].argmin(axis=1)
    return labels


def cluster_ward_on_mds(D, k, seed=42):
    """Embed via MDS, then run Ward on Euclidean coords."""
    coords = MDS(n_components=min(20, D.shape[0] - 1),
                 dissimilarity="precomputed", random_state=seed,
                 normalized_stress="auto").fit_transform(D)
    Z = linkage(coords, method="ward")
    return fcluster(Z, k, criterion="maxclust") - 1


def main():
    D = np.load(os.path.join(os.path.dirname(__file__),
                             "..", "..", "tacl2025", "figures",
                             "unsup_D_total.npy"))
    print(f"Distance matrix: {D.shape}")
    n = D.shape[0]

    results = {}
    for k in [2, 3, 4, 5, 6]:
        row = {"k": k}
        methods = {
            "ward_raw":   cluster_hier(D, "ward", k),
            "avg_raw":    cluster_hier(D, "average", k),
            "complete":   cluster_hier(D, "complete", k),
            "pam":        cluster_pam(D, k),
            "ward_mds":   cluster_ward_on_mds(D, k),
        }
        for name, labels in methods.items():
            try:
                sil = silhouette_score(D, labels, metric="precomputed")
            except Exception:
                sil = float("nan")
            sizes = sorted([(labels == c).sum() for c in np.unique(labels)],
                           reverse=True)
            row[f"{name}_sil"] = round(sil, 3)
            row[f"{name}_sizes"] = sizes
        # ARI: Ward_raw vs each alternative
        for name in ["avg_raw", "complete", "pam", "ward_mds"]:
            row[f"ari_ward_vs_{name}"] = round(
                adjusted_rand_score(methods["ward_raw"], methods[name]), 3)
        results[k] = row
        print(f"\n=== k={k} ===")
        for key, val in row.items():
            print(f"  {key}: {val}")

    out_path = os.path.join(os.path.dirname(__file__), "outputs",
                            "cluster_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if hasattr(x, "__iter__") else x)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
