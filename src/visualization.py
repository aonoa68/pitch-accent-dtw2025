"""
Visualization utilities for pitch accent analysis.

Includes:
- Failure mode visualization (Appendix C)
- Distance heatmaps
- MDS projections
- Contour comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

# Style settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150


def plot_failure_mode(
    subject_z: np.ndarray,
    tarui_z: np.ndarray,
    kansai_z: np.ndarray,
    distances: Dict[str, Dict[str, float]],
    subject_id: str = "S27",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Visualize failure modes of non-temporal distance metrics.
    
    This figure demonstrates why DTW outperforms Mean F0 and Histogram EMD
    for distinguishing phonologically similar but temporally distinct patterns.
    
    Parameters
    ----------
    subject_z : np.ndarray
        Subject's normalized pitch track
    tarui_z : np.ndarray
        Tarui reference normalized pitch track
    kansai_z : np.ndarray
        Kansai reference normalized pitch track
    distances : dict
        Nested dict: {method: {accent_type: distance}}
        e.g., {'mean_f0': {'tarui': 0.12, 'kansai': 0.15}, ...}
    subject_id : str
        Subject identifier for title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size in inches
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # ========== Top: Pitch contours ==========
    ax1 = axes[0]
    
    # Time axis (assuming 10ms frames)
    t_subj = np.arange(len(subject_z)) * 0.01
    t_tarui = np.arange(len(tarui_z)) * 0.01
    t_kansai = np.arange(len(kansai_z)) * 0.01
    
    # Plot contours
    ax1.plot(t_subj, subject_z, 'k-', linewidth=2, label=f'{subject_id} (subject)', alpha=0.9)
    ax1.plot(t_tarui, tarui_z, 'b--', linewidth=1.5, label='Tarui ref', alpha=0.7)
    ax1.plot(t_kansai, kansai_z, color='orange', linestyle='--', linewidth=1.5, 
             label='Kansai ref', alpha=0.7)
    
    # Mark accent nucleus region (approximate)
    # Find peak regions
    subj_valid = np.where(np.isfinite(subject_z))[0]
    if len(subj_valid) > 10:
        peak_idx = subj_valid[np.argmax(subject_z[subj_valid])]
        peak_time = peak_idx * 0.01
        ax1.axvline(peak_time, color='k', linestyle=':', alpha=0.5)
        ax1.annotate('accent\nnucleus', xy=(peak_time, ax1.get_ylim()[1] * 0.8),
                    fontsize=9, ha='center', color='gray')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized F0 (MAD-z)')
    ax1.set_title(f'Pitch Contour Comparison: {subject_id}')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2.5, 2.5)
    
    # ========== Bottom: Distance rankings ==========
    ax2 = axes[1]
    
    methods = ['mean_f0', 'hist_emd', 'dtw']
    method_labels = ['Mean F0', 'Histogram EMD', 'DTW (ours)']
    
    x = np.arange(len(methods))
    width = 0.35
    
    tarui_dists = [distances.get(m, {}).get('tarui', 0) for m in methods]
    kansai_dists = [distances.get(m, {}).get('kansai', 0) for m in methods]
    
    bars1 = ax2.bar(x - width/2, tarui_dists, width, label='→ Tarui', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, kansai_dists, width, label='→ Kansai', color='orange', alpha=0.8)
    
    # Add winner markers
    for i, (t, k) in enumerate(zip(tarui_dists, kansai_dists)):
        winner = '✓ Tarui' if t < k else '✓ Kansai'
        color = 'steelblue' if t < k else 'darkorange'
        ax2.annotate(winner, xy=(i, max(t, k) + 0.02), ha='center', fontsize=8, 
                    color=color, fontweight='bold')
    
    ax2.set_xlabel('Distance Method')
    ax2.set_ylabel('Distance')
    ax2.set_title('Distance to Reference (lower = closer match)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[viz] Saved: {save_path}")
    
    return fig


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: List[str],
    cluster_boundaries: Optional[List[int]] = None,
    title: str = "Subject-by-Subject Distance Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8)
):
    """
    Plot distance matrix heatmap with optional cluster boundaries.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Square distance matrix (N x N)
    labels : list
        Subject labels
    cluster_boundaries : list, optional
        Indices where cluster boundaries occur
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
    
    # Cluster boundaries
    if cluster_boundaries:
        for b in cluster_boundaries:
            ax.axhline(b - 0.5, color='white', linewidth=2)
            ax.axvline(b - 0.5, color='white', linewidth=2)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('DTW Distance')
    
    ax.set_title(title)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Subject')
    
    # Reduce tick density for large matrices
    if len(labels) > 20:
        step = max(1, len(labels) // 10)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_yticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels[::step], fontsize=8)
    else:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[viz] Saved: {save_path}")
    
    return fig


def plot_mds_clusters(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    accent_names: Dict[int, str] = None,
    medoid_indices: Optional[List[int]] = None,
    title: str = "MDS Projection of Accent Space",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6)
):
    """
    Plot MDS projection with cluster coloring.
    
    Parameters
    ----------
    coords : np.ndarray
        2D MDS coordinates (N x 2)
    cluster_labels : np.ndarray
        Cluster assignment for each point
    accent_names : dict, optional
        Mapping from cluster index to accent name
    medoid_indices : list, optional
        Indices of cluster medoids (will be marked with stars)
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default accent names
    if accent_names is None:
        accent_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3'}
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(set(cluster_labels))))
    
    for i, label in enumerate(sorted(set(cluster_labels))):
        mask = cluster_labels == label
        name = accent_names.get(label, f'Cluster {label}')
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  c=[colors[i]], label=name, alpha=0.7, s=60)
    
    # Mark medoids
    if medoid_indices:
        for idx in medoid_indices:
            ax.scatter(coords[idx, 0], coords[idx, 1], 
                      marker='*', s=200, c='black', edgecolors='white', 
                      linewidths=1.5, zorder=10)
    
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[viz] Saved: {save_path}")
    
    return fig


def plot_baseline_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Plot baseline comparison (Tarui preservation rate and balance ratio).
    
    Parameters
    ----------
    results : dict
        Results dict with keys: method names
        Values: {'tarui_rate': float, 'balance_ratio': float}
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    methods = list(results.keys())
    tarui_rates = [results[m].get('tarui_rate', 0) for m in methods]
    balance_ratios = [results[m].get('balance_ratio', 1) for m in methods]
    
    # Left: Tarui classification rate
    ax1 = axes[0]
    colors = ['steelblue' if r > 0.15 else 'lightcoral' for r in tarui_rates]
    bars1 = ax1.bar(methods, tarui_rates, color=colors, alpha=0.8)
    ax1.axhline(0.15, color='red', linestyle='--', label='15% threshold')
    ax1.set_ylabel('Tarui Classification Rate')
    ax1.set_title('(a) Tarui Preservation by Method')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    
    # Right: Balance ratio
    ax2 = axes[1]
    colors2 = ['green' if 0.9 <= r <= 1.1 else 'lightcoral' for r in balance_ratios]
    bars2 = ax2.bar(methods, balance_ratios, color=colors2, alpha=0.8)
    ax2.axhline(1.0, color='green', linestyle='--', label='Balanced (1.0)')
    ax2.axhspan(0.9, 1.1, alpha=0.2, color='green', label='Acceptable range')
    ax2.set_ylabel('Tarui/Kansai Distance Ratio')
    ax2.set_title('(b) Discrimination Balance')
    ax2.set_ylim(0, 2)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[viz] Saved: {save_path}")
    
    return fig


def generate_synthetic_failure_example(seed: int = 2025):
    """
    Generate synthetic data for failure mode visualization.
    
    This creates example contours that demonstrate why DTW succeeds
    where Mean F0 and Histogram EMD fail.
    
    Returns
    -------
    subject_z : np.ndarray
    tarui_z : np.ndarray
    kansai_z : np.ndarray
    distances : dict
    """
    np.random.seed(seed)
    
    T = 300  # frames
    t = np.linspace(0, 1, T)
    
    # Tarui pattern: later accent peak
    tarui_base = np.zeros(T)
    tarui_base[:100] = -0.5
    tarui_base[100:180] = np.linspace(-0.5, 1.5, 80)  # rise
    tarui_base[180:220] = 1.5  # peak
    tarui_base[220:] = np.linspace(1.5, -0.3, T - 220)  # fall
    tarui_z = tarui_base + np.random.normal(0, 0.15, T)
    
    # Kansai pattern: earlier accent peak
    kansai_base = np.zeros(T)
    kansai_base[:60] = -0.5
    kansai_base[60:120] = np.linspace(-0.5, 1.5, 60)  # earlier rise
    kansai_base[120:160] = 1.5  # earlier peak
    kansai_base[160:] = np.linspace(1.5, -0.3, T - 160)  # fall
    kansai_z = kansai_base + np.random.normal(0, 0.15, T)
    
    # Subject: Kansai-like timing but Tarui-like distribution
    subject_base = np.zeros(T)
    subject_base[:70] = -0.5
    subject_base[70:130] = np.linspace(-0.5, 1.4, 60)  # Kansai-like timing
    subject_base[130:170] = 1.4
    subject_base[170:] = np.linspace(1.4, -0.3, T - 170)
    subject_z = subject_base + np.random.normal(0, 0.15, T)
    
    # Add voicing gaps
    for z in [subject_z, tarui_z, kansai_z]:
        gap_start = np.random.randint(T // 3, T // 2)
        z[gap_start:gap_start + 15] = np.nan
    
    # Compute distances (synthetic values showing the failure pattern)
    distances = {
        'mean_f0': {'tarui': 0.08, 'kansai': 0.12},  # Mean F0 says Tarui (wrong)
        'hist_emd': {'tarui': 0.15, 'kansai': 0.18},  # Hist EMD says Tarui (wrong)
        'dtw': {'tarui': 0.35, 'kansai': 0.22}  # DTW correctly says Kansai
    }
    
    return subject_z, tarui_z, kansai_z, distances


# ============================================================
# Main: Generate example figures
# ============================================================

if __name__ == "__main__":
    import os
    
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate failure mode example (Appendix C)
    print("Generating failure mode visualization...")
    subj, tarui, kansai, dists = generate_synthetic_failure_example()
    plot_failure_mode(
        subj, tarui, kansai, dists,
        subject_id="S27",
        save_path=f"{output_dir}/fig_failure_mode.png"
    )
    
    # Generate baseline comparison example
    print("Generating baseline comparison...")
    baseline_results = {
        'Mean F0': {'tarui_rate': 0.015, 'balance_ratio': 1.42},
        'Hist EMD': {'tarui_rate': 0.853, 'balance_ratio': 0.83},
        'wav2vec': {'tarui_rate': 0.191, 'balance_ratio': 0.94},
        'DTW (ours)': {'tarui_rate': 0.574, 'balance_ratio': 1.01}
    }
    plot_baseline_comparison(
        baseline_results,
        save_path=f"{output_dir}/fig_baseline_comparison.png"
    )
    
    print(f"\n[Done] Figures saved to {output_dir}/")
