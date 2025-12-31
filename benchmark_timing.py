#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for computational cost measurement.
Generates synthetic F0 tracks and measures timing for each distance method.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/home/claude/pitch-accent')

from src.dtw_distance import series_distance, dtw_avg_cost, delta, _HAVE_NUMBA
from src.baselines import mean_f0_distance, histogram_emd_distance

# Check numba availability
print(f"Numba available: {_HAVE_NUMBA}")

# ============================================================
# Synthetic F0 Track Generation
# ============================================================

def generate_synthetic_f0(length=500, voiced_ratio=0.85, seed=None):
    """
    Generate a synthetic normalized F0 track.
    
    Simulates typical pitch patterns with:
    - Realistic length (~500 frames = ~3-5 seconds at 16kHz/160hop)
    - Natural pitch contour with declination
    - Voicing gaps (unvoiced frames as NaN)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base contour: slight declination + phrase structure
    t = np.linspace(0, 1, length)
    base = -0.5 * t  # Natural declination
    
    # Add pitch accent patterns (sine-like variations)
    accent = 0.8 * np.sin(2 * np.pi * t * 3)
    
    # Add small random variations
    noise = np.random.normal(0, 0.2, length)
    
    z = base + accent + noise
    
    # Add voicing gaps (NaN for unvoiced frames)
    mask = np.random.random(length) < voiced_ratio
    # Also add some contiguous gaps
    gap_starts = np.random.choice(length - 20, size=3, replace=False)
    for gs in gap_starts:
        gap_len = np.random.randint(5, 15)
        mask[gs:gs+gap_len] = False
    
    z[~mask] = np.nan
    
    return z


def generate_track_pair(length=500, similarity=0.7, seed=None):
    """Generate a pair of tracks with controlled similarity."""
    if seed is not None:
        np.random.seed(seed)
    
    z1 = generate_synthetic_f0(length, seed=seed)
    
    # Second track: similar but with variations
    z2 = z1.copy()
    valid = np.isfinite(z2)
    z2[valid] += np.random.normal(0, 0.3 * (1 - similarity), valid.sum())
    
    # Slightly different voicing pattern
    shift = np.random.randint(-10, 10)
    if shift > 0:
        z2 = np.concatenate([np.full(shift, np.nan), z2[:-shift]])
    elif shift < 0:
        z2 = np.concatenate([z2[-shift:], np.full(-shift, np.nan)])
    
    return z1, z2


# ============================================================
# Timing Functions
# ============================================================

def time_function(func, *args, n_runs=100, warmup=10, **kwargs):
    """
    Time a function with warmup runs.
    Returns mean and std of execution time in seconds.
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


# ============================================================
# Main Benchmark
# ============================================================

def run_benchmark(n_pairs=50, track_length=500, n_timing_runs=100):
    """
    Run comprehensive benchmark on all distance methods.
    """
    print("=" * 60)
    print("COMPUTATIONAL COST BENCHMARK")
    print("=" * 60)
    print(f"Track length: {track_length} frames (~{track_length * 0.01:.1f}s at 16kHz/160hop)")
    print(f"Number of pairs: {n_pairs}")
    print(f"Timing runs per pair: {n_timing_runs}")
    print()
    
    # Generate test data
    print("Generating synthetic track pairs...")
    pairs = [generate_track_pair(track_length, seed=i) for i in range(n_pairs)]
    
    results = {}
    
    # --------------------------------------------------------
    # Baseline 1: Mean F0 Distance
    # --------------------------------------------------------
    print("\n[1/4] Mean F0 Distance...")
    times_mean = []
    for z1, z2 in pairs:
        mean_t, _ = time_function(mean_f0_distance, z1, z2, n_runs=n_timing_runs, warmup=5)
        times_mean.append(mean_t)
    
    results['mean_f0'] = {
        'mean_ms': np.mean(times_mean) * 1000,
        'std_ms': np.std(times_mean) * 1000
    }
    print(f"   Mean F0: {results['mean_f0']['mean_ms']:.4f} ± {results['mean_f0']['std_ms']:.4f} ms")
    
    # --------------------------------------------------------
    # Baseline 2: Histogram EMD
    # --------------------------------------------------------
    print("\n[2/4] Histogram EMD...")
    times_emd = []
    for z1, z2 in pairs:
        emd_t, _ = time_function(histogram_emd_distance, z1, z2, n_runs=n_timing_runs, warmup=5)
        times_emd.append(emd_t)
    
    results['hist_emd'] = {
        'mean_ms': np.mean(times_emd) * 1000,
        'std_ms': np.std(times_emd) * 1000
    }
    print(f"   Histogram EMD: {results['hist_emd']['mean_ms']:.4f} ± {results['hist_emd']['std_ms']:.4f} ms")
    
    # --------------------------------------------------------
    # DTW (static only)
    # --------------------------------------------------------
    print("\n[3/4] DTW (static z only)...")
    times_dtw_static = []
    for z1, z2 in pairs:
        dtw_t, _ = time_function(dtw_avg_cost, z1, z2, n_runs=n_timing_runs, warmup=5)
        times_dtw_static.append(dtw_t)
    
    results['dtw_static'] = {
        'mean_ms': np.mean(times_dtw_static) * 1000,
        'std_ms': np.std(times_dtw_static) * 1000
    }
    print(f"   DTW (static): {results['dtw_static']['mean_ms']:.4f} ± {results['dtw_static']['std_ms']:.4f} ms")
    
    # --------------------------------------------------------
    # DTW (static + delta) - Full pipeline
    # --------------------------------------------------------
    print("\n[4/4] DTW (static + delta, α=0.7)...")
    times_dtw_full = []
    for z1, z2 in pairs:
        full_t, _ = time_function(series_distance, z1, z2, n_runs=n_timing_runs, warmup=5)
        times_dtw_full.append(full_t)
    
    results['dtw_full'] = {
        'mean_ms': np.mean(times_dtw_full) * 1000,
        'std_ms': np.std(times_dtw_full) * 1000
    }
    print(f"   DTW (full): {results['dtw_full']['mean_ms']:.4f} ± {results['dtw_full']['std_ms']:.4f} ms")
    
    # --------------------------------------------------------
    # Summary Table
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (per utterance pair)")
    print("=" * 60)
    print(f"{'Method':<20} {'Time (ms)':<15} {'Notes'}")
    print("-" * 60)
    print(f"{'Mean F0':<20} {results['mean_f0']['mean_ms']:.3f} ± {results['mean_f0']['std_ms']:.3f}{'':5} trivial, no temporal info")
    print(f"{'Histogram EMD':<20} {results['hist_emd']['mean_ms']:.3f} ± {results['hist_emd']['std_ms']:.3f}{'':5} distribution only")
    print(f"{'DTW (static)':<20} {results['dtw_static']['mean_ms']:.3f} ± {results['dtw_static']['std_ms']:.3f}{'':5} z(t) only")
    print(f"{'DTW (full)':<20} {results['dtw_full']['mean_ms']:.3f} ± {results['dtw_full']['std_ms']:.3f}{'':5} z(t) + Δz(t), α=0.7")
    print("=" * 60)
    
    # Relative speedup
    print("\nRelative to DTW (full):")
    base = results['dtw_full']['mean_ms']
    print(f"  Mean F0:       {base / results['mean_f0']['mean_ms']:.1f}x faster")
    print(f"  Histogram EMD: {base / results['hist_emd']['mean_ms']:.1f}x faster")
    print(f"  DTW (static):  {base / results['dtw_static']['mean_ms']:.1f}x faster")
    
    # Per-utterance estimate (assuming ~25 phrases, k=3 refs, 4 classes)
    n_comparisons = 25 * 3 * 4  # phrases × k × accent_types
    total_time_s = results['dtw_full']['mean_ms'] * n_comparisons / 1000
    print(f"\nEstimated time per speaker (68 speakers × 25 phrases × 12 refs):")
    print(f"  DTW (full): ~{total_time_s:.2f}s per speaker")
    print(f"  Full dataset (68 speakers): ~{total_time_s * 68 / 60:.1f} minutes")
    
    return results


# ============================================================
# Scaling Analysis
# ============================================================

def run_scaling_analysis():
    """Analyze how computation time scales with sequence length."""
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS (DTW full)")
    print("=" * 60)
    
    lengths = [100, 200, 300, 500, 750, 1000]
    times = []
    
    for L in lengths:
        z1, z2 = generate_track_pair(L, seed=42)
        t, _ = time_function(series_distance, z1, z2, n_runs=50, warmup=5)
        times.append(t * 1000)
        print(f"  Length {L:>4}: {t*1000:.3f} ms")
    
    # Estimate complexity
    print("\nScaling behavior:")
    ratio = times[-1] / times[0]
    len_ratio = lengths[-1] / lengths[0]
    estimated_exp = np.log(ratio) / np.log(len_ratio)
    print(f"  Time ratio ({lengths[-1]}/{lengths[0]}): {ratio:.2f}x")
    print(f"  Estimated complexity: O(n^{estimated_exp:.2f})")


if __name__ == "__main__":
    # Run main benchmark
    results = run_benchmark(n_pairs=30, track_length=500, n_timing_runs=50)
    
    # Run scaling analysis
    run_scaling_analysis()
    
    print("\n[Done] Benchmark complete.")
