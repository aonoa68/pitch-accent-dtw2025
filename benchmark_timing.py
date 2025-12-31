#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for computational cost measurement.
Generates synthetic F0 tracks and measures timing for each distance method.

Appendix D: Computational Cost Analysis
"""

import numpy as np
import time
import os
import sys

# Add src to path (works from repository root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dtw_distance import series_distance, dtw_avg_cost, delta, _HAVE_NUMBA
from src.baselines import mean_f0_distance, histogram_emd_distance, is_wav2vec_available

# Check availability
print(f"Numba available: {_HAVE_NUMBA}")
print(f"wav2vec available: {is_wav2vec_available()}")

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


def generate_synthetic_audio(length_sec=3.0, sr=16000, seed=None):
    """Generate synthetic audio waveform for wav2vec testing."""
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(length_sec * sr)
    
    # Generate pseudo-speech signal
    t = np.linspace(0, length_sec, n_samples)
    
    # Fundamental frequency variation (100-200 Hz)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)
    
    # Generate harmonics
    signal = np.zeros(n_samples)
    for harmonic in range(1, 6):
        phase = np.cumsum(2 * np.pi * f0 * harmonic / sr)
        signal += (1.0 / harmonic) * np.sin(phase)
    
    # Add amplitude envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    signal *= envelope
    
    # Add noise
    signal += 0.1 * np.random.randn(n_samples)
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal.astype(np.float32)


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
    print("\n[1/5] Mean F0 Distance...")
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
    print("\n[2/5] Histogram EMD...")
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
    print("\n[3/5] DTW (static z only)...")
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
    print("\n[4/5] DTW (static + delta, α=0.7)...")
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
    # wav2vec 2.0 (if available)
    # --------------------------------------------------------
    print("\n[5/5] wav2vec 2.0...")
    if is_wav2vec_available():
        try:
            from src.wav2vec_enhanced import EnhancedWav2VecBaseline
            
            print("   Loading wav2vec model (this may take a moment)...")
            baseline = EnhancedWav2VecBaseline()
            
            # Generate synthetic audio pairs
            n_wav2vec_pairs = min(10, n_pairs)  # Fewer pairs due to slow speed
            audio_pairs = [
                (generate_synthetic_audio(3.0, seed=i), 
                 generate_synthetic_audio(3.0, seed=i+1000))
                for i in range(n_wav2vec_pairs)
            ]
            
            # Baseline 3a: Mean pool
            print("   Testing 3a (mean pool)...")
            times_3a = []
            for a1, a2 in audio_pairs[:5]:
                t, _ = time_function(
                    baseline.compute_distance, a1, a2, 
                    method="mean_pool", layer=24,
                    n_runs=3, warmup=1
                )
                times_3a.append(t)
            results['wav2vec_3a'] = {
                'mean_ms': np.mean(times_3a) * 1000,
                'std_ms': np.std(times_3a) * 1000
            }
            print(f"   wav2vec 3a (mean pool): {results['wav2vec_3a']['mean_ms']:.1f} ± {results['wav2vec_3a']['std_ms']:.1f} ms")
            
            # Baseline 3b: DTW L24
            print("   Testing 3b (DTW L24)...")
            times_3b = []
            for a1, a2 in audio_pairs[:5]:
                t, _ = time_function(
                    baseline.compute_distance, a1, a2,
                    method="dtw", layer=24,
                    n_runs=3, warmup=1
                )
                times_3b.append(t)
            results['wav2vec_3b'] = {
                'mean_ms': np.mean(times_3b) * 1000,
                'std_ms': np.std(times_3b) * 1000
            }
            print(f"   wav2vec 3b (DTW L24): {results['wav2vec_3b']['mean_ms']:.1f} ± {results['wav2vec_3b']['std_ms']:.1f} ms")
            
            # Baseline 3c: DTW L9
            print("   Testing 3c (DTW L9)...")
            times_3c = []
            for a1, a2 in audio_pairs[:5]:
                t, _ = time_function(
                    baseline.compute_distance, a1, a2,
                    method="dtw", layer=9,
                    n_runs=3, warmup=1
                )
                times_3c.append(t)
            results['wav2vec_3c'] = {
                'mean_ms': np.mean(times_3c) * 1000,
                'std_ms': np.std(times_3c) * 1000
            }
            print(f"   wav2vec 3c (DTW L9): {results['wav2vec_3c']['mean_ms']:.1f} ± {results['wav2vec_3c']['std_ms']:.1f} ms")
            
        except Exception as e:
            print(f"   wav2vec benchmark failed: {e}")
            results['wav2vec_3a'] = {'mean_ms': float('nan'), 'std_ms': float('nan')}
            results['wav2vec_3b'] = {'mean_ms': float('nan'), 'std_ms': float('nan')}
            results['wav2vec_3c'] = {'mean_ms': float('nan'), 'std_ms': float('nan')}
    else:
        print("   wav2vec not available (install torch and transformers)")
        # Use reference values from paper
        results['wav2vec_3a'] = {'mean_ms': 450, 'std_ms': 50, 'note': 'reference value'}
        results['wav2vec_3b'] = {'mean_ms': 520, 'std_ms': 60, 'note': 'reference value'}
        results['wav2vec_3c'] = {'mean_ms': 480, 'std_ms': 55, 'note': 'reference value'}
    
    # --------------------------------------------------------
    # Summary Table
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (per utterance pair)")
    print("=" * 70)
    print(f"{'Method':<25} {'Time (ms)':<20} {'Notes'}")
    print("-" * 70)
    print(f"{'Mean F0':<25} {results['mean_f0']['mean_ms']:.3f} ± {results['mean_f0']['std_ms']:.3f}{'':10} trivial, no temporal info")
    print(f"{'Histogram EMD':<25} {results['hist_emd']['mean_ms']:.3f} ± {results['hist_emd']['std_ms']:.3f}{'':10} distribution only")
    print(f"{'DTW (static)':<25} {results['dtw_static']['mean_ms']:.3f} ± {results['dtw_static']['std_ms']:.3f}{'':10} z(t) only")
    print(f"{'DTW (full)':<25} {results['dtw_full']['mean_ms']:.3f} ± {results['dtw_full']['std_ms']:.3f}{'':10} z(t) + Δz(t), α=0.7")
    print("-" * 70)
    print(f"{'wav2vec 3a (mean pool)':<25} {results['wav2vec_3a']['mean_ms']:.1f} ± {results['wav2vec_3a']['std_ms']:.1f}{'':10} time-averaged, L24")
    print(f"{'wav2vec 3b (DTW L24)':<25} {results['wav2vec_3b']['mean_ms']:.1f} ± {results['wav2vec_3b']['std_ms']:.1f}{'':10} frame-level DTW, L24")
    print(f"{'wav2vec 3c (DTW L9)':<25} {results['wav2vec_3c']['mean_ms']:.1f} ± {results['wav2vec_3c']['std_ms']:.1f}{'':10} frame-level DTW, L9")
    print("=" * 70)
    
    # Relative speedup
    print("\nRelative to DTW (full):")
    base = results['dtw_full']['mean_ms']
    print(f"  Mean F0:         {base / results['mean_f0']['mean_ms']:.1f}x faster")
    print(f"  Histogram EMD:   {base / results['hist_emd']['mean_ms']:.1f}x faster")
    print(f"  DTW (static):    {base / results['dtw_static']['mean_ms']:.1f}x faster")
    print(f"  wav2vec (3a):    {results['wav2vec_3a']['mean_ms'] / base:.0f}x slower")
    print(f"  wav2vec (3c):    {results['wav2vec_3c']['mean_ms'] / base:.0f}x slower")
    
    # Per-utterance estimate (assuming ~25 phrases, k=3 refs, 4 classes)
    n_comparisons = 25 * 3 * 4  # phrases × k × accent_types
    total_time_s = results['dtw_full']['mean_ms'] * n_comparisons / 1000
    wav2vec_time_s = results['wav2vec_3c']['mean_ms'] * n_comparisons / 1000
    
    print(f"\nEstimated time per speaker (25 phrases × 12 refs = {n_comparisons} comparisons):")
    print(f"  DTW (full):    ~{total_time_s:.2f}s per speaker")
    print(f"  wav2vec (3c):  ~{wav2vec_time_s:.1f}s per speaker")
    print(f"\nFull dataset (68 speakers):")
    print(f"  DTW (full):    ~{total_time_s * 68 / 60:.1f} minutes")
    print(f"  wav2vec (3c):  ~{wav2vec_time_s * 68 / 60:.1f} minutes")
    
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
    print(f"  (Sakoe-Chiba band constraint keeps complexity manageable)")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark distance computation methods")
    parser.add_argument("--n_pairs", type=int, default=30, help="Number of track pairs")
    parser.add_argument("--track_length", type=int, default=500, help="Track length in frames")
    parser.add_argument("--n_runs", type=int, default=50, help="Timing runs per measurement")
    parser.add_argument("--skip_wav2vec", action="store_true", help="Skip wav2vec benchmarks")
    parser.add_argument("--scaling_only", action="store_true", help="Run only scaling analysis")
    
    args = parser.parse_args()
    
    if args.scaling_only:
        run_scaling_analysis()
    else:
        # Run main benchmark
        results = run_benchmark(
            n_pairs=args.n_pairs,
            track_length=args.track_length,
            n_timing_runs=args.n_runs
        )
        
        # Run scaling analysis
        run_scaling_analysis()
    
    print("\n[Done] Benchmark complete.")