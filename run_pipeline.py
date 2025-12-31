#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for pitch accent distance pipeline.

Usage:
    python run_pipeline.py \
        --subject_dir /path/to/subjects \
        --reference_dir /path/to/references \
        --excel_path data/accent_types.xlsx \
        --output_dir outputs/

For wav2vec enhanced baselines:
    python run_pipeline.py ... --wav2vec_enhanced

For unsupervised analysis:
    python run_pipeline.py ... --unsupervised
"""

import os
import glob
import argparse
import math
import random
import numpy as np
import pandas as pd
import yaml

from src.extractor import TrackExtractor
from src.classifier import AccentClassifier, two_stage_classify, calibrate_thresholds
from src.baselines import BaselineEvaluator, is_wav2vec_available
from src.virtual_reference import register_virtual_from_excel
from src.dtw_distance import series_distance

ACCENT_TYPES = ["tokyo", "kansai", "tarui", "kagoshima"]
SEED = 2025


def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def list_subject_files(subject_dir, exclude_files=None):
    """List subject audio files."""
    if not os.path.exists(subject_dir):
        print(f"[warn] SUBJECT_DIR not found: {subject_dir}")
        return []
    
    exclude = exclude_files or set()
    files = list(set(
        glob.glob(f"{subject_dir}/*.WAV") +
        glob.glob(f"{subject_dir}/*.wav")
    ))
    files = [f for f in files if os.path.basename(f) not in exclude]
    return sorted(files)


def list_reference_files(reference_dir):
    """List reference audio files by accent type."""
    if not os.path.exists(reference_dir):
        print(f"[warn] REFERENCE_DIR not found: {reference_dir}")
        return {}
    
    refs = {}
    for a in ACCENT_TYPES:
        cand = list(set(glob.glob(f"{reference_dir}/{a}*.[Ww][Aa][Vv]")))
        if a == "kagoshima":
            alias = list(set(glob.glob(f"{reference_dir}/none*.[Ww][Aa][Vv]")))
            cand = list(set(cand + alias))
        refs[a] = sorted(cand)
    return refs


def run_dtw_classification(subject_files, classifier, amb_abs, amb_rel):
    """Run DTW-based classification on all subjects."""
    rows = []
    
    for path in subject_files:
        z, vr, dist = classifier.distance_vector(path)
        
        if z is None:
            rows.append({
                'subject': os.path.basename(path),
                'voiced_ratio': vr,
                **{f'{a}_distance': np.inf for a in ACCENT_TYPES}
            })
            continue
        
        # Classification
        finite = {k: v for k, v in dist.items() if np.isfinite(v)}
        if len(finite) == 0:
            lab, ma, mr = 'unknown', np.nan, np.nan
        elif len(finite) == 1:
            lab, ma, mr = 'ambiguous', np.nan, np.nan
        else:
            lab, ma, mr = two_stage_classify(finite, amb_abs, amb_rel)
        
        rows.append({
            'subject': os.path.basename(path),
            'voiced_ratio': vr,
            **{f'{a}_distance': dist[a] for a in ACCENT_TYPES},
            'closest_accent': lab,
            'margin': ma,
            'rel_margin': mr
        })
    
    return pd.DataFrame(rows)


def run_baselines(subject_files, refs, extractor):
    """Run baseline evaluations (Mean F0, Histogram EMD, original wav2vec)."""
    evaluator = BaselineEvaluator(extractor, refs)
    rows = []
    
    for i, sp in enumerate(subject_files):
        if (i + 1) % 10 == 0:
            print(f"[baseline] Processing {i+1}/{len(subject_files)}...")
        rows.append(evaluator.evaluate_subject(sp))
    
    return pd.DataFrame(rows)


def run_wav2vec_enhanced(subject_files, refs, output_dir):
    """
    Run enhanced wav2vec baselines (3a, 3b, 3c).
    
    - 3a: Time-averaged embeddings (mean pool, L24)
    - 3b: Frame-level DTW on final layer (L24)
    - 3c: Frame-level DTW on middle layer (L9) - BEST
    """
    try:
        from src.wav2vec_enhanced import EnhancedWav2VecBaseline
    except ImportError as e:
        print(f"[wav2vec_enhanced] Import failed: {e}")
        print("[wav2vec_enhanced] Install: pip install torch torchaudio transformers")
        return None
    
    print("[wav2vec_enhanced] Loading model...")
    baseline = EnhancedWav2VecBaseline()
    
    # Flatten reference files
    ref_files = []
    ref_labels = []
    for accent, files in refs.items():
        for f in files:
            ref_files.append(f)
            ref_labels.append(accent)
    
    if not ref_files:
        print("[wav2vec_enhanced] No reference files found")
        return None
    
    results = []
    
    # Configurations to test
    configs = [
        ('3a', 'mean_pool', 24),
        ('3b', 'dtw', 24),
        ('3c', 'dtw', 9),
    ]
    
    for config_name, method, layer in configs:
        print(f"\n[wav2vec_enhanced] Running baseline {config_name} ({method}, L{layer})...")
        
        rows = []
        for i, subj_path in enumerate(subject_files):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i+1}/{len(subject_files)}...")
            
            try:
                # Compute distances to each accent class
                dists = {a: [] for a in ACCENT_TYPES}
                
                for ref_path, ref_label in zip(ref_files, ref_labels):
                    d = baseline.compute_distance_from_files(
                        subj_path, ref_path, method=method, layer=layer
                    )
                    if np.isfinite(d):
                        dists[ref_label].append(d)
                
                # Average distances per class
                avg_dists = {}
                for a in ACCENT_TYPES:
                    if dists[a]:
                        avg_dists[a] = np.mean(sorted(dists[a])[:3])  # k=3
                    else:
                        avg_dists[a] = np.inf
                
                # Nearest class
                finite = {k: v for k, v in avg_dists.items() if np.isfinite(v)}
                nearest = min(finite, key=finite.get) if finite else 'unknown'
                
                rows.append({
                    'subject': os.path.basename(subj_path),
                    **{f'{a}_dist': avg_dists[a] for a in ACCENT_TYPES},
                    'nearest': nearest
                })
                
            except Exception as e:
                print(f"  Error on {subj_path}: {e}")
                rows.append({
                    'subject': os.path.basename(subj_path),
                    **{f'{a}_dist': np.inf for a in ACCENT_TYPES},
                    'nearest': 'error'
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(f'{output_dir}/wav2vec_{config_name}_results.csv', index=False)
        
        # Compute Tarui preservation
        tarui_rate = (df['nearest'] == 'tarui').mean()
        tarui_kansai_ratios = []
        for _, row in df.iterrows():
            t, k = row['tarui_dist'], row['kansai_dist']
            if np.isfinite(t) and np.isfinite(k) and k > 0:
                tarui_kansai_ratios.append(t / k)
        
        balance = np.mean(tarui_kansai_ratios) if tarui_kansai_ratios else np.nan
        
        results.append({
            'config': config_name,
            'method': method,
            'layer': layer,
            'tarui_rate': tarui_rate,
            'balance_ratio': balance,
            'preserved': tarui_rate > 0.15
        })
        
        print(f"  {config_name}: Tarui={tarui_rate*100:.1f}%, Balance={balance:.2f}")
    
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(f'{output_dir}/wav2vec_enhanced_summary.csv', index=False)
    print(f"\nSaved: {output_dir}/wav2vec_enhanced_summary.csv")
    
    return df_summary


def run_unsupervised_analysis(subject_files, extractor, output_dir, k_values=[2, 3, 4, 5, 6]):
    """
    Run unsupervised clustering analysis.
    
    Computes subject-by-subject distance matrix and performs
    hierarchical clustering with silhouette analysis.
    """
    try:
        from src.unsupervised import (
            compute_distance_matrix, 
            hierarchical_clustering,
            analyze_silhouette,
            compute_mds,
            find_medoids,
            interpret_clusters,
            save_results
        )
    except ImportError as e:
        print(f"[unsupervised] Import failed: {e}")
        return None
    
    print("[unsupervised] Computing distance matrix...")
    
    # Distance function
    def dtw_distance(z1, z2):
        return series_distance(z1, z2, alpha=0.7, band_ratio=0.15)
    
    D, valid_files = compute_distance_matrix(extractor, subject_files, dtw_distance)
    
    print(f"[unsupervised] Matrix size: {D.shape[0]} x {D.shape[0]}")
    
    # Silhouette analysis
    print("[unsupervised] Running silhouette analysis...")
    silhouette_scores = analyze_silhouette(D, k_values)
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    
    print(f"[unsupervised] Silhouette scores:")
    for k, score in sorted(silhouette_scores.items()):
        marker = " ← optimal" if k == optimal_k else ""
        print(f"  k={k}: {score:.3f}{marker}")
    
    # Clustering with k=4 (interpretable) and optimal k
    results = {}
    for k in [4, optimal_k]:
        if k in results:
            continue
        
        labels = hierarchical_clustering(D, method='ward', k=k)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = sorted(counts, reverse=True)
        
        results[k] = {
            'labels': labels,
            'cluster_sizes': cluster_sizes,
            'silhouette': silhouette_scores.get(k, -1)
        }
        
        print(f"\n[unsupervised] k={k} cluster sizes: {cluster_sizes}")
        print(f"  Interpretation: {interpret_clusters(labels, cluster_sizes)}")
    
    # MDS projection
    print("[unsupervised] Computing MDS projection...")
    mds_coords = compute_mds(D)
    
    # Save results
    np.save(f'{output_dir}/unsup_D_total.npy', D)
    
    # Subject assignments (k=4)
    labels_k4 = results[4]['labels']
    df_subjects = pd.DataFrame({
        'file': [os.path.basename(f) for f in valid_files],
        'cluster': labels_k4,
        'mds_x': mds_coords[:, 0],
        'mds_y': mds_coords[:, 1]
    })
    df_subjects.to_csv(f'{output_dir}/unsup_subjects.csv', index=False)
    
    # Summary
    summary = {
        'n_subjects': len(valid_files),
        'silhouette_scores': {f'k{k}': v for k, v in silhouette_scores.items()},
        'optimal_k': optimal_k,
        'k4_cluster_sizes': results[4]['cluster_sizes'],
        'k4_dominant_pct': results[4]['cluster_sizes'][0] / len(valid_files) * 100
    }
    
    import json
    with open(f'{output_dir}/unsup_silhouette.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved unsupervised results to {output_dir}/")
    
    return summary


def compute_tarui_preservation(df, method_suffix):
    """Compute Tarui preservation metrics."""
    def get_nearest(row):
        dists = {a: row.get(f'{a}{method_suffix}', np.inf) for a in ACCENT_TYPES}
        finite = {k: v for k, v in dists.items() if np.isfinite(v)}
        if not finite:
            return 'unknown'
        return min(finite, key=finite.get)
    
    df = df.copy()
    df['nearest'] = df.apply(get_nearest, axis=1)
    tarui_rate = (df['nearest'] == 'tarui').mean()
    
    tarui_kansai_ratio = []
    for _, row in df.iterrows():
        t = row.get(f'tarui{method_suffix}', np.inf)
        k = row.get(f'kansai{method_suffix}', np.inf)
        if np.isfinite(t) and np.isfinite(k) and k > 0:
            tarui_kansai_ratio.append(t / k)
    
    return {
        'method': method_suffix.strip('_'),
        'tarui_classification_rate': tarui_rate,
        'mean_tarui_kansai_ratio': np.mean(tarui_kansai_ratio) if tarui_kansai_ratio else np.nan,
        'tarui_preserved': tarui_rate > 0.15
    }


def main():
    parser = argparse.ArgumentParser(description='Pitch Accent Distance Pipeline')
    parser.add_argument('--subject_dir', required=True, help='Directory containing subject audio files')
    parser.add_argument('--reference_dir', required=True, help='Directory containing reference audio files')
    parser.add_argument('--excel_path', default='data/accent_types.xlsx', help='Path to Excel file with patterns')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--config', default='configs/default.yaml', help='Configuration file')
    parser.add_argument('--skip_baselines', action='store_true', help='Skip baseline evaluation')
    parser.add_argument('--wav2vec_enhanced', action='store_true', help='Run enhanced wav2vec baselines (3a/3b/3c)')
    parser.add_argument('--unsupervised', action='store_true', help='Run unsupervised clustering analysis')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    set_seed(config.get('seed', SEED))
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List files
    refs = list_reference_files(args.reference_dir)
    subjects = list_subject_files(args.subject_dir)
    
    print(f'Found {sum(len(v) for v in refs.values())} reference files across 4 classes.')
    print(f'Found {len(subjects)} subject files.')
    
    if not subjects or not any(refs.values()):
        print("[fatal] missing subjects or references")
        return
    
    # Initialize
    extractor = TrackExtractor()
    classifier = AccentClassifier(extractor, refs, k_neighbors=3)
    
    # Virtual references from Excel
    if os.path.exists(args.excel_path):
        register_virtual_from_excel(classifier, extractor, subjects, args.excel_path)
    
    # ========== DTW Classification ==========
    print("\n[1/4] Running proposed method (DTW)...")
    
    # Preliminary pass to calibrate thresholds
    abs_tmp, rel_tmp = [], []
    for path in subjects:
        z, vr, dist = classifier.distance_vector(path)
        if z is None:
            continue
        vs = sorted(dist.items(), key=lambda x: x[1])
        if len(vs) >= 2 and math.isfinite(vs[0][1]) and math.isfinite(vs[1][1]):
            abs_tmp.append(vs[1][1] - vs[0][1])
            rel_tmp.append((vs[1][1] - vs[0][1]) / max(1e-9, vs[0][1]))
    
    amb_abs, amb_rel = calibrate_thresholds(abs_tmp, rel_tmp)
    print(f"[thresholds] amb_abs={amb_abs:.4f}, amb_rel={amb_rel:.4f}")
    
    # Full classification
    df_dtw = run_dtw_classification(subjects, classifier, amb_abs, amb_rel)
    df_dtw['amb_abs'] = amb_abs
    df_dtw['amb_rel'] = amb_rel
    df_dtw.to_csv(f'{args.output_dir}/supervised_results_dtw.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: {args.output_dir}/supervised_results_dtw.csv")
    
    # ========== Baselines ==========
    df_baseline = None
    if not args.skip_baselines:
        print("\n[2/4] Running baselines (Mean F0, Histogram EMD, wav2vec)...")
        df_baseline = run_baselines(subjects, refs, extractor)
        df_baseline.to_csv(f'{args.output_dir}/baseline_results.csv', index=False, encoding='utf-8-sig')
        print(f"Saved: {args.output_dir}/baseline_results.csv")
    
    # ========== Enhanced wav2vec Baselines ==========
    df_wav2vec_enhanced = None
    if args.wav2vec_enhanced:
        print("\n[2b/4] Running enhanced wav2vec baselines (3a/3b/3c)...")
        df_wav2vec_enhanced = run_wav2vec_enhanced(subjects, refs, args.output_dir)
    
    # ========== Unsupervised Analysis ==========
    if args.unsupervised:
        print("\n[3/4] Running unsupervised clustering analysis...")
        run_unsupervised_analysis(subjects, extractor, args.output_dir)
    
    # ========== Comparison Table ==========
    print("\n[4/4] Generating comparison table...")
    results = []
    
    # DTW
    df_dtw_renamed = df_dtw.copy()
    for a in ACCENT_TYPES:
        if f'{a}_distance' in df_dtw_renamed.columns:
            df_dtw_renamed[f'{a}_dtw'] = df_dtw_renamed[f'{a}_distance']
    results.append(compute_tarui_preservation(df_dtw_renamed, '_dtw'))
    
    # Baselines
    if df_baseline is not None:
        for suffix in ['_meanf0', '_histemd', '_wav2vec']:
            if f'tokyo{suffix}' in df_baseline.columns:
                results.append(compute_tarui_preservation(df_baseline, suffix))
    
    # Enhanced wav2vec (load from CSV if available)
    if df_wav2vec_enhanced is not None:
        for _, row in df_wav2vec_enhanced.iterrows():
            results.append({
                'method': f"wav2vec_{row['config']}",
                'tarui_classification_rate': row['tarui_rate'],
                'mean_tarui_kansai_ratio': row['balance_ratio'],
                'tarui_preserved': row['preserved']
            })
    
    df_comp = pd.DataFrame(results)
    df_comp.to_csv(f'{args.output_dir}/comparison_table.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print("=" * 70)
    
    # Highlight key finding
    dtw_rate = df_comp[df_comp['method'] == 'dtw']['tarui_classification_rate'].values
    if len(dtw_rate) > 0:
        print(f"\n★ DTW (proposed): {dtw_rate[0]*100:.1f}% Tarui preservation")
        
        best_baseline = df_comp[df_comp['method'] != 'dtw']['tarui_classification_rate'].max()
        if not np.isnan(best_baseline):
            improvement = dtw_rate[0] / max(best_baseline, 0.001)
            print(f"★ Improvement over best baseline: {improvement:.1f}×")
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()