#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity sweep for Table 2 of the JASA manuscript.

Varies (band_ratio, alpha) and reports:
  - Classification agreement with baseline (band=0.15, alpha=0.7)
  - Ambiguous rate

Thresholds are calibrated once at the baseline configuration and reused
across all (band, alpha) settings, matching the paper's stated protocol.
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extractor import TrackExtractor
from src.classifier import (
    AccentClassifier,
    two_stage_classify,
    calibrate_thresholds,
    ACCENT_TYPES,
)
from src.virtual_reference import register_virtual_from_excel
from src.dtw_distance import series_distance


def precompute_subject_tracks(extractor, subject_files):
    """Extract once; reuse across (band, alpha) settings."""
    tracks = {}
    for i, path in enumerate(subject_files):
        if (i + 1) % 10 == 0:
            print(f"  [extract] {i+1}/{len(subject_files)}")
        z, vr = extractor.extract(path)
        tracks[path] = (z, vr)
    return tracks


def distance_to_class_with_params(subj_track, ref_tracks_for_class, alpha, band, k):
    """Re-implement classifier.distance_to_class with explicit alpha/band."""
    if not ref_tracks_for_class:
        return np.inf
    dists = []
    for _, tr, _ in ref_tracks_for_class:
        d = series_distance(subj_track, tr, alpha=alpha, band_ratio=band)
        if np.isfinite(d):
            dists.append(d)
    if not dists:
        return np.inf
    dists.sort()
    return float(np.mean(dists[: min(k, len(dists))]))


def classify_all(subject_files, subj_tracks, classifier, alpha, band, amb_abs, amb_rel):
    """Run full classification with given (alpha, band) and FIXED thresholds."""
    rows = []
    for path in subject_files:
        z, vr = subj_tracks[path]
        if z is None:
            rows.append({"subject": os.path.basename(path), "closest": "unknown"})
            continue
        dist = {
            a: distance_to_class_with_params(
                z, classifier.ref_tracks.get(a, []), alpha, band, classifier.k
            )
            for a in ACCENT_TYPES
        }
        finite = {k: v for k, v in dist.items() if np.isfinite(v)}
        if len(finite) == 0:
            lab = "unknown"
        elif len(finite) == 1:
            lab = "ambiguous"
        else:
            lab, _, _ = two_stage_classify(finite, amb_abs, amb_rel)
        rows.append({"subject": os.path.basename(path), "closest": lab,
                     **{f"d_{a}": dist[a] for a in ACCENT_TYPES}})
    return pd.DataFrame(rows)


def calibrate_at_baseline(subject_files, subj_tracks, classifier, alpha, band):
    """Calibrate thresholds at the baseline configuration."""
    abs_tmp, rel_tmp = [], []
    for path in subject_files:
        z, vr = subj_tracks[path]
        if z is None:
            continue
        dist = {
            a: distance_to_class_with_params(
                z, classifier.ref_tracks.get(a, []), alpha, band, classifier.k
            )
            for a in ACCENT_TYPES
        }
        vs = sorted(dist.items(), key=lambda x: x[1])
        if len(vs) >= 2 and math.isfinite(vs[0][1]) and math.isfinite(vs[1][1]):
            abs_tmp.append(vs[1][1] - vs[0][1])
            rel_tmp.append((vs[1][1] - vs[0][1]) / max(1e-9, vs[0][1]))
    return calibrate_thresholds(abs_tmp, rel_tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_dir", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--excel_path", required=True)
    parser.add_argument("--subject_csv", required=True,
                        help="CSV listing 68 subject filenames (column 'subject')")
    parser.add_argument("--output_csv", default="sensitivity_results.csv")
    args = parser.parse_args()

    df_subj = pd.read_csv(args.subject_csv, encoding="utf-8-sig")
    subject_basenames = df_subj["subject"].tolist()
    subject_files = [os.path.join(args.subject_dir, b) for b in subject_basenames]

    refs = {}
    for a in ACCENT_TYPES:
        cand = [os.path.join(args.reference_dir, f"{a}.WAV"),
                os.path.join(args.reference_dir, f"{a}.wav")]
        if a == "kagoshima":
            cand += [os.path.join(args.reference_dir, "none.WAV"),
                     os.path.join(args.reference_dir, "none.wav")]
        refs[a] = [p for p in cand if os.path.exists(p)]

    print(f"Subjects: {len(subject_files)}")
    for a, lst in refs.items():
        print(f"  ref {a}: {lst}")

    extractor = TrackExtractor()
    classifier = AccentClassifier(extractor, refs, k_neighbors=3)
    if os.path.exists(args.excel_path):
        register_virtual_from_excel(classifier, extractor, subject_files, args.excel_path)

    print("\nExtracting subject tracks (once)...")
    subj_tracks = precompute_subject_tracks(extractor, subject_files)

    BASELINE_BAND, BASELINE_ALPHA = 0.15, 0.7
    print(f"\nCalibrating thresholds at baseline (band={BASELINE_BAND}, alpha={BASELINE_ALPHA})...")
    amb_abs, amb_rel = calibrate_at_baseline(
        subject_files, subj_tracks, classifier, BASELINE_ALPHA, BASELINE_BAND
    )
    print(f"  thresholds: amb_abs={amb_abs:.4f}, amb_rel={amb_rel:.4f}")

    print(f"\nRunning baseline classification...")
    baseline_df = classify_all(
        subject_files, subj_tracks, classifier,
        BASELINE_ALPHA, BASELINE_BAND, amb_abs, amb_rel
    )
    baseline_labels = baseline_df.set_index("subject")["closest"]
    print(f"  baseline label counts: {baseline_labels.value_counts().to_dict()}")
    baseline_ambig = (baseline_labels == "ambiguous").sum()
    print(f"  baseline ambiguous: {baseline_ambig}/{len(baseline_labels)} "
          f"= {100*baseline_ambig/len(baseline_labels):.1f}%")

    bands = [0.12, 0.15, 0.20]
    alphas = [0.5, 0.7, 0.9]
    results = []
    for band in bands:
        for alpha in alphas:
            print(f"\nConfig band={band}, alpha={alpha}...")
            df = classify_all(
                subject_files, subj_tracks, classifier,
                alpha, band, amb_abs, amb_rel
            )
            labels = df.set_index("subject")["closest"]
            agreement = (labels == baseline_labels).mean() * 100
            ambig_count = (labels == "ambiguous").sum()
            ambig_rate = 100 * ambig_count / len(labels)
            counts = labels.value_counts().to_dict()
            results.append({
                "band": band,
                "alpha": alpha,
                "agreement_pct": round(agreement, 1),
                "ambiguous_count": int(ambig_count),
                "ambiguous_pct": round(ambig_rate, 1),
                **{f"n_{k}": v for k, v in counts.items()},
            })
            print(f"  agreement={agreement:.1f}%, ambiguous={ambig_count} ({ambig_rate:.1f}%)")

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
