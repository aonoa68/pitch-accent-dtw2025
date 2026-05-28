#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity sweep with PER-CONFIG threshold calibration.

For each (band, alpha), recalibrate ambiguity thresholds from that
configuration's own margin distribution, then classify.
Reports agreement-with-baseline-labels and ambiguous rate.
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


def precompute(extractor, subject_files):
    tracks = {}
    for i, p in enumerate(subject_files):
        if (i + 1) % 10 == 0:
            print(f"  [extract] {i+1}/{len(subject_files)}")
        z, vr = extractor.extract(p)
        tracks[p] = (z, vr)
    return tracks


def dist_to_class(z, refs_for_class, alpha, band, k):
    if not refs_for_class:
        return np.inf
    ds = []
    for _, tr, _ in refs_for_class:
        d = series_distance(z, tr, alpha=alpha, band_ratio=band)
        if np.isfinite(d):
            ds.append(d)
    if not ds:
        return np.inf
    ds.sort()
    return float(np.mean(ds[: min(k, len(ds))]))


def compute_dist_vectors(subject_files, subj_tracks, classifier, alpha, band):
    out = []
    for p in subject_files:
        z, vr = subj_tracks[p]
        if z is None:
            out.append({a: np.inf for a in ACCENT_TYPES})
        else:
            out.append({a: dist_to_class(z, classifier.ref_tracks.get(a, []),
                                         alpha, band, classifier.k)
                        for a in ACCENT_TYPES})
    return out


def classify(dist_vectors, amb_abs, amb_rel):
    labs = []
    for dist in dist_vectors:
        finite = {k: v for k, v in dist.items() if np.isfinite(v)}
        if len(finite) == 0:
            labs.append("unknown")
        elif len(finite) == 1:
            labs.append("ambiguous")
        else:
            lab, _, _ = two_stage_classify(finite, amb_abs, amb_rel)
            labs.append(lab)
    return labs


def calibrate_from_dists(dist_vectors):
    a, r = [], []
    for d in dist_vectors:
        vs = sorted(d.items(), key=lambda x: x[1])
        if len(vs) >= 2 and math.isfinite(vs[0][1]) and math.isfinite(vs[1][1]):
            a.append(vs[1][1] - vs[0][1])
            r.append((vs[1][1] - vs[0][1]) / max(1e-9, vs[0][1]))
    return calibrate_thresholds(a, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True)
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--excel_path", required=True)
    ap.add_argument("--subject_csv", required=True)
    ap.add_argument("--output_csv", default="sensitivity_results_percfg.csv")
    args = ap.parse_args()

    basenames = pd.read_csv(args.subject_csv, encoding="utf-8-sig")["subject"].tolist()
    subj_files = [os.path.join(args.subject_dir, b) for b in basenames]
    refs = {}
    for a in ACCENT_TYPES:
        cand = [os.path.join(args.reference_dir, f"{a}.WAV")]
        if a == "kagoshima":
            cand += [os.path.join(args.reference_dir, "none.WAV")]
        refs[a] = [p for p in cand if os.path.exists(p)]

    print(f"Subjects: {len(subj_files)}")
    ext = TrackExtractor()
    clf = AccentClassifier(ext, refs, k_neighbors=3)
    if os.path.exists(args.excel_path):
        register_virtual_from_excel(clf, ext, subj_files, args.excel_path)

    print("\nExtracting once...")
    tracks = precompute(ext, subj_files)

    BAND0, ALPHA0 = 0.15, 0.7
    print(f"\nBaseline calibration (band={BAND0}, alpha={ALPHA0})...")
    base_dists = compute_dist_vectors(subj_files, tracks, clf, ALPHA0, BAND0)
    aabs0, arel0 = calibrate_from_dists(base_dists)
    print(f"  thresholds: amb_abs={aabs0:.4f}, amb_rel={arel0:.4f}")
    base_labels = classify(base_dists, aabs0, arel0)
    base_ambig = sum(1 for l in base_labels if l == "ambiguous")
    print(f"  baseline labels: {pd.Series(base_labels).value_counts().to_dict()}")
    print(f"  baseline ambiguous: {base_ambig}/{len(base_labels)} = {100*base_ambig/len(base_labels):.1f}%")

    bands = [0.12, 0.15, 0.20]
    alphas = [0.5, 0.7, 0.9]
    rows = []
    for b in bands:
        for a in alphas:
            print(f"\nConfig band={b}, alpha={a}...")
            dvs = compute_dist_vectors(subj_files, tracks, clf, a, b)
            aabs, arel = calibrate_from_dists(dvs)
            labs = classify(dvs, aabs, arel)
            agree = sum(1 for x, y in zip(labs, base_labels) if x == y) / len(labs) * 100
            ambig = sum(1 for l in labs if l == "ambiguous")
            cnts = pd.Series(labs).value_counts().to_dict()
            rows.append({
                "band": b, "alpha": a,
                "amb_abs": round(aabs, 4), "amb_rel": round(arel, 4),
                "agreement_pct": round(agree, 1),
                "ambiguous_count": ambig,
                "ambiguous_pct": round(100*ambig/len(labs), 1),
                **{f"n_{k}": v for k, v in cnts.items()},
            })
            print(f"  thresholds={aabs:.4f},{arel:.4f}  agreement={agree:.1f}%  ambig={ambig}({100*ambig/len(labs):.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
