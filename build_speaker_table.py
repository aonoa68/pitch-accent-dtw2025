#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Join speaker distance vectors with de-identified metadata and compute
continuous axis coordinates for stratified analysis.

Continuous axis measure:
    s_KT = (d_K - d_Tokyo) / (d_K + d_Tokyo)
           negative = Kansai pole, 0 = balanced, positive = Tokyo pole
    s_KTarui = (d_K - d_Tarui) / (d_K + d_Tarui)
           negative = Kansai, positive = Tarui

Inputs (expected in outputs/):
    - supervised_results_with_excel.csv (released here)
    - speaker_metadata_deid.csv (NOT included in this public
      repository because per-speaker hometown / school fields exceed the
      aggregate-distribution scope of the IRB consent; will be released
      under restricted access on request (see README).
      An aggregate summary by cohort is provided in
      outputs/speaker_metadata_aggregate.csv for sanity-checking the
      paper's tables.)

Writes:
    outputs/speaker_analysis_table.csv (when the per-speaker metadata
    is available)
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "outputs"

# Load supervised distances (with Excel virtual references)
dist = pd.read_csv(OUT / "supervised_results_with_excel.csv",
                   encoding="utf-8-sig")

# Build subject_id consistent with metadata file
def to_id(fname):
    name = re.sub(r"\.[Ww][Aa][Vv]$", "", fname)
    parts = name.split("_")
    out = []
    for p in parts:
        if re.search(r"[ぁ-んァ-ヶ一-龯]", p):
            break
        out.append(p)
    return "_".join(out)


dist["subject_id"] = dist["subject"].apply(to_id)

# Compute continuous coordinates
def sym_idx(a, b):
    """Return (a-b)/(a+b) where finite, else NaN."""
    valid = np.isfinite(a) & np.isfinite(b) & ((a + b) > 0)
    out = np.full_like(a, np.nan, dtype=float)
    out[valid] = (a[valid] - b[valid]) / (a[valid] + b[valid])
    return out


d_K = dist["kansai_distance"].astype(float).values
d_To = dist["tokyo_distance"].astype(float).values
d_Tar = dist["tarui_distance"].astype(float).values
d_Kago = dist["kagoshima_distance"].astype(float).values

dist["s_kansai_tokyo"] = sym_idx(d_K, d_To)
dist["s_kansai_tarui"] = sym_idx(d_K, d_Tar)
dist["balance_r_TK"] = np.where(np.isfinite(d_K) & np.isfinite(d_Tar) & (d_K > 0),
                                d_Tar / d_K, np.nan)

# Load metadata (not included in this public repository;
# see module docstring).
meta_path = OUT / "speaker_metadata_deid.csv"
if not meta_path.exists():
    print(f"[note] {meta_path.name} not present in this distribution.")
    print("       Per-speaker de-identified metadata will be released with the")
    print("       on request. See outputs/speaker_metadata_aggregate.csv")
    print("       for cohort-level aggregates that match the paper's tables.")
    sys.exit(0)

meta = pd.read_csv(meta_path, encoding="utf-8-sig")
print(f"distances: {len(dist)}, metadata: {len(meta)}")

# Join
joined = dist.merge(meta, on="subject_id", how="left")
print(f"joined: {len(joined)}")
missing = joined["age"].isna().sum()
print(f"  speakers without metadata: {missing}")

# Save key columns first, then everything
key_cols = ["subject_id", "closest_accent", "tokyo_distance", "kansai_distance",
            "tarui_distance", "kagoshima_distance", "s_kansai_tokyo",
            "s_kansai_tarui", "balance_r_TK", "age", "cohort", "sex",
            "birth_year", "hometown", "education", "out_residency_1",
            "father_origin", "mother_origin",
            "dial_self_kansai", "dial_self_kanto", "dial_self_local",
            "audio_aff_kansai", "audio_aff_tokyo", "audio_aff_tarui",
            "audio_aff_kagoshima"]
key_cols = [c for c in key_cols if c in joined.columns]
other_cols = [c for c in joined.columns if c not in key_cols]
joined = joined[key_cols + other_cols]

joined.to_csv(OUT / "speaker_analysis_table.csv",
              index=False, encoding="utf-8-sig")
print(f"\nWritten: speaker_analysis_table.csv")
print(f"  rows={len(joined)}, cols={len(joined.columns)}")

# Quick summary
print("\n--- s_kansai_tokyo by cohort ---")
print(joined.groupby("cohort")["s_kansai_tokyo"].describe().round(3))

print("\n--- s_kansai_tarui by cohort ---")
print(joined.groupby("cohort")["s_kansai_tarui"].describe().round(3))

print("\n--- s_kansai_tarui by sex ---")
print(joined.groupby("sex")["s_kansai_tarui"].describe().round(3))

print("\n--- closest_accent x cohort ---")
print(pd.crosstab(joined["closest_accent"], joined["cohort"], margins=True))
