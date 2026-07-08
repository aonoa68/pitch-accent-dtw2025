#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stratified analysis for the JASA manuscript (supplementary material).

Inputs:
    speaker_analysis_table.csv  (68 speakers x distances + metadata)

Outputs:
    outputs/fig_strat_*.png
    outputs/strat_summary_*.csv

Analyses:
  1. Instrument coverage by cohort
  2. Hard-label distribution by cohort and sex
  3. Self-reported dialect vs measured closest_accent
  4. Auditory affinity vs measured closest_accent
  5. Continuous coordinate by cohort/sex (small n, exploratory)
  6. Simple chi-square tests
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact

HERE = Path(__file__).parent
FIG = HERE / "outputs"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 100})

df = pd.read_csv(HERE / "outputs" / "speaker_analysis_table.csv", encoding="utf-8-sig")
print(f"Loaded {len(df)} speakers.")

COHORT_LABELS = {1.0: "C1 (younger)", 2.0: "C2 (adult)", 3.0: "C3 (elderly)"}
df["cohort_label"] = df["cohort"].map(COHORT_LABELS)

# =====================================================================
# 1. Instrument coverage by cohort
# =====================================================================
print("\n=== 1. Instrument coverage by cohort ===")
cov = df.groupby("cohort").apply(lambda g: pd.Series({
    "n": len(g),
    "n_unknown": (g["closest_accent"] == "unknown").sum(),
    "n_ambiguous": (g["closest_accent"] == "ambiguous").sum(),
    "n_classified": ((g["closest_accent"] != "unknown") &
                     (g["closest_accent"] != "ambiguous")).sum(),
    "unknown_pct": 100 * (g["closest_accent"] == "unknown").mean(),
    "classified_pct": 100 * ((g["closest_accent"] != "unknown") &
                             (g["closest_accent"] != "ambiguous")).mean(),
})).round(1)
print(cov)
cov.to_csv(HERE / "outputs" / "strat_coverage_by_cohort.csv")

# =====================================================================
# 2. Hard-label distribution by cohort
# =====================================================================
print("\n=== 2. closest_accent x cohort ===")
ct = pd.crosstab(df["closest_accent"], df["cohort_label"], margins=True)
print(ct)
ct.to_csv(HERE / "outputs" / "strat_label_x_cohort.csv")

# Chi-square: non-unknown labels vs cohort
sub = df[df["closest_accent"] != "unknown"]
ct2 = pd.crosstab(sub["closest_accent"], sub["cohort"])
chi2, p, dof, _ = chi2_contingency(ct2)
print(f"\nχ²(closest∈{{tarui,kansai,tokyo,ambiguous}} × cohort) = {chi2:.2f}, "
      f"df={dof}, p={p:.3f}")

# Plot stacked bar
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.4))
order = ["tokyo", "kansai", "tarui", "ambiguous", "unknown"]
colors = {"tokyo": "#1f77b4", "kansai": "#ff7f0e", "tarui": "#2ca02c",
          "ambiguous": "#999999", "unknown": "#cccccc"}
data = pd.crosstab(df["cohort_label"], df["closest_accent"])
data = data.reindex(columns=order, fill_value=0)
data_pct = data.div(data.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(data_pct))
for col in order:
    ax.bar(data_pct.index, data_pct[col], bottom=bottom,
           label=col, color=colors[col], edgecolor="white")
    bottom += data_pct[col].values
ax.set_ylabel("% of cohort")
ax.set_title("Classification distribution by cohort")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(FIG / "fig_strat_cohort_bars.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/fig_strat_cohort_bars.png")

# =====================================================================
# 3. Self-reported dialect vs measured closest_accent
# =====================================================================
print("\n=== 3. self-id vs measured ===")
df["self_label"] = "other/dk"
df.loc[df["dial_self_kansai"] == 1, "self_label"] = "kansai_self"
df.loc[df["dial_self_local"] == 1, "self_label"] = "local_self"
# If both flags set, prefer local
df.loc[(df["dial_self_local"] == 1) & (df["dial_self_kansai"] == 1),
       "self_label"] = "local_self"
ct_self = pd.crosstab(df["self_label"], df["closest_accent"], margins=True)
print(ct_self)
ct_self.to_csv(HERE / "outputs" / "strat_self_x_measured.csv")

# Among classified, how does self-id map to measured?
sub_cls = df[df["closest_accent"].isin(["tarui", "kansai", "tokyo"])]
ct_self_cls = pd.crosstab(sub_cls["self_label"], sub_cls["closest_accent"])
print("\nAmong classified (non-ambig/unknown):")
print(ct_self_cls)
if ct_self_cls.size >= 4:
    chi2, p, dof, _ = chi2_contingency(ct_self_cls)
    print(f"χ²(self × classified) = {chi2:.2f}, df={dof}, p={p:.3f}")

# =====================================================================
# 4. Auditory affinity vs measured
# =====================================================================
print("\n=== 4. auditory affinity (audio_aff_*) vs measured ===")
aff_cols = ["audio_aff_kansai", "audio_aff_tokyo",
            "audio_aff_tarui", "audio_aff_kagoshima"]
aff_summary = []
for col in aff_cols:
    cls_match = col.replace("audio_aff_", "")
    n_aff = (df[col] == 1).sum()
    n_aff_and_classified_that = ((df[col] == 1) &
                                  (df["closest_accent"] == cls_match)).sum()
    n_classified_that = (df["closest_accent"] == cls_match).sum()
    aff_summary.append({
        "axis": cls_match,
        "n_with_affinity": n_aff,
        "n_classified_as": n_classified_that,
        "n_both": n_aff_and_classified_that,
        "p_classified_given_aff": (
            n_aff_and_classified_that / n_aff if n_aff else np.nan),
        "p_aff_given_classified": (
            n_aff_and_classified_that / n_classified_that
            if n_classified_that else np.nan),
    })
aff_df = pd.DataFrame(aff_summary).round(3)
print(aff_df)
aff_df.to_csv(HERE / "outputs" / "strat_affinity_match.csv", index=False)

# =====================================================================
# 5. Continuous coordinate by cohort and sex (exploratory; small n)
# =====================================================================
print("\n=== 5. continuous coordinate (small n) ===")
finite = df[df["s_kansai_tarui"].notna()].copy()
print(f"  n with finite s_kansai_tarui: {len(finite)}")
print(finite.groupby("cohort_label")["s_kansai_tarui"].describe().round(3))
print(finite.groupby("sex")["s_kansai_tarui"].describe().round(3))

# Ridge-ish stripplot
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for i, group in enumerate(["cohort_label", "sex"]):
    ax = axes[i]
    for j, (lvl, sub) in enumerate(finite.groupby(group)):
        ax.scatter(sub["s_kansai_tarui"], np.full(len(sub), j) +
                   np.random.uniform(-0.1, 0.1, len(sub)),
                   s=30, alpha=0.7,
                   label=f"{lvl} (n={len(sub)})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("s = (d_K - d_T) / (d_K + d_T)  (← Kansai / Tarui →)")
    ax.set_yticks(range(len(finite[group].unique())))
    ax.set_yticklabels(finite[group].dropna().unique())
    ax.set_title(f"by {group}")
    ax.legend(loc="best", fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(FIG / "fig_strat_continuous.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: outputs/fig_strat_continuous.png")

# =====================================================================
# 6. Self-identified Kansai → measured: a key contrast
# =====================================================================
# Speakers who say "I speak Kansai" — do they actually measure as Kansai?
print("\n=== 6. Self-identified vs measured ===")
print("Q: among self-identified Kansai (n="
      f"{(df['dial_self_kansai']==1).sum()}), how many measured as Kansai vs Tarui?")
sub_k = df[df["dial_self_kansai"] == 1]
print(sub_k["closest_accent"].value_counts())
print(f"\nQ: among self-identified local (n="
      f"{(df['dial_self_local']==1).sum()}), how many measured as Tarui vs Kansai?")
sub_l = df[df["dial_self_local"] == 1]
print(sub_l["closest_accent"].value_counts())

# Save final summary
with open(HERE / "outputs" / "strat_summary.txt", "w") as f:
    f.write("Stratified analysis summary\n")
    f.write("===========================\n\n")
    f.write(f"Total speakers: {len(df)}\n")
    f.write(f"Cohorts: {dict(df['cohort_label'].value_counts())}\n\n")
    f.write("Unknown rate by cohort:\n")
    f.write(cov[["n", "n_unknown", "unknown_pct"]].to_string())
    f.write("\n\nSelf-identification:\n")
    f.write(f"  dial_self_kansai=1: {(df['dial_self_kansai']==1).sum()}/68\n")
    f.write(f"  dial_self_local=1:  {(df['dial_self_local']==1).sum()}/68\n")
    f.write(f"  dial_self_kanto=1:  {(df['dial_self_kanto']==1).sum()}/68 "
            "(nobody identifies as Kanto/Tokyo)\n")
    f.write("\nAuditory affinity to each axis:\n")
    f.write(aff_df.to_string(index=False))

print("\nDone. See outputs/ for plots and CSVs.")
