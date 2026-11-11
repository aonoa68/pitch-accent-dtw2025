# Speaker-normalized Acoustic Distance for Japanese Pitch Accent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository provides code and materials for the manuscript:

> **Speaker-normalized Acoustic Distance for Japanese Pitch Accent: Design Principles and Evaluation**

## Overview

Cross-speaker comparison of Japanese pitch accents is notoriously unstable due to individual variation in F0 range, speech rate, and phonation. This work designs a **measurement instrument** that positions Japanese speakers along documented pitch-accent gradients in a stable, interpretable, and linguistically anchored way. The axis is anchored by virtual reference contours synthesised from independent phonological descriptions of four canonical accent varieties (Tokyo, Kansai, Tarui, Kagoshima); the distance space is evaluated by whether it places 68 Kashiwabara speakers in the intermediate Tarui region that an independent dialectological description predicts.

### Key Contributions

1. **Normalization pipeline**: Semitone transform + MAD-based z-scoring that preserves phonological contrasts
2. **Symbolic-to-acoustic mapping**: L/H/R descriptions of canonical varieties are converted to virtual reference contours, anchoring the axis externally to the recorded corpus
3. **Evaluation framework**: Based on gradient preservation (intermediate-region placement rate, balance ratio, symmetric index) rather than discrete classification accuracy

### Main Results (Table 1 of the paper)

Three indicators on the Kansai–Tokyo segment of the externally anchored axis: intermediate-region placement rate (Inter.%), balance ratio r = mean(d_T/d_K), and symmetric index s = mean((d_K − d_T)/(d_K + d_T)).

| Method | Inter.% | r | s | Pattern |
| --- | --- | --- | --- | --- |
| Mean F0 | 1.5% | 1.42 | −0.05 | Gradient collapse |
| Histogram EMD | 85.3% | 0.83 | +0.12 | Gradient over-flattening |
| wav2vec 2.0 (3a: Mean Pool, L24) | 0% | 4.46 | −0.61 | Pole collapse |
| wav2vec 2.0 (3b: DTW, L24) | 0% | 1.55 | −0.21 | Pole collapse |
| wav2vec 2.0 (3c: DTW, L9) | 22.1% | 0.99 | +0.01 | Off-axis variation |
| **DTW (ours)** | **57.4%** | **1.01** | **−0.01** | **Gradient preserved** |

Only the proposed instrument simultaneously places a majority of speakers in the predicted intermediate region and keeps the axis symmetric (r ≈ 1, s ≈ 0). The contrast with wav2vec 3c is best read as a difference in **which axis is being measured** (the SSL instrument has an internal axis, but it is off-axis relative to the linguistically defined Kansai–Tokyo gradient), not as a difference in accuracy along a shared axis.

### Computational Cost

| Method | Time (ms) | Relative | Notes |
| --- | --- | --- | --- |
| Mean F0 | 0.04 | 1× | Trivial; no temporal info |
| Histogram EMD | 0.20 | 5× | Distribution only |
| DTW (static z) | 0.26 | 7× | Sakoe–Chiba band |
| DTW (full) | 0.99 | 25× | z(t) + Δz(t), α=0.7 |
| wav2vec 2.0 | ~450 | ~11000× | GPU recommended |

DTW-based distance remains **CPU-feasible** (<1ms per pair with Numba acceleration), processing the full dataset (68 speakers × 25 phrases × 4 references) in under 30 seconds.

## Installation

```bash
git clone https://github.com/aonoa68/pitch-accent-dtw2025.git
cd pitch-accent-dtw2025
pip install -r requirements.txt
```

### Optional dependencies

For GPU acceleration and wav2vec baseline:

```bash
pip install torch transformers
```

For faster DTW computation:

```bash
pip install numba fastdtw
```

## Quick Start

### Local Usage

```python
from src.extractor import TrackExtractor
from src.classifier import AccentClassifier
from src.dtw_distance import series_distance

# Initialize
extractor = TrackExtractor()
classifier = AccentClassifier(extractor, reference_files)

# Extract and classify
z, voiced_ratio = extractor.extract("path/to/audio.wav")
distances = classifier.distance_vector("path/to/audio.wav")
```

### Running the Full Pipeline

```bash
python run_pipeline.py \
    --subject_dir /path/to/subjects \
    --reference_dir /path/to/references \
    --excel_path data/accent_types.xlsx \
    --output_dir outputs/
```

### Running Benchmarks

To reproduce the computational cost measurements from Appendix D:

```bash
python benchmark_timing.py
```

This will output timing results for all distance metrics and scaling analysis.

### Generating Visualizations

To generate figures including the failure mode visualization (Appendix C):

```bash
python -m src.visualization
```

Output figures will be saved to `outputs/figures/`.

## Repository Structure

```
pitch-accent-dtw2025/
├── README.md
├── LICENSE
├── requirements.txt
├── run_pipeline.py                       # Main entry point
├── benchmark_timing.py                   # Computational cost measurement
├── sensitivity_sweep.py                  # Table 2: band/alpha sweep (fixed thresholds)
├── sensitivity_sweep_percfg.py           # Table 2 variant: per-config thresholds (reference)
├── cluster_comparison.py                 # Ward vs average vs PAM vs Ward-on-MDS
├── build_speaker_table.py                # Join distances with metadata (Appendix A.1)
├── stratified_analysis.py                # Appendix A.1: cohort/sex/affinity analysis
├── src/
│   ├── __init__.py
│   ├── extractor.py                      # F0 extraction & normalization
│   ├── dtw_distance.py                   # DTW-based distance computation
│   ├── baselines.py                      # Baseline methods (Mean F0, EMD, wav2vec)
│   ├── classifier.py                     # Two-stage accent classifier
│   ├── virtual_reference.py              # Excel-based symbolic references
│   ├── visualization.py                  # Figure generation
│   ├── unsupervised.py                   # Unsupervised clustering helpers
│   └── wav2vec_enhanced.py               # wav2vec 3-variant baselines (3a/3b/3c)
├── configs/
│   └── default.yaml                      # Default parameters
├── data/
│   ├── accent_types_20251027.xlsx        # Symbolic accent patterns (L/H/R)
│   └── data_README.md                    # Data access instructions
└── outputs/
    ├── baseline_results.csv              # Baseline distances
    ├── supervised_results_dtw.csv        # DTW distances (no Excel refs)
    ├── supervised_results_with_excel.csv # DTW distances (with Excel refs)
    ├── unsup_D_total.npy                 # 68x68 FastDTW matrix
    ├── unsup_subjects.csv                # MDS coordinates + cluster labels
    ├── unsup_silhouette.json             # Silhouette by k
    ├── wav2vec_enhanced_results.csv      # wav2vec layer sweep
    ├── wav2vec_enhanced_summary.csv      # wav2vec 3a/3b/3c summary
    ├── sensitivity_results.csv           # Table 2 (fixed thresholds)
    ├── sensitivity_results_percfg.csv    # Table 2 (per-config thresholds)
    ├── cluster_comparison.json           # Ward/average/PAM silhouette + ARI
    ├── speaker_metadata_aggregate.csv    # Cohort × attribute summary (Appendix A)
    ├── strat_coverage_by_cohort.csv      # Appendix A.1: Table A2
    ├── strat_label_x_cohort.csv          # Appendix A.1: cross-tab
    ├── strat_self_x_measured.csv         # Appendix A.1: self-id × measured
    ├── strat_affinity_match.csv          # Appendix A.1: auditory affinity
    ├── strat_summary.txt                 # Appendix A.1: text summary
    ├── fig_strat_cohort_bars.png         # Appendix A.1: Figure A1 (left)
    ├── fig_strat_continuous.png          # Appendix A.1: Figure A1 (right)
    └── ...
```

### Note on per-speaker de-identified metadata

The per-speaker de-identified metadata table (`speaker_metadata_deid.csv`) is **not included** in this public repository because it contains hometown and school-name fields whose combination exceeds the aggregate-distribution scope of the IRB consent. An aggregate summary by cohort (`speaker_metadata_aggregate.csv`), which matches the paper's Appendix A and A.1 tables, is included for sanity checking. The per-speaker de-identified file is available under restricted access on reasonable request, consistent with the IRB consent terms.

## Data

### Audio Data

The audio recordings used in this study are **not publicly available** due to privacy constraints. Researchers interested in accessing the data for replication purposes should contact the authors.

**Data specifications:**
- 68 native Japanese speakers
- ~25 read sentences per speaker
- 16 kHz, 16-bit, indoor recording
- Recording location: Kashiwabara, Maibara City, Shiga Prefecture
- Four reference accent types: Tokyo, Kansai, Tarui, Kagoshima

### Symbolic Accent Patterns

The Excel file `data/accent_types.xlsx` contains L/H/R symbolic patterns for each accent type, which are used to generate virtual reference contours. This file is publicly available in this repository.

## Method Details

### F0 Extraction and Normalization

1. **VAD**: Energy-based voice activity detection (top_db=25, min_segment=120ms)
2. **F0**: YIN with pYIN fallback (fmin=50Hz, fmax=700Hz)
3. **Semitone**: `s(t) = 12 * log2(F0(t) / median(F0))`
4. **MAD-z**: `z(t) = (s(t) - median(s)) / MAD(s)`
5. **Smoothing**: Savitzky-Golay filter (window=11, order=3)

### DTW Distance

```
D = α * DTW(z) + (1-α) * DTW(Δz),  α = 0.7
```

- Sakoe-Chiba band constraint (ratio=0.15)
- Costs: substitution=1.0, insertion=deletion=1.1
- Numba JIT acceleration available

### Two-Stage Classification (auxiliary hard-label diagnostic)

- **Stage 1**: {Tokyo, Kagoshima, Kansai-block}
- **Stage 2**: Kansai-block → {Kansai, Tarui}
- Ambiguity thresholds: m_abs < 0.011, m_rel < 0.035 (predetermined floors with optional quantile adjustment)
- Unknown: fewer than two finite Stage-1 distances available

The per-speaker continuous coordinates (4-pole distance vector) are the primary output; hard labels are reported only as supplementary diagnostics.

### wav2vec 2.0 Baselines

We evaluate three configurations of wav2vec 2.0 XLSR-53:

| Variant | Description | Layer | Distance |
| --- | --- | --- | --- |
| 3a | Mean Pool (Original) | L24 (final) | Cosine |
| 3b | Frame-level DTW | L24 (final) | DTW |
| 3c | Frame-level DTW | L9 (middle) | DTW |

Layer selection for 3c follows prior findings that prosodic information is better preserved in intermediate transformer layers (Pasad et al., 2021; Chen et al., 2022).

## Reproducing Paper Results

### Main Results (Table 1)

```bash
python run_pipeline.py \
    --subject_dir data/subjects \
    --reference_dir data/references \
    --output_dir outputs/
```

### wav2vec Enhanced Baselines (Table 1, rows 3a-3c)

```bash
python -m src.wav2vec_enhanced \
    --subject_dir data/subjects \
    --reference_dir data/references \
    --output_dir outputs/
```

### Computational Cost (Appendix D, Table 5)

```bash
python benchmark_timing.py
```

### Failure Mode Visualization (Appendix C)

```bash
python -m src.visualization
```

### Parameter Sensitivity (Table 2)

```bash
python sensitivity_sweep.py \
    --subject_dir /path/to/subjects \
    --reference_dir /path/to/references \
    --excel_path data/accent_types_20251027.xlsx \
    --subject_csv outputs/supervised_results_with_excel.csv \
    --output_csv outputs/sensitivity_results.csv
```

### Clustering Method Comparison

```bash
python cluster_comparison.py    # reads outputs/unsup_D_total.npy
```

### Stratified Analysis (Appendix A.1)

```bash
# Requires the per-speaker de-identified metadata (restricted access on request).
python build_speaker_table.py
python stratified_analysis.py
```

## Citation

```bibtex
@misc{onohara2026pitchaccent,
  title={Speaker-normalized Acoustic Distance for Japanese Pitch Accent: 
         Design Principles and Evaluation},
  author={Onohara, Ayaka},
  year={2026},
  howpublished={GitHub repository, \url{https://github.com/aonoa68/pitch-accent-dtw2025}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [librosa](https://librosa.org/) for audio processing
- [Numba](https://numba.pydata.org/) for JIT acceleration
- [wav2vec 2.0 XLSR-53](https://github.com/facebookresearch/fairseq) for SSL embeddings
- Linguistic descriptions from Uwano (1999), Kibe (2010), and others

## Contact

For questions about the code or data access, please open an issue.

---

**Note**: This repository is released under the MIT license (see LICENSE). It accompanies an unpublished manuscript, which is not currently under review at any journal.