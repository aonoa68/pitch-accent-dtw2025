# Speaker-normalized Acoustic Distance for Japanese Pitch Accent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository provides code and materials for the TACL 2025 paper:

> **Speaker-normalized Acoustic Distance for Japanese Pitch Accent: Design Principles and Evaluation**

## Overview

Cross-speaker comparison of Japanese pitch accents is notoriously unstable due to individual variation in F0 range, speech rate, and phonation. This work establishes **design principles for speaker-normalized pitch-accent distance**, demonstrating that interpretable distance design can substantially exceed black-box representations for prosodic typology tasks.

### Key Contributions

1. **Normalization pipeline**: Semitone transform + MAD-based z-scoring that preserves phonological contrasts
2. **Evaluation framework**: Prioritizes rank stability and intermediate-category preservation over classification accuracy
3. **Symbolic injection**: Method for injecting L/H/R accent patterns into acoustic space as virtual reference contours

### Main Results

We evaluate on Tarui preservation—a critical test for intermediate accent categories that data-driven methods tend to collapse.

| Method | Tarui Rate | Balance Ratio | Assessment |
|--------|-----------|---------------|------------|
| Mean F0 | 1.5% | 1.42 | Collapse |
| Histogram EMD | 85.3% | 0.83 | Over-assign |
| **wav2vec 2.0 variants:** | | | |
| ↳ Mean Pool (3a) | 0% | 4.46 | Collapse |
| ↳ DTW Layer 24 (3b) | 0% | 1.55 | Collapse |
| ↳ DTW Layer 9 (3c) | 22.1% | 0.99 | Preserved |
| **DTW (ours)** | **57.4%** | **1.01** | **Preserved** |

**Key finding**: Even with optimal configuration (middle-layer + frame-level DTW), wav2vec achieves only 22.1% Tarui preservation vs. our method's 57.4%—a **2.6× improvement** demonstrating the value of interpretable, phonologically grounded distance design.

### wav2vec 2.0 Analysis

We conducted comprehensive experiments to address concerns about baseline fairness:

- **Baseline 3a (Original)**: Time-averaged embeddings from final layer → Complete Tarui collapse (0%)
- **Baseline 3b (Frame-level DTW, L24)**: Preserving temporal structure with final layer → Still collapses (0%)
- **Baseline 3c (Frame-level DTW, L9)**: Middle layer following prior findings on prosodic encoding → Partial preservation (22.1%)

These results demonstrate that:
1. Time-averaging is not the sole cause of failure—final-layer embeddings lack prosodic information regardless of distance metric
2. Prosodic information *is* encoded in SSL models, but in intermediate layers
3. Even optimally configured SSL representations fall substantially short of explicit pitch contour comparison

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

### Using Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aonoa68/pitch-accent-dtw2025/blob/main/notebooks/tacl2025_demo.ipynb)

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

### Running Enhanced wav2vec Baselines

```bash
python run_pipeline.py \
    --subject_dir /path/to/subjects \
    --reference_dir /path/to/references \
    --run_wav2vec_enhanced \
    --wav2vec_layers 1,3,6,9,12,15,18,21,24 \
    --output_dir outputs/wav2vec_analysis/
```

## Repository Structure

```
pitch-accent-dtw2025/
├── README.md
├── LICENSE
├── requirements.txt
├── run_pipeline.py              # Main entry point
├── src/
│   ├── __init__.py
│   ├── extractor.py             # F0 extraction & normalization
│   ├── dtw_distance.py          # DTW-based distance computation
│   ├── baselines.py             # Baseline methods (Mean F0, EMD, wav2vec)
│   ├── wav2vec_enhanced.py      # Enhanced wav2vec baselines (NEW)
│   ├── classifier.py            # Two-stage accent classifier
│   ├── virtual_reference.py     # Excel-based symbolic references
│   ├── unsupervised.py          # Clustering analysis
│   └── visualization.py         # Figure generation
├── configs/
│   └── default.yaml             # Default parameters
├── data/
│   ├── accent_types.xlsx        # Symbolic accent patterns (L/H/R)
│   └── README.md                # Data access instructions
├── notebooks/
│   └── tacl2025_demo.ipynb      # Interactive demo
└── outputs/                     # Example outputs
    ├── supervised_results.csv
    ├── baseline_results.csv
    ├── wav2vec_enhanced_results.csv  # NEW
    └── figures/
```

## Data

### Speaker Population

The 68 speakers in this study were recorded in **Kashiwabara, Maibara City, Shiga Prefecture**—a location within the Tarui accent distribution zone at the eastern boundary of the Keihan (Kansai) accent region. This linguistic context is crucial for interpreting results:

- The speaker population is relatively **homogeneous**, sharing intermediate accentual characteristics
- Unsupervised clustering correctly identifies a single dominant cluster (not four discrete categories)
- This validates our finding that **symbolic prototype injection is necessary** to preserve intermediate accent categories

### Audio Data

The audio recordings are **not publicly available** due to privacy constraints. Researchers interested in accessing the data for replication should contact the authors.

**Specifications:**
- 68 native Japanese speakers
- ~25 read sentences per speaker
- 16 kHz, 16-bit, indoor recording
- Four reference accent types: Tokyo, Kansai, Tarui, Kagoshima

### Symbolic Accent Patterns

The Excel file `data/accent_types.xlsx` contains L/H/R symbolic patterns for each accent type, used to generate virtual reference contours. This file is publicly available.

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

### Two-Stage Classification

- **Stage 1**: {Tokyo, Kagoshima, Kansai-block}
- **Stage 2**: Kansai-block → {Kansai, Tarui}
- Ambiguity thresholds: m_abs < 0.01, m_rel < 0.03

### Enhanced wav2vec Baselines

We implement three wav2vec 2.0 configurations using XLSR-53:

```python
from src.wav2vec_enhanced import EnhancedWav2VecBaseline

baseline = EnhancedWav2VecBaseline(model_name="facebook/wav2vec2-large-xlsr-53")

# 3a: Original (time-averaged, final layer)
d_3a = baseline.compute_distance(audio1, audio2, method="mean_pool", layer=24)

# 3b: Frame-level DTW, final layer
d_3b = baseline.compute_distance(audio1, audio2, method="dtw", layer=24)

# 3c: Frame-level DTW, middle layer (recommended)
d_3c = baseline.compute_distance(audio1, audio2, method="dtw", layer=9)
```

## Reproducing Results

### Main Experiments

```bash
# 1. Baseline comparison (Table 1)
python run_pipeline.py --experiment baselines --output_dir outputs/

# 2. Enhanced wav2vec analysis
python run_pipeline.py --experiment wav2vec_enhanced --output_dir outputs/

# 3. Supervised classification with symbolic references
python run_pipeline.py --experiment supervised --excel_path data/accent_types.xlsx

# 4. Unsupervised clustering
python run_pipeline.py --experiment unsupervised --output_dir outputs/
```

### Expected Outputs

After running the full pipeline:

```
outputs/
├── baseline_results.csv           # Table 1 data
├── wav2vec_enhanced_results.csv   # Layer-wise wav2vec analysis
├── supervised_results.csv         # Classification with symbolic refs
├── unsup_subjects.csv             # Cluster assignments
├── unsup_silhouette.json          # Silhouette scores (k=2: 0.87, k=4: 0.32)
└── figures/
    ├── fig02_baseline_comparison.png
    ├── fig03_supervised_heatmap.png
    └── fig04_unsup_mds.png
```

## Citation

```bibtex
@article{anonymous2025pitchaccent,
  title={Speaker-normalized Acoustic Distance for Japanese Pitch Accent: Design Principles and Evaluation},
  author={Anonymous},
  journal={Transactions of the Association for Computational Linguistics},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [librosa](https://librosa.org/) for audio processing
- [wav2vec 2.0](https://github.com/facebookresearch/fairseq) for SSL embeddings
- [transformers](https://huggingface.co/transformers/) for model access
- Linguistic descriptions from Uwano (1999), Kibe (2010), and others

## References

- Chen, S., et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE JSTSP*.
- Pasad, A., et al. (2021). Layer-wise Analysis of a Self-supervised Speech Representation Model. *Proc. IEEE ASRU*.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning for Speech Recognition. *Proc. Interspeech*.

## Contact

For questions about the code or data access, please open an issue.

---

*Last updated: December 2024*
