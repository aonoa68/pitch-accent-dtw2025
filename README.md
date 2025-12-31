# Speaker-normalized Acoustic Distance for Japanese Pitch Accent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository provides code and materials for the TACL 2025 paper:

> **Speaker-normalized Acoustic Distance for Japanese Pitch Accent: Design Principles and Evaluation**

## Overview

Cross-speaker comparison of Japanese pitch accents is notoriously unstable due to individual variation in F0 range, speech rate, and phonation. This work establishes **design principles for speaker-normalized pitch-accent distance**, demonstrating that interpretable distance design can match or exceed black-box representations for prosodic typology tasks.

### Key Contributions

1. **Normalization pipeline**: Semitone transform + MAD-based z-scoring that preserves phonological contrasts
2. **Evaluation framework**: Prioritizes rank stability and intermediate-category preservation over classification accuracy
3. **Symbolic injection**: Method for injecting L/H/R accent patterns into acoustic space as virtual reference contours

### Main Results

| Method | Tarui Rate | Balance Ratio | Assessment |
|--------|------------|---------------|------------|
| Mean F0 | 1.5% | 1.42 | Collapse |
| Histogram EMD | 85.3% | 0.83 | Over-assign |
| wav2vec 2.0 | 19.1% | 0.94 | Partial |
| **DTW (ours)** | **57.4%** | **1.01** | **Balanced** |

### Computational Cost

| Method | Time (ms) | Notes |
|--------|-----------|-------|
| Mean F0 | 0.04 | Trivial; no temporal info |
| Histogram EMD | 0.20 | Distribution only |
| DTW (full) | 0.99 | z(t) + Δz(t), α=0.7, Numba-accelerated |
| wav2vec 2.0 | ~450 | GPU recommended |

DTW-based distance remains **CPU-feasible** (<1ms per pair with Numba), processing the full dataset (68 speakers × 25 phrases × 12 references) in under 30 seconds.

## Installation

```bash
git clone https://github.com/yourrepo/pitch-accent-dtw2025.git
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourrepo/pitch-accent-dtw2025/blob/main/notebooks/tacl2025_demo.ipynb)

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
├── run_pipeline.py              # Main entry point
├── benchmark_timing.py          # Computational cost measurement
├── src/
│   ├── __init__.py
│   ├── extractor.py             # F0 extraction & normalization
│   ├── dtw_distance.py          # DTW-based distance computation
│   ├── baselines.py             # Baseline methods (Mean F0, EMD, wav2vec)
│   ├── classifier.py            # Two-stage accent classifier
│   ├── virtual_reference.py     # Excel-based symbolic references
│   └── visualization.py         # Figure generation (NEW)
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
    └── figures/
        ├── fig_failure_mode.png      # Appendix C
        └── fig_baseline_comparison.png
```

## Data

### Audio Data

The audio recordings used in this study are **not publicly available** due to privacy constraints. Researchers interested in accessing the data for replication purposes should contact the authors.

**Data specifications:**
- 68 native Japanese speakers
- ~25 read sentences per speaker
- 16 kHz, 16-bit, indoor recording
- Four accent types: Tokyo, Kansai, Tarui, Kagoshima

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

### Two-Stage Classification

- **Stage 1**: {Tokyo, Kagoshima, Kansai-block}
- **Stage 2**: Kansai-block → {Kansai, Tarui}
- Ambiguity thresholds: m_abs < 0.01, m_rel < 0.03

## Reproducing Paper Results

### Main Results (Table 1)

```bash
python run_pipeline.py \
    --subject_dir data/subjects \
    --reference_dir data/references \
    --output_dir outputs/
```

### Computational Cost (Appendix D)

```bash
python benchmark_timing.py
```

### Failure Mode Visualization (Appendix C)

```bash
python -m src.visualization
```

## Citation

```bibtex
@article{anonymous2025pitchaccent,
  title={Speaker-normalized Acoustic Distance for Japanese Pitch Accent: 
         Design Principles and Evaluation},
  author={Anonymous},
  journal={Transactions of the Association for Computational Linguistics},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [librosa](https://librosa.org/) for audio processing
- [Numba](https://numba.pydata.org/) for JIT acceleration
- [wav2vec 2.0](https://github.com/facebookresearch/fairseq) for SSL embeddings
- Linguistic descriptions from Uwano (1999), Kibe (2010), and others

## Contact

For questions about the code or data access, please open an issue or contact the authors.
