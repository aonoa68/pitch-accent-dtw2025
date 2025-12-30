# Data

## Audio Recordings (Not Public)

The audio recordings used in this study are **not publicly available** due to privacy constraints on the speakers' voices.

### Data Specifications

- **Speakers**: 68 native Japanese speakers
- **Content**: ~25 read sentences per speaker
- **Format**: WAV, 16 kHz, 16-bit mono
- **Recording**: Indoor environment
- **Accent types**: Tokyo, Kansai, Tarui, Kagoshima

### Data Access

Researchers interested in accessing the audio data for replication purposes should contact the authors. Please include:

1. Your institutional affiliation
2. Intended use of the data
3. Agreement to use the data only for academic research

## Symbolic Accent Patterns (Public)

The file `accent_types.xlsx` contains L/H/R symbolic patterns for each accent type and is publicly available.

### Excel File Structure

| Column | Content |
|--------|---------|
| A | Accent type label (e.g., "tokyo", "kansai", "tarui", "kagoshima") |
| B+ | L/H/R pattern strings (e.g., "LHH", "HLL", "LHL") |

### Symbol Definitions

| Symbol | Level | Phonological Interpretation |
|--------|-------|----------------------------|
| L | -1.0 | Phrase-initial or post-accent low |
| H | +1.0 | Accent peak or phrase-level high |
| R | +0.5 | Transitional rise, accent nucleus approach |

### Usage

```python
from src.virtual_reference import register_virtual_from_excel

# Add virtual references from Excel
register_virtual_from_excel(
    classifier=clf,
    extractor=ext,
    subject_files=subject_list,
    excel_path="data/accent_types.xlsx"
)
```

## Directory Structure

```
data/
├── README.md                 # This file
├── accent_types.xlsx         # Symbolic patterns (public)
└── (audio files)             # Not included (contact authors)
```

## References

The accent type descriptions are based on:

- Uwano, Z. (1999). Accent. In *The Handbook of Japanese Linguistics*.
- Kibe, N. (2010). *Nihongo Akusento-shi Kenkyu* [Studies in the History of Japanese Accent].
- Venditti, J.J. (2005). The J-ToBI model of Japanese intonation.
