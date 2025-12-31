# Data Directory

This directory contains data files for the pitch accent classification experiments.

## Available Files

### `accent_types.xlsx`

Symbolic accent patterns (L/H/R) for each accent type, used to generate virtual reference contours.

| Column | Description |
|--------|-------------|
| phrase_id | Unique identifier for each phrase |
| text | Japanese text of the phrase |
| tokyo | L/H/R pattern for Tokyo accent |
| kansai | L/H/R pattern for Kansai accent |
| tarui | L/H/R pattern for Tarui accent |
| kagoshima | L/H/R pattern for Kagoshima accent |

**Symbol meanings:**
- `L` (Low): Phrase-initial or post-accent low pitch (-1.0)
- `H` (High): Accent peak or phrase-level high pitch (+1.0)
- `R` (Rising): Transitional rise toward accent nucleus (+0.5)

## Audio Data

The audio recordings used in this study are **not publicly available** due to privacy constraints.

### Dataset Specifications

- **Speakers**: 68 native Japanese speakers
- **Location**: Kashiwabara, Maibara City, Shiga Prefecture
- **Linguistic context**: Tarui accent distribution zone (transition between Kansai and Tokyo)
- **Recording**: ~25 read sentences per speaker
- **Format**: 16 kHz, 16-bit, mono, indoor recording
- **Reference types**: Tokyo, Kansai, Tarui, Kagoshima

### Data Access

Researchers interested in accessing the audio data for replication purposes should:

1. Contact the authors via the issue tracker
2. Provide institutional affiliation and research purpose
3. Sign a data use agreement

### Privacy Considerations

All data handling complies with:
- Japanese Personal Information Protection Act
- Institutional ethics board approval
- Speaker consent agreements

## Directory Structure

```
data/
├── README.md              # This file
├── accent_types.xlsx      # Symbolic accent patterns (public)
└── audio/                 # Audio files (not included, see above)
    ├── subjects/          # Subject recordings
    └── references/        # Reference recordings
```

## Citation

If you use the symbolic accent patterns in your research, please cite:

```bibtex
@article{anonymous2025pitchaccent,
  title={Speaker-normalized Acoustic Distance for Japanese Pitch Accent: 
         Design Principles and Evaluation},
  author={Anonymous},
  journal={Transactions of the Association for Computational Linguistics},
  year={2025}
}
```

## References

- Uwano, Z. (1999). Outline of Japanese accentology. In *Handbook of Japanese Linguistics*.
- Kibe, N. (2010). *Japanese Dialectology*. Iwanami Shoten.
