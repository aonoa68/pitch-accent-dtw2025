"""
Speaker-normalized Acoustic Distance for Japanese Pitch Accent
TACL 2025
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .extractor import TrackExtractor
from .dtw_distance import series_distance, dtw_avg_cost
from .classifier import AccentClassifier
from .baselines import BaselineEvaluator

__all__ = [
    "TrackExtractor",
    "series_distance",
    "dtw_avg_cost",
    "AccentClassifier",
    "BaselineEvaluator",
]
