"""
miRNA-Candidiasis Bioinformatics Package

A comprehensive framework for in silico prediction of candidiasis-related microRNAs.
Includes modules for data processing, feature engineering, machine learning models,
and visualization utilities for miRNA-disease association prediction.
"""

__version__ = "0.1.0"
__author__ = "Tiago Miranda"

from . import data_processing
from . import feature_engineering
from . import models
from . import visualization

__all__ = [
    "data_processing",
    "feature_engineering",
    "models",
    "visualization",
]
