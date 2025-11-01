"""
Feature engineering module for miRNA-disease association prediction.

This module provides utilities for computing similarity metrics and statistical tests,
including Jaccard index and hypergeometric tests.
"""

from .jaccard_similarity import JaccardSimilarity
from .hypergeometric_test import HypergeometricTest
from .feature_extractor import FeatureExtractor

__all__ = [
    "JaccardSimilarity",
    "HypergeometricTest",
    "FeatureExtractor",
]
