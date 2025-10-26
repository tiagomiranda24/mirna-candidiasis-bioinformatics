"""
Data processing module for miRNA-disease association prediction.

This module provides utilities for processing data from miRDB and CTD databases.
"""

from .mirdb_processor import MiRDBProcessor
from .ctd_processor import CTDProcessor

__all__ = [
    "MiRDBProcessor",
    "CTDProcessor",
]
