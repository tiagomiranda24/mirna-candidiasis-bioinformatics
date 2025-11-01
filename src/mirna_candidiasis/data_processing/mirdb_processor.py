"""
miRDB Database Processor

This module provides utilities for processing and analyzing data from the miRDB database.
miRDB is a database for miRNA target prediction and functional annotations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set


class MiRDBProcessor:
    """
    Processor for miRDB database data.
    
    Handles loading, processing, and extracting features from miRDB data,
    including miRNA-target interactions and prediction scores.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the MiRDBProcessor.
        
        Args:
            data_path: Path to the miRDB data file (optional)
        """
        self.data_path = data_path
        self.data = None
        self.mirna_targets = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load miRDB data from a file.
        
        Args:
            filepath: Path to the miRDB data file
            
        Returns:
            DataFrame containing the miRDB data
        """
        try:
            self.data = pd.read_csv(filepath, sep='\t', header=None,
                                   names=['miRNA', 'Target_Gene', 'Score'])
            return self.data
        except Exception as e:
            raise ValueError(f"Error loading miRDB data: {e}")
    
    def filter_by_score(self, min_score: float = 80.0) -> pd.DataFrame:
        """
        Filter miRNA-target interactions by prediction score.
        
        Args:
            min_score: Minimum prediction score threshold
            
        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data[self.data['Score'] >= min_score]
    
    def get_mirna_targets(self, mirna: str) -> List[str]:
        """
        Get all target genes for a specific miRNA.
        
        Args:
            mirna: miRNA identifier
            
        Returns:
            List of target gene names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        targets = self.data[self.data['miRNA'] == mirna]['Target_Gene'].tolist()
        return targets
    
    def get_target_mirnas(self, target: str) -> List[str]:
        """
        Get all miRNAs targeting a specific gene.
        
        Args:
            target: Target gene identifier
            
        Returns:
            List of miRNA names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        mirnas = self.data[self.data['Target_Gene'] == target]['miRNA'].tolist()
        return mirnas
    
    def build_mirna_target_dict(self, min_score: float = 80.0) -> Dict[str, Set[str]]:
        """
        Build a dictionary mapping miRNAs to their target genes.
        
        Args:
            min_score: Minimum prediction score threshold
            
        Returns:
            Dictionary mapping miRNA names to sets of target genes
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        filtered_data = self.filter_by_score(min_score)
        
        mirna_targets = {}
        for _, row in filtered_data.iterrows():
            mirna = row['miRNA']
            target = row['Target_Gene']
            
            if mirna not in mirna_targets:
                mirna_targets[mirna] = set()
            mirna_targets[mirna].add(target)
        
        self.mirna_targets = mirna_targets
        return mirna_targets
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """
        Get summary statistics of the miRDB data.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        stats = {
            'total_interactions': len(self.data),
            'unique_mirnas': self.data['miRNA'].nunique(),
            'unique_targets': self.data['Target_Gene'].nunique(),
            'mean_score': self.data['Score'].mean(),
            'median_score': self.data['Score'].median(),
            'min_score': self.data['Score'].min(),
            'max_score': self.data['Score'].max(),
        }
        
        return stats
