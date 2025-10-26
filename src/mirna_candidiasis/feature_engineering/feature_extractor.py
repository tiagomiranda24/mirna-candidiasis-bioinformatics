"""
Feature Extractor

This module provides utilities for extracting and combining features for
miRNA-disease association prediction using various similarity metrics and
statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from .jaccard_similarity import JaccardSimilarity
from .hypergeometric_test import HypergeometricTest


class FeatureExtractor:
    """
    Feature extractor for miRNA-disease association prediction.
    
    Combines various similarity metrics and statistical tests to create
    feature vectors for machine learning models.
    """
    
    def __init__(self, population_size: Optional[int] = None):
        """
        Initialize the FeatureExtractor.
        
        Args:
            population_size: Total population size for hypergeometric tests (optional)
        """
        self.population_size = population_size
        self.features = None
        
    def extract_jaccard_features(
        self,
        mirna_targets: Dict[str, Set],
        disease_genes: Dict[str, Set]
    ) -> pd.DataFrame:
        """
        Extract Jaccard similarity features between miRNAs and diseases.
        
        Args:
            mirna_targets: Dictionary mapping miRNA names to target gene sets
            disease_genes: Dictionary mapping disease names to associated gene sets
            
        Returns:
            DataFrame with Jaccard similarity features
        """
        features = []
        
        for mirna, targets in mirna_targets.items():
            for disease, genes in disease_genes.items():
                jaccard_sim = JaccardSimilarity.compute(targets, genes)
                
                features.append({
                    'miRNA': mirna,
                    'Disease': disease,
                    'Jaccard_Similarity': jaccard_sim
                })
        
        return pd.DataFrame(features)
    
    def extract_hypergeometric_features(
        self,
        mirna_targets: Dict[str, Set],
        disease_genes: Dict[str, Set],
        population_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract hypergeometric test features between miRNAs and diseases.
        
        Args:
            mirna_targets: Dictionary mapping miRNA names to target gene sets
            disease_genes: Dictionary mapping disease names to associated gene sets
            population_size: Total population size (uses instance value if not provided)
            
        Returns:
            DataFrame with hypergeometric test features
        """
        if population_size is None:
            population_size = self.population_size
        
        if population_size is None:
            raise ValueError("Population size must be provided")
        
        features = []
        
        for mirna, targets in mirna_targets.items():
            for disease, genes in disease_genes.items():
                overlap, pvalue = HypergeometricTest.compute_from_sets(
                    targets, genes, population_size
                )
                
                # Calculate fold enrichment
                expected = (len(targets) * len(genes)) / population_size if population_size > 0 else 0
                fold_enrichment = overlap / expected if expected > 0 else 0
                
                features.append({
                    'miRNA': mirna,
                    'Disease': disease,
                    'Overlap': overlap,
                    'Hypergeometric_Pvalue': pvalue,
                    'Fold_Enrichment': fold_enrichment,
                    'Negative_Log_Pvalue': -np.log10(pvalue + 1e-300)  # Add small value to avoid log(0)
                })
        
        return pd.DataFrame(features)
    
    def extract_combined_features(
        self,
        mirna_targets: Dict[str, Set],
        disease_genes: Dict[str, Set],
        population_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract combined features using multiple similarity metrics.
        
        Args:
            mirna_targets: Dictionary mapping miRNA names to target gene sets
            disease_genes: Dictionary mapping disease names to associated gene sets
            population_size: Total population size (uses instance value if not provided)
            
        Returns:
            DataFrame with combined features
        """
        # Extract Jaccard features
        jaccard_features = self.extract_jaccard_features(mirna_targets, disease_genes)
        
        # Extract hypergeometric features
        if population_size is not None or self.population_size is not None:
            hyper_features = self.extract_hypergeometric_features(
                mirna_targets, disease_genes, population_size
            )
            
            # Merge features
            features = pd.merge(
                jaccard_features,
                hyper_features,
                on=['miRNA', 'Disease'],
                how='outer'
            )
        else:
            features = jaccard_features
        
        # Add additional features
        features['miRNA_Target_Count'] = features['miRNA'].apply(
            lambda x: len(mirna_targets.get(x, set()))
        )
        features['Disease_Gene_Count'] = features['Disease'].apply(
            lambda x: len(disease_genes.get(x, set()))
        )
        
        self.features = features
        return features
    
    def add_network_features(
        self,
        features: pd.DataFrame,
        mirna_similarity_matrix: Optional[np.ndarray] = None,
        disease_similarity_matrix: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Add network-based features to existing feature set.
        
        Args:
            features: Existing feature DataFrame
            mirna_similarity_matrix: Similarity matrix for miRNAs (optional)
            disease_similarity_matrix: Similarity matrix for diseases (optional)
            
        Returns:
            DataFrame with added network features
        """
        # This is a placeholder for network-based features
        # In practice, you would compute features like:
        # - miRNA-miRNA similarity scores
        # - Disease-disease similarity scores
        # - Network centrality measures
        
        return features
    
    def normalize_features(
        self,
        features: pd.DataFrame,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Normalize numeric features.
        
        Args:
            features: Feature DataFrame
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            DataFrame with normalized features
        """
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = features[col].min()
                max_val = features[col].max()
                if max_val > min_val:
                    features[col] = (features[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = features[col].mean()
                std_val = features[col].std()
                if std_val > 0:
                    features[col] = (features[col] - mean_val) / std_val
        
        return features
