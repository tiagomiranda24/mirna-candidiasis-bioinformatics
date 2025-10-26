"""
Hypergeometric Test

This module provides utilities for performing hypergeometric tests to assess
the statistical significance of overlaps between sets, commonly used in
enrichment analysis and miRNA-disease association studies.
"""

import numpy as np
from scipy.stats import hypergeom
from typing import Set, List, Dict, Tuple


class HypergeometricTest:
    """
    Calculator for hypergeometric test p-values.
    
    The hypergeometric test is used to determine if the overlap between two sets
    is statistically significant given the population size.
    """
    
    @staticmethod
    def compute_pvalue(
        overlap: int,
        set1_size: int,
        set2_size: int,
        population_size: int
    ) -> float:
        """
        Compute hypergeometric test p-value for set overlap.
        
        Args:
            overlap: Size of the intersection between the two sets
            set1_size: Size of the first set
            set2_size: Size of the second set
            population_size: Total population size
            
        Returns:
            P-value from hypergeometric test
        """
        if population_size == 0:
            return 1.0
        
        # Probability of observing at least 'overlap' successes
        # M: population size
        # n: number of success states in population (set1_size)
        # N: number of draws (set2_size)
        # k: number of observed successes (overlap)
        
        pvalue = hypergeom.sf(overlap - 1, population_size, set1_size, set2_size)
        
        return pvalue
    
    @staticmethod
    def compute_from_sets(
        set1: Set,
        set2: Set,
        population_size: int
    ) -> Tuple[int, float]:
        """
        Compute hypergeometric test for two sets.
        
        Args:
            set1: First set
            set2: Second set
            population_size: Total population size
            
        Returns:
            Tuple of (overlap_size, p_value)
        """
        overlap = len(set1.intersection(set2))
        set1_size = len(set1)
        set2_size = len(set2)
        
        pvalue = HypergeometricTest.compute_pvalue(
            overlap, set1_size, set2_size, population_size
        )
        
        return overlap, pvalue
    
    @staticmethod
    def enrichment_analysis(
        query_set: Set,
        reference_sets: Dict[str, Set],
        population_size: int,
        significance_threshold: float = 0.05
    ) -> List[Dict]:
        """
        Perform enrichment analysis for a query set against multiple reference sets.
        
        Args:
            query_set: Query set to test for enrichment
            reference_sets: Dictionary of reference sets to test against
            population_size: Total population size
            significance_threshold: P-value threshold for significance
            
        Returns:
            List of dictionaries containing enrichment results
        """
        results = []
        
        for identifier, ref_set in reference_sets.items():
            overlap = len(query_set.intersection(ref_set))
            query_size = len(query_set)
            ref_size = len(ref_set)
            
            pvalue = HypergeometricTest.compute_pvalue(
                overlap, ref_size, query_size, population_size
            )
            
            # Calculate fold enrichment
            expected = (query_size * ref_size) / population_size if population_size > 0 else 0
            fold_enrichment = overlap / expected if expected > 0 else 0
            
            result = {
                'identifier': identifier,
                'overlap': overlap,
                'query_size': query_size,
                'reference_size': ref_size,
                'pvalue': pvalue,
                'fold_enrichment': fold_enrichment,
                'significant': pvalue < significance_threshold
            }
            
            results.append(result)
        
        # Sort by p-value
        results.sort(key=lambda x: x['pvalue'])
        
        return results
    
    @staticmethod
    def bonferroni_correction(pvalues: List[float]) -> List[float]:
        """
        Apply Bonferroni correction for multiple testing.
        
        Args:
            pvalues: List of p-values
            
        Returns:
            List of Bonferroni-corrected p-values
        """
        n_tests = len(pvalues)
        if n_tests == 0:
            return []
        
        corrected_pvalues = [min(p * n_tests, 1.0) for p in pvalues]
        return corrected_pvalues
    
    @staticmethod
    def fdr_correction(pvalues: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Apply Benjamini-Hochberg FDR correction for multiple testing.
        
        Args:
            pvalues: List of p-values
            alpha: False discovery rate threshold
            
        Returns:
            List of boolean values indicating significance after FDR correction
        """
        n_tests = len(pvalues)
        if n_tests == 0:
            return []
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = np.array(pvalues)[sorted_indices]
        
        # Calculate critical values
        critical_values = [(i + 1) / n_tests * alpha for i in range(n_tests)]
        
        # Find largest i where p(i) <= critical_value(i)
        significant = [False] * n_tests
        for i in range(n_tests - 1, -1, -1):
            if sorted_pvalues[i] <= critical_values[i]:
                # All tests up to i are significant
                for j in range(i + 1):
                    original_idx = sorted_indices[j]
                    significant[original_idx] = True
                break
        
        return significant
