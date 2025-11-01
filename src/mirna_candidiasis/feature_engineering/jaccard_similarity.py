"""
Jaccard Similarity Calculator

This module provides utilities for computing Jaccard similarity coefficients
between sets, commonly used for measuring similarity between miRNA target sets
or disease-associated gene sets.
"""

import numpy as np
from typing import Set, List, Dict, Union


class JaccardSimilarity:
    """
    Calculator for Jaccard similarity coefficient between sets.
    
    The Jaccard similarity coefficient is defined as the size of the intersection
    divided by the size of the union of two sets.
    """
    
    @staticmethod
    def compute(set1: Set, set2: Set) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity coefficient (0 to 1)
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def compute_distance(set1: Set, set2: Set) -> float:
        """
        Compute Jaccard distance between two sets.
        
        Jaccard distance = 1 - Jaccard similarity
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard distance (0 to 1)
        """
        return 1.0 - JaccardSimilarity.compute(set1, set2)
    
    @staticmethod
    def compute_matrix(sets_dict: Dict[str, Set]) -> np.ndarray:
        """
        Compute pairwise Jaccard similarity matrix for multiple sets.
        
        Args:
            sets_dict: Dictionary mapping identifiers to sets
            
        Returns:
            NumPy array containing pairwise Jaccard similarities
        """
        identifiers = list(sets_dict.keys())
        n = len(identifiers)
        
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    set1 = sets_dict[identifiers[i]]
                    set2 = sets_dict[identifiers[j]]
                    similarity_matrix[i, j] = JaccardSimilarity.compute(set1, set2)
        
        return similarity_matrix
    
    @staticmethod
    def get_top_similar(
        query_set: Set,
        reference_sets: Dict[str, Set],
        top_n: int = 10
    ) -> List[tuple]:
        """
        Get top N most similar sets to a query set.
        
        Args:
            query_set: Query set to compare against
            reference_sets: Dictionary of reference sets
            top_n: Number of top similar sets to return
            
        Returns:
            List of tuples (identifier, similarity_score) sorted by similarity
        """
        similarities = []
        
        for identifier, ref_set in reference_sets.items():
            similarity = JaccardSimilarity.compute(query_set, ref_set)
            similarities.append((identifier, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    @staticmethod
    def compute_weighted(
        set1: Set,
        set2: Set,
        weights: Dict[any, float]
    ) -> float:
        """
        Compute weighted Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            weights: Dictionary mapping elements to their weights
            
        Returns:
            Weighted Jaccard similarity coefficient
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # Calculate weighted intersection
        weighted_intersection = sum(weights.get(elem, 1.0) for elem in intersection)
        
        # Calculate weighted union
        weighted_union = sum(weights.get(elem, 1.0) for elem in union)
        
        if weighted_union == 0:
            return 0.0
        
        return weighted_intersection / weighted_union
