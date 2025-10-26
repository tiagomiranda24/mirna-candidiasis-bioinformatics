"""
CTD (Comparative Toxicogenomics Database) Processor

This module provides utilities for processing and analyzing data from the CTD database.
CTD provides information about chemical-gene interactions, gene-disease associations,
and chemical-disease relationships.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set


class CTDProcessor:
    """
    Processor for CTD (Comparative Toxicogenomics Database) data.
    
    Handles loading, processing, and extracting features from CTD data,
    including gene-disease associations and chemical-disease relationships.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the CTDProcessor.
        
        Args:
            data_path: Path to the CTD data file (optional)
        """
        self.data_path = data_path
        self.data = None
        self.gene_disease_map = {}
        self.disease_gene_map = {}
        
    def load_gene_disease_data(self, filepath: str) -> pd.DataFrame:
        """
        Load CTD gene-disease association data from a file.
        
        Args:
            filepath: Path to the CTD gene-disease data file
            
        Returns:
            DataFrame containing the CTD gene-disease associations
        """
        try:
            # CTD files typically have comment lines starting with '#'
            self.data = pd.read_csv(filepath, sep='\t', comment='#')
            return self.data
        except Exception as e:
            raise ValueError(f"Error loading CTD data: {e}")
    
    def filter_by_disease(self, disease_name: str) -> pd.DataFrame:
        """
        Filter associations by disease name.
        
        Args:
            disease_name: Name of the disease to filter by
            
        Returns:
            Filtered DataFrame containing associations for the specified disease
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        # Assuming column name is 'DiseaseName' or similar
        disease_col = self._find_disease_column()
        return self.data[self.data[disease_col].str.contains(disease_name, case=False, na=False)]
    
    def filter_by_gene(self, gene_symbol: str) -> pd.DataFrame:
        """
        Filter associations by gene symbol.
        
        Args:
            gene_symbol: Gene symbol to filter by
            
        Returns:
            Filtered DataFrame containing associations for the specified gene
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        gene_col = self._find_gene_column()
        return self.data[self.data[gene_col] == gene_symbol]
    
    def get_disease_genes(self, disease_name: str) -> List[str]:
        """
        Get all genes associated with a specific disease.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            List of gene symbols associated with the disease
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        filtered_data = self.filter_by_disease(disease_name)
        gene_col = self._find_gene_column()
        return filtered_data[gene_col].unique().tolist()
    
    def get_gene_diseases(self, gene_symbol: str) -> List[str]:
        """
        Get all diseases associated with a specific gene.
        
        Args:
            gene_symbol: Gene symbol
            
        Returns:
            List of diseases associated with the gene
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        filtered_data = self.filter_by_gene(gene_symbol)
        disease_col = self._find_disease_column()
        return filtered_data[disease_col].unique().tolist()
    
    def build_gene_disease_dict(self) -> Dict[str, Set[str]]:
        """
        Build a dictionary mapping genes to their associated diseases.
        
        Returns:
            Dictionary mapping gene symbols to sets of disease names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        gene_col = self._find_gene_column()
        disease_col = self._find_disease_column()
        
        gene_disease_map = {}
        for _, row in self.data.iterrows():
            gene = row[gene_col]
            disease = row[disease_col]
            
            if pd.notna(gene) and pd.notna(disease):
                if gene not in gene_disease_map:
                    gene_disease_map[gene] = set()
                gene_disease_map[gene].add(disease)
        
        self.gene_disease_map = gene_disease_map
        return gene_disease_map
    
    def build_disease_gene_dict(self) -> Dict[str, Set[str]]:
        """
        Build a dictionary mapping diseases to their associated genes.
        
        Returns:
            Dictionary mapping disease names to sets of gene symbols
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        gene_col = self._find_gene_column()
        disease_col = self._find_disease_column()
        
        disease_gene_map = {}
        for _, row in self.data.iterrows():
            gene = row[gene_col]
            disease = row[disease_col]
            
            if pd.notna(gene) and pd.notna(disease):
                if disease not in disease_gene_map:
                    disease_gene_map[disease] = set()
                disease_gene_map[disease].add(gene)
        
        self.disease_gene_map = disease_gene_map
        return disease_gene_map
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """
        Get summary statistics of the CTD data.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_gene_disease_data() first.")
        
        gene_col = self._find_gene_column()
        disease_col = self._find_disease_column()
        
        stats = {
            'total_associations': len(self.data),
            'unique_genes': self.data[gene_col].nunique() if gene_col in self.data.columns else 0,
            'unique_diseases': self.data[disease_col].nunique() if disease_col in self.data.columns else 0,
        }
        
        return stats
    
    def _find_gene_column(self) -> str:
        """
        Find the gene symbol column in the data.
        
        Returns:
            Name of the gene column
        """
        possible_names = ['GeneSymbol', 'Gene', 'GeneID', 'Symbol']
        for name in possible_names:
            if name in self.data.columns:
                return name
        
        # Return first column if not found
        return self.data.columns[0]
    
    def _find_disease_column(self) -> str:
        """
        Find the disease name column in the data.
        
        Returns:
            Name of the disease column
        """
        possible_names = ['DiseaseName', 'Disease', 'DiseaseID', 'Phenotype']
        for name in possible_names:
            if name in self.data.columns:
                return name
        
        # Return second column if not found
        return self.data.columns[1] if len(self.data.columns) > 1 else self.data.columns[0]
