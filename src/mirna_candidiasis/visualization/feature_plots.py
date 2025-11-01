"""
Feature Visualization Plots

This module provides visualization utilities for exploring and analyzing
features in the miRNA-disease association prediction framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from .plot_utils import PlotUtils


class FeaturePlots:
    """
    Visualization utilities for feature analysis and exploration.
    """
    
    @staticmethod
    def plot_feature_correlations(
        features: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 10),
        title: str = 'Feature Correlation Matrix'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot correlation matrix of features.
        
        Args:
            features: DataFrame containing features
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        # Calculate correlation matrix
        corr_matrix = features.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return fig, ax
    
    @staticmethod
    def plot_feature_distributions(
        features: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        cols: int = 3,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot distributions of multiple features.
        
        Args:
            features: DataFrame containing features
            feature_names: List of feature names to plot (all numeric if None)
            cols: Number of columns in subplot grid
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes array)
        """
        if feature_names is None:
            feature_names = features.select_dtypes(include=[np.number]).columns.tolist()
        
        n_features = len(feature_names)
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(feature_names):
            if feature in features.columns:
                axes[i].hist(features[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(feature)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        return fig, axes
    
    @staticmethod
    def plot_feature_importance(
        feature_importance: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        title: str = 'Feature Importance'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot feature importance scores.
        
        Args:
            feature_importance: DataFrame with 'Feature' and 'Importance' columns
            top_n: Number of top features to display
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['Importance'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.grid(True, alpha=0.3, axis='x')
        
        return fig, ax
    
    @staticmethod
    def plot_jaccard_similarity_heatmap(
        similarity_matrix: np.ndarray,
        labels: List[str],
        figsize: Tuple[int, int] = (12, 10),
        title: str = 'Jaccard Similarity Heatmap'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Jaccard similarity matrix as a heatmap.
        
        Args:
            similarity_matrix: Similarity matrix
            labels: Labels for rows and columns
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.plot_heatmap(
            similarity_matrix,
            title=title,
            xticklabels=labels,
            yticklabels=labels,
            cmap='YlOrRd',
            figsize=figsize,
            annot=False
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return fig, ax
    
    @staticmethod
    def plot_pvalue_distribution(
        pvalues: np.ndarray,
        significance_threshold: float = 0.05,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'P-value Distribution'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot distribution of p-values.
        
        Args:
            pvalues: Array of p-values
            significance_threshold: Threshold for significance
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        ax.hist(pvalues, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(
            significance_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'α = {significance_threshold}'
        )
        ax.set_xlabel('P-value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text showing number of significant results
        n_significant = np.sum(pvalues < significance_threshold)
        n_total = len(pvalues)
        ax.text(
            0.95, 0.95,
            f'Significant: {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)',
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        return fig, ax
    
    @staticmethod
    def plot_enrichment_analysis(
        enrichment_results: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        title: str = 'Enrichment Analysis'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot enrichment analysis results.
        
        Args:
            enrichment_results: DataFrame with enrichment results
            top_n: Number of top results to display
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        # Get top N results sorted by p-value
        top_results = enrichment_results.head(top_n).copy()
        top_results['-log10(p)'] = -np.log10(top_results['pvalue'] + 1e-300)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_results))
        colors = ['red' if sig else 'gray' for sig in top_results['significant']]
        
        ax.barh(y_pos, top_results['-log10(p)'], alpha=0.8, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_results['identifier'])
        ax.invert_yaxis()
        ax.set_xlabel('-log10(p-value)')
        ax.set_ylabel('Category')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance threshold line
        threshold_line = -np.log10(0.05)
        ax.axvline(threshold_line, color='blue', linestyle='--', linewidth=1, label='p=0.05')
        ax.legend()
        
        return fig, ax
