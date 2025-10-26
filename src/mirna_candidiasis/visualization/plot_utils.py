"""
General Plot Utilities

This module provides general-purpose plotting utilities for the
miRNA-disease association prediction framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Any


class PlotUtils:
    """
    General utilities for creating plots and visualizations.
    """
    
    @staticmethod
    def set_style(style: str = 'whitegrid', context: str = 'notebook'):
        """
        Set the plotting style.
        
        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        """
        sns.set_style(style)
        sns.set_context(context)
    
    @staticmethod
    def create_figure(
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure with specified size and title.
        
        Args:
            figsize: Figure size (width, height)
            title: Figure title (optional)
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig, ax
    
    @staticmethod
    def save_figure(
        fig: plt.Figure,
        filepath: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ):
        """
        Save a figure to file.
        
        Args:
            fig: Figure to save
            filepath: Output file path
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
    
    @staticmethod
    def plot_distribution(
        data: np.ndarray,
        title: str = 'Distribution',
        xlabel: str = 'Value',
        ylabel: str = 'Frequency',
        bins: int = 30,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot distribution histogram.
        
        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of histogram bins
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_boxplot(
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: str = 'Box Plot',
        figsize: Tuple[int, int] = (10, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a box plot.
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis (optional)
            y: Column name for y-axis (optional)
            title: Plot title
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        sns.boxplot(data=data, x=x, y=y, ax=ax)
        
        if x:
            plt.xticks(rotation=45, ha='right')
        
        return fig, ax
    
    @staticmethod
    def plot_heatmap(
        data: np.ndarray,
        title: str = 'Heatmap',
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xticklabels: Optional[List[str]] = None,
        yticklabels: Optional[List[str]] = None,
        cmap: str = 'viridis',
        figsize: Tuple[int, int] = (10, 8),
        annot: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a heatmap.
        
        Args:
            data: 2D array to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            xticklabels: Labels for x-axis ticks
            yticklabels: Labels for y-axis ticks
            cmap: Color map
            figsize: Figure size
            annot: Whether to annotate cells with values
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            annot=annot,
            fmt='.2f' if annot else None,
            cbar_kws={'label': 'Value'}
        )
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        return fig, ax
    
    @staticmethod
    def plot_scatter(
        x: np.ndarray,
        y: np.ndarray,
        title: str = 'Scatter Plot',
        xlabel: str = 'X',
        ylabel: str = 'Y',
        figsize: Tuple[int, int] = (10, 6),
        alpha: float = 0.6
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a scatter plot.
        
        Args:
            x: X-axis data
            y: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            alpha: Point transparency
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        ax.scatter(x, y, alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
