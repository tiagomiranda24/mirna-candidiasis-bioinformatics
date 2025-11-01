"""
Performance Visualization Plots

This module provides visualization utilities for model performance evaluation
and comparison in the miRNA-disease association prediction framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Optional, Tuple, List, Dict
from .plot_utils import PlotUtils


class PerformancePlots:
    """
    Visualization utilities for model performance evaluation.
    """
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        label: str = 'Model',
        figsize: Tuple[int, int] = (8, 8),
        title: str = 'ROC Curve'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            label: Label for the curve
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_multiple_roc_curves(
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        figsize: Tuple[int, int] = (10, 8),
        title: str = 'ROC Curves Comparison'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple ROC curves for model comparison.
        
        Args:
            results: Dictionary mapping model names to (y_true, y_prob) tuples
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        for model_name, (y_true, y_prob) in results.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        label: str = 'Model',
        figsize: Tuple[int, int] = (8, 8),
        title: str = 'Precision-Recall Curve'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            label: Label for the curve
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Plot PR curve
        ax.plot(recall, precision, linewidth=2, label=f'{label} (AUC = {pr_auc:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 6),
        title: str = 'Confusion Matrix',
        normalize: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            figsize: Figure size
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        cm = confusion_matrix.copy()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig, ax
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6),
        title: str = 'Model Performance Comparison'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot comparison of multiple metrics across models.
        
        Args:
            metrics_df: DataFrame with models as rows and metrics as columns
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        # Prepare data for grouped bar plot
        metrics_df_plot = metrics_df.set_index('Model') if 'Model' in metrics_df.columns else metrics_df
        
        # Plot grouped bars
        metrics_df_plot.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        return fig, ax
    
    @staticmethod
    def plot_learning_curve(
        train_scores: List[float],
        val_scores: List[float],
        train_sizes: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Learning Curve'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot learning curve showing training and validation scores.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes (optional)
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        if train_sizes is None:
            train_sizes = list(range(1, len(train_scores) + 1))
        
        ax.plot(train_sizes, train_scores, 'o-', linewidth=2, label='Training Score')
        ax.plot(train_sizes, val_scores, 's-', linewidth=2, label='Validation Score')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_cross_validation_scores(
        cv_scores: Dict[str, List[float]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = 'Cross-Validation Scores'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot cross-validation scores for multiple models.
        
        Args:
            cv_scores: Dictionary mapping model names to lists of CV scores
            figsize: Figure size
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = PlotUtils.create_figure(figsize, title)
        
        # Prepare data for box plot
        data = []
        labels = []
        
        for model_name, scores in cv_scores.items():
            data.append(scores)
            labels.append(model_name)
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Cross-Validation Score')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        return fig, ax
