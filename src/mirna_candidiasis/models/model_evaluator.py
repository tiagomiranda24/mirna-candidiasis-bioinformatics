"""
Model Evaluator

This module provides utilities for evaluating machine learning models
for miRNA-disease association prediction, including metrics calculation
and performance visualization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional, Tuple, Any


class ModelEvaluator:
    """
    Evaluator for miRNA-disease association prediction models.
    
    Provides comprehensive evaluation metrics and comparison utilities.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.results = {}
        
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = 'model'
    ) -> Dict[str, float]:
        """
        Evaluate model predictions with various metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                if y_prob.ndim > 1:
                    # For binary classification, use positive class probabilities
                    y_prob_positive = y_prob[:, 1]
                else:
                    y_prob_positive = y_prob
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob_positive)
                metrics['average_precision'] = average_precision_score(y_true, y_prob_positive)
            except Exception as e:
                print(f"Warning: Could not compute probability-based metrics: {e}")
        
        self.results[model_name] = metrics
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate a text classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes (optional)
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def compare_models(self, metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            metric: Metric to compare models by
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_predictions() first.")
        
        comparison = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Sort by the specified metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    def calculate_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate specificity (true negative rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] < 2 or cm.shape[1] < 2:
            return 0.0
        
        tn = cm[0, 0]
        fp = cm[0, 1]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return specificity
    
    def calculate_mcc(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Matthews Correlation Coefficient.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            MCC score
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] < 2 or cm.shape[1] < 2:
            return 0.0
        
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        mcc = numerator / denominator if denominator > 0 else 0.0
        
        return mcc
    
    def evaluate_cross_validation(
        self,
        cv_scores: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate cross-validation results.
        
        Args:
            cv_scores: List of cross-validation scores
            
        Returns:
            Dictionary with CV statistics
        """
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'median_score': np.median(cv_scores)
        }
    
    def get_metrics_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get a comprehensive summary of all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            DataFrame with all metrics
        """
        metrics = self.evaluate_predictions(y_true, y_pred, y_prob, 'summary')
        
        # Add additional metrics
        metrics['specificity'] = self.calculate_specificity(y_true, y_pred)
        metrics['mcc'] = self.calculate_mcc(y_true, y_pred)
        
        df = pd.DataFrame([metrics])
        df = df.T
        df.columns = ['Score']
        df['Metric'] = df.index
        df = df[['Metric', 'Score']]
        
        return df
