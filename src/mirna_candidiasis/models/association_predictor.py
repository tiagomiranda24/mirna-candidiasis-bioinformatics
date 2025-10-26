"""
Association Predictor

This module provides machine learning models for predicting miRNA-disease
associations using various classifiers and ensemble methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any


class AssociationPredictor:
    """
    Predictor for miRNA-disease associations.
    
    Supports multiple machine learning algorithms including Random Forest,
    Gradient Boosting, SVM, and Logistic Regression.
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize the AssociationPredictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'svm', 'logistic')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._initialize_model(model_type, **kwargs)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def _initialize_model(self, model_type: str, **kwargs) -> Any:
        """
        Initialize the machine learning model.
        
        Args:
            model_type: Type of model to initialize
            **kwargs: Model parameters
            
        Returns:
            Initialized model
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif model_type == 'svm':
            return SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                probability=kwargs.get('probability', True),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif model_type == 'logistic':
            return LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        target_column: str = 'label',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            features: Feature DataFrame
            target_column: Name of the target column
            feature_columns: List of feature column names (uses all numeric if None)
            
        Returns:
            Tuple of (X, y) arrays
        """
        if feature_columns is None:
            # Use all numeric columns except target
            feature_columns = [col for col in features.select_dtypes(include=[np.number]).columns
                             if col != target_column]
        
        self.feature_names = feature_columns
        
        X = features[feature_columns].values
        y = features[target_column].values
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale: bool = True
    ) -> 'AssociationPredictor':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target labels
            scale: Whether to scale features
            
        Returns:
            Self (for method chaining)
        """
        if scale:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Predict labels for new data.
        
        Args:
            X: Feature matrix
            scale: Whether to scale features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if scale:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            X: Feature matrix
            scale: Whether to scale features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if scale:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'accuracy',
        scale: bool = True
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            scale: Whether to scale features
            
        Returns:
            Dictionary with cross-validation results
        """
        if scale:
            X = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (for tree-based models).
        
        Returns:
            Feature importance scores or None
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        return None
    
    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance as a DataFrame.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importances = self.get_feature_importance()
        
        if importances is None or self.feature_names is None:
            return None
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        
        df = df.sort_values('Importance', ascending=False)
        
        return df
