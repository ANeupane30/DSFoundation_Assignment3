#!/usr/bin/env python3
"""
Model Training Module for Bat Seasonal Behavior Analysis
HIT140 Assessment 3 - Professional Data Analytics Pipeline

This module handles Linear Regression model training, evaluation, and seasonal
comparison analysis for bat behavioral patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import joblib
import os
from pathlib import Path

# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, max_error
)
from sklearn.feature_selection import SelectKBest, f_regression

# Statistical analysis
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""
    r2_score: float
    mse: float
    mae: float
    rmse: float
    explained_variance: float
    max_error: float
    n_samples: int
    n_features: int


@dataclass
class ModelResults:
    """Data class to store complete model results."""
    model: LinearRegression
    metrics: ModelMetrics
    feature_importance: pd.DataFrame
    predictions: np.ndarray
    residuals: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]


class BatBehaviorModeler:
    """
    Comprehensive modeling class for bat seasonal behavior analysis.
    
    Handles Linear Regression model training, evaluation, and seasonal
    comparison analysis with professional best practices.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the modeler.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        
    def prepare_modeling_data(self, df: pd.DataFrame, target_column: str,
                            feature_columns: Optional[List[str]] = None,
                            exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling by selecting features and target.
        
        Args:
            df: Input DataFrame
            target_column: Name of target/response variable
            feature_columns: List of feature columns (if None, auto-select)
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing modeling data")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Extract target variable
        y = df[target_column].copy()
        
        # Prepare features
        if feature_columns is None:
            # Auto-select numeric columns, excluding target and specified exclusions
            exclude_list = [target_column] + (exclude_columns or [])
            
            # Also exclude non-predictive columns
            auto_exclude = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['date', 'id', 'index', 'unnamed'])]
            exclude_list.extend(auto_exclude)
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_cols if col not in exclude_list]
        
        X = df[feature_columns].copy()
        
        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected in features, filling with median")
            X = X.fillna(X.median())
        
        if y.isnull().any():
            logger.warning("Missing values detected in target, dropping those rows")
            mask = y.notna()
            X = X[mask]
            y = y[mask]
        
        self.feature_names = feature_columns
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_columns)} features")
        logger.info(f"Target variable '{target_column}' range: {y.min():.2f} to {y.max():.2f}")
        
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                        max_features: int = 15, method: str = 'f_regression') -> pd.DataFrame:
        """
        Perform feature selection to identify most important predictors.
        
        Args:
            X: Feature matrix
            y: Target vector
            max_features: Maximum number of features to select
            method: Feature selection method
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Performing feature selection: {method} method, max {max_features} features")
        
        if len(X.columns) <= max_features:
            logger.info("Number of features already within limit, no selection needed")
            return X
        
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            # Correlation-based selection as fallback
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(max_features).index.tolist()
            X_selected = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features, 1):
            logger.info(f"  {i:2d}. {feature}")
        
        self.feature_names = selected_features
        return X[selected_features]
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                  stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            stratify: Whether to stratify split (for classification)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_param = None
        if stratify:
            # Create stratification bins for continuous target
            stratify_param = pd.qcut(y, q=5, labels=False, duplicates='drop')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                              scale_features: bool = True, model_name: str = "overall") -> LinearRegression:
        """
        Train a Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            scale_features: Whether to standardize features
            model_name: Name identifier for the model
            
        Returns:
            Trained LinearRegression model
        """
        logger.info(f"Training Linear Regression model: {model_name}")
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train.values
            self.scalers[model_name] = None
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Store model
        self.models[model_name] = model
        
        logger.info(f"Model {model_name} trained successfully")
        logger.info(f"  Intercept: {model.intercept_:.4f}")
        logger.info(f"  Number of coefficients: {len(model.coef_)}")
        
        return model
    
    def evaluate_model(self, model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series,
                      scaler: Optional[StandardScaler] = None, model_name: str = "model") -> ModelMetrics:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            scaler: Feature scaler (if used during training)
            model_name: Model name for logging
            
        Returns:
            ModelMetrics object with evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Scale test features if scaler was used
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test.values
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        explained_var = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        
        metrics = ModelMetrics(
            r2_score=r2,
            mse=mse,
            mae=mae,
            rmse=rmse,
            explained_variance=explained_var,
            max_error=max_err,
            n_samples=len(y_test),
            n_features=X_test.shape[1]
        )
        
        # Log results
        logger.info(f"Model {model_name} evaluation results:")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  Explained Variance: {explained_var:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model: LinearRegression, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from linear regression coefficients.
        
        Args:
            model: Trained LinearRegression model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance rankings
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Add direction indicators
        importance_df['direction'] = importance_df['coefficient'].apply(
            lambda x: '↑ Positive' if x > 0 else '↓ Negative'
        )
        
        return importance_df
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
                           scale_features: bool = True) -> Dict[str, float]:
        """
        Perform cross-validation to assess model stability.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Prepare model
        if scale_features:
            from sklearn.pipeline import Pipeline
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        else:
            model = LinearRegression()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        cv_results = {
            'mean_r2': cv_scores.mean(),
            'std_r2': cv_scores.std(),
            'min_r2': cv_scores.min(),
            'max_r2': cv_scores.max(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation results:")
        logger.info(f"  Mean R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        logger.info(f"  Range: {cv_results['min_r2']:.4f} to {cv_results['max_r2']:.4f}")
        
        return cv_results
    
    def train_seasonal_models(self, df: pd.DataFrame, target_column: str,
                            feature_columns: Optional[List[str]] = None,
                            seasons: List[str] = ['summer', 'autumn']) -> Dict[str, ModelResults]:
        """
        Train separate models for different seasons.
        
        Args:
            df: Complete dataset
            target_column: Target variable name
            feature_columns: Feature column names
            seasons: List of seasons to model
            
        Returns:
            Dictionary of seasonal model results
        """
        logger.info(f"Training seasonal models for: {seasons}")
        
        seasonal_results = {}
        
        for season in seasons:
            logger.info(f"Training model for {season} season")
            
            # Filter data for this season
            season_mask = df[f'is_{season}'] == 1
            season_data = df[season_mask].copy()
            
            if len(season_data) < 30:
                logger.warning(f"Insufficient data for {season} ({len(season_data)} samples), skipping")
                continue
            
            # Prepare data
            X, y = self.prepare_modeling_data(season_data, target_column, feature_columns,
                                            exclude_columns=[f'is_{s}' for s in ['winter', 'spring', 'summer', 'autumn']])
            
            if len(X) < 20:
                logger.warning(f"Too few samples for {season} after preparation, skipping")
                continue
            
            # Feature selection
            if len(X.columns) > 12:
                X = self.feature_selection(X, y, max_features=12)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=0.2)
            
            # Train model
            model = self.train_linear_regression(X_train, y_train, model_name=f"{season}_model")
            scaler = self.scalers[f"{season}_model"]
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, scaler, f"{season}_model")
            
            # Get feature importance
            importance = self.get_feature_importance(model, X.columns.tolist())
            
            # Make predictions on test set
            X_test_scaled = scaler.transform(X_test) if scaler else X_test.values
            predictions = model.predict(X_test_scaled)
            residuals = y_test - predictions
            
            # Store results
            seasonal_results[season] = ModelResults(
                model=model,
                metrics=metrics,
                feature_importance=importance,
                predictions=predictions,
                residuals=residuals,
                scaler=scaler,
                feature_names=X.columns.tolist()
            )
            
            logger.info(f"✅ {season.capitalize()} model completed: R² = {metrics.r2_score:.4f}")
        
        return seasonal_results
    
    def compare_seasonal_models(self, seasonal_results: Dict[str, ModelResults]) -> pd.DataFrame:
        """
        Compare performance across seasonal models.
        
        Args:
            seasonal_results: Dictionary of seasonal model results
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing seasonal model performance")
        
        comparison_data = []
        
        for season, results in seasonal_results.items():
            comparison_data.append({
                'season': season.capitalize(),
                'r2_score': results.metrics.r2_score,
                'mse': results.metrics.mse,
                'mae': results.metrics.mae,
                'rmse': results.metrics.rmse,
                'n_samples': results.metrics.n_samples,
                'n_features': results.metrics.n_features
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best performing model
        best_season = comparison_df.loc[comparison_df['r2_score'].idxmax(), 'season']
        best_r2 = comparison_df['r2_score'].max()
        
        logger.info(f"Best performing model: {best_season} (R² = {best_r2:.4f})")
        logger.info("Model comparison summary:")
        logger.info(f"\n{comparison_df.round(4)}")
        
        return comparison_df
    
    def analyze_coefficient_differences(self, seasonal_results: Dict[str, ModelResults]) -> pd.DataFrame:
        """
        Analyze differences in coefficients between seasonal models.
        
        Args:
            seasonal_results: Dictionary of seasonal model results
            
        Returns:
            DataFrame with coefficient comparison
        """
        logger.info("Analyzing coefficient differences between seasonal models")
        
        if len(seasonal_results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return pd.DataFrame()
        
        # Get common features across models
        all_features = set()
        for results in seasonal_results.values():
            all_features.update(results.feature_names)
        
        # Create coefficient comparison
        coeff_data = []
        seasons = list(seasonal_results.keys())
        
        for feature in all_features:
            row = {'feature': feature}
            
            for season in seasons:
                if feature in seasonal_results[season].feature_names:
                    feature_idx = seasonal_results[season].feature_names.index(feature)
                    coeff = seasonal_results[season].model.coef_[feature_idx]
                    row[f'{season}_coeff'] = coeff
                else:
                    row[f'{season}_coeff'] = np.nan
            
            # Calculate difference between first two seasons
            if len(seasons) >= 2:
                coeff1 = row[f'{seasons[0]}_coeff']
                coeff2 = row[f'{seasons[1]}_coeff']
                if not (pd.isna(coeff1) or pd.isna(coeff2)):
                    row['coefficient_difference'] = coeff1 - coeff2
                    row['abs_difference'] = abs(coeff1 - coeff2)
            
            coeff_data.append(row)
        
        coeff_df = pd.DataFrame(coeff_data)
        
        # Sort by absolute difference
        if 'abs_difference' in coeff_df.columns:
            coeff_df = coeff_df.sort_values('abs_difference', ascending=False)
        
        logger.info("Top coefficient differences:")
        for _, row in coeff_df.head().iterrows():
            logger.info(f"  {row['feature']}: {row.get('coefficient_difference', 'N/A'):.4f}")
        
        return coeff_df
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models and scalers to disk.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}")
        
        for model_name, model in self.models.items():
            # Save model
            model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
            joblib.dump(model, model_path)
            
            # Save scaler if exists
            if model_name in self.scalers and self.scalers[model_name] is not None:
                scaler_path = os.path.join(output_dir, f"{model_name}_scaler.joblib")
                joblib.dump(self.scalers[model_name], scaler_path)
        
        logger.info(f"✅ {len(self.models)} models saved successfully")
    
    def load_models(self, input_dir: str) -> None:
        """
        Load saved models and scalers from disk.
        
        Args:
            input_dir: Directory containing saved models
        """
        logger.info(f"Loading models from {input_dir}")
        
        model_files = [f for f in os.listdir(input_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '')
            
            # Load model
            model_path = os.path.join(input_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            
            # Load scaler if exists
            scaler_file = f"{model_name}_scaler.joblib"
            scaler_path = os.path.join(input_dir, scaler_file)
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
            else:
                self.scalers[model_name] = None
        
        logger.info(f"✅ {len(self.models)} models loaded successfully")


if __name__ == "__main__":
    # Example usage
    modeler = BatBehaviorModeler()
    logger.info("BatBehaviorModeler module loaded successfully")