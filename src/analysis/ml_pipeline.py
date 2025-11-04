"""
Machine Learning pipeline for Islamabad Smog Detection System.

This module provides comprehensive machine learning functionality including
feature engineering, data preprocessing, model training, and evaluation
for air pollution prediction and analysis.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import joblib
import json

# Machine Learning libraries
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class MLPipeline:
    """Comprehensive machine learning pipeline for air pollution prediction."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ML pipeline.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.ml_config = self.config.get_section('ml_pipeline')

        # Feature engineering parameters
        self.feature_config = self.ml_config.get('features', {})
        self.lag_features = self.feature_config.get('lag_features', [1, 3, 7, 14])
        self.rolling_windows = self.feature_config.get('rolling_windows', [7, 14, 30])
        self.temporal_features = self.feature_config.get('temporal_features', ['day_of_year', 'month', 'season'])

        # Data splitting parameters
        self.split_config = self.ml_config.get('data_split', {})
        self.train_ratio = self.split_config.get('train_ratio', 0.7)
        self.validation_ratio = self.split_config.get('validation_ratio', 0.15)
        self.test_ratio = self.split_config.get('test_ratio', 0.15)
        self.temporal_split = self.split_config.get('temporal_split', True)

        # Scaling parameters
        self.scaling_config = self.ml_config.get('scaling', {})
        self.scaling_method = self.scaling_config.get('method', 'standard')
        self.log_transform_pollutants = self.scaling_config.get('log_transform_pollutants', True)

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.ml_ready_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed')) / 'ml_ready'
        )

        # Create subdirectories
        self.training_data_path = FileUtils.ensure_directory(self.ml_ready_path / 'training_data')
        self.validation_data_path = FileUtils.ensure_directory(self.ml_ready_path / 'validation_data')
        self.models_path = FileUtils.ensure_directory(self.ml_ready_path / 'models')
        self.results_path = FileUtils.ensure_directory(self.ml_ready_path / 'results')

        # Islamabad region
        self.region = GeoUtils.create_islamabad_region(buffer_km=50)

        logger.info("ML pipeline initialized")

    def prepare_ml_dataset(self, dataset: xr.Dataset, target_var: str,
                           save_to_disk: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for machine learning.

        Args:
            dataset: Input dataset
            target_var: Target variable for prediction
            save_to_disk: Whether to save prepared dataset

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Preparing ML dataset for target variable: {target_var}")

        try:
            # Extract time series data
            if 'time' not in dataset.dims:
                logger.error("Dataset must have time dimension for ML preparation")
                return pd.DataFrame(), pd.Series()

            # Create spatially averaged data
            spatial_data = {}
            for var_name in dataset.data_vars:
                if var_name.lower() in ['lat', 'lon']:
                    continue

                data_var = dataset[var_name]
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                spatial_data[var_name] = spatial_mean.to_pandas()

            # Create DataFrame
            df = pd.DataFrame(spatial_data)

            if target_var not in df.columns:
                logger.error(f"Target variable {target_var} not found in dataset")
                return pd.DataFrame(), pd.Series()

            # Remove rows with missing target
            df = df.dropna(subset=[target_var])

            # Feature engineering
            df_features = self._engineer_features(df, target_var)

            # Prepare target variable
            target = df_features[target_var].copy()

            # Remove target from features
            features = df_features.drop(columns=[target_var])

            # Handle remaining missing values
            features = self._handle_missing_values(features)

            # Data validation
            features, target = self._validate_ml_data(features, target)

            # Save prepared dataset if requested
            if save_to_disk:
                self._save_ml_dataset(features, target, target_var)

            logger.info(f"ML dataset prepared: {features.shape[0]} samples, {features.shape[1]} features")

            return features, target

        except Exception as e:
            logger.error(f"Failed to prepare ML dataset: {e}")
            return pd.DataFrame(), pd.Series()

    def _engineer_features(self, df: pd.DataFrame, target_var: str) -> pd.DataFrame:
        """
        Engineer features for machine learning.

        Args:
            df: Input DataFrame
            target_var: Target variable name

        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()

        # Temporal features
        df_engineered = self._add_temporal_features(df_engineered)

        # Lag features for target variable
        if target_var in df_engineered.columns:
            for lag in self.lag_features:
                df_engineered[f'{target_var}_lag_{lag}'] = df_engineered[target_var].shift(lag)

        # Lag features for other variables
        for col in df_engineered.columns:
            if col != target_var and col.lower() not in ['day', 'month', 'season']:
                for lag in [1, 3, 7]:  # Fewer lags for other variables
                    df_engineered[f'{col}_lag_{lag}'] = df_engineered[col].shift(lag)

        # Rolling window features
        for col in df_engineered.columns:
            if col.lower() not in ['day', 'month', 'season']:
                for window in self.rolling_windows:
                    # Only apply rolling features to numeric columns
                    if df_engineered[col].dtype in ['float64', 'int64']:
                        # Rolling statistics
                        df_engineered[f'{col}_rolling_mean_{window}'] = df_engineered[col].rolling(window).mean()
                        df_engineered[f'{col}_rolling_std_{window}'] = df_engineered[col].rolling(window).std()
                        df_engineered[f'{col}_rolling_min_{window}'] = df_engineered[col].rolling(window).min()
                        df_engineered[f'{col}_rolling_max_{window}'] = df_engineered[col].rolling(window).max()

        # Difference features
        for col in df_engineered.columns:
            if col.lower() not in ['day', 'month', 'season'] and df_engineered[col].dtype in ['float64', 'int64']:
                df_engineered[f'{col}_diff_1'] = df_engineered[col].diff(1)
                df_engineered[f'{col}_diff_7'] = df_engineered[col].diff(7)

        # Interaction features (simple interactions)
        pollutant_vars = [col for col in df_engineered.columns
                         if col.lower() not in ['day', 'month', 'season'] and df_engineered[col].dtype in ['float64', 'int64']]

        if len(pollutant_vars) >= 2:
            # Add a few key interactions
            for i, var1 in enumerate(pollutant_vars[:3]):  # Limit to first 3 variables
                for var2 in pollutant_vars[i+1:i+2]:  # One interaction per variable
                    df_engineered[f'{var1}_x_{var2}'] = df_engineered[var1] * df_engineered[var2]

        # Remove rows with NaN values created by feature engineering
        df_engineered = df_engineered.dropna()

        return df_engineered

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to DataFrame."""
        df_temporal = df.copy()

        if df_temporal.index.dtype.kind in ['M', 'm']:  # Check if index is datetime-like
            # Basic temporal features
            df_temporal['day_of_year'] = df_temporal.index.dayofyear
            df_temporal['month'] = df_temporal.index.month
            df_temporal['day_of_week'] = df_temporal.index.dayofweek
            df_temporal['quarter'] = df_temporal.index.quarter

            # Season (Northern Hemisphere)
            def get_season(month):
                if month in [12, 1, 2]:
                    return 0  # Winter
                elif month in [3, 4, 5]:
                    return 1  # Spring
                elif month in [6, 7, 8]:
                    return 2  # Summer
                else:
                    return 3  # Fall

            df_temporal['season'] = df_temporal['month'].apply(get_season)

            # Cyclical encoding for temporal features
            df_temporal['day_of_year_sin'] = np.sin(2 * np.pi * df_temporal['day_of_year'] / 365)
            df_temporal['day_of_year_cos'] = np.cos(2 * np.pi * df_temporal['day_of_year'] / 365)
            df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
            df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)

            # Weekend indicator
            df_temporal['is_weekend'] = (df_temporal.index.dayofweek >= 5).astype(int)

        return df_temporal

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # For now, simple forward fill then backward fill
        # In practice, you might want more sophisticated imputation
        df_filled = df.fillna(method='ffill').fillna(method='bfill')

        # If still missing values, use mean imputation
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())

        return df_filled

    def _validate_ml_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate ML data for quality and consistency."""
        # Ensure features and target have same index
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        target = target.replace([np.inf, -np.inf], np.nan)

        # Drop rows with remaining NaN values
        valid_mask = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_mask]
        target = target[valid_mask]

        # Basic sanity checks
        if len(features) == 0:
            logger.warning("No valid data remaining after cleaning")
            return features, target

        if len(features.columns) == 0:
            logger.warning("No features remaining after cleaning")
            return features, target

        return features, target

    def _save_ml_dataset(self, features: pd.DataFrame, target: pd.Series, target_var: str) -> None:
        """Save prepared ML dataset."""
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ml_dataset_{target_var}_{timestamp}"

            # Save features and target separately
            features_path = self.training_data_path / f"{filename}_features.csv"
            target_path = self.training_data_path / f"{filename}_target.csv"

            features.to_csv(features_path)
            target.to_csv(target_path)

            # Save metadata
            metadata = {
                'target_variable': target_var,
                'num_samples': len(features),
                'num_features': len(features.columns),
                'feature_names': list(features.columns),
                'date_range': [str(features.index.min()), str(features.index.max())],
                'preparation_date': timestamp
            }

            metadata_path = self.training_data_path / f"{filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"ML dataset saved: {len(features)} samples, {len(features.columns)} features")

        except Exception as e:
            logger.error(f"Failed to save ML dataset: {e}")

    def split_data(self, features: pd.DataFrame, target: pd.Series,
                   validation_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets.

        Args:
            features: Feature DataFrame
            target: Target Series
            validation_size: Validation set size (overrides config if provided)

        Returns:
            Dictionary with split data
        """
        if validation_size is None:
            validation_size = self.validation_ratio

        try:
            n_samples = len(features)
            test_size = int(n_samples * self.test_ratio)
            val_size = int(n_samples * validation_size)
            train_size = n_samples - test_size - val_size

            if self.temporal_split:
                # Temporal split (no data leakage)
                # Split chronologically
                X_train = features.iloc[:train_size]
                y_train = target.iloc[:train_size]

                X_val = features.iloc[train_size:train_size + val_size]
                y_val = target.iloc[train_size:train_size + val_size]

                X_test = features.iloc[train_size + val_size:]
                y_test = target.iloc[train_size + val_size:]
            else:
                # Random split
                # First split: separate test set
                X_temp, X_test, y_temp, y_test = train_test_split(
                    features, target, test_size=self.test_ratio, random_state=42
                )

                # Second split: separate train and validation from temp
                val_ratio_adjusted = validation_size / (1 - self.test_ratio)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42
                )

            split_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'split_info': {
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test),
                    'total_size': n_samples,
                    'temporal_split': self.temporal_split
                }
            }

            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            return split_data

        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            return {}

    def preprocess_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                           X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
        """
        Preprocess and scale features.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features

        Returns:
            Tuple of (scaled_train, scaled_val, scaled_test, scaler)
        """
        try:
            # Identify numeric columns for scaling
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

            # Remove temporal cyclical features from scaling (they're already normalized)
            cyclical_features = [col for col in numeric_cols if any(cf in col for cf in ['_sin', '_cos'])]
            scale_cols = [col for col in numeric_cols if col not in cyclical_features]

            if not scale_cols:
                logger.warning("No columns found for scaling")
                return X_train, X_val, X_test, None

            # Apply log transformation to skewed pollutant data
            if self.log_transform_pollutants:
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                X_test_scaled = X_test.copy()

                for col in scale_cols:
                    if col in X_train_scaled.columns:
                        # Check if data is positively skewed
                        skewness = X_train_scaled[col].skew()
                        if skewness > 1.0 and (X_train_scaled[col] > 0).all():
                            X_train_scaled[col] = np.log1p(X_train_scaled[col])
                            X_val_scaled[col] = np.log1p(X_val_scaled[col])
                            X_test_scaled[col] = np.log1p(X_test_scaled[col])
            else:
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                X_test_scaled = X_test.copy()

            # Apply scaling
            if self.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.scaling_method}, using standard")
                scaler = StandardScaler()

            # Fit scaler on training data and transform all sets
            X_train_scaled[scale_cols] = scaler.fit_transform(X_train_scaled[scale_cols])
            X_val_scaled[scale_cols] = scaler.transform(X_val_scaled[scale_cols])
            X_test_scaled[scale_cols] = scaler.transform(X_test_scaled[scale_cols])

            logger.info(f"Features preprocessed using {self.scaling_method} scaling")

            return X_train_scaled, X_val_scaled, X_test_scaled, scaler

        except Exception as e:
            logger.error(f"Failed to preprocess features: {e}")
            return X_train, X_val, X_test, None

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train multiple models and evaluate them.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            models: List of model names to train

        Returns:
            Dictionary with trained models and their performance
        """
        if models is None:
            models = ['linear', 'ridge', 'random_forest', 'xgboost'] if XGB_AVAILABLE else ['linear', 'ridge', 'random_forest']

        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available for model training")
            return {}

        logger.info(f"Training models: {models}")

        trained_models = {}

        for model_name in models:
            try:
                logger.info(f"Training {model_name} model")

                if model_name == 'linear':
                    model = LinearRegression()
                elif model_name == 'ridge':
                    model = Ridge(alpha=1.0)
                elif model_name == 'lasso':
                    model = Lasso(alpha=1.0)
                elif model_name == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_name == 'xgboost' and XGB_AVAILABLE:
                    model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                elif model_name == 'lightgbm' and XGB_AVAILABLE:
                    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_train_pred)
                val_metrics = self._calculate_metrics(y_val, y_val_pred)

                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))

                trained_models[model_name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_importance': feature_importance,
                    'model_type': model_name
                }

                logger.info(f"{model_name} - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        return trained_models

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else np.inf
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}

    def select_best_model(self, trained_models: Dict[str, Any],
                         metric: str = 'r2') -> Tuple[str, Any]:
        """
        Select the best model based on validation metrics.

        Args:
            trained_models: Dictionary of trained models
            metric: Metric to use for selection

        Returns:
            Tuple of (best_model_name, best_model_data)
        """
        if not trained_models:
            logger.warning("No trained models available for selection")
            return None, None

        best_model_name = None
        best_score = -np.inf if metric == 'r2' else np.inf
        best_model_data = None

        for model_name, model_data in trained_models.items():
            val_metrics = model_data.get('val_metrics', {})

            if metric in val_metrics:
                score = val_metrics[metric]

                if metric == 'r2':
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_data = model_data
                else:  # For metrics where lower is better (MSE, RMSE, MAE)
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_data = model_data

        if best_model_name:
            logger.info(f"Best model selected: {best_model_name} ({metric}: {best_score:.4f})")
        else:
            logger.warning(f"Could not find best model using metric: {metric}")

        return best_model_name, best_model_data

    def save_models(self, trained_models: Dict[str, Any], target_var: str) -> None:
        """
        Save trained models to disk.

        Args:
            trained_models: Dictionary of trained models
            target_var: Target variable name
        """
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

            for model_name, model_data in trained_models.items():
                # Save model
                model_filename = f"{target_var}_{model_name}_model_{timestamp}.joblib"
                model_path = self.models_path / model_filename
                joblib.dump(model_data['model'], model_path)

                # Save metadata
                metadata = {
                    'model_name': model_name,
                    'target_variable': target_var,
                    'train_metrics': model_data['train_metrics'],
                    'val_metrics': model_data['val_metrics'],
                    'feature_importance': model_data['feature_importance'],
                    'model_type': model_data['model_type'],
                    'timestamp': timestamp
                }

                metadata_filename = f"{target_var}_{model_name}_metadata_{timestamp}.json"
                metadata_path = self.models_path / metadata_filename

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Models saved for {target_var}: {list(trained_models.keys())}")

        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def generate_predictions(self, model: Any, X_test: pd.DataFrame,
                           target_var: str, save_to_disk: bool = True) -> np.ndarray:
        """
        Generate predictions using trained model.

        Args:
            model: Trained model
            X_test: Test features
            target_var: Target variable name
            save_to_disk: Whether to save predictions

        Returns:
            Predictions array
        """
        try:
            predictions = model.predict(X_test)

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                predictions_df = pd.DataFrame({
                    'index': X_test.index,
                    'prediction': predictions
                })

                predictions_filename = f"{target_var}_predictions_{timestamp}.csv"
                predictions_path = self.results_path / predictions_filename
                predictions_df.to_csv(predictions_path, index=False)

                logger.info(f"Predictions saved to {predictions_path}")

            return predictions

        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            return np.array([])

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      target_var: str) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            target_var: Target variable name

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Generate predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            test_metrics = self._calculate_metrics(y_test, y_pred)

            # Create residuals analysis
            residuals = y_test - y_pred
            residual_analysis = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'skewness_residual': float(stats.skew(residuals)),
                'kurtosis_residual': float(stats.kurtosis(residuals))
            }

            # Prediction intervals (simplified - using residual standard error)
            residual_std = np.std(residuals)
            prediction_intervals = {
                'lower_95': y_pred - 1.96 * residual_std,
                'upper_95': y_pred + 1.96 * residual_std
            }

            evaluation_results = {
                'test_metrics': test_metrics,
                'residual_analysis': residual_analysis,
                'prediction_intervals': prediction_intervals,
                'target_variable': target_var,
                'evaluation_date': pd.Timestamp.now().isoformat()
            }

            # Save evaluation results
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            eval_filename = f"{target_var}_evaluation_{timestamp}.json"
            eval_path = self.results_path / eval_filename

            with open(eval_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)

            logger.info(f"Model evaluation saved to {eval_path}")

            return evaluation_results

        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            return {}

    def feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series,
                         method: str = 'univariate', k: int = 10) -> List[str]:
        """
        Perform feature selection.

        Args:
            X_train: Training features
            y_train: Training target
            method: Feature selection method
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("sklearn not available for feature selection")
                return list(X_train.columns)

            if method == 'univariate':
                selector = SelectKBest(score_func=f_regression, k=k)
                selector.fit(X_train, y_train)
                selected_features = X_train.columns[selector.get_support()].tolist()

            elif method == 'rfe':
                # Use Random Forest for RFE
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = RFE(estimator=estimator, n_features_to_select=k)
                selector.fit(X_train, y_train)
                selected_features = X_train.columns[selector.get_support()].tolist()

            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return list(X_train.columns)

            logger.info(f"Feature selection ({method}): selected {len(selected_features)} features")

            return selected_features

        except Exception as e:
            logger.error(f"Failed to perform feature selection: {e}")
            return list(X_train.columns)

    def run_full_pipeline(self, dataset: xr.Dataset, target_var: str,
                         models: Optional[List[str]] = None,
                         feature_selection: bool = True) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.

        Args:
            dataset: Input dataset
            target_var: Target variable
            models: List of models to train
            feature_selection: Whether to perform feature selection

        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Running full ML pipeline for target: {target_var}")

        pipeline_results = {}

        try:
            # Step 1: Prepare ML dataset
            features, target = self.prepare_ml_dataset(dataset, target_var, save_to_disk=True)

            if features.empty or target.empty:
                logger.error("Failed to prepare ML dataset")
                return pipeline_results

            # Step 2: Split data
            split_data = self.split_data(features, target)

            if not split_data:
                logger.error("Failed to split data")
                return pipeline_results

            # Step 3: Feature selection (optional)
            if feature_selection:
                selected_features = self.feature_selection(
                    split_data['X_train'], split_data['y_train'], k=20
                )

                # Update datasets with selected features
                for key in ['X_train', 'X_val', 'X_test']:
                    split_data[key] = split_data[key][selected_features]

                pipeline_results['selected_features'] = selected_features

            # Step 4: Preprocess features
            X_train_pp, X_val_pp, X_test_pp, scaler = self.preprocess_features(
                split_data['X_train'], split_data['X_val'], split_data['X_test']
            )

            # Step 5: Train models
            trained_models = self.train_models(
                X_train_pp, split_data['y_train'],
                X_val_pp, split_data['y_val'], models
            )

            if not trained_models:
                logger.error("Failed to train models")
                return pipeline_results

            # Step 6: Select best model
            best_model_name, best_model_data = self.select_best_model(trained_models)

            # Step 7: Save models
            self.save_models(trained_models, target_var)

            # Step 8: Evaluate best model on test set
            if best_model_data:
                evaluation = self.evaluate_model(
                    best_model_data['model'], X_test_pp, split_data['y_test'], target_var
                )

                # Generate predictions
                predictions = self.generate_predictions(
                    best_model_data['model'], X_test_pp, target_var, save_to_disk=True
                )

                pipeline_results.update({
                    'best_model': best_model_name,
                    'best_model_data': best_model_data,
                    'evaluation': evaluation,
                    'predictions': predictions.tolist(),
                    'all_models': trained_models,
                    'scaler': scaler,
                    'feature_names': list(X_train_pp.columns)
                })

            pipeline_results['pipeline_info'] = {
                'target_variable': target_var,
                'total_samples': len(features),
                'num_features': len(features.columns),
                'feature_selection_used': feature_selection,
                'scaling_method': self.scaling_method,
                'completion_date': pd.Timestamp.now().isoformat()
            }

            # Save complete pipeline results
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"pipeline_results_{target_var}_{timestamp}.json"
            results_path = self.results_path / results_filename

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                return obj

            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=convert_numpy)

            logger.info(f"Full ML pipeline completed for {target_var}")
            logger.info(f"Best model: {best_model_name}, Test R²: {evaluation['test_metrics']['r2']:.4f}")

        except Exception as e:
            logger.error(f"ML pipeline failed: {e}")

        return pipeline_results