"""
Statistical analysis module for Islamabad Smog Detection System.

This module provides comprehensive statistical analysis capabilities including
correlation analysis, regression modeling, hypothesis testing, and spatial
statistics for air pollution data in the Islamabad region.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from scipy.spatial import distance_matrix
    from scipy.stats import pearsonr, spearmanr, kendalltau
    SPATIAL_STATS_AVAILABLE = True
except ImportError:
    SPATIAL_STATS_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for air pollution data."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize statistical analyzer.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.processed_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed')) / 'analysis_ready'
        )

        # Islamabad region
        self.region = GeoUtils.create_islamabad_region(buffer_km=50)

        logger.info("Statistical analyzer initialized")

    def perform_comprehensive_analysis(self, dataset: xr.Dataset,
                                     analysis_types: Optional[List[str]] = None,
                                     save_to_disk: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.

        Args:
            dataset: Input dataset
            analysis_types: List of analysis types to perform
            save_to_disk: Whether to save results to disk

        Returns:
            Dictionary with analysis results
        """
        if analysis_types is None:
            analysis_types = ['descriptive', 'correlation', 'regression', 'spatial']

        logger.info(f"Performing statistical analysis: {analysis_types}")

        results = {}

        for analysis_type in analysis_types:
            try:
                logger.info(f"Performing {analysis_type} analysis")

                if analysis_type == 'descriptive':
                    results['descriptive'] = self.descriptive_statistics(dataset)
                elif analysis_type == 'correlation':
                    results['correlation'] = self.correlation_analysis(dataset)
                elif analysis_type == 'regression':
                    results['regression'] = self.regression_analysis(dataset)
                elif analysis_type == 'spatial':
                    results['spatial'] = self.spatial_statistics(dataset)
                elif analysis_type == 'temporal':
                    results['temporal'] = self.temporal_statistics(dataset)
                elif analysis_type == 'multivariate':
                    results['multivariate'] = self.multivariate_analysis(dataset)
                else:
                    logger.warning(f"Unknown analysis type: {analysis_type}")

            except Exception as e:
                logger.error(f"Failed to perform {analysis_type} analysis: {e}")
                continue

        # Save results if requested
        if save_to_disk:
            self._save_analysis_results(results, dataset, analysis_types)

        return results

    def descriptive_statistics(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with descriptive statistics
        """
        stats_results = {}

        for var_name in dataset.data_vars:
            if var_name.lower() in ['lat', 'lon', 'time']:
                continue

            data_var = dataset[var_name]
            data_values = data_var.values
            valid_data = data_values[~np.isnan(data_values)]

            if len(valid_data) == 0:
                continue

            try:
                # Basic statistics
                var_stats = {
                    'count': len(valid_data),
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'std': float(np.std(valid_data)),
                    'var': float(np.var(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'range': float(np.max(valid_data) - np.min(valid_data)),
                    'q1': float(np.percentile(valid_data, 25)),
                    'q3': float(np.percentile(valid_data, 75)),
                    'iqr': float(np.percentile(valid_data, 75) - np.percentile(valid_data, 25)),
                    'skewness': float(stats.skew(valid_data)),
                    'kurtosis': float(stats.kurtosis(valid_data)),
                    'coefficient_of_variation': float(np.std(valid_data) / np.mean(valid_data)) if np.mean(valid_data) != 0 else 0
                }

                # Additional statistics for sufficient data
                if len(valid_data) >= 30:
                    # Confidence intervals
                    ci_mean = stats.t.interval(0.95, len(valid_data)-1, loc=np.mean(valid_data), scale=stats.sem(valid_data))
                    var_stats['confidence_interval_95'] = [float(ci_mean[0]), float(ci_mean[1])]

                    # Normality tests
                    shapiro_stat, shapiro_p = stats.shapiro(valid_data[:5000] if len(valid_data) > 5000 else valid_data)
                    var_stats['shapiro_wilk'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }

                    # Anderson-Darling test
                    ad_result = stats.anderson(valid_data)
                    var_stats['anderson_darling'] = {
                        'statistic': float(ad_result.statistic),
                        'critical_values': ad_result.critical_values.tolist(),
                        'significance_levels': ad_result.significance_levels.tolist()
                    }

                # Data quality metrics
                var_stats['data_quality'] = {
                    'missing_percentage': float((data_values.size - len(valid_data)) / data_values.size * 100),
                    'outlier_percentage': self._calculate_outlier_percentage(valid_data),
                    'data_range_reasonableness': self._assess_data_range(var_name, valid_data)
                }

                stats_results[var_name] = var_stats

            except Exception as e:
                logger.error(f"Failed to calculate descriptive statistics for {var_name}: {e}")
                continue

        return stats_results

    def correlation_analysis(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Perform correlation analysis between variables.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with correlation analysis results
        """
        correlation_results = {}

        # Extract pollutant variables (exclude coordinates and quality flags)
        pollutant_vars = [var for var in dataset.data_vars
                         if var.lower() not in ['lat', 'lon', 'time', 'qa_value', 'quality']]

        if len(pollutant_vars) < 2:
            logger.warning("Need at least 2 variables for correlation analysis")
            return correlation_results

        try:
            # Create spatially averaged data for each variable
            var_data = {}
            for var in pollutant_vars:
                data_var = dataset[var]
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    # Spatial averaging
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                elif 'time' in data_var.dims:
                    # Time averaging
                    spatial_mean = data_var.mean(dim='time')
                else:
                    spatial_mean = data_var

                var_data[var] = spatial_mean.to_pandas().dropna()

            # Find common time index
            common_index = None
            for var, data in var_data.items():
                if common_index is None:
                    common_index = data.index
                else:
                    common_index = common_index.intersection(data.index)

            if len(common_index) == 0:
                logger.warning("No common time index found for correlation analysis")
                return correlation_results

            # Align all data to common index
            aligned_data = {}
            for var, data in var_data.items():
                aligned_data[var] = data.reindex(common_index).dropna()

            # Create DataFrame for correlation analysis
            df = pd.DataFrame(aligned_data)

            # Calculate correlation matrices
            correlation_results['pearson'] = df.corr(method='pearson').to_dict()
            correlation_results['spearman'] = df.corr(method='spearman').to_dict()
            correlation_results['kendall'] = df.corr(method='kendall').to_dict()

            # Calculate p-values for correlations
            p_values = {}
            for var1 in pollutant_vars:
                p_values[var1] = {}
                for var2 in pollutant_vars:
                    if var1 in df.columns and var2 in df.columns:
                        try:
                            _, p_val = pearsonr(df[var1].dropna(), df[var2].dropna())
                            p_values[var1][var2] = float(p_val)
                        except:
                            p_values[var1][var2] = np.nan

            correlation_results['p_values'] = p_values

            # Significant correlations (p < 0.05)
            significant_correlations = {}
            for var1 in p_values:
                significant_correlations[var1] = {}
                for var2, p_val in p_values[var1].items():
                    if p_val < 0.05:
                        significant_correlations[var1][var2] = {
                            'correlation': correlation_results['pearson'][var1][var2],
                            'p_value': p_val,
                            'significance': 'significant'
                        }

            correlation_results['significant_correlations'] = significant_correlations

            # Partial correlations (controlling for other variables)
            if len(pollutant_vars) >= 3 and STATSMODELS_AVAILABLE:
                partial_corrs = self._calculate_partial_correlations(df, pollutant_vars)
                correlation_results['partial_correlations'] = partial_corrs

        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")

        return correlation_results

    def _calculate_partial_correlations(self, df: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Calculate partial correlations controlling for other variables."""
        partial_corrs = {}

        for i, var1 in enumerate(variables):
            partial_corrs[var1] = {}
            for j, var2 in enumerate(variables):
                if i == j:
                    partial_corrs[var1][var2] = 1.0
                    continue

                # Control variables (all except var1 and var2)
                control_vars = [v for v in variables if v not in [var1, var2]]

                if not control_vars:
                    partial_corrs[var1][var2] = df.corr().loc[var1, var2]
                    continue

                try:
                    # Calculate partial correlation using linear regression
                    X_control = df[control_vars].dropna()
                    y1 = df[var1].loc[X_control.index]
                    y2 = df[var2].loc[X_control.index]

                    # Remove effects of control variables
                    model1 = sm.OLS(y1, sm.add_constant(X_control)).fit()
                    model2 = sm.OLS(y2, sm.add_constant(X_control)).fit()

                    residuals1 = model1.resid
                    residuals2 = model2.resid

                    # Correlation of residuals is the partial correlation
                    partial_corr, _ = pearsonr(residuals1, residuals2)
                    partial_corrs[var1][var2] = float(partial_corr)

                except Exception as e:
                    logger.warning(f"Failed to calculate partial correlation for {var1}-{var2}: {e}")
                    partial_corrs[var1][var2] = np.nan

        return partial_corrs

    def regression_analysis(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Perform regression analysis.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with regression analysis results
        """
        regression_results = {}

        try:
            # Prepare data for regression
            pollutant_vars = [var for var in dataset.data_vars
                             if var.lower() not in ['lat', 'lon', 'time', 'qa_value', 'quality']]

            if len(pollutant_vars) < 2:
                logger.warning("Need at least 2 variables for regression analysis")
                return regression_results

            # Create spatially averaged time series data
            var_data = {}
            for var in pollutant_vars:
                data_var = dataset[var]
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                var_data[var] = spatial_mean.to_pandas().dropna()

            # Find common time index
            common_index = None
            for var, data in var_data.items():
                if common_index is None:
                    common_index = data.index
                else:
                    common_index = common_index.intersection(data.index)

            if len(common_index) == 0:
                logger.warning("No common time index found for regression analysis")
                return regression_results

            # Align all data
            aligned_data = {}
            for var, data in var_data.items():
                aligned_data[var] = data.reindex(common_index).dropna()

            df = pd.DataFrame(aligned_data)

            # Simple linear regressions
            regression_results['simple_linear'] = {}
            for i, target_var in enumerate(pollutant_vars):
                for j, predictor_var in enumerate(pollutant_vars):
                    if i == j:
                        continue

                    try:
                        X = df[predictor_var].values.reshape(-1, 1)
                        y = df[target_var].values

                        # Remove NaN values
                        valid_mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                        X_clean = X[valid_mask]
                        y_clean = y[valid_mask]

                        if len(X_clean) < 10:
                            continue

                        # Fit linear regression
                        model = LinearRegression()
                        model.fit(X_clean, y_clean)

                        # Calculate metrics
                        y_pred = model.predict(X_clean)
                        r2 = r2_score(y_clean, y_pred)
                        mse = mean_squared_error(y_clean, y_pred)

                        regression_results['simple_linear'][f"{target_var}_vs_{predictor_var}"] = {
                            'coefficient': float(model.coef_[0]),
                            'intercept': float(model.intercept_),
                            'r_squared': float(r2),
                            'mse': float(mse),
                            'rmse': float(np.sqrt(mse)),
                            'sample_size': len(X_clean)
                        }

                        # Statistical significance using scipy
                        if len(X_clean) >= 3:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(X_clean.flatten(), y_clean)
                            regression_results['simple_linear'][f"{target_var}_vs_{predictor_var}"].update({
                                'p_value': float(p_value),
                                'std_error': float(std_err),
                                'is_significant': p_value < 0.05
                            })

                    except Exception as e:
                        logger.warning(f"Failed to fit regression {target_var} vs {predictor_var}: {e}")
                        continue

            # Multiple regression (if enough variables)
            if len(pollutant_vars) >= 3 and SKLEARN_AVAILABLE:
                regression_results['multiple'] = {}
                for target_var in pollutant_vars:
                    try:
                        # Use other variables as predictors
                        predictor_vars = [v for v in pollutant_vars if v != target_var]
                        X = df[predictor_vars]
                        y = df[target_var]

                        # Remove rows with NaN
                        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                        X_clean = X[valid_mask]
                        y_clean = y[valid_mask]

                        if len(X_clean) < len(predictor_vars) * 2:
                            continue

                        # Fit multiple regression
                        model = LinearRegression()
                        model.fit(X_clean, y_clean)

                        y_pred = model.predict(X_clean)
                        r2 = r2_score(y_clean, y_pred)
                        mse = mean_squared_error(y_clean, y_pred)

                        # Cross-validation
                        cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='r2')

                        regression_results['multiple'][target_var] = {
                            'coefficients': {var: float(coef) for var, coef in zip(predictor_vars, model.coef_)},
                            'intercept': float(model.intercept_),
                            'r_squared': float(r2),
                            'mse': float(mse),
                            'rmse': float(np.sqrt(mse)),
                            'cv_r2_mean': float(cv_scores.mean()),
                            'cv_r2_std': float(cv_scores.std()),
                            'sample_size': len(X_clean),
                            'predictors': predictor_vars
                        }

                    except Exception as e:
                        logger.warning(f"Failed to fit multiple regression for {target_var}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Regression analysis failed: {e}")

        return regression_results

    def spatial_statistics(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Perform spatial statistics analysis.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with spatial statistics results
        """
        spatial_results = {}

        for var_name in dataset.data_vars:
            if var_name.lower() in ['lat', 'lon', 'time']:
                continue

            data_var = dataset[var_name]

            # Check if data has spatial dimensions
            if 'lat' not in data_var.dims or 'lon' not in data_var.dims:
                continue

            try:
                # Get data for a specific time if time dimension exists
                if 'time' in data_var.dims:
                    spatial_data = data_var.isel(time=0)
                else:
                    spatial_data = data_var

                data_values = spatial_data.values
                valid_data = data_values[~np.isnan(data_values)]

                if len(valid_data) == 0:
                    continue

                # Basic spatial statistics
                var_spatial_stats = {
                    'spatial_mean': float(np.mean(valid_data)),
                    'spatial_std': float(np.std(valid_data)),
                    'spatial_range': float(np.max(valid_data) - np.min(valid_data)),
                    'spatial_coefficient_of_variation': float(np.std(valid_data) / np.mean(valid_data)) if np.mean(valid_data) != 0 else 0
                }

                # Spatial autocorrelation (Moran's I - simplified implementation)
                if len(valid_data) > 10:
                    moran_i = self._calculate_simplified_morans_i(spatial_data)
                    var_spatial_stats['morans_i'] = moran_i

                # Spatial pattern analysis
                var_spatial_stats['spatial_patterns'] = self._analyze_spatial_patterns(spatial_data)

                # Hotspot analysis (simplified)
                hotspots = self._identify_hotspots(spatial_data)
                var_spatial_stats['hotspot_analysis'] = hotspots

                spatial_results[var_name] = var_spatial_stats

            except Exception as e:
                logger.error(f"Failed to calculate spatial statistics for {var_name}: {e}")
                continue

        return spatial_results

    def _calculate_simplified_morans_i(self, spatial_data: xr.DataArray) -> float:
        """Calculate simplified Moran's I spatial autocorrelation."""
        try:
            data_values = spatial_data.values
            valid_mask = ~np.isnan(data_values)

            if np.sum(valid_mask) < 10:
                return 0.0

            # Create weight matrix (simplified - using contiguity)
            rows, cols = data_values.shape
            n = np.sum(valid_mask)

            # Calculate spatial lag
            spatial_lag = np.zeros_like(data_values)
            weights = np.zeros_like(data_values)

            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if valid_mask[i, j]:
                        # Calculate average of neighbors
                        neighbors = []
                        neighbor_weights = []

                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if (0 <= ni < rows and 0 <= nj < cols and valid_mask[ni, nj]):
                                    neighbors.append(data_values[ni, nj])
                                    neighbor_weights.append(1.0)

                        if neighbors:
                            spatial_lag[i, j] = np.mean(neighbors)
                            weights[i, j] = len(neighbor_weights)

            # Calculate Moran's I
            valid_values = data_values[valid_mask]
            valid_lag = spatial_lag[valid_mask]
            valid_weights = weights[valid_mask]

            if np.sum(valid_weights) == 0:
                return 0.0

            mean_val = np.mean(valid_values)
            numerator = np.sum(valid_weights * (valid_values - mean_val) * (valid_lag - mean_val))
            denominator = np.sum(valid_weights * (valid_values - mean_val) ** 2)

            if denominator == 0:
                return 0.0

            morans_i = numerator / denominator
            return float(morans_i)

        except Exception as e:
            logger.warning(f"Failed to calculate Moran's I: {e}")
            return 0.0

    def _analyze_spatial_patterns(self, spatial_data: xr.DataArray) -> Dict[str, Any]:
        """Analyze spatial patterns in the data."""
        data_values = spatial_data.values
        valid_mask = ~np.isnan(data_values)

        if np.sum(valid_mask) == 0:
            return {}

        try:
            patterns = {}

            # Gradient analysis
            if data_values.ndim == 2:
                # Calculate gradients
                grad_y, grad_x = np.gradient(data_values)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                patterns['gradient'] = {
                    'mean_gradient': float(np.mean(grad_magnitude[valid_mask])),
                    'max_gradient': float(np.max(grad_magnitude[valid_mask])),
                    'gradient_variability': float(np.std(grad_magnitude[valid_mask]))
                }

                # Texture analysis (simplified)
                patterns['texture'] = self._calculate_texture_metrics(data_values, valid_mask)

            return patterns

        except Exception as e:
            logger.warning(f"Failed to analyze spatial patterns: {e}")
            return {}

    def _calculate_texture_metrics(self, data: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
        """Calculate texture metrics for spatial data."""
        try:
            valid_data = data[valid_mask]

            # Local variance as texture measure
            local_variance = ndimage.generic_filter(
                data, np.var, size=3, mode='constant', cval=np.nan
            )

            texture_metrics = {
                'local_variance_mean': float(np.nanmean(local_variance)),
                'local_variance_std': float(np.nanstd(local_variance)),
                'data_variance': float(np.var(valid_data))
            }

            return texture_metrics

        except Exception as e:
            logger.warning(f"Failed to calculate texture metrics: {e}")
            return {}

    def _identify_hotspots(self, spatial_data: xr.DataArray, threshold_percentile: float = 90) -> Dict[str, Any]:
        """Identify spatial hotspots."""
        try:
            data_values = spatial_data.values
            valid_mask = ~np.isnan(data_values)

            if np.sum(valid_mask) == 0:
                return {}

            # Calculate threshold
            valid_data = data_values[valid_mask]
            threshold = np.percentile(valid_data, threshold_percentile)

            # Identify hotspots
            hotspots = data_values > threshold
            hotspot_mask = hotspots & valid_mask

            # Calculate hotspot statistics
            hotspot_analysis = {
                'threshold': float(threshold),
                'num_hotspots': int(np.sum(hotspot_mask)),
                'hotspot_percentage': float(np.sum(hotspot_mask) / np.sum(valid_mask) * 100),
                'hotspot_mean': float(np.mean(data_values[hotspot_mask])) if np.sum(hotspot_mask) > 0 else 0,
                'hotspot_max': float(np.max(data_values[hotspot_mask])) if np.sum(hotspot_mask) > 0 else 0
            }

            return hotspot_analysis

        except Exception as e:
            logger.warning(f"Failed to identify hotspots: {e}")
            return {}

    def temporal_statistics(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Perform temporal statistics analysis.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with temporal statistics
        """
        temporal_results = {}

        for var_name in dataset.data_vars:
            if var_name.lower() in ['lat', 'lon']:
                continue

            data_var = dataset[var_name]

            # Check if data has time dimension
            if 'time' not in data_var.dims:
                continue

            try:
                # Create spatial mean for temporal analysis
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    temporal_data = data_var.mean(dim=['lat', 'lon'])
                else:
                    temporal_data = data_var

                ts_series = temporal_data.to_pandas().dropna()

                if len(ts_series) == 0:
                    continue

                var_temporal_stats = {
                    'start_date': str(ts_series.index.min()),
                    'end_date': str(ts_series.index.max()),
                    'duration_days': (ts_series.index.max() - ts_series.index.min()).days,
                    'num_observations': len(ts_series)
                }

                # Trend analysis
                if len(ts_series) >= 10:
                    x = np.arange(len(ts_series))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_series)
                    var_temporal_stats['trend'] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'is_significant': p_value < 0.05,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'no trend'
                    }

                # Seasonal patterns
                if len(ts_series) >= 24:  # At least 2 years of monthly data
                    seasonal_stats = {}
                    for season, season_data in ts_series.groupby(ts_series.index.season):
                        seasonal_stats[season] = {
                            'mean': float(season_data.mean()),
                            'std': float(season_data.std()),
                            'count': len(season_data)
                        }
                    var_temporal_stats['seasonal_patterns'] = seasonal_stats

                    # Monthly patterns
                    monthly_stats = {}
                    for month, month_data in ts_series.groupby(ts_series.index.month):
                        monthly_stats[month] = {
                            'mean': float(month_data.mean()),
                            'std': float(month_data.std()),
                            'count': len(month_data)
                        }
                    var_temporal_stats['monthly_patterns'] = monthly_stats

                temporal_results[var_name] = var_temporal_stats

            except Exception as e:
                logger.error(f"Failed to calculate temporal statistics for {var_name}: {e}")
                continue

        return temporal_results

    def multivariate_analysis(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Perform multivariate analysis.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with multivariate analysis results
        """
        multivariate_results = {}

        try:
            # Extract pollutant variables
            pollutant_vars = [var for var in dataset.data_vars
                             if var.lower() not in ['lat', 'lon', 'time', 'qa_value', 'quality']]

            if len(pollutant_vars) < 2:
                logger.warning("Need at least 2 variables for multivariate analysis")
                return multivariate_results

            # Create spatially and temporally averaged data
            var_data = {}
            for var in pollutant_vars:
                data_var = dataset[var]

                # Average over spatial dimensions
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                # Average over time if multiple time points
                if 'time' in spatial_mean.dims:
                    temporal_mean = spatial_mean.mean(dim='time')
                else:
                    temporal_mean = spatial_mean

                var_data[var] = float(temporal_mean.values)

            # Create DataFrame for analysis
            df = pd.DataFrame([var_data])

            # Principal Component Analysis
            if SKLEARN_AVAILABLE and len(pollutant_vars) >= 2:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler

                # Prepare data matrix (transpose to have variables as columns)
                data_matrix = np.array([var_data[var] for var in pollutant_vars]).reshape(1, -1)

                if data_matrix.size > 0:
                    # Standardize data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data_matrix)

                    # PCA
                    pca = PCA(n_components=min(len(pollutant_vars), data_matrix.shape[1]))
                    pca_result = pca.fit(scaled_data)

                    multivariate_results['pca'] = {
                        'explained_variance_ratio': pca_result.explained_variance_ratio_.tolist(),
                        'cumulative_variance_ratio': np.cumsum(pca_result.explained_variance_ratio_).tolist(),
                        'components': pca_result.components_.tolist(),
                        'n_components': pca_result.n_components_
                    }

            # Variable importance analysis
            var_importance = {}
            for var in pollutant_vars:
                # Calculate importance based on variance and range
                if var in dataset:
                    data_var = dataset[var]
                    data_values = data_var.values[~np.isnan(data_var.values)]

                    if len(data_values) > 0:
                        importance = {
                            'variance': float(np.var(data_values)),
                            'range': float(np.max(data_values) - np.min(data_values)),
                            'coefficient_of_variation': float(np.std(data_values) / np.mean(data_values)) if np.mean(data_values) != 0 else 0
                        }
                        var_importance[var] = importance

            multivariate_results['variable_importance'] = var_importance

            # Clustering tendency
            if len(pollutant_vars) >= 3:
                clustering_score = self._assess_clustering_tendency(df)
                multivariate_results['clustering_tendency'] = clustering_score

        except Exception as e:
            logger.error(f"Multivariate analysis failed: {e}")

        return multivariate_results

    def _assess_clustering_tendency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the tendency of data to form clusters."""
        try:
            # Hopkins statistic for clustering tendency
            # Simplified implementation
            n_samples, n_features = df.shape

            if n_samples < 10 or n_features < 2:
                return {'hopkins_statistic': None, 'clustering_tendency': 'insufficient_data'}

            # Generate random uniform data
            np.random.seed(42)
            random_data = np.random.uniform(
                low=df.min().min(),
                high=df.max().max(),
                size=(n_samples, n_features)
            )

            # Calculate distances
            real_distances = pdist(df.values)
            random_distances = pdist(random_data)

            # Simplified Hopkins statistic
            hopkins = np.mean(random_distances) / (np.mean(real_distances) + np.mean(random_distances))

            clustering_tendency = 'high' if hopkins > 0.75 else 'moderate' if hopkins > 0.5 else 'low'

            return {
                'hopkins_statistic': float(hopkins),
                'clustering_tendency': clustering_tendency
            }

        except Exception as e:
            logger.warning(f"Failed to assess clustering tendency: {e}")
            return {'hopkins_statistic': None, 'clustering_tendency': 'unknown'}

    def _calculate_outlier_percentage(self, data: np.ndarray) -> float:
        """Calculate percentage of outliers using IQR method."""
        if len(data) == 0:
            return 0.0

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (data < lower_bound) | (data > upper_bound)
        return float(np.sum(outliers) / len(data) * 100)

    def _assess_data_range(self, var_name: str, data: np.ndarray) -> str:
        """Assess if data range is reasonable for the variable type."""
        if len(data) == 0:
            return 'no_data'

        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min

        # Simple range assessment for common pollutant variables
        if 'no2' in var_name.lower():
            if 0 <= data_min and data_max <= 100:  # mol/m2
                return 'reasonable'
        elif 'so2' in var_name.lower():
            if 0 <= data_min and data_max <= 50:  # mol/m2
                return 'reasonable'
        elif 'co' in var_name.lower():
            if 0 <= data_min and data_max <= 0.1:  # mol/m2
                return 'reasonable'
        elif 'aod' in var_name.lower():
            if 0 <= data_min and data_max <= 5:  # dimensionless
                return 'reasonable'

        # Generic assessment
        if data_range == 0:
            return 'constant'
        elif data_range > np.mean(data) * 10:
            return 'wide_range'
        else:
            return 'unknown'

    def _save_analysis_results(self, results: Dict[str, Any], dataset: xr.Dataset,
                             analysis_types: List[str]) -> None:
        """Save analysis results to disk."""
        try:
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            original_title = dataset.attrs.get('title', 'dataset').replace(' ', '_')
            analysis_str = '_'.join(analysis_types)
            filename = f"statistical_analysis_{original_title}_{analysis_str}_{timestamp}.json"

            output_path = self.processed_data_path / filename

            # Save results as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Saved statistical analysis results to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")

    def generate_statistical_plots(self, dataset: xr.Dataset,
                                  output_dir: Optional[Path] = None) -> None:
        """
        Generate statistical visualization plots.

        Args:
            dataset: Input dataset
            output_dir: Directory to save plots
        """
        try:
            if output_dir:
                FileUtils.ensure_directory(output_dir)

            # Extract pollutant variables
            pollutant_vars = [var for var in dataset.data_vars
                             if var.lower() not in ['lat', 'lon', 'time', 'qa_value', 'quality']]

            if len(pollutant_vars) < 2:
                logger.warning("Need at least 2 variables for statistical plots")
                return

            # Prepare data for plotting
            var_data = {}
            for var in pollutant_vars:
                data_var = dataset[var]
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                var_data[var] = spatial_mean.to_pandas().dropna()

            # Find common time index
            common_index = None
            for var, data in var_data.items():
                if common_index is None:
                    common_index = data.index
                else:
                    common_index = common_index.intersection(data.index)

            if len(common_index) == 0:
                return

            # Align data
            aligned_data = {}
            for var, data in var_data.items():
                aligned_data[var] = data.reindex(common_index).dropna()

            df = pd.DataFrame(aligned_data)

            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Pollutant Correlation Matrix')
            plt.tight_layout()

            if output_dir:
                plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Pairplot
            if len(pollutant_vars) <= 5:  # Limit to avoid too many plots
                plt.figure(figsize=(12, 10))
                sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.6})
                plt.suptitle('Pollutant Relationships', y=1.02)
                plt.tight_layout()

                if output_dir:
                    plt.savefig(output_dir / 'pollutant_pairplot.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, var in enumerate(pollutant_vars[:4]):
                axes[i].hist(df[var].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{var.upper()} Distribution')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(len(pollutant_vars), 4):
                axes[i].set_visible(False)

            plt.suptitle('Pollutant Distributions')
            plt.tight_layout()

            if output_dir:
                plt.savefig(output_dir / 'pollutant_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Statistical plots saved to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to generate statistical plots: {e}")