"""
Time series processor for Islamabad Smog Detection System.

This module provides comprehensive time series analysis capabilities including
decomposition, trend analysis, anomaly detection, and temporal aggregation
for air pollution data in the Islamabad region.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
from scipy import stats, signal, interpolate
from scipy.stats import zscore, variation
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """Comprehensive time series analysis for air pollution data."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize time series processor.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.time_series_config = self.config.get_section('time_series')

        # Time series processing parameters
        self.aggregation_config = self.time_series_config.get('aggregation', {})
        self.gap_filling_config = self.time_series_config.get('gap_filling', {})
        self.trend_config = self.time_series_config.get('trend_analysis', {})

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.processed_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed')) / 'timeseries'
        )

        # Create subdirectories for different aggregation levels
        self.daily_path = FileUtils.ensure_directory(self.processed_data_path / 'daily')
        self.weekly_path = FileUtils.ensure_directory(self.processed_data_path / 'weekly')
        self.monthly_path = FileUtils.ensure_directory(self.processed_data_path / 'monthly')

        logger.info("Time series processor initialized")

    def process_time_series(self, dataset: xr.Dataset, operations: Optional[List[str]] = None,
                           save_to_disk: bool = True) -> xr.Dataset:
        """
        Process time series data with various analysis operations.

        Args:
            dataset: Input dataset with time dimension
            operations: List of time series operations to apply
            save_to_disk: Whether to save processed data to disk

        Returns:
            Processed time series dataset
        """
        if operations is None:
            operations = ['aggregate', 'decompose', 'detect_anomalies', 'trend_analysis']

        if 'time' not in dataset.dims:
            logger.warning("No time dimension found in dataset")
            return dataset

        logger.info(f"Processing time series with operations: {operations}")

        processed_dataset = dataset.copy()

        # Apply each time series operation
        for operation in operations:
            try:
                logger.info(f"Applying {operation} operation")

                if operation == 'aggregate':
                    processed_dataset = self._aggregate_time_series(processed_dataset)
                elif operation == 'decompose':
                    processed_dataset = self._decompose_time_series(processed_dataset)
                elif operation == 'detect_anomalies':
                    processed_dataset = self._detect_anomalies(processed_dataset)
                elif operation == 'trend_analysis':
                    processed_dataset = self._analyze_trends(processed_dataset)
                elif operation == 'gap_filling':
                    processed_dataset = self._fill_gaps(processed_dataset)
                elif operation == 'smooth':
                    processed_dataset = self._smooth_time_series(processed_dataset)
                else:
                    logger.warning(f"Unknown time series operation: {operation}")

            except Exception as e:
                logger.error(f"Failed to apply {operation}: {e}")
                continue

        # Update dataset attributes
        processed_dataset.attrs.update({
            'time_series_processing': operations,
            'processing_date': pd.Timestamp.now().isoformat(),
            'original_dataset': dataset.attrs.get('title', 'Unknown')
        })

        # Save processed dataset if requested
        if save_to_disk:
            self._save_time_series(processed_dataset, dataset, operations)

        return processed_dataset

    def _aggregate_time_series(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Aggregate time series to different temporal resolutions.

        Args:
            dataset: Input dataset

        Returns:
            Dataset with aggregated time series
        """
        aggregated_dataset = dataset.copy()

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            try:
                # Daily aggregation (if not already daily)
                if self.aggregation_config.get('daily', True):
                    daily_agg = data_var.resample(time='1D').agg({
                        'mean': 'mean',
                        'median': 'median',
                        'std': 'std',
                        'min': 'min',
                        'max': 'max',
                        'count': 'count'
                    })

                    for agg_name in ['mean', 'median', 'std', 'min', 'max', 'count']:
                        var_name_agg = f"{var_name}_daily_{agg_name}"
                        aggregated_dataset[var_name_agg] = daily_agg[agg_name]

                # Weekly aggregation
                if self.aggregation_config.get('weekly', True):
                    weekly_agg = data_var.resample(time='1W').agg({
                        'mean': 'mean',
                        'median': 'median',
                        'std': 'std',
                        'min': 'min',
                        'max': 'max'
                    })

                    for agg_name in ['mean', 'median', 'std', 'min', 'max']:
                        var_name_agg = f"{var_name}_weekly_{agg_name}"
                        aggregated_dataset[var_name_agg] = weekly_agg[agg_name]

                # Monthly aggregation
                if self.aggregation_config.get('monthly', True):
                    monthly_agg = data_var.resample(time='1M').agg({
                        'mean': 'mean',
                        'median': 'median',
                        'std': 'std',
                        'min': 'min',
                        'max': 'max'
                    })

                    for agg_name in ['mean', 'median', 'std', 'min', 'max']:
                        var_name_agg = f"{var_name}_monthly_{agg_name}"
                        aggregated_dataset[var_name_agg] = monthly_agg[agg_name]

                # Seasonal aggregation
                if self.aggregation_config.get('seasonal', True):
                    seasonal_agg = data_var.groupby('time.season').agg({
                        'mean': 'mean',
                        'median': 'median',
                        'std': 'std'
                    })

                    for agg_name in ['mean', 'median', 'std']:
                        var_name_agg = f"{var_name}_seasonal_{agg_name}"
                        # Convert to DataArray with season coordinate
                        for season, values in seasonal_agg[agg_name].items():
                            seasonal_da = xr.DataArray(
                                values,
                                coords={'season': [season]},
                                dims=['season'],
                                attrs=data_var.attrs.copy()
                            )
                            aggregated_dataset[var_name_agg] = seasonal_da

            except Exception as e:
                logger.error(f"Failed to aggregate {var_name}: {e}")
                continue

        return aggregated_dataset

    def _decompose_time_series(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            dataset: Input dataset

        Returns:
            Dataset with decomposed components
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available, skipping decomposition")
            return dataset

        decomposed_dataset = dataset.copy()

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            # Create spatial mean for decomposition
            if 'lat' in data_var.dims and 'lon' in data_var.dims:
                spatial_mean = data_var.mean(dim=['lat', 'lon'])
            else:
                spatial_mean = data_var

            try:
                # Convert to pandas Series for decomposition
                ts_df = spatial_mean.to_pandas()
                ts_series = ts_df.dropna()

                if len(ts_series) < 24:  # Need minimum data points for decomposition
                    logger.warning(f"Insufficient data points for {var_name} decomposition")
                    continue

                # Perform seasonal decomposition
                if len(ts_series) >= 24:  # At least 2 years of monthly data
                    decomposition = seasonal_decompose(
                        ts_series,
                        model='additive',
                        period=12  # Monthly seasonality
                    )
                else:
                    # Use STL for shorter time series
                    stl = STL(ts_series, seasonal=7)
                    decomposition = stl.fit()

                # Add components to dataset
                trend_da = xr.DataArray(
                    decomposition.trend.values,
                    coords={'time': decomposition.trend.index},
                    dims=['time'],
                    attrs={'description': f'Trend component for {var_name}'}
                )
                decomposed_dataset[f"{var_name}_trend"] = trend_da

                seasonal_da = xr.DataArray(
                    decomposition.seasonal.values,
                    coords={'time': decomposition.seasonal.index},
                    dims=['time'],
                    attrs={'description': f'Seasonal component for {var_name}'}
                )
                decomposed_dataset[f"{var_name}_seasonal"] = seasonal_da

                residual_da = xr.DataArray(
                    decomposition.resid.values,
                    coords={'time': decomposition.resid.index},
                    dims=['time'],
                    attrs={'description': f'Residual component for {var_name}'}
                )
                decomposed_dataset[f"{var_name}_residual"] = residual_da

            except Exception as e:
                logger.error(f"Failed to decompose {var_name}: {e}")
                continue

        return decomposed_dataset

    def _detect_anomalies(self, dataset: xr.Dataset, method: str = 'zscore') -> xr.Dataset:
        """
        Detect anomalies in time series data.

        Args:
            dataset: Input dataset
            method: Anomaly detection method ('zscore', 'iqr', 'isolation_forest')

        Returns:
            Dataset with anomaly flags
        """
        anomaly_dataset = dataset.copy()

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            try:
                # Create spatial mean for anomaly detection
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                # Convert to pandas Series
                ts_series = spatial_mean.to_pandas().dropna()

                if method == 'zscore':
                    anomalies = self._detect_anomalies_zscore(ts_series)
                elif method == 'iqr':
                    anomalies = self._detect_anomalies_iqr(ts_series)
                elif method == 'isolation_forest':
                    anomalies = self._detect_anomalies_isolation_forest(ts_series)
                else:
                    logger.warning(f"Unknown anomaly detection method: {method}")
                    continue

                # Create anomaly DataArray
                anomaly_da = xr.DataArray(
                    anomalies.values,
                    coords={'time': anomalies.index},
                    dims=['time'],
                    attrs={
                        'description': f'Anomaly flags for {var_name}',
                        'method': method,
                        'anomaly_threshold': 3.0 if method == 'zscore' else 1.5
                    }
                )
                anomaly_dataset[f"{var_name}_anomalies"] = anomaly_da

                # Add anomaly statistics
                anomaly_stats = {
                    'total_anomalies': int(np.sum(anomalies)),
                    'anomaly_percentage': float(np.sum(anomalies) / len(anomalies) * 100),
                    'method': method
                }

                anomaly_dataset.attrs[f"{var_name}_anomaly_stats"] = anomaly_stats

            except Exception as e:
                logger.error(f"Failed to detect anomalies for {var_name}: {e}")
                continue

        return anomaly_dataset

    def _detect_anomalies_zscore(self, ts_series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect anomalies using z-score method."""
        z_scores = np.abs(zscore(ts_series))
        anomalies = z_scores > threshold
        return pd.Series(anomalies, index=ts_series.index)

    def _detect_anomalies_iqr(self, ts_series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect anomalies using IQR method."""
        Q1 = ts_series.quantile(0.25)
        Q3 = ts_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        anomalies = (ts_series < lower_bound) | (ts_series > upper_bound)
        return anomalies

    def _detect_anomalies_isolation_forest(self, ts_series: pd.Series, contamination: float = 0.1) -> pd.Series:
        """Detect anomalies using Isolation Forest."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using z-score instead")
            return self._detect_anomalies_zscore(ts_series)

        from sklearn.ensemble import IsolationForest

        # Reshape data for sklearn
        X = ts_series.values.reshape(-1, 1)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        anomalies = pd.Series(anomaly_labels == -1, index=ts_series.index)

        return anomalies

    def _analyze_trends(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Analyze trends in time series data.

        Args:
            dataset: Input dataset

        Returns:
            Dataset with trend analysis results
        """
        trend_dataset = dataset.copy()

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            try:
                # Create spatial mean for trend analysis
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                # Convert to pandas Series
                ts_series = spatial_mean.to_pandas().dropna()

                if len(ts_series) < 10:
                    logger.warning(f"Insufficient data points for {var_name} trend analysis")
                    continue

                # Linear trend analysis
                x = np.arange(len(ts_series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_series)

                # Mann-Kendall trend test
                mk_result = self._mann_kendall_test(ts_series.values)

                # Add trend analysis results
                trend_stats = {
                    'linear_slope': float(slope),
                    'linear_intercept': float(intercept),
                    'linear_r_squared': float(r_value ** 2),
                    'linear_p_value': float(p_value),
                    'mann_kendall_statistic': float(mk_result['statistic']),
                    'mann_kendall_p_value': float(mk_result['p_value']),
                    'trend_direction': mk_result['trend']
                }

                trend_dataset.attrs[f"{var_name}_trend_stats"] = trend_stats

                # Create trend line
                trend_line = slope * x + intercept
                trend_da = xr.DataArray(
                    trend_line,
                    coords={'time': ts_series.index},
                    dims=['time'],
                    attrs={'description': f'Linear trend for {var_name}'}
                )
                trend_dataset[f"{var_name}_trend_line"] = trend_da

                # Create detrended series
                detrended = ts_series.values - trend_line
                detrended_da = xr.DataArray(
                    detrended,
                    coords={'time': ts_series.index},
                    dims=['time'],
                    attrs={'description': f'Detrended {var_name}'}
                )
                trend_dataset[f"{var_name}_detrended"] = detrended_da

            except Exception as e:
                logger.error(f"Failed to analyze trends for {var_name}: {e}")
                continue

        return trend_dataset

    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform Mann-Kendall trend test."""
        n = len(data)

        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1

        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Determine trend direction
        if z > 1.96:
            trend = 'increasing'
        elif z < -1.96:
            trend = 'decreasing'
        else:
            trend = 'no trend'

        return {
            'statistic': s,
            'z_score': z,
            'p_value': p_value,
            'trend': trend
        }

    def _fill_gaps(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Fill gaps in time series data.

        Args:
            dataset: Input dataset

        Returns:
            Dataset with filled gaps
        """
        gap_filled_dataset = dataset.copy()

        max_gap_days = self.gap_filling_config.get('max_gap_days', 3)
        interpolation_method = self.gap_filling_config.get('interpolation_method', 'linear')

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            try:
                # Convert to pandas DataFrame for gap filling
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    # Handle spatial data - fill gaps at each spatial point
                    gap_filled_var = data_var.copy()

                    # Iterate over spatial dimensions
                    if data_var.ndim == 3:  # time x lat x lon
                        for lat_idx in range(data_var.sizes['lat']):
                            for lon_idx in range(data_var.sizes['lon']):
                                ts_values = data_var[:, lat_idx, lon_idx].to_pandas()
                                filled_ts = self._fill_time_series_gaps(
                                    ts_values, max_gap_days, interpolation_method
                                )
                                gap_filled_var[:, lat_idx, lon_idx] = filled_ts.values

                    gap_filled_dataset[var_name] = gap_filled_var
                else:
                    # Handle single time series
                    ts_series = data_var.to_pandas()
                    filled_ts = self._fill_time_series_gaps(
                        ts_series, max_gap_days, interpolation_method
                    )

                    filled_da = xr.DataArray(
                        filled_ts.values,
                        coords=data_var.coords,
                        dims=data_var.dims,
                        attrs=data_var.attrs.copy()
                    )
                    gap_filled_dataset[var_name] = filled_da

            except Exception as e:
                logger.error(f"Failed to fill gaps for {var_name}: {e}")
                continue

        gap_filled_dataset.attrs['gap_filled'] = True

        return gap_filled_dataset

    def _fill_time_series_gaps(self, ts_series: pd.Series, max_gap_days: int,
                               method: str = 'linear') -> pd.Series:
        """Fill gaps in a single time series."""
        # Create complete time index
        full_index = pd.date_range(
            start=ts_series.index.min(),
            end=ts_series.index.max(),
            freq='D'
        )

        # Reindex to include all dates
        ts_reindexed = ts_series.reindex(full_index)

        # Fill gaps based on method
        if method == 'linear':
            ts_filled = ts_reindexed.interpolate(method='linear')
        elif method == 'polynomial':
            ts_filled = ts_reindexed.interpolate(method='polynomial', order=2)
        elif method == 'spline':
            ts_filled = ts_reindexed.interpolate(method='spline', order=3)
        else:
            ts_filled = ts_reindexed.interpolate(method='linear')

        # Only fill gaps smaller than max_gap_days
        gaps = ts_reindexed.isna()
        gap_lengths = self._calculate_consecutive_gaps(gaps)

        for start, length in gap_lengths:
            if length > max_gap_days:
                # Keep NaNs for large gaps
                gap_end = start + pd.Timedelta(days=length)
                ts_filled[start:gap_end] = np.nan

        return ts_filled

    def _calculate_consecutive_gaps(self, na_series: pd.Series) -> List[Tuple[pd.Timestamp, int]]:
        """Calculate consecutive NaN gaps in time series."""
        gaps = []
        current_gap_start = None
        current_gap_length = 0

        for i, is_na in enumerate(na_series):
            if is_na:
                if current_gap_start is None:
                    current_gap_start = na_series.index[i]
                    current_gap_length = 1
                else:
                    current_gap_length += 1
            else:
                if current_gap_start is not None:
                    gaps.append((current_gap_start, current_gap_length))
                    current_gap_start = None
                    current_gap_length = 0

        # Handle gap at end of series
        if current_gap_start is not None:
            gaps.append((current_gap_start, current_gap_length))

        return gaps

    def _smooth_time_series(self, dataset: xr.Dataset, window_size: int = 7) -> xr.Dataset:
        """
        Apply smoothing to time series data.

        Args:
            dataset: Input dataset
            window_size: Moving window size for smoothing

        Returns:
            Dataset with smoothed time series
        """
        smoothed_dataset = dataset.copy()

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            try:
                # Apply moving average smoothing
                smoothed_var = data_var.rolling(time=window_size, center=True).mean()
                smoothed_dataset[f"{var_name}_smoothed"] = smoothed_var

            except Exception as e:
                logger.error(f"Failed to smooth {var_name}: {e}")
                continue

        return smoothed_dataset

    def calculate_temporal_statistics(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Calculate comprehensive temporal statistics.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with temporal statistics
        """
        stats = {}

        for var_name in dataset.data_vars:
            if 'time' not in dataset[var_name].dims:
                continue

            data_var = dataset[var_name]

            try:
                # Create spatial mean for statistics
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    spatial_mean = data_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = data_var

                ts_series = spatial_mean.to_pandas().dropna()

                if len(ts_series) > 0:
                    stats[var_name] = {
                        'start_date': str(ts_series.index.min()),
                        'end_date': str(ts_series.index.max()),
                        'duration_days': (ts_series.index.max() - ts_series.index.min()).days,
                        'num_observations': len(ts_series),
                        'mean': float(ts_series.mean()),
                        'median': float(ts_series.median()),
                        'std': float(ts_series.std()),
                        'min': float(ts_series.min()),
                        'max': float(ts_series.max()),
                        'range': float(ts_series.max() - ts_series.min()),
                        'coefficient_of_variation': float(ts_series.std() / ts_series.mean()) if ts_series.mean() != 0 else 0
                    }

                    # Seasonal statistics
                    if len(ts_series) >= 12:  # At least one year
                        seasonal_stats = {}
                        for season, season_data in ts_series.groupby(ts_series.index.season):
                            seasonal_stats[season] = {
                                'mean': float(season_data.mean()),
                                'std': float(season_data.std()),
                                'count': len(season_data)
                            }
                        stats[var_name]['seasonal'] = seasonal_stats

                    # Monthly statistics
                    monthly_stats = {}
                    for month, month_data in ts_series.groupby(ts_series.index.month):
                        monthly_stats[month] = {
                            'mean': float(month_data.mean()),
                            'std': float(month_data.std()),
                            'count': len(month_data)
                        }
                    stats[var_name]['monthly'] = monthly_stats

            except Exception as e:
                logger.error(f"Failed to calculate statistics for {var_name}: {e}")
                continue

        return stats

    def _save_time_series(self, processed_dataset: xr.Dataset,
                         original_dataset: xr.Dataset,
                         operations: List[str]) -> None:
        """Save processed time series dataset to disk."""
        try:
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            original_title = original_dataset.attrs.get('title', 'dataset').replace(' ', '_')
            operations_str = '_'.join(operations)
            filename = f"timeseries_{original_title}_{operations_str}_{timestamp}.nc"

            output_path = self.processed_data_path / filename

            # Save as NetCDF
            processed_dataset.to_netcdf(output_path)

            # Also save temporal statistics
            stats = self.calculate_temporal_statistics(processed_dataset)
            stats_path = output_path.with_suffix('.json')
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            logger.info(f"Saved time series dataset to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save time series dataset: {e}")

    def generate_time_series_plots(self, dataset: xr.Dataset, var_name: str,
                                  output_dir: Optional[Path] = None) -> None:
        """
        Generate comprehensive time series visualization plots.

        Args:
            dataset: Input dataset
            var_name: Variable name to plot
            output_dir: Directory to save plots
        """
        try:
            if var_name not in dataset.data_vars:
                logger.error(f"Variable {var_name} not found in dataset")
                return

            data_var = dataset[var_name]

            # Create spatial mean for plotting
            if 'lat' in data_var.dims and 'lon' in data_var.dims:
                spatial_mean = data_var.mean(dim=['lat', 'lon'])
            else:
                spatial_mean = data_var

            ts_series = spatial_mean.to_pandas().dropna()

            if len(ts_series) == 0:
                logger.warning(f"No valid data for {var_name}")
                return

            # Create output directory
            if output_dir:
                FileUtils.ensure_directory(output_dir)

            # Create comprehensive plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Original time series
            axes[0, 0].plot(ts_series.index, ts_series.values, linewidth=1)
            axes[0, 0].set_title(f'{var_name} Time Series')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True, alpha=0.3)

            # Monthly aggregation
            if len(ts_series) >= 12:
                monthly_data = ts_series.resample('M').mean()
                axes[0, 1].plot(monthly_data.index, monthly_data.values, linewidth=2, color='red')
                axes[0, 1].set_title(f'{var_name} Monthly Mean')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].grid(True, alpha=0.3)

            # Seasonal pattern
            if len(ts_series) >= 24:  # At least 2 years
                seasonal_data = ts_series.groupby(ts_series.index.month).mean()
                axes[1, 0].bar(seasonal_data.index, seasonal_data.values)
                axes[1, 0].set_title(f'{var_name} Seasonal Pattern')
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Mean Value')
                axes[1, 0].grid(True, alpha=0.3)

            # Distribution
            axes[1, 1].hist(ts_series.values, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(f'{var_name} Distribution')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f'Time Series Analysis: {var_name}', fontsize=16)
            plt.tight_layout()

            if output_dir:
                output_path = output_dir / f'{var_name}_timeseries_analysis.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved time series plot to {output_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Failed to generate time series plots for {var_name}: {e}")