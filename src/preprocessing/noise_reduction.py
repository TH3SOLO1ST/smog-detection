"""
Noise reduction pipeline for Islamabad Smog Detection System.

This module provides comprehensive noise reduction algorithms for satellite
imagery and time series data, including spatial filtering, temporal smoothing,
and advanced denoising techniques to improve signal quality for smog analysis.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import cv2
from scipy import ndimage, signal
from scipy.stats import zscore
from skimage import restoration, filters
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class NoiseReducer:
    """Comprehensive noise reduction for satellite imagery and time series."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize noise reducer.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.processing_config = self.config.get_section('processing')

        # Noise reduction parameters
        self.noise_params = self.processing_config.get('noise_reduction', {})
        self.gaussian_kernel = self.noise_params.get('gaussian_kernel', 3)
        self.median_filter_size = self.noise_params.get('median_filter_size', 3)
        self.bilateral_d = self.noise_params.get('bilateral_d', 9)
        self.bilateral_sigma_color = self.noise_params.get('bilateral_sigma_color', 75)
        self.bilateral_sigma_space = self.noise_params.get('bilateral_sigma_space', 75)

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.processed_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed')) / 'noise_reduced'
        )

        logger.info("Noise reducer initialized")

    def reduce_noise(self, dataset: xr.Dataset, methods: Optional[List[str]] = None,
                    save_to_disk: bool = True) -> xr.Dataset:
        """
        Apply noise reduction to satellite dataset.

        Args:
            dataset: Input satellite dataset
            methods: List of noise reduction methods to apply
            save_to_disk: Whether to save reduced dataset to disk

        Returns:
            Noise reduced dataset
        """
        if methods is None:
            methods = ['gaussian', 'median', 'bilateral']

        logger.info(f"Applying noise reduction using methods: {methods}")

        reduced_dataset = dataset.copy()

        # Apply noise reduction to each data variable
        for var_name in dataset.data_vars:
            data_var = dataset[var_name]

            # Skip coordinate variables and quality flags
            if var_name.lower() in ['lat', 'lon', 'time', 'qa_value', 'quality']:
                continue

            try:
                reduced_var = data_var.copy()

                # Apply each noise reduction method
                for method in methods:
                    logger.info(f"Applying {method} noise reduction to {var_name}")
                    reduced_var = self._apply_noise_reduction_method(reduced_var, method)

                # Update dataset with reduced variable
                reduced_dataset[var_name] = reduced_var

                # Add noise reduction metadata
                reduced_dataset[var_name].attrs.update({
                    'noise_reduction_methods': methods,
                    'gaussian_kernel': self.gaussian_kernel,
                    'median_filter_size': self.median_filter_size,
                    'noise_reduction_applied': True,
                    'processing_date': pd.Timestamp.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Failed to apply noise reduction to {var_name}: {e}")
                # Keep original variable if reduction fails
                continue

        # Update dataset attributes
        reduced_dataset.attrs.update({
            'noise_reduction_methods': methods,
            'noise_reduction_applied': True,
            'processing_date': pd.Timestamp.now().isoformat(),
            'original_dataset': dataset.attrs.get('title', 'Unknown')
        })

        # Save reduced dataset if requested
        if save_to_disk:
            self._save_reduced_dataset(reduced_dataset, dataset, methods)

        return reduced_dataset

    def _apply_noise_reduction_method(self, data_var: xr.DataArray, method: str) -> xr.DataArray:
        """
        Apply specific noise reduction method to data variable.

        Args:
            data_var: Input data variable
            method: Noise reduction method

        Returns:
            Noise reduced data variable
        """
        data_values = data_var.values

        if method == 'gaussian':
            reduced_values = self._apply_gaussian_filter(data_values)
        elif method == 'median':
            reduced_values = self._apply_median_filter(data_values)
        elif method == 'bilateral':
            reduced_values = self._apply_bilateral_filter(data_values)
        elif method == 'non_local_means':
            reduced_values = self._apply_non_local_means(data_values)
        elif method == 'wiener':
            reduced_values = self._apply_wiener_filter(data_values)
        elif method == 'temporal_smoothing':
            reduced_values = self._apply_temporal_smoothing(data_values)
        elif method == 'wavelet_denoising':
            reduced_values = self._apply_wavelet_denoising(data_values)
        elif method == 'pca_denoising':
            reduced_values = self._apply_pca_denoising(data_values)
        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            return data_var

        # Create reduced DataArray with same coordinates and attributes
        reduced_var = xr.DataArray(
            reduced_values,
            coords=data_var.coords,
            dims=data_var.dims,
            attrs=data_var.attrs.copy()
        )

        return reduced_var

    def _apply_gaussian_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter for noise reduction."""
        try:
            # Create Gaussian kernel
            kernel_size = self.gaussian_kernel if self.gaussian_kernel % 2 == 1 else self.gaussian_kernel + 1
            sigma = kernel_size / 6.0  # Standard deviation

            if data.ndim == 2:
                # Apply 2D Gaussian filter
                reduced = ndimage.gaussian_filter(data, sigma=sigma)
            elif data.ndim == 3:
                # Apply to each 2D slice (assuming first dimension is time)
                reduced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    reduced[i] = ndimage.gaussian_filter(data[i], sigma=sigma)
            elif data.ndim == 4:
                # Apply to each 2D slice (assuming last two dimensions are spatial)
                reduced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        reduced[i, j] = ndimage.gaussian_filter(data[i, j], sigma=sigma)
            else:
                logger.warning(f"Unsupported data dimensionality for Gaussian filter: {data.ndim}")
                return data

            return reduced

        except Exception as e:
            logger.error(f"Gaussian filter failed: {e}")
            return data

    def _apply_median_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply median filter for noise reduction."""
        try:
            kernel_size = self.median_filter_size if self.median_filter_size % 2 == 1 else self.median_filter_size + 1

            if data.ndim == 2:
                reduced = ndimage.median_filter(data, size=kernel_size)
            elif data.ndim == 3:
                reduced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    reduced[i] = ndimage.median_filter(data[i], size=kernel_size)
            elif data.ndim == 4:
                reduced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        reduced[i, j] = ndimage.median_filter(data[i, j], size=kernel_size)
            else:
                logger.warning(f"Unsupported data dimensionality for median filter: {data.ndim}")
                return data

            return reduced

        except Exception as e:
            logger.error(f"Median filter failed: {e}")
            return data

    def _apply_bilateral_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise reduction."""
        try:
            # Normalize data to 0-255 range for OpenCV
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return data

            data_min, data_max = np.min(valid_data), np.max(valid_data)
            if data_max - data_min == 0:
                return data

            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

            if data.ndim == 2:
                reduced_normalized = cv2.bilateralFilter(
                    normalized, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space
                )
            elif data.ndim == 3:
                reduced_normalized = np.zeros_like(normalized)
                for i in range(normalized.shape[0]):
                    reduced_normalized[i] = cv2.bilateralFilter(
                        normalized[i], self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space
                    )
            elif data.ndim == 4:
                reduced_normalized = np.zeros_like(normalized)
                for i in range(normalized.shape[0]):
                    for j in range(normalized.shape[1]):
                        reduced_normalized[i, j] = cv2.bilateralFilter(
                            normalized[i, j], self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space
                        )
            else:
                logger.warning(f"Unsupported data dimensionality for bilateral filter: {data.ndim}")
                return data

            # Scale back to original range
            reduced = (reduced_normalized.astype(np.float32) / 255.0) * (data_max - data_min) + data_min

            # Restore NaN values
            reduced[np.isnan(data)] = np.nan

            return reduced

        except Exception as e:
            logger.error(f"Bilateral filter failed: {e}")
            return data

    def _apply_non_local_means(self, data: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        try:
            # Normalize to 0-255 range
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return data

            data_min, data_max = np.min(valid_data), np.max(valid_data)
            if data_max - data_min == 0:
                return data

            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

            if data.ndim == 2:
                reduced_normalized = cv2.fastNlMeansDenoising(normalized, None, h=10, templateWindowSize=7, searchWindowSize=21)
            elif data.ndim == 3:
                reduced_normalized = np.zeros_like(normalized)
                for i in range(normalized.shape[0]):
                    reduced_normalized[i] = cv2.fastNlMeansDenoising(normalized[i], None, h=10, templateWindowSize=7, searchWindowSize=21)
            else:
                logger.warning(f"Non-local means not supported for {data.ndim}D data")
                return data

            # Scale back to original range
            reduced = (reduced_normalized.astype(np.float32) / 255.0) * (data_max - data_min) + data_min

            # Restore NaN values
            reduced[np.isnan(data)] = np.nan

            return reduced

        except Exception as e:
            logger.error(f"Non-local means denoising failed: {e}")
            return data

    def _apply_wiener_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply Wiener filter for noise reduction."""
        try:
            if data.ndim == 2:
                # Estimate noise variance
                noise_var = np.var(data[~np.isnan(data)]) * 0.1  # Assume 10% noise
                reduced = restoration.wiener(data, noise=noise_var)
            elif data.ndim == 3:
                reduced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    noise_var = np.var(data[i][~np.isnan(data[i])]) * 0.1
                    reduced[i] = restoration.wiener(data[i], noise=noise_var)
            else:
                logger.warning(f"Wiener filter not supported for {data.ndim}D data")
                return data

            return reduced

        except Exception as e:
            logger.error(f"Wiener filter failed: {e}")
            return data

    def _apply_temporal_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing for time series data."""
        try:
            if data.ndim < 3:
                logger.warning("Temporal smoothing requires at least 3D data")
                return data

            # Apply moving average along time dimension (first dimension)
            window_size = 3  # 3-point moving average
            if data.shape[0] < window_size:
                return data

            reduced = np.zeros_like(data)

            for i in range(data.shape[0]):
                # Define window boundaries
                start = max(0, i - window_size // 2)
                end = min(data.shape[0], i + window_size // 2 + 1)

                # Apply moving average
                if data.ndim == 3:
                    reduced[i] = np.nanmean(data[start:end], axis=0)
                elif data.ndim == 4:
                    reduced[i] = np.nanmean(data[start:end], axis=0)

            return reduced

        except Exception as e:
            logger.error(f"Temporal smoothing failed: {e}")
            return data

    def _apply_wavelet_denoising(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising."""
        try:
            if data.ndim == 2:
                # 2D wavelet denoising
                reduced = restoration.denoise_wavelet(data, method='BayesShrink', mode='soft')
            elif data.ndim == 3:
                reduced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    reduced[i] = restoration.denoise_wavelet(data[i], method='BayesShrink', mode='soft')
            else:
                logger.warning(f"Wavelet denoising not supported for {data.ndim}D data")
                return data

            return reduced

        except Exception as e:
            logger.error(f"Wavelet denoising failed: {e}")
            return data

    def _apply_pca_denoising(self, data: np.ndarray) -> np.ndarray:
        """Apply PCA-based denoising."""
        try:
            if data.ndim == 3:
                # Treat time dimension as observations
                original_shape = data.shape
                # Reshape to (time, pixels)
                data_2d = data.reshape(original_shape[0], -1)

                # Remove NaN values
                valid_mask = ~np.isnan(data_2d)
                if not np.any(valid_mask):
                    return data

                # Simple PCA denoising (keeping top components)
                pca = PCA(n_components=min(10, original_shape[0]))

                # Handle NaN values by filling with column means
                data_filled = data_2d.copy()
                col_means = np.nanmean(data_2d, axis=0)
                for j in range(data_2d.shape[1]):
                    mask = valid_mask[:, j]
                    if np.any(~mask):
                        data_filled[~mask, j] = col_means[j]

                # Apply PCA
                data_pca = pca.fit_transform(data_filled)
                data_reconstructed = pca.inverse_transform(data_pca)

                # Restore original shape
                reduced = data_reconstructed.reshape(original_shape)

                # Restore NaN values
                reduced[~valid_mask.reshape(original_shape)] = np.nan

            else:
                logger.warning(f"PCA denoising not supported for {data.ndim}D data")
                return data

            return reduced

        except Exception as e:
            logger.error(f"PCA denoising failed: {e}")
            return data

    def remove_outliers(self, dataset: xr.Dataset, method: str = 'zscore',
                       threshold: float = 3.0) -> xr.Dataset:
        """
        Remove outliers from dataset.

        Args:
            dataset: Input dataset
            method: Outlier detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            Dataset with outliers removed
        """
        logger.info(f"Removing outliers using {method} method with threshold {threshold}")

        cleaned_dataset = dataset.copy()

        for var_name in dataset.data_vars:
            data_var = dataset[var_name]

            # Skip coordinate variables
            if var_name.lower() in ['lat', 'lon', 'time']:
                continue

            try:
                if method == 'zscore':
                    cleaned_var = self._remove_outliers_zscore(data_var, threshold)
                elif method == 'iqr':
                    cleaned_var = self._remove_outliers_iqr(data_var, threshold)
                elif method == 'isolation_forest':
                    cleaned_var = self._remove_outlers_isolation_forest(data_var, threshold)
                else:
                    logger.warning(f"Unknown outlier removal method: {method}")
                    cleaned_var = data_var

                cleaned_dataset[var_name] = cleaned_var

                # Add outlier removal metadata
                cleaned_dataset[var_name].attrs.update({
                    'outlier_removal_method': method,
                    'outlier_threshold': threshold,
                    'outliers_removed': True
                })

            except Exception as e:
                logger.error(f"Failed to remove outliers from {var_name}: {e}")
                continue

        return cleaned_dataset

    def _remove_outliers_zscore(self, data_var: xr.DataArray, threshold: float) -> xr.DataArray:
        """Remove outliers using z-score method."""
        data_values = data_var.values
        valid_mask = ~np.isnan(data_values)

        if np.any(valid_mask):
            valid_data = data_values[valid_mask]
            z_scores = np.abs(zscore(valid_data))
            outlier_mask = z_scores > threshold

            # Create full mask for all data
            full_outlier_mask = np.zeros_like(data_values, dtype=bool)
            full_outlier_mask[valid_mask] = outlier_mask

            # Set outliers to NaN
            cleaned_data = data_values.copy()
            cleaned_data[full_outlier_mask] = np.nan

            cleaned_var = xr.DataArray(
                cleaned_data,
                coords=data_var.coords,
                dims=data_var.dims,
                attrs=data_var.attrs.copy()
            )

            return cleaned_var
        else:
            return data_var

    def _remove_outliers_iqr(self, data_var: xr.DataArray, multiplier: float = 1.5) -> xr.DataArray:
        """Remove outliers using IQR method."""
        data_values = data_var.values
        valid_mask = ~np.isnan(data_values)

        if np.any(valid_mask):
            valid_data = data_values[valid_mask]
            Q1 = np.percentile(valid_data, 25)
            Q3 = np.percentile(valid_data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outlier_mask = (valid_data < lower_bound) | (valid_data > upper_bound)

            # Create full mask for all data
            full_outlier_mask = np.zeros_like(data_values, dtype=bool)
            full_outlier_mask[valid_mask] = outlier_mask

            # Set outliers to NaN
            cleaned_data = data_values.copy()
            cleaned_data[full_outlier_mask] = np.nan

            cleaned_var = xr.DataArray(
                cleaned_data,
                coords=data_var.coords,
                dims=data_var.dims,
                attrs=data_var.attrs.copy()
            )

            return cleaned_var
        else:
            return data_var

    def _remove_outlers_isolation_forest(self, data_var: xr.DataArray, contamination: float) -> xr.DataArray:
        """Remove outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not available for Isolation Forest")
            return data_var

        data_values = data_var.values
        valid_mask = ~np.isnan(data_values)

        if np.any(valid_mask):
            valid_data = data_values[valid_mask]

            # Reshape for sklearn
            X = valid_data.reshape(-1, 1)

            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            outlier_mask = outlier_labels == -1

            # Create full mask for all data
            full_outlier_mask = np.zeros_like(data_values, dtype=bool)
            full_outlier_mask[valid_mask] = outlier_mask

            # Set outliers to NaN
            cleaned_data = data_values.copy()
            cleaned_data[full_outlier_mask] = np.nan

            cleaned_var = xr.DataArray(
                cleaned_data,
                coords=data_var.coords,
                dims=data_var.dims,
                attrs=data_var.attrs.copy()
            )

            return cleaned_var
        else:
            return data_var

    def calculate_noise_metrics(self, original_data: xr.Dataset,
                               reduced_data: xr.Dataset) -> Dict[str, Any]:
        """
        Calculate noise reduction metrics.

        Args:
            original_data: Original dataset
            reduced_data: Noise reduced dataset

        Returns:
            Dictionary with noise reduction metrics
        """
        metrics = {}

        for var_name in original_data.data_vars:
            if var_name in reduced_data.data_vars:
                orig_var = original_data[var_name]
                redu_var = reduced_data[var_name]

                orig_values = orig_var.values[~np.isnan(orig_var.values)]
                redu_values = redu_var.values[~np.isnan(redu_values.values)]

                if len(orig_values) > 0 and len(redu_values) > 0:
                    # Signal-to-noise ratio
                    snr_original = self._calculate_snr(orig_values)
                    snr_reduced = self._calculate_snr(redu_values)

                    # Noise variance (using high-frequency components)
                    noise_var_original = self._estimate_noise_variance(orig_values)
                    noise_var_reduced = self._estimate_noise_variance(redu_values)

                    metrics[var_name] = {
                        'snr_original': snr_original,
                        'snr_reduced': snr_reduced,
                        'snr_improvement': snr_reduced - snr_original,
                        'noise_variance_original': noise_var_original,
                        'noise_variance_reduced': noise_var_reduced,
                        'noise_reduction_ratio': noise_var_reduced / noise_var_original if noise_var_original > 0 else 1.0,
                        'edge_preservation': self._calculate_edge_preservation(orig_values, redu_values)
                    }

        return metrics

    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        signal = np.mean(data)
        noise = np.std(data)
        return signal / noise if noise > 0 else float('inf')

    def _estimate_noise_variance(self, data: np.ndarray) -> float:
        """Estimate noise variance using high-pass filter."""
        if data.ndim == 2:
            # Apply Laplacian filter to get high-frequency components
            laplacian = ndimage.laplace(data)
            return np.var(laplacian[~np.isnan(laplacian)])
        else:
            return np.var(data)

    def _calculate_edge_preservation(self, original: np.ndarray, reduced: np.ndarray) -> float:
        """Calculate edge preservation metric."""
        if original.ndim == 2 and reduced.ndim == 2:
            # Calculate gradients using Sobel filter
            grad_original = filters.sobel_h(original) + filters.sobel_v(original)
            grad_reduced = filters.sobel_h(reduced) + filters.sobel_v(reduced)

            # Correlation between gradients
            original_flat = grad_original[~np.isnan(grad_original)].flatten()
            reduced_flat = grad_reduced[~np.isnan(grad_reduced)].flatten()

            if len(original_flat) > 0 and len(reduced_flat) > 0:
                correlation = np.corrcoef(original_flat, reduced_flat[:len(original_flat)])[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0
        else:
            return 0.0

    def _save_reduced_dataset(self, reduced_dataset: xr.Dataset,
                             original_dataset: xr.Dataset,
                             methods: List[str]) -> None:
        """Save noise reduced dataset to disk."""
        try:
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            original_title = original_dataset.attrs.get('title', 'dataset').replace(' ', '_')
            methods_str = '_'.join(methods)
            filename = f"noise_reduced_{original_title}_{methods_str}_{timestamp}.nc"

            output_path = self.processed_data_path / filename

            # Save as NetCDF
            reduced_dataset.to_netcdf(output_path)

            # Also save noise reduction metrics
            metrics = self.calculate_noise_metrics(original_dataset, reduced_dataset)
            metrics_path = output_path.with_suffix('.json')
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved noise reduced dataset to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save reduced dataset: {e}")

    def generate_noise_reduction_visualization(self, original_data: xr.Dataset,
                                              reduced_data: xr.Dataset,
                                              var_name: str,
                                              output_path: Optional[Path] = None) -> None:
        """
        Generate visualization showing noise reduction results.

        Args:
            original_data: Original dataset
            reduced_data: Noise reduced dataset
            var_name: Variable name to visualize
            output_path: Optional output path for saving plot
        """
        try:
            if var_name not in original_data.data_vars or var_name not in reduced_data.data_vars:
                logger.error(f"Variable {var_name} not found in datasets")
                return

            orig_var = original_data[var_name]
            redu_var = reduced_data[var_name]

            # Get the first 2D slice for visualization
            if orig_var.ndim >= 2:
                orig_2d = orig_var.isel(time=0) if 'time' in orig_var.dims else orig_var
                redu_2d = redu_var.isel(time=0) if 'time' in redu_var.dims else redu_var

                # Reduce to 2D if still higher dimensional
                while orig_2d.ndim > 2:
                    orig_2d = orig_2d.isel({orig_2d.dims[0]: 0})
                    redu_2d = redu_2d.isel({redu_2d.dims[0]: 0})

                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Original data
                im1 = axes[0, 0].imshow(orig_2d.values, cmap='viridis')
                axes[0, 0].set_title('Original Data')
                plt.colorbar(im1, ax=axes[0, 0])

                # Reduced data
                im2 = axes[0, 1].imshow(redu_2d.values, cmap='viridis')
                axes[0, 1].set_title('Noise Reduced')
                plt.colorbar(im2, ax=axes[0, 1])

                # Difference
                diff = redu_2d.values - orig_2d.values
                im3 = axes[1, 0].imshow(diff, cmap='RdBu_r')
                axes[1, 0].set_title('Difference (Reduced - Original)')
                plt.colorbar(im3, ax=axes[1, 0])

                # Noise profile comparison
                # Take a horizontal line through the middle
                mid_row = orig_2d.shape[0] // 2
                orig_line = orig_2d.values[mid_row, :]
                redu_line = redu_2d.values[mid_row, :]

                axes[1, 1].plot(orig_line, label='Original', alpha=0.7)
                axes[1, 1].plot(redu_line, label='Reduced', alpha=0.7)
                axes[1, 1].set_title('Horizontal Profile Comparison')
                axes[1, 1].set_xlabel('Column Index')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.suptitle(f'Noise Reduction Results: {var_name}')
                plt.tight_layout()

                if output_path:
                    FileUtils.ensure_directory(output_path.parent)
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved noise reduction visualization to {output_path}")

                plt.close()

        except Exception as e:
            logger.error(f"Failed to generate noise reduction visualization: {e}")