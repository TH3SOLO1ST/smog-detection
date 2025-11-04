"""
Contrast enhancement module for Islamabad Smog Detection System.

This module provides comprehensive contrast enhancement algorithms for satellite
imagery, including histogram equalization, adaptive methods, and multi-scale
processing to improve feature visibility for smog analysis.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import cv2
from scipy import ndimage, stats
from skimage import exposure, filters
import matplotlib.pyplot as plt

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class ContrastEnhancer:
    """Comprehensive contrast enhancement for satellite imagery."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize contrast enhancer.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.processing_config = self.config.get_section('processing')

        # Contrast enhancement parameters
        self.enhance_params = self.processing_config.get('contrast_enhancement', {})
        self.clahe_clip_limit = self.enhance_params.get('clahe_clip_limit', 2.0)
        self.clahe_grid_size = self.enhance_params.get('clahe_grid_size', 8)
        self.gamma_correction = self.enhance_params.get('gamma_correction', 1.2)
        self.histogram_equalization = self.enhance_params.get('histogram_equalization', True)

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.processed_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed')) / 'enhanced'
        )

        logger.info("Contrast enhancer initialized")

    def enhance_contrast(self, dataset: xr.Dataset, methods: Optional[List[str]] = None,
                        save_to_disk: bool = True) -> xr.Dataset:
        """
        Apply contrast enhancement to satellite dataset.

        Args:
            dataset: Input satellite dataset
            methods: List of contrast enhancement methods to apply
            save_to_disk: Whether to save enhanced dataset to disk

        Returns:
            Contrast enhanced dataset
        """
        if methods is None:
            methods = ['clahe', 'gamma', 'histogram_equalization']

        logger.info(f"Applying contrast enhancement using methods: {methods}")

        enhanced_dataset = dataset.copy()

        # Apply contrast enhancement to each data variable
        for var_name in dataset.data_vars:
            data_var = dataset[var_name]

            # Skip coordinate variables and quality flags
            if var_name.lower() in ['lat', 'lon', 'time', 'qa_value', 'quality']:
                continue

            try:
                enhanced_var = data_var.copy()

                # Apply each contrast enhancement method
                for method in methods:
                    logger.info(f"Applying {method} contrast enhancement to {var_name}")
                    enhanced_var = self._apply_contrast_enhancement_method(enhanced_var, method)

                # Update dataset with enhanced variable
                enhanced_dataset[var_name] = enhanced_var

                # Add enhancement metadata
                enhanced_dataset[var_name].attrs.update({
                    'contrast_enhancement_methods': methods,
                    'clahe_clip_limit': self.clahe_clip_limit,
                    'clahe_grid_size': self.clahe_grid_size,
                    'gamma_correction': self.gamma_correction,
                    'contrast_enhancement_applied': True,
                    'processing_date': pd.Timestamp.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Failed to apply contrast enhancement to {var_name}: {e}")
                # Keep original variable if enhancement fails
                continue

        # Update dataset attributes
        enhanced_dataset.attrs.update({
            'contrast_enhancement_methods': methods,
            'contrast_enhancement_applied': True,
            'processing_date': pd.Timestamp.now().isoformat(),
            'original_dataset': dataset.attrs.get('title', 'Unknown')
        })

        # Save enhanced dataset if requested
        if save_to_disk:
            self._save_enhanced_dataset(enhanced_dataset, dataset, methods)

        return enhanced_dataset

    def _apply_contrast_enhancement_method(self, data_var: xr.DataArray, method: str) -> xr.DataArray:
        """
        Apply specific contrast enhancement method to data variable.

        Args:
            data_var: Input data variable
            method: Contrast enhancement method

        Returns:
            Contrast enhanced data variable
        """
        data_values = data_var.values

        if method == 'clahe':
            enhanced_values = self._apply_clahe(data_values)
        elif method == 'gamma_correction':
            enhanced_values = self._apply_gamma_correction(data_values)
        elif method == 'histogram_equalization':
            enhanced_values = self._apply_histogram_equalization(data_values)
        elif method == 'adaptive_equalization':
            enhanced_values = self._apply_adaptive_equalization(data_values)
        elif method == 'contrast_stretching':
            enhanced_values = self._apply_contrast_stretching(data_values)
        elif method == 'multiscale_enhancement':
            enhanced_values = self._apply_multiscale_enhancement(data_values)
        elif method == 'unsharp_masking':
            enhanced_values = self._apply_unsharp_masking(data_values)
        elif method == 'color_space_enhancement':
            enhanced_values = self._apply_color_space_enhancement(data_values)
        else:
            logger.warning(f"Unknown contrast enhancement method: {method}")
            return data_var

        # Create enhanced DataArray with same coordinates and attributes
        enhanced_var = xr.DataArray(
            enhanced_values,
            coords=data_var.coords,
            dims=data_var.dims,
            attrs=data_var.attrs.copy()
        )

        return enhanced_var

    def _apply_clahe(self, data: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
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
                enhanced_normalized = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
                ).apply(normalized)
            elif data.ndim == 3:
                enhanced_normalized = np.zeros_like(normalized)
                for i in range(normalized.shape[0]):
                    enhanced_normalized[i] = cv2.createCLAHE(
                        clipLimit=self.clahe_clip_limit,
                        tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
                    ).apply(normalized[i])
            elif data.ndim == 4:
                enhanced_normalized = np.zeros_like(normalized)
                for i in range(normalized.shape[0]):
                    for j in range(normalized.shape[1]):
                        enhanced_normalized[i, j] = cv2.createCLAHE(
                            clipLimit=self.clahe_clip_limit,
                            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
                        ).apply(normalized[i, j])
            else:
                logger.warning(f"CLAHE not supported for {data.ndim}D data")
                return data

            # Scale back to original range
            enhanced = (enhanced_normalized.astype(np.float32) / 255.0) * (data_max - data_min) + data_min

            # Restore NaN values
            enhanced[np.isnan(data)] = np.nan

            return enhanced

        except Exception as e:
            logger.error(f"CLAHE failed: {e}")
            return data

    def _apply_gamma_correction(self, data: np.ndarray) -> np.ndarray:
        """Apply gamma correction."""
        try:
            # Handle NaN values
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data

            # Normalize to 0-1 range
            valid_data = data[valid_mask]
            data_min, data_max = np.min(valid_data), np.max(valid_data)

            if data_max - data_min == 0:
                return data

            normalized = (data - data_min) / (data_max - data_min)

            # Apply gamma correction
            enhanced_normalized = np.power(normalized, 1.0 / self.gamma_correction)

            # Scale back to original range
            enhanced = enhanced_normalized * (data_max - data_min) + data_min

            # Restore NaN values
            enhanced[~valid_mask] = np.nan

            return enhanced

        except Exception as e:
            logger.error(f"Gamma correction failed: {e}")
            return data

    def _apply_histogram_equalization(self, data: np.ndarray) -> np.ndarray:
        """Apply global histogram equalization."""
        try:
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data

            if data.ndim == 2:
                # Flatten for histogram calculation
                valid_data = data[valid_mask]
                enhanced_data = data.copy()

                # Calculate CDF
                hist, bin_edges = np.histogram(valid_data, bins=256)
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())

                # Apply histogram equalization
                enhanced_data[valid_mask] = np.interp(valid_data, bin_edges[:-1], cdf_normalized)

                return enhanced_data

            elif data.ndim == 3:
                enhanced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    enhanced[i] = self._apply_histogram_equalization(data[i])
                return enhanced
            else:
                logger.warning(f"Histogram equalization not supported for {data.ndim}D data")
                return data

        except Exception as e:
            logger.error(f"Histogram equalization failed: {e}")
            return data

    def _apply_adaptive_equalization(self, data: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization."""
        try:
            if data.ndim == 2:
                # Use skimage's adaptive histogram equalization
                enhanced = exposure.equalize_adapthist(data, clip_limit=0.03)
                return enhanced
            elif data.ndim == 3:
                enhanced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    enhanced[i] = exposure.equalize_adapthist(data[i], clip_limit=0.03)
                return enhanced
            else:
                logger.warning(f"Adaptive equalization not supported for {data.ndim}D data")
                return data

        except Exception as e:
            logger.error(f"Adaptive equalization failed: {e}")
            return data

    def _apply_contrast_stretching(self, data: np.ndarray) -> np.ndarray:
        """Apply percentile-based contrast stretching."""
        try:
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data

            if data.ndim == 2:
                enhanced = data.copy()
                valid_data = data[valid_mask]

                # Use 2nd and 98th percentiles for stretching
                p2, p98 = np.percentile(valid_data, [2, 98])

                if p98 - p2 > 0:
                    enhanced[valid_mask] = np.clip(
                        (data[valid_mask] - p2) / (p98 - p2),
                        0, 1
                    )

                return enhanced

            elif data.ndim == 3:
                enhanced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    enhanced[i] = self._apply_contrast_stretching(data[i])
                return enhanced
            else:
                logger.warning(f"Contrast stretching not supported for {data.ndim}D data")
                return data

        except Exception as e:
            logger.error(f"Contrast stretching failed: {e}")
            return data

    def _apply_multiscale_enhancement(self, data: np.ndarray) -> np.ndarray:
        """Apply multi-scale enhancement using Laplacian pyramid."""
        try:
            if data.ndim == 2:
                return self._multiscale_enhance_2d(data)
            elif data.ndim == 3:
                enhanced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    enhanced[i] = self._multiscale_enhance_2d(data[i])
                return enhanced
            else:
                logger.warning(f"Multi-scale enhancement not supported for {data.ndim}D data")
                return data

        except Exception as e:
            logger.error(f"Multi-scale enhancement failed: {e}")
            return data

    def _multiscale_enhance_2d(self, data: np.ndarray) -> np.ndarray:
        """Apply multi-scale enhancement to 2D data."""
        # Create Gaussian pyramid
        levels = 4
        pyramid = [data]
        for i in range(levels - 1):
            smoothed = cv2.GaussianBlur(pyramid[-1], (5, 5), 0)
            pyramid.append(smoothed)

        # Create Laplacian pyramid
        laplacian = []
        for i in range(levels - 1):
            laplacian.append(pyramid[i] - pyramid[i + 1])
        laplacian.append(pyramid[-1])

        # Enhance contrast in each level
        enhanced_laplacian = []
        for i, level in enumerate(laplacian):
            if i < levels - 1:
                # Enhance high-frequency components more
                enhancement_factor = 1.0 + 0.2 * (i / (levels - 1))
                enhanced_laplacian.append(level * enhancement_factor)
            else:
                enhanced_laplacian.append(level)

        # Reconstruct image
        enhanced = enhanced_laplacian[-1].copy()
        for i in range(levels - 2, -1, -1):
            enhanced = enhanced_laplacian[i] + enhanced

        # Restore NaN values
        enhanced[np.isnan(data)] = np.nan

        return enhanced

    def _apply_unsharp_masking(self, data: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement."""
        try:
            if data.ndim == 2:
                # Create blurred version
                blurred = cv2.GaussianBlur(data, (5, 5), 0)
                # Apply unsharp masking
                enhanced = cv2.addWeighted(data, 1.5, blurred, -0.5, 0)
                return enhanced
            elif data.ndim == 3:
                enhanced = np.zeros_like(data)
                for i in range(data.shape[0]):
                    blurred = cv2.GaussianBlur(data[i], (5, 5), 0)
                    enhanced[i] = cv2.addWeighted(data[i], 1.5, blurred, -0.5, 0)
                return enhanced
            else:
                logger.warning(f"Unsharp masking not supported for {data.ndim}D data")
                return data

        except Exception as e:
            logger.error(f"Unsharp masking failed: {e}")
            return data

    def _apply_color_space_enhancement(self, data: np.ndarray) -> np.ndarray:
        """Apply enhancement in different color spaces."""
        # This is a placeholder implementation
        # In practice, you would convert to LAB, HSV, or YUV color spaces
        logger.info("Color space enhancement (placeholder implementation)")
        return data

    def calculate_enhancement_metrics(self, original_data: xr.Dataset,
                                    enhanced_data: xr.Dataset) -> Dict[str, Any]:
        """
        Calculate contrast enhancement metrics.

        Args:
            original_data: Original dataset
            enhanced_data: Contrast enhanced dataset

        Returns:
            Dictionary with enhancement metrics
        """
        metrics = {}

        for var_name in original_data.data_vars:
            if var_name in enhanced_data.data_vars:
                orig_var = original_data[var_name]
                enh_var = enhanced_data[var_name]

                orig_values = orig_var.values[~np.isnan(orig_var.values)]
                enh_values = enh_var.values[~np.isnan(enh_var.values)]

                if len(orig_values) > 0 and len(enh_values) > 0:
                    # Contrast metrics
                    orig_contrast = np.std(orig_values)
                    enh_contrast = np.std(enh_values)

                    # Dynamic range
                    orig_dynamic = np.max(orig_values) - np.min(orig_values)
                    enh_dynamic = np.max(enh_values) - np.min(enh_values)

                    # Entropy (information content)
                    orig_entropy = self._calculate_entropy(orig_values)
                    enh_entropy = self._calculate_entropy(enh_values)

                    # Edge content
                    orig_edge_content = self._calculate_edge_content(orig_values)
                    enh_edge_content = self._calculate_edge_content(enh_values)

                    metrics[var_name] = {
                        'contrast_original': orig_contrast,
                        'contrast_enhanced': enh_contrast,
                        'contrast_improvement': enh_contrast - orig_contrast,
                        'dynamic_range_original': orig_dynamic,
                        'dynamic_range_enhanced': enh_dynamic,
                        'dynamic_range_improvement': enh_dynamic - orig_dynamic,
                        'entropy_original': orig_entropy,
                        'entropy_enhanced': enh_entropy,
                        'entropy_improvement': enh_entropy - orig_entropy,
                        'edge_content_original': orig_edge_content,
                        'edge_content_enhanced': enh_edge_content,
                        'edge_content_improvement': enh_edge_content - orig_edge_content
                    }

        return metrics

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        # Create histogram
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log2(hist))

    def _calculate_edge_content(self, data: np.ndarray) -> float:
        """Calculate edge content using Sobel filter."""
        if data.ndim == 2:
            # Apply Sobel filter
            sobel_x = filters.sobel_h(data)
            sobel_y = filters.sobel_v(data)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return np.mean(edge_magnitude[~np.isnan(edge_magnitude)])
        else:
            return 0.0

    def _save_enhanced_dataset(self, enhanced_dataset: xr.Dataset,
                              original_dataset: xr.Dataset,
                              methods: List[str]) -> None:
        """Save enhanced dataset to disk."""
        try:
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            original_title = original_dataset.attrs.get('title', 'dataset').replace(' ', '_')
            methods_str = '_'.join(methods)
            filename = f"enhanced_{original_title}_{methods_str}_{timestamp}.nc"

            output_path = self.processed_data_path / filename

            # Save as NetCDF
            enhanced_dataset.to_netcdf(output_path)

            # Also save enhancement metrics
            metrics = self.calculate_enhancement_metrics(original_dataset, enhanced_dataset)
            metrics_path = output_path.with_suffix('.json')
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved enhanced dataset to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save enhanced dataset: {e}")

    def generate_enhancement_visualization(self, original_data: xr.Dataset,
                                         enhanced_data: xr.Dataset,
                                         var_name: str,
                                         output_path: Optional[Path] = None) -> None:
        """
        Generate visualization showing contrast enhancement results.

        Args:
            original_data: Original dataset
            enhanced_data: Contrast enhanced dataset
            var_name: Variable name to visualize
            output_path: Optional output path for saving plot
        """
        try:
            if var_name not in original_data.data_vars or var_name not in enhanced_data.data_vars:
                logger.error(f"Variable {var_name} not found in datasets")
                return

            orig_var = original_data[var_name]
            enh_var = enhanced_data[var_name]

            # Get the first 2D slice for visualization
            if orig_var.ndim >= 2:
                orig_2d = orig_var.isel(time=0) if 'time' in orig_var.dims else orig_var
                enh_2d = enh_var.isel(time=0) if 'time' in enh_var.dims else enh_var

                # Reduce to 2D if still higher dimensional
                while orig_2d.ndim > 2:
                    orig_2d = orig_2d.isel({orig_2d.dims[0]: 0})
                    enh_2d = enh_2d.isel({enh_2d.dims[0]: 0})

                # Create visualization
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Original data
                im1 = axes[0, 0].imshow(orig_2d.values, cmap='viridis')
                axes[0, 0].set_title('Original Data')
                plt.colorbar(im1, ax=axes[0, 0])

                # Enhanced data
                im2 = axes[0, 1].imshow(enh_2d.values, cmap='viridis')
                axes[0, 1].set_title('Enhanced Data')
                plt.colorbar(im2, ax=axes[0, 1])

                # Difference
                diff = enh_2d.values - orig_2d.values
                im3 = axes[0, 2].imshow(diff, cmap='RdBu_r')
                axes[0, 2].set_title('Difference (Enhanced - Original)')
                plt.colorbar(im3, ax=axes[0, 2])

                # Histogram comparison
                orig_flat = orig_2d.values[~np.isnan(orig_2d.values)].flatten()
                enh_flat = enh_2d.values[~np.isnan(enh_2d.values)].flatten()

                axes[1, 0].hist(orig_flat, bins=50, alpha=0.7, label='Original', color='blue')
                axes[1, 0].hist(enh_flat, bins=50, alpha=0.7, label='Enhanced', color='red')
                axes[1, 0].set_title('Histogram Comparison')
                axes[1, 0].set_xlabel('Value')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # Cumulative distribution
                axes[1, 1].hist(orig_flat, bins=50, alpha=0.7, label='Original',
                               cumulative=True, color='blue', density=True)
                axes[1, 1].hist(enh_flat, bins=50, alpha=0.7, label='Enhanced',
                               cumulative=True, color='red', density=True)
                axes[1, 1].set_title('Cumulative Distribution')
                axes[1, 1].set_xlabel('Value')
                axes[1, 1].set_ylabel('Cumulative Probability')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                # Profile comparison
                mid_row = orig_2d.shape[0] // 2
                orig_line = orig_2d.values[mid_row, :]
                enh_line = enh_2d.values[mid_row, :]

                axes[1, 2].plot(orig_line, label='Original', alpha=0.7)
                axes[1, 2].plot(enh_line, label='Enhanced', alpha=0.7)
                axes[1, 2].set_title('Horizontal Profile Comparison')
                axes[1, 2].set_xlabel('Column Index')
                axes[1, 2].set_ylabel('Value')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

                plt.suptitle(f'Contrast Enhancement Results: {var_name}')
                plt.tight_layout()

                if output_path:
                    FileUtils.ensure_directory(output_path.parent)
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved enhancement visualization to {output_path}")

                plt.close()

        except Exception as e:
            logger.error(f"Failed to generate enhancement visualization: {e}")

    def batch_enhance_datasets(self, datasets: List[xr.Dataset],
                             methods: Optional[List[str]] = None) -> List[xr.Dataset]:
        """
        Apply contrast enhancement to multiple datasets.

        Args:
            datasets: List of datasets to enhance
            methods: Enhancement methods

        Returns:
            List of enhanced datasets
        """
        enhanced_datasets = []

        for i, dataset in enumerate(datasets):
            try:
                logger.info(f"Enhancing dataset {i+1}/{len(datasets)}")
                enhanced = self.enhance_contrast(dataset, methods=methods, save_to_disk=True)
                enhanced_datasets.append(enhanced)
            except Exception as e:
                logger.error(f"Failed to enhance dataset {i+1}: {e}")
                # Add original dataset if enhancement fails
                enhanced_datasets.append(dataset)

        logger.info(f"Enhanced {len(enhanced_datasets)}/{len(datasets)} datasets")
        return enhanced_datasets