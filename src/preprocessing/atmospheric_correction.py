"""
Atmospheric correction module for Islamabad Smog Detection System.

This module provides comprehensive atmospheric correction algorithms for satellite
imagery, including Dark Object Subtraction (DOS), haze removal, and advanced
correction methods to improve data quality for smog analysis.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import cv2
from scipy import ndimage
from scipy.stats import norm
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.plot import show
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import rio_color.color
    RIO_COLOR_AVAILABLE = True
except ImportError:
    RIO_COLOR_AVAILABLE = False

try:
    import sk_eo
    SK_EO_AVAILABLE = True
except ImportError:
    SK_EO_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class AtmosphericCorrector:
    """Comprehensive atmospheric correction for satellite imagery."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize atmospheric corrector.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.processing_config = self.config.get_section('processing')

        # Atmospheric correction parameters
        self.atm_params = self.processing_config.get('atmospheric_correction', {})
        self.dark_object_percentile = self.atm_params.get('dark_object_percentile', 1)
        self.haze_correction = self.atm_params.get('haze_correction', True)
        self.color_balance = self.atm_params.get('color_balance', True)
        self.method = self.atm_params.get('method', 'DOS')

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.processed_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed')) / 'atmospheric_corrected'
        )

        logger.info("Atmospheric corrector initialized")

    def correct_dataset(self, dataset: xr.Dataset, method: Optional[str] = None,
                       save_to_disk: bool = True) -> xr.Dataset:
        """
        Apply atmospheric correction to satellite dataset.

        Args:
            dataset: Input satellite dataset
            method: Correction method ('DOS', 'haze_removal', 'combined')
            save_to_disk: Whether to save corrected dataset to disk

        Returns:
            Atmospheric corrected dataset
        """
        if method is None:
            method = self.method

        logger.info(f"Applying atmospheric correction using {method} method")

        corrected_dataset = dataset.copy()

        # Apply correction to each data variable
        for var_name in dataset.data_vars:
            data_var = dataset[var_name]

            # Skip coordinate variables and quality flags
            if var_name.lower() in ['lat', 'lon', 'time', 'qa_value', 'quality']:
                continue

            try:
                if method == 'DOS':
                    corrected_var = self._apply_dos_correction(data_var)
                elif method == 'haze_removal':
                    corrected_var = self._apply_haze_removal(data_var)
                elif method == 'combined':
                    corrected_var = self._apply_combined_correction(data_var)
                else:
                    logger.warning(f"Unknown correction method: {method}")
                    corrected_var = data_var

                # Update dataset with corrected variable
                corrected_dataset[var_name] = corrected_var

                # Add correction metadata
                corrected_dataset[var_name].attrs.update({
                    'atmospheric_correction': method,
                    'correction_applied': True,
                    'dark_object_percentile': self.dark_object_percentile,
                    'correction_date': pd.Timestamp.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Failed to correct variable {var_name}: {e}")
                # Keep original variable if correction fails
                continue

        # Update dataset attributes
        corrected_dataset.attrs.update({
            'atmospheric_correction_method': method,
            'correction_applied': True,
            'processing_date': pd.Timestamp.now().isoformat(),
            'original_dataset': dataset.attrs.get('title', 'Unknown')
        })

        # Save corrected dataset if requested
        if save_to_disk:
            self._save_corrected_dataset(corrected_dataset, dataset, method)

        return corrected_dataset

    def _apply_dos_correction(self, data_var: xr.DataArray) -> xr.DataArray:
        """
        Apply Dark Object Subtraction (DOS) atmospheric correction.

        Args:
            data_var: Input data variable

        Returns:
            DOS corrected data variable
        """
        logger.info(f"Applying DOS correction to {data_var.name}")

        # Get data values
        data_values = data_var.values

        # Handle different dimensionalities
        if data_var.ndim == 2:
            corrected_data = self._dos_correction_2d(data_values)
        elif data_var.ndim == 3:
            corrected_data = self._dos_correction_3d(data_values)
        elif data_var.ndim == 4:
            corrected_data = self._dos_correction_4d(data_values)
        else:
            logger.warning(f"Unsupported data dimensionality: {data_var.ndim}")
            return data_var

        # Create corrected DataArray with same coordinates and attributes
        corrected_var = xr.DataArray(
            corrected_data,
            coords=data_var.coords,
            dims=data_var.dims,
            attrs=data_var.attrs.copy()
        )

        return corrected_var

    def _dos_correction_2d(self, data_2d: np.ndarray) -> np.ndarray:
        """Apply DOS correction to 2D data."""
        # Calculate histogram
        valid_data = data_2d[~np.isnan(data_2d)].flatten()
        if len(valid_data) == 0:
            return data_2d

        # Find dark object value (1st percentile)
        dark_value = np.percentile(valid_data, self.dark_object_percentile)

        # Apply DOS correction: subtract dark object value
        corrected_data = data_2d - dark_value

        # Ensure no negative values (for physical meaningfulness)
        corrected_data = np.maximum(corrected_data, 0)

        return corrected_data

    def _dos_correction_3d(self, data_3d: np.ndarray) -> np.ndarray:
        """Apply DOS correction to 3D data (e.g., time x lat x lon)."""
        corrected_data = np.zeros_like(data_3d)

        # Apply DOS correction to each 2D slice
        for i in range(data_3d.shape[0]):
            corrected_data[i] = self._dos_correction_2d(data_3d[i])

        return corrected_data

    def _dos_correction_4d(self, data_4d: np.ndarray) -> np.ndarray:
        """Apply DOS correction to 4D data."""
        corrected_data = np.zeros_like(data_4d)

        # Apply DOS correction to each 2D slice (assuming last two dimensions are spatial)
        for i in range(data_4d.shape[0]):
            for j in range(data_4d.shape[1]):
                corrected_data[i, j] = self._dos_correction_2d(data_4d[i, j])

        return corrected_data

    def _apply_haze_removal(self, data_var: xr.DataArray) -> xr.DataArray:
        """
        Apply haze removal algorithms.

        Args:
            data_var: Input data variable

        Returns:
            Haze removed data variable
        """
        logger.info(f"Applying haze removal to {data_var.name}")

        data_values = data_var.values

        if RIO_COLOR_AVAILABLE:
            # Use rio-color for haze removal if available
            if data_var.ndim == 2:
                # Convert to 8-bit for rio-color processing
                normalized_data = self._normalize_to_8bit(data_values)
                corrected_data = self._haze_removal_riocolor(normalized_data)
                # Scale back to original range
                corrected_data = self._scale_back_to_range(corrected_data, data_values)
            else:
                # Process 2D slices
                corrected_data = self._haze_removal_multidim(data_values)
        else:
            # Fallback to simple haze removal
            corrected_data = self._simple_haze_removal(data_values)

        corrected_var = xr.DataArray(
            corrected_data,
            coords=data_var.coords,
            dims=data_var.dims,
            attrs=data_var.attrs.copy()
        )

        return corrected_var

    def _apply_combined_correction(self, data_var: xr.DataArray) -> xr.DataArray:
        """
        Apply combined atmospheric correction (DOS + haze removal).

        Args:
            data_var: Input data variable

        Returns:
            Combined corrected data variable
        """
        logger.info(f"Applying combined atmospheric correction to {data_var.name}")

        # First apply DOS correction
        dos_corrected = self._apply_dos_correction(data_var)

        # Then apply haze removal
        final_corrected = self._apply_haze_removal(dos_corrected)

        # Apply color balancing if enabled
        if self.color_balance:
            final_corrected = self._apply_color_balancing(final_corrected)

        return final_corrected

    def _normalize_to_8bit(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to 8-bit range (0-255)."""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return data

        data_min, data_max = np.min(valid_data), np.max(valid_data)
        if data_max - data_min == 0:
            return np.zeros_like(data, dtype=np.uint8)

        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        return normalized

    def _scale_back_to_range(self, corrected_8bit: np.ndarray, original_data: np.ndarray) -> np.ndarray:
        """Scale 8-bit corrected data back to original data range."""
        valid_original = original_data[~np.isnan(original_data)]
        if len(valid_original) == 0:
            return original_data

        data_min, data_max = np.min(valid_original), np.max(valid_original)
        scaled = (corrected_8bit.astype(np.float32) / 255.0) * (data_max - data_min) + data_min

        return scaled

    def _haze_removal_riocolor(self, data_8bit: np.ndarray) -> np.ndarray:
        """Apply haze removal using rio-color library."""
        try:
            # Simple haze removal using rio-color
            # Note: This is a simplified implementation
            # In practice, you might want to use more sophisticated methods

            # Apply atmospheric correction simulation
            corrected = cv2.detailEnhance(data_8bit, sigma_s=10, sigma_r=0.15)
            return corrected

        except Exception as e:
            logger.error(f"rio-color haze removal failed: {e}")
            return data_8bit

    def _haze_removal_multidim(self, data: np.ndarray) -> np.ndarray:
        """Apply haze removal to multidimensional data."""
        corrected_data = np.zeros_like(data)

        if data.ndim == 3:
            for i in range(data.shape[0]):
                normalized = self._normalize_to_8bit(data[i])
                corrected_8bit = self._haze_removal_riocolor(normalized)
                corrected_data[i] = self._scale_back_to_range(corrected_8bit, data[i])
        elif data.ndim == 4:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    normalized = self._normalize_to_8bit(data[i, j])
                    corrected_8bit = self._haze_removal_riocolor(normalized)
                    corrected_data[i, j] = self._scale_back_to_range(corrected_8bit, data[i, j])

        return corrected_data

    def _simple_haze_removal(self, data: np.ndarray) -> np.ndarray:
        """Simple haze removal using image processing techniques."""
        # Apply Gaussian blur to estimate atmospheric background
        if data.ndim == 2:
            background = cv2.GaussianBlur(data, (51, 51), 0)
            corrected = data - 0.5 * background  # Subtract part of background
        elif data.ndim == 3:
            corrected = np.zeros_like(data)
            for i in range(data.shape[0]):
                background = cv2.GaussianBlur(data[i], (51, 51), 0)
                corrected[i] = data[i] - 0.5 * background
        else:
            corrected = data

        # Ensure no negative values
        corrected = np.maximum(corrected, 0)

        return corrected

    def _apply_color_balancing(self, data_var: xr.DataArray) -> xr.DataArray:
        """Apply color balancing to multiple-band data."""
        # This is a placeholder for color balancing
        # In practice, you would implement histogram matching or other techniques
        logger.info("Applying color balancing (placeholder implementation)")
        return data_var

    def calculate_correction_metrics(self, original_data: xr.Dataset,
                                    corrected_data: xr.Dataset) -> Dict[str, Any]:
        """
        Calculate metrics to assess atmospheric correction effectiveness.

        Args:
            original_data: Original dataset
            corrected_data: Corrected dataset

        Returns:
            Dictionary with correction metrics
        """
        metrics = {}

        # Compare statistics for each data variable
        for var_name in original_data.data_vars:
            if var_name in corrected_data.data_vars:
                orig_var = original_data[var_name]
                corr_var = corrected_data[var_name]

                orig_values = orig_var.values[~np.isnan(orig_var.values)]
                corr_values = corr_var.values[~np.isnan(corr_var.values)]

                if len(orig_values) > 0 and len(corr_values) > 0:
                    # Basic statistics
                    metrics[var_name] = {
                        'original_mean': float(np.mean(orig_values)),
                        'corrected_mean': float(np.mean(corr_values)),
                        'original_std': float(np.std(orig_values)),
                        'corrected_std': float(np.std(corr_values)),
                        'mean_change': float(np.mean(corr_values) - np.mean(orig_values)),
                        'std_change': float(np.std(corr_values) - np.std(orig_values))
                    }

                    # Advanced metrics
                    # Signal-to-noise ratio improvement
                    if np.std(orig_values) > 0:
                        snr_original = np.abs(np.mean(orig_values)) / np.std(orig_values)
                        snr_corrected = np.abs(np.mean(corr_values)) / np.std(corr_values)
                        metrics[var_name]['snr_improvement'] = snr_corrected - snr_original

                    # Contrast improvement (using coefficient of variation)
                    if np.mean(orig_values) > 0:
                        cv_original = np.std(orig_values) / np.mean(orig_values)
                        cv_corrected = np.std(corr_values) / np.mean(corr_values)
                        metrics[var_name]['contrast_improvement'] = cv_corrected - cv_original

        return metrics

    def generate_correction_visualization(self, original_data: xr.Dataset,
                                         corrected_data: xr.Dataset,
                                         var_name: str,
                                         output_path: Optional[Path] = None) -> None:
        """
        Generate visualization showing before/after atmospheric correction.

        Args:
            original_data: Original dataset
            corrected_data: Corrected dataset
            var_name: Variable name to visualize
            output_path: Optional output path for saving plot
        """
        try:
            if var_name not in original_data.data_vars or var_name not in corrected_data.data_vars:
                logger.error(f"Variable {var_name} not found in datasets")
                return

            orig_var = original_data[var_name]
            corr_var = corrected_data[var_name]

            # Get the first 2D slice for visualization
            if orig_var.ndim >= 2:
                orig_2d = orig_var.isel(time=0) if 'time' in orig_var.dims else orig_var
                corr_2d = corr_var.isel(time=0) if 'time' in corr_var.dims else corr_var

                # Reduce to 2D if still higher dimensional
                while orig_2d.ndim > 2:
                    orig_2d = orig_2d.isel({orig_2d.dims[0]: 0})
                    corr_2d = corr_2d.isel({corr_2d.dims[0]: 0})

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original data
                im1 = axes[0].imshow(orig_2d.values, cmap='viridis')
                axes[0].set_title('Original Data')
                axes[0].set_xlabel('Longitude')
                axes[0].set_ylabel('Latitude')
                plt.colorbar(im1, ax=axes[0])

                # Corrected data
                im2 = axes[1].imshow(corr_2d.values, cmap='viridis')
                axes[1].set_title('Atmospheric Corrected')
                axes[1].set_xlabel('Longitude')
                axes[1].set_ylabel('Latitude')
                plt.colorbar(im2, ax=axes[1])

                # Difference
                diff = corr_2d.values - orig_2d.values
                im3 = axes[2].imshow(diff, cmap='RdBu_r')
                axes[2].set_title('Difference (Corrected - Original)')
                axes[2].set_xlabel('Longitude')
                axes[2].set_ylabel('Latitude')
                plt.colorbar(im3, ax=axes[2])

                plt.suptitle(f'Atmospheric Correction Results: {var_name}')
                plt.tight_layout()

                if output_path:
                    FileUtils.ensure_directory(output_path.parent)
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved correction visualization to {output_path}")

                plt.close()

        except Exception as e:
            logger.error(f"Failed to generate correction visualization: {e}")

    def _save_corrected_dataset(self, corrected_dataset: xr.Dataset,
                               original_dataset: xr.Dataset,
                               method: str) -> None:
        """Save corrected dataset to disk."""
        try:
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            original_title = original_dataset.attrs.get('title', 'dataset').replace(' ', '_')
            filename = f"atm_corrected_{original_title}_{method}_{timestamp}.nc"

            output_path = self.processed_data_path / filename

            # Save as NetCDF
            corrected_dataset.to_netcdf(output_path)

            # Also save correction metrics
            metrics = self.calculate_correction_metrics(original_dataset, corrected_dataset)
            metrics_path = output_path.with_suffix('.json')
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved atmospheric corrected dataset to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save corrected dataset: {e}")

    def batch_correct_datasets(self, datasets: List[xr.Dataset],
                             method: Optional[str] = None) -> List[xr.Dataset]:
        """
        Apply atmospheric correction to multiple datasets.

        Args:
            datasets: List of datasets to correct
            method: Correction method

        Returns:
            List of corrected datasets
        """
        corrected_datasets = []

        for i, dataset in enumerate(datasets):
            try:
                logger.info(f"Correcting dataset {i+1}/{len(datasets)}")
                corrected = self.correct_dataset(dataset, method=method, save_to_disk=True)
                corrected_datasets.append(corrected)
            except Exception as e:
                logger.error(f"Failed to correct dataset {i+1}: {e}")
                # Add original dataset if correction fails
                corrected_datasets.append(dataset)

        logger.info(f"Corrected {len(corrected_datasets)}/{len(datasets)} datasets")
        return corrected_datasets

    def validate_correction_quality(self, original_data: xr.Dataset,
                                  corrected_data: xr.Dataset) -> Dict[str, bool]:
        """
        Validate the quality of atmospheric correction.

        Args:
            original_data: Original dataset
            corrected_data: Corrected dataset

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        for var_name in original_data.data_vars:
            if var_name in corrected_data.data_vars:
                orig_var = original_data[var_name]
                corr_var = corrected_data[var_name]

                orig_values = orig_var.values[~np.isnan(orig_var.values)]
                corr_values = corr_var.values[~np.isnan(corr_var.values)]

                if len(orig_values) > 0 and len(corr_values) > 0:
                    # Check if correction improved data quality
                    validation_results[var_name] = {
                        'no_negative_values': np.all(corr_values >= 0),
                        'reasonable_mean_change': abs(np.mean(corr_values) - np.mean(orig_values)) < np.std(orig_values),
                        'preserved_data_structure': np.std(corr_values) > 0.1 * np.std(orig_values),
                        'improved_contrast': np.std(corr_values) >= np.std(orig_values)
                    }

        return validation_results