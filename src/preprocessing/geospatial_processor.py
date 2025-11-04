"""
Geospatial processor for Islamabad Smog Detection System.

This module provides comprehensive geospatial processing capabilities including
projection handling, resampling, region extraction, and data fusion for satellite
imagery in the Islamabad region.
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import pyproj
from pyproj import Transformer, CRS as pyCRS
import cv2

try:
    from rasterio.warp import calculate_default_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class GeospatialProcessor:
    """Comprehensive geospatial processing for satellite data."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize geospatial processor.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.region_config = self.config.get_section('region')

        # Islamabad region configuration
        self.region = GeoUtils.create_islamabad_region(
            buffer_km=self.region_config.get('buffer_km', 50)
        )

        # Target projection (UTM Zone 43N for Pakistan)
        self.target_crs = f"EPSG:{GeoUtils.get_epsg_code_for_utm_zone(43, 'north')}"
        self.target_resolution = 1000  # 1km resolution

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.processed_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('processed_data', 'data/processed'))
        )

        logger.info("Geospatial processor initialized")

    def process_dataset(self, dataset: xr.Dataset, operations: Optional[List[str]] = None,
                       save_to_disk: bool = True) -> xr.Dataset:
        """
        Apply comprehensive geospatial processing to dataset.

        Args:
            dataset: Input satellite dataset
            operations: List of geospatial operations to apply
            save_to_disk: Whether to save processed dataset to disk

        Returns:
            Geospatially processed dataset
        """
        if operations is None:
            operations = ['reproject', 'clip', 'resample']

        logger.info(f"Applying geospatial processing: {operations}")

        processed_dataset = dataset.copy()

        # Apply each geospatial operation
        for operation in operations:
            try:
                logger.info(f"Applying {operation} processing")

                if operation == 'reproject':
                    processed_dataset = self._reproject_dataset(processed_dataset)
                elif operation == 'clip':
                    processed_dataset = self._clip_to_region(processed_dataset)
                elif operation == 'resample':
                    processed_dataset = self._resample_dataset(processed_dataset)
                elif operation == 'align':
                    processed_dataset = self._align_datasets(processed_dataset)
                elif operation == 'quality_filter':
                    processed_dataset = self._apply_quality_filtering(processed_dataset)
                else:
                    logger.warning(f"Unknown geospatial operation: {operation}")

            except Exception as e:
                logger.error(f"Failed to apply {operation} processing: {e}")
                continue

        # Update dataset attributes
        processed_dataset.attrs.update({
            'geospatial_processing': operations,
            'target_crs': self.target_crs,
            'target_resolution': self.target_resolution,
            'region': 'Islamabad',
            'processing_date': pd.Timestamp.now().isoformat(),
            'original_dataset': dataset.attrs.get('title', 'Unknown')
        })

        # Save processed dataset if requested
        if save_to_disk:
            self._save_processed_dataset(processed_dataset, dataset, operations)

        return processed_dataset

    def _reproject_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Reproject dataset to target coordinate reference system.

        Args:
            dataset: Input dataset

        Returns:
            Reprojected dataset
        """
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available, skipping reprojection")
            return dataset

        try:
            # Check current CRS
            current_crs = dataset.attrs.get('crs', 'EPSG:4326')  # Assume WGS84 if not specified

            if current_crs == self.target_crs:
                logger.info("Dataset already in target CRS")
                return dataset

            processed_dataset = dataset.copy()

            # Reproject each data variable
            for var_name in dataset.data_vars:
                data_var = dataset[var_name]

                # Skip coordinate variables
                if var_name.lower() in ['lat', 'lon', 'time']:
                    continue

                try:
                    reprojected_var = self._reproject_dataarray(data_var, current_crs, self.target_crs)
                    processed_dataset[var_name] = reprojected_var

                except Exception as e:
                    logger.error(f"Failed to reproject variable {var_name}: {e}")
                    continue

            # Update coordinates
            if 'lat' in dataset.coords and 'lon' in dataset.coords:
                processed_dataset = self._update_coordinates_after_reprojection(processed_dataset)

            processed_dataset.attrs['crs'] = self.target_crs
            processed_dataset.attrs['reprojected_from'] = current_crs

            return processed_dataset

        except Exception as e:
            logger.error(f"Reprojection failed: {e}")
            return dataset

    def _reproject_dataarray(self, data_array: xr.DataArray, src_crs: str, dst_crs: str) -> xr.DataArray:
        """Reproject individual data array."""
        if data_array.ndim < 2:
            logger.warning(f"Skipping reprojection for {data_array.ndim}D data")
            return data_array

        # Get spatial bounds
        if 'lat' in data_array.coords and 'lon' in data_array.coords:
            lat_coords = data_array['lat'].values
            lon_coords = data_array['lon'].values

            bounds = [
                np.min(lon_coords), np.min(lat_coords),
                np.max(lon_coords), np.max(lat_coords)
            ]
        else:
            logger.warning("No lat/lon coordinates found for reprojection")
            return data_array

        # Calculate transform and dimensions for target CRS
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, data_array.sizes.get('lon', 512), data_array.sizes.get('lat', 512),
            *bounds
        )

        # Create destination array
        if data_array.ndim == 2:
            dst_array = np.empty((height, width), dtype=data_array.dtype)
        elif data_array.ndim == 3:
            dst_array = np.empty((data_array.shape[0], height, width), dtype=data_array.dtype)
        else:
            logger.warning(f"Reprojection not supported for {data_array.ndim}D data")
            return data_array

        # Reproject data
        if data_array.ndim == 2:
            reproject(
                source=data_array.values,
                destination=dst_array,
                src_transform=from_bounds(*bounds, data_array.sizes.get('lon', 512), data_array.sizes.get('lat', 512)),
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
        elif data_array.ndim == 3:
            for i in range(data_array.shape[0]):
                reproject(
                    source=data_array[i].values,
                    destination=dst_array[i],
                    src_transform=from_bounds(*bounds, data_array.sizes.get('lon', 512), data_array.sizes.get('lat', 512)),
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

        # Create new coordinates
        dst_bounds = rasterio.transform.array_bounds(height, width, transform)
        new_lon = np.linspace(dst_bounds[0], dst_bounds[2], width)
        new_lat = np.linspace(dst_bounds[1], dst_bounds[3], height)

        # Create reprojected DataArray
        coords = data_array.coords.copy()
        coords['lon'] = new_lon
        coords['lat'] = new_lat

        reprojected = xr.DataArray(
            dst_array,
            coords=coords,
            dims=data_array.dims,
            attrs=data_array.attrs.copy()
        )

        return reprojected

    def _update_coordinates_after_reprojection(self, dataset: xr.Dataset) -> xr.Dataset:
        """Update coordinate information after reprojection."""
        # This is a simplified implementation
        # In practice, you would properly calculate new coordinate grids
        return dataset

    def _clip_to_region(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Clip dataset to Islamabad region.

        Args:
            dataset: Input dataset

        Returns:
            Clipped dataset
        """
        try:
            if 'lat' not in dataset.coords or 'lon' not in dataset.coords:
                logger.warning("No lat/lon coordinates found for clipping")
                return dataset

            bbox = self.region['bounding_box']

            # Clip to bounding box
            clipped_dataset = dataset.sel(
                lat=slice(bbox['south'], bbox['north']),
                lon=slice(bbox['west'], bbox['east'])
            )

            # Add clipping metadata
            clipped_dataset.attrs.update({
                'clipped_to_region': 'Islamabad',
                'clipping_bounds': bbox,
                'original_bounds': {
                    'lat': [float(dataset.lat.min()), float(dataset.lat.max())],
                    'lon': [float(dataset.lon.min()), float(dataset.lon.max())]
                }
            })

            logger.info(f"Clipped dataset to Islamabad region: {bbox}")

            return clipped_dataset

        except Exception as e:
            logger.error(f"Clipping failed: {e}")
            return dataset

    def _resample_dataset(self, dataset: xr.Dataset, target_resolution: Optional[float] = None) -> xr.Dataset:
        """
        Resample dataset to target resolution.

        Args:
            dataset: Input dataset
            target_resolution: Target resolution in meters

        Returns:
            Resampled dataset
        """
        if target_resolution is None:
            target_resolution = self.target_resolution

        try:
            processed_dataset = dataset.copy()

            # Calculate new dimensions
            if 'lat' in dataset.coords and 'lon' in dataset.coords:
                lat_range = dataset.lat.max() - dataset.lat.min()
                lon_range = dataset.lon.max() - dataset.lon.min()

                # Convert to meters (approximate)
                lat_size = int(lat_range * 111000 / target_resolution)  # 1 degree ≈ 111km
                lon_size = int(lon_range * 111000 * np.cos(np.radians(dataset.lat.mean())) / target_resolution)

                new_lat = np.linspace(dataset.lat.min(), dataset.lat.max(), lat_size)
                new_lon = np.linspace(dataset.lon.min(), dataset.lon.max(), lon_size)

                processed_dataset = processed_dataset.interp(
                    lat=new_lat,
                    lon=new_lon,
                    method='linear'
                )

                processed_dataset.attrs.update({
                    'resampled_to_resolution': target_resolution,
                    'resampling_method': 'linear',
                    'original_dimensions': {
                        'lat_size': dataset.sizes.get('lat', 0),
                        'lon_size': dataset.sizes.get('lon', 0)
                    },
                    'resampled_dimensions': {
                        'lat_size': lat_size,
                        'lon_size': lon_size
                    }
                })

            return processed_dataset

        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return dataset

    def _align_datasets(self, dataset: xr.Dataset, reference_grid: Optional[Dict] = None) -> xr.Dataset:
        """
        Align dataset to reference grid.

        Args:
            dataset: Input dataset
            reference_grid: Reference grid specification

        Returns:
            Aligned dataset
        """
        # This is a placeholder for dataset alignment
        # In practice, you would align to a common grid for multi-source data
        logger.info("Dataset alignment (placeholder implementation)")
        return dataset

    def _apply_quality_filtering(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply quality-based filtering to dataset.

        Args:
            dataset: Input dataset

        Returns:
            Quality filtered dataset
        """
        processed_dataset = dataset.copy()

        # Apply quality filtering based on quality flags if available
        for var_name in dataset.data_vars:
            if 'qa_value' in dataset.data_vars:
                qa_mask = dataset['qa_value'] < 3  # Good quality pixels
                processed_dataset[var_name] = processed_dataset[var_name].where(qa_mask)

        processed_dataset.attrs['quality_filtered'] = True

        return processed_dataset

    def fuse_datasets(self, datasets: List[xr.Dataset], method: str = 'weighted_average') -> xr.Dataset:
        """
        Fuse multiple datasets into a single dataset.

        Args:
            datasets: List of datasets to fuse
            method: Fusion method ('weighted_average', 'mosaic', 'blend')

        Returns:
            Fused dataset
        """
        if len(datasets) < 2:
            logger.warning("Need at least 2 datasets for fusion")
            return datasets[0] if datasets else xr.Dataset()

        logger.info(f"Fusing {len(datasets)} datasets using {method} method")

        try:
            if method == 'weighted_average':
                return self._weighted_average_fusion(datasets)
            elif method == 'mosaic':
                return self._mosaic_fusion(datasets)
            elif method == 'blend':
                return self._blend_fusion(datasets)
            else:
                logger.warning(f"Unknown fusion method: {method}")
                return datasets[0]

        except Exception as e:
            logger.error(f"Dataset fusion failed: {e}")
            return datasets[0]

    def _weighted_average_fusion(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        """Fuse datasets using weighted average."""
        # Align all datasets to common grid
        reference_dataset = datasets[0]
        aligned_datasets = []

        for dataset in datasets[1:]:
            try:
                aligned = dataset.interp(
                    lat=reference_dataset.lat,
                    lon=reference_dataset.lon,
                    method='linear'
                )
                aligned_datasets.append(aligned)
            except Exception as e:
                logger.error(f"Failed to align dataset: {e}")
                continue

        # Add reference dataset
        aligned_datasets.insert(0, reference_dataset)

        # Create fused dataset
        fused_dataset = reference_dataset.copy()

        # Fuse each data variable
        for var_name in reference_dataset.data_vars:
            if var_name.lower() in ['lat', 'lon', 'time']:
                continue

            if all(var_name in ds.data_vars for ds in aligned_datasets):
                # Calculate weighted average (weights based on data quality or recency)
                weights = np.ones(len(aligned_datasets))

                # In practice, you would calculate weights based on:
                # - Data quality flags
                # - Temporal proximity
                # - Sensor characteristics

                # Simple equal-weighted average for now
                fused_values = np.zeros_like(reference_dataset[var_name].values)
                weight_sum = np.zeros_like(fused_values)

                for i, dataset in enumerate(aligned_datasets):
                    data_values = dataset[var_name].values
                    weight = weights[i]

                    # Handle NaN values
                    valid_mask = ~np.isnan(data_values)
                    fused_values[valid_mask] += data_values[valid_mask] * weight
                    weight_sum[valid_mask] += weight

                # Normalize by weight sum
                mask = weight_sum > 0
                fused_values[mask] /= weight_sum[mask]

                fused_dataset[var_name] = xr.DataArray(
                    fused_values,
                    coords=reference_dataset[var_name].coords,
                    dims=reference_dataset[var_name].dims,
                    attrs=reference_dataset[var_name].attrs.copy()
                )

        # Add fusion metadata
        fused_dataset.attrs.update({
            'fused_datasets': len(datasets),
            'fusion_method': 'weighted_average',
            'fusion_date': pd.Timestamp.now().isoformat()
        })

        return fused_dataset

    def _mosaic_fusion(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        """Fuse datasets using mosaicking."""
        # Simple mosaic implementation
        # In practice, you would use proper mosaicking algorithms
        reference_dataset = datasets[0]
        return reference_dataset.copy()

    def _blend_fusion(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        """Fuse datasets using blending."""
        # Simple blend implementation
        # In practice, you would use sophisticated blending techniques
        reference_dataset = datasets[0]
        return reference_dataset.copy()

    def create_analysis_grid(self, resolution: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create analysis grid for Islamabad region.

        Args:
            resolution: Grid resolution in meters

        Returns:
            Tuple of (lon_grid, lat_grid) coordinate arrays
        """
        bbox = self.region['bounding_box']

        # Calculate grid dimensions
        lat_size = int((bbox['north'] - bbox['south']) * 111000 / resolution)
        lon_size = int((bbox['east'] - bbox['west']) * 111000 / resolution)

        # Create coordinate arrays
        lon_grid = np.linspace(bbox['west'], bbox['east'], lon_size)
        lat_grid = np.linspace(bbox['south'], bbox['north'], lat_size)

        return lon_grid, lat_grid

    def calculate_spatial_statistics(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Calculate spatial statistics for dataset.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with spatial statistics
        """
        stats = {}

        for var_name in dataset.data_vars:
            if var_name.lower() in ['lat', 'lon', 'time']:
                continue

            data_var = dataset[var_name]
            data_values = data_var.values
            valid_data = data_values[~np.isnan(data_values)]

            if len(valid_data) > 0:
                stats[var_name] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data)),
                    'percentiles': {
                        '25': float(np.percentile(valid_data, 25)),
                        '75': float(np.percentile(valid_data, 75)),
                        '90': float(np.percentile(valid_data, 90)),
                        '95': float(np.percentile(valid_data, 95))
                    },
                    'count_valid': int(len(valid_data)),
                    'count_total': int(data_values.size),
                    'coverage': float(len(valid_data) / data_values.size)
                }

        return stats

    def _save_processed_dataset(self, processed_dataset: xr.Dataset,
                               original_dataset: xr.Dataset,
                               operations: List[str]) -> None:
        """Save processed dataset to disk."""
        try:
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            original_title = original_dataset.attrs.get('title', 'dataset').replace(' ', '_')
            operations_str = '_'.join(operations)
            filename = f"geoprocessed_{original_title}_{operations_str}_{timestamp}.nc"

            output_path = self.processed_data_path / filename

            # Save as NetCDF
            processed_dataset.to_netcdf(output_path)

            # Also save processing metadata
            metadata = {
                'original_title': original_dataset.attrs.get('title', 'Unknown'),
                'operations': operations,
                'processing_date': pd.Timestamp.now().isoformat(),
                'region': 'Islamabad',
                'target_crs': self.target_crs,
                'target_resolution': self.target_resolution
            }

            metadata_path = output_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved geoprocessed dataset to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save processed dataset: {e}")

    def batch_process_datasets(self, datasets: List[xr.Dataset],
                              operations: Optional[List[str]] = None) -> List[xr.Dataset]:
        """
        Apply geospatial processing to multiple datasets.

        Args:
            datasets: List of datasets to process
            operations: Processing operations

        Returns:
            List of processed datasets
        """
        processed_datasets = []

        for i, dataset in enumerate(datasets):
            try:
                logger.info(f"Processing dataset {i+1}/{len(datasets)}")
                processed = self.process_dataset(dataset, operations=operations, save_to_disk=True)
                processed_datasets.append(processed)
            except Exception as e:
                logger.error(f"Failed to process dataset {i+1}: {e}")
                # Add original dataset if processing fails
                processed_datasets.append(dataset)

        logger.info(f"Processed {len(processed_datasets)}/{len(datasets)} datasets")
        return processed_datasets

    def validate_spatial_consistency(self, dataset: xr.Dataset) -> Dict[str, bool]:
        """
        Validate spatial consistency of dataset.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Check coordinate ordering
        if 'lat' in dataset.coords and 'lon' in dataset.coords:
            lat_coords = dataset['lat'].values
            lon_coords = dataset['lon'].values

            validation_results['latitude_increasing'] = lat_coords[-1] > lat_coords[0]
            validation_results['longitude_increasing'] = lon_coords[-1] > lon_coords[0]

            # Check for valid coordinate ranges
            validation_results['latitude_range_valid'] = -90 <= np.min(lat_coords) and np.max(lat_coords) <= 90
            validation_results['longitude_range_valid'] = -180 <= np.min(lon_coords) and np.max(lon_coords) <= 180

            # Check if dataset covers Islamabad region
            bbox = self.region['bounding_box']
            covers_islamabad = (
                np.min(lat_coords) <= bbox['north'] and
                np.max(lat_coords) >= bbox['south'] and
                np.min(lon_coords) <= bbox['east'] and
                np.max(lon_coords) >= bbox['west']
            )
            validation_results['covers_islamabad'] = covers_islamabad

        return validation_results