"""
Google Earth Engine data collector for Islamabad Smog Detection System.

This module provides functionality to retrieve pre-processed satellite data
from Google Earth Engine, including Sentinel-5P Level 3 data, MODIS collections,
and Landsat imagery for the Islamabad region.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import pandas as pd
import numpy as np

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    ee = None

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class GEECollector:
    """Collects satellite data via Google Earth Engine."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Google Earth Engine collector.

        Args:
            config_path: Optional path to configuration file
        """
        if not GEE_AVAILABLE:
            raise ImportError("Google Earth Engine Python API not available")

        self.config = get_config(config_path)
        self.api_config = self.config.get_api_config('google_earth_engine')
        self.region_config = self.config.get_section('region')
        self.data_sources = self.config.get_section('data_sources')

        # GEE configuration
        self.project_id = self.api_config.get('project_id')
        self.service_account_key = self.api_config.get('service_account_key')
        self.max_pixels = self.api_config.get('max_pixels', 1e10)

        # Data collections
        self.collections = self.data_sources.get('google_earth_engine', {}).get('collections', {})

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.raw_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('raw_data', 'data/raw')) / 'gee'
        )

        # Initialize region
        self.region = GeoUtils.create_islamabad_region(
            buffer_km=self.region_config.get('buffer_km', 50)
        )

        # Initialize GEE
        self._initialize_gee()

        logger.info("Google Earth Engine collector initialized")

    def _initialize_gee(self) -> None:
        """Initialize Google Earth Engine authentication and connection."""
        try:
            if self.service_account_key and os.path.exists(self.service_account_key):
                # Service account authentication
                service_account = os.getenv('GEE_SERVICE_ACCOUNT')
                if service_account:
                    credentials = ee.ServiceAccountCredentials(service_account, self.service_account_key)
                    ee.Initialize(credentials, project=self.project_id)
                    logger.info("GEE initialized with service account")
                else:
                    raise ValueError("GEE_SERVICE_ACCOUNT environment variable not set")
            else:
                # Default authentication (user account)
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
                logger.info("GEE initialized with user account")

        except Exception as e:
            logger.error(f"Failed to initialize Google Earth Engine: {e}")
            raise

    def collect_sentinel5p_data(self, product: str, start_date: str, end_date: str,
                              save_to_disk: bool = True) -> Optional[ee.ImageCollection]:
        """
        Collect Sentinel-5P Level 3 data from GEE.

        Args:
            product: Product type (no2, so2, co, o3)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_disk: Whether to export data to disk

        Returns:
            GEE ImageCollection
        """
        collection_id = self.collections.get(product)
        if not collection_id:
            raise ValueError(f"Unknown product: {product}")

        try:
            logger.info(f"Collecting Sentinel-5P {product} data from {start_date} to {end_date}")

            # Get image collection
            collection = ee.ImageCollection(collection_id) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region['polygon'])

            # Apply quality filtering and cloud masking
            collection = self._apply_sentinel5p_quality_filter(collection, product)

            # Clip to region
            collection = collection.map(lambda img: img.clip(self.region['polygon']))

            logger.info(f"Found {collection.size().getInfo()} images in collection")

            if save_to_disk:
                self._export_collection(collection, product, start_date, end_date, 'sentinel5p')

            return collection

        except Exception as e:
            logger.error(f"Failed to collect Sentinel-5P {product} data: {e}")
            return None

    def collect_modis_data(self, product: str, start_date: str, end_date: str,
                          save_to_disk: bool = True) -> Optional[ee.ImageCollection]:
        """
        Collect MODIS data from GEE.

        Args:
            product: MODIS product name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_disk: Whether to export data to disk

        Returns:
            GEE ImageCollection
        """
        collection_id = self.collections.get(product)
        if not collection_id:
            raise ValueError(f"Unknown product: {product}")

        try:
            logger.info(f"Collecting MODIS {product} data from {start_date} to {end_date}")

            # Get image collection
            collection = ee.ImageCollection(collection_id) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region['polygon'])

            # Apply quality filtering based on product type
            collection = self._apply_modis_quality_filter(collection, product)

            # Clip to region
            collection = collection.map(lambda img: img.clip(self.region['polygon']))

            logger.info(f"Found {collection.size().getInfo()} images in collection")

            if save_to_disk:
                self._export_collection(collection, product, start_date, end_date, 'modis')

            return collection

        except Exception as e:
            logger.error(f"Failed to collect MODIS {product} data: {e}")
            return None

    def collect_landsat_data(self, product: str, start_date: str, end_date: str,
                            save_to_disk: bool = True) -> Optional[ee.ImageCollection]:
        """
        Collect Landsat data from GEE.

        Args:
            product: Landsat product name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_disk: Whether to export data to disk

        Returns:
            GEE ImageCollection
        """
        collection_id = self.collections.get(product)
        if not collection_id:
            raise ValueError(f"Unknown product: {product}")

        try:
            logger.info(f"Collecting Landsat {product} data from {start_date} to {end_date}")

            # Get image collection
            collection = ee.ImageCollection(collection_id) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region['polygon']) \
                .filter(ee.Filter.lt('CLOUD_COVER', 20))  # Cloud cover filter

            # Apply cloud masking
            collection = self._apply_landsat_cloud_mask(collection, product)

            # Clip to region
            collection = collection.map(lambda img: img.clip(self.region['polygon']))

            logger.info(f"Found {collection.size().getInfo()} images in collection")

            if save_to_disk:
                self._export_collection(collection, product, start_date, end_date, 'landsat')

            return collection

        except Exception as e:
            logger.error(f"Failed to collect Landsat {product} data: {e}")
            return None

    def extract_time_series(self, collection: ee.ImageCollection, region: Optional[Dict] = None,
                           reducer: str = 'mean', scale: int = 1000) -> pd.DataFrame:
        """
        Extract time series data from image collection.

        Args:
            collection: GEE ImageCollection
            region: Region geometry (uses Islamabad region if None)
            reducer: Reduction method ('mean', 'median', 'min', 'max')
            scale: Scale in meters for spatial reduction

        Returns:
            DataFrame with time series data
        """
        if region is None:
            region = self.region['polygon']

        try:
            # Define reduction function
            if reducer == 'mean':
                ee_reducer = ee.Reducer.mean()
            elif reducer == 'median':
                ee_reducer = ee.Reducer.median()
            elif reducer == 'min':
                ee_reducer = ee.Reducer.min()
            elif reducer == 'max':
                ee_reducer = ee.Reducer.max()
            else:
                raise ValueError(f"Unknown reducer: {reducer}")

            # Map over collection to extract time series
            def extract_time(img):
                date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
                reduced = img.reduceRegion(
                    reducer=ee_reducer,
                    geometry=region,
                    scale=scale,
                    maxPixels=self.max_pixels
                )
                return reduced.set('date', date)

            time_series = collection.map(extract_time)

            # Get the data as a dictionary
            time_dict = time_series.getInfo()

            # Convert to DataFrame
            data = []
            for feature in time_dict['features']:
                row = feature['properties']
                if 'date' in row:
                    data.append(row)

            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                return df
            else:
                logger.warning("No time series data extracted")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to extract time series: {e}")
            return pd.DataFrame()

    def _apply_sentinel5p_quality_filter(self, collection: ee.ImageCollection,
                                       product: str) -> ee.ImageCollection:
        """Apply quality filtering to Sentinel-5P data."""
        def filter_image(img):
            # Quality filtering based on qa_value
            qa = img.select('qa_value')
            good_quality = qa.lt(3)  # Quality flags < 3 are good

            # Apply mask and select relevant band
            if product == 'no2':
                data_band = img.select('tropospheric_NO2_column_number_density')
            elif product == 'so2':
                data_band = img.select('SO2_column_number_density')
            elif product == 'co':
                data_band = img.select('CO_column_number_density')
            elif product == 'o3':
                data_band = img.select('O3_column_number_density')
            else:
                return img

            return data_band.updateMask(good_quality).copyProperties(img)

        return collection.map(filter_image)

    def _apply_modis_quality_filter(self, collection: ee.ImageCollection,
                                   product: str) -> ee.ImageCollection:
        """Apply quality filtering to MODIS data."""
        def filter_image(img):
            # Common MODIS quality filtering
            if 'Deep_Blue_Aerosol_Optical_Depth_550_Land' in img.bandNames().getInfo():
                # MOD04 aerosol product
                qa = img.select('Deep_Blue_AOD_QA')
                good_quality = qa.lt(3)  # Good quality pixels
                aod = img.select('Deep_Blue_Aerosol_Optical_Depth_550_Land')
                return aod.updateMask(good_quality).copyProperties(img)
            else:
                # Fallback: apply cloud mask if available
                if 'QC' in img.bandNames().getInfo():
                    qa = img.select('QC')
                    good_quality = qa.lt(2)
                    return img.updateMask(good_quality)
                return img

        return collection.map(filter_image)

    def _apply_landsat_cloud_mask(self, collection: ee.ImageCollection,
                                 product: str) -> ee.ImageCollection:
        """Apply cloud masking to Landsat data."""
        def mask_clouds(img):
            # Get cloud shadow and cloud bits from QA_PIXEL
            qa = img.select('QA_PIXEL')
            cloud_shadow_bit_mask = 1 << 3
            clouds_bit_mask = 1 << 5
            dilated_cloud_bit_mask = 1 << 1

            # Create cloud mask
            cloud_mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
                        .And(qa.bitwiseAnd(clouds_bit_mask).eq(0)) \
                        .And(qa.bitwiseAnd(dilated_cloud_bit_mask).eq(0))

            # Apply mask to optical bands
            optical_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            masked_img = img.select(optical_bands).updateMask(cloud_mask)

            # Copy properties
            return masked_img.copyProperties(img).copyProperties(img, ['system:time_start'])

        return collection.map(mask_clouds)

    def _export_collection(self, collection: ee.ImageCollection, product: str,
                          start_date: str, end_date: str, source: str) -> None:
        """
        Export image collection to Google Drive or local storage.

        Args:
            collection: ImageCollection to export
            product: Product name
            start_date: Start date
            end_date: End date
            source: Data source (sentinel5p, modis, landsat)
        """
        try:
            # Create output directory
            output_dir = FileUtils.ensure_directory(
                self.raw_data_path / source / product.lower()
            )

            # Get collection size
            collection_size = collection.size().getInfo()
            logger.info(f"Exporting {collection_size} images for {product}")

            if collection_size == 0:
                logger.warning(f"No images to export for {product}")
                return

            # Export as single mosaic (most recent)
            mosaic = collection.mosaic()
            description = f"{source}_{product}_{start_date}_{end_date}"

            # Export to Google Drive
            task = ee.batch.Export.image.toDrive(
                image=mosaic,
                description=description,
                folder='smog_detection',
                fileNamePrefix=description,
                scale=1000,  # 1km resolution
                region=self.region['polygon'].bounds(),
                fileFormat='GeoTIFF',
                maxPixels=self.max_pixels
            )

            task.start()
            logger.info(f"Started export task: {description}")

            # For smaller collections, also export time series
            if collection_size <= 30:
                self._export_time_series(collection, product, start_date, end_date, source)

        except Exception as e:
            logger.error(f"Failed to export collection {product}: {e}")

    def _export_time_series(self, collection: ee.ImageCollection, product: str,
                           start_date: str, end_date: str, source: str) -> None:
        """Export time series data as CSV."""
        try:
            # Extract time series
            ts_df = self.extract_time_series(collection)

            if not ts_df.empty:
                # Save to CSV
                filename = f"{source}_{product}_timeseries_{start_date}_{end_date}.csv"
                output_path = FileUtils.ensure_directory(
                    self.raw_data_path / source / 'timeseries'
                ) / filename

                FileUtils.save_dataframe(ts_df, output_path, format='csv')
                logger.info(f"Saved time series to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export time series for {product}: {e}")

    def get_collection_info(self, collection_id: str) -> Dict[str, Any]:
        """Get information about a GEE collection."""
        try:
            collection = ee.ImageCollection(collection_id)
            info = collection.limit(1).getInfo()

            if 'properties' in info:
                return {
                    'id': collection_id,
                    'title': info['properties'].get('title', 'Unknown'),
                    'description': info['properties'].get('description', ''),
                    'bands': info.get('bands', []),
                    'properties': info['properties']
                }
            else:
                return {'id': collection_id, 'error': 'No info available'}

        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_id}: {e}")
            return {'id': collection_id, 'error': str(e)}

    def get_available_collections(self) -> Dict[str, str]:
        """Get dictionary of available collections."""
        return self.collections.copy()

    def validate_collection_id(self, collection_id: str) -> bool:
        """Validate that a collection ID exists and is accessible."""
        try:
            collection = ee.ImageCollection(collection_id)
            size = collection.size().getInfo()
            return isinstance(size, (int, float))
        except Exception as e:
            logger.error(f"Collection {collection_id} validation failed: {e}")
            return False

    def get_region_statistics(self, collection: ee.ImageCollection,
                            geometry: Optional[Dict] = None) -> Dict[str, Any]:
        """Get statistical summary for collection over region."""
        if geometry is None:
            geometry = self.region['polygon']

        try:
            # Calculate mean image
            mean_image = collection.mean()

            # Get regional statistics
            stats = mean_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=1000,
                maxPixels=self.max_pixels
            ).getInfo()

            # Add collection metadata
            stats.update({
                'collection_size': collection.size().getInfo(),
                'date_range': collection.aggregate_min('system:time_start').getInfo(),
                'region_area_km2': GeoUtils.calculate_polygon_area_km2(self.region['polygon'])
            })

            return stats

        except Exception as e:
            logger.error(f"Failed to get region statistics: {e}")
            return {}

    def check_collection_availability(self, collection_id: str,
                                    start_date: str, end_date: str) -> bool:
        """Check if data is available for collection in date range."""
        try:
            collection = ee.ImageCollection(collection_id) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region['polygon'])

            size = collection.size().getInfo()
            return size > 0

        except Exception as e:
            logger.error(f"Failed to check availability for {collection_id}: {e}")
            return False

    def get_optimal_dates(self, collection_id: str, start_date: str, end_date: str,
                         max_images: int = 10) -> List[str]:
        """Get optimal dates with least cloud cover for collection."""
        try:
            collection = ee.ImageCollection(collection_id) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region['polygon'])

            # If collection has cloud cover property, filter and sort
            if 'CLOUD_COVER' in collection.first().bandNames().getInfo():
                collection = collection.sort('CLOUD_COVER')

            # Get limited number of images
            limited_collection = collection.limit(max_images)

            # Extract dates
            dates = limited_collection.aggregate_array('system:time_start').getInfo()
            date_strings = [datetime.datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in dates]

            return date_strings

        except Exception as e:
            logger.error(f"Failed to get optimal dates for {collection_id}: {e}")
            return []