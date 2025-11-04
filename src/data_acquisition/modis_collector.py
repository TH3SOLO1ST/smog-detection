"""
NASA MODIS data collector for Islamabad Smog Detection System.

This module provides functionality to retrieve atmospheric and aerosol data
from NASA's MODIS sensors (Terra and Aqua satellites) via FIRMS API and
Earthdata for the Islamabad region.
"""

import os
import requests
import time
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import xarray as xr
import numpy as np
import pandas as pd
import h5py
import netCDF4 as nc
from urllib.parse import urljoin
import tempfile

try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("earthaccess not available, some features may be limited")

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class MODISCollector:
    """Collects MODIS atmospheric and aerosol data via NASA APIs."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MODIS data collector.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.api_config = self.config.get_api_config('nasa')
        self.region_config = self.config.get_section('region')
        self.data_sources = self.config.get_section('data_sources')

        # FIRMS API configuration
        self.firms_api_url = self.api_config.get('firms_api_url',
            'https://firms.modaps.eosdis.nasa.gov/api/area/json')

        # Authentication for Earthdata
        self.earthdata_username = self.api_config.get('earthdata_username')
        self.earthdata_password = self.api_config.get('earthdata_password')

        # Data products
        self.products = self.data_sources.get('modis', {}).get('products', {})
        self.country_code = self.data_sources.get('modis', {}).get('country_code', 'PER')

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.raw_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('raw_data', 'data/raw')) / 'modis'
        )

        # Initialize region
        self.region = GeoUtils.create_islamabad_region(
            buffer_km=self.region_config.get('buffer_km', 50)
        )

        # Initialize earthaccess if available
        if EARTHACCESS_AVAILABLE and self.earthdata_username and self.earthdata_password:
            try:
                earthaccess.login(strategy='userpass',
                                username=self.earthdata_username,
                                password=self.earthdata_password)
                logger.info("Successfully logged into NASA Earthdata")
            except Exception as e:
                logger.warning(f"Failed to login to NASA Earthdata: {e}")

        logger.info("MODIS collector initialized")

    def collect_firms_data(self, product: str, start_date: str, end_date: Optional[str] = None,
                          save_to_disk: bool = True) -> Optional[pd.DataFrame]:
        """
        Collect data from NASA FIRMS API.

        Args:
            product: FIRMS product (e.g., 'MODIS_AF_MOD04_L2')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD, defaults to start_date)
            save_to_disk: Whether to save data to disk

        Returns:
            DataFrame with collected data
        """
        if end_date is None:
            end_date = start_date

        try:
            # FIRMS API request parameters
            params = {
                'area': self.country_code,  # Pakistan
                'product': product,
                'date': start_date if start_date == end_date else f"{start_date}/{end_date}",
                'format': 'json'
            }

            logger.info(f"Requesting FIRMS data for {product} from {start_date} to {end_date}")

            # Make request with retry logic
            response = self._make_firms_request(params)

            if response.status_code != 200:
                logger.error(f"FIRS API returned status {response.status_code}")
                return None

            # Parse JSON response
            data = response.json()

            if not data:
                logger.warning(f"No data returned from FIRMS for {product} on {start_date}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Process and clean data
            df = self._process_firms_data(df, product)

            # Save to disk if requested
            if save_to_disk:
                output_path = self._get_firms_output_path(product, start_date, end_date)
                FileUtils.save_dataframe(df, output_path, format='csv', index=False)
                logger.info(f"Saved FIRMS {product} data to {output_path}")

            return df

        except Exception as e:
            logger.error(f"Failed to collect FIRMS {product} data: {e}")
            return None

    def collect_earthaccess_data(self, product: str, start_date: str, end_date: str,
                                save_to_disk: bool = True) -> List[xr.Dataset]:
        """
        Collect MODIS data using earthaccess library.

        Args:
            product: MODIS product name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_disk: Whether to save data to disk

        Returns:
            List of xarray Datasets
        """
        if not EARTHACCESS_AVAILABLE:
            raise ImportError("earthaccess library not available")

        try:
            # Get collection ID for product
            collection_id = self.products.get(product)
            if not collection_id:
                raise ValueError(f"Unknown product: {product}")

            logger.info(f"Searching for {product} granules from {start_date} to {end_date}")

            # Search for granules
            results = earthaccess.search_data(
                short_name=product.split('/')[-1],  # Remove collection prefix
                temporal=(start_date, end_date),
                bounding_box=(
                    self.region['bounding_box']['west'],
                    self.region['bounding_box']['south'],
                    self.region['bounding_box']['east'],
                    self.region['bounding_box']['north']
                ),
                count=100
            )

            if not results:
                logger.warning(f"No granules found for {product}")
                return []

            logger.info(f"Found {len(results)} granules for {product}")

            # Download and process granules
            datasets = []
            for i, granule in enumerate(results):
                try:
                    logger.info(f"Processing granule {i+1}/{len(results)}: {granule['title']}")

                    # Download granule data
                    data_files = earthaccess.download([granule], local_path=self.raw_data_path)

                    # Process downloaded files
                    for data_file in data_files:
                        dataset = self._process_modis_granule(data_file, product, start_date, end_date)
                        if dataset is not None:
                            datasets.append(dataset)

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Failed to process granule {granule['title']}: {e}")
                    continue

            if save_to_disk:
                # Save processed datasets
                self._save_datasets(datasets, product, start_date, end_date)

            return datasets

        except Exception as e:
            logger.error(f"Failed to collect earthaccess data for {product}: {e}")
            return []

    def _make_firms_request(self, params: Dict, max_retries: int = 3) -> requests.Response:
        """
        Make FIRMS API request with retry logic.

        Args:
            params: Request parameters
            max_retries: Maximum number of retries

        Returns:
            Response object
        """
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(self.firms_api_url, params=params, timeout=60)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()

            except requests.RequestException as e:
                if attempt == max_retries:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

        raise Exception("Failed to complete FIRMS request")

    def _process_firms_data(self, df: pd.DataFrame, product: str) -> pd.DataFrame:
        """
        Process and clean FIRMS data.

        Args:
            df: Raw DataFrame from FIRMS
            product: Product name

        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df

        # Convert timestamp
        if 'acq_date' in df.columns and 'acq_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H:%M:%S')

        # Convert coordinates to numeric
        coordinate_columns = ['latitude', 'longitude']
        for col in coordinate_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter to Islamabad region
        bbox = self.region['bounding_box']
        if 'latitude' in df.columns and 'longitude' in df.columns:
            mask = (
                (df['latitude'] >= bbox['south']) &
                (df['latitude'] <= bbox['north']) &
                (df['longitude'] >= bbox['west']) &
                (df['longitude'] <= bbox['east'])
            )
            df = df[mask].copy()

        # Add product metadata
        df['product'] = product
        df['region'] = 'Pakistan'

        # Sort by datetime
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)

        return df

    def _process_modis_granule(self, file_path: Path, product: str,
                              start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """
        Process individual MODIS granule file.

        Args:
            file_path: Path to granule file
            product: Product name
            start_date: Start date
            end_date: End date

        Returns:
            Processed xarray Dataset
        """
        try:
            # Determine file type and open accordingly
            if file_path.suffix.lower() in ['.hdf', '.h4']:
                return self._process_hdf_file(file_path, product, start_date, end_date)
            elif file_path.suffix.lower() in ['.nc']:
                return self._process_netcdf_file(file_path, product, start_date, end_date)
            elif file_path.suffix.lower() in ['.he5']:
                return self._process_he5_file(file_path, product, start_date, end_date)
            else:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to process granule {file_path}: {e}")
            return None

    def _process_hdf_file(self, file_path: Path, product: str,
                         start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Process HDF format MODIS file."""
        try:
            # Open HDF file using h5py
            with h5py.File(file_path, 'r') as hdf_file:
                # Extract data based on product type
                if 'MOD04' in product or 'MYD04' in product:  # Aerosol products
                    return self._extract_aerosol_data(hdf_file, file_path, product, start_date, end_date)
                else:
                    logger.warning(f"Unsupported HDF product: {product}")
                    return None

        except Exception as e:
            logger.error(f"Failed to process HDF file {file_path}: {e}")
            return None

    def _process_netcdf_file(self, file_path: Path, product: str,
                           start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Process NetCDF format MODIS file."""
        try:
            # Open NetCDF file
            ds = xr.open_dataset(file_path)

            # Clip to Islamabad region
            bbox = self.region['bounding_box']
            if 'lon' in ds.coords and 'lat' in ds.coords:
                ds = ds.sel(
                    lon=slice(bbox['west'], bbox['east']),
                    lat=slice(bbox['south'], bbox['north'])
                )

            # Add metadata
            ds.attrs.update({
                'title': f'MODIS {product} data for Islamabad',
                'region': 'Islamabad, Pakistan',
                'bbox': bbox,
                'date_range': f"{start_date} to {end_date}",
                'source_file': str(file_path)
            })

            return ds

        except Exception as e:
            logger.error(f"Failed to process NetCDF file {file_path}: {e}")
            return None

    def _process_he5_file(self, file_path: Path, product: str,
                         start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Process HDF-EOS5 format MODIS file."""
        try:
            # Open HE5 file
            ds = xr.open_dataset(file_path, engine='netcdf4')

            # Extract relevant data variables
            # This would need to be customized based on specific product structure
            if 'AOD_550_Dark_Target_Deep_Blue_Combined_Mean' in ds:
                # MOD04 aerosol product
                data_var = 'AOD_550_Dark_Target_Deep_Blue_Combined_Mean'
            elif 'AOD_550_Dark_Target_Deep_Blue_Combined' in ds:
                data_var = 'AOD_550_Dark_Target_Deep_Blue_Combined'
            else:
                # Find first numeric data variable
                data_vars = [var for var in ds.data_vars if ds[var].dtype.kind in 'biufc']
                if data_vars:
                    data_var = data_vars[0]
                else:
                    logger.warning(f"No suitable data variable found in {file_path}")
                    return None

            # Create new dataset with selected variable
            processed_ds = ds[[data_var]].copy()

            # Clip to Islamabad region
            bbox = self.region['bounding_box']
            if 'lon' in processed_ds.coords and 'lat' in processed_ds.coords:
                processed_ds = processed_ds.sel(
                    lon=slice(bbox['west'], bbox['east']),
                    lat=slice(bbox['south'], bbox['north'])
                )

            # Add metadata
            processed_ds.attrs.update({
                'title': f'MODIS {product} data for Islamabad',
                'region': 'Islamabad, Pakistan',
                'bbox': bbox,
                'date_range': f"{start_date} to {end_date}",
                'source_file': str(file_path),
                'product': product
            })

            return processed_ds

        except Exception as e:
            logger.error(f"Failed to process HE5 file {file_path}: {e}")
            return None

    def _extract_aerosol_data(self, hdf_file, file_path: Path, product: str,
                             start_date: str, end_date: str) -> Optional[xr.Dataset]:
        """Extract aerosol optical depth data from HDF file."""
        try:
            # This is a simplified extraction - actual implementation would need
            # to handle specific HDF structure for each product
            data_dict = {}

            # Look for aerosol-related datasets
            for key in hdf_file.keys():
                if 'AOD' in key or 'aerosol' in key.lower() or 'Optical_Depth' in key:
                    dataset = hdf_file[key]
                    if len(dataset.shape) >= 2:
                        data_dict[key] = dataset[:]

            if not data_dict:
                logger.warning(f"No aerosol data found in {file_path}")
                return None

            # Create coordinate arrays (simplified)
            bbox = self.region['bounding_box']
            first_dataset = list(data_dict.values())[0]
            ny, nx = first_dataset.shape[:2]

            lon = np.linspace(bbox['west'], bbox['east'], nx)
            lat = np.linspace(bbox['south'], bbox['north'], ny)

            # Create xarray dataset
            coords = {'lon': lon, 'lat': lat}
            ds = xr.Dataset(data_dict, coords=coords)

            # Add metadata
            ds.attrs.update({
                'title': f'MODIS {product} aerosol data for Islamabad',
                'region': 'Islamabad, Pakistan',
                'bbox': bbox,
                'date_range': f"{start_date} to {end_date}",
                'source_file': str(file_path),
                'product': product
            })

            return ds

        except Exception as e:
            logger.error(f"Failed to extract aerosol data from {file_path}: {e}")
            return None

    def _get_firms_output_path(self, product: str, start_date: str, end_date: str) -> Path:
        """Generate output path for FIRMS data."""
        product_dir = FileUtils.ensure_directory(self.raw_data_path / 'firms')

        if start_date == end_date:
            year = start_date[:4]
            month = start_date[5:7]
            filename = f"firms_{product}_{start_date}.csv"
        else:
            year = start_date[:4]
            month = start_date[5:7]
            filename = f"firms_{product}_{start_date}_{end_date}.csv"

        year_dir = FileUtils.ensure_directory(product_dir / year)
        return year_dir / month / filename

    def _save_datasets(self, datasets: List[xr.Dataset], product: str,
                      start_date: str, end_date: str) -> None:
        """Save processed datasets to disk."""
        if not datasets:
            return

        product_dir = FileUtils.ensure_directory(self.raw_data_path / product.lower())
        year = start_date[:4]
        year_dir = FileUtils.ensure_directory(product_dir / year)

        for i, dataset in enumerate(datasets):
            # Generate filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{product}_{start_date}_{timestamp}_{i+1}.nc"

            output_path = year_dir / filename

            # Save as NetCDF
            dataset.to_netcdf(output_path)
            logger.info(f"Saved dataset to {output_path}")

    def get_available_products(self) -> List[str]:
        """Get list of available MODIS products."""
        return list(self.products.keys())

    def collect_time_series(self, product: str, start_date: str, end_date: str,
                           method: str = 'firms', save_to_disk: bool = True) -> List[Any]:
        """
        Collect time series data for multiple dates.

        Args:
            product: Product name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            method: Collection method ('firms' or 'earthaccess')
            save_to_disk: Whether to save data to disk

        Returns:
            List of collected data (DataFrames for FIRMS, Datasets for earthaccess)
        """
        if method == 'firms':
            return self._collect_firms_time_series(product, start_date, end_date, save_to_disk)
        elif method == 'earthaccess':
            return self.collect_earthaccess_data(product, start_date, end_date, save_to_disk)
        else:
            raise ValueError(f"Unknown collection method: {method}")

    def _collect_firms_time_series(self, product: str, start_date: str, end_date: str,
                                  save_to_disk: bool = True) -> List[pd.DataFrame]:
        """Collect FIRMS time series data."""
        datasets = []
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')

            logger.info(f"Collecting FIRMS {product} data for {date_str}")
            dataset = self.collect_firms_data(product, date_str, date_str, save_to_disk)

            if dataset is not None and not dataset.empty:
                datasets.append(dataset)

            current_dt += datetime.timedelta(days=1)

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"Collected {len(datasets)} FIRMS {product} datasets from {start_date} to {end_date}")
        return datasets

    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate date range for data collection."""
        try:
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

            # MODIS Terra launched in 1999, Aqua in 2002
            modis_start = datetime.datetime(2000, 1, 1)

            if start_dt < modis_start:
                logger.warning(f"Start date {start_date} is before MODIS era")
                return False

            if start_dt > end_dt:
                logger.error("Start date must be before end date")
                return False

            # Don't allow dates too far in the future
            future_limit = datetime.datetime.now() + datetime.timedelta(days=1)
            if end_dt > future_limit:
                logger.warning(f"End date {end_date} is in the future")
                return False

            return True

        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return False

    def get_product_info(self, product: str) -> Dict[str, Any]:
        """Get information about a specific MODIS product."""
        if product not in self.products:
            raise ValueError(f"Unknown product: {product}")

        return {
            'name': product,
            'collection_id': self.products[product],
            'spatial_resolution': self._get_product_resolution(product),
            'temporal_resolution': 'daily',
            'description': f'MODIS {product} atmospheric measurements'
        }

    def _get_product_resolution(self, product: str) -> str:
        """Get spatial resolution for product."""
        if 'MOD04' in product or 'MYD04' in product:
            return '10km'
        elif 'MOD08' in product:
            return '1degree'
        elif 'MOD13' in product:
            return '250m'
        else:
            return 'unknown'