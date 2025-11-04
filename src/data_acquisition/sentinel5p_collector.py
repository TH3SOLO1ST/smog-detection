"""
Sentinel-5P data collector for Islamabad Smog Detection System.

This module provides functionality to retrieve atmospheric data from the
Copernicus Sentinel-5P satellite (TROPOMI instrument) for the Islamabad region.
Supports NO2, SO2, CO, O3, and aerosol index data acquisition.
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
from urllib.parse import urljoin

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class Sentinel5PCollector:
    """Collects Sentinel-5P atmospheric data via Copernicus Data Space Ecosystem API."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Sentinel-5P data collector.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.api_config = self.config.get_api_config('copernicus')
        self.region_config = self.config.get_section('region')
        self.data_sources = self.config.get_section('data_sources')

        # API endpoints
        self.base_url = self.api_config.get('base_url', 'https://dataspace.copernicus.eu')
        self.auth_url = self.api_config.get('auth_url',
            'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token')
        self.process_api_url = f"{self.base_url}/api/v1/process"

        # Authentication
        self.access_token = None
        self.token_expires_at = None

        # Data collections
        self.collections = self.data_sources.get('sentinel5p', {}).get('collections', {})

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.raw_data_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('raw_data', 'data/raw')) / 'sentinel5p'
        )

        # Initialize region
        self.region = GeoUtils.create_islamabad_region(
            buffer_km=self.region_config.get('buffer_km', 50)
        )

        logger.info("Sentinel-5P collector initialized")

    def _get_access_token(self) -> str:
        """
        Get OAuth2 access token for Copernicus API.

        Returns:
            Access token string
        """
        # Check if current token is still valid
        if (self.access_token and self.token_expires_at and
            datetime.datetime.now() < self.token_expires_at):
            return self.access_token

        # Request new token
        client_id = self.api_config.get('client_id')
        client_secret = self.api_config.get('client_secret')

        if not client_id or not client_secret:
            raise ValueError("Copernicus API credentials not configured")

        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }

        try:
            response = requests.post(self.auth_url, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 3600)
            self.token_expires_at = datetime.datetime.now() + datetime.timedelta(seconds=expires_in - 300)  # Refresh 5 min early

            logger.info("Successfully obtained Copernicus access token")
            return self.access_token

        except requests.RequestException as e:
            logger.error(f"Failed to get access token: {e}")
            raise

    def _create_evalscript(self, product: str) -> str:
        """
        Create evalscript for processing Sentinel-5P data.

        Args:
            product: Product type (no2, so2, co, o3, aerosol)

        Returns:
            Evalscript string
        """
        evalscripts = {
            'no2': """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands: ["NO2", "qa_value"],
                            units: ["mol m-2", "DN"]
                        }],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    var qa = sample.qa_value;
                    var no2 = sample.NO2;

                    // Quality filtering: use pixels with good quality (qa_value < 3)
                    if (qa >= 3) {
                        return [NaN];
                    }

                    return [no2];
                }
            """,
            'so2': """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands: ["SO2", "qa_value"],
                            units: ["mol m-2", "DN"]
                        }],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    var qa = sample.qa_value;
                    var so2 = sample.SO2;

                    // Quality filtering
                    if (qa >= 3) {
                        return [NaN];
                    }

                    return [so2];
                }
            """,
            'co': """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands: ["CO", "qa_value"],
                            units: ["mol m-2", "DN"]
                        }],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    var qa = sample.qa_value;
                    var co = sample.CO;

                    // Quality filtering
                    if (qa >= 3) {
                        return [NaN];
                    }

                    return [co];
                }
            """,
            'o3': """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands: ["O3", "qa_value"],
                            units: ["mol m-2", "DN"]
                        }],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    var qa = sample.qa_value;
                    var o3 = sample.O3;

                    // Quality filtering
                    if (qa >= 3) {
                        return [NaN];
                    }

                    return [o3];
                }
            """,
            'aerosol': """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands: ["aerosol_index_354_388", "qa_value"],
                            units: ["DN", "DN"]
                        }],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    var qa = sample.qa_value;
                    var ai = sample.aerosol_index_354_388;

                    // Quality filtering
                    if (qa >= 3) {
                        return [NaN];
                    }

                    return [ai];
                }
            """
        }

        if product not in evalscripts:
            raise ValueError(f"Unsupported product type: {product}")

        return evalscripts[product]

    def _make_request_with_retry(self, request_func, max_retries: int = 3,
                                backoff_factor: float = 2.0) -> requests.Response:
        """
        Make HTTP request with retry logic.

        Args:
            request_func: Function that makes the request
            max_retries: Maximum number of retries
            backoff_factor: Backoff multiplier for retry delays

        Returns:
            Response object
        """
        for attempt in range(max_retries + 1):
            try:
                response = request_func()
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == max_retries:
                    logger.error(f"Request failed after {max_retries} retries: {e}")
                    raise

                wait_time = backoff_factor ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}), "
                             f"retrying in {wait_time} seconds: {e}")
                time.sleep(wait_time)

    def _create_request_body(self, product: str, start_date: str, end_date: str) -> Dict:
        """
        Create request body for Sentinel Hub Process API.

        Args:
            product: Product type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Request body dictionary
        """
        bbox = self.region['bounding_box']

        return {
            "input": {
                "bounds": {
                    "bbox": [bbox['west'], bbox['south'], bbox['east'], bbox['north']]
                },
                "data": [{
                    "type": "sentinel-5p",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z"
                        },
                        "mosaickingOrder": "mostRecent"
                    }
                }]
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }]
            },
            "evalscript": self._create_evalscript(product)
        }

    def collect_data(self, product: str, start_date: str, end_date: Optional[str] = None,
                    save_to_disk: bool = True) -> Optional[xr.Dataset]:
        """
        Collect Sentinel-5P data for specified product and date range.

        Args:
            product: Product type (no2, so2, co, o3, aerosol)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD, defaults to start_date)
            save_to_disk: Whether to save data to disk

        Returns:
            xarray Dataset with collected data
        """
        if product not in self.collections:
            raise ValueError(f"Unsupported product: {product}. "
                           f"Available: {list(self.collections.keys())}")

        if end_date is None:
            end_date = start_date

        try:
            # Get access token
            token = self._get_access_token()

            # Create request body
            request_body = self._create_request_body(product, start_date, end_date)

            # Make request
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            def make_request():
                return requests.post(self.process_api_url,
                                   headers=headers,
                                   json=request_body,
                                   timeout=300)

            response = self._make_request_with_retry(make_request)

            # Save raw data if requested
            if save_to_disk:
                output_path = self._get_output_path(product, start_date, end_date)
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Saved {product} data to {output_path}")

                # Load and return as xarray Dataset
                return self._load_data(output_path, product, start_date, end_date)

            else:
                # Load data from response content
                return self._load_data_from_content(response.content, product, start_date, end_date)

        except Exception as e:
            logger.error(f"Failed to collect {product} data for {start_date}-{end_date}: {e}")
            return None

    def collect_time_series(self, product: str, start_date: str, end_date: str,
                           save_to_disk: bool = True) -> List[xr.Dataset]:
        """
        Collect time series data for multiple dates.

        Args:
            product: Product type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_disk: Whether to save data to disk

        Returns:
            List of xarray Datasets for each date
        """
        datasets = []
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')

            logger.info(f"Collecting {product} data for {date_str}")
            dataset = self.collect_data(product, date_str, date_str, save_to_disk)

            if dataset is not None:
                datasets.append(dataset)

            current_dt += datetime.timedelta(days=1)

            # Rate limiting to avoid API quota issues
            time.sleep(1)

        logger.info(f"Collected {len(datasets)} {product} datasets from {start_date} to {end_date}")
        return datasets

    def _get_output_path(self, product: str, start_date: str, end_date: str) -> Path:
        """
        Generate output file path for collected data.

        Args:
            product: Product type
            start_date: Start date
            end_date: End date

        Returns:
            Output file path
        """
        # Create product subdirectory
        product_dir = FileUtils.ensure_directory(self.raw_data_path / product)

        # Create year subdirectory
        year = start_date[:4]
        year_dir = FileUtils.ensure_directory(product_dir / year)

        # Create month subdirectory
        month = start_date[5:7]
        month_dir = FileUtils.ensure_directory(year_dir / month)

        # Generate filename
        if start_date == end_date:
            filename = f"s5p_{product}_{start_date}.tif"
        else:
            filename = f"s5p_{product}_{start_date}_{end_date}.tif"

        return month_dir / filename

    def _load_data(self, file_path: Path, product: str, start_date: str, end_date: str) -> xr.Dataset:
        """
        Load data from file into xarray Dataset.

        Args:
            file_path: Path to data file
            product: Product type
            start_date: Start date
            end_date: End date

        Returns:
            xarray Dataset
        """
        try:
            # Load GeoTIFF data
            ds = xr.open_dataset(file_path, engine='rasterio')

            # Rename bands and add metadata
            if 'band_data' in ds:
                ds = ds.rename({'band_data': product})

            # Add coordinate information
            bbox = self.region['bounding_box']

            # Create spatial coordinates
            if 'x' in ds.coords and 'y' in ds.coords:
                # Use existing coordinates
                pass
            else:
                # Create coordinate arrays
                lon = np.linspace(bbox['west'], bbox['east'], ds.sizes.get('x', 512))
                lat = np.linspace(bbox['south'], bbox['north'], ds.sizes.get('y', 512))
                ds = ds.assign_coords({
                    'lon': ('x', lon),
                    'lat': ('y', lat)
                })

            # Add time coordinate
            time_coord = pd.to_datetime(start_date)
            ds = ds.expand_dims({'time': [time_coord]})

            # Add attributes
            ds[product].attrs.update({
                'long_name': f'Sentinel-5P {product.upper()} concentration',
                'units': self._get_product_units(product),
                'source': 'Copernicus Sentinel-5P/TROPOMI',
                'processing_level': 'Level 2'
            })

            ds.attrs.update({
                'title': f'Sentinel-5P {product.upper()} data for Islamabad',
                'region': 'Islamabad, Pakistan',
                'bbox': bbox,
                'collection': self.collections.get(product),
                'date_range': f"{start_date} to {end_date}"
            })

            return ds

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def _load_data_from_content(self, content: bytes, product: str,
                               start_date: str, end_date: str) -> xr.Dataset:
        """
        Load data from response content into xarray Dataset.

        Args:
            content: Response content bytes
            product: Product type
            start_date: Start date
            end_date: End date

        Returns:
            xarray Dataset
        """
        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        try:
            return self._load_data(tmp_path, product, start_date, end_date)
        finally:
            # Clean up temporary file
            tmp_path.unlink()

    def _get_product_units(self, product: str) -> str:
        """Get units for specific product."""
        units = {
            'no2': 'mol m-2',
            'so2': 'mol m-2',
            'co': 'mol m-2',
            'o3': 'mol m-2',
            'aerosol': 'dimensionless'
        }
        return units.get(product, 'unknown')

    def get_available_products(self) -> List[str]:
        """
        Get list of available Sentinel-5P products.

        Returns:
            List of product names
        """
        return list(self.collections.keys())

    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Validate date range for data collection.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if valid
        """
        try:
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

            # Sentinel-5P was launched in October 2017
            sentinel5p_start = datetime.datetime(2017, 10, 13)

            if start_dt < sentinel5p_start:
                logger.warning(f"Start date {start_date} is before Sentinel-5P launch")
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

    def get_collection_info(self, product: str) -> Dict[str, Any]:
        """
        Get information about a specific collection.

        Args:
            product: Product name

        Returns:
            Collection information
        """
        if product not in self.collections:
            raise ValueError(f"Unknown product: {product}")

        return {
            'name': product,
            'collection_id': self.collections[product],
            'spatial_resolution': 7000,  # 7km for most products
            'temporal_resolution': 'daily',
            'units': self._get_product_units(product),
            'description': f'Sentinel-5P {product.upper()} atmospheric measurements'
        }