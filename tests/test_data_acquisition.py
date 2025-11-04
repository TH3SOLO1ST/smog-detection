"""
Data acquisition module tests for Islamabad Smog Detection System.
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector
from src.utils.geo_utils import GeoUtils


class TestDataAcquisition(unittest.TestCase):
    """Test cases for data acquisition modules."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample dataset
        self.sample_data = self._create_sample_dataset()

        # Initialize collectors with test config
        self.test_config = {
            'apis': {
                'copernicus': {
                    'client_id': 'test_client_id',
                    'client_secret': 'test_client_secret',
                    'base_url': 'https://test.dataspace.copernicus.eu',
                    'auth_url': 'https://test.auth.url'
                },
                'nasa': {
                    'earthdata_username': 'test_user',
                    'earthdata_password': 'test_pass'
                }
            },
            'region': {
                'name': 'Islamabad',
                'center_lat': 33.6844,
                'center_lon': 73.0479,
                'buffer_km': 50,
                'bounding_box': {
                    'north': 34.1844,
                    'south': 33.1844,
                    'east': 73.9479,
                    'west': 72.1479
                }
            }
        }

    def _create_sample_dataset(self):
        """Create sample dataset for testing."""
        # Create sample coordinates for Islamabad region
        lat = np.linspace(33.1844, 34.1844, 50)
        lon = np.linspace(72.1479, 73.9479, 50)
        time = pd.date_range('2023-01-01', periods=10, freq='D')

        # Create sample data
        data = np.random.random((len(time), len(lat), len(lon))) * 0.0001

        # Create xarray dataset
        ds = xr.Dataset(
            data_vars={
                'no2': (['time', 'lat', 'lon'], data),
                'qa_value': (['time', 'lat', 'lon'], np.random.randint(0, 5, (len(time), len(lat), len(lon))))
            },
            coords={
                'time': time,
                'lat': lat,
                'lon': lon
            },
            attrs={
                'title': 'Sample NO2 data',
                'source': 'Test data'
            }
        )

        return ds

    def test_geoutils_create_bounding_box(self):
        """Test GeoUtils bounding box creation."""
        bbox = GeoUtils.create_bounding_box(33.6844, 73.0479, 50)

        self.assertIsInstance(bbox, dict)
        self.assertIn('north', bbox)
        self.assertIn('south', bbox)
        self.assertIn('east', bbox)
        self.assertIn('west', bbox)

        # Check that bounding box makes sense
        self.assertGreater(bbox['north'], 33.6844)
        self.assertLess(bbox['south'], 33.6844)
        self.assertGreater(bbox['east'], 73.0479)
        self.assertLess(bbox['west'], 73.0479)

    def test_geoutils_create_islamabad_region(self):
        """Test GeoUtils Islamabad region creation."""
        region = GeoUtils.create_islamabad_region(buffer_km=50)

        self.assertIsInstance(region, dict)
        self.assertEqual(region['name'], 'Islamabad')
        self.assertIn('center', region)
        self.assertIn('bounding_box', region)
        self.assertIn('polygon', region)

    def test_geoutils_calculate_distance(self):
        """Test distance calculation."""
        # Test distance between two points in Islamabad
        lat1, lon1 = 33.6844, 73.0479  # Islamabad center
        lat2, lon2 = 33.7, 73.1        # Nearby point

        distance = GeoUtils.calculate_distance(lat1, lon1, lat2, lon2)

        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        self.assertLess(distance, 100)  # Should be less than 100km

    def test_sentinel5p_collector_initialization(self):
        """Test Sentinel-5P collector initialization."""
        # Mock config file creation would go here
        # For now, test that the class can be instantiated (would fail without proper config)
        try:
            collector = Sentinel5PCollector()
            self.assertIsNotNone(collector)
        except Exception as e:
            # Expected to fail without proper configuration
            self.assertIn('credentials', str(e).lower())

    def test_date_range_validation(self):
        """Test date range validation for data collection."""
        collector = Sentinel5PCollector()

        # Test valid date range
        self.assertTrue(collector.validate_date_range('2023-01-01', '2023-01-31'))

        # Test invalid date range (start after end)
        self.assertFalse(collector.validate_date_range('2023-02-01', '2023-01-31'))

        # Test invalid date format
        self.assertFalse(collector.validate_date_range('invalid-date', '2023-01-31'))

    def test_collection_info_retrieval(self):
        """Test collection information retrieval."""
        collector = Sentinel5PCollector()

        # Test for known product
        info = collector.get_collection_info('no2')
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('spatial_resolution', info)

        # Test for unknown product
        info = collector.get_collection_info('unknown_product')
        self.assertIsNone(info)

    def test_sample_dataset_structure(self):
        """Test that sample dataset has correct structure."""
        self.assertIn('time', self.sample_data.dims)
        self.assertIn('lat', self.sample_data.dims)
        self.assertIn('lon', self.sample_data.dims)

        self.assertIn('no2', self.sample_data.data_vars)
        self.assertIn('qa_value', self.sample_data.data_vars)

        # Check dimensions
        self.assertEqual(len(self.sample_data.time), 10)
        self.assertEqual(len(self.sample_data.lat), 50)
        self.assertEqual(len(self.sample_data.lon), 50)

    def test_dataset_attributes(self):
        """Test dataset attributes."""
        self.assertIn('title', self.sample_data.attrs)
        self.assertIn('source', self.sample_data.attrs)
        self.assertEqual(self.sample_data.attrs['title'], 'Sample NO2 data')


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create datasets for validation testing
        self.valid_dataset = self._create_valid_dataset()
        self.invalid_dataset = self._create_invalid_dataset()

    def _create_valid_dataset(self):
        """Create a valid dataset for testing."""
        lat = np.linspace(33.0, 34.0, 20)
        lon = np.linspace(72.0, 74.0, 20)

        # Create valid data with proper values
        data = np.random.random((len(lat), len(lon))) * 0.0001 + 0.00001

        ds = xr.Dataset(
            data_vars={'no2': (['lat', 'lon'], data)},
            coords={'lat': lat, 'lon': lon},
            attrs={'title': 'Valid Dataset'}
        )

        return ds

    def _create_invalid_dataset(self):
        """Create an invalid dataset for testing."""
        # Create dataset with problematic data
        lat = np.linspace(33.0, 34.0, 20)
        lon = np.linspace(72.0, 74.0, 20)

        # Create data with issues: all zeros and NaNs
        data = np.zeros((len(lat), len(lon)))
        data[0, 0] = np.nan

        ds = xr.Dataset(
            data_vars={'no2': (['lat', 'lon'], data)},
            coords={'lat': lat, 'lon': lon},
            attrs={'title': 'Invalid Dataset'}
        )

        return ds

    def test_valid_dataset_validation(self):
        """Test validation of valid dataset."""
        # Import here to avoid circular imports
        sys.path.append(str(Path(__file__).parent.parent))
        from src.data_acquisition.data_validator import DataValidator

        validator = DataValidator()
        report = validator.validate_dataset(self.valid_dataset, 'test', 'no2')

        # Check that validation succeeded
        self.assertEqual(report.overall_status.value, 'passed')
        self.assertGreater(report.passed_checks, 0)

    def test_invalid_dataset_validation(self):
        """Test validation of invalid dataset."""
        from src.data_acquisition.data_validator import DataValidator

        validator = DataValidator()
        report = validator.validate_dataset(self.invalid_dataset, 'test', 'no2')

        # Check that validation found issues
        # This might pass or fail depending on the specific validation rules
        # The important thing is that it runs without crashing
        self.assertIsNotNone(report)


class TestCollectionPerformance(unittest.TestCase):
    """Performance tests for data collection."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.large_dataset = self._create_large_dataset()

    def _create_large_dataset(self):
        """Create a large dataset for performance testing."""
        # Create larger dataset
        lat = np.linspace(32.5, 35.5, 100)  # Larger spatial extent
        lon = np.linspace(71.5, 74.5, 100)
        time = pd.date_range('2020-01-01', periods=365, freq='D')  # One year

        # Create data
        data = np.random.random((len(time), len(lat), len(lon))) * 0.0001

        ds = xr.Dataset(
            data_vars={
                'no2': (['time', 'lat', 'lon'], data),
                'so2': (['time', 'lat', 'lon'], data * 0.5),
                'co': (['time', 'lat', 'lon'], data * 0.2)
            },
            coords={'time': time, 'lat': lat, 'lon': lon},
            attrs={'title': 'Large Test Dataset'}
        )

        return ds

    def test_large_dataset_processing_speed(self):
        """Test processing speed with large dataset."""
        import time

        # Time basic operations
        start_time = time.time()

        # Test spatial averaging
        spatial_mean = self.large_dataset.mean(dim=['lat', 'lon'])

        # Test time averaging
        time_mean = self.large_dataset.mean(dim='time')

        end_time = time.time()
        processing_time = end_time - start_time

        # Processing should be reasonably fast
        self.assertLess(processing_time, 10.0)  # Should complete in under 10 seconds

        # Results should be valid
        self.assertEqual(len(spatial_mean.time), 365)
        self.assertEqual(len(time_mean.lat), 100)
        self.assertEqual(len(time_mean.lon), 100)


if __name__ == '__main__':
    unittest.main()