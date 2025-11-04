"""
Preprocessing module tests for Islamabad Smog Detection System.
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.atmospheric_correction import AtmosphericCorrector
from src.preprocessing.noise_reduction import NoiseReducer
from src.preprocessing.contrast_enhancement import ContrastEnhancer
from src.preprocessing.geospatial_processor import GeospatialProcessor


class TestAtmosphericCorrection(unittest.TestCase):
    """Test cases for atmospheric correction."""

    def setUp(self):
        """Set up test fixtures."""
        self.corrector = AtmosphericCorrector()
        self.sample_data = self._create_sample_image_data()

    def _create_sample_image_data(self):
        """Create sample 2D image data for testing."""
        # Create sample 2D array representing satellite image
        np.random.seed(42)
        data = np.random.random((100, 100)) * 0.0001 + 0.00001

        # Add some atmospheric interference (brighter values in corners)
        data[:10, :10] *= 2  # Top-left corner
        data[-10:, -10:] *= 2  # Bottom-right corner

        # Create xarray dataset
        ds = xr.Dataset(
            data_vars={'no2': (['y', 'x'], data)},
            coords={
                'y': np.linspace(33.0, 34.0, 100),
                'x': np.linspace(72.0, 74.0, 100)
            },
            attrs={'title': 'Sample Image Data'}
        )

        return ds

    def test_dos_correction(self):
        """Test Dark Object Subtraction correction."""
        corrected_dataset = self.corrector.correct_dataset(self.sample_data, methods=['DOS'])

        self.assertIn('no2', corrected_dataset.data_vars)

        # Check that correction was applied
        original_mean = self.sample_data['no2'].mean().values
        corrected_mean = corrected_dataset['no2'].mean().values

        # DOS correction should reduce overall brightness
        self.assertLess(corrected_mean, original_mean)

    def test_haze_removal(self):
        """Test haze removal correction."""
        corrected_dataset = self.corrector.correct_dataset(self.sample_data, methods=['haze_removal'])

        self.assertIn('no2', corrected_dataset.data_vars)

        # Check correction metadata
        self.assertIn('atmospheric_correction', corrected_dataset['no2'].attrs)
        self.assertEqual(corrected_dataset['no2'].attrs['atmospheric_correction'], 'haze_removal')

    def test_combined_correction(self):
        """Test combined atmospheric correction."""
        corrected_dataset = self.corrector.correct_dataset(self.sample_data, methods=['combined'])

        self.assertIn('no2', corrected_dataset.data_vars)
        self.assertIn('atmospheric_correction', corrected_dataset.attrs)
        self.assertIn('combined', corrected_dataset.attrs['atmospheric_correction_methods'])

    def test_correction_metrics(self):
        """Test correction metrics calculation."""
        corrected_dataset = self.corrector.correct_dataset(self.sample_data, methods=['DOS'])
        metrics = self.corrector.calculate_correction_metrics(self.sample_data, corrected_dataset)

        self.assertIn('no2', metrics)
        self.assertIn('snr_improvement', metrics['no2'])
        self.assertIn('contrast_improvement', metrics['no2'])


class TestNoiseReduction(unittest.TestCase):
    """Test cases for noise reduction."""

    def setUp(self):
        """Set up test fixtures."""
        self.reducer = NoiseReducer()
        self.noisy_data = self._create_noisy_data()

    def _create_noisy_data(self):
        """Create noisy data for testing."""
        np.random.seed(42)
        # Create clean signal
        clean_signal = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.0001 + 0.0001

        # Add noise
        noise = np.random.normal(0, 0.00002, 100)
        noisy_signal = clean_signal + noise

        # Create dataset
        ds = xr.Dataset(
            data_vars={'no2': (['time'], noisy_signal)},
            coords={'time': pd.date_range('2023-01-01', periods=100, freq='D')},
            attrs={'title': 'Noisy Time Series'}
        )

        return ds

    def test_gaussian_filter(self):
        """Test Gaussian noise reduction."""
        reduced_dataset = self.reducer.reduce_noise(self.noisy_data, methods=['gaussian'])

        self.assertIn('no2', reduced_dataset.data_vars)
        self.assertTrue(reduced_dataset['no2'].attrs.get('noise_reduction_applied', False))

    def test_median_filter(self):
        """Test median noise reduction."""
        reduced_dataset = self.reducer.reduce_noise(self.noisy_data, methods=['median'])

        self.assertIn('no2', reduced_dataset.data_vars)
        self.assertTrue(reduced_dataset['noisy'].attrs.get('noise_reduction_applied', False))

    def test_bilateral_filter(self):
        """Test bilateral noise reduction."""
        reduced_dataset = self.reducer.reduce_noise(self.noisy_data, methods=['bilateral'])

        self.assertIn('no2', reduced_dataset.data_vars)

    def test_outlier_removal(self):
        """Test outlier removal."""
        cleaned_dataset = self.reducer.remove_outliers(self.noisy_data, method='zscore')

        self.assertIn('no2_anomalies', cleaned_dataset.data_vars)

    def test_noise_metrics(self):
        """Test noise reduction metrics calculation."""
        reduced_dataset = self.reducer.reduce_noise(self.noisy_data, methods=['gaussian'])
        metrics = self.reducer.calculate_noise_metrics(self.noisy_data, reduced_dataset)

        self.assertIn('no2', metrics)
        self.assertIn('snr_improvement', metrics['no2'])


class TestContrastEnhancement(unittest.TestCase):
    """Test cases for contrast enhancement."""

    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = ContrastEnhancer()
        self.low_contrast_data = self._create_low_contrast_data()

    def _create_low_contrast_data(self):
        """Create low contrast image data."""
        # Create data with low contrast (small range)
        np.random.seed(42)
        base_value = 0.0001
        small_variation = np.random.random((50, 50)) * 0.000001  # Very small variation
        data = base_value + small_variation

        ds = xr.Dataset(
            data_vars={'no2': (['y', 'x'], data)},
            coords={
                'y': np.linspace(33.0, 34.0, 50),
                'x': np.linspace(72.0, 74.0, 50)
            },
            attrs={'title': 'Low Contrast Image'}
        )

        return ds

    def test_clahe_enhancement(self):
        """Test CLAHE contrast enhancement."""
        enhanced_dataset = self.enhancer.enhance_contrast(self.low_contrast_data, methods=['clahe'])

        self.assertIn('no2', enhanced_dataset.data_vars)
        self.assertTrue(enhanced_dataset['no2'].attrs.get('contrast_enhancement_applied', False))

    def test_gamma_correction(self):
        """Test gamma correction."""
        enhanced_dataset = self.enhancer.enhance_contrast(self.low_contrast_data, methods=['gamma_correction'])

        self.assertIn('no2', enhanced_dataset.data_vars)
        self.assertIn('gamma_correction', enhanced_dataset['no2'].attrs.get('contrast_enhancement_methods', []))

    def test_histogram_equalization(self):
        """Test histogram equalization."""
        enhanced_dataset = self.enhancer.enhance_contrast(self.low_contrast_data, methods=['histogram_equalization'])

        self.assertIn('no2', enhanced_dataset.data_vars)

    def test_enhancement_metrics(self):
        """Test enhancement metrics calculation."""
        enhanced_dataset = self.enhancer.enhance_contrast(self.low_contrast_data, methods=['clahe'])
        metrics = self.enhancer.calculate_enhancement_metrics(self.low_contrast_data, enhanced_dataset)

        self.assertIn('no2', metrics)
        self.assertIn('contrast_improvement', metrics['no2'])
        self.assertIn('entropy_improvement', metrics['no2'])


class TestGeospatialProcessor(unittest.TestCase):
    """Test cases for geospatial processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = GeospatialProcessor()
        self.spatial_data = self._create_spatial_data()

    def _create_spatial_data(self):
        """Create spatial data for testing."""
        lat = np.linspace(32.5, 35.5, 20)
        lon = np.linspace(71.5, 74.5, 20)

        # Create 2D spatial data
        np.random.seed(42)
        data = np.random.random((len(lat), len(lon))) * 0.0001

        ds = xr.Dataset(
            data_vars={'no2': (['lat', 'lon'], data)},
            coords={'lat': lat, 'lon': lon},
            attrs={'title': 'Spatial Data', 'crs': 'EPSG:4326'}
        )

        return ds

    def test_region_clipping(self):
        """Test clipping to Islamabad region."""
        processed_dataset = self.processor.process_dataset(self.spatial_data, operations=['clip'])

        self.assertIn('no2', processed_dataset.data_vars)
        self.assertTrue(processed_dataset.attrs.get('clipped_to_region', False))

    def test_resampling(self):
        """Test spatial resampling."""
        processed_dataset = self.processor.process_dataset(self.spatial_data, operations=['resample'])

        self.assertIn('no2', processed_dataset.data_vars)

    def test_spatial_statistics(self):
        """Test spatial statistics calculation."""
        stats = self.processor.calculate_spatial_statistics(self.spatial_data)

        self.assertIn('no2', stats)
        self.assertIn('spatial_mean', stats['no2'])
        self.assertIn('spatial_std', stats['no2'])

    def test_dataset_fusion(self):
        """Test dataset fusion."""
        # Create a second dataset for fusion
        np.random.seed(123)
        data2 = np.random.random((len(self.spatial_data.lat), len(self.spatial_data.lon))) * 0.0001

        dataset2 = xr.Dataset(
            data_vars={'so2': (['lat', 'lon'], data2)},
            coords={'lat': self.spatial_data.lat, 'lon': self.spatial_data.lon}
        )

        fused_dataset = self.processor.fuse_datasets([self.spatial_data, dataset2])

        # Check that fusion was attempted
        self.assertIsInstance(fused_dataset, xr.Dataset)

    def test_spatial_consistency_validation(self):
        """Test spatial consistency validation."""
        validation_results = self.processor.validate_spatial_consistency(self.spatial_data)

        self.assertIsInstance(validation_results, dict)
        self.assertIn('latitude_range_valid', validation_results)
        self.assertIn('longitude_range_valid', validation_results)


class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests for preprocessing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.raw_data = self._create_raw_satellite_data()

    def _create_raw_satellite_data(self):
        """Create raw satellite-like data."""
        np.random.seed(42)

        # Create time series with spatial dimensions
        time = pd.date_range('2023-01-01', periods=10, freq='D')
        lat = np.linspace(33.0, 34.0, 20)
        lon = np.linspace(72.0, 74.0, 20)

        # Create data with atmospheric interference and noise
        base_data = np.random.random((len(time), len(lat), len(lon))) * 0.0001

        # Add atmospheric effects (corner brightening)
        base_data[:, :5, :5] *= 1.5
        base_data[:, -5:, -5:] *= 1.5

        # Add noise
        noise = np.random.normal(0, 0.00001, base_data.shape)
        final_data = base_data + noise

        ds = xr.Dataset(
            data_vars={'no2': (['time', 'lat', 'lon'], final_data)},
            coords={'time': time, 'lat': lat, 'lon': lon},
            attrs={'title': 'Raw Satellite Data', 'source': 'Test Satellite'}
        )

        return ds

    def test_complete_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Initialize processors
        atm_corrector = AtmosphericCorrector()
        noise_reducer = NoiseReducer()
        enhancer = ContrastEnhancer()
        geo_processor = GeospatialProcessor()

        # Apply preprocessing steps
        # 1. Atmospheric correction
        corrected = atm_corrector.correct_dataset(self.raw_data, methods=['DOS'])

        # 2. Noise reduction
        denoised = noise_reducer.reduce_noise(corrected, methods=['gaussian'])

        # 3. Contrast enhancement
        enhanced = enhancer.enhance_contrast(denoised, methods=['clahe'])

        # 4. Geospatial processing
        processed = geo_processor.process_dataset(enhanced, operations=['clip', 'resample'])

        # Validate pipeline
        self.assertIn('no2', processed.data_vars)
        self.assertTrue(processed.attrs.get('atmospheric_correction_applied', False))
        self.assertTrue(processed.attrs.get('noise_reduction_applied', False))
        self.assertTrue(processed.attrs.get('contrast_enhancement_applied', False))

    def test_preprocessing_metrics(self):
        """Test preprocessing metrics calculation."""
        # Apply preprocessing and calculate metrics
        atm_corrector = AtmosphericCorrector()
        corrected = atm_corrector.correct_dataset(self.raw_data, methods=['DOS'])
        correction_metrics = atm_corrector.calculate_correction_metrics(self.raw_data, corrected)

        self.assertIn('no2', correction_metrics)
        self.assertIn('snr_improvement', correction_metrics['no2'])

        noise_reducer = NoiseReducer()
        denoised = noise_reducer.reduce_noise(corrected, methods=['gaussian'])
        noise_metrics = noise_reducer.calculate_noise_metrics(self.raw_data, denoised)

        self.assertIn('no2', noise_metrics)
        self.assertIn('noise_reduction_ratio', noise_metrics['no2'])

        enhancer = ContrastEnhancer()
        enhanced = enhancer.enhance_contrast(denoised, methods=['clahe'])
        enhancement_metrics = enhancer.calculate_enhancement_metrics(denoised, enhanced)

        self.assertIn('no2', enhancement_metrics)
        self.assertIn('contrast_improvement', enhancement_metrics['no2'])


if __name__ == '__main__':
    unittest.main()