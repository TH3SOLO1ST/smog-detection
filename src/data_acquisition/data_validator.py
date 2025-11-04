"""
Data validator for Islamabad Smog Detection System.

This module provides comprehensive data validation functionality for
satellite data collected from various sources, ensuring data quality,
completeness, and integrity before processing.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import json
import datetime
from dataclasses import dataclass
from enum import Enum

from ..utils.config import get_config
from ..utils.file_utils import FileUtils, FileInfo
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Validation status outcomes."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some checks failed but data is usable
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    test_name: str
    status: ValidationStatus
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime.datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()


@dataclass
class ValidationReport:
    """Comprehensive validation report for a dataset."""
    dataset_id: str
    source: str
    product: str
    date_range: str
    overall_status: ValidationStatus
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: int
    critical: int
    results: List[ValidationResult]
    timestamp: datetime.datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            'dataset_id': self.dataset_id,
            'source': self.source,
            'product': self.product,
            'date_range': self.date_range,
            'overall_status': self.overall_status.value,
            'summary': {
                'total_checks': self.total_checks,
                'passed_checks': self.passed_checks,
                'failed_checks': self.failed_checks,
                'warnings': self.warnings,
                'errors': self.errors,
                'critical': self.critical
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'level': r.level.value,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ],
            'timestamp': self.timestamp.isoformat()
        }


class DataValidator:
    """Comprehensive data validation for satellite imagery and time series data."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data validator.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.quality_config = self.config.get('processing.quality_thresholds', {})

        # Default validation thresholds
        self.thresholds = {
            'cloud_cover_max': self.quality_config.get('cloud_cover_max', 20),
            'data_quality_min': self.quality_config.get('data_quality_min', 0.7),
            'min_data_coverage': 0.5,  # Minimum 50% valid data
            'max_missing_percentage': 0.3,  # Maximum 30% missing data
            'spatial_consistency_tolerance': 0.1,  # 10% tolerance
            'temporal_consistency_hours': 24,  # 24 hour tolerance
            'min_data_points': 10,  # Minimum data points for time series
            ' outlier_threshold': 3.0  # Standard deviations
        }

        # Product-specific validation rules
        self.product_rules = {
            'no2': {
                'valid_range': (0, 100),  # Typical range in mol/m2
                'expected_units': 'mol m-2',
                'expected_resolution': 7000  # meters
            },
            'so2': {
                'valid_range': (0, 50),
                'expected_units': 'mol m-2',
                'expected_resolution': 7000
            },
            'co': {
                'valid_range': (0, 0.1),
                'expected_units': 'mol m-2',
                'expected_resolution': 7000
            },
            'o3': {
                'valid_range': (0, 0.5),
                'expected_units': 'mol m-2',
                'expected_resolution': 7000
            },
            'aod': {
                'valid_range': (0, 5),
                'expected_units': 'dimensionless',
                'expected_resolution': 10000  # 10km
            }
        }

        # Islamabad region for spatial validation
        self.region = GeoUtils.create_islamabad_region(buffer_km=50)

        logger.info("Data validator initialized")

    def validate_dataset(self, data: Union[xr.Dataset, pd.DataFrame, Path],
                        source: str, product: str,
                        date_range: Optional[str] = None) -> ValidationReport:
        """
        Perform comprehensive validation of a dataset.

        Args:
            data: Dataset to validate (xarray Dataset, DataFrame, or file path)
            source: Data source (sentinel5p, modis, gee, etc.)
            product: Product type (no2, so2, co, o3, aod)
            date_range: Date range string

        Returns:
            Comprehensive validation report
        """
        results = []

        # Load data if path provided
        if isinstance(data, Path):
            try:
                loaded_data = self._load_data(data)
            except Exception as e:
                results.append(ValidationResult(
                    test_name="data_loading",
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL,
                    message=f"Failed to load data from {data}: {e}"
                ))
                return self._create_report(
                    dataset_id=str(data),
                    source=source,
                    product=product,
                    date_range=date_range or "unknown",
                    results=results
                )
        else:
            loaded_data = data

        # Determine data type and run appropriate validations
        if isinstance(loaded_data, xr.Dataset):
            results.extend(self._validate_xarray_dataset(loaded_data, product))
        elif isinstance(loaded_data, pd.DataFrame):
            results.extend(self._validate_dataframe(loaded_data, product))
        else:
            results.append(ValidationResult(
                test_name="data_type",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.ERROR,
                message=f"Unsupported data type: {type(loaded_data)}"
            ))

        # Product-specific validations
        results.extend(self._validate_product_specific(loaded_data, product))

        # Source-specific validations
        results.extend(self._validate_source_specific(loaded_data, source, product))

        # Spatial validations
        if isinstance(loaded_data, xr.Dataset):
            results.extend(self._validate_spatial_data(loaded_data, product))

        # Temporal validations
        results.extend(self._validate_temporal_data(loaded_data, product))

        # Data quality validations
        results.extend(self._validate_data_quality(loaded_data, product))

        # Generate dataset ID
        dataset_id = self._generate_dataset_id(loaded_data, source, product, date_range)

        return self._create_report(
            dataset_id=dataset_id,
            source=source,
            product=product,
            date_range=date_range or "unknown",
            results=results
        )

    def _validate_xarray_dataset(self, dataset: xr.Dataset, product: str) -> List[ValidationResult]:
        """Validate xarray Dataset."""
        results = []

        # Check if dataset is empty
        if not dataset.data_vars:
            results.append(ValidationResult(
                test_name="dataset_empty",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL,
                message="Dataset contains no data variables"
            ))
            return results

        # Check dimensions
        expected_dims = ['time', 'lat', 'lon'] if 'time' in dataset.dims else ['lat', 'lon']
        missing_dims = [dim for dim in expected_dims if dim not in dataset.dims]
        if missing_dims:
            results.append(ValidationResult(
                test_name="dimensions",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message=f"Missing expected dimensions: {missing_dims}",
                details={'present_dims': list(dataset.dims)}
            ))

        # Check coordinate systems
        if 'lat' in dataset.coords and 'lon' in dataset.coords:
            lat_valid = self._validate_coordinates(dataset['lat'].values, 'latitude')
            lon_valid = self._validate_coordinates(dataset['lon'].values, 'longitude')

            if not lat_valid:
                results.append(ValidationResult(
                    test_name="latitude_coordinates",
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message="Latitude coordinates appear invalid"
                ))

            if not lon_valid:
                results.append(ValidationResult(
                    test_name="longitude_coordinates",
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message="Longitude coordinates appear invalid"
                ))

        # Check for valid data values
        for var_name in dataset.data_vars:
            data_var = dataset[var_name]
            results.extend(self._validate_data_variable(data_var, var_name, product))

        # Check dataset attributes
        if not dataset.attrs:
            results.append(ValidationResult(
                test_name="metadata",
                status=ValidationStatus.WARNING,
                level=ValidationLevel.WARNING,
                message="Dataset has no metadata attributes"
            ))

        return results

    def _validate_dataframe(self, df: pd.DataFrame, product: str) -> List[ValidationResult]:
        """Validate pandas DataFrame."""
        results = []

        # Check if DataFrame is empty
        if df.empty:
            results.append(ValidationResult(
                test_name="dataframe_empty",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL,
                message="DataFrame is empty"
            ))
            return results

        # Check for required columns
        required_columns = self._get_required_columns(product)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results.append(ValidationResult(
                test_name="required_columns",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.ERROR,
                message=f"Missing required columns: {missing_columns}",
                details={'present_columns': list(df.columns)}
            ))

        # Check data types
        if 'latitude' in df.columns and 'longitude' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['latitude']):
                results.append(ValidationResult(
                    test_name="latitude_type",
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message="Latitude column is not numeric"
                ))

            if not pd.api.types.is_numeric_dtype(df['longitude']):
                results.append(ValidationResult(
                    test_name="longitude_type",
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message="Longitude column is not numeric"
                ))

        # Check for missing values
        missing_stats = df.isnull().sum()
        high_missing = missing_stats[missing_stats > len(df) * 0.3]
        if not high_missing.empty:
            results.append(ValidationResult(
                test_name="missing_values",
                status=ValidationStatus.WARNING,
                level=ValidationLevel.WARNING,
                message=f"High missing values in columns: {list(high_missing.index)}",
                details={'missing_percentages': (high_missing / len(df) * 100).to_dict()}
            ))

        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['latitude', 'longitude']:
                continue  # Skip coordinate columns for outlier detection

            outliers = self._detect_outliers(df[col].dropna())
            if len(outliers) > len(df) * 0.1:  # More than 10% outliers
                results.append(ValidationResult(
                    test_name=f"outliers_{col}",
                    status=ValidationStatus.WARNING,
                    level=ValidationLevel.WARNING,
                    message=f"High number of outliers in {col}: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)"
                ))

        return results

    def _validate_data_variable(self, data_var: xr.DataArray, var_name: str, product: str) -> List[ValidationResult]:
        """Validate individual data variable."""
        results = []

        # Get data values
        data_values = data_var.values

        # Check for all NaN values
        if np.all(np.isnan(data_values)):
            results.append(ValidationResult(
                test_name=f"all_nan_{var_name}",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL,
                message=f"Variable {var_name} contains only NaN values"
            ))
            return results

        # Check data coverage
        valid_data = data_values[~np.isnan(data_values)]
        coverage = len(valid_data) / len(data_values.flatten())
        if coverage < self.thresholds['min_data_coverage']:
            results.append(ValidationResult(
                test_name=f"data_coverage_{var_name}",
                status=ValidationStatus.PARTIAL,
                level=ValidationLevel.WARNING,
                message=f"Low data coverage for {var_name}: {coverage:.2%}",
                details={'coverage': coverage, 'threshold': self.thresholds['min_data_coverage']}
            ))

        # Check data range if product rules exist
        if product in self.product_rules:
            valid_range = self.product_rules[product]['valid_range']
            out_of_range = valid_data[(valid_data < valid_range[0]) | (valid_data > valid_range[1])]
            if len(out_of_range) > 0:
                results.append(ValidationResult(
                    test_name=f"range_validation_{var_name}",
                    status=ValidationStatus.WARNING,
                    level=ValidationLevel.WARNING,
                    message=f"{len(out_of_range)} values outside expected range for {var_name}",
                    details={
                        'valid_range': valid_range,
                        'actual_range': [float(np.min(valid_data)), float(np.max(valid_data))],
                        'out_of_range_count': len(out_of_range)
                    }
                ))

        return results

    def _validate_product_specific(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> List[ValidationResult]:
        """Product-specific validation rules."""
        results = []

        if product not in self.product_rules:
            results.append(ValidationResult(
                test_name="product_validation",
                status=ValidationStatus.PARTIAL,
                level=ValidationLevel.WARNING,
                message=f"No specific validation rules for product: {product}"
            ))
            return results

        rules = self.product_rules[product]

        # Check units if metadata available
        if isinstance(data, xr.Dataset) and 'units' in data.attrs:
            units = data.attrs.get('units')
            expected_units = rules.get('expected_units')
            if expected_units and units != expected_units:
                results.append(ValidationResult(
                    test_name="units_validation",
                    status=ValidationStatus.WARNING,
                    level=ValidationLevel.WARNING,
                    message=f"Units mismatch: expected {expected_units}, got {units}"
                ))

        return results

    def _validate_source_specific(self, data: Union[xr.Dataset, pd.DataFrame],
                                 source: str, product: str) -> List[ValidationResult]:
        """Source-specific validation rules."""
        results = []

        # Sentinel-5P specific validations
        if source == 'sentinel5p':
            results.extend(self._validate_sentinel5p(data, product))
        # MODIS specific validations
        elif source == 'modis':
            results.extend(self._validate_modis(data, product))
        # GEE specific validations
        elif source == 'gee':
            results.extend(self._validate_gee(data, product))

        return results

    def _validate_sentinel5p(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> List[ValidationResult]:
        """Sentinel-5P specific validations."""
        results = []

        # Check spatial resolution (approximately)
        if isinstance(data, xr.Dataset):
            if 'lat' in data.coords and 'lon' in data.coords:
                lat_res = float(abs(data['lat'].values[1] - data['lat'].values[0]))
                lon_res = float(abs(data['lon'].values[1] - data['lon'].values[0]))
                expected_res = 0.0625  # Approximately 7km at equator

                if abs(lat_res - expected_res) > 0.02:
                    results.append(ValidationResult(
                        test_name="sentinel5p_resolution",
                        status=ValidationStatus.WARNING,
                        level=ValidationLevel.WARNING,
                        message=f"Unexpected spatial resolution: {lat_res:.4f}° (expected ~{expected_res:.4f}°)"
                    ))

        return results

    def _validate_modis(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> List[ValidationResult]:
        """MODIS specific validations."""
        results = []

        # Check for quality flags if available
        if isinstance(data, xr.Dataset):
            qa_vars = [var for var in data.data_vars if 'qa' in var.lower() or 'QC' in var]
            if not qa_vars:
                results.append(ValidationResult(
                    test_name="modis_quality_flags",
                    status=ValidationStatus.WARNING,
                    level=ValidationLevel.WARNING,
                    message="No quality control flags found in MODIS data"
                ))

        return results

    def _validate_gee(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> List[ValidationResult]:
        """Google Earth Engine specific validations."""
        results = []

        # GEE typically provides cloud-masked data, check for reasonable cloud filtering
        if isinstance(data, xr.Dataset):
            # Check data density (GEE should provide good coverage)
            for var_name in data.data_vars:
                data_values = data[var_name].values
                valid_data = data_values[~np.isnan(data_values)]
                coverage = len(valid_data) / len(data_values.flatten())

                if coverage < 0.7:  # Expect at least 70% coverage from GEE
                    results.append(ValidationResult(
                        test_name=f"gee_coverage_{var_name}",
                        status=ValidationStatus.WARNING,
                        level=ValidationLevel.WARNING,
                        message=f"Low GEE data coverage: {coverage:.2%} (expected >= 70%)"
                    ))

        return results

    def _validate_spatial_data(self, dataset: xr.Dataset, product: str) -> List[ValidationResult]:
        """Validate spatial aspects of the data."""
        results = []

        # Check if data covers Islamabad region
        if 'lat' in dataset.coords and 'lon' in dataset.coords:
            lat_coords = dataset['lat'].values
            lon_coords = dataset['lon'].values

            bbox = self.region['bounding_box']

            # Check coordinate ranges
            if (np.min(lat_coords) > bbox['south'] or np.max(lat_coords) < bbox['north'] or
                np.min(lon_coords) > bbox['west'] or np.max(lon_coords) < bbox['east']):
                results.append(ValidationResult(
                    test_name="spatial_coverage",
                    status=ValidationStatus.WARNING,
                    level=ValidationLevel.WARNING,
                    message="Data does not fully cover Islamabad region",
                    details={
                        'data_bbox': [float(np.min(lon_coords)), float(np.min(lat_coords)),
                                    float(np.max(lon_coords)), float(np.max(lat_coords))],
                        'islamabad_bbox': [bbox['west'], bbox['south'], bbox['east'], bbox['north']]
                    }
                ))

        # Check spatial consistency
        for var_name in dataset.data_vars:
            data_var = dataset[var_name]
            if len(data_var.dims) >= 2:  # 2D or higher
                results.extend(self._check_spatial_consistency(data_var, var_name))

        return results

    def _validate_temporal_data(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> List[ValidationResult]:
        """Validate temporal aspects of the data."""
        results = []

        # Check time series consistency
        if isinstance(data, xr.Dataset) and 'time' in data.coords:
            time_coords = data['time'].values
            if len(time_coords) > 1:
                # Check for time gaps
                time_diffs = np.diff(time_coords)
                expected_diff = np.median(time_diffs)

                large_gaps = time_diffs > expected_diff * 2
                if np.any(large_gaps):
                    results.append(ValidationResult(
                        test_name="temporal_gaps",
                        status=ValidationStatus.WARNING,
                        level=ValidationLevel.WARNING,
                        message=f"Found {np.sum(large_gaps)} large temporal gaps in data"
                    ))

        elif isinstance(data, pd.DataFrame) and 'datetime' in data.columns:
            # Check temporal consistency in DataFrame
            data_sorted = data.sort_values('datetime')
            time_diffs = data_sorted['datetime'].diff().dropna()
            if len(time_diffs) > 0:
                expected_diff = time_diffs.median()
                large_gaps = time_diffs > expected_diff * 2
                if len(large_gaps) > 0:
                    results.append(ValidationResult(
                        test_name="temporal_gaps",
                        status=ValidationStatus.WARNING,
                        level=ValidationLevel.WARNING,
                        message=f"Found {len(large_gaps)} large temporal gaps in DataFrame"
                    ))

        return results

    def _validate_data_quality(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> List[ValidationResult]:
        """Validate overall data quality metrics."""
        results = []

        # Check overall data quality score
        quality_score = self._calculate_quality_score(data, product)

        if quality_score < self.thresholds['data_quality_min']:
            results.append(ValidationResult(
                test_name="overall_quality",
                status=ValidationStatus.FAILED,
                level=ValidationLevel.ERROR,
                message=f"Data quality score {quality_score:.2f} below threshold {self.thresholds['data_quality_min']}",
                details={'quality_score': quality_score}
            ))

        return results

    def _calculate_quality_score(self, data: Union[xr.Dataset, pd.DataFrame], product: str) -> float:
        """Calculate overall data quality score (0-1)."""
        score_components = []

        # Data coverage component
        if isinstance(data, xr.Dataset):
            for var_name in data.data_vars:
                data_values = data[var_name].values
                valid_data = data_values[~np.isnan(data_values)]
                coverage = len(valid_data) / len(data_values.flatten())
                score_components.append(coverage)
        elif isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            total_cells = len(numeric_data) * len(numeric_data.columns)
            valid_cells = numeric_data.notna().sum().sum()
            coverage = valid_cells / total_cells if total_cells > 0 else 0
            score_components.append(coverage)

        # Range validation component
        if product in self.product_rules:
            valid_range = self.product_rules[product]['valid_range']
            # This would need to be implemented based on actual data structure

        # Return average of components
        return np.mean(score_components) if score_components else 0.0

    def _load_data(self, file_path: Path) -> Union[xr.Dataset, pd.DataFrame]:
        """Load data from file path."""
        file_ext = file_path.suffix.lower()

        if file_ext in ['.nc', '.netcdf']:
            return xr.open_dataset(file_path)
        elif file_ext in ['.csv']:
            return pd.read_csv(file_path)
        elif file_ext in ['.parquet']:
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _validate_coordinates(self, coords: np.ndarray, coord_type: str) -> bool:
        """Validate coordinate arrays."""
        if coord_type == 'latitude':
            return -90 <= np.min(coords) and np.max(coords) <= 90
        elif coord_type == 'longitude':
            return -180 <= np.min(coords) and np.max(coords) <= 180
        return False

    def _get_required_columns(self, product: str) -> List[str]:
        """Get required columns for product type."""
        base_columns = ['latitude', 'longitude']
        product_columns = {
            'no2': ['no2'],
            'so2': ['so2'],
            'co': ['co'],
            'o3': ['o3'],
            'aod': ['aod']
        }
        return base_columns + product_columns.get(product, [])

    def _detect_outliers(self, data: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data < lower_bound) | (data > upper_bound)]

    def _check_spatial_consistency(self, data_var: xr.DataArray, var_name: str) -> List[ValidationResult]:
        """Check spatial consistency of data variable."""
        results = []

        # Simple check: data should have reasonable spatial gradients
        if len(data_var.dims) >= 2:
            # Calculate spatial gradients
            if data_var.ndim == 2:
                # 2D data
                grad_x = np.gradient(data_var.values, axis=1)
                grad_y = np.gradient(data_var.values, axis=0)
            else:
                # Multi-dimensional data, take mean over other dimensions
                reduced_data = data_var.mean(dim=[d for d in data_var.dims if d not in ['lat', 'lon']])
                grad_x = np.gradient(reduced_data.values, axis=1)
                grad_y = np.gradient(reduced_data.values, axis=0)

            # Check for extremely high gradients (potential data errors)
            max_grad = np.max(np.abs(np.concatenate([grad_x.flatten(), grad_y.flatten()])))
            if max_grad > np.nanstd(data_var.values) * 10:
                results.append(ValidationResult(
                    test_name=f"spatial_consistency_{var_name}",
                    status=ValidationStatus.WARNING,
                    level=ValidationLevel.WARNING,
                    message=f"High spatial gradients detected in {var_name}: {max_grad:.4f}"
                ))

        return results

    def _generate_dataset_id(self, data: Union[xr.Dataset, pd.DataFrame],
                           source: str, product: str, date_range: Optional[str]) -> str:
        """Generate unique dataset ID."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{source}_{product}_{date_range or 'unknown'}_{timestamp}"

    def _create_report(self, dataset_id: str, source: str, product: str,
                      date_range: str, results: List[ValidationResult]) -> ValidationReport:
        """Create validation report from results."""
        # Count results by status and level
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        partial = sum(1 for r in results if r.status == ValidationStatus.PARTIAL)

        warnings = sum(1 for r in results if r.level == ValidationLevel.WARNING)
        errors = sum(1 for r in results if r.level == ValidationLevel.ERROR)
        critical = sum(1 for r in results if r.level == ValidationLevel.CRITICAL)

        # Determine overall status
        if critical > 0:
            overall_status = ValidationStatus.FAILED
        elif errors > 0:
            overall_status = ValidationStatus.FAILED
        elif failed > 0:
            overall_status = ValidationStatus.PARTIAL
        else:
            overall_status = ValidationStatus.PASSED

        return ValidationReport(
            dataset_id=dataset_id,
            source=source,
            product=product,
            date_range=date_range,
            overall_status=overall_status,
            total_checks=len(results),
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            errors=errors,
            critical=critical,
            results=results
        )

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to file."""
        output_path = Path(output_path)
        FileUtils.ensure_directory(output_path.parent)

        report_dict = report.to_dict()

        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
        else:
            # Default to JSON
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(report_dict, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")

    def validate_batch(self, data_files: List[Path], source: str, product: str) -> List[ValidationReport]:
        """Validate multiple datasets."""
        reports = []
        for file_path in data_files:
            try:
                report = self.validate_dataset(file_path, source, product)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to validate {file_path}: {e}")
                # Create failed report
                results = [ValidationResult(
                    test_name="file_validation",
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL,
                    message=f"Failed to validate file: {e}"
                )]
                report = self._create_report(
                    dataset_id=str(file_path),
                    source=source,
                    product=product,
                    date_range="unknown",
                    results=results
                )
                reports.append(report)

        return reports