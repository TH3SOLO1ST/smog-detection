# Islamabad Smog Detection System

A comprehensive Python-based system for acquiring real satellite imagery, processing atmospheric data, and analyzing air pollution patterns for Islamabad, Pakistan using Sentinel-5P/TOMI, NASA MODIS, and Google Earth Engine data.

## Overview

This system provides end-to-end capabilities for satellite-based smog detection and air pollution analysis:

- **Real-time Data Acquisition**: Multiple satellite data sources with automated APIs
- **Advanced Image Processing**: Atmospheric correction, noise reduction, and contrast enhancement
- **Statistical Analysis**: Time series analysis, correlation studies, and spatial statistics
- **Machine Learning**: Data preparation, model training, and pollution prediction
- **Interactive Visualizations**: Maps, charts, dashboards, and real-time monitoring tools

## Features

### Data Acquisition
- **Sentinel-5P/TROPOMI**: NO2, SO2, CO, O3, and aerosol index data via Copernicus Data Space Ecosystem
- **NASA MODIS**: Aerosol optical depth and atmospheric data via FIRMS API
- **Google Earth Engine**: Pre-processed collections with built-in corrections
- **Data Validation**: Quality control, integrity checks, and completeness verification

### Image Preprocessing
- **Atmospheric Correction**: Dark Object Subtraction (DOS) and haze removal algorithms
- **Noise Reduction**: Gaussian, median, bilateral, and non-local means filtering
- **Contrast Enhancement**: CLAHE, histogram equalization, and adaptive methods
- **Geospatial Processing**: Reprojection, resampling, and region extraction

### Analysis Capabilities
- **Time Series Processing**: Aggregation, decomposition, trend analysis, and anomaly detection
- **Statistical Analysis**: Descriptive statistics, correlation matrices, regression modeling
- **Machine Learning Pipeline**: Feature engineering, model training, and evaluation
- **Spatial Statistics**: Hotspot detection, autocorrelation analysis, pattern recognition

### Visualization & Reporting
- **Interactive Maps**: Real-time pollution heatmaps with multi-pollutant overlays
- **Time Series Charts**: Line plots, area charts, and seasonal pattern analysis
- **Statistical Visualizations**: Distribution plots, correlation heatmaps, and scatter matrices
- **Comprehensive Dashboards**: HTML, PDF, and interactive dashboards with alert systems

## Project Structure

```
smog-detection/
├── src/                          # Source code modules
│   ├── data_acquisition/        # Satellite data collectors
│   │   ├── __init__.py
│   │   ├── sentinel5p_collector.py      # Sentinel-5P/TROPOMI data
│   │   ├── modis_collector.py           # NASA MODIS data
│   │   ├── gee_collector.py             # Google Earth Engine
│   │   └── data_validator.py            # Data quality validation
│   ├── preprocessing/              # Image processing modules
│   │   ├── __init__.py
│   │   ├── atmospheric_correction.py    # Atmospheric correction
│   │   ├── noise_reduction.py           # Noise reduction
│   │   ├── contrast_enhancement.py      # Contrast enhancement
│   │   └── geospatial_processor.py      # Geospatial processing
│   ├── analysis/                    # Analysis modules
│   │   ├── __init__.py
│   │   ├── time_series_processor.py     # Time series analysis
│   │   ├── statistical_analysis.py      # Statistical analysis
│   │   └── ml_pipeline.py               # Machine learning
│   ├── visualization/               # Visualization modules
│   │   ├── __init__.py
│   │   ├── mapping_tools.py           # Interactive maps
│   │   ├── charting_tools.py          # Charts and plots
│   │   └── dashboard_generator.py     # Dashboard generation
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── geo_utils.py              # Geographic utilities
│       └── file_utils.py              # File operations
├── data/                          # Data directories
│   ├── raw/                         # Original satellite data
│   ├── processed/                   # Preprocessed data
│   └── exports/                     # Final outputs
├── notebooks/                      # Analysis notebooks
├── tests/                         # Unit and integration tests
├── requirements.txt                # Python dependencies
├── config.yaml                    # System configuration
└── docs/                          # Documentation
```

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/smog-detection.git
cd smog-detection
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure API credentials**:
   - Copy `config.yaml.example` to `config.yaml`
   - Fill in your API credentials:
     - Copernicus client ID and secret
     - NASA Earthdata credentials
     - Google Earth Engine service account

5. **Install Google Earth Engine (optional)**:
```bash
pip install earthaccess
earthaccess authenticate
```

## Usage

### Basic Data Collection

```python
from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector
from src.utils.config import get_config

# Initialize collector
collector = Sentinel5PCollector()

# Collect NO2 data for Islamabad
dataset = collector.collect_data(
    product='no2',
    start_date='2023-01-01',
    end_date='2023-01-31'
)
```

### Image Preprocessing

```python
from src.preprocessing.atmospheric_correction import AtmosphericCorrector
from src.preprocessing.noise_reduction import NoiseReducer

# Apply atmospheric correction
corrector = AtmosphericCorrector()
corrected_dataset = corrector.correct_dataset(dataset)

# Apply noise reduction
reducer = NoiseReducer()
processed_dataset = reducer.reduce_noise(corrected_dataset)
```

### Statistical Analysis

```python
from src.analysis.statistical_analysis import StatisticalAnalyzer

# Perform comprehensive analysis
analyzer = StatisticalAnalyzer()
results = analyzer.perform_comprehensive_analysis(
    processed_dataset,
    analysis_types=['descriptive', 'correlation', 'spatial']
)
```

### Machine Learning

```python
from src.analysis.ml_pipeline import MLPipeline

# Prepare data and train models
pipeline = MLPipeline()
results = pipeline.run_full_pipeline(
    processed_dataset,
    target_var='no2',
    models=['random_forest', 'xgboost']
)
```

### Visualization

```python
from src.visualization.dashboard_generator import DashboardGenerator

# Generate comprehensive dashboard
generator = DashboardGenerator()
dashboard_paths = generator.generate_comprehensive_dashboard(
    processed_dataset,
    pollutants=['no2', 'so2', 'co'],
    dashboard_type='interactive'
)
```

## Configuration

The system is configured through `config.yaml`. Key configuration sections:

### API Credentials
```yaml
apis:
  copernicus:
    client_id: "${COPERNICUS_CLIENT_ID}"
    client_secret: "${COPERNICUS_CLIENT_SECRET}"
  nasa:
    earthdata_username: "${NASA_EARTHDATA_USERNAME}"
    earthdata_password: "${NASA_EARTHDATA_PASSWORD}"
  google_earth_engine:
    project_id: "${GEE_PROJECT_ID}"
    service_account_key: "${GEE_SERVICE_ACCOUNT_KEY}"
```

### Region Configuration
```yaml
region:
  name: "Islamabad"
  center_lat: 33.6844
  center_lon: 73.0479
  buffer_km: 50
  bounding_box:
    north: 34.1844
    south: 33.1844
    east: 73.9479
    west: 72.1479
```

### Processing Parameters
```yaml
processing:
  atmospheric_correction:
    method: "DOS"
    dark_object_percentile: 1
  noise_reduction:
    gaussian_kernel: 3
    median_filter_size: 3
  quality_thresholds:
    cloud_cover_max: 20
    data_quality_min: 0.7
```

## Data Sources

### Satellite Specifications

| Platform | Sensor | Products | Resolution | Coverage |
|----------|--------|----------|-----------|
| Sentinel-5P | TROPOMI | NO2, SO2, CO, O3, AOD | 7km × 3.5km | Global |
| MODIS Terra/Aqua | MODIS/MYD04_L2 | Aerosol AOD | 1km | Global |
| Landsat 8 | OLI/TIRS | Surface Reflectance | 30m | Global |

### Pollutant Information

| Pollutant | Typical Range | Units | Health Impact |
|----------|---------------|-------|--------------|
| NO2 | 0-100 µmol/m² | Micrograms per cubic meter | Respiratory irritation |
| SO2 | 0-50 µmol/m² | Micrograms per cubic meter | Respiratory issues |
| CO | 0-0.1 mg/m³ | Milligrams per cubic meter | Reduced oxygen transport |
| O3 | 0-0.5 DU | Dobson Units | Respiratory irritation |
| AOD | 0-5 | Unitless | Reduced visibility |

## API Reference

### Data Acquisition

#### Sentinel5PCollector

```python
class Sentinel5PCollector:
    def collect_data(self, product, start_date, end_date=None):
        """Collect satellite data for specified product and date range."""
        pass

    def collect_time_series(self, product, start_date, end_date):
        """Collect time series data."""
        pass

    def get_available_products(self):
        """Get list of available Sentinel-5P products."""
        pass
```

### Preprocessing

#### AtmosphericCorrector

```python
class AtmosphericCorrector:
    def correct_dataset(self, dataset, methods=['DOS', 'combined']):
        """Apply atmospheric correction to dataset."""
        pass

    def calculate_correction_metrics(self, original, corrected):
        """Calculate correction effectiveness metrics."""
        pass
```

### Analysis

#### TimeSeriesProcessor

```python
class TimeSeriesProcessor:
    def process_dataset(self, dataset, operations):
        """Process time series data with various analyses."""
        pass

    def aggregate_data(self, dataset):
        """Aggregate data to different temporal resolutions."""
        pass

    def detect_anomalies(self, dataset, method='zscore'):
        """Detect anomalies in time series data."""
        pass
```

#### StatisticalAnalyzer

```python
class StatisticalAnalyzer:
    def perform_comprehensive_analysis(self, dataset, analysis_types):
        """Perform comprehensive statistical analysis."""
        pass

    def correlation_analysis(self, dataset):
        """Perform correlation analysis between variables."""
        pass
```

#### MLPipeline

```python
class MLPipeline:
    def run_full_pipeline(self, dataset, target_var, models=None):
        """Run complete machine learning pipeline."""
        pass

    def prepare_ml_dataset(self, dataset, target_var):
        """Prepare dataset for machine learning."""
        pass
```

### Visualization

#### DashboardGenerator

```python
class DashboardGenerator:
    def generate_comprehensive_dashboard(self, dataset, pollutants):
        """Generate comprehensive dashboard."""
        pass

    def create_real_time_dashboard(self, dataset, pollutants):
        """Create real-time monitoring dashboard."""
        pass
```

## Examples

### Example 1: Complete Data Pipeline

```python
from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector
from src.preprocessing.atmospheric_correction import AtmosphericCorrector
from src.analysis.statistical_analysis import StatisticalAnalyzer
from src.visualization.dashboard_generator import DashboardGenerator

# 1. Collect data
collector = Sentinel5PCollector()
raw_data = collector.collect_data(
    product='no2',
    start_date='2023-10-01',
    end_date='2023-10-31'
)

# 2. Preprocess
corrector = AtmosphericCorrector()
processed_data = corrector.correct_dataset(raw_data)

# 3. Analyze
analyzer = StatisticalAnalyzer()
analysis_results = analyzer.perform_comprehensive_analysis(
    processed_data,
    analysis_types=['descriptive', 'correlation']
)

# 4. Visualize
generator = DashboardGenerator()
dashboard = generator.generate_comprehensive_dashboard(
    processed_data,
    pollutants=['no2']
)

print(f"Dashboard created: {dashboard}")
```

### Example 2: Machine Learning Prediction

```python
from src.analysis.ml_pipeline import MLPipeline

# Create synthetic time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
no2_data = np.sin(np.arange(365) * 2 * np.pi / 365) * 0.00005 + 0.0001

# Create xarray dataset
ds = xr.Dataset(
    data_vars={'no2': (['time'], no2_data)},
    coords={'time': dates}
)

# Run ML pipeline
pipeline = MLPipeline()
results = pipeline.run_full_pipeline(
    ds,
    target_var='no2',
    models=['random_forest', 'xgboost'],
    feature_selection=True
)

print(f"Model performance: {results['evaluation']['test_metrics']}")
```

### Example 3: Real-time Monitoring

```python
from src.visualization.dashboard_generator import DashboardGenerator

# Generate real-time dashboard with alerts
alert_dashboard = generator.create_real_time_dashboard(
    current_data,
    pollutants=['no2', 'so2', 'co'],
    thresholds={
        'no2': {'moderate': 0.05, 'warning': 0.1, 'critical': 0.2},
        'so2': {'moderate': 0.02, 'warning': 0.05, 'critical': 0.1}
    },
    update_interval=300  # 5 minutes
)

print(f"Real-time dashboard: {alert_dashboard}")
```

## Testing

Run the test suite to verify installation and functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_data_acquisition.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_analysis.py -v
```

### Test Coverage

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Performance

### System Requirements

- **Minimum**: Python 3.8, 8GB RAM, 50GB disk space
- **Recommended**: Python 3.10+, 16GB RAM, 100GB disk space

### Optimizations

- **Parallel Processing**: Dask-based parallel processing for large datasets
- **Memory Management**: Chunked processing for high-resolution satellite data
- **Caching**: Local caching of frequently accessed data
- **Batch Processing**: Efficient handling of multiple date ranges

### Benchmarks

| Operation | Dataset Size | Processing Time | Memory Usage |
|----------|--------------|----------------|-------------|
| Sentinel-5P Collection | 1 month | ~30 seconds | ~2GB |
| MODIS Collection | 1 month | ~20 seconds | ~1GB |
| Preprocessing | 100MB | ~10 seconds | ~1GB |
| ML Training | 10K samples | ~5 seconds | ~500MB |

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API credentials in `config.yaml`
   - Check internet connectivity
   - Ensure API quotas are not exceeded

2. **Memory Issues**
   - Use chunked processing for large datasets
   - Reduce spatial resolution
   - Process smaller time ranges

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Ensure virtual environment is activated

4. **Data Quality Issues**
   - Run data validation before processing
   - Check for missing data periods
   - Verify coordinate systems

### Debug Mode

Enable debug logging in `config.yaml`:

```yaml
logging:
  level: "DEBUG"
```

Or in code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/smog-detection.git
cd smog-detection

# Create development environment
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements.txt
pip install -e .[dev]  # Development dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Copernicus Sentinel-5P**: Atmospheric composition data
- **NASA MODIS**: Aerosol and atmospheric data
- **Google Earth Engine**: Cloud-based geospatial processing
- **ESA**: Space agency collaboration and data sharing

## Citation

If you use this system in your research, please cite:

```
Islamabad Smog Detection System (2025)
Satellite-based air pollution monitoring and analysis for Islamabad, Pakistan
GitHub Repository: https://github.com/TH3SOLO1ST/smog-detection
```

## Contact

For questions, issues, or contributions:

- GitHub Issues: https://github.com/TH3SOLO1ST/smog-detection/issues
- Email: excel3227@gmail.com
- Project Lead: excel3227@gmail.com

---

**Note**: This system is designed for research and educational purposes. For operational air quality monitoring, please refer to official environmental agency resources.
