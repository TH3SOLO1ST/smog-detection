# Islamabad Smog Detection System - User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Data Collection](#data-collection)
   - [Sentinel-5P Data](#sentinel5p-data)
   - [NASA MODIS Data](#modis-data)
   - [Google Earth Engine](#gee-data)
4. [Data Preprocessing](#preprocessing)
   - [Atmospheric Correction](#atmospheric-correction)
   - [Noise Reduction](#noise-reduction)
   - [Contrast Enhancement](#contrast-enhancement)
   - [Geospatial Processing](#geospatial-processing)
5. [Analysis](#analysis)
   - [Time Series Analysis](#time-series-analysis)
   - [Statistical Analysis](#statistical-analysis)
   - [Machine Learning](#machine-learning)
6. [Visualization](#visualization)
   - [Interactive Maps](#interactive-maps)
   - [Charts and Plots](#charts-and-plots)
   - [Dashboards](#dashboards)
7. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Advanced Analysis](#advanced-analysis)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+
- Git
- API credentials for satellite data sources (see Configuration section)

### Installation

1. Clone and set up the environment:
```bash
git clone https://github.com/your-repo/smog-detection.git
cd smog-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure API credentials:
   - Copy `config.yaml.example` to `config.yaml`
   - Fill in your API credentials

3. Run your first data collection:
```python
from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector

collector = Sentinel5PCollector()
dataset = collector.collect_data(
    product='no2',
    start_date='2023-01-01',
    end_date='2023-01-31'
)
```

## Configuration

The system uses `config.yaml` for configuration. Key sections:

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
  center_lon:   permeation: 50
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

## Data Collection

### Sentinel-5P Data

The Sentinel-5P/TROPOMI sensor provides atmospheric composition data at high spatial resolution:

```python
from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector

# Initialize collector
collector = Sentinel5PCollector()

# Collect different pollutants
no2_data = collector.collect_data('no2', '2023-01-01', '2023-01-31')
so2_data = collector.collect_data('so2', '2023-01-01', '2023-01-31')
aod_data = collector.collect_data('aod', '2023-01-01', '2023-01-31')

# Collect time series
monthly_data = collector.collect_time_series('no2', '2023-01-01', '2023-12-31')
```

### NASA MODIS Data

NASA FIRMS provides active fire and aerosol data:

```python
from src.data_acquisition.modis_collector import MODISCollector

collector = MODISCollector()

# Get aerosol data for Pakistan
firms_data = collector.collect_firms_data('MODIS_AF_MOD04_L2', '2023-10-01', '2023-10-31')

# Get multiple products
mod04_data = collector.collect_earthaccess_data('MOD04_L2', '2023-01-01', '2023-01-31')
mod13_data = collector.collect_earthaccess_data('MOD13Q1', '2023-01-01', '2023-01-31')
```

### Google Earth Engine

Google Earth Engine provides pre-processed data:

```python
from src.data_acquisition.gee_collector import GEECollector

collector = GEECollector()

# Get Level 3 NO2 data
no2_gee = collector.collect_sentinel5p_data('no2', '2023-01-01', '2023-12-31')

# Get MODIS aerosol data
modis_gee = collector.collect_modis_data('MOD04_L2', '2023-01-01', '2023-01-31')
```

## Data Preprocessing

### Atmospheric Correction

Apply atmospheric correction to remove atmospheric interference:

```python
from src.preprocessing.atmospheric_correction import AtmosphericCorrector

corrector = AtmosphericCorrector()

# Different correction methods
dos_corrected = corrector.correct_dataset(dataset, methods=['DOS'])
haze_corrected = corrector.correct_dataset(dataset, methods=['haze_removal'])
combined_corrected = corrector.correct_dataset(dataset, methods=['combined'])

# Calculate improvement metrics
metrics = corrector.calculate_correction_metrics(original_data, corrected_dataset)
```

### Noise Reduction

Remove sensor noise while preserving important features:

```python
from src.preprocessing.noise_reduction import NoiseReducer

reducer = NoiseReducer()

# Apply different filters
gaussian_filtered = reducer.reduce_noise(dataset, methods=['gaussian'])
median_filtered = reducer.reduce_noise(dataset, methods=['median'])
bilateral_filtered = reducer.reduce_noise(dataset, methods=['bilateral'])
```

### Contrast Enhancement

Improve data visibility for analysis:

```python
from src.preprocessing.contrast_enhancement import ContrastEnhancer

enhancer = ContrastEnhancer()

# Apply CLAHE for adaptive contrast
clahe_enhanced = enhancer.enhance_contrast(dataset, methods=['clahe'])

# Apply gamma correction
gamma_enhanced = enhancer.enhance_contrast(dataset, methods=['gamma_correction'])
```

### Geospatial Processing

Process data for Islamabad region:

```python
from src.preprocessing.geospatial_processor import GeospatialProcessor

processor = GeospatialProcessor()

# Clip to Islamabad region
clipped_data = processor.process_dataset(dataset, operations=['clip'])

# Resample to uniform resolution
resampled_data = processor.process_dataset(dataset, operations=['resample'])

# Apply spatial consistency checks
validated_data = processor.validate_spatial_consistency(clipped_data)
```

## Analysis

### Time Series Analysis

Analyze temporal patterns and trends:

```python
from src.analysis.time_series_processor import TimeSeriesProcessor

processor = TimeSeriesProcessor()

# Aggregate data
daily_data = processor.process_dataset(dataset, operations=['aggregate'])
monthly_data = processor.process_dataset(dataset, operations=['aggregate', 'monthly'])

# Detect anomalies
anomaly_data = processor.process_dataset(dataset, operations=['detect_anomalies'])

# Decompose time series
decomposed = processor.process_dataset(dataset, operations=['decompose'])
```

### Statistical Analysis

Perform comprehensive statistical analysis:

```python
from src.analysis.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Descriptive statistics
desc_stats = analyzer.descriptive_statistics(dataset)

# Correlation analysis
correlations = analyzer.correlation_analysis(dataset)

# Regression analysis
regressions = analyzer.regression_analysis(dataset)

# Spatial statistics
spatial_stats = analyzer.spatial_statistics(dataset)
```

### Machine Learning Pipeline

Train pollution prediction models:

```python
from src.analysis.ml_pipeline import MLPipeline

pipeline = MLPipeline()

# Prepare data
features, target = pipeline.prepare_ml_dataset(dataset, 'no2')

# Split data
split_data = pipeline.split_data(features, target)

# Train models
models = pipeline.train_models(
    split_data['X_train'], split_data['y_train'],
    split_data['X_val'], split_data['y_val'],
    models=['random_forest', 'gradient_boosting']
)

# Evaluate models
evaluation = pipeline.evaluate_model(best_model, X_test, y_test, 'no2')
```

## Visualization

### Interactive Maps

Create interactive pollution maps:

```python
from src.visualization.mapping_tools import MappingTools

mapper = MappingTools()

# Create pollution heatmap
no2_map = mapper.create_pollution_heatmap(dataset, 'no2')
multi_polutant_map = mapper.create_multi_pollutant_comparison_map(
    dataset, ['no2', 'so2', 'co']
)
```

### Charts and Plots

Generate comprehensive visualizations:

```python
from src.visualization.charting_tools import ChartingTools

charting = ChartingTools()

# Time series plots
ts_charts = charting.create_time_series_chart(
    dataset, ['no2', 'so2', 'co'],
    chart_type='interactive'
)

# Distribution plots
dist_plots = charting.create_distribution_plots(
    dataset, ['no2', 'so2', 'co']
)

# Statistical visualizations
correlation_plots = charting.create_correlation_analysis(dataset, ['no2', 'so2', 'co'])
```

### Dashboards

Generate comprehensive monitoring dashboards:

```python
from src.visualization.dashboard_generator import DashboardGenerator

generator = DashboardGenerator()

# Interactive HTML dashboard
dashboard = generator.generate_comprehensive_dashboard(
    dataset, ['no2', 'so2', 'co'],
    dashboard_type='interactive'
)

# PDF report
report = generator.generate_comprehensive_dashboard(
    dataset, ['no2', 'so2', 'co'],
    dashboard_type='pdf'
)

# Real-time monitoring dashboard with alerts
alert_dashboard = generator.create_real_time_dashboard(
    current_data, ['no2', 'so2', 'co'],
    thresholds={
        'no2': {'moderate': 0.05, 'warning': 0.1, 'critical': 0.2}
    }
)
```

## Examples

### Example 1: Complete Data Pipeline

```python
# Collect data
from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector
from src.preprocessing.atmospheric_correction import AtmosphericCorrector
from src.analysis.statistical_analysis import StatisticalAnalyzer
from src.visualization.dashboard_generator import DashboardGenerator

# 1. Collect data
collector = Sentinel5PCollector()
raw_data = collector.collect_data('no2', '2023-10-01', '2023-10-31')

# 2. Preprocess
corrector = AtmosphericCorrector()
processed_data = corrector.correct_dataset(raw_data)

# 3. Analyze
analyzer = StatisticalAnalyzer()
results = analyzer.perform_comprehensive_analysis(
    processed_data,
    analysis_types=['descriptive', 'correlation']
)

# 4. Visualize
generator = DashboardGenerator()
dashboard = generator.generate_comprehensive_dashboard(
    processed_data,
    pollutants=['no2']
)
```

### Example 2: Machine Learning Prediction

```python
from src.analysis.ml_pipeline import MLPipeline

# Prepare and train model
pipeline = MLPipeline()
results = pipeline.run_full_pipeline(
    dataset, 'no2',
    models=['random_forest', 'xgboost'],
    feature_selection=True
)

# Check model performance
print(f"R² Score: {results['evaluation']['test_metrics']['r2']:.4f}")
print(f"RMSE: {results['evaluation']['test_metrics']['rmse']:.4f}")
```

### Example 3: Real-time Monitoring

```python
# Create real-time dashboard with alerts
alert_dashboard = generator.create_real_time_dashboard(
    current_data,
    pollutants=['no2', 'so2', 'co'],
    thresholds={
        'no2': {'moderate': 0.05, 'warning': 0.1, 'critical': 0.2}
    },
    update_interval=300  # 5 minutes
)

# Generate real-time dashboard with auto-refresh
alert_dashboard = generator.create_real_time_dashboard(
    dataset, ['no2', 'so2'],
    save_to_disk=True
)
```

### Example 4: Advanced Analysis

```python
# Multi-source data fusion
from src.data_acquisition.gee_collector import GEECollector
from src.preprocessing.geospatial_processor import GeospatialProcessor

# Collect data from multiple sources
gee_data = GEECollector()
gee_no2 = gee_data.collect_sentinel5p_data('no2', '2023-01-01', '2023-12-31')
gee_modis = gee_data.collect_modis_data('MOD04_L2', '2023-01-01', '2023-12-31')

# Combine data using weighted average
fused_dataset = processor.fuse_datasets([gee_no2, gee_modis])

# Advanced statistical analysis
advanced_results = analyzer.perform_comprehensive_analysis(
    fused_dataset,
    analysis_types=['descriptive', 'correlation', 'regression', 'spatial', 'multivariate']
)
```

## Examples

### Example 1: Basic Data Collection and Processing

```python
from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector
from src.preprocessing.atmospheric_correction import AtmosphericCorrector
from src.analysis.statistical_analysis import StatisticalAnalyzer
from src.visualization.dashboard_generator import DashboardGenerator

# Initialize and collect data
collector = Sentinel5PCollector()

# Collect NO2 data for Islamabad (smog season: October-November)
dataset = collector.collect_data('no2', '2023-10-01', '2023-11-30')

# Apply atmospheric correction
corrector = AtmosphericCorrector()
corrected_data = corrector.correct_dataset(dataset, methods=['DOS', 'haze_removal'])

# Statistical analysis
analyzer = StatisticalAnalyzer()
stats = analyzer.perform_comprehensive_analysis(corrected_data,
    analysis_types=['descriptive', 'correlation'])

# Generate dashboard
generator = DashboardGenerator()
dashboard = generator.generate_comprehensive_dashboard(
    corrected_data,
    pollutants=['no2']
)

# Results
print(f"Dashboard generated at: {dashboard}")
print(f"NO2 mean: {stats['no2']['mean']:.4f}")
```

### Example 2: Multi-Pollutant Comparative Analysis

```python
from src.data_acquisition import Sentinelp5PCollector, MODISCollector
from src.preprocessing.geospatial_processor import Geosprocessor
from src.analysis.statistical_analysis import StatisticalAnalyzer

# Initialize collectors
sentinel_collector = Sentinelp5PCollector()
modis_collector = MODISCollector()

# Collect data from multiple sources
sentinel_data = sentinel_collector.collect_time_series('no2', '2023-01-01', '2023-12-31')
modis_data = modis_collector.collect_firms_data('MODIS_AF_MOD04_L2', '2023-01-01', '2023-12-31')

# Process datasets independently
sentinel_processed = GeospatialProcessor().process_dataset(sentinel_data, operations=['clip', 'resample'])
modis_processed = GeospatialProcessor().process_dataset(modis_data, operations=['clip', 'resample'])

# Analyze both datasets
sentinel_analysis = StatisticalAnalyzer()
modis_analysis = StatisticalAnalyzer()

sentinel_results = sentinel_analysis.perform_comprehensive_analysis(
    sentinel_processed,
    analysis_types=['descriptive', 'spatial']
)

modis_results = modis_analysis.perform_comprehensive_analysis(
    modis_processed,
    analysis_types=['descriptive', 'spatial']
)

# Comparative analysis
comparison = StatisticalAnalyzer()

# Compare spatial means
sentinel_mean = sentinel_results['spatial']['no2']['spatial_mean']
modis_mean = modis_results['spatial']['no2']['spatial_mean']

correlation = StatisticalAnalyzer.correlation_analysis(
    sentinel_processed.merge(sentinel_processed)
)

print(f"Mean NO2 (Sentinel-5P): {sentinel_mean:.6f}")
print(f"Mean AOD (MODIS): {modis_mean:.6f}")
print(f"Correlation: {correlation['pearson']['no2_modis']:.3f}")
```

### Example 3: Seasonal Pattern Analysis

```python
from src.analysis.time_series_processor import TimeSeriesProcessor
from src.visualization.charting_tools import ChartingTools

# Initialize processor
processor = TimeSeriesProcessor()

# Process for seasonal analysis
seasonal_data = processor.process_dataset(
    dataset,
    operations=['aggregate', 'decompose', 'seasonal']
)

# Generate seasonal plots
seasonal_plots = ChartingTools.create_seasonal_analysis_charts(
    dataset,
    pollutants=['no2', 'so2']
)

# Monthly patterns
monthly_data = processor.process_dataset(
    dataset,
    operations=['aggregate', 'monthly']
)

# Export seasonal data
export_data = processor._save_time_series(
    seasonal_data, 'no2', save_to_disk=True
)

print(f"Seasonal analysis completed for {len(seasonal_data['no2'])} time periods")
```

### Example 4: Machine Learning Model Evaluation

```python
from src.analysis.ml_pipeline import MLPipeline
from src.utils.config import get_config

# Initialize pipeline
pipeline = MLPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(
    dataset,
    target_var='no2',
    models=['linear', 'random_forest', 'xgboost'],
    feature_selection=True,
    save_to_disk=True
)

# Compare model performance
for model_name, model_data in results['all_models'].items():
    print(f"{model_name}:")
    print(f"  R²: {model_data['val_metrics']['r2']:.4f}")
    print(f"  RMSE: {model_data['val_metrics']['rmse']:.4f}")
    print(f"  MAE: {model_data['val_metrics']['mae']:.4f}")
    print()

# Select best model
best_model_name, best_model_data = pipeline.select_best_model(
    results['all_models'],
    metric='r2'
)

print(f"Best model: {best_model_name}")
print(f"Test R²: {best_model_data['evaluation']['test_metrics']['r2']:.4f}")
```

## Troubleshooting

### Installation Issues

**Q: Getting import errors**
```bash
# Ensure virtual environment is activated
source venv/binactivate  # Linux/Mac
# Or on Windows
venv\Scripts\activate  # Windows
```

**Q: API authentication failures**
```bash
# Check config.yaml has correct credentials
cat config.yaml

# Verify environment variables
echo $COPERNICUS_CLIENT_ID
echo $COPERNICUS_CLIENT_SECRET
```

**Q: Memory errors with large datasets**
```python
# Use chunked processing for large datasets
from src.utils.file_utils import FileUtils

# Process in chunks
chunk_size = 100
for chunk in range(0, len(data.time), chunk_size):
    chunk = data.isel(time=slice(chunk, min(chunk+1, len(data.time)))
    processed_chunk = processor.process_dataset(chunk)
    # Save chunk results
    FileUtils.save_dataframe(processed_chunk, f"chunk_{chunk}.csv")
```

**Q: Slow processing**
```python
# Enable parallel processing
from dask import delayed, compute
from multiprocessing import cpu_count

# Use Dask for parallel time series
results = []
for date in date_range:
    delayed_results = delayed(process_date(date, pollutant))
    results.append(delayed_results)
```

### Data Quality Issues

**Q: Data gaps**
```python
# Check for missing data
from src.data_acquisition.data_validator import DataValidator

validator = DataValidator()
validation_report = validator.validate_dataset(dataset, 'test', 'no2')

# Print validation summary
print(f"Validation status: {validation_report.overall_status.value}")
print(f"Total checks: {validation_report.total_checks}")
print(f"Passed: {validation_report.passed_checks}")
print(f"Failed: {validation_report.failed_checks}")
```

### Performance Issues

**Q: Processing bottlenecks**
```python
# Monitor processing time
import time
start_time = time.time()
processed = processor.process_dataset(large_dataset)
end_time = time.time()

print(f"Processing time: {end_time - start_time:.2f} seconds")
```

**Q: Memory leaks**
```python
# Monitor memory usage
import psutil
import gc

process_memory = psutil.Process().memory_info()
print(f"Memory usage: {process_memory.rss:.2f} MB")
```

### API Issues

**Q: API rate limits**
```python
# Implement retry logic with exponential backoff
collector = Sentinel5PCollector()

# Collector includes retry logic automatically
dataset = collector.collect_data(
    product='no2',
    start_date='2023-01-01',
    end_date='2023-01-31',
    save_to_disk=True
)
```

**Q: Google Earth Engine quota**
```python
# Monitor GEE compute unit usage
from src.data_acquisition.gee_collector import GEECollector
import ee

collector = GEECollector()

# Check available quota
usage = collector.get_available_products('no2', '2023-01-01', '2023-12-31')

if usage:
    print(f"Data available for NO2: {usage}")
else:
    print("No data available - check GEE settings")
```

## Best Practices

### Data Management

**Storage Organization:**
- Use consistent directory structure
- Separate raw, processed, and exported data
- Implement automated cleanup of old data
- Track data provenance and metadata

**Data Validation:**
- Always validate before processing
- Check for temporal and spatial consistency
- Verify coordinate systems
- Monitor data quality indicators

**Processing Pipeline:**
- Process data in logical order
- Save intermediate results
- Use appropriate correction methods
- Document all processing parameters

### Analysis Quality

**Statistical Rigor:**
- Use appropriate statistical tests
- Validate assumptions with domain experts
- Check significance levels
- Document analysis limitations

**Machine Learning:**
- Follow proper data splitting procedures
- Use cross-validation with time series splits
- Evaluate on independent test set
- Save model artifacts and metadata

### Visualization Quality

**Map Design:**
- Use appropriate color schemes for different pollutants
- Add interactive controls
- Include coordinate reference systems
- Provide clear legends and titles

**Chart Clarity:**
- Use descriptive titles and labels
- Include axis labels and units
- Add confidence intervals where applicable
- Explain statistical significance

## Monitoring and Alerting

### Real-time Monitoring

Create automated alert systems for:

- Pollution threshold breaches
- Data pipeline failures
- API quota exhaustion
- Model performance degradation

### Alert Configuration

```yaml
monitoring:
  alerts:
    enabled: true
    smtp_server: "${SMTP_SERVER}"
    smtp_port: 587
    recipients: ["admin@domain.com"]
  thresholds:
    no2_alert: 0.1
    so2_alert: 0.05
    co_alert: 0.02
```

### Dashboard Auto-refresh

Set up auto-refresh for real-time monitoring:

```html
<!-- Auto-refresh every 5 minutes -->
<meta http-equiv="refresh" content="300">
```

## Contributing

We welcome contributions! Please see our guidelines:

1. **Code Quality**
   - Follow existing code patterns
   - Add comprehensive tests
   - Document new functionality
   - Use type hints and documentation

2. **Documentation**
   - Keep README updated
   - Add user guide examples
   - Document API changes

3. **Testing**
   - Add unit tests for new features
   - Test with realistic data
   - Validate with edge cases

## Resources

- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- [NASA FIRMS API Documentation](https://firms.modaps.eosdis.nasa.gov/api/area/json)
- [Google Earth Engine Documentation](https://earthengine.google.com/)
- [Satellite Data Processing Guidelines](https://docs.scipy.org/)

## Contact

For questions, issues, or contributions:

- GitHub Issues: https://github.com/your-repo/smog-detection/issues
- Project Documentation: [docs/user_guide.md](docs/user_guide.md)
- API Reference: [docs/api_documentation.md](docs/api_documentation.md)

---

**Note**: This system processes satellite data for research purposes. For operational monitoring, please refer to official environmental agency resources.