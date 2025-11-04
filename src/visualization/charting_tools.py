"""
Charting tools for Islamabad Smog Detection System.

This module provides comprehensive charting and visualization capabilities for
air pollution data, including time series plots, correlation matrices,
distribution plots, and analytical visualizations.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import base64
import io

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Statistical plotting
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class ChartingTools:
    """Comprehensive charting tools for air pollution data visualization."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize charting tools.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.viz_config = self.config.get_section('visualization')

        # Chart configuration
        self.chart_config = self.viz_config.get('charts', {})
        self.figure_size = self.chart_config.get('figure_size', [12, 8])
        self.dpi = self.chart_config.get('dpi', 300)
        self.style = self.chart_config.get('style', 'seaborn-v0_8')

        # Set matplotlib style
        plt.style.use(self.style)
        sns.set_palette("husl")

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.exports_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('exports', 'data/exports')) / 'charts'
        )

        # Islamabad region
        self.region = GeoUtils.create_islamabad_region(buffer_km=50)

        # Color schemes for pollutants
        self.pollutant_colors = {
            'no2': '#FF6B6B',
            'so2': '#4ECDC4',
            'co': '#45B7D1',
            'o3': '#96CEB4',
            'aod': '#FFEAA7',
            'pm25': '#DDA0DD',
            'pm10': '#F0A0A0'
        }

        logger.info("Charting tools initialized")

    def create_time_series_chart(self, dataset: xr.Dataset, pollutants: List[str],
                                chart_type: str = 'line',
                                save_to_disk: bool = True) -> Dict[str, str]:
        """
        Create time series charts for multiple pollutants.

        Args:
            dataset: Input dataset with time dimension
            pollutants: List of pollutant names
            chart_type: Type of chart ('line', 'area', 'stacked')
            save_to_disk: Whether to save charts to disk

        Returns:
            Dictionary mapping pollutant to file paths
        """
        chart_paths = {}

        try:
            if 'time' not in dataset.dims:
                logger.warning("Dataset must have time dimension for time series charts")
                return chart_paths

            logger.info(f"Creating time series charts for: {pollutants}")

            # Extract time series data
            time_series_data = self._extract_time_series_data(dataset, pollutants)

            if not time_series_data:
                logger.error("No time series data available")
                return chart_paths

            # Create chart based on type
            if chart_type == 'line':
                chart_paths = self._create_line_charts(time_series_data, save_to_disk)
            elif chart_type == 'area':
                chart_paths = self._create_area_charts(time_series_data, save_to_disk)
            elif chart_type == 'stacked':
                chart_paths = self._create_stacked_charts(time_series_data, save_to_disk)
            elif chart_type == 'interactive':
                chart_paths = self._create_interactive_time_series(time_series_data, save_to_disk)
            else:
                logger.warning(f"Unknown chart type: {chart_type}")

        except Exception as e:
            logger.error(f"Failed to create time series charts: {e}")

        return chart_paths

    def _extract_time_series_data(self, dataset: xr.Dataset,
                                 pollutants: List[str]) -> Dict[str, pd.Series]:
        """Extract time series data for specified pollutants."""
        time_series_data = {}

        try:
            for pollutant in pollutants:
                # Find the data variable
                pollutant_var = None
                for var_name in dataset.data_vars:
                    if pollutant.lower() in var_name.lower():
                        pollutant_var = dataset[var_name]
                        break

                if pollutant_var is None:
                    logger.warning(f"Pollutant {pollutant} not found in dataset")
                    continue

                # Create spatial mean time series
                if 'lat' in pollutant_var.dims and 'lon' in pollutant_var.dims:
                    spatial_mean = pollutant_var.mean(dim=['lat', 'lon'])
                elif 'time' in pollutant_var.dims:
                    spatial_mean = pollutant_var
                else:
                    logger.warning(f"No temporal data found for {pollutant}")
                    continue

                # Convert to pandas Series
                ts_series = spatial_mean.to_pandas().dropna()
                if not ts_series.empty:
                    time_series_data[pollutant] = ts_series

        except Exception as e:
            logger.error(f"Failed to extract time series data: {e}")

        return time_series_data

    def _create_line_charts(self, time_series_data: Dict[str, pd.Series],
                           save_to_disk: bool) -> Dict[str, str]:
        """Create line charts for time series data."""
        chart_paths = {}

        try:
            for pollutant, ts_series in time_series_data.items():
                # Create figure
                fig, ax = plt.subplots(figsize=self.figure_size)

                # Plot time series
                ax.plot(ts_series.index, ts_series.values,
                       color=self.pollutant_colors.get(pollutant, 'blue'),
                       linewidth=2, alpha=0.8)

                # Customize chart
                ax.set_title(f'Islamabad {pollutant.upper()} Time Series', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Concentration', fontsize=12)
                ax.grid(True, alpha=0.3)

                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                # Add statistics
                mean_val = ts_series.mean()
                std_val = ts_series.std()
                ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
                ax.axhline(y=mean_val + std_val, color='orange', linestyle='--', alpha=0.5, label=f'+1 STD: {mean_val + std_val:.4f}')
                ax.axhline(y=mean_val - std_val, color='orange', linestyle='--', alpha=0.5, label=f'-1 STD: {mean_val - std_val:.4f}')

                ax.legend()

                plt.tight_layout()

                if save_to_disk:
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{pollutant}_timeseries_{timestamp}.png"
                    output_path = self.exports_path / filename

                    plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close()

                    chart_paths[pollutant] = str(output_path)
                    logger.info(f"Time series chart saved: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create line charts: {e}")

        return chart_paths

    def _create_area_charts(self, time_series_data: Dict[str, pd.Series],
                           save_to_disk: bool) -> Dict[str, str]:
        """Create area charts for time series data."""
        chart_paths = {}

        try:
            for pollutant, ts_series in time_series_data.items():
                fig, ax = plt.subplots(figsize=self.figure_size)

                # Create area chart
                ax.fill_between(ts_series.index, ts_series.values,
                               color=self.pollutant_colors.get(pollutant, 'blue'),
                               alpha=0.6, label=pollutant.upper())

                # Add line for emphasis
                ax.plot(ts_series.index, ts_series.values,
                       color=self.pollutant_colors.get(pollutant, 'blue'),
                       linewidth=2, alpha=0.8)

                # Customize chart
                ax.set_title(f'Islamabad {pollutant.upper()} Concentration Over Time',
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Concentration', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                plt.tight_layout()

                if save_to_disk:
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{pollutant}_area_chart_{timestamp}.png"
                    output_path = self.exports_path / filename

                    plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close()

                    chart_paths[pollutant] = str(output_path)

        except Exception as e:
            logger.error(f"Failed to create area charts: {e}")

        return chart_paths

    def _create_stacked_charts(self, time_series_data: Dict[str, pd.Series],
                              save_to_disk: bool) -> Dict[str, str]:
        """Create stacked area chart for multiple pollutants."""
        chart_paths = {}

        try:
            if len(time_series_data) < 2:
                logger.warning("Need at least 2 pollutants for stacked chart")
                return chart_paths

            # Combine all time series into DataFrame
            combined_df = pd.DataFrame(time_series_data)

            # Create stacked area chart
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Plot stacked areas
            ax.stackplot(combined_df.index, combined_df.values.T,
                         labels=[col.upper() for col in combined_df.columns],
                         colors=[self.pollutant_colors.get(col.lower(), 'blue')
                                for col in combined_df.columns],
                         alpha=0.7)

            # Customize chart
            ax.set_title('Islamabad Multi-Pollutant Concentrations Over Time',
                       fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Concentration', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"multi_pollutant_stacked_{timestamp}.png"
                output_path = self.exports_path / filename

                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                chart_paths['stacked'] = str(output_path)

        except Exception as e:
            logger.error(f"Failed to create stacked chart: {e}")

        return chart_paths

    def _create_interactive_time_series(self, time_series_data: Dict[str, pd.Series],
                                       save_to_disk: bool) -> Dict[str, str]:
        """Create interactive time series chart using Plotly."""
        chart_paths = {}

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive charts")
            return chart_paths

        try:
            # Create figure
            fig = go.Figure()

            # Add trace for each pollutant
            for pollutant, ts_series in time_series_data.items():
                fig.add_trace(go.Scatter(
                    x=ts_series.index,
                    y=ts_series.values,
                    mode='lines',
                    name=pollutant.upper(),
                    line=dict(color=self.pollutant_colors.get(pollutant, 'blue'), width=2)
                ))

            # Update layout
            fig.update_layout(
                title='Islamabad Multi-Pollutant Time Series',
                xaxis_title='Date',
                yaxis_title='Concentration',
                hovermode='x unified',
                showlegend=True,
                width=1000,
                height=600
            )

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"interactive_timeseries_{timestamp}.html"
                output_path = self.exports_path / filename

                fig.write_html(str(output_path))
                chart_paths['interactive'] = str(output_path)

        except Exception as e:
            logger.error(f"Failed to create interactive time series: {e}")

        return chart_paths

    def create_distribution_plots(self, dataset: xr.Dataset, pollutants: List[str],
                                save_to_disk: bool = True) -> Dict[str, str]:
        """
        Create distribution plots for pollutants.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            save_to_disk: Whether to save plots to disk

        Returns:
            Dictionary mapping pollutant to file paths
        """
        plot_paths = {}

        try:
            # Extract data for distribution analysis
            distribution_data = self._extract_distribution_data(dataset, pollutants)

            if not distribution_data:
                logger.error("No distribution data available")
                return plot_paths

            # Create individual distribution plots
            for pollutant, data in distribution_data.items():
                plot_paths.update(self._create_distribution_plot(data, pollutant, save_to_disk))

            # Create combined distribution plot
            plot_paths['combined'] = self._create_combined_distribution_plot(
                distribution_data, save_to_disk
            )

        except Exception as e:
            logger.error(f"Failed to create distribution plots: {e}")

        return plot_paths

    def _extract_distribution_data(self, dataset: xr.Dataset,
                                  pollutants: List[str]) -> Dict[str, np.ndarray]:
        """Extract distribution data for specified pollutants."""
        distribution_data = {}

        try:
            for pollutant in pollutants:
                # Find the data variable
                pollutant_var = None
                for var_name in dataset.data_vars:
                    if pollutant.lower() in var_name.lower():
                        pollutant_var = dataset[var_name]
                        break

                if pollutant_var is None:
                    continue

                # Extract all data values (flatten spatial dimensions)
                data_values = pollutant_var.values
                valid_data = data_values[~np.isnan(data_values)]

                if len(valid_data) > 0:
                    distribution_data[pollutant] = valid_data

        except Exception as e:
            logger.error(f"Failed to extract distribution data: {e}")

        return distribution_data

    def _create_distribution_plot(self, data: np.ndarray, pollutant: str,
                                 save_to_disk: bool) -> Dict[str, str]:
        """Create individual distribution plot for a pollutant."""
        plot_paths = {}

        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Islamabad {pollutant.upper()} Distribution Analysis', fontsize=16)

            # Histogram
            axes[0, 0].hist(data, bins=50, alpha=0.7, color=self.pollutant_colors.get(pollutant, 'blue'), edgecolor='black')
            axes[0, 0].set_title('Distribution Histogram')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

            # Box plot
            axes[0, 1].boxplot(data, vert=True)
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)

            # Q-Q plot
            if SCIPY_AVAILABLE:
                stats.probplot(data, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
                axes[1, 0].grid(True, alpha=0.3)

            # Statistics text
            axes[1, 1].axis('off')
            stats_text = f"""
            Statistical Summary:
            ─────────────────────
            Count: {len(data):,}
            Mean: {np.mean(data):.4f}
            Median: {np.median(data):.4f}
            Std Dev: {np.std(data):.4f}
            Min: {np.min(data):.4f}
            Max: {np.max(data):.4f}
            25th Percentile: {np.percentile(data, 25):.4f}
            75th Percentile: {np.percentile(data, 75):.4f}
            Skewness: {stats.skew(data):.4f}
            Kurtosis: {stats.kurtosis(data):.4f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                            fontfamily='monospace')

            plt.tight_layout()

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{pollutant}_distribution_{timestamp}.png"
                output_path = self.exports_path / filename

                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                plot_paths[f"{pollutant}_distribution"] = str(output_path)

        except Exception as e:
            logger.error(f"Failed to create distribution plot for {pollutant}: {e}")

        return plot_paths

    def _create_combined_distribution_plot(self, distribution_data: Dict[str, np.ndarray],
                                         save_to_disk: bool) -> Optional[str]:
        """Create combined distribution plot for all pollutants."""
        try:
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Multi-Pollutant Distribution Comparison', fontsize=16)

            # Combined histogram
            for pollutant, data in distribution_data.items():
                axes[0, 0].hist(data, bins=50, alpha=0.6, label=pollutant.upper(),
                             color=self.pollutant_colors.get(pollutant, 'blue'))
            axes[0, 0].set_title('Combined Histogram')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Combined box plot
            box_data = [data for data in distribution_data.values()]
            box_labels = list(distribution_data.keys())
            bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)

            # Color box plots
            for patch, pollutant in zip(bp['boxes'], distribution_data.keys()):
                patch.set_facecolor(self.pollutant_colors.get(pollutant, 'blue'))
                patch.set_alpha(0.7)

            axes[0, 1].set_title('Combined Box Plot')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)

            # Statistics comparison table
            axes[1, 0].axis('off')
            stats_data = []
            for pollutant, data in distribution_data.items():
                stats_data.append([
                    pollutant.upper(),
                    f"{np.mean(data):.4f}",
                    f"{np.std(data):.4f}",
                    f"{np.min(data):.4f}",
                    f"{np.max(data):.4f}",
                    f"{stats.skew(data):.4f}"
                ])

            stats_table = axes[1, 0].table(cellText=stats_data,
                                          colLabels=['Pollutant', 'Mean', 'Std', 'Min', 'Max', 'Skew'],
                                          cellLoc='center', loc='center')
            stats_table.auto_set_font_size(False)
            stats_table.set_fontsize(10)
            stats_table.scale(1, 1.5)

            # Correlation matrix (if multiple pollutants)
            if len(distribution_data) >= 2:
                correlation_data = []
                for pollutant1, data1 in distribution_data.items():
                    row = []
                    for pollutant2, data2 in distribution_data.items():
                        if pollutant1 == pollutant2:
                            row.append(1.0)
                        else:
                            corr, _ = stats.pearsonr(data1, data2)
                            row.append(corr)
                    correlation_data.append(row)

                im = axes[1, 1].imshow(correlation_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                axes[1, 1].set_xticks(range(len(distribution_data)))
                axes[1, 1].set_yticks(range(len(distribution_data)))
                axes[1, 1].set_xticklabels([p.upper() for p in distribution_data.keys()])
                axes[1, 1].set_yticklabels([p.upper() for p in distribution_data.keys()])
                axes[1, 1].set_title('Correlation Matrix')

                # Add correlation values
                for i in range(len(distribution_data)):
                    for j in range(len(distribution_data)):
                        text = axes[1, 1].text(j, i, f'{correlation_data[i][j]:.2f}',
                                             ha="center", va="center", color="black")

                plt.colorbar(im, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'Need multiple pollutants\nfor correlation matrix',
                               ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"combined_distribution_{timestamp}.png"
                output_path = self.exports_path / filename

                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create combined distribution plot: {e}")
            return None

    def create_correlation_analysis(self, dataset: xr.Dataset, pollutants: List[str],
                                  save_to_disk: bool = True) -> Dict[str, str]:
        """
        Create correlation analysis plots.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            save_to_disk: Whether to save plots to disk

        Returns:
            Dictionary with plot file paths
        """
        plot_paths = {}

        try:
            if len(pollutants) < 2:
                logger.warning("Need at least 2 pollutants for correlation analysis")
                return plot_paths

            # Extract time series data
            time_series_data = self._extract_time_series_data(dataset, pollutants)

            if len(time_series_data) < 2:
                logger.warning("Insufficient data for correlation analysis")
                return plot_paths

            # Create correlation matrix plot
            plot_paths['correlation_matrix'] = self._create_correlation_matrix_plot(
                time_series_data, save_to_disk
            )

            # Create scatter plot matrix
            plot_paths['scatter_matrix'] = self._create_scatter_matrix(
                time_series_data, save_to_disk
            )

            # Create pairplot if seaborn available
            if hasattr(sns, 'pairplot'):
                plot_paths['pairplot'] = self._create_pairplot(
                    time_series_data, save_to_disk
                )

        except Exception as e:
            logger.error(f"Failed to create correlation analysis: {e}")

        return plot_paths

    def _create_correlation_matrix_plot(self, time_series_data: Dict[str, pd.Series],
                                        save_to_disk: bool) -> Optional[str]:
        """Create correlation matrix heatmap."""
        try:
            # Create DataFrame
            df = pd.DataFrame(time_series_data)

            # Calculate correlation matrix
            corr_matrix = df.corr()

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Create heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)

            ax.set_title('Pollutant Correlation Matrix', fontsize=16, fontweight='bold')

            plt.tight_layout()

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"correlation_matrix_{timestamp}.png"
                output_path = self.exports_path / filename

                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create correlation matrix plot: {e}")
            return None

    def _create_scatter_matrix(self, time_series_data: Dict[str, pd.Series],
                              save_to_disk: bool) -> Optional[str]:
        """Create scatter plot matrix."""
        try:
            # Create DataFrame
            df = pd.DataFrame(time_series_data)

            # Create scatter matrix
            n_vars = len(df.columns)
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(15, 15))
            fig.suptitle('Pollutant Scatter Matrix', fontsize=16)

            for i, col1 in enumerate(df.columns):
                for j, col2 in enumerate(df.columns):
                    ax = axes[i, j]

                    if i == j:
                        # Diagonal: histogram
                        ax.hist(df[col1], bins=30, alpha=0.7, color='blue', edgecolor='black')
                        ax.set_title(col1.upper())
                    else:
                        # Off-diagonal: scatter plot
                        ax.scatter(df[col2], df[col1], alpha=0.6, s=20)

                        # Add correlation coefficient
                        corr, _ = stats.pearsonr(df[col2], df[col1])
                        ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    if i == n_vars - 1:
                        ax.set_xlabel(col2.upper())
                    if j == 0:
                        ax.set_ylabel(col1.upper())

                    ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"scatter_matrix_{timestamp}.png"
                output_path = self.exports_path / filename

                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create scatter matrix: {e}")
            return None

    def _create_pairplot(self, time_series_data: Dict[str, pd.Series],
                        save_to_disk: bool) -> Optional[str]:
        """Create seaborn pairplot."""
        try:
            # Create DataFrame
            df = pd.DataFrame(time_series_data)

            # Create pairplot
            g = sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30})

            g.fig.suptitle('Pollutant Pairplot', y=1.02)

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"pairplot_{timestamp}.png"
                output_path = self.exports_path / filename

                g.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create pairplot: {e}")
            return None

    def create_seasonal_analysis_charts(self, dataset: xr.Dataset, pollutants: List[str],
                                       save_to_disk: bool = True) -> Dict[str, str]:
        """
        Create seasonal analysis charts.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            save_to_disk: Whether to save charts to disk

        Returns:
            Dictionary with chart file paths
        """
        chart_paths = {}

        try:
            # Extract seasonal data
            seasonal_data = self._extract_seasonal_data(dataset, pollutants)

            if not seasonal_data:
                logger.error("No seasonal data available")
                return chart_paths

            # Create seasonal patterns chart
            chart_paths['seasonal_patterns'] = self._create_seasonal_patterns_chart(
                seasonal_data, save_to_disk
            )

            # Create monthly comparison chart
            chart_paths['monthly_comparison'] = self._create_monthly_comparison_chart(
                seasonal_data, save_to_disk
            )

            # Create seasonal heatmap
            chart_paths['seasonal_heatmap'] = self._create_seasonal_heatmap(
                seasonal_data, save_to_disk
            )

        except Exception as e:
            logger.error(f"Failed to create seasonal analysis charts: {e}")

        return chart_paths

    def _extract_seasonal_data(self, dataset: xr.Dataset,
                              pollutants: List[str]) -> Dict[str, pd.DataFrame]:
        """Extract seasonal data for specified pollutants."""
        seasonal_data = {}

        try:
            for pollutant in pollutants:
                # Find the data variable
                pollutant_var = None
                for var_name in dataset.data_vars:
                    if pollutant.lower() in var_name.lower():
                        pollutant_var = dataset[var_name]
                        break

                if pollutant_var is None or 'time' not in pollutant_var.dims:
                    continue

                # Create spatial mean time series
                if 'lat' in pollutant_var.dims and 'lon' in pollutant_var.dims:
                    spatial_mean = pollutant_var.mean(dim=['lat', 'lon'])
                else:
                    spatial_mean = pollutant_var

                # Convert to pandas DataFrame with seasonal features
                ts_series = spatial_mean.to_pandas()
                if ts_series.empty:
                    continue

                # Add seasonal features
                df = pd.DataFrame(ts_series, columns=['value'])
                df['month'] = df.index.month
                df['season'] = df.index.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
                df['day_of_year'] = df.index.dayofyear

                seasonal_data[pollutant] = df

        except Exception as e:
            logger.error(f"Failed to extract seasonal data: {e}")

        return seasonal_data

    def _create_seasonal_patterns_chart(self, seasonal_data: Dict[str, pd.DataFrame],
                                       save_to_disk: bool) -> Optional[str]:
        """Create seasonal patterns chart."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Seasonal Pollution Patterns', fontsize=16)

            # Seasonal box plots
            seasonal_means = []
            season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
            pollutant_labels = list(seasonal_data.keys())

            for i, (pollutant, df) in enumerate(seasonal_data.items()):
                seasonal_means.append([
                    df[df['season'] == 1]['value'].mean(),  # Winter
                    df[df['season'] == 2]['value'].mean(),  # Spring
                    df[df['season'] == 3]['value'].mean(),  # Summer
                    df[df['season'] == 4]['value'].mean()   # Fall
                ])

            # Seasonal bar chart
            x = np.arange(len(season_labels))
            width = 0.8 / len(pollutant_labels)

            for i, (pollutant, means) in enumerate(zip(pollutant_labels, seasonal_means)):
                axes[0, 0].bar(x + i * width, means, width,
                               label=pollutant.upper(),
                               color=self.pollutant_colors.get(pollutant, 'blue'),
                               alpha=0.7)

            axes[0, 0].set_title('Seasonal Mean Concentrations')
            axes[0, 0].set_xlabel('Season')
            axes[0, 0].set_ylabel('Mean Concentration')
            axes[0, 0].set_xticks(x + width * (len(pollutant_labels) - 1) / 2)
            axes[0, 0].set_xticklabels(season_labels)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Monthly patterns
            for pollutant, df in seasonal_data.items():
                monthly_means = df.groupby('month')['value'].mean()
                axes[0, 1].plot(monthly_means.index, monthly_means.values,
                               marker='o', label=pollutant.upper(),
                               color=self.pollutant_colors.get(pollutant, 'blue'),
                               linewidth=2)

            axes[0, 1].set_title('Monthly Patterns')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Mean Concentration')
            axes[0, 1].set_xticks(range(1, 13))
            axes[0, 1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Day of year patterns (violin plots)
            if len(seasonal_data) > 0:
                # Combine all pollutants for day of year analysis
                all_data = []
                all_labels = []
                for pollutant, df in seasonal_data.items():
                    all_data.extend(df['value'].values)
                    all_labels.extend([pollutant.upper()] * len(df))

                # Simple scatter plot for day of year
                for pollutant, df in seasonal_data.items():
                    axes[1, 0].scatter(df['day_of_year'], df['value'],
                                       alpha=0.5, s=10,
                                       label=pollutant.upper(),
                                       color=self.pollutant_colors.get(pollutant, 'blue'))

                axes[1, 0].set_title('Day of Year Patterns')
                axes[1, 0].set_xlabel('Day of Year')
                axes[1, 0].set_ylabel('Concentration')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Summary statistics table
            axes[1, 1].axis('off')
            summary_data = []
            for pollutant, df in seasonal_data.items():
                summary_data.append([
                    pollutant.upper(),
                    f"{df['value'].mean():.4f}",
                    f"{df['value'].std():.4f}",
                    f"{df.groupby('season')['value'].mean().max():.4f}",
                    f"{df.groupby('season')['value'].mean().min():.4f}"
                ])

            summary_table = axes[1, 1].table(cellText=summary_data,
                                           colLabels=['Pollutant', 'Overall Mean', 'Std', 'Max Season', 'Min Season'],
                                           cellLoc='center', loc='center')
            summary_table.auto_set_font_size(False)
            summary_table.set_fontsize(10)
            summary_table.scale(1, 1.5)

            plt.tight_layout()

            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"seasonal_patterns_{timestamp}.png"
                output_path = self.exports_path / filename

                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()

                return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create seasonal patterns chart: {e}")
            return None

    def _create_monthly_comparison_chart(self, seasonal_data: Dict[str, pd.DataFrame],
                                        save_to_disk: bool) -> Optional[str]:
        """Create monthly comparison chart."""
        # This would be similar to seasonal patterns but focused on monthly data
        # Implementation omitted for brevity
        return None

    def _create_seasonal_heatmap(self, seasonal_data: Dict[str, pd.DataFrame],
                                save_to_disk: bool) -> Optional[str]:
        """Create seasonal heatmap."""
        # This would create a heatmap of seasonal patterns
        # Implementation omitted for brevity
        return None

    def export_charts_data(self, chart_data: Dict[str, Any],
                          output_format: str = 'json') -> str:
        """
        Export chart data for external use.

        Args:
            chart_data: Chart data dictionary
            output_format: Export format ('json', 'csv', 'excel')

        Returns:
            File path
        """
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

            if output_format == 'json':
                filename = f"chart_data_{timestamp}.json"
                output_path = self.exports_path / filename

                with open(output_path, 'w') as f:
                    json.dump(chart_data, f, indent=2, default=str)

            elif output_format == 'csv':
                # Convert to DataFrame if possible
                if isinstance(chart_data, dict):
                    df = pd.DataFrame(chart_data)
                    filename = f"chart_data_{timestamp}.csv"
                    output_path = self.exports_path / filename
                    df.to_csv(output_path, index=False)

            elif output_format == 'excel':
                if isinstance(chart_data, dict):
                    df = pd.DataFrame(chart_data)
                    filename = f"chart_data_{timestamp}.xlsx"
                    output_path = self.exports_path / filename
                    df.to_excel(output_path, index=False)

            logger.info(f"Chart data exported: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export chart data: {e}")
            return ""

    def create_dashboard_summary(self, dataset: xr.Dataset,
                               pollutants: List[str]) -> Dict[str, Any]:
        """
        Create comprehensive dashboard summary with key metrics.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names

        Returns:
            Dictionary with dashboard summary data
        """
        summary = {}

        try:
            # Extract basic statistics
            time_series_data = self._extract_time_series_data(dataset, pollutants)
            distribution_data = self._extract_distribution_data(dataset, pollutants)

            # Current values
            current_values = {}
            for pollutant, ts_series in time_series_data.items():
                if not ts_series.empty:
                    current_values[pollutant] = float(ts_series.iloc[-1])

            summary['current_values'] = current_values

            # Recent trends
            recent_trends = {}
            for pollutant, ts_series in time_series_data.items():
                if len(ts_series) >= 7:
                    recent_mean = ts_series.tail(7).mean()
                    previous_mean = ts_series.tail(14).head(7).mean()
                    trend = (recent_mean - previous_mean) / previous_mean * 100
                    recent_trends[pollutant] = {
                        'trend_percent': float(trend),
                        'direction': 'increasing' if trend > 0 else 'decreasing'
                    }

            summary['recent_trends'] = recent_trends

            # Statistics summary
            stats_summary = {}
            for pollutant, data in distribution_data.items():
                stats_summary[pollutant] = {
                    'mean': float(np.mean(data)),
                    'median': float(np.median(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }

            summary['statistics'] = stats_summary

            # Quality indicators
            quality_indicators = {}
            for pollutant, ts_series in time_series_data.items():
                if not ts_series.empty:
                    # Calculate data completeness
                    expected_days = (ts_series.index.max() - ts_series.index.min()).days
                    actual_days = len(ts_series)
                    completeness = actual_days / expected_days * 100

                    quality_indicators[pollutant] = {
                        'data_completeness': float(completeness),
                        'data_points': int(len(ts_series)),
                        'date_range': [str(ts_series.index.min()), str(ts_series.index.max())]
                    }

            summary['quality_indicators'] = quality_indicators

            # Alert levels (simplified)
            alert_levels = {}
            threshold_levels = {
                'no2': {'good': 20, 'moderate': 40, 'poor': 60},
                'so2': {'good': 10, 'moderate': 20, 'poor': 30},
                'co': {'good': 0.05, 'moderate': 0.1, 'poor': 0.2}
            }

            for pollutant, current_val in current_values.items():
                thresholds = threshold_levels.get(pollutant, {})
                if current_val < thresholds.get('good', float('inf')):
                    level = 'good'
                elif current_val < thresholds.get('moderate', float('inf')):
                    level = 'moderate'
                else:
                    level = 'poor'

                alert_levels[pollutant] = {
                    'current_level': level,
                    'current_value': current_val,
                    'thresholds': thresholds
                }

            summary['alert_levels'] = alert_levels

        except Exception as e:
            logger.error(f"Failed to create dashboard summary: {e}")

        return summary