"""
Mapping tools for Islamabad Smog Detection System.

This module provides comprehensive mapping and visualization capabilities for
air pollution data, including interactive maps, heat maps, and spatial
analysis visualizations for the Islamabad region.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import base64
import io

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Mapping libraries
try:
    import folium
    from folium import plugins
    import branca.colormap as cm
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils

logger = logging.getLogger(__name__)


class MappingTools:
    """Comprehensive mapping tools for air pollution visualization."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize mapping tools.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.viz_config = self.config.get_section('visualization')

        # Mapping configuration
        self.map_config = self.viz_config.get('maps', {})
        self.default_center = self.map_config.get('default_center', [33.6844, 73.0479])
        self.default_zoom = self.map_config.get('default_zoom', 10)

        # Color schemes for different pollutants
        self.color_schemes = {
            'no2': 'YlOrRd',
            'so2': 'PuRd',
            'co': 'Oranges',
            'o3': 'BuPu',
            'aod': 'Greys',
            'pm25': 'Reds',
            'pm10': 'Oranges'
        }

        # Islamabad region
        self.region = GeoUtils.create_islamabad_region(buffer_km=50)

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.exports_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('exports', 'data/exports')) / 'maps'
        )

        logger.info("Mapping tools initialized")

    def create_pollution_heatmap(self, dataset: xr.Dataset, pollutant: str,
                                 timestamp: Optional[str] = None,
                                 save_to_disk: bool = True) -> Optional[str]:
        """
        Create pollution heatmap using Folium.

        Args:
            dataset: Input dataset with spatial data
            pollutant: Pollutant name (no2, so2, co, o3, aod)
            timestamp: Specific timestamp to visualize
            save_to_disk: Whether to save map to disk

        Returns:
            HTML file path or None if failed
        """
        if not FOLIUM_AVAILABLE:
            logger.error("Folium not available for mapping")
            return None

        try:
            logger.info(f"Creating pollution heatmap for {pollutant}")

            # Extract data for visualization
            data_var = self._extract_pollutant_data(dataset, pollutant, timestamp)

            if data_var is None:
                logger.error(f"No data found for pollutant {pollutant}")
                return None

            # Create base map
            m = folium.Map(
                location=self.default_center,
                zoom_start=self.default_zoom,
                tiles='OpenStreetMap'
            )

            # Add layer control
            folium.LayerControl().add_to(m)

            # Add Islamabad region boundary
            self._add_region_boundary(m)

            # Create heatmap data
            heatmap_data = self._prepare_heatmap_data(data_var)

            if heatmap_data:
                # Add heatmap layer
                heatmap = plugins.HeatMap(
                    heatmap_data,
                    name=f'{pollutant.upper()} Heatmap',
                    radius=15,
                    blur=10,
                    gradient=self._get_color_gradient(pollutant)
                )
                m.add_child(heatmap)

            # Add colorbar
            self._add_colorbar(m, data_var, pollutant)

            # Add measurement points
            self._add_measurement_points(m, data_var, pollutant)

            # Add title and legend
            self._add_map_metadata(m, pollutant, timestamp)

            # Save map
            if save_to_disk:
                timestamp_str = timestamp.replace('-', '_').replace(':', '_') if timestamp else 'latest'
                filename = f"{pollutant}_heatmap_{timestamp_str}.html"
                output_path = self.exports_path / filename

                m.save(str(output_path))
                logger.info(f"Pollution heatmap saved to {output_path}")

                return str(output_path)
            else:
                # Return as HTML string
                html = m._repr_html_()
                return html

        except Exception as e:
            logger.error(f"Failed to create pollution heatmap: {e}")
            return None

    def create_interactive_pollution_map(self, dataset: xr.Dataset, pollutant: str,
                                       save_to_disk: bool = True) -> Optional[str]:
        """
        Create interactive pollution map with time slider using Plotly.

        Args:
            dataset: Input dataset
            pollutant: Pollutant name
            save_to_disk: Whether to save map to disk

        Returns:
            HTML file path or None if failed
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for interactive mapping")
            return None

        try:
            logger.info(f"Creating interactive pollution map for {pollutant}")

            # Extract time series data
            time_series_data = self._extract_time_series_data(dataset, pollutant)

            if time_series_data is None:
                logger.error(f"No time series data found for pollutant {pollutant}")
                return None

            # Create interactive map with animation
            fig = go.Figure()

            # Add frames for each time step
            frames = []
            for timestamp, data_slice in time_series_data.items():
                frame_data = self._create_map_frame(data_slice, pollutant, timestamp)
                frames.append(frame_data)

            fig.frames = frames

            # Create initial frame
            initial_data = frames[0]['data'][0] if frames else None

            if initial_data:
                fig.add_trace(initial_data)

            # Add layout with animation controls
            fig.update_layout(
                title=f'Islamabad {pollutant.upper()} Concentration Over Time',
                geo=dict(
                    center=dict(lat=self.default_center[0], lon=self.default_center[1]),
                    projection_type='mercator',
                    showland=True,
                    landcolor='lightgray',
                    showocean=True,
                    oceancolor='lightblue',
                    showframe=False,
                    showcountries=True,
                    countrycolor='gray'
                ),
                updatemenus=[dict(
                    type='buttons',
                    direction='left',
                    buttons=list([
                        dict(
                            args=[None, {'frame': {'duration': 500, 'redraw': True},
                                     'fromcurrent': True, 'transition': {'duration': 300}}],
                            label='Play',
                            method='animate'
                        ),
                        dict(
                            args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                     'fromcurrent': True, 'transition': {'duration': 0}}],
                            label='Pause',
                            method='animate'
                        )
                    ]),
                    pad=dict(r=10, t=87),
                    showactive=False,
                    x=0.011,
                    xanchor='right',
                    y=0,
                    yanchor='top'
                )],
                sliders=[dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue={'font': {'size': 20}},
                    transition={'duration': 300, 'easing': 'cubic-in-out'},
                    pad={'b': 10, 't': 50},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[dict(
                        args=[[frame.name], {'frame': {'duration': 400, 'redraw': True},
                                           'mode': 'immediate', 'transition': {'duration': 300}}],
                        label=frame.name,
                        method='animate'
                    ) for frame in frames]
                )]
            )

            # Save map
            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"interactive_{pollutant}_map_{timestamp}.html"
                output_path = self.exports_path / filename

                fig.write_html(str(output_path))
                logger.info(f"Interactive pollution map saved to {output_path}")

                return str(output_path)
            else:
                return fig.to_html()

        except Exception as e:
            logger.error(f"Failed to create interactive pollution map: {e}")
            return None

    def create_multi_pollutant_comparison_map(self, dataset: xr.Dataset,
                                            pollutants: List[str],
                                            save_to_disk: bool = True) -> Optional[str]:
        """
        Create map comparing multiple pollutants.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            save_to_disk: Whether to save map to disk

        Returns:
            HTML file path or None if failed
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for multi-pollutant mapping")
            return None

        try:
            logger.info(f"Creating multi-pollutant comparison map for: {pollutants}")

            # Create subplots for each pollutant
            specs = [[{'type': 'geo'} for _ in pollutants]]
            fig = make_subplots(
                rows=1, cols=len(pollutants),
                specs=specs,
                subplot_titles=[p.upper() for p in pollutants],
                horizontal_spacing=0.01
            )

            # Add trace for each pollutant
            for i, pollutant in enumerate(pollutants):
                data_slice = self._extract_pollutant_data(dataset, pollutant)

                if data_slice is not None:
                    trace = self._create_comparison_trace(data_slice, pollutant, i+1)
                    fig.add_trace(trace, row=1, col=i+1)

            # Update layout
            fig.update_layout(
                title='Multi-Pollutant Comparison - Islamabad',
                height=600,
                showlegend=False
            )

            # Update geo settings for each subplot
            for i in range(1, len(pollutants) + 1):
                fig.update_geos(
                    center=dict(lat=self.default_center[0], lon=self.default_center[1]),
                    projection_type='mercator',
                    showland=True,
                    landcolor='lightgray',
                    showframe=False,
                    row=1, col=i
                )

            # Save map
            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                pollutant_str = '_'.join(pollutants)
                filename = f"multi_pollutant_{pollutant_str}_{timestamp}.html"
                output_path = self.exports_path / filename

                fig.write_html(str(output_path))
                logger.info(f"Multi-pollutant comparison map saved to {output_path}")

                return str(output_path)
            else:
                return fig.to_html()

        except Exception as e:
            logger.error(f"Failed to create multi-pollutant comparison map: {e}")
            return None

    def create_wind_pollution_map(self, pollution_data: xr.Dataset,
                                 wind_data: Optional[pd.DataFrame] = None,
                                 save_to_disk: bool = True) -> Optional[str]:
        """
        Create map showing pollution with wind vectors.

        Args:
            pollution_data: Pollution dataset
            wind_data: Wind speed and direction data
            save_to_disk: Whether to save map to disk

        Returns:
            HTML file path or None if failed
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for wind-pollution mapping")
            return None

        try:
            logger.info("Creating wind-pollution map")

            # Create base pollution map
            pollutant = 'no2'  # Default to NO2 for wind analysis
            data_slice = self._extract_pollutant_data(pollution_data, pollutant)

            if data_slice is None:
                logger.error("No pollution data available for wind map")
                return None

            fig = go.Figure()

            # Add pollution contour
            contour = self._create_pollution_contour(data_slice, pollutant)
            fig.add_trace(contour)

            # Add wind vectors if available
            if wind_data is not None:
                wind_trace = self._create_wind_vectors(wind_data)
                fig.add_trace(wind_trace)

            # Update layout
            fig.update_layout(
                title='Islamabad Pollution with Wind Patterns',
                geo=dict(
                    center=dict(lat=self.default_center[0], lon=self.default_center[1]),
                    projection_type='mercator',
                    showland=True,
                    landcolor='lightgray'
                )
            )

            # Save map
            if save_to_disk:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"wind_pollution_map_{timestamp}.html"
                output_path = self.exports_path / filename

                fig.write_html(str(output_path))
                logger.info(f"Wind-pollution map saved to {output_path}")

                return str(output_path)
            else:
                return fig.to_html()

        except Exception as e:
            logger.error(f"Failed to create wind-pollution map: {e}")
            return None

    def _extract_pollutant_data(self, dataset: xr.Dataset, pollutant: str,
                               timestamp: Optional[str] = None) -> Optional[xr.DataArray]:
        """Extract pollutant data for visualization."""
        try:
            # Find the data variable
            pollutant_var = None
            for var_name in dataset.data_vars:
                if pollutant.lower() in var_name.lower():
                    pollutant_var = dataset[var_name]
                    break

            if pollutant_var is None:
                logger.warning(f"Pollutant {pollutant} not found in dataset")
                return None

            # Handle time dimension
            if 'time' in pollutant_var.dims:
                if timestamp:
                    # Extract specific timestamp
                    time_dim = pollutant_var['time']
                    time_values = pd.to_datetime(time_dim.values)

                    # Find closest time to requested timestamp
                    target_time = pd.to_datetime(timestamp)
                    closest_idx = np.argmin(np.abs(time_values - target_time))
                    data_slice = pollutant_var.isel(time=closest_idx)
                else:
                    # Use most recent time
                    data_slice = pollutant_var.isel(time=-1)
            else:
                data_slice = pollutant_var

            # Handle spatial dimensions
            if 'lat' in data_slice.dims and 'lon' in data_slice.dims:
                return data_slice
            else:
                logger.warning(f"Data for {pollutant} does not have spatial dimensions")
                return None

        except Exception as e:
            logger.error(f"Failed to extract {pollutant} data: {e}")
            return None

    def _extract_time_series_data(self, dataset: xr.Dataset,
                                 pollutant: str) -> Optional[Dict[str, xr.DataArray]]:
        """Extract time series data for animation."""
        try:
            pollutant_var = None
            for var_name in dataset.data_vars:
                if pollutant.lower() in var_name.lower():
                    pollutant_var = dataset[var_name]
                    break

            if pollutant_var is None or 'time' not in pollutant_var.dims:
                return None

            time_series = {}
            for i, time_val in enumerate(pollutant_var['time'].values):
                time_str = str(pd.to_datetime(time_val))
                time_series[time_str] = pollutant_var.isel(time=i)

            return time_series

        except Exception as e:
            logger.error(f"Failed to extract time series data: {e}")
            return None

    def _prepare_heatmap_data(self, data_var: xr.DataArray) -> List[List[float]]:
        """Prepare data for Folium heatmap."""
        try:
            heatmap_data = []

            if 'lat' in data_var.dims and 'lon' in data_var.dims:
                lat_coords = data_var['lat'].values
                lon_coords = data_var['lon'].values
                data_values = data_var.values

                # Sample points to avoid overcrowding
                step = max(1, len(lat_coords) // 20)
                for i in range(0, len(lat_coords), step):
                    for j in range(0, len(lon_coords), step):
                        lat = lat_coords[i]
                        lon = lon_coords[j]
                        value = data_values[i, j]

                        if not np.isnan(value) and value > 0:
                            heatmap_data.append([lat, lon, float(value)])

            return heatmap_data

        except Exception as e:
            logger.error(f"Failed to prepare heatmap data: {e}")
            return []

    def _get_color_gradient(self, pollutant: str) -> List[str]:
        """Get color gradient for specific pollutant."""
        gradients = {
            'no2': ['blue', 'cyan', 'yellow', 'orange', 'red'],
            'so2': ['purple', 'red', 'orange', 'yellow'],
            'co': ['green', 'yellow', 'orange', 'red', 'darkred'],
            'o3': ['blue', 'purple', 'pink', 'red'],
            'aod': ['white', 'lightgray', 'gray', 'darkgray', 'black']
        }
        return gradients.get(pollutant, ['blue', 'green', 'yellow', 'red'])

    def _add_colorbar(self, m, data_var: xr.DataArray, pollutant: str):
        """Add colorbar to Folium map."""
        try:
            # Calculate min and max values
            valid_data = data_var.values[~np.isnan(data_var.values)]
            if len(valid_data) > 0:
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))

                # Create colormap
                colormap = cm.LinearColormap(
                    colors=self._get_color_gradient(pollutant),
                    vmin=min_val,
                    vmax=max_val
                )

                # Add colormap to map
                colormap.caption = f'{pollutant.upper()} Concentration'
                m.add_child(colormap)

        except Exception as e:
            logger.error(f"Failed to add colorbar: {e}")

    def _add_region_boundary(self, m):
        """Add Islamabad region boundary to map."""
        try:
            bbox = self.region['bounding_box']

            # Create rectangle for Islamabad region
            bounds = [
                [bbox['south'], bbox['west']],
                [bbox['north'], bbox['east']]
            ]

            folium.Rectangle(
                bounds=bounds,
                popup='Islamabad Region',
                color='red',
                weight=2,
                fill=False,
                opacity=0.8
            ).add_to(m)

        except Exception as e:
            logger.error(f"Failed to add region boundary: {e}")

    def _add_measurement_points(self, m, data_var: xr.DataArray, pollutant: str):
        """Add measurement points to map."""
        try:
            if 'lat' in data_var.dims and 'lon' in data_var.dims:
                lat_coords = data_var['lat'].values
                lon_coords = data_var['lon'].values
                data_values = data_var.values

                # Sample measurement points
                step = max(1, len(lat_coords) // 10)
                for i in range(0, len(lat_coords), step):
                    for j in range(0, len(lon_coords), step):
                        lat = lat_coords[i]
                        lon = lon_coords[j]
                        value = data_values[i, j]

                        if not np.isnan(value):
                            # Determine color based on value
                            color = self._get_value_color(value, pollutant)

                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=5,
                                popup=f'{pollutant.upper()}: {value:.4f}',
                                color=color,
                                fill=True,
                                fillColor=color,
                                fillOpacity=0.7
                            ).add_to(m)

        except Exception as e:
            logger.error(f"Failed to add measurement points: {e}")

    def _get_value_color(self, value: float, pollutant: str) -> str:
        """Get color for specific value."""
        # Simple color mapping based on percentile
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        if value < 0.2:
            return colors[0]
        elif value < 0.4:
            return colors[1]
        elif value < 0.6:
            return colors[2]
        elif value < 0.8:
            return colors[3]
        else:
            return colors[4]

    def _add_map_metadata(self, m, pollutant: str, timestamp: Optional[str]):
        """Add title and metadata to map."""
        try:
            title = f'Islamabad {pollutant.upper()} Pollution Map'
            if timestamp:
                title += f' - {timestamp}'

            # Add title as HTML
            title_html = '''
            <h3 align="center" style="font-size:16px"><b>{}</b></h3>
            '''.format(title)

            m.get_root().html.add_child(folium.Element(title_html))

        except Exception as e:
            logger.error(f"Failed to add map metadata: {e}")

    def _create_map_frame(self, data_slice: xr.DataArray, pollutant: str,
                         timestamp: str) -> Dict:
        """Create map frame for animation."""
        try:
            # Extract coordinates and values
            lat_coords = data_slice['lat'].values
            lon_coords = data_slice['lon'].values
            data_values = data_slice.values

            # Create trace
            trace = go.Scattermapbox(
                lat=lat_coords.flatten(),
                lon=lon_coords.flatten(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=data_values.flatten(),
                    colorscale=self.color_schemes.get(pollutant, 'Viridis'),
                    showscale=True,
                    colorbar=dict(title=f'{pollutant.upper()} Concentration')
                ),
                text=[f'Value: {v:.4f}' for v in data_values.flatten()],
                name=timestamp
            )

            return {'data': [trace], 'name': timestamp}

        except Exception as e:
            logger.error(f"Failed to create map frame: {e}")
            return {'data': [], 'name': timestamp}

    def _create_comparison_trace(self, data_slice: xr.DataArray,
                               pollutant: str, col_idx: int) -> go.Scattermapbox:
        """Create trace for comparison map."""
        try:
            lat_coords = data_slice['lat'].values
            lon_coords = data_slice['lon'].values
            data_values = data_slice.values

            return go.Scattermapbox(
                lat=lat_coords.flatten(),
                lon=lon_coords.flatten(),
                mode='markers',
                marker=dict(
                    size=6,
                    color=data_values.flatten(),
                    colorscale=self.color_schemes.get(pollutant, 'Viridis'),
                    showscale=(col_idx == 1),  # Only show colorbar for first subplot
                    colorbar=dict(title=f'{pollutant.upper()}', x=0.45 if col_idx == 1 else None)
                ),
                name=pollutant.upper()
            )

        except Exception as e:
            logger.error(f"Failed to create comparison trace: {e}")
            return go.Scattermapbox()

    def _create_pollution_contour(self, data_slice: xr.DataArray, pollutant: str) -> go.Contour:
        """Create pollution contour trace."""
        try:
            lat_coords = data_slice['lat'].values
            lon_coords = data_slice['lon'].values
            data_values = data_slice.values

            return go.Contour(
                z=data_values,
                x=lon_coords,
                y=lat_coords,
                colorscale=self.color_schemes.get(pollutant, 'Viridis'),
                colorbar=dict(title=f'{pollutant.upper()} Concentration'),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                name=f'{pollutant.upper()}'
            )

        except Exception as e:
            logger.error(f"Failed to create pollution contour: {e}")
            return go.Contour()

    def _create_wind_vectors(self, wind_data: pd.DataFrame) -> go.Scattermapbox:
        """Create wind vector trace."""
        try:
            if 'lat' in wind_data.columns and 'lon' in wind_data.columns:
                return go.Scattermapbox(
                    lat=wind_data['lat'],
                    lon=wind_data['lon'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        symbol='arrow',
                        color='blue',
                        angle=wind_data.get('direction', 0),
                        anglesrc='previous'
                    ),
                    text=wind_data.get('speed', 0),
                    name='Wind Vectors'
                )
            else:
                return go.Scattermapbox()

        except Exception as e:
            logger.error(f"Failed to create wind vectors: {e}")
            return go.Scattermapbox()

    def create_static_pollution_maps(self, dataset: xr.Dataset,
                                    pollutants: List[str],
                                    save_to_disk: bool = True) -> Dict[str, str]:
        """
        Create static pollution maps using Matplotlib.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            save_to_disk: Whether to save maps to disk

        Returns:
            Dictionary mapping pollutant to file paths
        """
        static_maps = {}

        try:
            for pollutant in pollutants:
                logger.info(f"Creating static map for {pollutant}")

                data_var = self._extract_pollutant_data(dataset, pollutant)

                if data_var is None:
                    logger.warning(f"No data available for {pollutant}")
                    continue

                # Create static map
                fig, ax = plt.subplots(figsize=(12, 10))

                # Create contour plot
                lat_coords = data_var['lat'].values
                lon_coords = data_var['lon'].values
                data_values = data_var.values

                # Create contour plot
                contour = ax.contourf(lon_coords, lat_coords, data_values,
                                    levels=20, cmap=self.color_schemes.get(pollutant, 'viridis'))

                # Add colorbar
                cbar = plt.colorbar(contour, ax=ax)
                cbar.set_label(f'{pollutant.upper()} Concentration')

                # Add Islamabad region boundary
                bbox = self.region['bounding_box']
                rect = plt.Rectangle((bbox['west'], bbox['south']),
                                   bbox['east'] - bbox['west'],
                                   bbox['north'] - bbox['south'],
                                   fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

                # Set labels and title
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'Islamabad {pollutant.upper()} Concentration')
                ax.grid(True, alpha=0.3)

                # Save map
                if save_to_disk:
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"static_{pollutant}_map_{timestamp}.png"
                    output_path = self.exports_path / filename

                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    static_maps[pollutant] = str(output_path)
                    logger.info(f"Static map saved: {output_path}")
                else:
                    # Save to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode()
                    plt.close()

                    static_maps[pollutant] = f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Failed to create static pollution maps: {e}")

        return static_maps

    def export_map_data(self, dataset: xr.Dataset, pollutants: List[str],
                       output_format: str = 'geojson') -> Dict[str, str]:
        """
        Export map data for external use.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            output_format: Export format ('geojson', 'csv', 'shapefile')

        Returns:
            Dictionary mapping pollutant to file paths
        """
        export_paths = {}

        try:
            for pollutant in pollutants:
                data_var = self._extract_pollutant_data(dataset, pollutant)

                if data_var is None:
                    continue

                # Convert to GeoDataFrame if possible
                if GEOPANDAS_AVAILABLE:
                    gdf = self._dataarray_to_geodataframe(data_var, pollutant)

                    if output_format == 'geojson':
                        filename = f"{pollutant}_map_data.geojson"
                        output_path = self.exports_path / filename
                        gdf.to_file(output_path, driver='GeoJSON')
                        export_paths[pollutant] = str(output_path)

                    elif output_format == 'csv':
                        filename = f"{pollutant}_map_data.csv"
                        output_path = self.exports_path / filename
                        gdf.to_csv(output_path, index=False)
                        export_paths[pollutant] = str(output_path)

                    elif output_format == 'shapefile':
                        filename = f"{pollutant}_map_data.shp"
                        output_path = self.exports_path / filename
                        gdf.to_file(output_path, driver='ESRI Shapefile')
                        export_paths[pollutant] = str(output_path)

                else:
                    # Fallback to CSV export
                    filename = f"{pollutant}_map_data.csv"
                    output_path = self.exports_path / filename

                    # Flatten data to CSV
                    lat_coords = data_var['lat'].values
                    lon_coords = data_var['lon'].values
                    data_values = data_var.values

                    export_data = []
                    for i in range(len(lat_coords)):
                        for j in range(len(lon_coords)):
                            export_data.append({
                                'lat': lat_coords[i],
                                'lon': lon_coords[j],
                                'value': data_values[i, j],
                                'pollutant': pollutant
                            })

                    df = pd.DataFrame(export_data)
                    df.to_csv(output_path, index=False)
                    export_paths[pollutant] = str(output_path)

        except Exception as e:
            logger.error(f"Failed to export map data: {e}")

        return export_paths

    def _dataarray_to_geodataframe(self, data_var: xr.DataArray,
                                   pollutant: str) -> gpd.GeoDataFrame:
        """Convert xarray DataArray to GeoDataFrame."""
        try:
            lat_coords = data_var['lat'].values
            lon_coords = data_var['lon'].values
            data_values = data_var.values

            # Create point geometries and data
            points = []
            values = []

            for i in range(len(lat_coords)):
                for j in range(len(lon_coords)):
                    points.append(Point(lon_coords[j], lat_coords[i]))
                    values.append(data_values[i, j])

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': points,
                'value': values,
                'pollutant': pollutant
            }, crs='EPSG:4326')

            # Remove NaN values
            gdf = gdf.dropna(subset=['value'])

            return gdf

        except Exception as e:
            logger.error(f"Failed to convert DataArray to GeoDataFrame: {e}")
            return gpd.GeoDataFrame()