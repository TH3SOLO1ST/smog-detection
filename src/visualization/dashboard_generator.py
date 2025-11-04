"""
Dashboard generator for Islamabad Smog Detection System.

This module provides comprehensive dashboard generation capabilities including
HTML dashboards, PDF reports, and interactive visualizations for
air pollution monitoring and analysis.
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

# Template libraries
try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Web generation
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Chart generation
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import get_config
from ..utils.file_utils import FileUtils
from ..utils.geo_utils import GeoUtils
from .mapping_tools import MappingTools
from .charting_tools import ChartingTools

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Comprehensive dashboard generator for air pollution monitoring."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize dashboard generator.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config(config_path)
        self.viz_config = self.config.get_section('visualization')

        # Dashboard configuration
        self.dashboard_config = self.viz_config.get('dashboard', {})
        self.auto_refresh_interval = self.dashboard_config.get('auto_refresh_interval', 3600)

        # Storage paths
        self.storage_config = self.config.get_storage_config()
        self.exports_path = FileUtils.ensure_directory(
            Path(self.storage_config.get('exports', 'data/exports')) / 'reports'
        )

        # Initialize tools
        self.mapping_tools = MappingTools(config_path)
        self.charting_tools = ChartingTools(config_path)

        # Islamabad region
        self.region = GeoUtils.create_islamabad_region(buffer_km=50)

        # Dashboard templates
        self.templates = self._load_templates()

        logger.info("Dashboard generator initialized")

    def _load_templates(self) -> Dict[str, Template]:
        """Load HTML templates for dashboard generation."""
        templates = {}

        if JINJA2_AVAILABLE:
            try:
                # Simple inline templates
                templates['main_dashboard'] = Template(self._get_main_dashboard_template())
                templates['summary_report'] = Template(self._get_summary_report_template())
                templates['detail_analysis'] = Template(self._get_detail_analysis_template())
            except Exception as e:
                logger.error(f"Failed to load templates: {e}")
                JINJA2_AVAILABLE = False

        return templates

    def generate_comprehensive_dashboard(self, dataset: xr.Dataset,
                                         pollutants: List[str],
                                         dashboard_type: str = 'html',
                                         save_to_disk: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive dashboard with multiple visualizations.

        Args:
            dataset: Input dataset
            pollutants: List of pollutant names
            dashboard_type: Type of dashboard ('html', 'pdf', 'interactive')
            save_to_disk: Whether to save dashboard to disk

        Returns:
            Dictionary with dashboard file paths
        """
        dashboard_paths = {}

        try:
            logger.info(f"Generating {dashboard_type} dashboard for pollutants: {pollutants}")

            # Generate visualizations
            viz_paths = self._generate_visualizations(dataset, pollutants)

            # Create dashboard summary
            dashboard_summary = self.charting_tools.create_dashboard_summary(dataset, pollutants)

            # Generate dashboard based on type
            if dashboard_type == 'html':
                dashboard_paths['html'] = self._generate_html_dashboard(
                    viz_paths, dashboard_summary, pollutants, save_to_disk
                )
            elif dashboard_type == 'pdf':
                dashboard_paths['pdf'] = self._generate_pdf_dashboard(
                    viz_paths, dashboard_summary, pollutants, save_to_disk
                )
            elif dashboard_type == 'interactive':
                dashboard_paths['interactive'] = self._generate_interactive_dashboard(
                    dataset, pollutants, save_to_disk
                )
            elif dashboard_type == 'all':
                # Generate all types
                dashboard_paths['html'] = self._generate_html_dashboard(
                    viz_paths, dashboard_summary, pollutants, save_to_disk
                )
                dashboard_paths['pdf'] = self._generate_pdf_dashboard(
                    viz_paths, dashboard_summary, pollutants, save_to_disk
                )
                dashboard_paths['interactive'] = self._generate_interactive_dashboard(
                    dataset, pollutants, save_to_disk
                )
            else:
                logger.warning(f"Unknown dashboard type: {dashboard_type}")

            logger.info(f"Dashboard generation completed: {list(dashboard_paths.keys())}")

        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")

        return dashboard_paths

    def _generate_visualizations(self, dataset: xr.Dataset,
                                pollutants: List[str]) -> Dict[str, Dict[str, str]]:
        """Generate all visualizations needed for dashboard."""
        viz_paths = {}

        try:
            # Generate time series charts
            viz_paths['time_series'] = self.charting_tools.create_time_series_chart(
                dataset, pollutants, chart_type='interactive', save_to_disk=True
            )

            # Generate distribution plots
            viz_paths['distribution'] = self.charting_tools.create_distribution_plots(
                dataset, pollutants, save_to_disk=True
            )

            # Generate correlation analysis
            viz_paths['correlation'] = self.charting_tools.create_correlation_analysis(
                dataset, pollutants, save_to_disk=True
            )

            # Generate seasonal analysis
            viz_paths['seasonal'] = self.charting_tools.create_seasonal_analysis_charts(
                dataset, pollutants, save_to_disk=True
            )

            # Generate maps
            viz_paths['maps'] = {}
            for pollutant in pollutants:
                map_path = self.mapping_tools.create_pollution_heatmap(
                    dataset, pollutant, save_to_disk=True
                )
                if map_path:
                    viz_paths['maps'][pollutant] = map_path

            # Generate multi-pollutant comparison
            if len(pollutants) > 1:
                comparison_map = self.mapping_tools.create_multi_pollutant_comparison_map(
                    dataset, pollutants, save_to_disk=True
                )
                if comparison_map:
                    viz_paths['maps']['comparison'] = comparison_map

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

        return viz_paths

    def _generate_html_dashboard(self, viz_paths: Dict[str, Dict[str, str]],
                                dashboard_summary: Dict[str, Any],
                                pollutants: List[str],
                                save_to_disk: bool) -> Optional[str]:
        """Generate HTML dashboard."""
        if not JINJA2_AVAILABLE:
            logger.error("Jinja2 not available for HTML dashboard generation")
            return None

        try:
            # Prepare template data
            template_data = {
                'title': 'Islamabad Smog Detection Dashboard',
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pollutants': pollutants,
                'summary': dashboard_summary,
                'visualizations': self._prepare_visualization_data(viz_paths),
                'region_info': {
                    'name': 'Islamabad',
                    'center': [33.6844, 73.0479],
                    'buffer_km': 50,
                    'bounding_box': self.region['bounding_box']
                }
            }

            # Render HTML template
            if 'main_dashboard' in self.templates:
                html_content = self.templates['main_dashboard'].render(**template_data)
            else:
                html_content = self._get_simple_html_dashboard(template_data)

            if save_to_disk:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"dashboard_{timestamp}.html"
                output_path = self.exports_path / filename

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                logger.info(f"HTML dashboard saved: {output_path}")
                return str(output_path)
            else:
                return html_content

        except Exception as e:
            logger.error(f"Failed to generate HTML dashboard: {e}")
            return None

    def _generate_pdf_dashboard(self, viz_paths: Dict[str, Dict[str, str]],
                               dashboard_summary: Dict[str, Any],
                               pollutants: List[str],
                               save_to_disk: bool) -> Optional[str]:
        """Generate PDF dashboard report."""
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab not available for PDF dashboard generation")
            return None

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dashboard_report_{timestamp}.pdf"
            output_path = self.exports_path / filename

            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title page
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )

            story.append(Paragraph("Islamabad Smog Detection Dashboard", title_style))
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading1']))
            story.append(Spacer(1, 12))

            # Current pollution levels
            current_data = []
            current_data.append(['Pollutant', 'Current Value', 'Trend', 'Status'])
            current_values = dashboard_summary.get('current_values', {})
            recent_trends = dashboard_summary.get('recent_trends', {})
            alert_levels = dashboard_summary.get('alert_levels', {})

            for pollutant in pollutants:
                current_val = current_values.get(pollutant, 'N/A')
                trend = recent_trends.get(pollutant, {}).get('direction', 'N/A')
                status = alert_levels.get(pollutant, {}).get('current_level', 'N/A')

                current_data.append([
                    pollutant.upper(),
                    f"{current_val:.4f}" if isinstance(current_val, (int, float)) else str(current_val),
                    str(trend).title(),
                    str(status).title()
                ])

            current_table = Table(current_data)
            current_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(current_table)
            story.append(Spacer(1, 20))

            # Statistics Summary
            story.append(Paragraph("Statistical Summary", styles['Heading1']))
            story.append(Spacer(1, 12))

            stats = dashboard_summary.get('statistics', {})
            stats_data = [['Pollutant', 'Mean', 'Std Dev', 'Min', 'Max']]

            for pollutant in pollutants:
                pollutant_stats = stats.get(pollutant, {})
                stats_data.append([
                    pollutant.upper(),
                    f"{pollutant_stats.get('mean', 0):.4f}",
                    f"{pollutant_stats.get('std', 0):.4f}",
                    f"{pollutant_stats.get('min', 0):.4f}",
                    f"{pollutant_stats.get('max', 0):.4f}"
                ])

            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(stats_table)
            story.append(Spacer(1, 20))

            # Include visualizations
            story.append(Paragraph("Visualizations", styles['Heading1']))
            story.append(Spacer(1, 12))

            # Add images if available
            for viz_type, viz_dict in viz_paths.items():
                story.append(Paragraph(f"{viz_type.replace('_', ' ').title()}", styles['Heading2']))

                if isinstance(viz_dict, dict):
                    for name, path in viz_dict.items():
                        if path and Path(path).exists():
                            try:
                                img = Image(str(path), width=6*inch, height=4*inch)
                                story.append(img)
                                story.append(Spacer(1, 12))
                            except Exception as e:
                                logger.warning(f"Failed to include image {path}: {e}")

            # Build PDF
            doc.build(story)

            logger.info(f"PDF dashboard report saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate PDF dashboard: {e}")
            return None

    def _generate_interactive_dashboard(self, dataset: xr.Dataset,
                                       pollutants: List[str],
                                       save_to_disk: bool) -> Optional[str]:
        """Generate interactive dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for interactive dashboard")
            return None

        try:
            # Create subplot layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Current Pollution Levels', 'Time Series', 'Distribution', 'Correlation', 'Map View', 'Seasonal Patterns'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "heatmap"}],
                       [{"type": "scattermapbox"}, {"type": "bar"}]],
                vertical_spacing=0.1
            )

            # Extract time series data
            time_series_data = self.charting_tools._extract_time_series_data(dataset, pollutants)
            distribution_data = self.charting_tools._extract_distribution_data(dataset, pollutants)

            # Current pollution levels (gauge charts)
            for i, pollutant in enumerate(pollutants):
                if pollutant in time_series_data and not time_series_data[pollutant].empty:
                    current_value = time_series_data[pollutant].iloc[-1]
                    mean_value = time_series_data[pollutant].mean()

                    fig.add_trace(go.Indicator(
                        mode="number+gauge+delta",
                        value=current_value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': pollutant.upper()},
                        delta={'reference': mean_value},
                        gauge={'axis': {'range': [None, mean_value * 2]},
                              'bar': {'color': self.charting_tools.pollutant_colors.get(pollutant, "blue")}},
                              'steps': [{'range': [0, mean_value * 0.5], 'color': "lightgray"},
                                        {'range': [mean_value * 0.5, mean_value], 'color': "gray"}],
                              'threshold': {'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75, 'value': mean_value * 1.5}}
                    ), row=1, col=1)
                    break

            # Time series
            for pollutant in pollutants:
                if pollutant in time_series_data and not time_series_data[pollutant].empty:
                    fig.add_trace(go.Scatter(
                        x=time_series_data[pollutant].index,
                        y=time_series_data[pollutant].values,
                        mode='lines',
                        name=pollutant.upper(),
                        line=dict(color=self.charting_tools.pollutant_colors.get(pollutant, "blue"))
                    ), row=2, col=1)

            # Distribution
            if distribution_data:
                for pollutant, data in distribution_data.items():
                    fig.add_trace(go.Histogram(
                        x=data,
                        name=pollutant.upper(),
                        opacity=0.7
                    ), row=2, col=2)

            # Correlation heatmap
            if len(time_series_data) >= 2:
                df = pd.DataFrame(time_series_data)
                corr_matrix = df.corr()

                fig.add_trace(go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlation'
                ), row=3, col=2)

            # Map view (simplified)
            # This would require more complex implementation
            # For now, placeholder
            fig.add_trace(go.Scattermapbox(
                lat=[33.6844],
                lon=[73.0479],
                mode='markers',
                marker=dict(size=20, color='red'),
                name='Islamabad Center'
            ), row=3, col=1)

            # Update layout
            fig.update_layout(
                title="Islamabad Smog Detection Interactive Dashboard",
                height=1200,
                showlegend=True,
                template='plotly_white'
            )

            # Save interactive dashboard
            if save_to_disk:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"interactive_dashboard_{timestamp}.html"
                output_path = self.exports_path / filename

                fig.write_html(str(output_path))
                logger.info(f"Interactive dashboard saved: {output_path}")
                return str(output_path)
            else:
                return fig.to_html()

        except Exception as e:
            logger.error(f"Failed to generate interactive dashboard: {e}")
            return None

    def _prepare_visualization_data(self, viz_paths: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Prepare visualization data for HTML template."""
        prepared_data = {}

        for viz_type, viz_dict in viz_paths.items():
            prepared_data[viz_type] = {}

            if isinstance(viz_dict, dict):
                for name, path in viz_dict.items():
                    if path and Path(path).exists():
                        # Convert image to base64 for embedding
                        try:
                            with open(path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                                prepared_data[viz_type][name] = f"data:image/png;base64,{img_data}"
                        except Exception as e:
                            logger.warning(f"Failed to encode image {path}: {e}")
                            prepared_data[viz_type][name] = path
                    else:
                        prepared_data[viz_type][name] = path
            else:
                prepared_data[viz_type] = viz_dict

        return prepared_data

    def _get_main_dashboard_template(self) -> str:
        """Get main dashboard HTML template."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { margin: 0; font-size: 2.5em; }
        .header p { margin: 10px 0 0 0; opacity: 0.8; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .summary-card h3 { margin: 0 0 10px 0; color: #333; }
        .summary-card .value { font-size: 2em; font-weight: bold; color: #667eea; }
        .summary-card .trend { font-size: 0.9em; margin-top: 5px; }
        .trend.up { color: #e74c3c; }
        .trend.down { color: #27ae60; }
        .visualization { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .visualization h2 { margin: 0 0 20px 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .visualization img { max-width: 100%; height: auto; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on {{ generation_time }}</p>
    </div>

    <div class="summary">
        {% for pollutant in pollutants %}
        <div class="summary-card">
            <h3>{{ pollutant.upper() }}</h3>
            <div class="value">{{ summary.current_values.get(pollutant, 'N/A') }}</div>
            <div class="trend {{ summary.recent_trends.get(pollutant, {}).get('direction', '') }}">
                {{ summary.recent_trends.get(pollutant, {}).get('direction', 'N/A') }}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="visualization">
        <h2>Time Series Analysis</h2>
        {% if visualizations.time_series %}
            {% for name, path in visualizations.time_series.items() %}
                {% if path.startswith('data:image') %}
                    <img src="{{ path }}" alt="{{ name }}">
                {% else %}
                    <p>Chart: {{ name }}</p>
                {% endif %}
            {% endfor %}
        {% else %}
            <p>No time series visualizations available.</p>
        {% endif %}
    </div>

    <div class="visualization">
        <h2>Distribution Analysis</h2>
        {% if visualizations.distribution %}
            {% for name, path in visualizations.distribution.items() %}
                {% if path.startswith('data:image') %}
                    <img src="{{ path }}" alt="{{ name }}">
                {% else %}
                    <p>Distribution: {{ name }}</p>
                {% endif %}
            {% endfor %}
        {% else %}
            <p>No distribution visualizations available.</p>
        {% endif %}
    </div>

    <div class="visualization">
        <h2>Pollution Maps</h2>
        {% if visualizations.maps %}
            {% for name, path in visualizations.maps.items() %}
                {% if path.startswith('data:image') %}
                    <img src="{{ path }}" alt="{{ name }}">
                {% else %}
                    <p>Map: {{ name }}</p>
                {% endif %}
            {% endfor %}
        {% else %}
            <p>No map visualizations available.</p>
        {% endif %}
    </div>

    <div class="footer">
        <p>Islamabad Smog Detection System © {{ generation_time[:4] }}</p>
    </div>
</body>
</html>
        '''

    def _get_simple_html_dashboard(self, template_data: Dict[str, Any]) -> str:
        """Get simple HTML dashboard template."""
        return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{template_data['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .pollutant {{ margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #667eea; }}
    </style>
</head>
<body>
    <h1>{template_data['title']}</h1>
    <p>Generated on {template_data['generation_time']}</p>

    <div class="summary">
        <h2>Current Pollution Levels</h2>
        {% for pollutant in template_data['pollutants'] %}
        <div class="pollutant">
            <strong>{{ pollutant.upper() }}:</strong>
            {{ template_data['summary']['current_values'].get(pollutant, 'N/A') }}
        </div>
        {% endfor %}
    </div>
</body>
</html>
        '''

    def _get_summary_report_template(self) -> str:
        """Get summary report template."""
        return '''
        <h1>{{ title }}</h1>
        <p>Generated: {{ generation_time }}</p>

        <h2>Executive Summary</h2>
        <div class="summary-grid">
            {% for pollutant in pollutants %}
            <div class="metric-card">
                <h3>{{ pollutant.upper() }}</h3>
                <p class="current-value">{{ summary.current_values.get(pollutant, 'N/A') }}</p>
                <p class="trend">{{ summary.recent_trends.get(pollutant, {}).get('direction', 'N/A') }}</p>
            </div>
            {% endfor %}
        </div>
        '''

    def _get_detail_analysis_template(self) -> str:
        """Get detailed analysis template."""
        return '''
        <h1>Detailed Analysis Report</h1>

        <h2>Statistical Analysis</h2>
        <div class="stats-table">
            <!-- Statistics table would be populated here -->
        </div>

        <h2>Visualizations</h2>
        <div class="viz-grid">
            <!-- Visualizations would be embedded here -->
        </div>
        '''

    def create_real_time_dashboard(self, dataset: xr.Dataset,
                                  pollutants: List[str],
                                  update_interval: int = 300) -> str:
        """
        Create real-time dashboard with auto-refresh capability.

        Args:
            dataset: Current dataset
            pollutants: List of pollutants
            update_interval: Update interval in seconds

        Returns:
            HTML file path
        """
        try:
            # Generate base dashboard
            dashboard_paths = self.generate_comprehensive_dashboard(
                dataset, pollutants, dashboard_type='html', save_to_disk=True
            )

            if 'html' not in dashboard_paths:
                return None

            # Add auto-refresh capability
            html_path = dashboard_paths['html']
            with open(html_path, 'r') as f:
                html_content = f.read()

            # Add auto-refresh meta tag
            auto_refresh_tag = f'<meta http-equiv="refresh" content="{update_interval}">'
            html_content = html_content.replace('<head>', f'<head>{auto_refresh_tag}')

            # Save updated HTML
            with open(html_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Real-time dashboard created with {update_interval}s refresh interval")
            return html_path

        except Exception as e:
            logger.error(f"Failed to create real-time dashboard: {e}")
            return None

    def generate_alert_dashboard(self, dataset: xr.Dataset,
                                pollutants: List[str],
                                thresholds: Dict[str, Dict[str, float]],
                                save_to_disk: bool = True) -> Optional[str]:
        """
        Generate alert dashboard based on threshold values.

        Args:
            dataset: Input dataset
            pollutants: List of pollutants
            thresholds: Threshold values for alerts
            save_to_disk: Whether to save dashboard

        Returns:
            HTML file path
        """
        try:
            logger.info("Generating alert dashboard")

            # Extract current values
            time_series_data = self.charting_tools._extract_time_series_data(dataset, pollutants)
            current_values = {}
            alerts = {}

            for pollutant in pollutants:
                if pollutant in time_series_data and not time_series_data[pollutant].empty:
                    current_val = float(time_series_data[pollutant].iloc[-1])
                    current_values[pollutant] = current_val

                    # Check against thresholds
                    pollutant_thresholds = thresholds.get(pollutant, {})
                    if pollutant_thresholds:
                        if current_val >= pollutant_thresholds.get('critical', float('inf')):
                            alerts[pollutant] = {'level': 'critical', 'value': current_val, 'threshold': pollutant_thresholds.get('critical')}
                        elif current_val >= pollutant_thresholds.get('warning', float('inf')):
                            alerts[pollutant] = {'level': 'warning', 'value': current_val, 'threshold': pollutant_thresholds.get('warning')}
                        elif current_val >= pollutant_thresholds.get('moderate', float('inf')):
                            alerts[pollutant] = {'level': 'moderate', 'value': current_val, 'threshold': pollutant_thresholds.get('moderate')}
                        else:
                            alerts[pollutant] = {'level': 'good', 'value': current_val}

            # Generate alert dashboard HTML
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alert_dashboard_{timestamp}.html"
            output_path = self.exports_path / filename

            html_content = self._create_alert_dashboard_html(current_values, alerts, thresholds)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Alert dashboard saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate alert dashboard: {e}")
            return None

    def _create_alert_dashboard_html(self, current_values: Dict[str, float],
                                    alerts: Dict[str, Dict[str, Any]],
                                    thresholds: Dict[str, Dict[str, float]]) -> str:
        """Create HTML content for alert dashboard."""
        html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Islamabad Air Quality Alert Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
                .header {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 30px; }}
                .alert-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .alert-card {{ padding: 20px; border-radius: 10px; text-align: center; color: white; }}
                .alert-critical {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
                .alert-warning {{ background: linear-gradient(135deg, #f39c12, #e67e22); }}
                .alert-moderate {{ background: linear-gradient(135deg, #3498db, #2980b9); }}
                .alert-good {{ background: linear-gradient(135deg, #27ae60, #229954); }}
                .alert-value {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; }}
                .alert-level {{ font-size: 1.2em; margin: 5px 0; text-transform: uppercase; }}
                .alert-threshold {{ font-size: 0.9em; opacity: 0.8; }}
                .status-legend {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .legend-item {{ display: inline-block; margin: 10px; padding: 10px; border-radius: 5px; color: white; }}
                .last-updated {{ text-align: center; color: #666; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 Islamabad Air Quality Alert Dashboard</h1>
                <p>Real-time pollution monitoring with threshold-based alerts</p>
            </div>

            <div class="alert-grid">
        '''

        for pollutant, alert_data in alerts.items():
            level = alert_data['level']
            value = alert_data['value']
            threshold = alert_data.get('threshold', 'N/A')

            emoji = {'critical': '🔴', 'warning': '🟡', 'moderate': '🟡', 'good': '🟢'}.get(level, '⚪')

            html += f'''
                <div class="alert-card alert-{level}">
                    <h3>{emoji} {pollutant.upper()}</h3>
                    <div class="alert-value">{value:.4f}</div>
                    <div class="alert-level">{level}</div>
                    <div class="alert-threshold">Threshold: {threshold}</div>
                </div>
            '''

        html += f'''
            </div>

            <div class="status-legend">
                <h3>Status Legend</h3>
                <div class="legend-item alert-critical">🔴 Critical - Immediate Action Required</div>
                <div class="legend-item alert-warning">🟡 Warning - Unhealthy Air Quality</div>
                <div class="legend-item alert-moderate">🟡 Moderate - Acceptable Range</div>
                <div class="legend-item alert-good">🟢 Good - Healthy Air Quality</div>
            </div>

            <div class="last-updated">
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <!-- Auto-refresh every 5 minutes -->
            <meta http-equiv="refresh" content="300">

        </body>
        </html>
        '''

        return html

    def export_dashboard_data(self, dashboard_data: Dict[str, Any],
                             output_format: str = 'json') -> str:
        """
        Export dashboard data for external systems.

        Args:
            dashboard_data: Dashboard data dictionary
            output_format: Export format ('json', 'xml', 'api')

        Returns:
            Export file path or API response
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if output_format == 'json':
                filename = f"dashboard_data_{timestamp}.json"
                output_path = self.exports_path / filename

                with open(output_path, 'w') as f:
                    json.dump(dashboard_data, f, indent=2, default=str)

            elif output_format == 'api':
                # Create API-like response format
                api_response = {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': dashboard_data
                }

                filename = f"dashboard_api_response_{timestamp}.json"
                output_path = self.exports_path / filename

                with open(output_path, 'w') as f:
                    json.dump(api_response, f, indent=2)

            logger.info(f"Dashboard data exported: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            return ""

    def schedule_dashboard_generation(self, dataset: xr.Dataset,
                                      pollutants: List[str],
                                      schedule_type: str = 'daily') -> None:
        """
        Schedule regular dashboard generation (placeholder).

        Args:
            dataset: Dataset for dashboard generation
            pollutants: List of pollutants
            schedule_type: Schedule type ('daily', 'weekly', 'monthly')
        """
        # This would typically integrate with a task scheduler like Celery or APScheduler
        logger.info(f"Scheduling dashboard generation: {schedule_type}")

        # Implementation would depend on the specific scheduling system used
        # This is a placeholder for the scheduling functionality
        pass

    def validate_dashboard_components(self) -> Dict[str, bool]:
        """
        Validate all dashboard components and dependencies.

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        try:
            # Check template availability
            validation_results['jinja2_available'] = JINJA2_AVAILABLE
            validation_results['plotly_available'] = PLOTLY_AVAILABLE
            validation_results['reportlab_available'] = REPORTLAB_AVAILABLE

            # Check directory structure
            validation_results['exports_directory'] = self.exports_path.exists()
            validation_results['mapping_tools_available'] = self.mapping_tools is not None
            validation_results['charting_tools_available'] = self.charting_tools is not None

            # Check required libraries
            validation_results['matplotlib_available'] = True  # Always available as we're using it
            validation_results['seaborn_available'] = True  # Always available as we're using it

        except Exception as e:
            logger.error(f"Dashboard validation failed: {e}")
            validation_results['error'] = str(e)

        return validation_results

    def get_dashboard_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about available dashboard capabilities.

        Returns:
            Dictionary with dashboard metadata
        """
        metadata = {
            'supported_formats': ['html', 'pdf', 'interactive'],
            'supported_visualizations': [
                'time_series', 'distribution', 'correlation', 'maps',
                'seasonal_analysis', 'alert_dashboard'
            ],
            'dependencies': {
                'jinja2_available': JINJA2_AVAILABLE,
                'plotly_available': PLOTLY_AVAILABLE,
                'reportlab_available': REPORTLAB_AVAILABLE
            },
            'configuration': {
                'auto_refresh_interval': self.auto_refresh_interval,
                'exports_directory': str(self.exports_path)
            },
            'region_info': {
                'name': 'Islamabad',
                'center': [33.6844, 73.0479],
                'buffer_km': 50
            }
        }

        return metadata