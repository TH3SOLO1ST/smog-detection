"""
Main Flask Application for Islamabad Smog Detection System
Web-based dashboard for non-technical users
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_cors import CORS
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from celery import Celery
import redis

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_acquisition.sentinel5p_collector import Sentinel5PCollector
from src.data_acquisition.modis_collector import MODISCollector
from src.data_acquisition.gee_collector import GEECollector
from src.preprocessing.atmospheric_correction import AtmosphericCorrector
from src.analysis.time_series_processor import TimeSeriesProcessor
from src.visualization.mapping_tools import MappingTools
from src.visualization.charting_tools import ChartingTools
from src.utils.config import ConfigManager
from src.automation.email_reports import EmailReporter
from src.automation.pdf_reports import PDFReporter

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Enable CORS
CORS(app)

# Configuration
config_manager = ConfigManager()
config = config_manager.get_config()

# Redis configuration
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.from_url(redis_url)

# Celery configuration
app.config['CELERY_BROKER_URL'] = redis_url
app.config['CELERY_RESULT_BACKEND'] = redis_url

def make_celery(app):
    """Create Celery instance"""
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        """Make celery tasks work with Flask app context."""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery_app = make_celery(app)

# Initialize data collectors and processors
try:
    sentinel_collector = Sentinel5PCollector(config)
    modis_collector = MODISCollector(config)
    gee_collector = GEECollector(config)
    atmospheric_corrector = AtmosphericCorrector(config)
    ts_processor = TimeSeriesProcessor(config)
    mapping_tools = MappingTools(config)
    charting_tools = ChartingTools(config)
    email_reporter = EmailReporter(config)
    pdf_reporter = PDFReporter(config)
except Exception as e:
    print(f"Warning: Could not initialize some components: {e}")
    # Continue with limited functionality
    sentinel_collector = None
    modis_collector = None
    gee_collector = None
    atmospheric_corrector = None
    ts_processor = None
    mapping_tools = None
    charting_tools = None
    email_reporter = None
    pdf_reporter = None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/current_conditions')
def get_current_conditions():
    """Get current air quality conditions for Islamabad"""
    try:
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')

        # Try to get real data from collectors
        current_data = {
            'timestamp': datetime.now().isoformat(),
            'location': 'Islamabad, Pakistan',
            'coordinates': {'lat': 33.6844, 'lon': 73.0479},
            'pollutants': {},
            'overall_status': 'unknown',
            'data_source': 'demo'
        }

        # Demo data for development - replace with real data when available
        current_data['pollutants'] = {
            'NO2': {
                'value': 25.3,
                'unit': 'µg/m³',
                'status': 'moderate',
                'color': '#FFC107'
            },
            'SO2': {
                'value': 12.1,
                'unit': 'µg/m³',
                'status': 'good',
                'color': '#4CAF50'
            },
            'CO': {
                'value': 0.8,
                'unit': 'mg/m³',
                'status': 'good',
                'color': '#4CAF50'
            },
            'O3': {
                'value': 65.2,
                'unit': 'µg/m³',
                'status': 'moderate',
                'color': '#FFC107'
            },
            'PM2.5': {
                'value': 35.7,
                'unit': 'µg/m³',
                'status': 'unhealthy_sensitive',
                'color': '#FF9800'
            },
            'AOD': {
                'value': 0.42,
                'unit': 'dimensionless',
                'status': 'moderate',
                'color': '#FFC107'
            }
        }

        # Determine overall status based on worst pollutant
        status_priority = ['hazardous', 'very_unhealthy', 'unhealthy', 'unhealthy_sensitive', 'moderate', 'good']
        statuses = [pollutant['status'] for pollutant in current_data['pollutants'].values()]

        for status in status_priority:
            if status in statuses:
                current_data['overall_status'] = status
                break

        return jsonify(current_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical_data')
def get_historical_data():
    """Get historical air quality data"""
    try:
        days = request.args.get('days', 30, type=int)

        # Generate demo historical data
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]

        historical_data = {
            'dates': dates,
            'pollutants': {
                'NO2': [20 + (i % 10) for i in range(days)],
                'SO2': [10 + (i % 5) for i in range(days)],
                'CO': [0.5 + (i % 3) * 0.1 for i in range(days)],
                'O3': [60 + (i % 15) for i in range(days)],
                'PM2.5': [30 + (i % 20) for i in range(days)],
                'AOD': [0.3 + (i % 4) * 0.05 for i in range(days)]
            }
        }

        return jsonify(historical_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/map_data')
def get_map_data():
    """Get data for interactive map"""
    try:
        # Generate demo map data for Islamabad region
        map_data = {
            'center': {'lat': 33.6844, 'lon': 73.0479},
            'zoom': 10,
            'locations': [
                {
                    'name': 'Islamabad City Center',
                    'lat': 33.6844,
                    'lon': 73.0479,
                    'pollution_level': 'moderate',
                    'pm25': 35.7,
                    'no2': 25.3,
                    'color': '#FFC107'
                },
                {
                    'name': 'Faisal Mosque',
                    'lat': 33.7295,
                    'lon': 73.0365,
                    'pollution_level': 'good',
                    'pm25': 28.2,
                    'no2': 22.1,
                    'color': '#4CAF50'
                },
                {
                    'name': 'Islamabad Airport',
                    'lat': 33.6151,
                    'lon': 73.0994,
                    'pollution_level': 'unhealthy_sensitive',
                    'pm25': 42.3,
                    'no2': 31.5,
                    'color': '#FF9800'
                }
            ],
            'heat_map_data': []  # Could be populated with grid data
        }

        return jsonify(map_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get current air quality alerts"""
    try:
        alerts = [
            {
                'id': 1,
                'type': 'warning',
                'message': 'PM2.5 levels elevated in industrial areas',
                'timestamp': datetime.now().isoformat(),
                'locations': ['Industrial Area', 'Islamabad Airport'],
                'severity': 'medium'
            },
            {
                'id': 2,
                'type': 'info',
                'message': 'Good air quality expected tomorrow due to favorable winds',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'locations': ['All Islamabad'],
                'severity': 'low'
            }
        ]

        return jsonify({'alerts': alerts})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report')
def generate_report():
    """Generate and download air quality report"""
    try:
        report_type = request.args.get('type', 'daily')
        format_type = request.args.get('format', 'pdf')

        # Trigger report generation in background
        task = generate_report_task.delay(report_type, format_type)

        return jsonify({
            'task_id': task.id,
            'message': 'Report generation started',
            'estimated_time': '30 seconds'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report_status/<task_id>')
def get_report_status(task_id):
    """Check report generation status"""
    try:
        task = generate_report_task.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'status': task.info.get('status', '')
            }
            if task.result:
                response['result'] = task.result
        else:
            response = {
                'state': task.state,
                'status': str(task.info)
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/refresh_data')
def refresh_data():
    """Force refresh of air quality data"""
    try:
        # Trigger data refresh in background
        task = refresh_data_task.delay()

        return jsonify({
            'task_id': task.id,
            'message': 'Data refresh started',
            'estimated_time': '2-5 minutes'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Configuration routes
@app.route('/config')
def config_page():
    """Configuration page"""
    return render_template('config.html')

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        try:
            current_config = config_manager.get_config()
            # Remove sensitive information
            safe_config = {
                'region': current_config.get('region', {}),
                'apis': {k: {'configured': bool(v.get('client_id') or v.get('username'))}
                        for k, v in current_config.get('apis', {}).items()},
                'processing': current_config.get('processing', {}),
                'alerts': current_config.get('alerts', {})
            }
            return jsonify(safe_config)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif request.method == 'POST':
        try:
            new_config = request.json
            config_manager.update_config(new_config)
            return jsonify({'message': 'Configuration updated successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Celery tasks
@celery_app.task
def generate_report_task(report_type, format_type):
    """Generate report in background"""
    try:
        # Simulate report generation
        import time
        time.sleep(10)  # Simulate processing time

        if pdf_reporter:
            report_path = pdf_reporter.generate_report(report_type)
            return {'status': 'completed', 'file_path': report_path}
        else:
            return {'status': 'demo', 'message': 'Demo report generation'}

    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

@celery_app.task
def refresh_data_task():
    """Refresh air quality data in background"""
    try:
        # Simulate data refresh
        import time
        time.sleep(30)  # Simulate API calls and processing

        return {'status': 'completed', 'message': 'Data refreshed successfully'}

    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

@celery_app.task
def send_daily_report():
    """Send daily email report"""
    try:
        if email_reporter:
            email_reporter.send_daily_report()
            return {'status': 'completed', 'message': 'Daily report sent'}
        else:
            return {'status': 'demo', 'message': 'Demo daily report'}

    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

# Schedule daily reports
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup scheduled tasks"""
    # Send daily report at 8 AM
    sender.add_periodic_task(
        crontab(hour=8, minute=0),
        send_daily_report.s(),
        name='send-daily-report'
    )

def create_app():
    """Create and configure Flask app"""
    return app

if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=True)