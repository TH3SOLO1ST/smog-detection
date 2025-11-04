"""
Email Report Generator for Islamabad Smog Detection System
Automated daily and weekly air quality reports via email
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests
from jinja2 import Template
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailReporter:
    """Handles automated email report generation and delivery"""

    def __init__(self, config):
        """
        Initialize Email Reporter

        Args:
            config (dict): System configuration
        """
        self.config = config
        self.smtp_config = config.get('email', {})
        self.alerts_config = config.get('alerts', {})

        # Email settings from environment variables
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self.email_username = os.environ.get('EMAIL_USERNAME', '')
        self.email_password = os.environ.get('EMAIL_PASSWORD', '')

        # Report storage
        self.reports_dir = Path('data/exports/reports')
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def send_daily_report(self):
        """
        Generate and send daily air quality report

        Returns:
            dict: Status of the operation
        """
        try:
            logger.info("Generating daily air quality report")

            # Generate report content
            report_data = self._generate_daily_report_data()

            # Create HTML email content
            html_content = self._create_daily_report_html(report_data)

            # Create plain text version
            text_content = self._create_daily_report_text(report_data)

            # Get recipients
            recipients = self._get_recipients()

            if not recipients:
                logger.warning("No email recipients configured")
                return {'status': 'failed', 'message': 'No recipients configured'}

            # Send email
            result = self._send_email(
                subject=f"Daily Air Quality Report - Islamabad - {datetime.now().strftime('%Y-%m-%d')}",
                recipients=recipients,
                html_content=html_content,
                text_content=text_content,
                attachments=self._get_daily_attachments(report_data)
            )

            if result['success']:
                logger.info(f"Daily report sent successfully to {len(recipients)} recipients")
                return {'status': 'success', 'message': 'Daily report sent successfully'}
            else:
                logger.error(f"Failed to send daily report: {result['error']}")
                return {'status': 'failed', 'error': result['error']}

        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {'status': 'failed', 'error': str(e)}

    def send_weekly_report(self):
        """
        Generate and send weekly air quality report

        Returns:
            dict: Status of the operation
        """
        try:
            logger.info("Generating weekly air quality report")

            # Generate report content
            report_data = self._generate_weekly_report_data()

            # Create HTML email content
            html_content = self._create_weekly_report_html(report_data)

            # Create plain text version
            text_content = self._create_weekly_report_text(report_data)

            # Get recipients
            recipients = self._get_recipients()

            # Send email
            result = self._send_email(
                subject=f"Weekly Air Quality Report - Islamabad - {datetime.now().strftime('%Y-%m-%d')}",
                recipients=recipients,
                html_content=html_content,
                text_content=text_content,
                attachments=self._get_weekly_attachments(report_data)
            )

            if result['success']:
                logger.info(f"Weekly report sent successfully to {len(recipients)} recipients")
                return {'status': 'success', 'message': 'Weekly report sent successfully'}
            else:
                logger.error(f"Failed to send weekly report: {result['error']}")
                return {'status': 'failed', 'error': result['error']}

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return {'status': 'failed', 'error': str(e)}

    def send_alert_email(self, alert_data):
        """
        Send immediate alert email for high pollution levels

        Args:
            alert_data (dict): Alert information

        Returns:
            dict: Status of the operation
        """
        try:
            logger.info(f"Sending alert email: {alert_data['message']}")

            # Create alert email content
            html_content = self._create_alert_html(alert_data)
            text_content = self._create_alert_text(alert_data)

            # Get recipients (for alerts, we might have a separate list)
            recipients = self._get_alert_recipients()

            if not recipients:
                logger.warning("No alert recipients configured")
                return {'status': 'failed', 'message': 'No alert recipients configured'}

            # Send email with high priority
            result = self._send_email(
                subject=f"🚨 AIR QUALITY ALERT - {alert_data['severity'].upper()} - Islamabad",
                recipients=recipients,
                html_content=html_content,
                text_content=text_content,
                priority='high'
            )

            if result['success']:
                logger.info(f"Alert email sent successfully to {len(recipients)} recipients")
                return {'status': 'success', 'message': 'Alert sent successfully'}
            else:
                logger.error(f"Failed to send alert: {result['error']}")
                return {'status': 'failed', 'error': result['error']}

        except Exception as e:
            logger.error(f"Error sending alert email: {e}")
            return {'status': 'failed', 'error': str(e)}

    def test_email_configuration(self):
        """
        Test email configuration by sending a test email

        Returns:
            dict: Test result
        """
        try:
            logger.info("Testing email configuration")

            test_content = """
            <h2>Email Configuration Test</h2>
            <p>This is a test email from the Islamabad Smog Detection System.</p>
            <p>If you received this email, your email configuration is working correctly.</p>
            <p><strong>Sent:</strong> {}</p>
            <p><strong>System:</strong> Islamabad Smog Detection System v1.0</p>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Get test recipient (first configured recipient)
            recipients = self._get_recipients()
            if not recipients:
                return {'success': False, 'error': 'No recipients configured for testing'}

            result = self._send_email(
                subject="Email Configuration Test - Islamabad Smog Monitor",
                recipients=[recipients[0]],  # Send to first recipient only
                html_content=test_content,
                text_content="Email configuration test from Islamabad Smog Detection System"
            )

            return result

        except Exception as e:
            logger.error(f"Error testing email configuration: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_daily_report_data(self):
        """Generate data for daily report"""
        today = datetime.now().strftime('%Y-%m-%d')

        # This would normally fetch real data from the system
        # For now, we'll generate demo data
        report_data = {
            'date': today,
            'overall_status': 'moderate',
            'air_quality_index': 85,
            'pollutants': {
                'PM2.5': {'value': 35.7, 'unit': 'µg/m³', 'status': 'moderate'},
                'NO2': {'value': 25.3, 'unit': 'µg/m³', 'status': 'good'},
                'SO2': {'value': 12.1, 'unit': 'µg/m³', 'status': 'good'},
                'CO': {'value': 0.8, 'unit': 'mg/m³', 'status': 'good'},
                'O3': {'value': 65.2, 'unit': 'µg/m³', 'status': 'moderate'}
            },
            'health_recommendations': [
                "Air quality is acceptable for most people",
                "Sensitive individuals may experience minor issues",
                "Limit prolonged outdoor exertion if you are sensitive"
            ],
            'peak_times': {
                'morning_peak': '08:00 - 10:00',
                'evening_peak': '18:00 - 20:00'
            },
            'forecast': 'Similar conditions expected tomorrow',
            'data_sources': ['Copernicus Sentinel-5P', 'NASA MODIS', 'Local monitoring stations']
        }

        return report_data

    def _generate_weekly_report_data(self):
        """Generate data for weekly report"""
        week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        week_end = datetime.now().strftime('%Y-%m-%d')

        # Generate demo weekly data
        report_data = {
            'week_start': week_start,
            'week_end': week_end,
            'average_aqi': 92,
            'peak_aqi': 145,
            'lowest_aqi': 45,
            'trend': 'improving',
            'pollutant_averages': {
                'PM2.5': {'avg': 38.2, 'peak': 65.4, 'unit': 'µg/m³'},
                'NO2': {'avg': 27.8, 'peak': 45.1, 'unit': 'µg/m³'},
                'SO2': {'avg': 14.3, 'peak': 28.7, 'unit': 'µg/m³'},
                'CO': {'avg': 0.9, 'peak': 1.4, 'unit': 'mg/m³'},
                'O3': {'avg': 68.5, 'peak': 95.2, 'unit': 'µg/m³'}
            },
            'notable_events': [
                {
                    'date': '2024-01-15',
                    'event': 'High PM2.5 levels detected in industrial area',
                    'peak_value': '65.4 µg/m³'
                }
            ],
            'weekly_summary': "Air quality showed gradual improvement throughout the week with weekend conditions being the best.",
            'forecast': "Expected improvement in early next week due to changing weather patterns"
        }

        return report_data

    def _create_daily_report_html(self, data):
        """Create HTML content for daily report"""
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Daily Air Quality Report - Islamabad</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 20px; margin-bottom: 30px; }
                .status-box { text-align: center; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .good { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .moderate { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
                .unhealthy { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
                .pollutant-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin: 20px 0; }
                .pollutant-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }
                .value { font-size: 24px; font-weight: bold; color: #495057; }
                .unit { font-size: 12px; color: #6c757d; }
                .status { font-size: 14px; margin-top: 5px; }
                .recommendations { background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌫️ Daily Air Quality Report</h1>
                    <h2>Islamabad, Pakistan</h2>
                    <p><strong>{{ date }}</strong></p>
                </div>

                <div class="status-box {{ data.overall_status }}">
                    <h3>Overall Air Quality: {{ data.overall_status.title() }}</h3>
                    <div style="font-size: 48px; margin: 10px 0;">{{ get_status_emoji(data.overall_status) }}</div>
                    <p>Air Quality Index: <strong>{{ data.air_quality_index }}</strong></p>
                </div>

                <h3>📊 Pollutant Levels</h3>
                <div class="pollutant-grid">
                    {% for pollutant, info in data.pollutants.items() %}
                    <div class="pollutant-card">
                        <div class="value">{{ info.value }}</div>
                        <div class="unit">{{ info.unit }}</div>
                        <div>{{ pollutant }}</div>
                        <div class="status">{{ info.status.title() }}</div>
                    </div>
                    {% endfor %}
                </div>

                <div class="recommendations">
                    <h3>🏥 Health Recommendations</h3>
                    <ul>
                        {% for rec in data.health_recommendations %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                </div>

                <h3>📈 Daily Patterns</h3>
                <p><strong>Morning Peak:</strong> {{ data.peak_times.morning_peak }}</p>
                <p><strong>Evening Peak:</strong> {{ data.peak_times.evening_peak }}</p>

                <h3>🔮 Tomorrow's Forecast</h3>
                <p>{{ data.forecast }}</p>

                <div class="footer">
                    <p>Generated by Islamabad Smog Detection System</p>
                    <p>Data sources: {{ data.data_sources | join(', ') }}</p>
                    <p>Generated: {{ generation_time }}</p>
                </div>
            </div>
        </body>
        </html>
        ''')

        return template.render(
            data=data,
            date=data['date'],
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            get_status_emoji=self._get_status_emoji
        )

    def _create_daily_report_text(self, data):
        """Create plain text content for daily report"""
        content = f"""
DAILY AIR QUALITY REPORT - ISLAMABAD
Date: {data['date']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATUS: {data['overall_status'].upper()}
Air Quality Index: {data['air_quality_index']}

POLLUTANT LEVELS:
"""
        for pollutant, info in data['pollutants'].items():
            content += f"- {pollutant}: {info['value']} {info['unit']} ({info['status']})\n"

        content += f"""
HEALTH RECOMMENDATIONS:
"""
        for rec in data['health_recommendations']:
            content += f"- {rec}\n"

        content += f"""
PEAK TIMES:
- Morning: {data['peak_times']['morning_peak']}
- Evening: {data['peak_times']['evening_peak']}

FORECAST: {data['forecast']}

---
Generated by Islamabad Smog Detection System
Data sources: {', '.join(data['data_sources'])}
"""
        return content

    def _create_weekly_report_html(self, data):
        """Create HTML content for weekly report"""
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Weekly Air Quality Report - Islamabad</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 700px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 20px; margin-bottom: 30px; }
                .summary-box { background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .trend-good { color: #28a745; }
                .trend-bad { color: #dc3545; }
                .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
                .stat-card { text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; }
                .stat-value { font-size: 24px; font-weight: bold; color: #495057; }
                .stat-label { font-size: 12px; color: #6c757d; }
                .pollutant-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .pollutant-table th, .pollutant-table td { padding: 10px; text-align: center; border: 1px solid #dee2e6; }
                .pollutant-table th { background-color: #f8f9fa; }
                .events-list { background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }
                .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Weekly Air Quality Report</h1>
                    <h2>Islamabad, Pakistan</h2>
                    <p><strong>{{ data.week_start }} to {{ data.week_end }}</strong></p>
                </div>

                <div class="summary-box">
                    <h3>📈 Weekly Summary</h3>
                    <p>{{ data.weekly_summary }}</p>
                    <p><strong>Trend:</strong> <span class="trend-{{ 'good' if data.trend == 'improving' else 'bad' }}">
                        {{ data.trend.title() }} {{ get_trend_emoji(data.trend) }}
                    </span></p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{{ data.average_aqi }}</div>
                        <div class="stat-label">Average AQI</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ data.peak_aqi }}</div>
                        <div class="stat-label">Peak AQI</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ data.lowest_aqi }}</div>
                        <div class="stat-label">Lowest AQI</div>
                    </div>
                </div>

                <h3>🧪 Pollutant Averages</h3>
                <table class="pollutant-table">
                    <thead>
                        <tr>
                            <th>Pollutant</th>
                            <th>Average</th>
                            <th>Peak</th>
                            <th>Unit</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for pollutant, info in data.pollutant_averages.items() %}
                        <tr>
                            <td>{{ pollutant }}</td>
                            <td>{{ info.avg }}</td>
                            <td>{{ info.peak }}</td>
                            <td>{{ info.unit }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>

                {% if data.notable_events %}
                <div class="events-list">
                    <h3>⚠️ Notable Events</h3>
                    {% for event in data.notable_events %}
                    <p><strong>{{ event.date }}:</strong> {{ event.event }} (Peak: {{ event.peak_value }})</p>
                    {% endfor %}
                </div>
                {% endif %}

                <h3>🔮 Next Week Forecast</h3>
                <p>{{ data.forecast }}</p>

                <div class="footer">
                    <p>Generated by Islamabad Smog Detection System</p>
                    <p>Generated: {{ generation_time }}</p>
                </div>
            </div>
        </body>
        </html>
        ''')

        return template.render(
            data=data,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            get_trend_emoji=self._get_trend_emoji
        )

    def _create_weekly_report_text(self, data):
        """Create plain text content for weekly report"""
        content = f"""
WEEKLY AIR QUALITY REPORT - ISLAMABAD
Period: {data['week_start']} to {data['week_end']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

WEEKLY SUMMARY:
{data['weekly_summary']}

KEY STATISTICS:
- Average AQI: {data['average_aqi']}
- Peak AQI: {data['peak_aqi']}
- Lowest AQI: {data['lowest_aqi']}
- Trend: {data['trend'].title()}

POLLUTANT AVERAGES:
"""
        for pollutant, info in data['pollutant_averages'].items():
            content += f"- {pollutant}: {info['avg']} {info['unit']} (Peak: {info['peak']})\n"

        if data['notable_events']:
            content += f"""
NOTABLE EVENTS:
"""
            for event in data['notable_events']:
                content += f"- {event['date']}: {event['event']} (Peak: {event['peak_value']})\n"

        content += f"""
NEXT WEEK FORECAST:
{data['forecast']}

---
Generated by Islamabad Smog Detection System
"""
        return content

    def _create_alert_html(self, alert_data):
        """Create HTML content for alert email"""
        severity_colors = {
            'low': '#17a2b8',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }

        color = severity_colors.get(alert_data.get('severity', 'medium'), '#ffc107')

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Air Quality Alert - Islamabad</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .alert-header {{ background-color: {color}; color: white; padding: 20px; border-radius: 10px 10px 0 0; margin: -30px -30px 30px -30px; }}
                .alert-details {{ background-color: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .urgent {{ font-size: 24px; font-weight: bold; color: {color}; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="alert-header">
                    <h1>🚨 AIR QUALITY ALERT</h1>
                    <p>Islamabad, Pakistan</p>
                    <p><strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong></p>
                </div>

                <div class="urgent">
                    {alert_data['message'].upper()}
                </div>

                <div class="alert-details">
                    <h3>📍 Affected Areas</h3>
                    <p>{', '.join(alert_data.get('locations', ['Various locations']))}</p>

                    <h3>⚠️ Severity</h3>
                    <p><strong>{alert_data.get('severity', 'unknown').upper()}</strong></p>
                </div>

                <h3>🏥 Recommended Actions</h3>
                <ul>
                    <li>Limit outdoor activities</li>
                    <li>Keep windows closed</li>
                    <li>Use air purifiers if available</li>
                    <li>Monitor symptoms if you have respiratory conditions</li>
                </ul>

                <p><small>This alert was generated automatically by the Islamabad Smog Detection System.</small></p>
            </div>
        </body>
        </html>
        '''

        return html

    def _create_alert_text(self, alert_data):
        """Create plain text content for alert email"""
        content = f"""
AIR QUALITY ALERT - ISLAMABAD
{'=' * 50}

ALERT: {alert_data['message'].upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Severity: {alert_data.get('severity', 'unknown').upper()}

Affected Areas: {', '.join(alert_data.get('locations', ['Various locations']))}

RECOMMENDED ACTIONS:
- Limit outdoor activities
- Keep windows closed
- Use air purifiers if available
- Monitor symptoms if you have respiratory conditions

This alert was generated automatically by the Islamabad Smog Detection System.
"""
        return content

    def _get_recipients(self):
        """Get list of email recipients from configuration"""
        recipients = self.alerts_config.get('email', {}).get('recipients', [])

        # Also check environment variable
        env_recipients = os.environ.get('ALERT_RECIPIENTS', '')
        if env_recipients:
            env_list = [email.strip() for email in env_recipients.split(',') if email.strip()]
            recipients.extend(env_list)

        return list(set(recipients))  # Remove duplicates

    def _get_alert_recipients(self):
        """Get list of alert-specific recipients"""
        # For alerts, we might want a different distribution list
        return self._get_recipients()  # For now, use same list

    def _get_daily_attachments(self, report_data):
        """Get attachments for daily report"""
        attachments = []

        # Add daily chart if available
        chart_path = self.reports_dir / f"daily_chart_{report_data['date']}.png"
        if chart_path.exists():
            attachments.append(str(chart_path))

        return attachments

    def _get_weekly_attachments(self, report_data):
        """Get attachments for weekly report"""
        attachments = []

        # Add weekly chart if available
        chart_path = self.reports_dir / f"weekly_chart_{report_data['week_end']}.png"
        if chart_path.exists():
            attachments.append(str(chart_path))

        return attachments

    def _send_email(self, subject, recipients, html_content=None, text_content=None,
                   attachments=None, priority='normal'):
        """
        Send email using SMTP

        Args:
            subject (str): Email subject
            recipients (list): List of recipient email addresses
            html_content (str): HTML content
            text_content (str): Plain text content
            attachments (list): List of file paths to attach
            priority (str): Email priority ('normal', 'high')

        Returns:
            dict: Result with success status and error message if failed
        """
        try:
            if not self.email_username or not self.email_password:
                return {'success': False, 'error': 'Email credentials not configured'}

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = ', '.join(recipients)

            # Add priority header if high priority
            if priority == 'high':
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'

            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)

            # Add HTML content
            if html_content:
                html_part = MIMEText(html_content, 'html')
                msg.attach(html_part)

            # Add attachments
            if attachments:
                for file_path in attachments:
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {Path(file_path).name}'
                            )
                            msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {len(recipients)} recipients")
            return {'success': True}

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {'success': False, 'error': str(e)}

    def _get_status_emoji(self, status):
        """Get emoji for air quality status"""
        emojis = {
            'good': '😊',
            'moderate': '😐',
            'unhealthy_sensitive': '😷',
            'unhealthy': '🤢',
            'very_unhealthy': '😵',
            'hazardous': '☠️'
        }
        return emojis.get(status, '❓')

    def _get_trend_emoji(self, trend):
        """Get emoji for trend"""
        emojis = {
            'improving': '📈',
            'worsening': '📉',
            'stable': '➡️'
        }
        return emojis.get(trend, '➡️')