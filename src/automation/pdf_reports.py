"""
PDF Report Generator for Islamabad Smog Detection System
Professional PDF reports for sharing and archiving
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFReporter:
    """Handles PDF report generation for air quality data"""

    def __init__(self, config):
        """
        Initialize PDF Reporter

        Args:
            config (dict): System configuration
        """
        self.config = config
        self.reports_dir = Path('data/exports/reports')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = Path('data/exports/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        # PDF styling
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Organization info
        self.org_name = config.get('organization', {}).get('name', 'Islamabad Environmental Agency')
        self.org_contact = config.get('organization', {}).get('contact', 'contact@islamabad.gov.pk')

    def generate_daily_report(self, date=None):
        """
        Generate daily air quality report in PDF format

        Args:
            date (str): Date for the report (YYYY-MM-DD). Defaults to today.

        Returns:
            str: Path to generated PDF file
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Generating daily PDF report for {date}")

            # Generate report data
            report_data = self._generate_daily_report_data(date)

            # Generate charts
            charts = self._generate_daily_charts(report_data, date)

            # Create PDF
            filename = f"daily_air_quality_report_{date}.pdf"
            filepath = self.reports_dir / filename

            doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)

            story = []

            # Title page
            story.extend(self._create_title_page(report_data, 'Daily'))

            # Executive summary
            story.extend(self._create_executive_summary(report_data))

            # Current conditions
            story.extend(self._create_current_conditions(report_data))

            # Charts
            if charts:
                story.extend(self._create_charts_section(charts))

            # Health recommendations
            story.extend(self._create_health_recommendations(report_data))

            # Footer and metadata
            story.append(Spacer(1, 20))
            story.extend(self._create_footer())

            # Build PDF
            doc.build(story, onFirstPage=self._add_header, onLaterPages=self._add_header)

            logger.info(f"Daily PDF report generated: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating daily PDF report: {e}")
            raise

    def generate_weekly_report(self, week_end=None):
        """
        Generate weekly air quality report in PDF format

        Args:
            week_end (str): End date of the week (YYYY-MM-DD). Defaults to today.

        Returns:
            str: Path to generated PDF file
        """
        try:
            if week_end is None:
                week_end = datetime.now().strftime('%Y-%m-%d')

            week_start = (datetime.strptime(week_end, '%Y-%m-%d') - timedelta(days=6)).strftime('%Y-%m-%d')

            logger.info(f"Generating weekly PDF report for {week_start} to {week_end}")

            # Generate report data
            report_data = self._generate_weekly_report_data(week_start, week_end)

            # Generate charts
            charts = self._generate_weekly_charts(report_data, week_start, week_end)

            # Create PDF
            filename = f"weekly_air_quality_report_{week_end}.pdf"
            filepath = self.reports_dir / filename

            doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)

            story = []

            # Title page
            story.extend(self._create_title_page(report_data, 'Weekly'))

            # Executive summary
            story.extend(self._create_executive_summary(report_data))

            # Weekly statistics
            story.extend(self._create_weekly_statistics(report_data))

            # Pollutant analysis
            story.extend(self._create_pollutant_analysis(report_data))

            # Charts
            if charts:
                story.extend(self._create_charts_section(charts))

            # Notable events
            story.extend(self._create_notable_events(report_data))

            # Footer and metadata
            story.append(Spacer(1, 20))
            story.extend(self._create_footer())

            # Build PDF
            doc.build(story, onFirstPage=self._add_header, onLaterPages=self._add_header)

            logger.info(f"Weekly PDF report generated: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating weekly PDF report: {e}")
            raise

    def generate_monthly_report(self, month=None, year=None):
        """
        Generate monthly air quality report in PDF format

        Args:
            month (int): Month number (1-12). Defaults to current month.
            year (int): Year. Defaults to current year.

        Returns:
            str: Path to generated PDF file
        """
        try:
            if month is None:
                month = datetime.now().month
            if year is None:
                year = datetime.now().year

            logger.info(f"Generating monthly PDF report for {month}/{year}")

            # Generate report data
            report_data = self._generate_monthly_report_data(month, year)

            # Generate charts
            charts = self._generate_monthly_charts(report_data, month, year)

            # Create PDF
            month_name = datetime(year, month, 1).strftime('%B')
            filename = f"monthly_air_quality_report_{month_name.lower()}_{year}.pdf"
            filepath = self.reports_dir / filename

            doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)

            story = []

            # Title page
            story.extend(self._create_title_page(report_data, 'Monthly'))

            # Executive summary
            story.extend(self._create_executive_summary(report_data))

            # Monthly overview
            story.extend(self._create_monthly_overview(report_data))

            # Detailed analysis
            story.extend(self._create_detailed_analysis(report_data))

            # Charts
            if charts:
                story.extend(self._create_charts_section(charts))

            # Conclusions and recommendations
            story.extend(self._create_conclusions(report_data))

            # Footer and metadata
            story.append(Spacer(1, 20))
            story.extend(self._create_footer())

            # Build PDF
            doc.build(story, onFirstPage=self._add_header, onLaterPages=self._add_header)

            logger.info(f"Monthly PDF report generated: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error generating monthly PDF report: {e}")
            raise

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            borderWidth=0,
            borderColor=colors.darkblue,
            borderPadding=5
        ))

    def _generate_daily_report_data(self, date):
        """Generate data for daily report"""
        # This would normally fetch real data from the system
        # For now, we'll generate demo data
        return {
            'date': date,
            'overall_status': 'moderate',
            'air_quality_index': 85,
            'pollutants': {
                'PM2.5': {'value': 35.7, 'unit': 'µg/m³', 'status': 'moderate', 'who_guideline': 15},
                'NO2': {'value': 25.3, 'unit': 'µg/m³', 'status': 'good', 'who_guideline': 40},
                'SO2': {'value': 12.1, 'unit': 'µg/m³', 'status': 'good', 'who_guideline': 50},
                'CO': {'value': 0.8, 'unit': 'mg/m³', 'status': 'good', 'who_guideline': 10},
                'O3': {'value': 65.2, 'unit': 'µg/m³', 'status': 'moderate', 'who_guideline': 100},
                'AOD': {'value': 0.42, 'unit': 'dimensionless', 'status': 'moderate'}
            },
            'health_recommendations': [
                "Air quality is acceptable for most people",
                "Sensitive individuals may experience minor issues",
                "Limit prolonged outdoor exertion if you are sensitive",
                "Keep windows closed during peak pollution hours"
            ],
            'peak_times': {
                'morning_peak': '08:00 - 10:00',
                'evening_peak': '18:00 - 20:00'
            },
            'forecast': 'Similar conditions expected tomorrow',
            'data_sources': ['Copernicus Sentinel-5P', 'NASA MODIS', 'Local monitoring stations'],
            'temperature': {'avg': 22, 'min': 16, 'max': 28},
            'humidity': {'avg': 65, 'min': 45, 'max': 80},
            'wind_speed': {'avg': 12, 'max': 20},
            'historical_comparison': {
                'same_day_last_year': 92,
                'monthly_average': 88,
                'yearly_average': 91
            }
        }

    def _generate_weekly_report_data(self, week_start, week_end):
        """Generate data for weekly report"""
        return {
            'week_start': week_start,
            'week_end': week_end,
            'average_aqi': 92,
            'peak_aqi': 145,
            'lowest_aqi': 45,
            'trend': 'improving',
            'pollutant_averages': {
                'PM2.5': {'avg': 38.2, 'peak': 65.4, 'unit': 'µg/m³', 'who_guideline': 15},
                'NO2': {'avg': 27.8, 'peak': 45.1, 'unit': 'µg/m³', 'who_guideline': 40},
                'SO2': {'avg': 14.3, 'peak': 28.7, 'unit': 'µg/m³', 'who_guideline': 50},
                'CO': {'avg': 0.9, 'peak': 1.4, 'unit': 'mg/m³', 'who_guideline': 10},
                'O3': {'avg': 68.5, 'peak': 95.2, 'unit': 'µg/m³', 'who_guideline': 100}
            },
            'daily_data': self._generate_demo_daily_data(week_start, week_end),
            'notable_events': [
                {
                    'date': '2024-01-15',
                    'event': 'High PM2.5 levels detected in industrial area',
                    'peak_value': '65.4 µg/m³',
                    'cause': 'Industrial activity combined with low wind speeds'
                },
                {
                    'date': '2024-01-18',
                    'event': 'Improved air quality due to rainfall',
                    'lowest_value': '28.1 µg/m³',
                    'impact': 'Rain helped wash away pollutants'
                }
            ],
            'weekly_summary': "Air quality showed gradual improvement throughout the week with weekend conditions being the best. The mid-week period experienced elevated pollution levels due to industrial activity and unfavorable meteorological conditions.",
            'forecast': "Expected improvement in early next week due to changing weather patterns and increased wind speeds.",
            'compliance_status': {
                'days_within_guidelines': 3,
                'days_exceeding_guidelines': 4,
                'compliance_percentage': 43
            }
        }

    def _generate_monthly_report_data(self, month, year):
        """Generate data for monthly report"""
        month_name = datetime(year, month, 1).strftime('%B')
        days_in_month = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days if month < 12 else 31

        return {
            'month': month_name,
            'year': year,
            'average_aqi': 95,
            'peak_aqi': 165,
            'lowest_aqi': 38,
            'monthly_trend': 'stable',
            'pollutant_monthly_averages': {
                'PM2.5': {'avg': 41.5, 'peak': 78.2, 'min': 18.3, 'unit': 'µg/m³'},
                'NO2': {'avg': 29.8, 'peak': 52.1, 'min': 15.6, 'unit': 'µg/m³'},
                'SO2': {'avg': 16.2, 'peak': 35.7, 'min': 8.1, 'unit': 'µg/m³'},
                'CO': {'avg': 1.1, 'peak': 2.3, 'min': 0.4, 'unit': 'mg/m³'},
                'O3': {'avg': 72.3, 'peak': 118.5, 'min': 45.2, 'unit': 'µg/m³'}
            },
            'weekly_breakdown': self._generate_demo_weekly_data(month, year),
            'meteorological_summary': {
                'avg_temperature': 18.5,
                'total_rainfall': 45.2,
                'avg_humidity': 62,
                'dominant_wind_direction': 'Northwest',
                'calm_days': 3
            },
            'compliance_analysis': {
                'days_within_who_guidelines': 8,
                'days_exceeding_guidelines': 22,
                'compliance_percentage': 26.7,
                'critical_pollutant': 'PM2.5',
                'exceedance_days': 18
            },
            'monthly_summary': f"The month of {month_name} {year} showed variable air quality conditions with periodic episodes of elevated pollution levels. The overall trend remained stable compared to previous months, with PM2.5 being the primary concern, exceeding WHO guidelines on 18 days.",
            'recommendations': [
                "Implement stricter industrial emission controls during winter months",
                "Increase green cover along major transportation corridors",
                "Promote public transportation to reduce vehicle emissions",
                "Establish more monitoring stations in industrial areas",
                "Develop early warning system for pollution episodes"
            ]
        }

    def _generate_demo_daily_data(self, start_date, end_date):
        """Generate demo daily data for weekly report"""
        from datetime import datetime, timedelta

        daily_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

        while current_date <= end_date_obj:
            # Generate random-ish values with some pattern
            base_aqi = 80 + (current_date.weekday() * 5)  # Weekends better
            aqi = base_aqi + (hash(str(current_date)) % 30) - 15  # Add some randomness

            daily_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day_name': current_date.strftime('%A'),
                'aqi': max(30, min(200, aqi)),  # Keep within reasonable range
                'pm25': max(10, min(100, aqi * 0.4)),
                'no2': max(5, min(80, aqi * 0.3)),
                'status': 'good' if aqi < 50 else 'moderate' if aqi < 100 else 'unhealthy_sensitive'
            })
            current_date += timedelta(days=1)

        return daily_data

    def _generate_demo_weekly_data(self, month, year):
        """Generate demo weekly data for monthly report"""
        weekly_data = []
        start_date = datetime(year, month, 1)

        for week in range(5):  # Max 5 weeks in a month
            week_start = start_date + timedelta(weeks=week)
            if week_start.month != month:
                break

            week_end = week_start + timedelta(days=6)
            if week_end.month != month:
                week_end = datetime(year, month, 1) + timedelta(days=32)  # End of month
                week_end = week_end.replace(day=1) - timedelta(days=1)

            avg_aqi = 85 + (week * 10) + (hash(f"week_{week}") % 20) - 10

            weekly_data.append({
                'week': week + 1,
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'average_aqi': max(40, min(180, avg_aqi)),
                'peak_aqi': avg_aqi + 30,
                'status': 'good' if avg_aqi < 50 else 'moderate' if avg_aqi < 100 else 'unhealthy_sensitive'
            })

        return weekly_data

    def _generate_daily_charts(self, data, date):
        """Generate charts for daily report"""
        charts = {}

        try:
            # Pollutant comparison chart
            pollutants = list(data['pollutants'].keys())
            values = [data['pollutants'][p]['value'] for p in pollutants]
            units = [data['pollutants'][p]['unit'] for p in pollutants]
            guidelines = [data['pollutants'][p].get('who_guideline', values[i]) for i, p in enumerate(pollutants)]

            # Normalize values to same scale for comparison
            normalized_values = []
            normalized_guidelines = []
            for i, (value, guideline, unit) in enumerate(zip(values, guidelines, units)):
                if unit == 'mg/m³':
                    # Convert to µg/m³ for comparison
                    value *= 1000
                    guideline *= 1000
                normalized_values.append(value)
                normalized_guidelines.append(guideline)

            plt.figure(figsize=(10, 6))
            x_pos = range(len(pollutants))
            bars = plt.bar(x_pos, normalized_values, color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db', '#1abc9c'])
            plt.axhline(y=guidelines[0], color='red', linestyle='--', alpha=0.7, label='WHO Guidelines')

            plt.xlabel('Pollutants')
            plt.ylabel('Concentration (µg/m³)')
            plt.title(f'Pollutant Levels - {date}')
            plt.xticks(x_pos, pollutants)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save chart
            chart_path = self.charts_dir / f"daily_pollutants_{date}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['pollutants'] = chart_path

            # Historical comparison chart
            historical = data['historical_comparison']
            categories = ['Today', 'Same Day Last Year', 'Monthly Average', 'Yearly Average']
            values_hist = [data['air_quality_index'], historical['same_day_last_year'],
                          historical['monthly_average'], historical['yearly_average']]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, values_hist, color=['#3498db', '#95a5a6', '#ecf0f1', '#bdc3c7'])
            plt.ylabel('Air Quality Index')
            plt.title('Historical AQI Comparison')
            plt.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars, values_hist):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(value), ha='center', va='bottom')

            chart_path = self.charts_dir / f"daily_historical_{date}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['historical'] = chart_path

        except Exception as e:
            logger.error(f"Error generating daily charts: {e}")

        return charts

    def _generate_weekly_charts(self, data, week_start, week_end):
        """Generate charts for weekly report"""
        charts = {}

        try:
            # Daily AQI trend chart
            daily_data = data['daily_data']
            dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_data]
            aqi_values = [d['aqi'] for d in daily_data]
            pm25_values = [d['pm25'] for d in daily_data]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # AQI trend
            ax1.plot(dates, aqi_values, marker='o', linewidth=2, markersize=6, color='#3498db')
            ax1.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
            ax1.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Unhealthy Threshold')
            ax1.set_ylabel('Air Quality Index')
            ax1.set_title('Daily AQI Trend')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))

            # PM2.5 trend
            ax2.plot(dates, pm25_values, marker='s', linewidth=2, markersize=6, color='#e74c3c')
            ax2.axhline(y=35, color='orange', linestyle='--', alpha=0.7, label='Pakistani Standard')
            ax2.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='WHO Guideline')
            ax2.set_ylabel('PM2.5 (µg/m³)')
            ax2.set_title('Daily PM2.5 Levels')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))

            plt.tight_layout()

            chart_path = self.charts_dir / f"weekly_trends_{week_end}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['trends'] = chart_path

            # Pollutant averages comparison
            pollutants = list(data['pollutant_averages'].keys())
            averages = [data['pollutant_averages'][p]['avg'] for p in pollutants]
            peaks = [data['pollutant_averages'][p]['peak'] for p in pollutants]

            plt.figure(figsize=(12, 6))
            x_pos = range(len(pollutants))
            width = 0.35

            plt.bar([x - width/2 for x in x_pos], averages, width, label='Average', color='#3498db')
            plt.bar([x + width/2 for x in x_pos], peaks, width, label='Peak', color='#e74c3c')

            plt.xlabel('Pollutants')
            plt.ylabel('Concentration')
            plt.title('Weekly Pollutant Averages vs Peaks')
            plt.xticks(x_pos, pollutants)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

            chart_path = self.charts_dir / f"weekly_pollutants_{week_end}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['pollutants'] = chart_path

        except Exception as e:
            logger.error(f"Error generating weekly charts: {e}")

        return charts

    def _generate_monthly_charts(self, data, month, year):
        """Generate charts for monthly report"""
        charts = {}

        try:
            # Weekly breakdown chart
            weekly_data = data['weekly_breakdown']
            weeks = [f"Week {w['week']}" for w in weekly_data]
            avg_aqi = [w['average_aqi'] for w in weekly_data]
            peak_aqi = [w['peak_aqi'] for w in weekly_data]

            plt.figure(figsize=(12, 6))
            x_pos = range(len(weeks))
            width = 0.35

            plt.bar([x - width/2 for x in x_pos], avg_aqi, width, label='Average AQI', color='#3498db')
            plt.bar([x + width/2 for x in x_pos], peak_aqi, width, label='Peak AQI', color='#e74c3c')

            plt.xlabel('Weeks')
            plt.ylabel('Air Quality Index')
            plt.title(f'Weekly AQI Breakdown - {data["month"]} {year}')
            plt.xticks(x_pos, weeks)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

            chart_path = self.charts_dir / f"monthly_weekly_{data['month'].lower()}_{year}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['weekly'] = chart_path

            # Compliance analysis chart
            compliance = data['compliance_analysis']
            categories = ['Within WHO Guidelines', 'Exceeding Guidelines']
            values = [compliance['days_within_who_guidelines'], compliance['days_exceeding_guidelines']]
            colors = ['#2ecc71', '#e74c3c']

            plt.figure(figsize=(8, 8))
            plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Monthly Compliance Analysis - {data["month"]} {year}')

            chart_path = self.charts_dir / f"monthly_compliance_{data['month'].lower()}_{year}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['compliance'] = chart_path

        except Exception as e:
            logger.error(f"Error generating monthly charts: {e}")

        return charts

    def _create_title_page(self, data, report_type):
        """Create title page content"""
        story = []

        # Main title
        title = f"{report_type.upper()} AIR QUALITY REPORT"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))

        # Subtitle
        if report_type == 'Daily':
            subtitle = f"Islamabad, Pakistan - {data['date']}"
        elif report_type == 'Weekly':
            subtitle = f"Islamabad, Pakistan - {data['week_start']} to {data['week_end']}"
        else:  # Monthly
            subtitle = f"Islamabad, Pakistan - {data['month']} {data['year']}"

        story.append(Paragraph(subtitle, self.styles['CustomSubtitle']))
        story.append(Spacer(1, 40))

        # Organization info
        org_text = f"""
        <para align="center">
        <b>{self.org_name}</b><br/>
        Environmental Monitoring Division<br/>
        Air Quality Assessment Report
        </para>
        """
        story.append(Paragraph(org_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Report date
        report_date_text = f"""
        <para align="center">
        <b>Report Generated:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}<br/>
        <b>Report ID:</b> {self._generate_report_id(report_type, data)}
        </para>
        """
        story.append(Paragraph(report_date_text, self.styles['Normal']))

        return story

    def _create_executive_summary(self, data):
        """Create executive summary section"""
        story = []
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

        if 'overall_status' in data:  # Daily report
            summary_text = f"""
            On {data['date']}, Islamabad recorded an Air Quality Index (AQI) of {data['air_quality_index']},
            indicating {data['overall_status']} air quality conditions. The primary pollutant of concern
            was PM2.5 with concentrations reaching {data['pollutants']['PM2.5']['value']} µg/m³.
            """
        elif 'trend' in data:  # Weekly report
            summary_text = f"""
            During the period from {data['week_start']} to {data['week_end']}, Islamabad experienced
            an average AQI of {data['average_aqi']}, with a peak of {data['peak_aqi']}. The overall trend
            for the week was {data['trend']}, with {data.get('compliance_status', {}).get('compliance_percentage', 0)}% of days
            meeting WHO air quality guidelines.
            """
        else:  # Monthly report
            summary_text = f"""
            In {data['month']} {data['year']}, Islamabad maintained an average AQI of {data['average_aqi']},
            with monthly compliance with WHO guidelines at {data['compliance_analysis']['compliance_percentage']:.1f}%.
            The primary pollutant of concern was {data['compliance_analysis']['critical_pollutant']}.
            """

        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        return story

    def _create_current_conditions(self, data):
        """Create current conditions section"""
        story = []
        story.append(Paragraph("Current Air Quality Conditions", self.styles['SectionHeader']))

        # Create pollutant table
        table_data = [['Pollutant', 'Concentration', 'WHO Guideline', 'Status']]

        for pollutant, info in data['pollutants'].items():
            if pollutant != 'AOD':  # Skip AOD for main table
                status_emoji = self._get_status_emoji(info['status'])
                table_data.append([
                    pollutant,
                    f"{info['value']} {info['unit']}",
                    f"{info.get('who_guideline', 'N/A')} {info['unit']}",
                    f"{info['status'].title()} {status_emoji}"
                ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        return story

    def _create_health_recommendations(self, data):
        """Create health recommendations section"""
        story = []
        story.append(Paragraph("Health Recommendations", self.styles['SectionHeader']))

        for recommendation in data['health_recommendations']:
            story.append(Paragraph(f"• {recommendation}", self.styles['Normal']))

        story.append(Spacer(1, 20))

        return story

    def _create_charts_section(self, charts):
        """Create charts section"""
        story = []
        story.append(Paragraph("Visual Analysis", self.styles['SectionHeader']))

        for chart_name, chart_path in charts.items():
            if chart_path.exists():
                try:
                    img = ReportLabImage(str(chart_path), width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                except Exception as e:
                    logger.error(f"Error including chart {chart_name}: {e}")

        return story

    def _create_weekly_statistics(self, data):
        """Create weekly statistics section"""
        story = []
        story.append(Paragraph("Weekly Statistics", self.styles['SectionHeader']))

        # Key metrics table
        stats_data = [
            ['Metric', 'Value'],
            ['Average AQI', str(data['average_aqi'])],
            ['Peak AQI', str(data['peak_aqi'])],
            ['Lowest AQI', str(data['lowest_aqi'])],
            ['Weekly Trend', data['trend'].title()]
        ]

        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(stats_table)
        story.append(Spacer(1, 20))

        return story

    def _create_pollutant_analysis(self, data):
        """Create pollutant analysis section"""
        story = []
        story.append(Paragraph("Pollutant Analysis", self.styles['SectionHeader']))

        # Pollutant averages table
        table_data = [['Pollutant', 'Average', 'Peak', 'Unit']]

        for pollutant, info in data['pollutant_averages'].items():
            table_data.append([
                pollutant,
                f"{info['avg']:.1f}",
                f"{info['peak']:.1f}",
                info['unit']
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        return story

    def _create_notable_events(self, data):
        """Create notable events section"""
        story = []

        if data.get('notable_events'):
            story.append(Paragraph("Notable Events", self.styles['SectionHeader']))

            for event in data['notable_events']:
                event_text = f"""
                <b>{event['date']}:</b> {event['event']}<br/>
                <i>Details:</i> {event.get('cause', event.get('impact', 'No additional details available.'))}
                """
                story.append(Paragraph(event_text, self.styles['Normal']))
                story.append(Spacer(1, 10))

        return story

    def _create_monthly_overview(self, data):
        """Create monthly overview section"""
        story = []
        story.append(Paragraph("Monthly Overview", self.styles['SectionHeader']))

        # Summary text
        story.append(Paragraph(data['monthly_summary'], self.styles['Normal']))
        story.append(Spacer(1, 20))

        return story

    def _create_detailed_analysis(self, data):
        """Create detailed analysis section"""
        story = []
        story.append(Paragraph("Detailed Analysis", self.styles['SectionHeader']))

        # Meteorological conditions
        story.append(Paragraph("Meteorological Conditions", self.styles['Heading3']))
        met_data = data['meteorological_summary']

        met_text = f"""
        Average Temperature: {met_data['avg_temperature']}°C<br/>
        Total Rainfall: {met_data['total_rainfall']} mm<br/>
        Average Humidity: {met_data['avg_humidity']}%<br/>
        Dominant Wind Direction: {met_data['dominant_wind_direction']}<br/>
        Calm Days: {met_data['calm_days']}
        """
        story.append(Paragraph(met_text, self.styles['Normal']))
        story.append(Spacer(1, 15))

        # Compliance analysis
        compliance = data['compliance_analysis']
        story.append(Paragraph("Compliance Analysis", self.styles['Heading3']))

        compliance_text = f"""
        Days within WHO Guidelines: {compliance['days_within_who_guidelines']}<br/>
        Days Exceeding Guidelines: {compliance['days_exceeding_guidelines']}<br/>
        Compliance Percentage: {compliance['compliance_percentage']:.1f}%<br/>
        Critical Pollutant: {compliance['critical_pollutant']}<br/>
        Guideline Exceedance Days: {compliance['exceedance_days']}
        """
        story.append(Paragraph(compliance_text, self.styles['Normal']))
        story.append(Spacer(1, 20))

        return story

    def _create_conclusions(self, data):
        """Create conclusions and recommendations section"""
        story = []
        story.append(Paragraph("Conclusions and Recommendations", self.styles['SectionHeader']))

        # Summary conclusions
        story.append(Paragraph("Key Findings", self.styles['Heading3']))
        story.append(Paragraph(data['monthly_summary'], self.styles['Normal']))
        story.append(Spacer(1, 15))

        # Recommendations
        story.append(Paragraph("Recommendations", self.styles['Heading3']))
        for recommendation in data['recommendations']:
            story.append(Paragraph(f"• {recommendation}", self.styles['Normal']))

        story.append(Spacer(1, 20))

        return story

    def _create_footer(self):
        """Create footer content"""
        story = []

        footer_text = f"""
        <para align="center" fontSize="8" textColor="gray">
        <br/><br/>
        Generated by Islamabad Smog Detection System v1.0<br/>
        {self.org_name} | {self.org_contact}<br/>
        This report contains data from satellite monitoring and ground-based sensors.<br/>
        Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}
        </para>
        """

        story.append(Paragraph(footer_text, self.styles['Normal']))

        return story

    def _add_header(self, canvas, doc):
        """Add header to each page"""
        canvas.saveState()

        # Header text
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.darkblue)
        canvas.drawString(inch, A4[1] - 0.5 * inch, "Islamabad Smog Detection System - Air Quality Report")

        # Page number
        canvas.setFillColor(colors.black)
        canvas.drawRightString(A4[0] - inch, A4[1] - 0.5 * inch, f"Page {doc.page}")

        canvas.restoreState()

    def _generate_report_id(self, report_type, data):
        """Generate unique report ID"""
        if report_type == 'Daily':
            return f"DAILY_{data['date'].replace('-', '')}_{datetime.now().strftime('%H%M')}"
        elif report_type == 'Weekly':
            return f"WEEKLY_{data['week_end'].replace('-', '')}_{datetime.now().strftime('%H%M')}"
        else:  # Monthly
            return f"MONTHLY_{data['month'].lower()}_{data['year']}_{datetime.now().strftime('%H%M')}"

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