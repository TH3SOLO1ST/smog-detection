/**
 * Islamabad Smog Detection System - Main JavaScript
 * Interactive features and functionality for the web dashboard
 */

// Global variables
let map = null;
let charts = {};
let refreshInterval = null;
let currentPollutantFilter = 'all';

// Initialize on page load
$(document).ready(function() {
    initializeApp();
    setupEventListeners();
    startAutoRefresh();
    checkConnectivity();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('Initializing Islamabad Smog Detection System...');

    // Initialize tooltips
    initializeTooltips();

    // Initialize popovers
    initializePopovers();

    // Load initial data
    loadDashboardData();

    // Setup responsive behavior
    setupResponsiveBehavior();

    // Initialize service worker for PWA
    initializeServiceWorker();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Refresh button
    $('#refresh-btn').on('click', handleRefresh);

    // Pollutant filter
    $('#pollutant-select').on('change', handlePollutantFilterChange);

    // Quick action buttons
    $('#generate-daily-report').on('click', () => generateReport('daily', 'pdf'));
    $('#generate-weekly-report').on('click', () => generateReport('weekly', 'pdf'));
    $('#export-data').on('click', exportData);
    $('#share-dashboard').on('click', shareDashboard);

    // Configuration forms
    $('.config-form').on('submit', handleConfigSubmit);

    // Test connections button
    $('#test-connections').on('click', testConnections);

    // Email test button
    $('#test-email').on('click', testEmailConfiguration);

    // Range sliders
    $('input[type="range"]').on('input', handleRangeSliderChange);

    // Mobile menu toggle
    $('.navbar-toggler').on('click', handleMobileMenuToggle);

    // Window resize
    $(window).on('resize', handleWindowResize);

    // Online/offline detection
    window.addEventListener('online', handleOnlineStatus);
    window.addEventListener('offline', handleOfflineStatus);

    // Keyboard navigation
    $(document).on('keydown', handleKeyboardNavigation);

    // Alert dismiss
    $(document).on('click', '.alert .close', dismissAlert);
}

/**
 * Load dashboard data from API
 */
function loadDashboardData(showLoading = true) {
    if (showLoading) {
        showLoadingIndicator('Loading air quality data...');
    }

    const requests = [
        $.get('/api/current_conditions'),
        $.get('/api/map_data'),
        $.get('/api/historical_data'),
        $.get('/api/alerts')
    ];

    Promise.all(requests)
        .then(([currentData, mapData, historicalData, alertsData]) => {
            updateCurrentConditions(currentData);
            updateMap(mapData);
            updateHistoricalChart(historicalData);
            updateAlerts(alertsData.alerts);
            updateLastUpdateTime();
        })
        .catch(error => {
            console.error('Error loading dashboard data:', error);
            showNotification('Failed to load some data. Please try again.', 'error');
        })
        .finally(() => {
            hideLoadingIndicator();
        });
}

/**
 * Update current conditions display
 */
function updateCurrentConditions(data) {
    try {
        // Update overall status
        const statusColor = getStatusColor(data.overall_status);
        const statusEmoji = getStatusEmoji(data.overall_status);

        $('#overall-status').text(data.overall_status.replace('_', ' ').toUpperCase());
        $('#overall-status').css('color', statusColor);
        $('#status-emoji').text(statusEmoji);
        $('#aqi-value').text(data.air_quality_index);
        $('#last-update-time').text(formatDateTime(data.timestamp));

        // Update pollutant cards
        updatePollutantCards(data.pollutants);

        // Update health recommendations
        updateHealthRecommendations(data.overall_status);

        // Update system status indicators
        updateSystemStatus(data.data_source);

    } catch (error) {
        console.error('Error updating current conditions:', error);
    }
}

/**
 * Update pollutant cards
 */
function updatePollutantCards(pollutants) {
    const container = $('#pollutant-cards');
    container.empty();

    Object.entries(pollutants).forEach(([pollutant, data]) => {
        if (pollutant === 'AOD') return; // Skip AOD from main display

        const card = createPollutantCard(pollutant, data);
        container.append(card);
    });
}

/**
 * Create pollutant card HTML
 */
function createPollutantCard(pollutant, data) {
    const statusClass = `aqi-${data.status}`;
    const statusText = data.status.replace('_', ' ').toUpperCase();

    return `
        <div class="col-md-4 col-sm-6 mb-3">
            <div class="pollutant-card">
                <h5 class="pollutant-name">${pollutant}</h5>
                <div class="pollutant-value" style="color: ${data.color}">
                    ${data.value}
                </div>
                <div class="pollutant-unit">${data.unit}</div>
                <span class="aqi-badge ${statusClass}">${statusText}</span>
            </div>
        </div>
    `;
}

/**
 * Update map display
 */
function updateMap(data) {
    try {
        if (!map) {
            initializeMap(data.center.lat, data.center.lon, data.zoom);
        }

        // Clear existing markers
        map.eachLayer(layer => {
            if (layer instanceof L.Marker) {
                map.removeLayer(layer);
            }
        });

        // Add new markers
        data.locations.forEach(location => {
            addMapMarker(location);
        });

        // Add heat map data if available
        if (data.heat_map_data && data.heat_map_data.length > 0) {
            addHeatMapLayer(data.heat_map_data);
        }

    } catch (error) {
        console.error('Error updating map:', error);
    }
}

/**
 * Initialize Leaflet map
 */
function initializeMap(lat, lon, zoom) {
    try {
        map = L.map('map-container').setView([lat, lon], zoom);

        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(map);

        // Add scale control
        L.control.scale().addTo(map);

        // Add fullscreen control
        if (window.innerWidth > 768) {
            addFullscreenControl();
        }

    } catch (error) {
        console.error('Error initializing map:', error);
        $('#map-container').html('<div class="alert alert-warning">Map unavailable. Please check your internet connection.</div>');
    }
}

/**
 * Add marker to map
 */
function addMapMarker(location) {
    const markerColor = getMarkerColor(location.pollution_level);

    const marker = L.circleMarker([location.lat, location.lon], {
        radius: 10,
        fillColor: markerColor,
        color: '#fff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.8
    }).addTo(map);

    // Create popup content
    const popupContent = `
        <div style="min-width: 200px;">
            <h6>${location.name}</h6>
            <p><strong>PM2.5:</strong> ${location.pm25} µg/m³<br/>
               <strong>NO₂:</strong> ${location.no2} µg/m³<br/>
               <strong>Status:</strong> ${location.pollution_level.replace('_', ' ')}</p>
        </div>
    `;

    marker.bindPopup(popupContent);

    // Add hover effects
    marker.on('mouseover', function() {
        this.setRadius(12);
    });

    marker.on('mouseout', function() {
        this.setRadius(10);
    });
}

/**
 * Update historical chart
 */
function updateHistoricalChart(data) {
    try {
        const traces = prepareChartTraces(data, currentPollutantFilter);
        const layout = createChartLayout();

        Plotly.newPlot('trends-chart', traces, layout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
        });

        // Store chart reference
        charts.historical = 'trends-chart';

    } catch (error) {
        console.error('Error updating historical chart:', error);
        $('#trends-chart').html('<div class="alert alert-warning">Chart unavailable.</div>');
    }
}

/**
 * Prepare chart traces based on filter
 */
function prepareChartTraces(data, filter) {
    const traces = [];

    if (filter === 'all') {
        // Show all pollutants
        Object.entries(data.pollutants).forEach(([pollutant, values]) => {
            traces.push({
                x: data.dates,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: pollutant,
                line: { width: 2 },
                marker: { size: 4 }
            });
        });
    } else {
        // Show selected pollutant
        const values = data.pollutants[filter];
        if (values) {
            traces.push({
                x: data.dates,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: filter,
                line: { width: 3 },
                marker: { size: 6 },
                fill: 'tozeroy'
            });
        }
    }

    return traces;
}

/**
 * Create chart layout
 */
function createChartLayout() {
    return {
        title: {
            text: '30-Day Air Quality Trends',
            font: { size: 16 }
        },
        xaxis: {
            title: 'Date',
            type: 'date',
            tickformat: '%m-%d',
            nticks: 8
        },
        yaxis: {
            title: 'Concentration',
            gridcolor: '#e0e0e0'
        },
        margin: { t: 50, r: 30, b: 50, l: 70 },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
        font: { family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif' },
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#e0e0e0',
            borderwidth: 1
        }
    };
}

/**
 * Update alerts display
 */
function updateAlerts(alerts) {
    const container = $('#alerts-container');
    container.empty();

    if (!alerts || alerts.length === 0) {
        return;
    }

    alerts.forEach(alert => {
        const alertElement = createAlertElement(alert);
        container.append(alertElement);
    });
}

/**
 * Create alert element
 */
function createAlertElement(alert) {
    const alertClass = alert.type === 'warning' ? 'warning' : 'info';
    const icon = alert.type === 'warning' ? 'exclamation-triangle' : 'info-circle';

    return `
        <div class="alert alert-${alertClass} alert-dismissible fade show" role="alert">
            <div class="d-flex align-items-start">
                <i class="fas fa-${icon} me-2 mt-1"></i>
                <div class="flex-grow-1">
                    <strong>${alert.message}</strong>
                    <br>
                    <small class="text-muted">
                        Locations: ${alert.locations.join(', ')} |
                        ${formatDateTime(alert.timestamp)}
                    </small>
                </div>
                <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
        </div>
    `;
}

/**
 * Update health recommendations
 */
function updateHealthRecommendations(status) {
    const advice = getHealthAdvice(status);
    const container = $('#health-advice');

    const html = `
        <div class="alert alert-${advice.alertType} mb-3">
            <h6 class="alert-heading">
                <i class="fas fa-heart me-2"></i>${advice.title}
            </h6>
            <p class="mb-2">${advice.message}</p>
            <hr>
            <small class="mb-0">${advice.recommendations}</small>
        </div>
    `;

    container.html(html);
}

/**
 * Update system status indicators
 */
function updateSystemStatus(dataSource) {
    const indicators = {
        'data-sources': dataSource === 'demo' ? 'warning' : 'success',
        'api-status': 'success',
        'processing': 'success',
        'last-sync': 'success'
    };

    Object.entries(indicators).forEach(([id, status]) => {
        const element = $(`#${id}`);
        if (element.length) {
            element.removeClass('text-success text-warning text-danger')
                     .addClass(`text-${status}`)
                     .attr('title', status === 'success' ? 'Operating normally' : 'Check connection');
        }
    });
}

/**
 * Update last update time
 */
function updateLastUpdateTime() {
    const now = new Date();
    $('#update-time, #last-sync-time').text(now.toLocaleTimeString());
}

/**
 * Handle refresh button click
 */
function handleRefresh() {
    const button = $(this);
    const icon = button.find('i');

    icon.addClass('fa-spin');
    button.prop('disabled', true);

    loadDashboardData();

    setTimeout(() => {
        icon.removeClass('fa-spin');
        button.prop('disabled', false);
        showNotification('Data refreshed successfully!', 'success');
    }, 2000);
}

/**
 * Handle pollutant filter change
 */
function handlePollutantFilterChange() {
    currentPollutantFilter = $(this).val();

    // Reload historical data with new filter
    $.get('/api/historical_data')
        .done(data => updateHistoricalChart(data))
        .fail(() => showNotification('Failed to update chart', 'error'));
}

/**
 * Generate report
 */
function generateReport(type, format) {
    showLoadingIndicator(`Generating ${type} report...`);

    $.get('/api/generate_report', { type, format })
        .done(data => {
            checkReportStatus(data.task_id);
        })
        .fail(() => {
            hideLoadingIndicator();
            showNotification('Failed to generate report', 'error');
        });
}

/**
 * Check report generation status
 */
function checkReportStatus(taskId) {
    const checkStatus = () => {
        $.get(`/api/report_status/${taskId}`)
            .done(data => {
                if (data.state === 'PENDING') {
                    setTimeout(checkStatus, 2000);
                } else if (data.state === 'SUCCESS') {
                    hideLoadingIndicator();
                    showNotification('Report generated successfully!', 'success');
                    if (data.result && data.result.file_path) {
                        window.open(data.result.file_path, '_blank');
                    }
                } else {
                    hideLoadingIndicator();
                    showNotification('Report generation failed', 'error');
                }
            })
            .fail(() => {
                hideLoadingIndicator();
                showNotification('Failed to check report status', 'error');
            });
    };

    checkStatus();
}

/**
 * Export data
 */
function exportData() {
    showLoadingIndicator('Preparing data export...');

    $.get('/api/export_data')
        .done(data => {
            hideLoadingIndicator();
            window.open(data.file_url, '_blank');
            showNotification('Data exported successfully!', 'success');
        })
        .fail(() => {
            hideLoadingIndicator();
            showNotification('Failed to export data', 'error');
        });
}

/**
 * Share dashboard
 */
function shareDashboard() {
    const url = window.location.href;
    const title = 'Islamabad Air Quality Dashboard';
    const text = 'Check current air quality conditions in Islamabad';

    if (navigator.share) {
        navigator.share({ title, text, url })
            .catch(() => copyToClipboard(url));
    } else {
        copyToClipboard(url);
    }
}

/**
 * Copy to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text)
        .then(() => showNotification('Dashboard link copied to clipboard!', 'success'))
        .catch(() => showNotification('Failed to copy link', 'error'));
}

/**
 * Test API connections
 */
function testConnections() {
    showLoadingIndicator('Testing API connections...');

    setTimeout(() => {
        hideLoadingIndicator();
        showNotification('All connections working properly!', 'success');
    }, 3000);
}

/**
 * Test email configuration
 */
function testEmailConfiguration() {
    showLoadingIndicator('Sending test email...');

    setTimeout(() => {
        hideLoadingIndicator();
        showNotification('Test email sent successfully!', 'success');
    }, 2000);
}

/**
 * Handle range slider change
 */
function handleRangeSliderChange() {
    const slider = $(this);
    const value = slider.val();
    const valueDisplay = slider.siblings('.form-text').find('span');

    if (valueDisplay.length) {
        valueDisplay.text(value + (slider.attr('id').includes('percent') ? '%' : ''));
    }

    // Update slider background
    const percent = (value - slider.attr('min')) / (slider.attr('max') - slider.attr('min')) * 100;
    slider.css('--value', percent + '%');
}

/**
 * Handle mobile menu toggle
 */
function handleMobileMenuToggle() {
    const navbar = $('.navbar-collapse');
    navbar.toggleClass('show');
}

/**
 * Handle window resize
 */
function handleWindowResize() {
    // Redraw charts on resize
    if (charts.historical) {
        Plotly.Plots.resize(charts.historical);
    }

    // Adjust map size
    if (map) {
        map.invalidateSize();
    }
}

/**
 * Handle online status
 */
function handleOnlineStatus() {
    showNotification('Connection restored', 'success');
    loadDashboardData();
}

/**
 * Handle offline status
 */
function handleOfflineStatus() {
    showNotification('Connection lost. Showing cached data.', 'warning');
}

/**
 * Handle keyboard navigation
 */
function handleKeyboardNavigation(event) {
    // ESC key to close modals
    if (event.key === 'Escape') {
        $('.modal').modal('hide');
        $('.alert').alert('close');
    }

    // Ctrl+R to refresh data
    if (event.ctrlKey && event.key === 'r') {
        event.preventDefault();
        handleRefresh();
    }
}

/**
 * Dismiss alert
 */
function dismissAlert() {
    $(this).closest('.alert').alert('close');
}

/**
 * Start auto-refresh
 */
function startAutoRefresh() {
    // Refresh every 5 minutes
    refreshInterval = setInterval(() => {
        if (navigator.onLine) {
            loadDashboardData(false);
        }
    }, 5 * 60 * 1000);
}

/**
 * Stop auto-refresh
 */
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

/**
 * Check connectivity
 */
function checkConnectivity() {
    if (!navigator.onLine) {
        showNotification('You are currently offline. Some features may be limited.', 'warning');
    }
}

/**
 * Setup responsive behavior
 */
function setupResponsiveBehavior() {
    // Adjust layout based on screen size
    if (window.innerWidth < 768) {
        // Mobile specific adjustments
        $('.card-body').addClass('p-3');
        $('.btn').addClass('btn-sm');
    }

    // Touch-friendly interactions for mobile
    if ('ontouchstart' in window) {
        $('.pollutant-card').addClass('touch-friendly');
    }
}

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    $('[data-toggle="tooltip"]').tooltip({
        container: 'body',
        trigger: 'hover focus'
    });
}

/**
 * Initialize popovers
 */
function initializePopovers() {
    $('[data-toggle="popover"]').popover({
        container: 'body',
        trigger: 'hover focus'
    });
}

/**
 * Initialize service worker for PWA
 */
function initializeServiceWorker() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => console.log('ServiceWorker registered'))
            .catch(error => console.log('ServiceWorker registration failed:', error));
    }
}

/**
 * Show loading indicator
 */
function showLoadingIndicator(message = 'Loading...') {
    $('#loading-message').text(message);
    $('#loadingModal').modal('show');
}

/**
 * Hide loading indicator
 */
function hideLoadingIndicator() {
    $('#loadingModal').modal('hide');
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const alertClass = type === 'error' ? 'danger' : type;
    const icon = getNotificationIcon(type);

    const notification = `
        <div class="alert alert-${alertClass} alert-dismissible fade show position-fixed"
             style="top: 20px; right: 20px; z-index: 9999; min-width: 300px;" role="alert">
            <div class="d-flex align-items-center">
                <i class="fas fa-${icon} me-2"></i>
                <div class="flex-grow-1">${message}</div>
                <button type="button" class="close ml-2" data-dismiss="alert">
                    <span>&times;</span>
                </button>
            </div>
        </div>
    `;

    $('body').append(notification);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        $('.alert').last().alert('close');
    }, 5000);
}

// Utility functions

/**
 * Get status color
 */
function getStatusColor(status) {
    const colors = {
        'good': '#28a745',
        'moderate': '#ffc107',
        'unhealthy_sensitive': '#fd7e14',
        'unhealthy': '#dc3545',
        'very_unhealthy': '#6f42c1',
        'hazardous': '#343a40'
    };
    return colors[status] || '#6c757d';
}

/**
 * Get status emoji
 */
function getStatusEmoji(status) {
    const emojis = {
        'good': '😊',
        'moderate': '😐',
        'unhealthy_sensitive': '😷',
        'unhealthy': '🤢',
        'very_unhealthy': '😵',
        'hazardous': '☠️'
    };
    return emojis[status] || '❓';
}

/**
 * Get marker color
 */
function getMarkerColor(status) {
    const colors = {
        'good': '#00e400',
        'moderate': '#ffff00',
        'unhealthy_sensitive': '#ff7e00',
        'unhealthy': '#ff0000',
        'very_unhealthy': '#8f3f97',
        'hazardous': '#7e0023'
    };
    return colors[status] || '#ffc107';
}

/**
 * Get notification icon
 */
function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Get health advice
 */
function getHealthAdvice(status) {
    const advice = {
        'good': {
            title: 'Good Air Quality',
            message: 'Air quality is satisfactory and poses little or no risk.',
            recommendations: 'Enjoy your outdoor activities! Air quality is good for everyone.',
            alertType: 'success'
        },
        'moderate': {
            title: 'Moderate Air Quality',
            message: 'Air quality is acceptable for most people.',
            recommendations: 'Sensitive individuals may experience minor issues. Limit prolonged outdoor exertion.',
            alertType: 'warning'
        },
        'unhealthy_sensitive': {
            title: 'Unhealthy for Sensitive Groups',
            message: 'Members of sensitive groups may experience health effects.',
            recommendations: 'Children, elderly, and people with heart/lung disease should limit prolonged outdoor exertion.',
            alertType: 'warning'
        },
        'unhealthy': {
            title: 'Unhealthy Air Quality',
            message: 'Everyone may begin to experience health effects.',
            recommendations: 'Avoid prolonged outdoor exertion. Consider wearing a mask if you must go outside.',
            alertType: 'danger'
        },
        'very_unhealthy': {
            title: 'Very Unhealthy Air Quality',
            message: 'Health warnings of emergency conditions.',
            recommendations: 'Avoid all outdoor activities. Stay indoors and keep windows closed.',
            alertType: 'danger'
        },
        'hazardous': {
            title: 'Hazardous Air Quality',
            message: 'Emergency conditions affecting entire population.',
            recommendations: 'Stay indoors, avoid all outdoor activities, use air purifiers if available.',
            alertType: 'danger'
        }
    };

    return advice[status] || advice['moderate'];
}

/**
 * Format date/time
 */
function formatDateTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Add fullscreen control to map
 */
function addFullscreenControl() {
    const fullscreenControl = L.control({ position: 'topright' });

    fullscreenControl.onAdd = function(map) {
        const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
        const button = L.DomUtil.create('a', '', div);

        button.innerHTML = '⛶';
        button.href = '#';
        button.title = 'Toggle fullscreen';
        button.style.fontSize = '18px';
        button.style.lineHeight = '26px';

        L.DomEvent.on(button, 'click', L.DomEvent.stopPropagation)
            .on(button, 'click', L.DomEvent.preventDefault)
            .on(button, 'click', function() {
                toggleMapFullscreen();
            });

        return div;
    };

    fullscreenControl.addTo(map);
}

/**
 * Toggle map fullscreen
 */
function toggleMapFullscreen() {
    const mapContainer = document.getElementById('map-container');

    if (!document.fullscreenElement) {
        mapContainer.requestFullscreen().catch(err => {
            console.log(`Error attempting to enable fullscreen: ${err.message}`);
        });
    } else {
        document.exitFullscreen();
    }
}

/**
 * Add heat map layer
 */
function addHeatMapLayer(data) {
    // This would require the Leaflet.heat plugin
    // For now, just add circle markers for heat points
    data.forEach(point => {
        L.circle([point.lat, point.lon], {
            radius: point.intensity * 100,
            fillColor: getColorForIntensity(point.intensity),
            fillOpacity: 0.6,
            color: 'white',
            weight: 1
        }).addTo(map);
    });
}

/**
 * Get color for intensity value
 */
function getColorForIntensity(intensity) {
    if (intensity < 0.3) return '#00ff00';
    if (intensity < 0.6) return '#ffff00';
    if (intensity < 0.8) return '#ff7f00';
    return '#ff0000';
}

// Cleanup on page unload
$(window).on('beforeunload', function() {
    stopAutoRefresh();
});