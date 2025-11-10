/**
 * Common JavaScript functions for SCEF Web Interface
 */

// Format number as percentage
function formatPercent(value, decimals = 2) {
    return (value * 100).toFixed(decimals) + '%';
}

// Format number as currency
function formatCurrency(value, currency = 'USD', decimals = 2) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Show toast notification
function showToast(message, type = 'success') {
    // Check if toast container exists
    let toastContainer = document.getElementById('toast-container');
    
    // Create toast container if it doesn't exist
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1050';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    // Toast content
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Initialize and show toast
    const bsToast = new bootstrap.Toast(toast, {
        delay: 5000,
        autohide: true
    });
    
    bsToast.show();
    
    // Remove toast from DOM after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

// Format status indicator
function getStatusIndicator(status) {
    const statusMap = {
        'running': { color: 'status-running', label: 'Running' },
        'paused': { color: 'status-paused', label: 'Paused' },
        'stopped': { color: 'status-stopped', label: 'Stopped' },
        'error': { color: 'status-error', label: 'Error' },
        'initializing': { color: 'status-paused', label: 'Initializing' }
    };
    
    const statusInfo = statusMap[status] || { color: '', label: status };
    
    return `
        <div>
            <span class="status-indicator ${statusInfo.color}"></span>
            <span>${statusInfo.label}</span>
        </div>
    `;
}

// Load data from API
async function fetchApi(url, options = {}) {
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API error:', error);
        showToast(error.message, 'danger');
        throw error;
    }
}

// Create metrics card
function createMetricCard(label, value, suffix = '', prefix = '') {
    return `
        <div class="col-md-3 col-sm-6 mb-4">
            <div class="metric-card">
                <div class="metric-value">${prefix}${value}${suffix}</div>
                <div class="metric-label">${label}</div>
            </div>
        </div>
    `;
}

// Format backtest metrics
function formatBacktestMetrics(metrics) {
    let html = '<div class="row">';
    
    // Total return
    html += createMetricCard(
        'Total Return',
        (metrics.total_return * 100).toFixed(2),
        '%'
    );
    
    // Annual return
    html += createMetricCard(
        'Annual Return',
        (metrics.annual_return * 100).toFixed(2),
        '%'
    );
    
    // Sharpe ratio
    html += createMetricCard(
        'Sharpe Ratio',
        metrics.sharpe_ratio.toFixed(2)
    );
    
    // Max drawdown
    html += createMetricCard(
        'Max Drawdown',
        (metrics.max_drawdown * 100).toFixed(2),
        '%',
        '-'
    );
    
    html += '</div><div class="row">';
    
    // Volatility
    html += createMetricCard(
        'Volatility',
        (metrics.volatility * 100).toFixed(2),
        '%'
    );
    
    // Win rate
    html += createMetricCard(
        'Win Rate',
        (metrics.win_rate * 100).toFixed(2),
        '%'
    );
    
    // Profit factor
    html += createMetricCard(
        'Profit Factor',
        metrics.profit_factor.toFixed(2)
    );
    
    // Number of trades
    html += createMetricCard(
        'Trades',
        metrics.num_trades
    );
    
    html += '</div>';
    
    return html;
}

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});
