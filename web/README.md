# SCEF Web Interface

This web interface provides a user-friendly way to interact with the Strategy Composition and Evaluation Framework (SCEF). It allows users to create, backtest, and deploy trading strategies through a browser-based interface.

## Features

- **Strategy Creation**: Build trading strategies by composing modular components
- **Market Data Management**: Upload or generate sample market data
- **Backtesting**: Test strategies against historical market data with visualization
- **Deployment**: Deploy strategies to the production engine with monitoring

## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web Interface**

   ```bash
   cd scef
   ./start.sh
   ```

   Or manually start the FastAPI application:

   ```bash
   cd scef/web
   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the Interface**

   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Usage Guide

### Creating Strategies

1. Go to the "Strategies" page
2. Click "Create New Strategy"
3. Add components from the left panel to build your strategy
4. Configure each component's parameters
5. Save the strategy

### Managing Market Data

1. Go to the "Market Data" page
2. Upload a CSV or Parquet file with OHLCV data, or
3. Create sample data with the "Create Sample Data" button

### Running Backtests

1. Go to the "Backtest" page
2. Select a strategy and market data
3. Configure backtest settings
4. Run the backtest
5. View the results with metrics and visualizations

### Deploying Strategies

1. Go to the "Deployment" page
2. Select a strategy and market data
3. Configure deployment settings
4. Deploy the strategy
5. Monitor performance from the deployment page

## System Requirements

- Python 3.8+
- Modern web browser
- Recommended: 4GB+ RAM, 2+ CPU cores

## Extending the Interface

The web interface is built with FastAPI and can be extended by:

1. Adding new API endpoints in `app.py`
2. Creating new HTML templates in `templates/`
3. Adding custom CSS styles in `static/css/styles.css`
4. Extending JavaScript functionality in `static/js/common.js`

## Troubleshooting

- **Interface not loading**: Check if the server is running and accessible
- **API errors**: Check console logs for error messages
- **Visualization issues**: Ensure Plotly is properly installed
- **File upload problems**: Verify file format and required columns

For more detailed information, refer to the main SCEF documentation.
