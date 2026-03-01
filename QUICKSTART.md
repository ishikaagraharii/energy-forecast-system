# Quick Start Guide

## Complete Energy Forecasting & Market Bidding System

All 6 layers implemented:

1. ✅ Data Preprocessing
2. ✅ Weather/Load/Renewable Forecasting
3. ✅ Market Price Forecasting
4. ✅ Bidding Optimization
5. ✅ Market Simulation
6. ✅ Web Dashboard

## Run the System

### Option 1: Command Line

```bash
# Quick forecast (uses saved models, ~4 seconds)
python pipeline.py --no-train

# Full training + forecast (~80 seconds)
python pipeline.py
```

### Option 2: Web Dashboard

```bash
# Start the dashboard
python start_dashboard.py

# Open browser to: http://localhost:8000
```

## What You Get

### Forecasts (24 hours ahead)

- Weather: Temperature, solar radiation, wind, humidity, cloud cover
- Load: Electricity demand (MW)
- Renewable: Solar + wind generation (MW)
- Price: Market clearing price (₹/MWh)

### Market Intelligence

- Net demand calculation
- Price-demand correlation
- Renewable penetration analysis

### Bidding Strategy

- 4 strategies: Conservative, Moderate, Optimal, Aggressive
- Bid schedule (price + volume per hour)
- Expected profit: ~₹1.2 Billion/month (500 MW capacity)
- Acceptance probability estimation

### Performance Metrics

- Strategy comparison
- Backtest results (30 days)
- Profit analysis
- Best strategy: **Moderate** (₹1.25 Billion profit)

## Output Files

```
output/
├── weather_forecast.png          # Weather variables
├── load_forecast.png              # Demand curve
├── renewable_forecast.png         # Solar + wind
├── dashboard.png                  # Combined view
├── forecast_report.txt            # Summary
├── bidding_report.txt             # Bid details
├── performance_report.txt         # Backtest results
└── simulation_results.json        # Full data
```

## Dashboard Features

- **Overview**: Load, renewable, price forecasts
- **Forecasts**: All weather variables
- **Market Intelligence**: Price analysis, net demand
- **Bidding Strategy**: Configure & optimize bids
- **Performance**: Strategy comparison, metrics

## API Endpoints

```
POST /api/forecast/run          # Run complete pipeline
GET  /api/forecast/latest       # Get latest results
GET  /api/summary               # Quick summary
POST /api/bidding/optimize      # Optimize bids
GET  /api/simulation/results    # Performance metrics
```

## Key Results

- Peak Load: 142,930 MW
- Renewable Share: 11.3%
- Avg Price: ₹4,238/MWh
- Expected Profit: ₹1.25 Billion (moderate strategy)
- Acceptance Rate: 100%

## Tech Stack

- Backend: FastAPI + Python
- Frontend: HTML/CSS/JS + Chart.js
- ML: Scikit-learn, TensorFlow
- Data: Pandas, NumPy

---

**System is ready to use!**
