# Energy Forecasting System - Technical Explanation

## Project Structure

```
h:\Energy\
├── data/                           → Raw and processed datasets
├── src/                            → Core modules
│   ├── data_preprocessor.py        → Consolidates Excel files into unified CSV
│   ├── weather_forecaster.py       → Predicts weather variables 24h ahead
│   ├── load_forecaster.py          → Predicts electricity demand 24h ahead
│   ├── renewable_forecaster.py     → Predicts solar/wind generation 24h ahead
│   └── visualizations.py           → Generates all plots
├── models/                         → Trained model files (.h5, .pkl)
├── output/                         → Forecast results and visualizations
├── pipeline.py                     → Main orchestrator
└── requirements.txt                → Python dependencies
```

---

## What Each File Does

### data_preprocessor.py
- Reads 29 monthly Excel files from POSOCO (India's grid operator)
- Extracts electricity demand from Sheet1
- Extracts solar+wind generation from Sheet2
- Resamples 5-minute data to hourly frequency
- Merges with weather data (temperature, humidity, wind, solar radiation, cloud cover)
- Adds calendar features: hour, day_of_week, month, is_weekend
- Creates lag features: demand at t-1, t-24, t-168 (previous hour, day, week)
- Outputs: `data/processed/merged_dataset.csv`

### weather_forecaster.py
- **Purpose**: Forecast weather 24 hours ahead
- **Models**:
  - ARIMA: Traditional time-series baseline (captures trends and seasonality)
  - LSTM: Deep learning model that learns temporal patterns from sequences
- **Inputs**: Historical weather data (72 hours lookback)
- **Outputs**: 24 hourly predictions for each variable
- **Variables forecasted**: temperature_c, humidity_percent, wind_speed_kmh, solar_radiation_wm2, cloud_cover_percent

### load_forecaster.py
- **Purpose**: Forecast electricity demand 24 hours ahead
- **Models**:
  - MLR (Multiple Linear Regression): Simple baseline showing linear relationships
  - Random Forest: Ensemble of decision trees, captures non-linear patterns
  - XGBoost: Gradient boosted trees, often best for tabular data
  - LSTM: Captures sequential dependencies in load patterns
- **Inputs**: 
  - Weather features (temperature affects cooling/heating load)
  - Calendar features (weekday vs weekend patterns differ)
  - Lag features (demand follows daily/weekly cycles)
- **Output**: 24 hourly demand values in MW

### renewable_forecaster.py
- **Purpose**: Forecast solar and wind generation 24 hours ahead
- **Solar model logic**:
  - Generation = f(solar_radiation, cloud_cover, temperature)
  - Temperature above 25°C reduces panel efficiency by 0.4%/°C
  - Clouds reduce output proportionally
- **Wind model logic**:
  - Uses wind power curve: cut-in at 3 m/s, rated at 12 m/s, cut-out at 25 m/s
  - Power proportional to cube of wind speed in ramping region
- **Models**: Random Forest + LSTM trained on actual generation data
- **Output**: Hourly solar_mw, wind_mw, total_renewable_mw

### visualizations.py
- Generates 4 output files with consistent styling
- All plots show 24-hour forecast horizon
- Includes annotations for peak/minimum values

### pipeline.py
- Orchestrates the multi-stage flow: Data → Weather → Load → Renewable → Visualization
- Supports two modes:
  - `python pipeline.py` → Full training + forecast (~80 seconds)
  - `python pipeline.py --no-train` → Load saved models + forecast (~3 seconds)

---

## Output Files Explanation

### weather_forecast.png
Contains 6 subplots:

| Plot | What it shows |
|------|---------------|
| Temperature | Predicted air temperature (°C) over 24 hours. Higher temps → more AC load |
| Solar Radiation | W/m² hitting ground. Zero at night, peaks midday. Drives solar generation |
| Wind Speed | km/h wind forecast. Drives wind turbine output |
| Humidity | Relative humidity %. Affects perceived temperature and minor load impact |
| Cloud Cover | % of sky covered. High clouds = low solar output |
| Summary | Min/max/average statistics for each variable |

### load_forecast.png
Single plot showing:
- **Blue line**: Predicted electricity demand (MW) for each hour
- **Shaded band**: ±5% confidence interval
- **Peak annotation**: Highest demand hour and value (typically evening)
- **Min annotation**: Lowest demand hour and value (typically early morning)
- **X-axis**: Hours (00:00 to 23:00)
- **Y-axis**: Demand in Megawatts

Typical pattern: Low at night (3-5 AM), rises morning (6-9 AM), peaks evening (6-9 PM).

### renewable_forecast.png
Two plots:

**Top plot (Stacked Area)**:
- Yellow area: Solar generation (MW)
- Blue area: Wind generation (MW)
- Black dashed line: Total renewable output
- Shows how solar and wind complement each other (solar peaks midday, wind varies)

**Bottom plot (Line comparison)**:
- Separate lines for solar and wind
- Solar follows bell curve (sunrise to sunset)
- Wind more variable, depends on weather patterns
- Annotations show peak generation times

### dashboard.png
Combined view with 6 panels:

| Panel | Content |
|-------|---------|
| Top-left | Temperature forecast curve |
| Top-center | Solar radiation forecast |
| Top-right | Wind speed forecast |
| Middle-left | Full load forecast with peak/min labels |
| Middle-right | Summary statistics box showing peak load, renewable share |
| Bottom | Stacked renewable generation (solar + wind) |

### forecast_report.txt
Text summary containing:

```
WEATHER FORECAST:
  temperature_c: min - max (avg)      → Range of predicted temperatures
  humidity_percent: min - max (avg)   → Humidity range
  wind_speed_kmh: min - max (avg)     → Wind speed range
  solar_radiation_wm2: min - max      → Solar intensity range
  cloud_cover_percent: min - max      → Cloud coverage range

LOAD FORECAST:
  Peak Load: XXX MW                   → Maximum demand in forecast period
  Min Load: XXX MW                    → Minimum demand (typically night)
  Average: XXX MW                     → Mean demand across 24 hours
  Total Energy: XXX MWh               → Sum of hourly demand (energy consumed)

RENEWABLE FORECAST:
  Solar Peak: XXX MW                  → Maximum solar generation
  Wind Avg: XXX MW                    → Average wind output
  Total RE Generation: XXX MWh        → Total renewable energy produced

  Renewable Share: XX.X% of load      → (Total RE / Total Load) × 100
```

---

## Key Metrics Interpretation

| Metric | Meaning | Why it matters |
|--------|---------|----------------|
| Peak Load | Highest hourly demand | Grid must have capacity to meet this |
| Renewable Share | % of demand met by solar+wind | Higher = cleaner grid |
| Solar Peak | Max solar output | Occurs around noon on clear days |
| Total Energy (MWh) | Energy consumed over 24h | Used for planning and billing |

---

## Model Flow

```
Historical Weather Data
        ↓
[Weather Forecaster] → 24h weather predictions
        ↓
        ├──────────────────────────────────┐
        ↓                                  ↓
[Load Forecaster]                [Renewable Forecaster]
        ↓                                  ↓
24h Demand Forecast              24h Solar + Wind Forecast
        ↓                                  ↓
        └──────────────┬───────────────────┘
                       ↓
              [Visualizations]
                       ↓
            PNG charts + TXT report
```

---

## Running the System

```bash
# First time (trains models, ~80 seconds)
python pipeline.py

# Subsequent runs (uses saved models, ~3 seconds)
python pipeline.py --no-train

# View outputs
output/weather_forecast.png
output/load_forecast.png
output/renewable_forecast.png
output/dashboard.png
output/forecast_report.txt
```
