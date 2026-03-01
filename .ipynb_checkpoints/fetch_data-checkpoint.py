"""
Data Fetching Script for Energy Forecasting Project
Fetches weather data and provides instructions for load/generation data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

DATA_DIR = "data"

def fetch_weather_data():
    """
    Fetch historical weather data from Open-Meteo API (free, no API key needed)
    Location: Delhi, India (representative for North India grid)
    """
    print("Fetching weather data from Open-Meteo API...")
    
    # Open-Meteo Historical Weather API
    # Coordinates for major Indian cities (we'll use Delhi as primary)
    locations = {
        "delhi": {"lat": 28.6139, "lon": 77.2090},
        "mumbai": {"lat": 19.0760, "lon": 72.8777},
        "chennai": {"lat": 13.0827, "lon": 80.2707},
        "kolkata": {"lat": 22.5726, "lon": 88.3639},
        "bangalore": {"lat": 12.9716, "lon": 77.5946}
    }
    
    # Fetch 2 years of data (matching Mendeley dataset period)
    start_date = "2021-09-01"
    end_date = "2023-12-31"
    
    all_weather_data = []
    
    for city, coords in locations.items():
        print(f"  Fetching data for {city}...")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m", 
                "wind_speed_10m",
                "shortwave_radiation",  # Solar radiation
                "cloud_cover"
            ],
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            hourly = data["hourly"]
            
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "city": city,
                "temperature_c": hourly["temperature_2m"],
                "humidity_percent": hourly["relative_humidity_2m"],
                "wind_speed_kmh": hourly["wind_speed_10m"],
                "solar_radiation_wm2": hourly["shortwave_radiation"],
                "cloud_cover_percent": hourly["cloud_cover"]
            })
            all_weather_data.append(df)
            print(f"    ✓ {len(df)} hourly records fetched")
        else:
            print(f"    ✗ Failed to fetch data for {city}: {response.status_code}")
    
    if all_weather_data:
        weather_df = pd.concat(all_weather_data, ignore_index=True)
        output_path = os.path.join(DATA_DIR, "weather_data.csv")
        weather_df.to_csv(output_path, index=False)
        print(f"\n✓ Weather data saved to {output_path}")
        print(f"  Total records: {len(weather_df)}")
        print(f"  Date range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
        return weather_df
    return None


def fetch_india_avg_weather():
    """
    Fetch aggregated weather data (India average) for simpler modeling
    """
    print("\nFetching India-average weather data...")
    
    # Central India coordinates (approximate centroid)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 22.5,  # Central India
        "longitude": 78.5,
        "start_date": "2021-09-01",
        "end_date": "2023-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation",
            "cloud_cover"
        ],
        "timezone": "Asia/Kolkata"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly = data["hourly"]
        
        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "temperature_c": hourly["temperature_2m"],
            "humidity_percent": hourly["relative_humidity_2m"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
            "solar_radiation_wm2": hourly["shortwave_radiation"],
            "cloud_cover_percent": hourly["cloud_cover"]
        })
        
        output_path = os.path.join(DATA_DIR, "weather_india_central.csv")
        df.to_csv(output_path, index=False)
        print(f"✓ India central weather data saved to {output_path}")
        print(f"  Records: {len(df)}")
        return df
    else:
        print(f"✗ Failed: {response.status_code}")
        return None


def create_sample_load_data():
    """
    Create sample load data structure matching Mendeley dataset format.
    User should replace with actual Mendeley data after manual download.
    """
    print("\nCreating sample load data template...")
    
    # Generate datetime index matching weather data period
    date_range = pd.date_range(
        start="2021-09-01", 
        end="2023-12-31 23:00:00", 
        freq="H",
        tz="Asia/Kolkata"
    )
    
    # Create template with expected columns from Mendeley dataset
    df = pd.DataFrame({
        "datetime": date_range,
        "demand_mw": None,  # To be filled from actual data
        "solar_generation_mw": None,
        "wind_generation_mw": None,
        "region": "ALL_INDIA"
    })
    
    output_path = os.path.join(DATA_DIR, "load_generation_template.csv")
    df.head(100).to_csv(output_path, index=False)  # Save template
    print(f"✓ Template saved to {output_path}")
    
    return df


def print_download_instructions():
    """Print instructions for manual dataset downloads"""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MANUAL DOWNLOAD REQUIRED                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 1: Download Electricity Load & Generation Data                        ║
║  ─────────────────────────────────────────────────────                       ║
║  URL: https://data.mendeley.com/datasets/y58jknpgs8                          ║
║                                                                              ║
║  1. Click "Download All" button on the page                                  ║
║  2. Extract the ZIP file                                                     ║
║  3. Move CSV files to: data/                                                 ║
║                                                                              ║
║  Expected files:                                                             ║
║  - Electricity demand (hourly, MW)                                           ║
║  - Solar generation (hourly, MW)                                             ║
║  - Wind generation (hourly, MW)                                              ║
║                                                                              ║
║  Period: September 2021 - December 2023 (or later)                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


if __name__ == "__main__":
    print("=" * 70)
    print("ENERGY FORECASTING - DATA FETCHER")
    print("=" * 70)
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Fetch weather data (automated)
    fetch_weather_data()
    fetch_india_avg_weather()
    
    # Create template for load data
    create_sample_load_data()
    
    # Print manual download instructions
    print_download_instructions()
    
    print("\n" + "=" * 70)
    print("Data fetching complete!")
    print("=" * 70)
