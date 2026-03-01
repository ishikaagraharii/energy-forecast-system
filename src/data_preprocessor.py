"""
Data Preprocessing Module
Consolidates Excel files into unified datasets for the forecasting pipeline
"""

import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime

DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "Electricity Demand, Solar and Wind Generation Data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_and_process_excel_files():
    """Load all Excel files and consolidate into unified datasets"""
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    excel_files = glob(os.path.join(RAW_DATA_DIR, "*.xlsx"))
    print(f"Found {len(excel_files)} Excel files to process")
    
    demand_data = []
    renewable_data = []
    
    for file_path in sorted(excel_files):
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")
        
        try:
            # Sheet1: Demand data
            df_demand = pd.read_excel(file_path, sheet_name='Sheet1')
            df_demand = df_demand.iloc[1:]  # Skip header row with labels
            df_demand.columns = ['datetime'] + [f'col_{i}' for i in range(1, len(df_demand.columns))]
            
            # First numeric column is typically total demand
            df_demand['datetime'] = pd.to_datetime(df_demand['datetime'], errors='coerce')
            df_demand = df_demand.dropna(subset=['datetime'])
            
            # Extract demand (first numeric column after time)
            demand_col = df_demand.columns[1]
            df_demand['demand_mw'] = pd.to_numeric(df_demand[demand_col], errors='coerce')
            
            demand_data.append(df_demand[['datetime', 'demand_mw']])
            
            # Sheet2: Renewable (Solar + Wind) data
            df_renew = pd.read_excel(file_path, sheet_name='Sheet2')
            df_renew = df_renew.iloc[1:]  # Skip header row
            df_renew.columns = ['datetime', 'solar_wind_mw', 'hourly_value']
            df_renew['datetime'] = pd.to_datetime(df_renew['datetime'], errors='coerce')
            df_renew = df_renew.dropna(subset=['datetime'])
            df_renew['solar_wind_mw'] = pd.to_numeric(df_renew['solar_wind_mw'], errors='coerce')
            
            renewable_data.append(df_renew[['datetime', 'solar_wind_mw']])
            
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            continue
    
    # Concatenate all data
    print("\nConsolidating demand data...")
    df_demand_all = pd.concat(demand_data, ignore_index=True)
    df_demand_all = df_demand_all.sort_values('datetime').reset_index(drop=True)
    
    print("Consolidating renewable data...")
    df_renew_all = pd.concat(renewable_data, ignore_index=True)
    df_renew_all = df_renew_all.sort_values('datetime').reset_index(drop=True)
    
    return df_demand_all, df_renew_all


def resample_to_hourly(df, value_col, agg_func='mean'):
    """Resample data to hourly frequency"""
    df = df.copy()
    df = df.set_index('datetime')
    df_hourly = df[[value_col]].resample('H').agg(agg_func)
    df_hourly = df_hourly.reset_index()
    return df_hourly


def merge_with_weather(df_demand, df_renewable, df_weather):
    """Merge load/renewable data with weather data"""
    
    # Ensure datetime columns are timezone-naive for merging
    df_demand['datetime'] = pd.to_datetime(df_demand['datetime']).dt.tz_localize(None)
    df_renewable['datetime'] = pd.to_datetime(df_renewable['datetime']).dt.tz_localize(None)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime']).dt.tz_localize(None)
    
    # Merge demand with weather
    df_merged = pd.merge(df_demand, df_weather, on='datetime', how='inner')
    
    # Merge with renewable
    df_merged = pd.merge(df_merged, df_renewable, on='datetime', how='left')
    
    return df_merged


def add_calendar_features(df):
    """Add calendar-based features for load forecasting"""
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def create_lag_features(df, target_col, lags=[1, 2, 3, 24, 48, 168]):
    """Create lag features for time series modeling"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def preprocess_all_data():
    """Main preprocessing pipeline"""
    
    print("=" * 70)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load and process Excel files
    print("\n[Step 1] Loading Excel files...")
    df_demand, df_renewable = load_and_process_excel_files()
    
    print(f"\nRaw demand records: {len(df_demand)}")
    print(f"Raw renewable records: {len(df_renewable)}")
    
    # Step 2: Resample to hourly
    print("\n[Step 2] Resampling to hourly frequency...")
    df_demand_hourly = resample_to_hourly(df_demand, 'demand_mw')
    df_renewable_hourly = resample_to_hourly(df_renewable, 'solar_wind_mw')
    
    print(f"Hourly demand records: {len(df_demand_hourly)}")
    print(f"Hourly renewable records: {len(df_renewable_hourly)}")
    
    # Step 3: Load weather data
    print("\n[Step 3] Loading weather data...")
    weather_path = os.path.join(DATA_DIR, "weather_india_central.csv")
    df_weather = pd.read_csv(weather_path)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    print(f"Weather records: {len(df_weather)}")
    
    # Step 4: Merge datasets
    print("\n[Step 4] Merging datasets...")
    df_merged = merge_with_weather(df_demand_hourly, df_renewable_hourly, df_weather)
    print(f"Merged records: {len(df_merged)}")
    
    # Step 5: Add features
    print("\n[Step 5] Adding calendar and lag features...")
    df_merged = add_calendar_features(df_merged)
    df_merged = create_lag_features(df_merged, 'demand_mw')
    df_merged = create_lag_features(df_merged, 'solar_wind_mw', lags=[1, 24])
    
    # Step 6: Handle missing values
    print("\n[Step 6] Handling missing values...")
    initial_len = len(df_merged)
    df_merged = df_merged.dropna()
    print(f"Records after dropping NaN: {len(df_merged)} (dropped {initial_len - len(df_merged)})")
    
    # Step 7: Save processed data
    print("\n[Step 7] Saving processed data...")
    
    # Save individual datasets
    df_demand_hourly.to_csv(os.path.join(PROCESSED_DIR, "demand_hourly.csv"), index=False)
    df_renewable_hourly.to_csv(os.path.join(PROCESSED_DIR, "renewable_hourly.csv"), index=False)
    
    # Save merged dataset
    output_path = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    df_merged.to_csv(output_path, index=False)
    
    print(f"\nSaved files:")
    print(f"  - {PROCESSED_DIR}/demand_hourly.csv")
    print(f"  - {PROCESSED_DIR}/renewable_hourly.csv")
    print(f"  - {PROCESSED_DIR}/merged_dataset.csv")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nFinal dataset shape: {df_merged.shape}")
    print(f"Date range: {df_merged['datetime'].min()} to {df_merged['datetime'].max()}")
    print(f"\nColumns: {df_merged.columns.tolist()}")
    
    return df_merged


if __name__ == "__main__":
    df = preprocess_all_data()
    print("\nFirst few rows:")
    print(df.head())
