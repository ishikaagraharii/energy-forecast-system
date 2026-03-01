"""
Module 3: Renewable Energy Generation Forecasting
Forecasts solar and wind generation based on weather conditions
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class RenewableForecaster:
    """Renewable energy generation forecasting"""
    
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {}
        
        # Weather features that affect renewable generation
        self.solar_features = [
            'solar_radiation_wm2', 'cloud_cover_percent', 
            'temperature_c', 'hour', 'month'
        ]
        self.wind_features = [
            'wind_speed_kmh', 'temperature_c', 
            'humidity_percent', 'hour'
        ]
        self._load_saved_models()
    
    def _load_saved_models(self):
        """Load previously saved models"""
        # Load scalers
        scaler_path = os.path.join(MODEL_DIR, 'renewable_scalers.pkl')
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        
        # Load RF models
        for name in ['solar', 'wind']:
            rf_path = os.path.join(MODEL_DIR, f'renewable_rf_{name}.pkl')
            if os.path.exists(rf_path):
                self.models[f'rf_{name}'] = joblib.load(rf_path)
            
            lstm_path = os.path.join(MODEL_DIR, f'renewable_lstm_{name}.h5')
            if os.path.exists(lstm_path):
                try:
                    self.models[f'lstm_{name}'] = load_model(lstm_path, compile=False)
                except:
                    pass
    
    def estimate_solar_generation(self, df, capacity_mw=50000):
        """
        Estimate solar generation from weather data
        Uses simplified physics-based model when actual data unavailable
        
        Solar output = capacity * efficiency * (radiation / 1000) * (1 - cloud_factor)
        """
        radiation = df['solar_radiation_wm2'].values
        cloud = df['cloud_cover_percent'].values / 100
        temp = df['temperature_c'].values
        
        # Temperature derating (efficiency drops above 25°C)
        temp_factor = np.where(temp > 25, 1 - 0.004 * (temp - 25), 1.0)
        
        # Cloud impact
        cloud_factor = 0.7 * cloud  # Clouds reduce output
        
        # Normalized radiation (assuming 1000 W/m² is peak)
        radiation_factor = np.clip(radiation / 1000, 0, 1)
        
        # Estimated generation
        solar_gen = capacity_mw * 0.18 * radiation_factor * (1 - cloud_factor) * temp_factor
        
        return np.maximum(solar_gen, 0)
    
    def estimate_wind_generation(self, df, capacity_mw=45000):
        """
        Estimate wind generation from wind speed
        Uses power curve approximation
        
        Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s
        """
        # Convert km/h to m/s
        wind_ms = df['wind_speed_kmh'].values / 3.6
        
        # Power curve approximation
        cut_in, rated, cut_out = 3, 12, 25
        
        power_factor = np.zeros_like(wind_ms)
        
        # Below cut-in: 0
        mask_below = wind_ms < cut_in
        
        # Between cut-in and rated: cubic relationship
        mask_ramp = (wind_ms >= cut_in) & (wind_ms < rated)
        power_factor[mask_ramp] = ((wind_ms[mask_ramp] - cut_in) / (rated - cut_in)) ** 3
        
        # At rated: 100%
        mask_rated = (wind_ms >= rated) & (wind_ms < cut_out)
        power_factor[mask_rated] = 1.0
        
        # Above cut-out: 0 (safety shutdown)
        mask_cutout = wind_ms >= cut_out
        
        # Capacity factor typically around 25-35%
        wind_gen = capacity_mw * 0.30 * power_factor
        
        return np.maximum(wind_gen, 0)
    
    def train_rf_model(self, X, y, name):
        """Train Random Forest for renewable prediction"""
        print(f"  Training RF for {name}...")
        model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
        model.fit(X, y)
        self.models[f'rf_{name}'] = model
        joblib.dump(model, os.path.join(MODEL_DIR, f'renewable_rf_{name}.pkl'))
        return model
    
    def train_lstm(self, X, y, name, seq_length=48, epochs=10):
        """Train LSTM for renewable generation"""
        print(f"  Training LSTM for {name}...")
        
        # Scale
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.scalers[f'{name}_X'] = scaler_X
        self.scalers[f'{name}_y'] = scaler_y
        
        # Prepare sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_length - self.forecast_horizon + 1):
            X_seq.append(X_scaled[i:i + seq_length])
            y_seq.append(y_scaled[i + seq_length:i + seq_length + self.forecast_horizon])
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # Split
        split = int(len(X_seq) * 0.9)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]
        
        # Model
        model = Sequential([
            LSTM(32, input_shape=(seq_length, X.shape[1])),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        self.models[f'lstm_{name}'] = model
        model.save(os.path.join(MODEL_DIR, f'renewable_lstm_{name}.h5'))
        
        return model
    
    def train_all(self, df):
        """Train all renewable forecasting models"""
        print("\n" + "=" * 50)
        print("RENEWABLE FORECASTING - MODEL TRAINING")
        print("=" * 50)
        
        # Prepare features for solar
        solar_cols = [c for c in self.solar_features if c in df.columns]
        X_solar = df[solar_cols].values
        
        # Prepare features for wind
        wind_cols = [c for c in self.wind_features if c in df.columns]
        X_wind = df[wind_cols].values
        
        # Check if we have actual renewable data
        if 'solar_wind_mw' in df.columns:
            # Use actual data
            y_total = df['solar_wind_mw'].values
            
            # Approximate split (solar typically 55%, wind 45% during day)
            # This is a simplification - actual split would need separate data
            y_solar = y_total * 0.55
            y_wind = y_total * 0.45
            print("  Using actual renewable generation data")
        else:
            # Estimate from physics models
            y_solar = self.estimate_solar_generation(df)
            y_wind = self.estimate_wind_generation(df)
            print("  Using physics-based estimates")
        
        # Train RF models
        self.train_rf_model(X_solar, y_solar, 'solar')
        self.train_rf_model(X_wind, y_wind, 'wind')
        
        # Train LSTM models
        try:
            self.train_lstm(X_solar[-5000:], y_solar[-5000:], 'solar', epochs=10)
            self.train_lstm(X_wind[-5000:], y_wind[-5000:], 'wind', epochs=10)
        except Exception as e:
            print(f"  LSTM training failed: {e}")
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(MODEL_DIR, 'renewable_scalers.pkl'))
        
        print("\n✓ Renewable forecasting models trained and saved")
    
    def forecast_rf(self, X, name):
        """Forecast using RF model"""
        model = self.models.get(f'rf_{name}')
        if model is None:
            return None
        return model.predict(X)
    
    def forecast_lstm(self, X, name, seq_length=48):
        """Forecast using LSTM"""
        model = self.models.get(f'lstm_{name}')
        scaler_X = self.scalers.get(f'{name}_X')
        scaler_y = self.scalers.get(f'{name}_y')
        
        if model is None or scaler_X is None:
            return None
        
        X_scaled = scaler_X.transform(X[-seq_length:])
        X_seq = X_scaled.reshape(1, seq_length, X.shape[1])
        
        pred_scaled = model.predict(X_seq, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        
        return pred.flatten()
    
    def forecast(self, df_recent, weather_forecast=None, model_type='lstm'):
        """Generate 24-hour renewable generation forecast"""
        print(f"\nGenerating {self.forecast_horizon}h renewable forecast using {model_type.upper()}...")
        
        # Use weather forecast if provided, else use recent data
        if weather_forecast is not None and len(weather_forecast) > 0:
            df_forecast = weather_forecast.copy()
            # Add hour/month features
            base_hour = df_recent['hour'].iloc[-1] if 'hour' in df_recent.columns else 0
            df_forecast['hour'] = [(base_hour + i) % 24 for i in range(len(df_forecast))]
            df_forecast['month'] = df_recent['month'].iloc[-1] if 'month' in df_recent.columns else 6
        else:
            df_forecast = df_recent.tail(self.forecast_horizon)
        
        # Prepare features
        solar_cols = [c for c in self.solar_features if c in df_forecast.columns]
        wind_cols = [c for c in self.wind_features if c in df_forecast.columns]
        
        X_solar = df_forecast[solar_cols].values if solar_cols else df_recent[self.solar_features[:3]].tail(72).values
        X_wind = df_forecast[wind_cols].values if wind_cols else df_recent[self.wind_features[:3]].tail(72).values
        
        # Forecast
        if model_type == 'lstm':
            solar_pred = self.forecast_lstm(df_recent[[c for c in self.solar_features if c in df_recent.columns]].values, 'solar')
            wind_pred = self.forecast_lstm(df_recent[[c for c in self.wind_features if c in df_recent.columns]].values, 'wind')
        else:
            solar_pred = self.forecast_rf(X_solar, 'solar')
            wind_pred = self.forecast_rf(X_wind, 'wind')
        
        results = {
            'solar_mw': solar_pred if solar_pred is not None else np.zeros(self.forecast_horizon),
            'wind_mw': wind_pred if wind_pred is not None else np.zeros(self.forecast_horizon)
        }
        
        results['total_renewable_mw'] = results['solar_mw'] + results['wind_mw']
        
        print(f"  Solar: {results['solar_mw'][0]:.0f} -> {results['solar_mw'][-1]:.0f} MW")
        print(f"  Wind: {results['wind_mw'][0]:.0f} -> {results['wind_mw'][-1]:.0f} MW")
        print(f"  Total: {results['total_renewable_mw'][0]:.0f} -> {results['total_renewable_mw'][-1]:.0f} MW")
        
        return pd.DataFrame(results)


def run_renewable_forecasting(df_merged, train_models=True):
    """Main entry point for renewable forecasting module"""
    
    forecaster = RenewableForecaster(forecast_horizon=24)
    
    if train_models:
        train_data = df_merged.iloc[:-168]
        forecaster.train_all(train_data)
    
    # Generate forecast
    recent_data = df_merged.tail(200)
    forecast_df = forecaster.forecast(recent_data, model_type='lstm')
    
    return forecaster, forecast_df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/merged_dataset.csv")
    forecaster, forecast = run_renewable_forecasting(df)
    print("\n24-hour Renewable Forecast:")
    print(forecast)
