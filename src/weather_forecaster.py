"""
Module 1: Weather Forecasting
Implements ARIMA (baseline) and LSTM models for weather prediction
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class WeatherForecaster:
    """Weather forecasting using ARIMA and LSTM models"""
    
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
        self.scalers = {}
        self.models = {}
        self.weather_features = [
            'temperature_c', 
            'humidity_percent', 
            'wind_speed_kmh',
            'solar_radiation_wm2', 
            'cloud_cover_percent'
        ]
        self._load_saved_models()
    
    def _load_saved_models(self):
        """Load previously saved models"""
        scaler_path = os.path.join(MODEL_DIR, 'weather_scalers.pkl')
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        
        for feature in self.weather_features:
            model_path = os.path.join(MODEL_DIR, f'lstm_{feature}.h5')
            if os.path.exists(model_path):
                try:
                    self.models[f'lstm_{feature}'] = load_model(model_path, compile=False)
                except:
                    pass
    
    def prepare_sequences(self, data, seq_length=72):
        """Prepare sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - seq_length - self.forecast_horizon + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def train_arima(self, series, feature_name, order=(2, 1, 2)):
        """Train ARIMA model for a single weather feature"""
        print(f"  Training ARIMA for {feature_name}...")
        try:
            model = ARIMA(series, order=order)
            fitted = model.fit()
            self.models[f'arima_{feature_name}'] = fitted
            return fitted
        except Exception as e:
            print(f"  ARIMA failed for {feature_name}: {e}")
            return None
    
    def train_lstm(self, train_data, feature_name, seq_length=72, epochs=10, batch_size=64):
        """Train LSTM model for a single weather feature"""
        print(f"  Training LSTM for {feature_name}...")
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train_data.reshape(-1, 1))
        self.scalers[feature_name] = scaler
        
        X, y = self.prepare_sequences(scaled_data.flatten(), seq_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(seq_length, 1)),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
        
        self.models[f'lstm_{feature_name}'] = model
        model.save(os.path.join(MODEL_DIR, f'lstm_{feature_name}.h5'))
        return model
    
    def forecast_arima(self, feature_name, steps=24):
        """Generate forecast using ARIMA"""
        model = self.models.get(f'arima_{feature_name}')
        if model is None:
            return None
        forecast = model.forecast(steps=steps)
        return forecast.values
    
    def forecast_lstm(self, recent_data, feature_name, seq_length=72):
        """Generate forecast using LSTM"""
        model = self.models.get(f'lstm_{feature_name}')
        scaler = self.scalers.get(feature_name)
        
        if model is None or scaler is None:
            return None
        
        scaled = scaler.transform(recent_data[-seq_length:].reshape(-1, 1))
        X = scaled.reshape(1, seq_length, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        return pred.flatten()
    
    def train_all(self, df_weather, use_lstm=True):
        """Train models for all weather features"""
        print("\n" + "=" * 50)
        print("WEATHER FORECASTING - MODEL TRAINING")
        print("=" * 50)
        
        for feature in self.weather_features:
            print(f"\n[{feature}]")
            series = df_weather[feature].values
            self.train_arima(series[-2000:], feature)
            if use_lstm:
                try:
                    self.train_lstm(series[-5000:], feature, epochs=10)
                except Exception as e:
                    print(f"  LSTM training failed: {e}")
        
        joblib.dump(self.scalers, os.path.join(MODEL_DIR, 'weather_scalers.pkl'))
        print("\n✓ Weather models trained and saved")
    
    def forecast_all(self, df_weather, model_type='lstm'):
        """Generate 24-hour forecast for all weather features"""
        print(f"\nGenerating {self.forecast_horizon}h weather forecast using {model_type.upper()}...")
        
        forecasts = {}
        for feature in self.weather_features:
            if model_type == 'lstm':
                forecast = self.forecast_lstm(df_weather[feature].values, feature)
            else:
                forecast = self.forecast_arima(feature, steps=self.forecast_horizon)
            
            if forecast is not None:
                forecasts[feature] = forecast
                print(f"  {feature}: {forecast[0]:.2f} -> {forecast[-1]:.2f}")
        
        return pd.DataFrame(forecasts) if forecasts else pd.DataFrame()


def run_weather_forecasting(df_weather, train_models=True):
    """Main entry point for weather forecasting module"""
    forecaster = WeatherForecaster(forecast_horizon=24)
    
    if train_models:
        train_data = df_weather.iloc[:-168]
        forecaster.train_all(train_data, use_lstm=True)
    
    recent_data = df_weather.tail(200)
    forecast_df = forecaster.forecast_all(recent_data, model_type='lstm')
    return forecaster, forecast_df


if __name__ == "__main__":
    df = pd.read_csv("data/weather_india_central.csv")
    forecaster, forecast = run_weather_forecasting(df)
    print("\n24-hour Weather Forecast:")
    print(forecast)
