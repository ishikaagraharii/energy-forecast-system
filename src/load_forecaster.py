"""
Module 2: Load Forecasting (Electricity Demand)
Implements MLR (baseline), Random Forest/XGBoost, and LSTM models
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


class LoadForecaster:
    """Electricity load forecasting using multiple models"""
    
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
        self.feature_scaler = None
        self.target_scaler = None
        self.models = {}
        
        # Features for ML models
        self.feature_cols = [
            'temperature_c', 'humidity_percent', 'wind_speed_kmh',
            'solar_radiation_wm2', 'cloud_cover_percent',
            'hour', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'demand_mw_lag_1', 'demand_mw_lag_24', 'demand_mw_lag_168'
        ]
        self._load_saved_models()
    
    def _load_saved_models(self):
        """Load previously saved models"""
        # Load scalers
        fs_path = os.path.join(MODEL_DIR, 'load_feature_scaler.pkl')
        ts_path = os.path.join(MODEL_DIR, 'load_target_scaler.pkl')
        if os.path.exists(fs_path):
            self.feature_scaler = joblib.load(fs_path)
        if os.path.exists(ts_path):
            self.target_scaler = joblib.load(ts_path)
        
        # Load LSTM model
        lstm_path = os.path.join(MODEL_DIR, 'load_lstm.h5')
        if os.path.exists(lstm_path):
            try:
                self.models['lstm'] = load_model(lstm_path, compile=False)
            except:
                pass
        
        # Load ML models
        for name in ['mlr', 'rf', 'xgb']:
            path = os.path.join(MODEL_DIR, f'load_{name}.pkl')
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
    
    def prepare_features(self, df):
        """Prepare feature matrix and target"""
        available_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[available_cols].values
        y = df['demand_mw'].values if 'demand_mw' in df.columns else None
        return X, y, available_cols
    
    def train_mlr(self, X_train, y_train):
        """Train Multiple Linear Regression (baseline)"""
        print("  Training MLR baseline...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['mlr'] = model
        joblib.dump(model, os.path.join(MODEL_DIR, 'load_mlr.pkl'))
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """Train Random Forest model"""
        print("  Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['rf'] = model
        joblib.dump(model, os.path.join(MODEL_DIR, 'load_rf.pkl'))
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train Gradient Boosting (XGBoost-like) model"""
        print("  Training XGBoost...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['xgb'] = model
        joblib.dump(model, os.path.join(MODEL_DIR, 'load_xgb.pkl'))
        return model
    
    def prepare_lstm_sequences(self, X, y, seq_length=168):
        """Prepare sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length - self.forecast_horizon + 1):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length:i + seq_length + self.forecast_horizon])
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm(self, X_train, y_train, seq_length=72, epochs=10, batch_size=64):
        """Train LSTM model for load forecasting"""
        print("  Training LSTM...")
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_train)
        
        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_lstm_sequences(X_scaled, y_scaled, seq_length)
        
        # Split train/val
        split_idx = int(len(X_seq) * 0.9)
        X_tr, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_tr, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build model
        n_features = X_train.shape[1]
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_length, n_features)),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        self.models['lstm'] = model
        model.save(os.path.join(MODEL_DIR, 'load_lstm.h5'))
        
        # Save scalers
        joblib.dump(self.feature_scaler, os.path.join(MODEL_DIR, 'load_feature_scaler.pkl'))
        joblib.dump(self.target_scaler, os.path.join(MODEL_DIR, 'load_target_scaler.pkl'))
        
        return model, history
    
    def train_all(self, df_train):
        """Train all models"""
        print("\n" + "=" * 50)
        print("LOAD FORECASTING - MODEL TRAINING")
        print("=" * 50)
        
        X, y, feature_cols = self.prepare_features(df_train)
        print(f"\nTraining samples: {len(X)}")
        print(f"Features: {len(feature_cols)}")
        
        # Train baseline models
        self.train_mlr(X, y)
        self.train_random_forest(X, y)
        self.train_xgboost(X, y)
        
        # Train LSTM
        try:
            self.train_lstm(X[-5000:], y[-5000:], epochs=10)
        except Exception as e:
            print(f"  LSTM training failed: {e}")
        
        print("\n✓ Load forecasting models trained and saved")
    
    def predict_ml(self, X, model_name='rf'):
        """Predict using ML models"""
        model = self.models.get(model_name)
        if model is None:
            return None
        return model.predict(X)
    
    def predict_lstm(self, X_recent, seq_length=72):
        """Predict using LSTM"""
        model = self.models.get('lstm')
        if model is None or self.feature_scaler is None:
            return None
        
        # Scale and prepare input
        X_scaled = self.feature_scaler.transform(X_recent[-seq_length:])
        X_seq = X_scaled.reshape(1, seq_length, X_scaled.shape[1])
        
        # Predict
        pred_scaled = model.predict(X_seq, verbose=0)
        pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        
        return pred.flatten()
    
    def forecast(self, df_recent, model_type='lstm', weather_forecast=None):
        """Generate 24-hour load forecast"""
        print(f"\nGenerating {self.forecast_horizon}h load forecast using {model_type.upper()}...")
        
        X, _, _ = self.prepare_features(df_recent)
        
        if model_type == 'lstm':
            predictions = self.predict_lstm(X)
        else:
            # For ML models, predict next 24 hours iteratively
            predictions = self.predict_ml(X[-self.forecast_horizon:], model_type)
        
        if predictions is not None:
            print(f"  Load forecast: {predictions[0]:.0f} MW -> {predictions[-1]:.0f} MW")
        
        return predictions
    
    def evaluate(self, df_test, model_type='rf'):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_type.upper()} model...")
        
        X, y_actual, _ = self.prepare_features(df_test)
        
        if model_type == 'lstm':
            y_pred = self.predict_lstm(X)
            y_actual = y_actual[-self.forecast_horizon:]
        else:
            y_pred = self.predict_ml(X, model_type)
        
        if y_pred is None or len(y_pred) != len(y_actual):
            print("  Evaluation failed - length mismatch")
            return None
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        print(f"  MAE: {mae:.2f} MW")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return metrics


def run_load_forecasting(df_merged, train_models=True):
    """Main entry point for load forecasting module"""
    
    forecaster = LoadForecaster(forecast_horizon=24)
    
    if train_models:
        # Train on all but last week
        train_data = df_merged.iloc[:-168]
        forecaster.train_all(train_data)
    
    # Generate forecast
    recent_data = df_merged.tail(200)
    forecast = forecaster.forecast(recent_data, model_type='lstm')
    
    return forecaster, forecast


if __name__ == "__main__":
    # Test with processed data
    df = pd.read_csv("data/processed/merged_dataset.csv")
    forecaster, forecast = run_load_forecasting(df)
    print("\n24-hour Load Forecast (MW):")
    print(forecast)
