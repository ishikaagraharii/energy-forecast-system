"""
Market Price Forecasting Module
Predicts Day-Ahead Market Clearing Price (MCP) based on net demand
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models disabled.")


class PriceForecaster:
    """
    Forecasts electricity market clearing prices (MCP)
    
    Models:
    - Random Forest: Baseline ensemble model
    - Gradient Boosting: Advanced tree-based model
    - LSTM: Deep learning for temporal patterns
    
    Features:
    - Net demand (load - renewable)
    - Hour of day, day of week
    - Historical prices (lags)
    - Weather features
    """
    
    def __init__(self, forecast_horizon=24, model_dir="models"):
        self.forecast_horizon = forecast_horizon
        self.model_dir = model_dir
        
        # Models
        self.rf_model = None
        self.gb_model = None
        self.lstm_model = None
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Feature columns
        self.price_features = [
            'net_demand_mw', 'hour', 'day_of_week', 'is_weekend',
            'temperature_c', 'demand_mw', 'solar_wind_mw',
            'hour_sin', 'hour_cos'
        ]
        
        self._load_saved_models()
    
    def _load_saved_models(self):
        """Load pre-trained models if available"""
        try:
            rf_path = os.path.join(self.model_dir, "price_rf.pkl")
            gb_path = os.path.join(self.model_dir, "price_gb.pkl")
            lstm_path = os.path.join(self.model_dir, "price_lstm.h5")
            scaler_path = os.path.join(self.model_dir, "price_scalers.pkl")
            
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                print("  Loaded saved RF price model")
            
            if os.path.exists(gb_path):
                self.gb_model = joblib.load(gb_path)
                print("  Loaded saved GB price model")
            
            # Load scalers FIRST before LSTM
            if os.path.exists(scaler_path):
                scalers = joblib.load(scaler_path)
                self.feature_scaler = scalers['feature']
                self.target_scaler = scalers['target']
                print("  Loaded saved price scalers")
            else:
                print("  Price scalers not found, will need training")
            
            if KERAS_AVAILABLE and os.path.exists(lstm_path):
                try:
                    self.lstm_model = load_model(lstm_path)
                    print("  Loaded saved LSTM price model")
                except Exception as e:
                    print(f"  Could not load LSTM model: {e}")
        
        except Exception as e:
            print(f"  Error loading models: {e}")
    
    def prepare_features(self, df):
        """Prepare features for price forecasting"""
        df = df.copy()
        
        # Calculate net demand
        if 'net_demand_mw' not in df.columns:
            df['net_demand_mw'] = df['demand_mw'] - df['solar_wind_mw']
        
        # Ensure required features exist
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Select features
        available_features = [f for f in self.price_features if f in df.columns]
        X = df[available_features].values
        
        return X, available_features
    
    def generate_synthetic_prices(self, df):
        """
        Generate synthetic price data based on net demand
        Uses realistic price-demand relationship for Indian DAM
        """
        df = df.copy()
        
        # Calculate net demand
        df['net_demand_mw'] = df['demand_mw'] - df['solar_wind_mw']
        
        # Base price model: Price increases with net demand
        # Indian DAM typical range: ₹2000-5000/MWh
        net_demand_normalized = (df['net_demand_mw'] - df['net_demand_mw'].min()) / \
                                (df['net_demand_mw'].max() - df['net_demand_mw'].min())
        
        base_price = 2000 + 3000 * net_demand_normalized
        
        # Hour-of-day effect (peak hours more expensive)
        hour_effect = np.where(
            (df['hour'] >= 18) & (df['hour'] <= 22), 1.2,  # Evening peak
            np.where((df['hour'] >= 6) & (df['hour'] <= 10), 1.1,  # Morning peak
                     np.where((df['hour'] >= 0) & (df['hour'] <= 5), 0.85, 1.0))  # Night valley
        )
        
        # Weekend effect (lower prices)
        weekend_effect = np.where(df['is_weekend'] == 1, 0.9, 1.0)
        
        # Renewable effect (high renewable = lower prices)
        renewable_share = df['solar_wind_mw'] / df['demand_mw']
        renewable_effect = 1.0 - 0.3 * renewable_share  # Up to 30% reduction
        
        # Combine effects
        price = base_price * hour_effect * weekend_effect * renewable_effect
        
        # Add realistic noise
        noise = np.random.normal(0, 150, len(df))
        price = price + noise
        
        # Clip to realistic bounds
        price = np.clip(price, 1500, 6000)
        
        df['price_inr_mwh'] = price
        
        return df
    
    def train_rf(self, X_train, y_train):
        """Train Random Forest model"""
        print("  Training Random Forest...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Save
        joblib.dump(self.rf_model, os.path.join(self.model_dir, "price_rf.pkl"))
        return self.rf_model
    
    def train_gb(self, X_train, y_train):
        """Train Gradient Boosting model"""
        print("  Training Gradient Boosting...")
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        
        # Save
        joblib.dump(self.gb_model, os.path.join(self.model_dir, "price_gb.pkl"))
        return self.gb_model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, seq_length=72):
        """Train LSTM model"""
        if not KERAS_AVAILABLE:
            print("  Skipping LSTM (TensorFlow not available)")
            return None
        
        print("  Training LSTM...")
        
        # Reshape for LSTM
        n_features = X_train.shape[1]
        X_train_seq = X_train[:len(X_train)//seq_length*seq_length].reshape(-1, seq_length, n_features)
        y_train_seq = y_train[:len(y_train)//seq_length*seq_length].reshape(-1, seq_length)
        
        X_val_seq = X_val[:len(X_val)//seq_length*seq_length].reshape(-1, seq_length, n_features)
        y_val_seq = y_val[:len(y_val)//seq_length*seq_length].reshape(-1, seq_length)
        
        # Build model
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(seq_length)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.lstm_model = model
        model.save(os.path.join(self.model_dir, "price_lstm.h5"))
        
        return model
    
    def train_all(self, df, use_lstm=True):
        """Train all price forecasting models"""
        print("\nTraining Price Forecasting Models...")
        print("-" * 50)
        
        # Generate synthetic prices if not present
        if 'price_inr_mwh' not in df.columns:
            print("  Generating synthetic price data...")
            df = self.generate_synthetic_prices(df)
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        y = df['price_inr_mwh'].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"  Training samples: {len(X)}")
        
        # Scale
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.2, shuffle=False
        )
        
        # Train models
        self.train_rf(X_train, y_train)
        self.train_gb(X_train, y_train)
        
        if use_lstm and KERAS_AVAILABLE:
            self.train_lstm(X_train, y_train, X_val, y_val)
        
        # Save scalers
        joblib.dump({
            'feature': self.feature_scaler,
            'target': self.target_scaler
        }, os.path.join(self.model_dir, "price_scalers.pkl"))
        
        # Evaluate
        self._evaluate_models(X_val, y_val)
        
        print("\n✓ Price models trained successfully")
    
    def _evaluate_models(self, X_val, y_val):
        """Evaluate model performance"""
        print("\n  Model Performance:")
        
        if self.rf_model:
            y_pred = self.rf_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            print(f"    RF MAE: ₹{self.target_scaler.inverse_transform([[mae]])[0][0]:.2f}/MWh")
        
        if self.gb_model:
            y_pred = self.gb_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            print(f"    GB MAE: ₹{self.target_scaler.inverse_transform([[mae]])[0][0]:.2f}/MWh")
    
    def forecast(self, recent_data, load_forecast, renewable_forecast, model_type='gb'):
        """
        Forecast prices for next 24 hours
        
        Args:
            recent_data: Recent historical data
            load_forecast: 24-hour load forecast (MW)
            renewable_forecast: DataFrame with solar_mw, wind_mw columns
            model_type: 'rf', 'gb', or 'lstm'
        
        Returns:
            Array of 24 hourly price forecasts (₹/MWh)
        """
        # Prepare forecast features
        forecast_df = pd.DataFrame()
        
        # Net demand
        total_renewable = renewable_forecast['solar_mw'].values + renewable_forecast['wind_mw'].values
        forecast_df['net_demand_mw'] = load_forecast - total_renewable
        forecast_df['demand_mw'] = load_forecast
        forecast_df['solar_wind_mw'] = total_renewable
        
        # Time features (next 24 hours)
        last_hour = recent_data['hour'].iloc[-1]
        forecast_df['hour'] = [(last_hour + i + 1) % 24 for i in range(24)]
        forecast_df['hour_sin'] = np.sin(2 * np.pi * forecast_df['hour'] / 24)
        forecast_df['hour_cos'] = np.cos(2 * np.pi * forecast_df['hour'] / 24)
        
        # Day of week
        last_dow = recent_data['day_of_week'].iloc[-1]
        forecast_df['day_of_week'] = [last_dow if i < (24 - last_hour - 1) else (last_dow + 1) % 7 
                                       for i in range(24)]
        forecast_df['is_weekend'] = (forecast_df['day_of_week'] >= 5).astype(int)
        
        # Weather (use last known or average)
        forecast_df['temperature_c'] = recent_data['temperature_c'].iloc[-24:].mean()
        
        # Prepare features
        X, _ = self.prepare_features(forecast_df)
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict
        if model_type == 'rf' and self.rf_model:
            y_pred_scaled = self.rf_model.predict(X_scaled)
        elif model_type == 'gb' and self.gb_model:
            y_pred_scaled = self.gb_model.predict(X_scaled)
        elif model_type == 'lstm' and self.lstm_model:
            # For LSTM, use simple approach (not sequence-based for forecast)
            y_pred_scaled = self.gb_model.predict(X_scaled) if self.gb_model else None
        else:
            # Fallback to simple model
            y_pred_scaled = self._simple_price_model(forecast_df)
            return y_pred_scaled
        
        # Inverse transform
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Ensure realistic bounds
        y_pred = np.clip(y_pred, 1500, 6000)
        
        return y_pred
    
    def _simple_price_model(self, df):
        """Simple rule-based price model as fallback"""
        net_demand_norm = (df['net_demand_mw'] - df['net_demand_mw'].min()) / \
                          (df['net_demand_mw'].max() - df['net_demand_mw'].min() + 1e-6)
        
        base_price = 2500 + 2000 * net_demand_norm
        
        # Peak hour adjustment
        hour_mult = np.where(
            (df['hour'] >= 18) & (df['hour'] <= 22), 1.15,
            np.where((df['hour'] >= 0) & (df['hour'] <= 5), 0.9, 1.0)
        )
        
        return base_price * hour_mult


def run_price_forecasting(recent_data, load_forecast, renewable_forecast, train=False):
    """Convenience function to run price forecasting"""
    forecaster = PriceForecaster()
    
    if train:
        forecaster.train_all(recent_data)
    
    price_forecast = forecaster.forecast(recent_data, load_forecast, renewable_forecast)
    
    return price_forecast
