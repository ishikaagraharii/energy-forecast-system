"""
Energy Forecasting Pipeline
Main orchestrator for the multi-stage forecasting system

Pipeline:
1. Data Preprocessing
2. Weather Forecasting (24-hour ahead)
3. Load Forecasting (24-hour ahead)
4. Renewable Generation Forecasting (24-hour ahead)
5. Visualization & Reporting
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessor import preprocess_all_data
from src.weather_forecaster import WeatherForecaster, run_weather_forecasting
from src.load_forecaster import LoadForecaster, run_load_forecasting
from src.renewable_forecaster import RenewableForecaster, run_renewable_forecasting
from src.visualizations import generate_all_visualizations, plot_combined_dashboard
from src.price_forecaster import PriceForecaster, run_price_forecasting
from src.bidding_optimizer import BiddingOptimizer, run_bidding_optimization
from src.market_simulator import MarketSimulator, run_market_simulation


class EnergyForecastingPipeline:
    """
    Multi-stage energy forecasting pipeline
    
    Stages:
    - Stage 1: Weather Forecast (temperature, solar radiation, wind, humidity, cloud cover)
    - Stage 2: Load Forecast (electricity demand)
    - Stage 3: Renewable Forecast (solar + wind generation)
    - Stage 4: Price Forecast (market clearing price)
    - Stage 5: Bidding Optimization (optimal bid strategy)
    - Stage 6: Market Simulation (performance evaluation)
    """
    
    def __init__(self, data_dir="data", model_dir="models", output_dir="output", capacity_mw=500):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.capacity_mw = capacity_mw
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        
        # Forecasters
        self.weather_forecaster = None
        self.load_forecaster = None
        self.renewable_forecaster = None
        self.price_forecaster = None
        self.bidding_optimizer = None
        self.market_simulator = None
        
        # Data
        self.df_weather = None
        self.df_merged = None
        
        # Forecasts
        self.weather_forecast = None
        self.load_forecast = None
        self.renewable_forecast = None
        self.price_forecast = None
        self.bid_schedule = None
        self.simulation_results = None
    
    def load_data(self):
        """Load preprocessed data or run preprocessing"""
        print("\n" + "=" * 70)
        print("STAGE 0: DATA LOADING")
        print("=" * 70)
        
        processed_path = os.path.join(self.data_dir, "processed", "merged_dataset.csv")
        weather_path = os.path.join(self.data_dir, "weather_india_central.csv")
        
        # Check if processed data exists
        if os.path.exists(processed_path):
            print("Loading preprocessed data...")
            self.df_merged = pd.read_csv(processed_path)
            self.df_merged['datetime'] = pd.to_datetime(self.df_merged['datetime'])
            print(f"  Loaded {len(self.df_merged)} records")
        else:
            print("Running data preprocessing...")
            self.df_merged = preprocess_all_data()
        
        # Load weather data
        if os.path.exists(weather_path):
            self.df_weather = pd.read_csv(weather_path)
            self.df_weather['datetime'] = pd.to_datetime(self.df_weather['datetime'])
            print(f"  Weather data: {len(self.df_weather)} records")
        
        print(f"\nData range: {self.df_merged['datetime'].min()} to {self.df_merged['datetime'].max()}")
        return True
    
    def run_weather_stage(self, train=True):
        """Stage 1: Weather Forecasting"""
        print("\n" + "=" * 70)
        print("STAGE 1: WEATHER FORECASTING")
        print("=" * 70)
        
        self.weather_forecaster = WeatherForecaster(forecast_horizon=24)
        
        if train and self.df_weather is not None:
            # Train on all but last week
            train_data = self.df_weather.iloc[:-168]
            self.weather_forecaster.train_all(train_data, use_lstm=True)
        
        # Generate 24-hour forecast
        recent_weather = self.df_weather.tail(200) if self.df_weather is not None else self.df_merged.tail(200)
        self.weather_forecast = self.weather_forecaster.forecast_all(recent_weather, model_type='lstm')
        
        if self.weather_forecast is None or len(self.weather_forecast) == 0:
            print("  Warning: No weather forecast generated, using fallback")
            self.weather_forecast = recent_weather[self.weather_forecaster.weather_features].tail(24).reset_index(drop=True)
        
        print(f"\n✓ Weather forecast generated: {len(self.weather_forecast)} hours")
        return self.weather_forecast
    
    def run_load_stage(self, train=True):
        """Stage 2: Load Forecasting"""
        print("\n" + "=" * 70)
        print("STAGE 2: LOAD FORECASTING")
        print("=" * 70)
        
        self.load_forecaster = LoadForecaster(forecast_horizon=24)
        
        if train:
            train_data = self.df_merged.iloc[:-168]
            self.load_forecaster.train_all(train_data)
        
        # Generate 24-hour forecast
        recent_data = self.df_merged.tail(200)
        self.load_forecast = self.load_forecaster.forecast(
            recent_data, 
            model_type='lstm',
            weather_forecast=self.weather_forecast
        )
        
        if self.load_forecast is None:
            print("  Warning: LSTM failed, using RF fallback")
            self.load_forecast = self.load_forecaster.forecast(recent_data, model_type='rf')
        
        if self.load_forecast is None:
            self.load_forecast = np.full(24, recent_data['demand_mw'].mean())
        
        print(f"\n✓ Load forecast generated: {len(self.load_forecast)} hours")
        return self.load_forecast
    
    def run_renewable_stage(self, train=True):
        """Stage 3: Renewable Generation Forecasting"""
        print("\n" + "=" * 70)
        print("STAGE 3: RENEWABLE GENERATION FORECASTING")
        print("=" * 70)
        
        self.renewable_forecaster = RenewableForecaster(forecast_horizon=24)
        
        if train:
            train_data = self.df_merged.iloc[:-168]
            self.renewable_forecaster.train_all(train_data)
        
        # Generate 24-hour forecast using weather forecast
        recent_data = self.df_merged.tail(200)
        self.renewable_forecast = self.renewable_forecaster.forecast(
            recent_data,
            weather_forecast=self.weather_forecast,
            model_type='lstm'
        )
        
        if self.renewable_forecast is None or len(self.renewable_forecast) == 0:
            print("  Warning: Using RF fallback")
            self.renewable_forecast = self.renewable_forecaster.forecast(recent_data, model_type='rf')
        
        if self.renewable_forecast is None or len(self.renewable_forecast) == 0:
            self.renewable_forecast = pd.DataFrame({
                'solar_mw': np.zeros(24),
                'wind_mw': np.zeros(24),
                'total_renewable_mw': np.zeros(24)
            })
        
        print(f"\n✓ Renewable forecast generated: {len(self.renewable_forecast)} hours")
        return self.renewable_forecast
    
    def run_price_stage(self, train=True):
        """Stage 4: Price Forecasting"""
        print("\n" + "=" * 70)
        print("STAGE 4: PRICE FORECASTING")
        print("=" * 70)
        
        self.price_forecaster = PriceForecaster(forecast_horizon=24)
        
        if train:
            train_data = self.df_merged.iloc[:-168]
            self.price_forecaster.train_all(train_data)
        
        # Generate 24-hour price forecast
        recent_data = self.df_merged.tail(200)
        self.price_forecast = self.price_forecaster.forecast(
            recent_data,
            self.load_forecast,
            self.renewable_forecast,
            model_type='gb'
        )
        
        if self.price_forecast is None:
            print("  Warning: Using fallback price model")
            self.price_forecast = np.full(24, 3000)
        
        print(f"\n✓ Price forecast generated: {len(self.price_forecast)} hours")
        print(f"  Price range: ₹{self.price_forecast.min():,.0f} - ₹{self.price_forecast.max():,.0f}/MWh")
        print(f"  Average price: ₹{self.price_forecast.mean():,.0f}/MWh")
        
        return self.price_forecast
    
    def run_bidding_stage(self, strategy='optimal'):
        """Stage 5: Bidding Optimization"""
        print("\n" + "=" * 70)
        print("STAGE 5: BIDDING OPTIMIZATION")
        print("=" * 70)
        
        self.bidding_optimizer = BiddingOptimizer(
            capacity_mw=self.capacity_mw,
            role='generator',
            output_dir=self.output_dir
        )
        
        # Generate bid schedule
        self.bid_schedule = self.bidding_optimizer.generate_bid_schedule(
            self.price_forecast,
            strategy=strategy
        )
        
        # Simulate market clearing
        simulation = self.bidding_optimizer.simulate_market_clearing(self.bid_schedule)
        
        # Generate report
        report = self.bidding_optimizer.generate_bid_report(self.bid_schedule, simulation)
        
        print(f"\n✓ Bid schedule generated: {len(self.bid_schedule)} hours")
        print(f"  Strategy: {strategy}")
        print(f"  Expected profit: ₹{simulation['summary']['net_profit']:,.2f}")
        print(f"  Avg acceptance prob: {self.bid_schedule['acceptance_prob'].mean():.1%}")
        
        return {
            'bid_schedule': self.bid_schedule,
            'simulation': simulation,
            'report': report
        }
    
    def run_simulation_stage(self):
        """Stage 6: Market Simulation & Evaluation"""
        print("\n" + "=" * 70)
        print("STAGE 6: MARKET SIMULATION & EVALUATION")
        print("=" * 70)
        
        self.market_simulator = MarketSimulator(output_dir=self.output_dir)
        
        # Compare strategies
        strategy_comparison = self.market_simulator.compare_strategies(
            self.df_merged,
            {
                'load': self.load_forecast,
                'renewable': self.renewable_forecast,
                'price': self.price_forecast
            },
            capacity_mw=self.capacity_mw
        )
        
        # Generate performance report
        report = self.market_simulator.generate_performance_report(strategy_comparison)
        
        print(f"\n✓ Market simulation completed")
        print(f"  Best strategy: {strategy_comparison.iloc[0]['strategy']}")
        print(f"  Expected profit: ₹{strategy_comparison.iloc[0]['total_profit']:,.2f}")
        
        self.simulation_results = {
            'strategy_comparison': strategy_comparison,
            'report': report
        }
        
        return self.simulation_results
    
    def generate_visualizations(self, show=False):
        """Generate all visualization outputs"""
        print("\n" + "=" * 70)
        print("STAGE 4: VISUALIZATION")
        print("=" * 70)
        
        generate_all_visualizations(
            self.weather_forecast,
            self.load_forecast,
            self.renewable_forecast,
            show=show
        )
        
        return True
    
    def generate_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE FORECAST & MARKET REPORT")
        print("=" * 70)
        
        report = []
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Forecast Horizon: 24 hours\n")
        
        # Weather summary
        report.append("WEATHER FORECAST:")
        report.append("-" * 40)
        for col in self.weather_forecast.columns:
            vals = self.weather_forecast[col]
            report.append(f"  {col}: {vals.min():.1f} - {vals.max():.1f} (avg: {vals.mean():.1f})")
        
        # Load summary
        report.append("\nLOAD FORECAST:")
        report.append("-" * 40)
        report.append(f"  Peak Load: {np.max(self.load_forecast):,.0f} MW")
        report.append(f"  Min Load: {np.min(self.load_forecast):,.0f} MW")
        report.append(f"  Average: {np.mean(self.load_forecast):,.0f} MW")
        report.append(f"  Total Energy: {np.sum(self.load_forecast):,.0f} MWh")
        
        # Renewable summary
        report.append("\nRENEWABLE FORECAST:")
        report.append("-" * 40)
        report.append(f"  Solar Peak: {self.renewable_forecast['solar_mw'].max():,.0f} MW")
        report.append(f"  Wind Avg: {self.renewable_forecast['wind_mw'].mean():,.0f} MW")
        report.append(f"  Total RE Generation: {self.renewable_forecast['total_renewable_mw'].sum():,.0f} MWh")
        
        # Renewable share
        re_share = 100 * self.renewable_forecast['total_renewable_mw'].mean() / np.mean(self.load_forecast)
        report.append(f"\n  Renewable Share: ~{re_share:.1f}% of load")
        
        # Price summary
        if self.price_forecast is not None:
            report.append("\nPRICE FORECAST:")
            report.append("-" * 40)
            report.append(f"  Peak Price: ₹{np.max(self.price_forecast):,.0f}/MWh")
            report.append(f"  Min Price: ₹{np.min(self.price_forecast):,.0f}/MWh")
            report.append(f"  Average Price: ₹{np.mean(self.price_forecast):,.0f}/MWh")
        
        # Bidding summary
        if self.bid_schedule is not None:
            report.append("\nBIDDING STRATEGY:")
            report.append("-" * 40)
            report.append(f"  Total Bid Volume: {self.bid_schedule['bid_volume'].sum():,.0f} MWh")
            report.append(f"  Avg Bid Price: ₹{self.bid_schedule['bid_price'].mean():,.0f}/MWh")
            report.append(f"  Expected Revenue: ₹{self.bid_schedule['expected_revenue'].sum():,.0f}")
            report.append(f"  Avg Acceptance Prob: {self.bid_schedule['acceptance_prob'].mean():.1%}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        report_path = os.path.join(self.output_dir, "forecast_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ Report saved: {report_path}")
        
        return report_text
    
    def run_full_pipeline(self, train_models=True, show_plots=False, include_market=True):
        """Execute the complete forecasting and market pipeline"""
        print("\n" + "=" * 70)
        print("  ENERGY FORECASTING & MARKET BIDDING PIPELINE")
        print("  Multi-Stage 24-Hour Ahead Forecast + Optimization")
        print("=" * 70)
        
        start_time = datetime.now()
        
        try:
            # Stage 0: Load data
            self.load_data()
            
            # Stage 1: Weather forecast
            self.run_weather_stage(train=train_models)
            
            # Stage 2: Load forecast
            self.run_load_stage(train=train_models)
            
            # Stage 3: Renewable forecast
            self.run_renewable_stage(train=train_models)
            
            if include_market:
                # Stage 4: Price forecast
                self.run_price_stage(train=train_models)
                
                # Stage 5: Bidding optimization
                bidding_results = self.run_bidding_stage(strategy='optimal')
                
                # Stage 6: Market simulation
                self.run_simulation_stage()
            
            # Visualizations
            self.generate_visualizations(show=show_plots)
            
            # Generate comprehensive report
            self.generate_report()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "=" * 70)
            print("       PIPELINE COMPLETED SUCCESSFULLY")
            print(f"       Total time: {elapsed:.1f} seconds")
            print("=" * 70)
            
            results = {
                'weather': self.weather_forecast,
                'load': self.load_forecast,
                'renewable': self.renewable_forecast
            }
            
            if include_market:
                results.update({
                    'price': self.price_forecast,
                    'bid_schedule': self.bid_schedule,
                    'simulation': self.simulation_results
                })
            
            return results
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def quick_forecast(self):
        """
        Quick forecast using pre-trained models
        (Skips training, loads saved models)
        """
        return self.run_full_pipeline(train_models=False, show_plots=False)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Energy Forecasting Pipeline')
    parser.add_argument('--no-train', action='store_true', help='Skip model training, use saved models')
    parser.add_argument('--show-plots', action='store_true', help='Display plots interactively')
    args = parser.parse_args()
    
    pipeline = EnergyForecastingPipeline()
    
    results = pipeline.run_full_pipeline(
        train_models=not args.no_train,
        show_plots=args.show_plots
    )
    
    if results:
        print("\nForecast outputs saved to: output/")
        print("  - weather_forecast.png")
        print("  - load_forecast.png")
        print("  - renewable_forecast.png")
        print("  - dashboard.png")
        print("  - forecast_report.txt")
    
    return results


if __name__ == "__main__":
    main()
