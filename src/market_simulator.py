"""
Market Simulation and Performance Evaluation Module
Backtests bidding strategies and compares performance
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


class MarketSimulator:
    """
    Simulates Day-Ahead Market operations and evaluates strategy performance
    
    Features:
    - Backtest bidding strategies on historical data
    - Compare multiple strategies
    - Calculate performance metrics
    - Generate evaluation reports
    """
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.simulation_results = []
    
    def backtest_strategy(self, historical_data, forecasts, bidding_optimizer, 
                         strategy='optimal', window_days=30):
        """
        Backtest a bidding strategy on historical data
        
        Args:
            historical_data: DataFrame with actual demand, renewable, prices
            forecasts: Dict with 'load', 'renewable', 'price' forecasts
            bidding_optimizer: BiddingOptimizer instance
            strategy: Strategy name to test
            window_days: Number of days to backtest
        
        Returns:
            Dictionary with backtest results
        """
        print(f"\nBacktesting '{strategy}' strategy...")
        print("-" * 50)
        
        # Simulate for each day
        daily_results = []
        
        # Use last window_days of data
        start_idx = max(0, len(historical_data) - window_days * 24)
        test_data = historical_data.iloc[start_idx:].copy()
        
        # Generate synthetic actual prices if not present
        if 'price_inr_mwh' not in test_data.columns:
            from src.price_forecaster import PriceForecaster
            pf = PriceForecaster()
            test_data = pf.generate_synthetic_prices(test_data)
        
        # Simulate day by day
        for day in range(min(window_days, len(test_data) // 24)):
            day_data = test_data.iloc[day*24:(day+1)*24]
            
            if len(day_data) < 24:
                continue
            
            # Use forecasted prices (or actual as proxy)
            forecast_prices = day_data['price_inr_mwh'].values
            actual_prices = day_data['price_inr_mwh'].values
            
            # Generate bid schedule
            bid_schedule = bidding_optimizer.generate_bid_schedule(
                forecast_prices, strategy
            )
            
            # Simulate market clearing
            simulation = bidding_optimizer.simulate_market_clearing(
                bid_schedule, actual_prices
            )
            
            daily_results.append({
                'day': day,
                'date': day_data['datetime'].iloc[0] if 'datetime' in day_data.columns else None,
                'net_profit': simulation['summary']['net_profit'],
                'acceptance_rate': simulation['summary']['acceptance_rate'],
                'total_energy': simulation['summary']['total_energy'],
                'avg_price': simulation['summary']['avg_cleared_price']
            })
        
        df_daily = pd.DataFrame(daily_results)
        
        # Calculate aggregate metrics
        total_profit = df_daily['net_profit'].sum()
        avg_acceptance = df_daily['acceptance_rate'].mean()
        total_energy = df_daily['total_energy'].sum()
        
        print(f"  Total Profit: ₹{total_profit:,.2f}")
        print(f"  Avg Acceptance Rate: {avg_acceptance:.1%}")
        print(f"  Total Energy: {total_energy:,.0f} MWh")
        
        return {
            'strategy': strategy,
            'daily_results': df_daily,
            'total_profit': total_profit,
            'avg_acceptance_rate': avg_acceptance,
            'total_energy': total_energy
        }
    
    def compare_strategies(self, historical_data, forecasts, capacity_mw=500):
        """
        Compare multiple bidding strategies
        
        Returns:
            DataFrame with strategy comparison
        """
        from src.bidding_optimizer import BiddingOptimizer
        
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        
        strategies = ['conservative', 'moderate', 'aggressive', 'optimal']
        comparison_results = []
        
        optimizer = BiddingOptimizer(capacity_mw=capacity_mw)
        
        for strategy in strategies:
            result = self.backtest_strategy(
                historical_data, forecasts, optimizer, strategy, window_days=30
            )
            
            comparison_results.append({
                'strategy': strategy,
                'total_profit': result['total_profit'],
                'avg_acceptance_rate': result['avg_acceptance_rate'],
                'total_energy': result['total_energy'],
                'profit_per_mwh': result['total_profit'] / result['total_energy'] if result['total_energy'] > 0 else 0
            })
        
        df_comparison = pd.DataFrame(comparison_results)
        df_comparison = df_comparison.sort_values('total_profit', ascending=False)
        
        # Add baseline comparison
        best_profit = df_comparison['total_profit'].max()
        df_comparison['vs_best'] = (df_comparison['total_profit'] / best_profit - 1) * 100
        
        return df_comparison
    
    def evaluate_forecast_accuracy(self, forecasts, actuals):
        """
        Evaluate forecasting accuracy
        
        Args:
            forecasts: Dict with 'load', 'renewable', 'price' forecasts
            actuals: Dict with actual values
        
        Returns:
            Dictionary with accuracy metrics
        """
        metrics = {}
        
        for key in ['load', 'price']:
            if key in forecasts and key in actuals:
                forecast = np.array(forecasts[key])
                actual = np.array(actuals[key])
                
                # Ensure same length
                min_len = min(len(forecast), len(actual))
                forecast = forecast[:min_len]
                actual = actual[:min_len]
                
                mae = mean_absolute_error(actual, forecast)
                mape = mean_absolute_percentage_error(actual, forecast) * 100
                rmse = np.sqrt(np.mean((actual - forecast) ** 2))
                
                metrics[key] = {
                    'MAE': mae,
                    'MAPE': mape,
                    'RMSE': rmse
                }
        
        return metrics
    
    def generate_performance_report(self, strategy_comparison, forecast_accuracy=None):
        """Generate comprehensive performance report"""
        report = []
        report.append("=" * 70)
        report.append("MARKET PERFORMANCE EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n" + "-" * 70)
        report.append("STRATEGY COMPARISON")
        report.append("-" * 70)
        
        for _, row in strategy_comparison.iterrows():
            report.append(f"\n{row['strategy'].upper()}:")
            report.append(f"  Total Profit: ₹{row['total_profit']:,.2f}")
            report.append(f"  Acceptance Rate: {row['avg_acceptance_rate']:.1%}")
            report.append(f"  Total Energy: {row['total_energy']:,.0f} MWh")
            report.append(f"  Profit per MWh: ₹{row['profit_per_mwh']:,.2f}")
            if 'vs_best' in row:
                report.append(f"  vs Best: {row['vs_best']:+.1f}%")
        
        # Best strategy
        best = strategy_comparison.iloc[0]
        report.append(f"\n✓ RECOMMENDED STRATEGY: {best['strategy'].upper()}")
        report.append(f"  Expected Profit: ₹{best['total_profit']:,.2f}")
        
        if forecast_accuracy:
            report.append("\n" + "-" * 70)
            report.append("FORECAST ACCURACY")
            report.append("-" * 70)
            
            for key, metrics in forecast_accuracy.items():
                report.append(f"\n{key.upper()} FORECAST:")
                report.append(f"  MAE: {metrics['MAE']:,.2f}")
                report.append(f"  MAPE: {metrics['MAPE']:.2f}%")
                report.append(f"  RMSE: {metrics['RMSE']:,.2f}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.output_dir, "performance_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ Performance report saved: {report_path}")
        
        return report_text
    
    def export_results_json(self, results, filename='simulation_results.json'):
        """Export simulation results to JSON"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert DataFrames to dict for JSON serialization
        export_data = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                export_data[key] = value.to_dict(orient='records')
            elif isinstance(value, (np.integer, np.floating)):
                export_data[key] = float(value)
            else:
                export_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"  Results exported: {filepath}")
        
        return filepath


def run_market_simulation(historical_data, forecasts, capacity_mw=500):
    """Convenience function to run complete market simulation"""
    simulator = MarketSimulator()
    
    # Compare strategies
    strategy_comparison = simulator.compare_strategies(
        historical_data, forecasts, capacity_mw
    )
    
    # Generate report
    report = simulator.generate_performance_report(strategy_comparison)
    
    # Export results
    results = {
        'strategy_comparison': strategy_comparison,
        'report': report
    }
    
    simulator.export_results_json(results)
    
    return results
