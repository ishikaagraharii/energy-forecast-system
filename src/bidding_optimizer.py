"""
Bidding Strategy Optimization Module
Determines optimal bid prices and volumes for Day-Ahead Market
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime


class BiddingOptimizer:
    """
    Optimizes bidding strategy for Day-Ahead Market
    
    Strategies:
    - Conservative: Bid below forecast to ensure acceptance
    - Aggressive: Bid higher to maximize profit (risk of rejection)
    - Optimal: Balance acceptance probability and profit
    
    Supports both generator (selling) and buyer (purchasing) perspectives
    """
    
    def __init__(self, capacity_mw=500, role='generator', output_dir='output'):
        """
        Args:
            capacity_mw: Generator capacity or demand obligation
            role: 'generator' (selling power) or 'buyer' (purchasing power)
            output_dir: Directory to save bid schedules
        """
        self.capacity_mw = capacity_mw
        self.role = role
        self.output_dir = output_dir
        
        # Strategy parameters
        self.strategies = {
            'conservative': {'bid_margin': -0.05, 'volume_factor': 0.95},  # Bid 5% below forecast
            'moderate': {'bid_margin': -0.02, 'volume_factor': 1.0},       # Bid 2% below forecast
            'aggressive': {'bid_margin': 0.02, 'volume_factor': 1.0},      # Bid 2% above forecast
            'optimal': {'bid_margin': -0.03, 'volume_factor': 0.98}        # Optimized balance
        }
        
        # Market parameters
        self.imbalance_penalty_rate = 0.15  # 15% penalty for imbalance
        self.min_bid_price = 1500  # ₹/MWh
        self.max_bid_price = 6000  # ₹/MWh
    
    def calculate_bid_price(self, forecast_price, strategy='optimal'):
        """
        Calculate bid price based on forecast and strategy
        
        Args:
            forecast_price: Forecasted MCP (₹/MWh)
            strategy: Bidding strategy name
        
        Returns:
            Bid price (₹/MWh)
        """
        params = self.strategies.get(strategy, self.strategies['optimal'])
        bid_margin = params['bid_margin']
        
        if self.role == 'generator':
            # Generator: Bid below forecast to increase acceptance
            bid_price = forecast_price * (1 + bid_margin)
        else:
            # Buyer: Bid above forecast to ensure purchase
            bid_price = forecast_price * (1 - bid_margin)
        
        # Clip to market bounds
        bid_price = np.clip(bid_price, self.min_bid_price, self.max_bid_price)
        
        return bid_price
    
    def calculate_bid_volume(self, hour, strategy='optimal'):
        """
        Calculate bid volume based on capacity and strategy
        
        Args:
            hour: Hour of day (0-23)
            strategy: Bidding strategy name
        
        Returns:
            Bid volume (MW)
        """
        params = self.strategies.get(strategy, self.strategies['optimal'])
        volume_factor = params['volume_factor']
        
        # Base volume
        base_volume = self.capacity_mw * volume_factor
        
        # Adjust for peak/off-peak (generators may offer more during peak)
        if self.role == 'generator':
            if 18 <= hour <= 22:  # Evening peak
                volume = base_volume * 1.0
            elif 0 <= hour <= 5:  # Night valley
                volume = base_volume * 0.8
            else:
                volume = base_volume
        else:
            # Buyer: Purchase more during expected low-price hours
            volume = base_volume
        
        return min(volume, self.capacity_mw)
    
    def generate_bid_schedule(self, price_forecast, strategy='optimal'):
        """
        Generate 24-hour bid schedule
        
        Args:
            price_forecast: Array of 24 forecasted prices
            strategy: Bidding strategy name
        
        Returns:
            DataFrame with columns: hour, forecast_price, bid_price, bid_volume
        """
        bid_schedule = []
        
        for hour in range(24):
            forecast_price = price_forecast[hour]
            bid_price = self.calculate_bid_price(forecast_price, strategy)
            bid_volume = self.calculate_bid_volume(hour, strategy)
            
            bid_schedule.append({
                'hour': hour,
                'forecast_price': forecast_price,
                'bid_price': bid_price,
                'bid_volume': bid_volume
            })
        
        df = pd.DataFrame(bid_schedule)
        
        # Calculate expected metrics
        df['expected_revenue'] = df['bid_price'] * df['bid_volume']
        df['acceptance_prob'] = self._estimate_acceptance_probability(
            df['bid_price'], df['forecast_price']
        )
        
        return df
    
    def _estimate_acceptance_probability(self, bid_price, forecast_price):
        """
        Estimate probability of bid acceptance
        
        Simple model: 
        - Bid below forecast = high acceptance
        - Bid above forecast = low acceptance
        """
        if self.role == 'generator':
            # Generator: Lower bid = higher acceptance
            ratio = bid_price / forecast_price
            prob = np.clip(1.2 - ratio, 0.3, 0.95)
        else:
            # Buyer: Higher bid = higher acceptance
            ratio = bid_price / forecast_price
            prob = np.clip(ratio - 0.8, 0.3, 0.95)
        
        return prob
    
    def simulate_market_clearing(self, bid_schedule, actual_prices=None):
        """
        Simulate market clearing and calculate realized profit
        
        Args:
            bid_schedule: DataFrame from generate_bid_schedule()
            actual_prices: Actual MCP (if available for backtesting)
        
        Returns:
            Dictionary with simulation results
        """
        if actual_prices is None:
            # Use forecast as proxy
            actual_prices = bid_schedule['forecast_price'].values
        
        results = []
        
        for idx, row in bid_schedule.iterrows():
            hour = row['hour']
            bid_price = row['bid_price']
            bid_volume = row['bid_volume']
            actual_price = actual_prices[idx]
            
            # Determine if bid is accepted
            if self.role == 'generator':
                accepted = bid_price <= actual_price
            else:
                accepted = bid_price >= actual_price
            
            # Calculate revenue/cost
            if accepted:
                cleared_volume = bid_volume
                cleared_price = actual_price  # Pay-as-cleared market
            else:
                cleared_volume = 0
                cleared_price = 0
            
            # Calculate imbalance (if any)
            if self.role == 'generator':
                imbalance = bid_volume - cleared_volume
                imbalance_cost = imbalance * actual_price * self.imbalance_penalty_rate
                net_revenue = cleared_volume * cleared_price - imbalance_cost
            else:
                imbalance = cleared_volume - bid_volume
                imbalance_cost = abs(imbalance) * actual_price * self.imbalance_penalty_rate
                net_cost = cleared_volume * cleared_price + imbalance_cost
                net_revenue = -net_cost  # Negative for buyer
            
            results.append({
                'hour': hour,
                'bid_price': bid_price,
                'bid_volume': bid_volume,
                'actual_price': actual_price,
                'accepted': accepted,
                'cleared_volume': cleared_volume,
                'cleared_price': cleared_price,
                'revenue': cleared_volume * cleared_price,
                'imbalance_cost': imbalance_cost,
                'net_revenue': net_revenue
            })
        
        df_results = pd.DataFrame(results)
        
        # Summary metrics
        summary = {
            'total_revenue': df_results['revenue'].sum(),
            'total_imbalance_cost': df_results['imbalance_cost'].sum(),
            'net_profit': df_results['net_revenue'].sum(),
            'acceptance_rate': df_results['accepted'].mean(),
            'avg_cleared_price': df_results[df_results['cleared_volume'] > 0]['cleared_price'].mean(),
            'total_energy': df_results['cleared_volume'].sum()
        }
        
        return {
            'hourly_results': df_results,
            'summary': summary
        }
    
    def compare_strategies(self, price_forecast, actual_prices=None):
        """
        Compare all bidding strategies
        
        Returns:
            DataFrame comparing strategy performance
        """
        comparison = []
        
        for strategy_name in self.strategies.keys():
            bid_schedule = self.generate_bid_schedule(price_forecast, strategy_name)
            simulation = self.simulate_market_clearing(bid_schedule, actual_prices)
            
            comparison.append({
                'strategy': strategy_name,
                'net_profit': simulation['summary']['net_profit'],
                'acceptance_rate': simulation['summary']['acceptance_rate'],
                'total_energy': simulation['summary']['total_energy'],
                'avg_price': simulation['summary']['avg_cleared_price']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('net_profit', ascending=False)
        
        return df_comparison
    
    def save_bid_schedule(self, bid_schedule, strategy='optimal'):
        """Save bid schedule to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"bid_schedule_{strategy}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        bid_schedule.to_csv(filepath, index=False)
        print(f"  Bid schedule saved: {filepath}")
        
        return filepath
    
    def generate_bid_report(self, bid_schedule, simulation_results):
        """Generate text report of bidding strategy"""
        report = []
        report.append("=" * 70)
        report.append("BIDDING STRATEGY REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Role: {self.role.upper()}")
        report.append(f"Capacity: {self.capacity_mw} MW")
        
        report.append("\n" + "-" * 70)
        report.append("BID SUMMARY")
        report.append("-" * 70)
        report.append(f"Total Bid Volume: {bid_schedule['bid_volume'].sum():,.0f} MWh")
        report.append(f"Avg Bid Price: ₹{bid_schedule['bid_price'].mean():,.2f}/MWh")
        report.append(f"Min Bid Price: ₹{bid_schedule['bid_price'].min():,.2f}/MWh")
        report.append(f"Max Bid Price: ₹{bid_schedule['bid_price'].max():,.2f}/MWh")
        report.append(f"Avg Acceptance Probability: {bid_schedule['acceptance_prob'].mean():.1%}")
        
        if simulation_results:
            summary = simulation_results['summary']
            report.append("\n" + "-" * 70)
            report.append("SIMULATION RESULTS")
            report.append("-" * 70)
            report.append(f"Total Revenue: ₹{summary['total_revenue']:,.2f}")
            report.append(f"Imbalance Cost: ₹{summary['total_imbalance_cost']:,.2f}")
            report.append(f"Net Profit: ₹{summary['net_profit']:,.2f}")
            report.append(f"Acceptance Rate: {summary['acceptance_rate']:.1%}")
            report.append(f"Total Energy Cleared: {summary['total_energy']:,.0f} MWh")
            report.append(f"Avg Clearing Price: ₹{summary['avg_cleared_price']:,.2f}/MWh")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.output_dir, "bidding_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


def run_bidding_optimization(price_forecast, capacity_mw=500, strategy='optimal'):
    """Convenience function to run bidding optimization"""
    optimizer = BiddingOptimizer(capacity_mw=capacity_mw)
    
    # Generate bid schedule
    bid_schedule = optimizer.generate_bid_schedule(price_forecast, strategy)
    
    # Simulate market
    simulation = optimizer.simulate_market_clearing(bid_schedule)
    
    # Generate report
    report = optimizer.generate_bid_report(bid_schedule, simulation)
    
    return {
        'bid_schedule': bid_schedule,
        'simulation': simulation,
        'report': report
    }
