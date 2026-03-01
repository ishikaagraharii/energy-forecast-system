"""
Visualization Module
Generates plots for weather, load, and renewable generation forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'solar': '#FFD700',
    'wind': '#4A90D9',
    'load': '#E74C3C',
    'temp': '#E74C3C',
    'humidity': '#3498DB',
    'radiation': '#F39C12'
}


def create_forecast_hours(start_hour=None, horizon=24):
    """Create datetime index for forecast horizon"""
    if start_hour is None:
        start_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    return pd.date_range(start=start_hour, periods=horizon, freq='h')


def plot_weather_forecast(weather_forecast, save=True, show=True):
    """
    Plot weather forecast variables
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('24-Hour Weather Forecast', fontsize=16, fontweight='bold')
    
    hours = create_forecast_hours()
    
    # Temperature
    ax = axes[0, 0]
    if 'temperature_c' in weather_forecast.columns:
        ax.plot(hours, weather_forecast['temperature_c'], color=COLORS['temp'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(hours, weather_forecast['temperature_c'], alpha=0.3, color=COLORS['temp'])
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Solar Radiation
    ax = axes[0, 1]
    if 'solar_radiation_wm2' in weather_forecast.columns:
        ax.plot(hours, weather_forecast['solar_radiation_wm2'], color=COLORS['radiation'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(hours, weather_forecast['solar_radiation_wm2'], alpha=0.3, color=COLORS['radiation'])
    ax.set_ylabel('W/m²')
    ax.set_title('Solar Radiation')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Wind Speed
    ax = axes[1, 0]
    if 'wind_speed_kmh' in weather_forecast.columns:
        ax.plot(hours, weather_forecast['wind_speed_kmh'], color=COLORS['wind'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(hours, weather_forecast['wind_speed_kmh'], alpha=0.3, color=COLORS['wind'])
    ax.set_ylabel('km/h')
    ax.set_title('Wind Speed')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Humidity
    ax = axes[1, 1]
    if 'humidity_percent' in weather_forecast.columns:
        ax.plot(hours, weather_forecast['humidity_percent'], color=COLORS['humidity'], linewidth=2, marker='o', markersize=4)
        ax.fill_between(hours, weather_forecast['humidity_percent'], alpha=0.3, color=COLORS['humidity'])
    ax.set_ylabel('%')
    ax.set_title('Relative Humidity')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Cloud Cover
    ax = axes[2, 0]
    if 'cloud_cover_percent' in weather_forecast.columns:
        ax.bar(hours, weather_forecast['cloud_cover_percent'], color=COLORS['primary'], alpha=0.7, width=0.03)
    ax.set_ylabel('%')
    ax.set_title('Cloud Cover')
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Summary stats
    ax = axes[2, 1]
    ax.axis('off')
    stats_text = "Weather Summary\n" + "=" * 30 + "\n"
    for col in weather_forecast.columns:
        if weather_forecast[col].dtype in ['float64', 'int64']:
            stats_text += f"{col}:\n  Min: {weather_forecast[col].min():.1f}\n  Max: {weather_forecast[col].max():.1f}\n  Avg: {weather_forecast[col].mean():.1f}\n\n"
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'weather_forecast.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/weather_forecast.png")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_load_forecast(load_forecast, historical=None, save=True, show=True):
    """
    Plot electricity load forecast
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hours = create_forecast_hours()
    
    # Plot historical if provided
    if historical is not None:
        hist_hours = create_forecast_hours(
            start_hour=hours[0] - timedelta(hours=len(historical)),
            horizon=len(historical)
        )
        ax.plot(hist_hours, historical, color='gray', linewidth=1.5, 
                label='Historical', alpha=0.7, linestyle='--')
    
    # Plot forecast
    ax.plot(hours, load_forecast, color=COLORS['load'], linewidth=2.5, 
            marker='o', markersize=5, label='Forecast')
    ax.fill_between(hours, load_forecast, alpha=0.2, color=COLORS['load'])
    
    # Confidence band (±5%)
    upper = load_forecast * 1.05
    lower = load_forecast * 0.95
    ax.fill_between(hours, lower, upper, alpha=0.1, color=COLORS['load'], label='±5% Confidence')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Load (MW)')
    ax.set_title('24-Hour Electricity Load Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    
    # Add annotations
    max_idx = np.argmax(load_forecast)
    min_idx = np.argmin(load_forecast)
    ax.annotate(f'Peak: {load_forecast[max_idx]:.0f} MW', 
                xy=(hours[max_idx], load_forecast[max_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color=COLORS['load'],
                arrowprops=dict(arrowstyle='->', color=COLORS['load']))
    ax.annotate(f'Min: {load_forecast[min_idx]:.0f} MW', 
                xy=(hours[min_idx], load_forecast[min_idx]),
                xytext=(10, -15), textcoords='offset points',
                fontsize=10, color=COLORS['load'])
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'load_forecast.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/load_forecast.png")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_renewable_forecast(renewable_forecast, save=True, show=True):
    """
    Plot renewable generation forecast (solar + wind)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    hours = create_forecast_hours()
    
    # Top: Stacked area chart
    ax = axes[0]
    ax.fill_between(hours, 0, renewable_forecast['solar_mw'], 
                    color=COLORS['solar'], alpha=0.8, label='Solar')
    ax.fill_between(hours, renewable_forecast['solar_mw'], 
                    renewable_forecast['solar_mw'] + renewable_forecast['wind_mw'],
                    color=COLORS['wind'], alpha=0.8, label='Wind')
    ax.plot(hours, renewable_forecast['total_renewable_mw'], 
            color='black', linewidth=2, linestyle='--', label='Total')
    
    ax.set_ylabel('Generation (MW)')
    ax.set_title('24-Hour Renewable Generation Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    
    # Bottom: Individual lines
    ax = axes[1]
    ax.plot(hours, renewable_forecast['solar_mw'], color=COLORS['solar'], 
            linewidth=2, marker='s', markersize=4, label='Solar')
    ax.plot(hours, renewable_forecast['wind_mw'], color=COLORS['wind'], 
            linewidth=2, marker='^', markersize=4, label='Wind')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Generation (MW)')
    ax.set_title('Solar vs Wind Generation', fontsize=12)
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    
    # Add peak annotations
    solar_peak = np.argmax(renewable_forecast['solar_mw'])
    ax.annotate(f'Solar Peak: {renewable_forecast["solar_mw"].iloc[solar_peak]:.0f} MW',
                xy=(hours[solar_peak], renewable_forecast['solar_mw'].iloc[solar_peak]),
                xytext=(5, 10), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'renewable_forecast.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/renewable_forecast.png")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_combined_dashboard(weather_forecast, load_forecast, renewable_forecast, save=True, show=True):
    """
    Create a combined dashboard with all forecasts
    """
    fig = plt.figure(figsize=(16, 12))
    
    hours = create_forecast_hours()
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Weather (3 subplots)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'temperature_c' in weather_forecast.columns:
        ax1.plot(hours, weather_forecast['temperature_c'], color=COLORS['temp'], linewidth=2)
        ax1.fill_between(hours, weather_forecast['temperature_c'], alpha=0.3, color=COLORS['temp'])
    ax1.set_ylabel('°C')
    ax1.set_title('Temperature', fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    
    ax2 = fig.add_subplot(gs[0, 1])
    if 'solar_radiation_wm2' in weather_forecast.columns:
        ax2.plot(hours, weather_forecast['solar_radiation_wm2'], color=COLORS['radiation'], linewidth=2)
        ax2.fill_between(hours, weather_forecast['solar_radiation_wm2'], alpha=0.3, color=COLORS['radiation'])
    ax2.set_ylabel('W/m²')
    ax2.set_title('Solar Radiation', fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    
    ax3 = fig.add_subplot(gs[0, 2])
    if 'wind_speed_kmh' in weather_forecast.columns:
        ax3.plot(hours, weather_forecast['wind_speed_kmh'], color=COLORS['wind'], linewidth=2)
        ax3.fill_between(hours, weather_forecast['wind_speed_kmh'], alpha=0.3, color=COLORS['wind'])
    ax3.set_ylabel('km/h')
    ax3.set_title('Wind Speed', fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    
    # Row 2: Load forecast (spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(hours, load_forecast, color=COLORS['load'], linewidth=2.5, marker='o', markersize=4)
    ax4.fill_between(hours, load_forecast, alpha=0.2, color=COLORS['load'])
    ax4.set_ylabel('MW')
    ax4.set_title('Electricity Load Forecast', fontweight='bold')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Row 2: Stats panel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats = f"""
FORECAST SUMMARY
{'='*25}

Load:
  Peak: {np.max(load_forecast):,.0f} MW
  Min:  {np.min(load_forecast):,.0f} MW
  Avg:  {np.mean(load_forecast):,.0f} MW

Renewables:
  Solar Peak: {renewable_forecast['solar_mw'].max():,.0f} MW
  Wind Avg:   {renewable_forecast['wind_mw'].mean():,.0f} MW
  Total Gen:  {renewable_forecast['total_renewable_mw'].sum():,.0f} MWh

Renewable Share:
  ~{100*renewable_forecast['total_renewable_mw'].mean()/np.mean(load_forecast):.1f}% of load
"""
    ax5.text(0.1, 0.9, stats, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Row 3: Renewable generation
    ax6 = fig.add_subplot(gs[2, :])
    ax6.fill_between(hours, 0, renewable_forecast['solar_mw'], 
                     color=COLORS['solar'], alpha=0.8, label='Solar')
    ax6.fill_between(hours, renewable_forecast['solar_mw'],
                     renewable_forecast['solar_mw'] + renewable_forecast['wind_mw'],
                     color=COLORS['wind'], alpha=0.8, label='Wind')
    ax6.plot(hours, renewable_forecast['total_renewable_mw'],
             color='black', linewidth=2, linestyle='--', label='Total RE')
    ax6.set_xlabel('Hour')
    ax6.set_ylabel('MW')
    ax6.set_title('Renewable Generation Forecast', fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    fig.suptitle('Energy Forecasting Dashboard - 24 Hour Ahead', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/dashboard.png")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def generate_all_visualizations(weather_forecast, load_forecast, renewable_forecast, show=False):
    """Generate all visualization plots"""
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50 + "\n")
    
    plot_weather_forecast(weather_forecast, save=True, show=show)
    plot_load_forecast(load_forecast, save=True, show=show)
    plot_renewable_forecast(renewable_forecast, save=True, show=show)
    plot_combined_dashboard(weather_forecast, load_forecast, renewable_forecast, save=True, show=show)
    
    print("\n✓ All visualizations generated successfully!")
    return True


if __name__ == "__main__":
    # Test with sample data
    hours = 24
    weather = pd.DataFrame({
        'temperature_c': 25 + 5 * np.sin(np.linspace(0, 2*np.pi, hours)),
        'humidity_percent': 60 + 20 * np.cos(np.linspace(0, 2*np.pi, hours)),
        'wind_speed_kmh': 10 + 5 * np.random.randn(hours),
        'solar_radiation_wm2': np.maximum(0, 500 * np.sin(np.linspace(-0.5, 3.5, hours))),
        'cloud_cover_percent': np.random.randint(0, 80, hours)
    })
    
    load = 120000 + 20000 * np.sin(np.linspace(0, 2*np.pi, hours)) + np.random.randn(hours) * 5000
    
    renewable = pd.DataFrame({
        'solar_mw': np.maximum(0, 15000 * np.sin(np.linspace(-0.5, 3.5, hours))),
        'wind_mw': 8000 + 2000 * np.random.randn(hours),
        'total_renewable_mw': np.zeros(hours)
    })
    renewable['total_renewable_mw'] = renewable['solar_mw'] + renewable['wind_mw']
    
    generate_all_visualizations(weather, load, renewable, show=True)
