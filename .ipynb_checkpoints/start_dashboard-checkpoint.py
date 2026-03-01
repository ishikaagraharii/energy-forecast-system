"""
Dashboard Startup Script
Launches the FastAPI server for the Energy Forecasting Dashboard
"""

import uvicorn
import sys
import os

# Add api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

if __name__ == "__main__":
    print("=" * 70)
    print("  ENERGY FORECASTING & MARKET DASHBOARD")
    print("=" * 70)
    print("\nStarting server...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
