"""
FastAPI Backend for Energy Forecasting Dashboard
Provides REST API endpoints for forecasting and market operations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import EnergyForecastingPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Energy Forecasting & Market Bidding API",
    description="API for electricity demand, renewable generation, price forecasting and optimal bidding",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web"), name="static")

# Global pipeline instance
pipeline = None
last_forecast_time = None


# Pydantic models
class ForecastRequest(BaseModel):
    train_models: bool = False
    include_market: bool = True
    capacity_mw: Optional[float] = 500
    strategy: Optional[str] = 'optimal'


class BiddingRequest(BaseModel):
    capacity_mw: float = 500
    strategy: str = 'optimal'


# Helper functions
def numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def serialize_forecast_results(results):
    """Serialize forecast results for JSON response"""
    serialized = {}
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            serialized[key] = value.to_dict(orient='records')
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, dict):
            serialized[key] = {k: numpy_to_python(v) for k, v in value.items()}
        else:
            serialized[key] = numpy_to_python(value)
    
    return serialized


# API Endpoints

@app.get("/")
async def root():
    """Serve the main dashboard HTML"""
    return FileResponse("web/index.html")


@app.get("/api/status")
async def get_status():
    """Get system status"""
    global pipeline, last_forecast_time
    
    return {
        "status": "online",
        "pipeline_initialized": pipeline is not None,
        "last_forecast": last_forecast_time.isoformat() if last_forecast_time else None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/forecast/run")
async def run_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """Run complete forecasting pipeline"""
    global pipeline, last_forecast_time
    
    try:
        # Initialize pipeline
        pipeline = EnergyForecastingPipeline(capacity_mw=request.capacity_mw)
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            train_models=request.train_models,
            show_plots=False,
            include_market=request.include_market
        )
        
        if results is None:
            raise HTTPException(status_code=500, detail="Pipeline execution failed")
        
        last_forecast_time = datetime.now()
        
        # Serialize results
        response = serialize_forecast_results(results)
        
        return {
            "success": True,
            "timestamp": last_forecast_time.isoformat(),
            "data": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/latest")
async def get_latest_forecast():
    """Get latest forecast results"""
    global pipeline, last_forecast_time
    
    if pipeline is None:
        raise HTTPException(status_code=404, detail="No forecast available. Run forecast first.")
    
    try:
        results = {
            'weather': pipeline.weather_forecast,
            'load': pipeline.load_forecast,
            'renewable': pipeline.renewable_forecast,
            'price': pipeline.price_forecast,
            'bid_schedule': pipeline.bid_schedule
        }
        
        response = serialize_forecast_results(results)
        
        return {
            "success": True,
            "timestamp": last_forecast_time.isoformat() if last_forecast_time else None,
            "data": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/weather")
async def get_weather_forecast():
    """Get weather forecast only"""
    global pipeline
    
    if pipeline is None or pipeline.weather_forecast is None:
        raise HTTPException(status_code=404, detail="Weather forecast not available")
    
    return {
        "success": True,
        "data": pipeline.weather_forecast.to_dict(orient='records')
    }


@app.get("/api/forecast/load")
async def get_load_forecast():
    """Get load forecast only"""
    global pipeline
    
    if pipeline is None or pipeline.load_forecast is None:
        raise HTTPException(status_code=404, detail="Load forecast not available")
    
    return {
        "success": True,
        "data": pipeline.load_forecast.tolist()
    }


@app.get("/api/forecast/renewable")
async def get_renewable_forecast():
    """Get renewable forecast only"""
    global pipeline
    
    if pipeline is None or pipeline.renewable_forecast is None:
        raise HTTPException(status_code=404, detail="Renewable forecast not available")
    
    return {
        "success": True,
        "data": pipeline.renewable_forecast.to_dict(orient='records')
    }


@app.get("/api/forecast/price")
async def get_price_forecast():
    """Get price forecast only"""
    global pipeline
    
    if pipeline is None or pipeline.price_forecast is None:
        raise HTTPException(status_code=404, detail="Price forecast not available")
    
    return {
        "success": True,
        "data": pipeline.price_forecast.tolist()
    }


@app.get("/api/bidding/schedule")
async def get_bid_schedule():
    """Get current bid schedule"""
    global pipeline
    
    if pipeline is None or pipeline.bid_schedule is None:
        raise HTTPException(status_code=404, detail="Bid schedule not available")
    
    return {
        "success": True,
        "data": pipeline.bid_schedule.to_dict(orient='records')
    }


@app.post("/api/bidding/optimize")
async def optimize_bidding(request: BiddingRequest):
    """Optimize bidding strategy"""
    global pipeline
    
    if pipeline is None or pipeline.price_forecast is None:
        raise HTTPException(status_code=404, detail="Price forecast required. Run forecast first.")
    
    try:
        from src.bidding_optimizer import BiddingOptimizer
        
        optimizer = BiddingOptimizer(
            capacity_mw=request.capacity_mw,
            role='generator'
        )
        
        bid_schedule = optimizer.generate_bid_schedule(
            pipeline.price_forecast,
            strategy=request.strategy
        )
        
        simulation = optimizer.simulate_market_clearing(bid_schedule)
        
        return {
            "success": True,
            "data": {
                "bid_schedule": bid_schedule.to_dict(orient='records'),
                "simulation": serialize_forecast_results(simulation)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulation/results")
async def get_simulation_results():
    """Get market simulation results"""
    global pipeline
    
    if pipeline is None or pipeline.simulation_results is None:
        raise HTTPException(status_code=404, detail="Simulation results not available")
    
    return {
        "success": True,
        "data": serialize_forecast_results(pipeline.simulation_results)
    }


@app.get("/api/summary")
async def get_summary():
    """Get comprehensive summary of all forecasts"""
    global pipeline, last_forecast_time
    
    if pipeline is None:
        raise HTTPException(status_code=404, detail="No forecast available")
    
    try:
        summary = {
            "timestamp": last_forecast_time.isoformat() if last_forecast_time else None,
            "load": {
                "peak": float(np.max(pipeline.load_forecast)),
                "min": float(np.min(pipeline.load_forecast)),
                "average": float(np.mean(pipeline.load_forecast)),
                "total_energy": float(np.sum(pipeline.load_forecast))
            },
            "renewable": {
                "solar_peak": float(pipeline.renewable_forecast['solar_mw'].max()),
                "wind_avg": float(pipeline.renewable_forecast['wind_mw'].mean()),
                "total_generation": float(pipeline.renewable_forecast['total_renewable_mw'].sum()),
                "renewable_share": float(100 * pipeline.renewable_forecast['total_renewable_mw'].mean() / np.mean(pipeline.load_forecast))
            }
        }
        
        if pipeline.price_forecast is not None:
            summary["price"] = {
                "peak": float(np.max(pipeline.price_forecast)),
                "min": float(np.min(pipeline.price_forecast)),
                "average": float(np.mean(pipeline.price_forecast))
            }
        
        if pipeline.bid_schedule is not None:
            summary["bidding"] = {
                "total_volume": float(pipeline.bid_schedule['bid_volume'].sum()),
                "avg_bid_price": float(pipeline.bid_schedule['bid_price'].mean()),
                "expected_revenue": float(pipeline.bid_schedule['expected_revenue'].sum()),
                "avg_acceptance_prob": float(pipeline.bid_schedule['acceptance_prob'].mean())
            }
        
        return {
            "success": True,
            "data": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
