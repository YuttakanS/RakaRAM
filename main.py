"""
RakaRAM - Serverless API Entrypoint
====================================
FastAPI application providing real-time RAM price estimation, future market trends,
and global DRAM news intelligence.

Designed for serverless deployment (Vercel, AWS Lambda, Cloud Run) and local ASGI execution.
The top-level ASGI framework instance is exposed globally as `app`.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core domain modules
from src.models.predict import RamPricePredictor
from src.news.news_analyzer import get_latest_news_analysis, analyze_news_articles

# ---------------------------------------------------------------------------
# Global Application Instance (Required for Vercel / Serverless ASGI handlers)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RakaRAM API",
    description="Real-time Computer RAM Price Predictor & Global Semiconductor Market Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Standard Serverless WSGI/ASGI Aliases (Vercel, AWS Lambda, etc.)
application = app
handler = app

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton model loader
_predictor_instance: Optional[RamPricePredictor] = None

def get_predictor() -> RamPricePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        try:
            _predictor_instance = RamPricePredictor()
        except Exception as e:
            raise RuntimeError(f"Could not load ML price model: {e}")
    return _predictor_instance


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    brand: str = Field(default="Kingston", example="Kingston", description="RAM Brand (e.g. Kingston, Corsair, G.Skill)")
    generation: str = Field(default="DDR5", example="DDR5", description="Memory generation: 'DDR4' or 'DDR5'")
    capacity_gb: int = Field(default=32, example=32, description="Total capacity in GB (e.g. 8, 16, 32, 64)")
    modules: int = Field(default=2, example=2, description="Number of modules/sticks (1 or 2)")
    speed_mhz: int = Field(default=6000, example=6000, description="Memory clock speed in MHz (e.g. 3200, 5600, 6000)")
    cas_latency: int = Field(default=36, example=36, description="CAS latency timing (e.g. 16, 30, 36, 40)")
    has_rgb: int = Field(default=1, example=1, description="RGB lighting indicator: 1 for Yes, 0 for No")
    form_factor: str = Field(default="UDIMM", example="UDIMM", description="Form factor: 'UDIMM' (Desktop) or 'SO-DIMM' (Laptop)")


class PredictResponse(BaseModel):
    estimated_price_thb: int
    price_range: Dict[str, int]
    forecast: Dict[str, Any]
    inputs: Dict[str, Any]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    """Root endpoint: Health status, documentation link, and API overview."""
    return {
        "service": "RakaRAM API",
        "description": "RAM Price Prediction & Global DRAM Market Intelligence",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "predict_get": "/predict?brand=Kingston&generation=DDR5&capacity_gb=32&modules=2&speed_mhz=6000&cas_latency=36&has_rgb=1",
            "predict_post": "/predict",
            "news": "/news",
            "market_timeline": "/market/timeline",
            "health": "/health"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health", tags=["General"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint: Verifies model accessibility and service readiness."""
    try:
        predictor = get_predictor()
        model_ready = predictor is not None
    except Exception as e:
        model_ready = False

    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_price_get(
    brand: str = Query("Kingston", description="Brand name (e.g. Kingston, Corsair, G.Skill, KLEVV)"),
    generation: str = Query("DDR5", pattern="^(DDR4|DDR5)$", description="Memory generation: DDR4 or DDR5"),
    capacity_gb: int = Query(32, ge=4, le=256, description="Total capacity in GB"),
    modules: int = Query(2, ge=1, le=4, description="Number of modules (1, 2, 4)"),
    speed_mhz: int = Query(6000, ge=2133, le=10000, description="Speed in MHz"),
    cas_latency: int = Query(36, ge=10, le=50, description="CAS latency"),
    has_rgb: int = Query(1, ge=0, le=1, description="RGB flag: 1 for Yes, 0 for No"),
    form_factor: str = Query("UDIMM", pattern="^(UDIMM|SO-DIMM)$", description="Form factor")
) -> Dict[str, Any]:
    """
    Real-time price prediction via GET query parameters.
    Returns fair market price in Thai Baht (THB), confidence interval, and 6-month forecast.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            brand=brand,
            generation=generation,
            capacity_gb=capacity_gb,
            modules=modules,
            speed_mhz=speed_mhz,
            cas_latency=cas_latency,
            has_rgb=has_rgb,
            form_factor=form_factor
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_price_post(payload: PredictRequest) -> Dict[str, Any]:
    """
    Real-time price prediction via POST JSON payload.
    Returns fair market price in Thai Baht (THB), confidence interval, and 6-month forecast.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            brand=payload.brand,
            generation=payload.generation,
            capacity_gb=payload.capacity_gb,
            modules=payload.modules,
            speed_mhz=payload.speed_mhz,
            cas_latency=payload.cas_latency,
            has_rgb=payload.has_rgb,
            form_factor=payload.form_factor
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/news", tags=["Market Intelligence"])
def get_news(force_refresh: bool = Query(False, description="Whether to trigger fresh news sync")) -> Dict[str, Any]:
    """
    Retrieves latest global DRAM news intelligence, sentiment classification,
    and net price multipliers for DDR4 and DDR5.
    """
    try:
        if force_refresh:
            return analyze_news_articles(force_refresh=True)
        return get_latest_news_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis error: {str(e)}")


@app.get("/market/timeline", tags=["Market Intelligence"])
def get_market_timeline() -> Dict[str, Any]:
    """
    Returns the projected 6-month price movement comparison for DDR4 vs DDR5,
    dynamically adjusted by global semiconductor intelligence.
    """
    months = ["Nov 26", "Dec 26", "Jan 27", "Feb 27", "Mar 27", "Apr 27", "May 27"]
    d4_base = [100.0, 100.3, 100.8, 101.5, 102.1, 102.7, 103.2]
    d5_base = [100.0, 98.7, 97.3, 96.0, 94.8, 93.8, 93.0]

    try:
        news_intel = get_latest_news_analysis()
        d4_impact = news_intel.get("net_impact_ddr4_pct", 0.0)
        d5_impact = news_intel.get("net_impact_ddr5_pct", 0.0)
    except Exception:
        d4_impact, d5_impact = 0.0, 0.0

    adj4 = (d4_impact / 100.0) * 0.4 if d4_impact else 0.0
    adj5 = (d5_impact / 100.0) * 0.4 if d5_impact else 0.0

    d4_timeline = [round(r + (adj4 * (i / 6.0) * 100.0), 1) for i, r in enumerate(d4_base)]
    d5_timeline = [round(r + (adj5 * (i / 6.0) * 100.0), 1) for i, r in enumerate(d5_base)]

    return {
        "months": months,
        "ddr4": {
            "name": "DDR4 (Legacy · Foundry supply shrinking)",
            "index_timeline": d4_timeline,
            "net_news_impact_pct": d4_impact,
            "trend": "upward_drift"
        },
        "ddr5": {
            "name": "DDR5 (Mainstream · Die yield expanding)",
            "index_timeline": d5_timeline,
            "net_news_impact_pct": d5_impact,
            "trend": "downward_easing"
        },
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


# ---------------------------------------------------------------------------
# Local Development Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
