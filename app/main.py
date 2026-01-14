"""
FastAPI application initialization and configuration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.endpoints import baseline, anomaly, health, dashboard  # NEW - Added dashboard

# Try to import OCR if available
try:
    from app.api.endpoints import ocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Personalized Health Monitoring with ML-based Anomaly Detection, OCR & Dashboard Analytics"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(baseline.router, prefix="/baseline", tags=["Baseline"])
app.include_router(anomaly.router, prefix="/anomaly", tags=["Anomaly Detection"])
app.include_router(health.router, prefix="/health", tags=["Health Check"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])  # NEW

if OCR_AVAILABLE:
    app.include_router(ocr.router, prefix="/ocr", tags=["OCR Text Extraction"])


@app.get("/")
async def root():
    """Root endpoint."""
    features = ["baseline_training", "anomaly_detection", "dashboard_analytics"]
    if OCR_AVAILABLE:
        features.append("ocr_extraction")

    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "features": features
    }


if __name__ == "__main__":
    import uvicorn

    print(f"\n🚀 Starting {settings.PROJECT_NAME}...")
    print(f"   📍 Server: http://{settings.HOST}:{settings.PORT}")
    print(f"   📚 Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   💾 Storage: {settings.STORAGE_TYPE.upper()}")

    if OCR_AVAILABLE and settings.OCR_SPACE_API_KEY:
        print(f"   📸 OCR: ENABLED")
    elif OCR_AVAILABLE:
        print(f"   📸 OCR: DISABLED (Add OCR_SPACE_API_KEY to .env)")

    print(f"   📊 Dashboard: ENABLED\n")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )