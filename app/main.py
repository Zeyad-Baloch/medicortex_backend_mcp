"""
FastAPI application initialization and configuration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.endpoints import baseline, anomaly, health


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Personalized Health Monitoring with ML-based Anomaly Detection"
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"\n🚀 Starting {settings.PROJECT_NAME}...")
    print(f"   📍 Server: http://{settings.HOST}:{settings.PORT}")
    print(f"   📚 Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   💾 Storage: {settings.STORAGE_TYPE.upper()}\n")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
