"""
Dashboard endpoints - OPTIMIZED VERSION
Single endpoint returns all dashboard data to minimize API calls.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from app.dependencies import get_storage
from app.services.dashboard import DashboardService

router = APIRouter()


# ============================================================================
# MAIN UNIFIED ENDPOINT - Use this for frontend!
# ============================================================================

@router.get("/data/{user_id}")
async def get_complete_dashboard_data(
        user_id: str,
        days: int = Query(7, ge=1, le=90, description="Number of days for trends/stats"),
        include_trends: bool = Query(True, description="Include trend data for charts"),
        include_anomalies: bool = Query(True, description="Include anomaly history"),
        include_history: bool = Query(False, description="Include raw health data"),
        history_limit: int = Query(50, ge=1, le=500, description="Max history records")
):
    """
    Get ALL dashboard data in a SINGLE API call.

    Returns:
    - Overview (latest reading, health score, stats)
    - Trends (time-series data for charts) - optional
    - Anomaly history - optional
    - Raw data history - optional

    Frontend only needs to call THIS endpoint once!

    Example:
        GET /dashboard/data/user_123?days=7

    Returns everything the dashboard needs in one response.
    """
    try:
        storage = get_storage()
        service = DashboardService(storage)

        # Get overview (always included)
        overview = await service.get_overview(user_id)

        if overview.get("status") == "error":
            raise HTTPException(status_code=404, detail=overview["message"])

        # Build response
        response = {
            "status": "success",
            "user_id": user_id,
            "timestamp": overview.get("latest_reading_time"),

            # Core data (always included)
            "overview": {
                "has_baseline": overview.get("has_baseline"),
                "latest_reading": overview.get("latest_reading"),
                "health_score": overview.get("health_score"),
                "health_status": overview.get("health_status"),
                "total_readings": overview.get("total_readings"),
                "recent_stats": overview.get("recent_stats")
            },

            # Anomaly summary (always included)
            "anomalies": {
                "total": overview.get("recent_anomaly_count", 0),
                "high_risk": overview.get("high_risk_anomalies", 0),
                "medium_risk": overview.get("medium_risk_anomalies", 0),
                "low_risk": overview.get("low_risk_anomalies", 0)
            }
        }

        # Optional: Trends for charts
        if include_trends:
            trends = await service.get_trends(user_id, days=days)
            response["trends"] = trends.get("metrics", [])

        # Optional: Anomaly history
        if include_anomalies:
            anomaly_history = await service.get_anomaly_history(user_id, days=days)
            response["anomaly_history"] = anomaly_history.get("anomalies", [])

        # Optional: Raw health data
        if include_history:
            history = await storage.get_health_data_range(
                user_id,
                limit=history_limit
            )
            response["health_history"] = history

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OPTIONAL GRANULAR ENDPOINTS (for specific use cases)
# ============================================================================

@router.get("/overview/{user_id}")
async def get_dashboard_overview(user_id: str):
    """
    Get just the overview (use /data endpoint instead for better performance).
    """
    try:
        storage = get_storage()
        service = DashboardService(storage)
        result = await service.get_overview(user_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{user_id}")
async def get_dashboard_trends(
        user_id: str,
        days: int = Query(7, ge=1, le=90),
        metrics: Optional[str] = Query(None, description="Comma-separated list")
):
    """
    Get just trends (use /data endpoint instead for better performance).
    """
    try:
        storage = get_storage()
        service = DashboardService(storage)

        metric_list = None
        if metrics:
            metric_list = [m.strip() for m in metrics.split(',')]

        result = await service.get_trends(user_id, days=days, metrics=metric_list)

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-score/{user_id}")
async def get_health_score(user_id: str):
    """
    Get just health score (use /data endpoint instead for better performance).
    """
    try:
        storage = get_storage()
        service = DashboardService(storage)
        result = await service.get_health_score(user_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_dashboard_users():
    """
    List all users with dashboard data.
    """
    try:
        storage = get_storage()
        users = await storage.list_users()

        return {
            "status": "success",
            "total_users": len(users),
            "users": users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))