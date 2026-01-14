"""
Dashboard service - handles dashboard data aggregation and analysis.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from app.storage.base import BaseStorage


class DashboardService:
    """Service for dashboard data aggregation and health scoring."""

    def __init__(self, storage: BaseStorage):
        self.storage = storage

    async def get_overview(self, user_id: str) -> Dict:
        """
        Get complete dashboard overview for a user.

        Args:
            user_id: Unique user identifier

        Returns:
            Dict with overview data
        """
        # Check if user has baseline
        baseline = await self.storage.get_baseline(user_id)
        has_baseline = baseline is not None

        if not has_baseline:
            return {
                "status": "error",
                "message": f"No baseline found for user {user_id}. Please train baseline first.",
                "user_id": user_id,
                "has_baseline": False
            }

        # Get latest reading
        latest_reading = await self.storage.get_latest_health_data(user_id)

        # Get recent stats (last 7 days)
        recent_stats = await self.storage.get_health_data_stats(user_id, days=7)

        # Get recent anomaly counts (last 7 days)
        anomaly_counts = await self.storage.get_anomaly_counts(user_id, days=7)

        # Calculate health score
        health_score_data = await self._calculate_health_score(
            user_id, baseline, recent_stats, anomaly_counts
        )

        return {
            "status": "success",
            "user_id": user_id,
            "has_baseline": True,
            "latest_reading": latest_reading,
            "latest_reading_time": latest_reading.get('timestamp') if latest_reading else None,
            "recent_stats": recent_stats,
            "total_readings": recent_stats.get('total_readings', 0),
            "recent_anomaly_count": anomaly_counts.get('total', 0),
            "high_risk_anomalies": anomaly_counts.get('High', 0),
            "medium_risk_anomalies": anomaly_counts.get('Medium', 0),
            "low_risk_anomalies": anomaly_counts.get('Low', 0),
            "health_score": health_score_data.get('health_score'),
            "health_status": health_score_data.get('health_status')
        }

    async def get_trends(self, user_id: str, days: int = 7, metrics: List[str] = None) -> Dict:
        """
        Get time-series trend data for charts.

        Args:
            user_id: Unique user identifier
            days: Number of days to include
            metrics: List of metrics to include (default: all)

        Returns:
            Dict with trend data
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get health data for range
        data = await self.storage.get_health_data_range(
            user_id,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

        if not data:
            return {
                "status": "error",
                "message": f"No health data found for user {user_id}",
                "user_id": user_id,
                "days": days,
                "metrics": []
            }

        # Default metrics
        if metrics is None:
            metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

        # Build trend data for each metric
        metric_trends = []
        for metric in metrics:
            trend_data = []
            values = []

            for point in data:
                value = point.get(metric)
                if value is not None:
                    trend_data.append({
                        'timestamp': point['timestamp'],
                        'value': float(value)
                    })
                    values.append(float(value))

            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)

                # Determine trend (simple linear regression)
                trend = self._calculate_trend(values)

                metric_trends.append({
                    'metric': metric.replace('_', ' ').title(),
                    'data': trend_data,
                    'avg': round(avg, 2),
                    'min': round(min_val, 2),
                    'max': round(max_val, 2),
                    'trend': trend
                })

        return {
            "status": "success",
            "user_id": user_id,
            "days": days,
            "metrics": metric_trends
        }

    async def get_anomaly_history(self, user_id: str, days: int = 30, risk_level: str = None) -> Dict:
        """
        Get anomaly history for dashboard.

        Args:
            user_id: Unique user identifier
            days: Number of days to include
            risk_level: Filter by risk level (High, Medium, Low)

        Returns:
            Dict with anomaly history
        """
        # Get anomaly history
        anomalies = await self.storage.get_anomaly_history(user_id, days, risk_level)

        # Get counts
        counts = await self.storage.get_anomaly_counts(user_id, days)

        # Format anomalies
        formatted_anomalies = []
        for anomaly in anomalies:
            formatted_anomalies.append({
                'timestamp': anomaly['timestamp'],
                'metric': anomaly['metric'],
                'current_value': round(anomaly['current_value'], 2),
                'baseline_value': round(anomaly['baseline_value'], 2),
                'deviation_pct': round(anomaly['deviation_pct'], 2),
                'risk_level': anomaly['risk_level'],
                'confidence': round(anomaly['confidence'], 2),
                'detected_at': anomaly['detected_at']
            })

        return {
            "status": "success",
            "user_id": user_id,
            "days": days,
            "total_anomalies": counts.get('total', 0),
            "high_risk_count": counts.get('High', 0),
            "medium_risk_count": counts.get('Medium', 0),
            "low_risk_count": counts.get('Low', 0),
            "anomalies": formatted_anomalies
        }

    async def get_health_score(self, user_id: str) -> Dict:
        """
        Calculate comprehensive health score.

        Args:
            user_id: Unique user identifier

        Returns:
            Dict with health score and details
        """
        # Get baseline
        baseline = await self.storage.get_baseline(user_id)
        if not baseline:
            return {
                "status": "error",
                "message": f"No baseline found for user {user_id}"
            }

        # Get recent stats and anomalies
        recent_stats = await self.storage.get_health_data_stats(user_id, days=7)
        anomaly_counts = await self.storage.get_anomaly_counts(user_id, days=7)

        # Calculate score
        score_data = await self._calculate_health_score(
            user_id, baseline, recent_stats, anomaly_counts
        )

        return {
            "status": "success",
            "user_id": user_id,
            **score_data
        }

    async def get_stats(self, user_id: str, days: int = 30) -> Dict:
        """
        Get aggregated statistics with baseline comparison.

        Args:
            user_id: Unique user identifier
            days: Number of days for statistics

        Returns:
            Dict with aggregated stats
        """
        # Get baseline
        baseline = await self.storage.get_baseline(user_id)
        if not baseline:
            return {
                "status": "error",
                "message": f"No baseline found for user {user_id}"
            }

        # Get stats
        stats = await self.storage.get_health_data_stats(user_id, days)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Compare with baseline
        vs_baseline = {}
        for metric in ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']:
            if metric in stats and 'avg' in stats[metric]:
                current_avg = stats[metric]['avg']
                baseline_mean = baseline[metric]['mean']

                deviation = current_avg - baseline_mean
                deviation_pct = (deviation / baseline_mean) * 100

                # Determine status (better/worse depends on metric)
                if metric in ['heart_rate', 'stress_level']:
                    # Lower is better for these metrics
                    status = "better" if deviation < 0 else "worse"
                else:
                    # Higher is better for these metrics
                    status = "better" if deviation > 0 else "worse"

                vs_baseline[metric] = {
                    "deviation": round(deviation, 2),
                    "deviation_pct": round(deviation_pct, 2),
                    "status": status if abs(deviation_pct) > 5 else "stable"
                }

        return {
            "status": "success",
            "user_id": user_id,
            "period": f"Last {days} days",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "stats": stats,
            "total_readings": stats.get('total_readings', 0),
            "vs_baseline": vs_baseline
        }

    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================

    async def _calculate_health_score(self, user_id: str, baseline: Dict,
                                      recent_stats: Dict, anomaly_counts: Dict) -> Dict:
        """Calculate overall health score (0-100)."""
        if recent_stats.get('total_readings', 0) == 0:
            return {
                "health_score": 0.0,
                "health_status": "No Data",
                "metric_scores": {},
                "positive_factors": [],
                "negative_factors": [],
                "recommendations": ["Start recording health data to get a health score"]
            }

        metric_scores = {}
        positive_factors = []
        negative_factors = []
        recommendations = []

        # Define ideal directions for each metric
        lower_is_better = {'heart_rate', 'stress_level'}
        higher_is_better = {'steps', 'sleep_quality', 'calories'}

        # Score each metric (0-100)
        for metric in ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']:
            if metric not in recent_stats or 'avg' not in recent_stats[metric]:
                continue

            current_avg = recent_stats[metric]['avg']
            baseline_mean = baseline[metric]['mean']
            baseline_std = baseline[metric]['std']

            # Skip if baseline std is 0 (no variation in training data)
            if baseline_std == 0:
                metric_scores[metric] = 100  # Assume good if no variation
                continue

            # Calculate Z-score (how many std devs away from baseline)
            z_score = (current_avg - baseline_mean) / baseline_std
            abs_z = abs(z_score)

            # Base score on how close to baseline (0-100)
            if abs_z <= 0.5:
                base_score = 100  # Very close to baseline
            elif abs_z <= 1.0:
                base_score = 95  # Within 1 std dev
            elif abs_z <= 1.5:
                base_score = 85  # Within 1.5 std dev
            elif abs_z <= 2.0:
                base_score = 75  # Within 2 std dev
            elif abs_z <= 2.5:
                base_score = 60  # Getting concerning
            else:
                base_score = 50  # Significantly different

            # Apply directional adjustment
            # Positive z_score means current > baseline
            # Negative z_score means current < baseline
            direction_bonus = 0

            if metric in lower_is_better:
                # For heart_rate and stress: lower is better
                if z_score < -0.5:  # Current is lower than baseline (good!)
                    direction_bonus = min(5, abs(z_score) * 2)
                    if abs_z >= 1.0:
                        positive_factors.append(
                            f"{metric.replace('_', ' ').title()} reduced from baseline (healthy)"
                        )
                elif z_score > 1.0:  # Current is higher than baseline (bad)
                    direction_bonus = max(-10, -z_score * 3)
                    negative_factors.append(
                        f"{metric.replace('_', ' ').title()} elevated above baseline"
                    )
                    recommendations.append(
                        f"Work on reducing {metric.replace('_', ' ')}"
                    )

            elif metric in higher_is_better:
                # For steps, sleep_quality, calories: higher is better
                if z_score > 0.5:  # Current is higher than baseline (good!)
                    direction_bonus = min(5, z_score * 2)
                    if abs_z >= 1.0:
                        positive_factors.append(
                            f"{metric.replace('_', ' ').title()} improved from baseline"
                        )
                elif z_score < -1.0:  # Current is lower than baseline (bad)
                    direction_bonus = max(-10, z_score * 3)
                    negative_factors.append(
                        f"{metric.replace('_', ' ').title()} declined from baseline"
                    )
                    recommendations.append(
                        f"Try to improve {metric.replace('_', ' ')}"
                    )

            # Calculate final metric score
            final_metric_score = max(0, min(100, base_score + direction_bonus))
            metric_scores[metric] = final_metric_score

        # Calculate average of all metric scores
        if metric_scores:
            avg_metric_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            return {
                "health_score": 50.0,
                "health_status": "Insufficient Data",
                "metric_scores": {},
                "positive_factors": ["Not enough metrics available"],
                "negative_factors": [],
                "recommendations": ["Ensure all health metrics are being tracked"]
            }

        # Apply moderate anomaly penalty
        total_anomalies = (
                anomaly_counts.get('High', 0) +
                anomaly_counts.get('Medium', 0) +
                anomaly_counts.get('Low', 0)
        )

        # Calculate penalty based on severity
        raw_penalty = (
                anomaly_counts.get('High', 0) * 4 +  # 4 points per high-risk
                anomaly_counts.get('Medium', 0) * 2 +  # 2 points per medium
                anomaly_counts.get('Low', 0) * 0.5  # 0.5 points per low
        )

        # Cap the penalty at 15 points maximum
        anomaly_penalty = min(15, raw_penalty)

        # Apply the penalty
        final_score = max(0, min(100, avg_metric_score - anomaly_penalty))

        # Add anomaly-related factors
        if anomaly_counts.get('High', 0) > 0:
            negative_factors.append(
                f"Detected {anomaly_counts['High']} high-risk anomal{'y' if anomaly_counts['High'] == 1 else 'ies'}"
            )
            recommendations.append(
                "Consult healthcare provider about high-risk anomalies"
            )

        if total_anomalies == 0:
            positive_factors.append("No anomalies detected in recent period")

        # Determine health status
        if final_score >= 90:
            status = "Excellent"
        elif final_score >= 75:
            status = "Good"
        elif final_score >= 60:
            status = "Fair"
        elif final_score >= 45:
            status = "Needs Attention"
        else:
            status = "Poor"

        # Add encouraging factor if score is good
        if final_score >= 80:
            positive_factors.append("Maintaining healthy metrics overall")

        # Remove duplicates while preserving order
        positive_factors = list(dict.fromkeys(positive_factors))[:5]
        negative_factors = list(dict.fromkeys(negative_factors))[:5]
        recommendations = list(dict.fromkeys(recommendations))[:5]

        return {
            "health_score": round(final_score, 1),
            "health_status": status,
            "metric_scores": {k: round(v, 1) for k, v in metric_scores.items()},
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "recommendations": recommendations
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend
        n = len(values)
        x = list(range(n))

        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine trend based on slope
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"