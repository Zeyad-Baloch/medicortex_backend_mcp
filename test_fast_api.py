
import requests
import json
from datetime import datetime, timedelta

BASE = "http://localhost:8000"


print("Testing FastAPI...")
r = requests.get(BASE)
print(r.json())


print("\nGenerating 7 days of data...")
data = []
start = datetime(2025, 12, 20, 0, 0, 0)

for day in range(7):
    for hour in range(24):
        ts = start + timedelta(days=day, hours=hour)
        data.append({
            "timestamp": ts.isoformat(),
            "heart_rate": 70 + (hour % 12),
            "steps": 500 if 6 <= hour <= 22 else 10,
            "sleep_quality": 80 if (hour >= 22 or hour <= 6) else 0,
            "stress_level": 30 + (hour % 10),
            "calories": 90 if 6 <= hour <= 22 else 30
        })

print(f"Generated {len(data)} data points")

print("\nTraining baseline...")
r = requests.post(f"{BASE}/baseline/train", json={
    "user_id": "test_123",
    "days": 7,
    "data": data
})
result = r.json()
print(f"Status: {result.get('status', 'error')}")
if result.get('status') == 'success':
    print(f"Baselines: {list(result['baselines'].keys())}")

# Test 3: Detect anomalies
print("\nDetecting anomalies...")
anomaly = [{
    "timestamp": "2025-12-27T14:00:00",
    "heart_rate": 145,
    "steps": 5,
    "sleep_quality": 0,
    "stress_level": 95,
    "calories": 15
}]

r = requests.post(f"{BASE}/anomaly/detect", json={
    "user_id": "test_123",
    "data": anomaly
})
result = r.json()
print(f"Total anomalies: {result['total_anomalies']}")
print(f"High risk: {result['high_risk_count']}")

if result['alerts']:
    print("\nAlerts:")
    for alert in result['alerts'][:3]:
        print(f"  - {alert['metric']}: {alert['deviation']}")
