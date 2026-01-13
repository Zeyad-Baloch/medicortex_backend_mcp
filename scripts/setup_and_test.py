"""
Complete workflow: Train baseline → Detect anomalies
Run this before using mcp_client_groq.py
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:8000"

print("=" * 70)
print("MediCortex - Setup & Test Workflow")
print("=" * 70)

# Step 1: Check if backend is running
print("\n[1/3] Checking backend connection...")
try:
    response = requests.get(BASE_URL, timeout=5)
    print(f"✓ Backend is running: {response.json()}")
except Exception as e:
    print(f"❌ Backend not running! Start it with: python backend.py")
    print(f"   Error: {e}")
    exit(1)

# Step 2: Generate training data (7 days of normal health data)
print("\n[2/3] Generating 7 days of training data...")
data = []
start = datetime(2025, 12, 20, 0, 0, 0)

for day in range(7):
    for hour in range(24):
        ts = start + timedelta(days=day, hours=hour)
        data.append({
            "timestamp": ts.isoformat(),
            "heart_rate": 70 + (hour % 12),  # Normal: 70-82 bpm
            "steps": 500 if 6 <= hour <= 22 else 10,  # Active during day
            "sleep_quality": 80 if (hour >= 22 or hour <= 6) else 0,  # Good sleep at night
            "stress_level": 30 + (hour % 10),  # Normal: 30-40
            "calories": 90 if 6 <= hour <= 22 else 30  # More calories when active
        })

print(f"✓ Generated {len(data)} data points (7 days × 24 hours)")

# Step 3: Train baseline for user_123
print("\n[3/3] Training baseline for user_123...")
try:
    response = requests.post(
        f"{BASE_URL}/baseline/train",
        json={
            "user_id": "user_123",
            "days": 7,
            "data": data
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Baseline trained successfully!")
        print(f"\n📊 Your Normal Health Baselines:")
        print("-" * 70)
        
        baselines = result.get('baselines', {})
        for metric, stats in baselines.items():
            metric_name = metric.replace('_', ' ').title()
            mean = stats.get('mean', 0)
            lower = stats.get('recommended_lower', 0)
            upper = stats.get('recommended_upper', 0)
            print(f"  {metric_name:<20}: {mean:.1f} (normal range: {lower:.1f} - {upper:.1f})")
        
        print("-" * 70)
        
        # Show model performance
        perf = result.get('model_performance', {})
        print(f"\n🎯 Model Training Accuracy:")
        for metric, scores in perf.items():
            metric_name = metric.replace('_', ' ').title()
            val_r2 = scores.get('val_r2', 0)
            print(f"  {metric_name:<20}: {val_r2*100:.1f}% accurate")
        
    else:
        print(f"❌ Training failed: {response.text}")
        exit(1)
        
except Exception as e:
    print(f"❌ Error training baseline: {e}")
    exit(1)

# Step 4: Test anomaly detection with abnormal data
print("\n" + "=" * 70)
print("Testing Anomaly Detection")
print("=" * 70)

print("\n[4/5] Creating abnormal health data (HR=145, Stress=95)...")
anomaly_data = [{
    "timestamp": "2026-01-09T14:00:00",
    "heart_rate": 145,  # Way above normal (70-82)
    "steps": 5,  # Very low
    "sleep_quality": 0,
    "stress_level": 95,  # Way above normal (30-40)
    "calories": 15  # Very low
}]

print("\n[5/5] Detecting anomalies...")
try:
    response = requests.post(
        f"{BASE_URL}/anomaly/detect",
        json={
            "user_id": "user_123",
            "data": anomaly_data
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Anomaly detection complete!")
        print(f"\n⚠️  Detected Anomalies:")
        print(f"  Total: {result['total_anomalies']}")
        print(f"  High Risk: {result['high_risk_count']} 🚨")
        print(f"  Medium Risk: {result['medium_risk_count']} ⚠️")
        print(f"  Low Risk: {result['low_risk_count']} ℹ️")
        
        if result['alerts']:
            print(f"\n📋 Detailed Alerts:")
            print("-" * 70)
            for alert in result['alerts']:
                metric = alert['metric']
                current = alert['current_value']
                normal = alert['your_normal']
                deviation = alert['deviation_pct']
                risk = alert['risk_level']
                confidence = alert['confidence']
                
                risk_icon = "🚨" if risk == "High" else "⚠️" if risk == "Medium" else "ℹ️"
                
                print(f"\n  {risk_icon} {metric} [{risk} Risk - {confidence*100:.0f}% confidence]")
                print(f"     Current Value: {current:.1f}")
                print(f"     Your Normal:   {normal:.1f}")
                print(f"     Deviation:     {deviation:+.1f}%")
    else:
        print(f"❌ Detection failed: {response.text}")
        
except Exception as e:
    print(f"❌ Error detecting anomalies: {e}")

print("\n" + "=" * 70)
print("✅ Setup Complete! Now you can use:")
print("   python mcp_client_groq.py")
print("=" * 70)
