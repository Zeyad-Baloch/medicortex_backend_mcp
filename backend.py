

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


app = FastAPI(
    title="MediCortex ML Backend",
    description="Baseline Learning and Anomaly Detection for Health Monitoring",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class HealthDataPoint(BaseModel):
    timestamp: str
    heart_rate: float
    steps: int
    sleep_quality: float
    stress_level: float
    calories: float


class BaselineTrainRequest(BaseModel):
    user_id: str
    data: List[Dict[str, Any]]
    days: int = 7


class AnomalyDetectRequest(BaseModel):
    user_id: str
    data: List[Dict[str, Any]]




class StorageManager:


    def __init__(self):
        self.baselines = {}
        self.models = {}
        self.scalers = {}

    def save_baseline(self, user_id: str, baseline: Dict):
        self.baselines[user_id] = baseline

    def get_baseline(self, user_id: str) -> Optional[Dict]:
        return self.baselines.get(user_id)

    def save_models(self, user_id: str, models: Dict, scalers: Dict):
        self.models[user_id] = models
        self.scalers[user_id] = scalers

    def get_models(self, user_id: str):
        return self.models.get(user_id), self.scalers.get(user_id)


storage = StorageManager()




class HealthDataPreprocessor:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

    def process(self) -> pd.DataFrame:
        self._handle_missing_values()
        self._remove_outliers()
        self._apply_smoothing()
        return self.data

    def _handle_missing_values(self):
        for metric in self.metrics:
            if self.data[metric].isna().sum() > 0:
                self.data[metric] = self.data[metric].interpolate(method='linear')

    def _remove_outliers(self, threshold=1.5):
        for metric in self.metrics:
            Q1 = self.data[metric].quantile(0.25)
            Q3 = self.data[metric].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = ((self.data[metric] < lower_bound) | (self.data[metric] > upper_bound))
            if outliers.sum() > 0:
                self.data.loc[outliers, metric] = self.data[metric].median()

    def _apply_smoothing(self, window=3):
        for metric in self.metrics:
            self.data[f'{metric}_smoothed'] = (
                self.data[metric].rolling(window=window, center=True).mean()
            )
            self.data[f'{metric}_smoothed'].fillna(self.data[metric], inplace=True)


class FeatureEngineer:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

    def engineer(self) -> pd.DataFrame:
        self._extract_temporal_features()
        self._compute_rolling_statistics()
        self.data.fillna(method='bfill', inplace=True)
        return self.data

    def _extract_temporal_features(self):
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)

    def _compute_rolling_statistics(self, windows=[6, 12, 24]):
        for metric in self.metrics:
            for window in windows:
                self.data[f'{metric}_rolling_mean_{window}h'] = (
                    self.data[f'{metric}_smoothed'].rolling(window=window).mean()
                )
                self.data[f'{metric}_rolling_std_{window}h'] = (
                    self.data[f'{metric}_smoothed'].rolling(window=window).std()
                )


class BaselineLearner:

    def __init__(self, data: pd.DataFrame, user_id: str):
        self.data = data
        self.user_id = user_id
        self.models = {}
        self.scalers = {}
        self.baselines = {}
        self.metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

    def learn(self) -> Dict:
        results = {}

        for metric in self.metrics:
            train_r2, val_r2 = self._train_model(metric)
            results[metric] = {'train_r2': train_r2, 'val_r2': val_r2}

        self._compute_statistics()

        return {
            'user_id': self.user_id,
            'baselines': self.baselines,
            'model_performance': results,
            'status': 'success'
        }

    def _train_model(self, metric: str):

        feature_cols = [
            'hour', 'day_of_week', 'is_weekend',
            f'{metric}_rolling_mean_6h', f'{metric}_rolling_std_6h',
            f'{metric}_rolling_mean_12h', f'{metric}_rolling_std_12h',
            f'{metric}_rolling_mean_24h', f'{metric}_rolling_std_24h'
        ]

        X = self.data[feature_cols].copy()
        y = self.data[f'{metric}_smoothed'].copy()

        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        n_estimators = 50 if metric in ['steps', 'calories'] else 100
        max_depth = 8 if metric in ['steps', 'calories'] else 10

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        self.models[metric] = model
        self.scalers[metric] = scaler

        return model.score(X_train_scaled, y_train), model.score(X_val_scaled, y_val)

    def _compute_statistics(self):

        for metric in self.metrics:
            smoothed_col = f'{metric}_smoothed'

            # Context-aware filtering
            if metric == 'sleep_quality':
                mask = (self.data['hour'] >= 22) | (self.data['hour'] <= 6)
                metric_data = self.data.loc[mask, smoothed_col]
            elif metric in ['steps', 'calories']:
                mask = (self.data['hour'] >= 6) & (self.data['hour'] <= 22)
                metric_data = self.data.loc[mask, smoothed_col]
            else:
                metric_data = self.data[smoothed_col]

            stats = {
                'mean': float(metric_data.mean()),
                'std': float(metric_data.std()),
                'median': float(metric_data.median()),
                'p10': float(metric_data.quantile(0.10)),
                'p90': float(metric_data.quantile(0.90)),
                'recommended_lower': float(metric_data.quantile(0.10)) if metric in ['steps', 'calories'] else float(
                    metric_data.mean() - 2 * metric_data.std()),
                'recommended_upper': float(metric_data.quantile(0.90)) if metric in ['steps', 'calories'] else float(
                    metric_data.mean() + 2 * metric_data.std()),
                'range_type': 'percentile' if metric in ['steps', 'calories'] else 'statistical'
            }


            if metric in ['steps', 'calories', 'heart_rate']:
                stats['recommended_lower'] = max(0, stats['recommended_lower'])
            if metric in ['sleep_quality', 'stress_level']:
                stats['recommended_lower'] = max(0, min(100, stats['recommended_lower']))
                stats['recommended_upper'] = max(0, min(100, stats['recommended_upper']))

            self.baselines[metric] = stats


#Anomaly Detection

class PersonalizedZScoreDetector:


    def __init__(self, baselines: Dict):
        self.baselines = baselines
        self.metrics = list(baselines.keys())

    def detect(self, data: pd.DataFrame, z_threshold=2.5) -> pd.DataFrame:

        z_scores = pd.DataFrame(index=data.index)
        anomalies = pd.DataFrame(index=data.index)

        for metric in self.metrics:
            if metric in data.columns:
                mean = self.baselines[metric]['mean']
                std = self.baselines[metric]['std']

                z = (data[metric] - mean) / std
                z_scores[f'{metric}_zscore'] = z


                median = z.median()
                mad = np.median(np.abs(z - median))
                modified_z = 0.6745 * (z - median) / mad

                is_anomaly = np.abs(modified_z) > z_threshold
                confidence = np.minimum(np.abs(z) / z_threshold, 2.0) / 2.0

                anomalies[f'{metric}_anomaly'] = is_anomaly
                anomalies[f'{metric}_confidence'] = confidence

        return anomalies


class IsolationForestDetector:


    def __init__(self, baselines: Dict):
        self.baselines = baselines
        self.metrics = list(baselines.keys())
        self.models = {}
        self.scalers = {}

    def train(self, data: pd.DataFrame):

        for metric in self.metrics:
            if metric in data.columns:
                X = data[[metric]].values

                scaler = StandardScaler()
                scaler.mean_ = np.array([self.baselines[metric]['mean']])
                scaler.scale_ = np.array([self.baselines[metric]['std']])
                X_scaled = scaler.transform(X)

                model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
                model.fit(X_scaled)

                self.models[metric] = model
                self.scalers[metric] = scaler

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:

        anomalies = pd.DataFrame(index=data.index)

        for metric in self.metrics:
            if metric in data.columns and metric in self.models:
                X = data[[metric]].values
                X_scaled = self.scalers[metric].transform(X)

                predictions = self.models[metric].predict(X_scaled)
                scores = self.models[metric].score_samples(X_scaled)
                confidence = 1 / (1 + np.exp(scores))

                anomalies[f'{metric}_if_anomaly'] = (predictions == -1)
                anomalies[f'{metric}_if_confidence'] = confidence

        return anomalies


class ConsensusDetector:


    def __init__(self, baselines: Dict):
        self.baselines = baselines
        self.metrics = list(baselines.keys())

    def calculate_consensus(self, zscore_results: pd.DataFrame, if_results: pd.DataFrame) -> pd.DataFrame:

        consensus = pd.DataFrame(index=zscore_results.index)

        for metric in self.metrics:
            votes = []
            confidences = []

            if f'{metric}_anomaly' in zscore_results.columns:
                votes.append(zscore_results[f'{metric}_anomaly'].astype(int))
                confidences.append(zscore_results[f'{metric}_confidence'])

            if f'{metric}_if_anomaly' in if_results.columns:
                votes.append(if_results[f'{metric}_if_anomaly'].astype(int))
                confidences.append(if_results[f'{metric}_if_confidence'])

            if votes:
                consensus[f'{metric}_final_anomaly'] = (
                        (np.sum(votes, axis=0) >= 1) |  # At least ONE method detects
                        (np.mean(confidences, axis=0) >= 0.75)  # OR decent confidence
                )
                consensus[f'{metric}_avg_confidence'] = np.mean(confidences, axis=0)

        return consensus

    def assess_risk(self, consensus: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:

        risk = pd.DataFrame(index=consensus.index)

        for metric in self.metrics:
            if f'{metric}_final_anomaly' in consensus.columns:
                anomaly_mask = consensus[f'{metric}_final_anomaly']
                confidence = consensus[f'{metric}_avg_confidence']

                mean = self.baselines[metric]['mean']
                deviation_pct = np.abs((data[metric] - mean) / mean * 100)

                risk_labels = np.array(['Normal'] * len(data), dtype=object)

                for idx in consensus.index[anomaly_mask]:
                    conf = confidence.loc[idx]
                    dev = deviation_pct.loc[idx]

                    if conf >= 0.85 or dev >= 30:
                        risk_labels[idx] = 'High'
                    elif conf >= 0.70 or dev >= 20:
                        risk_labels[idx] = 'Medium'
                    else:
                        risk_labels[idx] = 'Low'

                risk[f'{metric}_risk_level'] = risk_labels

        return risk




@app.get("/")
async def root():

    return {
        "service": "MediCortex ML Backend",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/baseline/train")
async def train_baseline(request: BaselineTrainRequest):

    try:

        df = pd.DataFrame(request.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        preprocessor = HealthDataPreprocessor(df)
        processed = preprocessor.process()

        fe_engineer = FeatureEngineer(processed)
        featured = fe_engineer.engineer()

        learner = BaselineLearner(featured, request.user_id)
        result = learner.learn()

        storage.save_baseline(request.user_id, result['baselines'])
        storage.save_models(request.user_id, learner.models, learner.scalers)

        return {
            "status": "success",
            "user_id": request.user_id,
            "baselines": result['baselines'],
            "model_performance": result['model_performance'],
            "message": f"Successfully learned baselines for {request.user_id} using {len(df)} data points"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baseline/{user_id}")
async def get_baseline(user_id: str):

    baseline = storage.get_baseline(user_id)

    if not baseline:
        raise HTTPException(status_code=404, detail=f"No baseline found for user {user_id}")

    return {
        "status": "success",
        "user_id": user_id,
        "baselines": baseline
    }


@app.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyDetectRequest):

    try:
        # Get user's baseline
        baseline = storage.get_baseline(request.user_id)
        if not baseline:
            raise HTTPException(status_code=404,
                                detail=f"No baseline found for user {request.user_id}. Train baseline first.")


        df = pd.DataFrame(request.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        zscore_detector = PersonalizedZScoreDetector(baseline)
        zscore_results = zscore_detector.detect(df)

        if_detector = IsolationForestDetector(baseline)
        if_detector.train(df)
        if_results = if_detector.detect(df)


        consensus_detector = ConsensusDetector(baseline)
        consensus = consensus_detector.calculate_consensus(zscore_results, if_results)
        risk = consensus_detector.assess_risk(consensus, df)


        alerts = []
        for idx in df.index:
            for metric in baseline.keys():
                if f'{metric}_final_anomaly' in consensus.columns and consensus.loc[idx, f'{metric}_final_anomaly']:
                    current = float(df.loc[idx, metric])
                    normal = baseline[metric]['mean']
                    deviation_pct = ((current - normal) / normal) * 100

                    alerts.append({
                        'timestamp': str(df.loc[idx, 'timestamp']),
                        'metric': metric.replace('_', ' ').title(),
                        'current_value': current,
                        'your_normal': normal,
                        'deviation_pct': deviation_pct,
                        'risk_level': str(risk.loc[idx, f'{metric}_risk_level']),
                        'confidence': float(consensus.loc[idx, f'{metric}_avg_confidence'])
                    })


        high_risk = len([a for a in alerts if a['risk_level'] == 'High'])
        medium_risk = len([a for a in alerts if a['risk_level'] == 'Medium'])
        low_risk = len([a for a in alerts if a['risk_level'] == 'Low'])

        return {
            "status": "success",
            "user_id": request.user_id,
            "total_anomalies": len(alerts),
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "alerts": alerts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "users_with_baselines": len(storage.baselines),
        "total_models": sum(len(models) for models in storage.models.values())
    }


if __name__ == "__main__":
    import uvicorn

    print("\nStarting MediCortex ML Backend...")
    print("   Server: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)