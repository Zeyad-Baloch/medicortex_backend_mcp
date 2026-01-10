"""
Machine Learning models for baseline learning and anomaly detection.
"""
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from app.config import settings


class BaselineLearner:
    """Learns personalized health baselines using RandomForest models."""
    
    def __init__(self, data: pd.DataFrame, user_id: str):
        self.data = data
        self.user_id = user_id
        self.models = {}
        self.scalers = {}
        self.baselines = {}
        self.metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

    def learn(self) -> Dict:
        """Train models and compute baselines for all metrics."""
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
        """Train RandomForest model for a specific metric."""
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

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=settings.RANDOM_STATE
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        n_estimators = 50 if metric in ['steps', 'calories'] else settings.N_ESTIMATORS
        max_depth = 8 if metric in ['steps', 'calories'] else settings.MAX_DEPTH

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        self.models[metric] = model
        self.scalers[metric] = scaler

        return model.score(X_train_scaled, y_train), model.score(X_val_scaled, y_val)

    def _compute_statistics(self):
        """Compute statistical baselines for each metric."""
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

            # Ensure valid ranges
            if metric in ['steps', 'calories', 'heart_rate']:
                stats['recommended_lower'] = max(0, stats['recommended_lower'])
            if metric in ['sleep_quality', 'stress_level']:
                stats['recommended_lower'] = max(0, min(100, stats['recommended_lower']))
                stats['recommended_upper'] = max(0, min(100, stats['recommended_upper']))

            self.baselines[metric] = stats


class PersonalizedZScoreDetector:
    """Detects anomalies using personalized Z-score method."""
    
    def __init__(self, baselines: Dict):
        self.baselines = baselines
        self.metrics = list(baselines.keys())

    def detect(self, data: pd.DataFrame, z_threshold=2.5) -> pd.DataFrame:
        """Detect anomalies using Z-scores."""
        z_scores = pd.DataFrame(index=data.index)
        anomalies = pd.DataFrame(index=data.index)

        for metric in self.metrics:
            if metric in data.columns:
                mean = self.baselines[metric]['mean']
                std = self.baselines[metric]['std']

                z = (data[metric] - mean) / std
                z_scores[f'{metric}_zscore'] = z

                # Modified Z-score for robustness
                median = z.median()
                mad = np.median(np.abs(z - median))
                modified_z = 0.6745 * (z - median) / mad

                is_anomaly = np.abs(modified_z) > z_threshold
                confidence = np.minimum(np.abs(z) / z_threshold, 2.0) / 2.0

                anomalies[f'{metric}_anomaly'] = is_anomaly
                anomalies[f'{metric}_confidence'] = confidence

        return anomalies


class IsolationForestDetector:
    """Detects anomalies using Isolation Forest algorithm."""
    
    def __init__(self, baselines: Dict):
        self.baselines = baselines
        self.metrics = list(baselines.keys())
        self.models = {}
        self.scalers = {}

    def train(self, data: pd.DataFrame):
        """Train Isolation Forest models."""
        for metric in self.metrics:
            if metric in data.columns:
                X = data[[metric]].values

                scaler = StandardScaler()
                scaler.mean_ = np.array([self.baselines[metric]['mean']])
                scaler.scale_ = np.array([self.baselines[metric]['std']])
                X_scaled = scaler.transform(X)

                model = IsolationForest(
                    contamination=settings.CONTAMINATION,
                    random_state=settings.RANDOM_STATE,
                    n_estimators=100
                )
                model.fit(X_scaled)

                self.models[metric] = model
                self.scalers[metric] = scaler

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using trained models."""
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
    """Combines multiple detection methods for robust anomaly detection."""
    
    def __init__(self, baselines: Dict):
        self.baselines = baselines
        self.metrics = list(baselines.keys())

    def calculate_consensus(self, zscore_results: pd.DataFrame, if_results: pd.DataFrame) -> pd.DataFrame:
        """Calculate consensus from multiple detection methods."""
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
        """Assess risk level for detected anomalies."""
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
