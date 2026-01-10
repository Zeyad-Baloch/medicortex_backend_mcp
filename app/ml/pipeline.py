"""
ML Pipeline for baseline training and anomaly detection.
Chains preprocessing, feature engineering, and ML models.
"""
import pandas as pd
from typing import Dict, List

from app.ml.preprocessing import HealthDataPreprocessor, FeatureEngineer
from app.ml.detectors import (
    BaselineLearner,
    PersonalizedZScoreDetector,
    IsolationForestDetector,
    ConsensusDetector
)


class MLPipeline:
    """Orchestrates the complete ML workflow."""
    
    def train_baseline(self, user_id: str, data: List[Dict]) -> Dict:
        """
        Complete pipeline for training a user's baseline.
        
        Args:
            user_id: Unique user identifier
            data: List of health data points
            
        Returns:
            Dict with baselines and model performance
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Step 1: Preprocessing
        preprocessor = HealthDataPreprocessor(df)
        processed = preprocessor.process()
        
        # Step 2: Feature Engineering
        engineer = FeatureEngineer(processed)
        featured = engineer.engineer()
        
        # Step 3: Baseline Learning
        learner = BaselineLearner(featured, user_id)
        result = learner.learn()
        
        # Return result with trained models
        return {
            'result': result,
            'models': learner.models,
            'scalers': learner.scalers
        }
    
    def detect_anomalies(self, baseline: Dict, data: List[Dict]) -> Dict:
        """
        Complete pipeline for detecting anomalies.
        
        Args:
            baseline: User's learned baseline
            data: Current health data points
            
        Returns:
            Dict with detected anomalies and risk assessments
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Step 1: Z-Score Detection
        zscore_detector = PersonalizedZScoreDetector(baseline)
        zscore_results = zscore_detector.detect(df)
        
        # Step 2: Isolation Forest Detection
        if_detector = IsolationForestDetector(baseline)
        if_detector.train(df)
        if_results = if_detector.detect(df)
        
        # Step 3: Consensus & Risk Assessment
        consensus_detector = ConsensusDetector(baseline)
        consensus = consensus_detector.calculate_consensus(zscore_results, if_results)
        risk = consensus_detector.assess_risk(consensus, df)
        
        # Step 4: Generate Alerts
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
        
        # Count risk levels
        high_risk = len([a for a in alerts if a['risk_level'] == 'High'])
        medium_risk = len([a for a in alerts if a['risk_level'] == 'Medium'])
        low_risk = len([a for a in alerts if a['risk_level'] == 'Low'])
        
        return {
            'total_anomalies': len(alerts),
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'alerts': alerts
        }
