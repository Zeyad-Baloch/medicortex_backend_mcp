"""
Data preprocessing and feature engineering for health data.
Contains HealthDataPreprocessor and FeatureEngineer classes.
"""
import pandas as pd


class HealthDataPreprocessor:
    """Cleans and preprocesses raw health data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

    def process(self) -> pd.DataFrame:
        """Run all preprocessing steps."""
        self._handle_missing_values()
        self._remove_outliers()
        self._apply_smoothing()
        return self.data

    def _handle_missing_values(self):
        """Fill missing values using linear interpolation."""
        for metric in self.metrics:
            if self.data[metric].isna().sum() > 0:
                self.data[metric] = self.data[metric].interpolate(method='linear')

    def _remove_outliers(self, threshold=1.5):
        """Remove outliers using IQR method."""
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
        """Apply rolling window smoothing."""
        for metric in self.metrics:
            self.data[f'{metric}_smoothed'] = (
                self.data[metric].rolling(window=window, center=True).mean()
            )
            self.data[f'{metric}_smoothed'].fillna(self.data[metric], inplace=True)


class FeatureEngineer:
    """Generates features from preprocessed health data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.metrics = ['heart_rate', 'steps', 'sleep_quality', 'stress_level', 'calories']

    def engineer(self) -> pd.DataFrame:
        """Generate all features."""
        self._extract_temporal_features()
        self._compute_rolling_statistics()
        self.data.fillna(method='bfill', inplace=True)
        return self.data

    def _extract_temporal_features(self):
        """Extract time-based features."""
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)

    def _compute_rolling_statistics(self, windows=[6, 12, 24]):
        """Compute rolling window statistics."""
        for metric in self.metrics:
            for window in windows:
                self.data[f'{metric}_rolling_mean_{window}h'] = (
                    self.data[f'{metric}_smoothed'].rolling(window=window).mean()
                )
                self.data[f'{metric}_rolling_std_{window}h'] = (
                    self.data[f'{metric}_smoothed'].rolling(window=window).std()
                )
