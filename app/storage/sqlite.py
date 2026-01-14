"""
SQLite storage implementation for persistent data.
Stores baselines and ML models in a local database file.
"""
import sqlite3
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.storage.base import BaseStorage
from app.config import settings


class SQLiteStorage(BaseStorage):
    """SQLite storage - persistent across restarts."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._init_database()

    def _init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baselines (
                user_id TEXT PRIMARY KEY,
                baseline_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                user_id TEXT PRIMARY KEY,
                model_data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # OCR Reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_reports (
                report_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                report_type TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES baselines(user_id)
            )
        ''')

        # ============================================================================
        # NEW - HEALTH DATA HISTORY TABLE
        # ============================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_data_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                heart_rate REAL,
                steps INTEGER,
                sleep_quality REAL,
                stress_level REAL,
                calories REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES baselines(user_id)
            )
        ''')

        # ============================================================================
        # NEW - ANOMALY HISTORY TABLE
        # ============================================================================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metric TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                deviation_pct REAL NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES baselines(user_id)
            )
        ''')

        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ocr_user_id 
            ON ocr_reports(user_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_health_user_timestamp 
            ON health_data_history(user_id, timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_anomaly_user_timestamp 
            ON anomaly_history(user_id, timestamp)
        ''')

        conn.commit()
        conn.close()

    async def save_baseline(self, user_id: str, baseline: Dict) -> bool:
        """Save baseline to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            baseline_json = json.dumps(baseline)

            cursor.execute('''
                INSERT OR REPLACE INTO baselines (user_id, baseline_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, baseline_json))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving baseline: {e}")
            return False

    async def get_baseline(self, user_id: str) -> Optional[Dict]:
        """Get baseline from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT baseline_data FROM baselines WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()

            conn.close()

            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            print(f"Error getting baseline: {e}")
            return None

    async def save_models(self, user_id: str, models: Dict, scalers: Dict) -> bool:
        """Save models to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serialize models and scalers
            model_data = self.serialize_models(models, scalers)

            cursor.execute('''
                INSERT OR REPLACE INTO models (user_id, model_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, model_data))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    async def get_models(self, user_id: str) -> Optional[tuple]:
        """Get models from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT model_data FROM models WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()

            conn.close()

            if row:
                return self.deserialize_models(row[0])
            return None
        except Exception as e:
            print(f"Error getting models: {e}")
            return None

    async def delete_user_data(self, user_id: str) -> bool:
        """Delete user data from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM baselines WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM models WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM ocr_reports WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM health_data_history WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM anomaly_history WHERE user_id = ?', (user_id,))

            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()
            return deleted
        except Exception as e:
            print(f"Error deleting user data: {e}")
            return False

    async def list_users(self) -> list:
        """List all users."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT DISTINCT user_id FROM baselines')
            rows = cursor.fetchall()

            conn.close()

            return [row[0] for row in rows]
        except Exception as e:
            print(f"Error listing users: {e}")
            return []

    # ============================================================================
    # OCR METHODS
    # ============================================================================

    async def save_ocr_report(self, user_id: str, report_id: str, report_data: Dict) -> bool:
        """Save OCR report to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract data
            extracted_text = report_data.get('extracted_text', '')
            confidence = report_data.get('confidence', 0.0)
            report_type = report_data.get('report_type', 'general_report')
            keywords = json.dumps(report_data.get('keywords', []))
            created_at = report_data.get('created_at')

            cursor.execute('''
                INSERT INTO ocr_reports 
                (report_id, user_id, extracted_text, confidence, report_type, keywords, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (report_id, user_id, extracted_text, confidence, report_type, keywords, created_at))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving OCR report: {e}")
            return False

    async def get_ocr_report(self, report_id: str) -> Optional[Dict]:
        """Get OCR report from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT report_id, user_id, extracted_text, confidence, 
                       report_type, keywords, created_at
                FROM ocr_reports 
                WHERE report_id = ?
            ''', (report_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'report_id': row[0],
                    'user_id': row[1],
                    'extracted_text': row[2],
                    'confidence': row[3],
                    'report_type': row[4],
                    'keywords': json.loads(row[5]) if row[5] else [],
                    'created_at': row[6]
                }
            return None
        except Exception as e:
            print(f"Error getting OCR report: {e}")
            return None

    async def get_user_ocr_reports(self, user_id: str) -> List[Dict]:
        """Get all OCR reports for a user from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT report_id, user_id, extracted_text, confidence, 
                       report_type, keywords, created_at
                FROM ocr_reports 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))

            rows = cursor.fetchall()
            conn.close()

            reports = []
            for row in rows:
                reports.append({
                    'report_id': row[0],
                    'user_id': row[1],
                    'extracted_text': row[2],
                    'confidence': row[3],
                    'report_type': row[4],
                    'keywords': json.loads(row[5]) if row[5] else [],
                    'created_at': row[6]
                })

            return reports
        except Exception as e:
            print(f"Error getting user OCR reports: {e}")
            return []

    async def delete_ocr_report(self, report_id: str) -> bool:
        """Delete OCR report from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM ocr_reports WHERE report_id = ?', (report_id,))
            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()
            return deleted
        except Exception as e:
            print(f"Error deleting OCR report: {e}")
            return False

    # ============================================================================
    # NEW - HEALTH DATA HISTORY METHODS
    # ============================================================================

    async def save_health_data(self, user_id: str, data_points: List[Dict]) -> bool:
        """Save multiple health data points to history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for point in data_points:
                cursor.execute('''
                    INSERT INTO health_data_history 
                    (user_id, timestamp, heart_rate, steps, sleep_quality, stress_level, calories)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    point.get('timestamp'),
                    point.get('heart_rate'),
                    point.get('steps'),
                    point.get('sleep_quality'),
                    point.get('stress_level'),
                    point.get('calories')
                ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving health data: {e}")
            return False

    async def get_health_data_range(self, user_id: str, start_date: str = None, end_date: str = None,
                                    limit: int = None) -> List[Dict]:
        """Get health data for a user within a date range."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = '''
                SELECT timestamp, heart_rate, steps, sleep_quality, stress_level, calories
                FROM health_data_history
                WHERE user_id = ?
            '''
            params = [user_id]

            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date)

            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date)

            query += ' ORDER BY timestamp DESC'

            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            data = []
            for row in rows:
                data.append({
                    'timestamp': row[0],
                    'heart_rate': row[1],
                    'steps': row[2],
                    'sleep_quality': row[3],
                    'stress_level': row[4],
                    'calories': row[5]
                })

            return data
        except Exception as e:
            print(f"Error getting health data range: {e}")
            return []

    async def get_latest_health_data(self, user_id: str) -> Optional[Dict]:
        """Get the most recent health data point for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT timestamp, heart_rate, steps, sleep_quality, stress_level, calories
                FROM health_data_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (user_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'timestamp': row[0],
                    'heart_rate': row[1],
                    'steps': row[2],
                    'sleep_quality': row[3],
                    'stress_level': row[4],
                    'calories': row[5]
                }
            return None
        except Exception as e:
            print(f"Error getting latest health data: {e}")
            return None

    async def get_health_data_stats(self, user_id: str, days: int = 30) -> Dict:
        """Get aggregated statistics for health data over a time period."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            cursor.execute('''
                SELECT 
                    AVG(heart_rate) as avg_hr,
                    MIN(heart_rate) as min_hr,
                    MAX(heart_rate) as max_hr,
                    AVG(steps) as avg_steps,
                    SUM(steps) as total_steps,
                    AVG(sleep_quality) as avg_sleep,
                    AVG(stress_level) as avg_stress,
                    AVG(calories) as avg_calories,
                    COUNT(*) as total_readings
                FROM health_data_history
                WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
            ''', (user_id, start_date.isoformat(), end_date.isoformat()))

            row = cursor.fetchone()
            conn.close()

            if row and row[8] > 0:  # total_readings > 0
                return {
                    'heart_rate': {
                        'avg': round(row[0], 1) if row[0] else 0,
                        'min': round(row[1], 1) if row[1] else 0,
                        'max': round(row[2], 1) if row[2] else 0
                    },
                    'steps': {
                        'avg': round(row[3], 0) if row[3] else 0,
                        'total': int(row[4]) if row[4] else 0
                    },
                    'sleep_quality': {
                        'avg': round(row[5], 1) if row[5] else 0
                    },
                    'stress_level': {
                        'avg': round(row[6], 1) if row[6] else 0
                    },
                    'calories': {
                        'avg': round(row[7], 1) if row[7] else 0
                    },
                    'total_readings': int(row[8]),
                    'days': days
                }

            return {
                'total_readings': 0,
                'days': days,
                'message': 'No data available for this period'
            }
        except Exception as e:
            print(f"Error getting health data stats: {e}")
            return {}

    # ============================================================================
    # NEW - ANOMALY HISTORY METHODS
    # ============================================================================

    async def save_anomalies(self, user_id: str, anomalies: List[Dict]) -> bool:
        """Save detected anomalies to history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for anomaly in anomalies:
                cursor.execute('''
                    INSERT INTO anomaly_history 
                    (user_id, timestamp, metric, current_value, baseline_value, 
                     deviation_pct, risk_level, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    anomaly.get('timestamp'),
                    anomaly.get('metric'),
                    anomaly.get('current_value'),
                    anomaly.get('your_normal'),
                    anomaly.get('deviation_pct'),
                    anomaly.get('risk_level'),
                    anomaly.get('confidence')
                ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving anomalies: {e}")
            return False

    async def get_anomaly_history(self, user_id: str, days: int = 30, risk_level: str = None) -> List[Dict]:
        """Get anomaly history for a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            query = '''
                SELECT timestamp, metric, current_value, baseline_value, 
                       deviation_pct, risk_level, confidence, detected_at
                FROM anomaly_history
                WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
            '''
            params = [user_id, start_date.isoformat(), end_date.isoformat()]

            if risk_level:
                query += ' AND risk_level = ?'
                params.append(risk_level)

            query += ' ORDER BY timestamp DESC'

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            anomalies = []
            for row in rows:
                anomalies.append({
                    'timestamp': row[0],
                    'metric': row[1],
                    'current_value': row[2],
                    'baseline_value': row[3],
                    'deviation_pct': row[4],
                    'risk_level': row[5],
                    'confidence': row[6],
                    'detected_at': row[7]
                })

            return anomalies
        except Exception as e:
            print(f"Error getting anomaly history: {e}")
            return []

    async def get_anomaly_counts(self, user_id: str, days: int = 30) -> Dict:
        """Get count of anomalies by risk level."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            cursor.execute('''
                SELECT risk_level, COUNT(*) as count
                FROM anomaly_history
                WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
                GROUP BY risk_level
            ''', (user_id, start_date.isoformat(), end_date.isoformat()))

            rows = cursor.fetchall()
            conn.close()

            counts = {
                'High': 0,
                'Medium': 0,
                'Low': 0,
                'total': 0
            }

            for row in rows:
                risk_level = row[0]
                count = row[1]
                counts[risk_level] = count
                counts['total'] += count

            counts['days'] = days
            return counts
        except Exception as e:
            print(f"Error getting anomaly counts: {e}")
            return {'High': 0, 'Medium': 0, 'Low': 0, 'total': 0, 'days': days}