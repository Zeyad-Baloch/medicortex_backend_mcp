"""
SQLite storage implementation for persistent data.
Stores baselines and ML models in a local database file.
"""
import sqlite3
import json
from typing import Dict, Optional
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
