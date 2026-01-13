"""
In-memory storage implementation (no persistence).
Good for development and testing.
"""
from typing import Dict, Optional, List
from app.storage.base import BaseStorage


class MemoryStorage(BaseStorage):
    """In-memory storage - data lost on restart."""

    def __init__(self):
        self.baselines = {}
        self.models = {}
        self.scalers = {}
        # NEW - OCR storage
        self.ocr_reports = {}  # report_id -> report_data
        self.user_reports = {}  # user_id -> [report_ids]

    async def save_baseline(self, user_id: str, baseline: Dict) -> bool:
        """Save baseline to memory."""
        self.baselines[user_id] = baseline
        return True

    async def get_baseline(self, user_id: str) -> Optional[Dict]:
        """Get baseline from memory."""
        return self.baselines.get(user_id)

    async def save_models(self, user_id: str, models: Dict, scalers: Dict) -> bool:
        """Save models to memory."""
        self.models[user_id] = models
        self.scalers[user_id] = scalers
        return True

    async def get_models(self, user_id: str) -> Optional[tuple]:
        """Get models from memory."""
        models = self.models.get(user_id)
        scalers = self.scalers.get(user_id)
        if models and scalers:
            return models, scalers
        return None

    async def delete_user_data(self, user_id: str) -> bool:
        """Delete user data from memory."""
        deleted = False
        if user_id in self.baselines:
            del self.baselines[user_id]
            deleted = True
        if user_id in self.models:
            del self.models[user_id]
        if user_id in self.scalers:
            del self.scalers[user_id]
        # Delete OCR reports
        if user_id in self.user_reports:
            for report_id in self.user_reports[user_id]:
                if report_id in self.ocr_reports:
                    del self.ocr_reports[report_id]
            del self.user_reports[user_id]
        return deleted

    async def list_users(self) -> list:
        """List all users."""
        return list(self.baselines.keys())

    # ============================================================================
    # NEW - OCR METHODS
    # ============================================================================

    async def save_ocr_report(self, user_id: str, report_id: str, report_data: Dict) -> bool:
        """Save OCR report to memory."""
        try:
            # Store report
            self.ocr_reports[report_id] = report_data

            # Add to user's report list
            if user_id not in self.user_reports:
                self.user_reports[user_id] = []

            if report_id not in self.user_reports[user_id]:
                self.user_reports[user_id].append(report_id)

            return True
        except Exception as e:
            print(f"Error saving OCR report: {e}")
            return False

    async def get_ocr_report(self, report_id: str) -> Optional[Dict]:
        """Get OCR report from memory."""
        return self.ocr_reports.get(report_id)

    async def get_user_ocr_reports(self, user_id: str) -> List[Dict]:
        """Get all OCR reports for a user from memory."""
        report_ids = self.user_reports.get(user_id, [])
        reports = []

        for report_id in report_ids:
            report = self.ocr_reports.get(report_id)
            if report:
                reports.append(report)

        # Sort by created_at (newest first)
        reports.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return reports

    async def delete_ocr_report(self, report_id: str) -> bool:
        """Delete OCR report from memory."""
        try:
            if report_id not in self.ocr_reports:
                return False

            # Get report to find user_id
            report = self.ocr_reports[report_id]
            user_id = report.get('user_id')

            # Delete from reports
            del self.ocr_reports[report_id]

            # Remove from user's report list
            if user_id and user_id in self.user_reports:
                if report_id in self.user_reports[user_id]:
                    self.user_reports[user_id].remove(report_id)

            return True
        except Exception as e:
            print(f"Error deleting OCR report: {e}")
            return False