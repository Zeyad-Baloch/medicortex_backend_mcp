"""
OCR Service - Medical report text extraction using OCR.space API.
NOW WITH SMART PARSING: Extracts structured health values!
"""
import requests
import uuid
from typing import Dict, Optional, Tuple
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from app.storage.base import BaseStorage
from app.config import settings
from app.ml.text_parser import MedicalTextParser  # NEW


class OCRService:
    """Service for OCR text extraction from medical reports."""

    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.api_key = settings.OCR_SPACE_API_KEY
        self.api_url = "https://api.ocr.space/parse/image"
        self.parser = MedicalTextParser()  # NEW - Initialize parser
        self.medical_keywords = [
            "glucose", "blood pressure", "heart rate", "cholesterol",
            "hemoglobin", "platelet", "wbc", "rbc", "creatinine",
            "sodium", "potassium", "calcium", "thyroid", "insulin",
            "vitamin", "iron", "ferritin", "uric acid", "bilirubin",
            "sgpt", "sgot", "alkaline phosphatase", "albumin"
        ]

    def preprocess_image(self, image_bytes: bytes) -> BytesIO:
        """
        Preprocess image for better OCR accuracy.
        - Convert to grayscale
        - Resize if too large
        - Enhance contrast
        - Sharpen
        """
        try:
            # Open image
            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB (remove alpha channel if present)
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background

            # Resize if too large (OCR works best at 1200-2000px width)
            max_width = 2000
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

            # Convert to grayscale for better OCR
            image = image.convert('L')

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)

            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)

            # Save to BytesIO
            output = BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)

            return output

        except Exception as e:
            print(f"Image preprocessing error: {e}")
            # Return original if preprocessing fails
            return BytesIO(image_bytes)

    def call_ocr_api(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Call OCR.space API to extract text from image.

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)

            # Prepare API request
            payload = {
                'apikey': self.api_key,
                'language': 'eng',
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2  # Engine 2 is more accurate for documents
            }

            files = {
                'file': ('image.jpg', processed_image, 'image/jpeg')
            }

            # Call API
            response = requests.post(
                self.api_url,
                files=files,
                data=payload,
                timeout=30
            )

            result = response.json()

            # Check for errors
            if result.get('IsErroredOnProcessing'):
                error_msg = result.get('ErrorMessage', ['Unknown error'])[0]
                raise Exception(f"OCR API error: {error_msg}")

            # Extract text
            parsed_results = result.get('ParsedResults', [])
            if not parsed_results:
                return "", 0.0

            extracted_text = parsed_results[0].get('ParsedText', '').strip()

            # Calculate confidence (OCR.space doesn't provide confidence, estimate it)
            confidence = 0.9 if len(extracted_text) > 50 else 0.7

            return extracted_text, confidence

        except Exception as e:
            print(f"OCR API call error: {e}")
            raise Exception(f"Failed to extract text: {str(e)}")

    def extract_keywords(self, text: str) -> list:
        """
        Extract medical keywords from text.
        """
        text_lower = text.lower()
        found_keywords = []

        for keyword in self.medical_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords

    def categorize_report(self, text: str) -> str:
        """
        Categorize the type of medical report based on content.
        """
        text_lower = text.lower()

        # Lab report indicators
        if any(word in text_lower for word in ['lab', 'test', 'result', 'glucose', 'hemoglobin']):
            return "lab_report"

        # Prescription indicators
        if any(word in text_lower for word in ['prescription', 'medication', 'tablet', 'capsule', 'mg']):
            return "prescription"

        # Imaging report indicators
        if any(word in text_lower for word in ['x-ray', 'ct scan', 'mri', 'ultrasound', 'radiology']):
            return "imaging_report"

        # Discharge summary indicators
        if any(word in text_lower for word in ['discharge', 'admitted', 'diagnosis', 'treatment plan']):
            return "discharge_summary"

        return "general_report"

    async def extract_text_from_image(
            self,
            user_id: str,
            image_bytes: bytes,
            report_type: Optional[str] = None
    ) -> Dict:
        """
        Extract text from medical report image and store it.
        NOW WITH SMART PARSING!

        Args:
            user_id: User identifier
            image_bytes: Image file bytes
            report_type: Optional report type (auto-detected if not provided)

        Returns:
            Dict with extracted text, parsed values, alerts, and metadata
        """
        try:
            # Generate report ID
            report_id = f"report_{uuid.uuid4().hex[:12]}"

            # Extract text using OCR API
            extracted_text, confidence = self.call_ocr_api(image_bytes)

            if not extracted_text:
                return {
                    "status": "error",
                    "message": "No text could be extracted from the image. Please ensure the image is clear and contains text."
                }

            # ========================================================================
            # NEW - SMART PARSING: Extract structured health values
            # ========================================================================
            parsed_data = self.parser.parse_text(extracted_text)

            # Extract keywords
            keywords = self.extract_keywords(extracted_text)

            # Categorize if not provided
            if not report_type:
                report_type = self.categorize_report(extracted_text)

            # Store in database
            ocr_data = {
                "report_id": report_id,
                "user_id": user_id,
                "extracted_text": extracted_text,
                "confidence": confidence,
                "report_type": report_type,
                "keywords": keywords,
                "created_at": datetime.utcnow().isoformat(),
                # NEW - Add parsed data
                "parsed_values": parsed_data.get('extracted_values', {}),
                "patient_info": parsed_data.get('patient_info', {}),
                "alerts": parsed_data.get('alerts', []),
                "metrics_found": parsed_data.get('total_metrics_found', 0),
            }

            success = await self.storage.save_ocr_report(user_id, report_id, ocr_data)

            if not success:
                return {
                    "status": "error",
                    "message": "Failed to save OCR report to database"
                }

            return {
                "status": "success",
                **ocr_data
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"OCR extraction failed: {str(e)}"
            }

    async def get_user_reports(self, user_id: str) -> Dict:
        """
        Get all OCR reports for a user.
        """
        try:
            reports = await self.storage.get_user_ocr_reports(user_id)

            return {
                "status": "success",
                "user_id": user_id,
                "total_reports": len(reports),
                "reports": reports
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to retrieve reports: {str(e)}"
            }

    async def get_report_by_id(self, report_id: str) -> Optional[Dict]:
        """
        Get specific OCR report by ID.
        """
        try:
            return await self.storage.get_ocr_report(report_id)
        except Exception as e:
            print(f"Error retrieving report: {e}")
            return None

    async def delete_report(self, report_id: str) -> bool:
        """
        Delete an OCR report.
        """
        try:
            return await self.storage.delete_ocr_report(report_id)
        except Exception as e:
            print(f"Error deleting report: {e}")
            return False