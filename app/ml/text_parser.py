"""
AI-Powered Medical Text Parser using Groq.
FIXED: Uses app settings for API key.
"""
import json
from typing import Dict, Any
from datetime import datetime
from groq import Groq
from app.config import settings


class MedicalTextParser:
    """Parse medical report text using Groq AI."""

    def __init__(self):
        # FIXED: Get API key from settings (which loads from .env)
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in settings. Add it to .env file.")

        self.client = Groq(api_key=api_key)

        self.normal_ranges = {
            'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
            'blood_pressure_systolic': {'min': 90, 'max': 120, 'unit': 'mmHg'},
            'blood_pressure_diastolic': {'min': 60, 'max': 80, 'unit': 'mmHg'},
            'heart_rate': {'min': 60, 'max': 100, 'unit': 'bpm'},
            'cholesterol_total': {'min': 0, 'max': 200, 'unit': 'mg/dL'},
            'cholesterol_hdl': {'min': 40, 'max': 999, 'unit': 'mg/dL'},
            'cholesterol_ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
            'hemoglobin': {'min': 12, 'max': 17, 'unit': 'g/dL'},
            'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'},
            'wbc': {'min': 4.0, 'max': 11.0, 'unit': '10³/µL'},
            'rbc': {'min': 4.5, 'max': 5.5, 'unit': '10⁶/µL'},
            'platelets': {'min': 150, 'max': 400, 'unit': '10³/µL'},
        }

    def parse_text(self, text: str) -> Dict[str, Any]:
        """
        Parse medical report text using Groq AI.

        Args:
            text: OCR extracted text from medical report

        Returns:
            Dict with extracted values, alerts, and metadata
        """
        try:
            # Create prompt for Groq
            prompt = f"""Extract health metrics from this medical report. Return ONLY a JSON object, no other text.

Medical Report:
{text}

Extract these values if present:
- glucose (value and unit)
- hemoglobin (value and unit)
- wbc (white blood cells, value and unit)
- rbc (red blood cells, value and unit)
- platelets (value and unit)
- cholesterol_total (value and unit)
- cholesterol_hdl (value and unit)
- cholesterol_ldl (value and unit)
- triglycerides (value and unit)
- blood_pressure (systolic, diastolic)
- heart_rate (value and unit)

Also extract patient info:
- name
- age
- gender
- report_date

Return JSON format:
{{
  "extracted_values": {{
    "glucose": {{"value": 120.0, "unit": "mg/dL"}},
    "blood_pressure": {{"systolic": 130, "diastolic": 85, "unit": "mmHg", "formatted": "130/85"}},
    ...
  }},
  "patient_info": {{
    "name": "John Doe",
    "age": 45,
    "gender": "Male",
    "report_date": "January 12, 2026"
  }}
}}

RETURN ONLY THE JSON OBJECT. NO MARKDOWN, NO EXPLANATIONS."""

            # Call Groq
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            # Get response
            ai_response = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]

            ai_response = ai_response.strip()

            # Parse JSON
            parsed_data = json.loads(ai_response)

            # Generate alerts
            alerts = self.generate_alerts(parsed_data.get('extracted_values', {}))

            return {
                'extracted_values': parsed_data.get('extracted_values', {}),
                'alerts': alerts,
                'patient_info': parsed_data.get('patient_info', {}),
                'total_metrics_found': len(parsed_data.get('extracted_values', {})),
                'total_alerts': len(alerts),
                'parsed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            print(f"Groq parsing error: {e}")
            # Return empty but valid structure
            return {
                'extracted_values': {},
                'alerts': [],
                'patient_info': {},
                'total_metrics_found': 0,
                'total_alerts': 0,
                'parsed_at': datetime.utcnow().isoformat()
            }

    def generate_alerts(self, extracted: Dict) -> list:
        """Generate health alerts for abnormal values."""
        alerts = []

        for metric, data in extracted.items():
            # Handle blood pressure
            if metric == 'blood_pressure':
                systolic = data.get('systolic', 0)
                diastolic = data.get('diastolic', 0)

                if systolic > 130 or diastolic > 85:
                    severity = 'high' if systolic > 140 or diastolic > 90 else 'medium'
                    alerts.append({
                        'metric': 'blood_pressure',
                        'value': f"{int(systolic)}/{int(diastolic)}",
                        'severity': severity,
                        'message': f"Blood pressure elevated ({int(systolic)}/{int(diastolic)} vs normal 120/80 mmHg)"
                    })
                continue

            # Check other metrics
            if metric in self.normal_ranges:
                normal = self.normal_ranges[metric]
                value = data.get('value', 0)

                if value < normal['min'] or value > normal['max']:
                    deviation = abs(value - normal['max']) / normal['max'] if value > normal['max'] else abs(
                        value - normal['min']) / normal['min']
                    severity = 'high' if deviation > 0.2 else 'medium'
                    status = 'low' if value < normal['min'] else 'elevated'

                    alerts.append({
                        'metric': metric,
                        'value': value,
                        'severity': severity,
                        'message': f"{metric.replace('_', ' ').title()} {status} ({value} vs normal {normal['min']}-{normal['max']} {normal['unit']})"
                    })

        return alerts