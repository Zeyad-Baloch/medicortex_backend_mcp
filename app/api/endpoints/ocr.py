"""
OCR endpoints for medical report text extraction.
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from app.api.schemas import (
    OCRExtractResponse,
    OCRReportsListResponse,
    OCRReportDetail
)
from app.services.ocr import OCRService
from app.dependencies import get_ocr_service

router = APIRouter()


@router.post("/extract", response_model=OCRExtractResponse)
async def extract_text_from_report(
        file: UploadFile = File(..., description="Medical report image (JPG, PNG, PDF)"),
        user_id: str = Form(..., description="User ID"),
        report_type: Optional[str] = Form(None, description="Report type (auto-detected if not provided)")
):
    """
    Extract text from medical report image using OCR.

    Supports: JPG, PNG, JPEG, BMP
    Max file size: 5MB

    Returns extracted text, confidence score, and detected keywords.
    """
    try:
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: JPG, PNG, BMP. Got: {file.content_type}"
            )

        # Read file
        image_bytes = await file.read()

        # Check file size (5MB limit)
        max_size = 5 * 1024 * 1024  # 5MB
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: 5MB. Your file: {len(image_bytes) / 1024 / 1024:.2f}MB"
            )

        # Get OCR service
        service = get_ocr_service()

        # Extract text
        result = await service.extract_text_from_image(
            user_id=user_id,
            image_bytes=image_bytes,
            report_type=report_type
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")


@router.get("/reports/{user_id}", response_model=OCRReportsListResponse)
async def get_user_reports(user_id: str):
    """
    Get all OCR reports for a specific user.

    Returns list of all medical reports processed for this user.
    """
    try:
        service = get_ocr_service()
        result = await service.get_user_reports(user_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{report_id}", response_model=OCRReportDetail)
async def get_report_by_id(report_id: str):
    """
    Get specific OCR report by report ID.

    Returns detailed information about a single report.
    """
    try:
        service = get_ocr_service()
        result = await service.get_report_by_id(report_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/report/{report_id}")
async def delete_report(report_id: str):
    """
    Delete an OCR report.

    Permanently removes the report from the database.
    """
    try:
        service = get_ocr_service()
        success = await service.delete_report(report_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

        return {
            "status": "success",
            "message": f"Report {report_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))