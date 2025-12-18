from typing import Dict
from fastapi import HTTPException, status

from ai_ml.MCQGenerator import MCQGenerator
from ai_ml.AIExceptions import (
    ModelLoadError,
    MCQGenerationError
)

# --------------------------------------------------
# MCQ Generation Service
# --------------------------------------------------

def generate_mcqs_service(input_request: Dict):
    """
    Service layer for MCQ generation.

    This function:
    - Validates required inputs
    - Initializes MCQGenerator
    - Calls generation logic
    - Returns structured response
    """

    # -------------------------------
    # 1️⃣ Basic Input Validation
    # -------------------------------
    required_fields = ["topic_id", "topic", "subject", "num_questions"]
    for field in required_fields:
        if field not in input_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field: {field}"
            )

    # -------------------------------
    # 2️⃣ Initialize Generator
    # -------------------------------
    try:
        generator = MCQGenerator(
            model_name="microsoft/phi-2"  # CPU safe model
        )
    except ModelLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    # -------------------------------
    # 3️⃣ Generate MCQs
    # -------------------------------
    try:
        result = generator.generate(input_request)
        return result

    except MCQGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected MCQ generation error: {str(e)}"
        )
