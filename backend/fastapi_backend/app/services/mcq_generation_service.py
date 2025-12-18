from typing import Dict
from fastapi import HTTPException, status
from app.schemas.mcq_generation import MCQGenerationRequest
from ai_ml.MCQGenerator import MCQGenerator
from ai_ml.AIExceptions import *
from app.core import models
from app.config import settings

model_name = settings.HF_EVAL_MODEL_NAME


def generate_mcqs_service(input_request: MCQGenerationRequest):

    input_request = input_request.model_dump()
    
    required_fields = ["topic_id", "topic", "subject", "num_questions"]

    for field in required_fields:
        if field not in input_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field: {field}"
            )

    try:
        generator = MCQGenerator(
            model_name = model_name,
            global_model = models.ai_model
        )
    except ModelLoadException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

   
    try:
        result = generator.generate(input_request)
        return result

    except MCQGenerationException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected MCQ generation error: {str(e)}"
        )
