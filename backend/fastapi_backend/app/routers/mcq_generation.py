from fastapi import APIRouter, HTTPException
from app.schemas.mcq_generation import (
    MCQGenerationRequest,
    MCQGenerationResponse
)
from app.services.mcq_generation_service import generation_service


router = APIRouter(prefix="/mcqs", tags=["MCQ Generation"])


@router.post("/generate", response_model=MCQGenerationResponse)
async def generate_mcqs(payload: MCQGenerationRequest):
    response = generation_service.generate_mcqs_service(payload)

    if not response:
        raise HTTPException(
            status_code=500,
            detail="Model failed to generate mcqs"
        )

    return response
