from fastapi import APIRouter
from app.schemas.mcq_generation import (
    MCQGenerationRequest,
    MCQGenerationResponse
)
from app.services.mcq_generation_service import generate_mcqs_service

router = APIRouter(prefix="/mcqs", tags=["MCQ Generation"])


@router.post("/generate", response_model=MCQGenerationResponse)
async def generate_mcqs(payload: MCQGenerationRequest):
    return generate_mcqs_service(payload.dict())
