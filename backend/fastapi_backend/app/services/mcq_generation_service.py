from fastapi import HTTPException, status
from app.schemas.mcq_generation import MCQGenerationRequest
from ai_ml.MCQGenerator import MCQGenerator
from ai_ml.AIExceptions import *
from app.core import models
from app.config import settings

model_name = settings.HF_EVAL_MODEL_NAME

class MCQGenerationService:

    def generate_mcqs_service(self, input_request: MCQGenerationRequest):

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

        except Exception as e:
            print("Generation error: ", e)

            return {
            "topic_id": input_request["topic_id"],
            "topic": input_request["topic"],
            "mcqs": [
                {
                    "question": "MCQ generation failed due to model error",
                    "options": [
                        {"option_id": "A", "text": "Retry later"},
                        {"option_id": "B", "text": "Retry later"},
                        {"option_id": "C", "text": "Retry later"},
                        {"option_id": "D", "text": "Retry later"},
                    ],
                    "correct_option": "A"
                }
            ]
        }

        
generation_service = MCQGenerationService()