from app.schemas.question_generation import QuestionGenerationRequest
from app.core import models
from ai_ml.QuestionsGenerator import QuestionsGenerator
from app.config import settings
from ai_ml.AIExceptions import *

from fastapi import HTTPException, status

model_name = settings.HF_EVAL_MODEL_NAME

class QuestionGenerationService:

    def generate(self, payload: QuestionGenerationRequest):

        data = payload.model_dump()

        try:

            # Use models.ai_model loaded during lifespan

            generator = QuestionsGenerator(
                model_name=model_name, 
                global_model=models.ai_model
            )
        
        except ModelLoadException as e:
            raise HTTPException(
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail = str(e)
            )
    
        try:
            result = generator.create_questions(data)
            

            required_keys = ["topic", "questions"]

            if (
                not result 
                or not isinstance(result, dict)
                or any(k not in result for k in required_keys)
            ):
                raise ValueError("Model returned invalid output.")
            
        except Exception as e:
            
            print("Generation error: ", e)

            return {
                "topic_id": payload.topic_id,
                "topic": payload.topic,
                "questions": ["No questions could be generated due to model error"]
            }

        result["topic_id"] = payload.topic_id
        return result

generation_service = QuestionGenerationService()