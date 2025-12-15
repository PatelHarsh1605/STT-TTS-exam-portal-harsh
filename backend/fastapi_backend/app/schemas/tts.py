from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    question_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=500)
    language: str = "en"
    slow: bool = False

