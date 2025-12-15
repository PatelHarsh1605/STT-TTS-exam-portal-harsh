from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    language: str = "en"
    slow: bool = False

