from pydantic import BaseModel, Field
from typing import List


class MCQOption(BaseModel):
    option_id: str
    text: str


class MCQ(BaseModel):
    question: str
    options: List[MCQOption]
    correct_option: str


class MCQGenerationRequest(BaseModel):
    topic_id: str
    topic: str
    subject: str
    num_questions: int = Field(..., gt=0)


class MCQGenerationResponse(BaseModel):
    topic_id: str
    topic: str
    mcqs: List[MCQ]
