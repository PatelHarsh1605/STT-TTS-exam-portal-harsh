from typing import List
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ai_ml.ModelCreator import HFModelCreation
from ai_ml.AIExceptions import *


# -------------------------
# Pydantic Output Schema
# -------------------------

class MCQOption(BaseModel):
    option_id: str = Field(..., description="Option label like A, B, C, D")
    text: str


class MCQ(BaseModel):
    question: str
    options: List[MCQOption]
    correct_option: str


class MCQOutput(BaseModel):
    topic_id: str
    topic: str
    mcqs: List[MCQ]


# -------------------------
# MCQ Generator Class
# -------------------------

class MCQGenerator:

    def __init__(self, model_name: str):
        self.model = HFModelCreation.hf_model_creator(model_name)
        if self.model is None:
            raise ModelLoadError("Failed to load HF model for MCQ generation")

    def _create_chain(self):
        parser = JsonOutputParser(pydantic_object=MCQOutput)

        template = """
You are an expert exam paper setter.

Generate {num_questions} multiple-choice questions (MCQs) based on the following topic.

Rules:
- Each MCQ must have exactly 4 options labeled A, B, C, D
- Only ONE option must be correct
- Questions must strictly belong to the subject
- Difficulty should be appropriate for exams
- Output MUST be valid JSON only

Topic ID: {topic_id}
Topic: {topic}
Subject: {subject}

{format_instructions}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["topic_id", "topic", "subject", "num_questions"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        chain = prompt | self.model
        return chain, parser

    def _sanitize_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def generate(self, input_request: dict) -> MCQOutput:
        try:
            chain, parser = self._create_chain()

            raw_output = chain.invoke(input_request)

            # Extract text safely
            if isinstance(raw_output, dict) and "text" in raw_output:
                output_text = raw_output["text"]
            elif hasattr(raw_output, "generations"):
                output_text = raw_output.generations[0][0].text
            else:
                output_text = str(raw_output)

            cleaned_json = self._sanitize_json(output_text)
            return parser.parse(cleaned_json)

        except Exception as e:
            raise MCQGenerationError(f"MCQ generation failed: {str(e)}")
