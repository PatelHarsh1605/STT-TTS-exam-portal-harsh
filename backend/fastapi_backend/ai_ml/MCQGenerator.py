from typing import List
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from ai_ml.ModelCreator import HFModelCreation
from ai_ml.AIExceptions import *



class MCQOption(BaseModel):
    option_id: str = Field(..., description="Option label like A, B, C, D")
    text: str


class MCQ(BaseModel):
    question: str
    options: List[MCQOption]
    correct_option: str


class MCQOutput(BaseModel):
    mcqs: List[MCQ]




class MCQGenerator:

    def __init__(self, model_name: str, global_model = None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = HFModelCreation.hf_model_creator(self.model_name)
        return self.model

    def create_chain(self):
        try:
            parser = JsonOutputParser(pydantic_object=MCQOutput)

            template = """
You are an expert exam paper setter.

Generate EXACTLY {num_questions} MCQs.

STRICT RULES:
- Output MUST be a SINGLE valid JSON object
- The JSON MUST follow this schema EXACTLY
- Do NOT add numbering like 1., 2.
- Do NOT add explanations
- Do NOT add markdown
- Do NOT add extra text

The root object MUST be:
{
  "mcqs": [ ... ]
}

Each MCQ:
- question: string
- options: exactly 4 options with A, B, C, D
- correct_option: one of A, B, C, D

Topic ID: {topic_id}
Topic: {topic}
Subject: {subject}

{format_instructions}
"""


            prompt = PromptTemplate(
                template=template,
                input_variables=["num_questions","topic_id", "topic", "subject"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                }
            )

            chain = prompt | self.get_model()

            return chain, parser
        
        except Exception as e:
            raise ChainCreationException(f"Could not create chain due to error: {str(e)}")
            

    def generate(self, input_request: dict) -> MCQOutput:
        chain, parser = self.create_chain()

        raw_output = chain.invoke(input_request)

        if hasattr(raw_output, "content"):
            output_text = raw_output.content
        else:
            output_text = str(raw_output)

        try:
            parsed = parser.parse(output_text)
        except Exception as e:
            raise MCQGenerationException(
                f"Invalid JSON from model.\nRaw output:\n{output_text}"
            )

        if not isinstance(parsed, MCQOutput):
            raise MCQGenerationException(
                f"Parser returned unexpected type: {type(parsed)}"
            )

        return parsed

    

    