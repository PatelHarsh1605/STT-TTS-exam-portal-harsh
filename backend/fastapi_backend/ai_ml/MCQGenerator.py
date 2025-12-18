from typing import List
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
    topic_id: str
    topic: str
    mcqs: List[MCQ]




class MCQGenerator:

    def __init__(self, model_name: str, global_model = None):
        self.model_name = model_name
        self.global_model = global_model

    def get_model(self):
        if self.global_model is None:
            HFModelCreation.hf_model_creator(self.model_name)
        return self.global_model

    def create_chain(self):
        try:
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

            chain = prompt | self.get_model()

            return chain, parser
        
        except Exception as e:
            raise ChainCreationException(f"Could not create chain due to error: {str(e)}")
            

    def sanitize_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def generate(self, input_request: dict) -> MCQOutput:
        try:
        
            if "topic" not in input_request:
                raise KeyError("Input request must contain the topic related to which you want questions")
            
            elif "subject" not in input_request:
                raise KeyError("Input request must contain the subject related to which you want questions")

            elif "num_questions" not in input_request:
                raise KeyError("Input request must contain the number of questions you want related to the topic")
        

            chain, parser = self.create_chain()

            raw_output = chain.invoke(input_request)

            # Extract text safely
            if isinstance(raw_output, dict) and "text" in raw_output:
                output_text = raw_output["text"]

            elif hasattr(raw_output, "generations"):
                output_text = raw_output.generations[0][0].text

            else:
                output_text = str(raw_output)

            cleaned_json = self.sanitize_json(output_text)

            return parser.parse(cleaned_json)

        except Exception as e:
            raise MCQGenerationException(f"MCQ generation failed: {str(e)}")
