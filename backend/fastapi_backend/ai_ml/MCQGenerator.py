from typing import List
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from ai_ml.ModelCreator import HFModelCreation
from ai_ml.AIExceptions import *


class MCQOption(BaseModel):
    option_id: str
    text: str


class MCQ(BaseModel):
    question: str
    options: List[MCQOption]
    correct_option: str


class MCQOutput(BaseModel):
    mcqs: List[MCQ]


class MCQGenerator:

    def __init__(self, model_name: str, global_model=None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = HFModelCreation.hf_model_creator(self.model_name)
        return self.model

    def create_chain(self):
        try:
            # Use simple text format instead of JSON - much easier for model to generate
            template = """You are an expert exam paper setter.

Generate EXACTLY {num_questions} MCQs on the following topic.

IMPORTANT: Number each question sequentially as Question 1:, Question 2:, Question 3:, etc.

FORMAT: Use this exact format for each question:

Question 1: [Question text here?]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [A/B/C/D]

---

Question 2: [Next question text here?]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [A/B/C/D]

Topic ID: {topic_id}
Topic: {topic}
Subject: {subject}

Now generate {num_questions} MCQs with sequential numbering:"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["num_questions", "topic_id", "topic", "subject"]
            )

            chain = prompt | self.get_model()

            return chain

        except Exception as e:
            raise ChainCreationException(
                f"Could not create chain due to error: {str(e)}")

    def extract_text(self, raw_output) -> str:
        if isinstance(raw_output, list) and len(raw_output) > 0:
            item = raw_output[0]
            if isinstance(item, dict) and "generated_text" in item:
                return item["generated_text"]
        raise MCQGenerationException(f"Could not extract text from model output: {type(raw_output)}")

    def generate(self, input_request: dict) -> MCQOutput:
        chain = self.create_chain()

        raw_output = chain.invoke(input_request)
        output_text = self.extract_text(raw_output)

        # Parse the plain text MCQs into structured format
        mcqs = self.parse_mcqs_from_text(output_text, input_request["num_questions"])
        
        return MCQOutput(mcqs=mcqs)

    def parse_mcqs_from_text(self, text: str, expected_count: int) -> List[MCQ]:
        mcqs = []
        question_pattern = r'Question\s+\d+:\s*(.+?)(?=Question\s+\d+:|$)'
        question_blocks = re.findall(question_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not question_blocks:
            raise MCQGenerationException(f"Could not find any MCQs in output. Expected {expected_count} MCQs.")

        for block in question_blocks:
            try:
                mcqs.append(self.parse_single_mcq(block))
            except Exception:
                continue

        if len(mcqs) > expected_count:
            mcqs = mcqs[:expected_count]
        elif len(mcqs) < expected_count:
            raise MCQGenerationException(f"Expected {expected_count} MCQs, but only got {len(mcqs)}")

        return mcqs

    def parse_single_mcq(self, block: str) -> MCQ:
        question_match = re.search(r'^(.+?)(?=\nA\)|A\))', block, re.DOTALL)
        if not question_match:
            raise MCQGenerationException("Could not find question text")
        
        question_text = re.sub(r'^Question\s+\d+:\s*', '', question_match.group(1).strip(), flags=re.IGNORECASE).strip()
        
        option_pattern = r'([A-D])\)\s*(.+?)(?=[A-D]\)|Answer:|$)'
        option_matches = re.findall(option_pattern, block, re.DOTALL | re.IGNORECASE)
        
        if len(option_matches) < 4:
            raise MCQGenerationException("Could not find 4 options")
        
        options = [
            MCQOption(option_id=opt_id.upper(), text=re.sub(r'\n+.*$', '', opt_text.strip()).strip())
            for opt_id, opt_text in option_matches[:4]
        ]
        
        answer_match = re.search(r'Answer:\s*([A-D])', block, re.IGNORECASE)
        if not answer_match:
            raise MCQGenerationException("Could not find Answer field")
        
        correct_option = answer_match.group(1).upper()
        
        option_ids = {opt.option_id for opt in options}
        if correct_option not in option_ids:
            raise MCQGenerationException(f"Correct option '{correct_option}' not in options {option_ids}")
        
        return MCQ(question=question_text, options=options, correct_option=correct_option)
