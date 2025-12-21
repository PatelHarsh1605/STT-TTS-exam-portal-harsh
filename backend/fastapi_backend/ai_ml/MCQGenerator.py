from typing import List
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

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

FORMAT: Use this exact format for each question:

Question {num}: [Question text here?]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [A/B/C/D]

---

Topic ID: {topic_id}
Topic: {topic}
Subject: {subject}

Generate the MCQs now:"""

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

        # Case 1: LangChain returns string
        if isinstance(raw_output, str):
            return raw_output

        # Case 2: HuggingFacePipeline returns list[dict]
        if isinstance(raw_output, list):
            if len(raw_output) == 0:
                raise MCQGenerationException("Empty output from model")

            item = raw_output[0]
            if isinstance(item, dict) and "generated_text" in item:
                # Extract only the newly generated text, not the input prompt
                full_text = item["generated_text"]
                return full_text

        # Case 3: LLMResult style
        if hasattr(raw_output, "generations"):
            return raw_output.generations[0][0].text

        # Case 4: content attribute
        if hasattr(raw_output, "content"):
            return raw_output.content

        raise MCQGenerationException(
            f"Could not extract text from model output: {type(raw_output)}\nOutput: {str(raw_output)[:500]}"
        )

    def generate(self, input_request: dict) -> MCQOutput:
        chain = self.create_chain()

        raw_output = chain.invoke(input_request)
        output_text = self.extract_text(raw_output)

        # Parse the plain text MCQs into structured format
        mcqs = self.parse_mcqs_from_text(output_text, input_request["num_questions"])
        
        return MCQOutput(mcqs=mcqs)

    def parse_mcqs_from_text(self, text: str, expected_count: int) -> List[MCQ]:
        """
        Parse MCQs from plain text format:
        
        Question 1: [text]?
        A) [text]
        B) [text]
        C) [text]
        D) [text]
        Answer: [A/B/C/D]
        """
        mcqs = []
        
        # Split by "Question N:" pattern
        question_pattern = r'Question\s+\d+:\s*(.+?)(?=Question\s+\d+:|$)'
        question_blocks = re.findall(question_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not question_blocks:
            raise MCQGenerationException(
                f"Could not find any MCQs in output. Expected {expected_count} MCQs.\n"
                f"Output was:\n{text[:500]}"
            )

        for block in question_blocks:
            try:
                mcq = self.parse_single_mcq(block)
                mcqs.append(mcq)
            except Exception as e:
                # Skip malformed questions and continue
                print(f"Warning: Could not parse MCQ block: {str(e)}")
                continue

        if len(mcqs) != expected_count:
            raise MCQGenerationException(
                f"Expected {expected_count} MCQs, but got {len(mcqs)}"
            )

        return mcqs

    def parse_single_mcq(self, block: str) -> MCQ:
        """Parse a single MCQ from text block"""
        
        # Extract question
        question_match = re.search(r'^(.+?)(?=\nA\)|A\))', block, re.DOTALL)
        if not question_match:
            raise MCQGenerationException("Could not find question text")
        
        question_text = question_match.group(1).strip()
        # Remove "Question N:" prefix if present
        question_text = re.sub(r'^Question\s+\d+:\s*', '', question_text, flags=re.IGNORECASE).strip()
        
        # Extract options
        options = []
        option_pattern = r'([A-D])\)\s*(.+?)(?=[A-D]\)|Answer:|$)'
        option_matches = re.findall(option_pattern, block, re.DOTALL | re.IGNORECASE)
        
        if len(option_matches) < 4:
            raise MCQGenerationException(f"Could not find 4 options. Found {len(option_matches)}")
        
        for option_id, option_text in option_matches[:4]:
            option_text = option_text.strip()
            # Remove trailing newlines and extra whitespace
            option_text = re.sub(r'\n+.*$', '', option_text).strip()
            options.append(MCQOption(option_id=option_id.upper(), text=option_text))
        
        # Extract correct answer
        answer_match = re.search(r'Answer:\s*([A-D])', block, re.IGNORECASE)
        if not answer_match:
            raise MCQGenerationException("Could not find Answer field")
        
        correct_option = answer_match.group(1).upper()
        
        # Validate correct option is in options
        option_ids = {opt.option_id for opt in options}
        if correct_option not in option_ids:
            raise MCQGenerationException(
                f"Correct option '{correct_option}' not in options {option_ids}"
            )
        
        return MCQ(
            question=question_text,
            options=options,
            correct_option=correct_option
        )
