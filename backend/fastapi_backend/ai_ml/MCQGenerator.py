from typing import List
import re
import json

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
            # Simple template without JsonOutputParser format instructions
            # We handle parsing manually in generate()
            template = """You are an expert exam paper setter.

Generate EXACTLY {num_questions} MCQs.

IMPORTANT: Return ONLY valid JSON, nothing else. No explanation, no markdown, no extra text.

The response must be a single valid JSON object with this exact structure:
{{
  "mcqs": [
    {{
      "question": "the question text here",
      "options": [
        {{"option_id": "A", "text": "option A text"}},
        {{"option_id": "B", "text": "option B text"}},
        {{"option_id": "C", "text": "option C text"}},
        {{"option_id": "D", "text": "option D text"}}
      ],
      "correct_option": "A"
    }}
  ]
}}

Topic ID: {topic_id}
Topic: {topic}
Subject: {subject}

Generate the JSON response now:"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["num_questions", "topic_id", "topic", "subject"]
            )

            chain = prompt | self.get_model()

            return chain, None  # Return None for parser since we handle it manually

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

    def extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON object from text that may contain other content.
        Finds the first { and last } that form a valid JSON object.
        """
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            raise MCQGenerationException(f"No JSON object found in output: {text[:300]}")
        
        # Try to find matching closing brace
        brace_count = 0
        end_idx = -1
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx == -1:
            raise MCQGenerationException(f"Malformed JSON in output: {text[:300]}")
        
        json_str = text[start_idx:end_idx]
        
        # Validate it's actual JSON
        try:
            json.loads(json_str)
        except json.JSONDecodeError as e:
            raise MCQGenerationException(f"Invalid JSON extracted: {str(e)}\nJSON: {json_str[:300]}")
        
        return json_str


    def generate(self, input_request: dict) -> MCQOutput:
        chain, parser = self.create_chain()

        raw_output = chain.invoke(input_request)
        output_text = self.extract_text(raw_output)

        # Extract JSON from potentially mixed text content
        try:
            json_text = self.extract_json_from_text(output_text)
        except MCQGenerationException:
            json_text = output_text

        # Parse JSON directly without LangChain's JsonOutputParser due to errors
        try:
            json_dict = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise MCQGenerationException(
                f"Failed to parse JSON: {str(e)}\n--- JSON TEXT ---\n{json_text[:500]}"
            )

        # Manually validate and create MCQOutput using Pydantic
        try:
            parsed = MCQOutput(**json_dict)
        except ValueError as e:
            raise MCQGenerationException(
                f"Invalid MCQ structure: {str(e)}\n--- JSON ---\n{json.dumps(json_dict, indent=2)[:500]}"
            )
        except Exception as e:
            raise MCQGenerationException(
                f"Unexpected error creating MCQOutput: {str(e)}\n--- JSON ---\n{json.dumps(json_dict, indent=2)[:500]}"
            )

        if parsed is None:
            raise MCQGenerationException(
                f"Failed to create MCQOutput from valid JSON: {json_text[:300]}"
            )

        if len(parsed.mcqs) != input_request["num_questions"]:
            raise MCQGenerationException(
                f"Expected {input_request['num_questions']} MCQs, "
                f"but got {len(parsed.mcqs)}"
            )

        for idx, mcq in enumerate(parsed.mcqs, start=1):
            option_ids = {opt.option_id for opt in mcq.options}
            if mcq.correct_option not in option_ids:
                raise MCQGenerationException(
                    f"MCQ {idx}: correct_option '{mcq.correct_option}' "
                    f"not in options {option_ids}"
                )

        return parsed
