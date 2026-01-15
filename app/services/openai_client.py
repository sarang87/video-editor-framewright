from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from app.models import AnalysisCriteria


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required to call OpenAI.")
        self.client = OpenAI(api_key=resolved_key)
        self.model_name = model_name

    def _build_prompt(self, criteria: AnalysisCriteria) -> str:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(criteria.questions)])
        extra = criteria.extra_instructions or ""
        extra_instruction_text = f"\n\nAdditional instructions: {extra}" if extra else ""
        
        return (
            "Analyze the video and answer the following questions based on what you observe:\n\n"
            f"{questions_text}\n\n"
            f"Answer style:\n"
            f"- Audience: {criteria.audience}\n"
            f"- Tone: {criteria.tone}\n"
            f"- Depth: {criteria.depth}\n"
            f"{extra_instruction_text}\n\n"
            "Output format: Markdown with a title, then numbered answers corresponding to each question.\n"
            "Be specific and reference observable events in the video. If a question cannot be answered from the video, state that clearly.\n"
        )

    def generate_answers_markdown(self, video_path: str, criteria: AnalysisCriteria) -> str:
        prompt = self._build_prompt(criteria)
        with open(video_path, "rb") as video_file:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "video",
                                "video": video_file,
                            },
                        ],
                    }
                ],
            )
        return response.choices[0].message.content or ""

