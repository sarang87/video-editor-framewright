from __future__ import annotations

import os
from typing import Optional

import google.generativeai as genai

from app.models import AnalysisCriteria


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro"):
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY is required to call Gemini.")
        genai.configure(api_key=resolved_key)
        self.model = genai.GenerativeModel(model_name)

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
        uploaded = genai.upload_file(video_path)
        try:
            response = self.model.generate_content([uploaded, prompt])
        finally:
            try:
                genai.delete_file(uploaded.name)
            except Exception:
                # Best-effort cleanup; do not hide original errors.
                pass
        return response.text or ""

