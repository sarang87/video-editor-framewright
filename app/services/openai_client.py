from __future__ import annotations

from typing import Optional

from openai import OpenAI

from app.models import AnalysisCriteria
from app.core.config import settings


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini", base_url: Optional[str] = None):
        resolved_key = api_key or settings.OPENAI_API_KEY
        # For vLLM, API key is often ignored but required by the library
        if not resolved_key and not base_url:
            raise ValueError("OPENAI_API_KEY is required to call OpenAI.")
        
        self.client = OpenAI(
            api_key=resolved_key or "token-is-ignored",
            base_url=base_url
        )
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
        from app.utils.logger import setup_logging
        from app.utils.video_utils import extract_frames_as_base64, build_image_payloads

        logger = setup_logging(__name__)
        prompt = self._build_prompt(criteria)
        
        # Manually extract frames to bypass vLLM's flaky video loader
        try:
            logger.info(f"Extracting frames for Q&A from: {video_path}")
            base64_frames = extract_frames_as_base64(video_path, num_frames=8)
            image_payloads = build_image_payloads(base64_frames)
            
            content_parts = [{"type": "text", "text": prompt}]
            content_parts.extend(image_payloads)
            
            logger.info(f"Sending request to OpenAI compatible API (Model: {self.model_name})")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                ],
            )
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Error processing video locally: {e}", exc_info=True)
            return f"Error processing video locally: {str(e)}"

