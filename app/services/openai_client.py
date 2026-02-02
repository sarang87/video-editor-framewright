from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from app.models import AnalysisCriteria


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini", base_url: Optional[str] = None):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
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
        import base64
        import io
        from decord import VideoReader, cpu
        from PIL import Image
        import numpy as np
        from pathlib import Path

        prompt = self._build_prompt(criteria)
        
        # Manually extract frames to bypass vLLM's flaky video loader
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            # Uniformly sample 8 frames
            indices = np.linspace(0, total_frames - 1, 8, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            
            content_parts = [{"type": "text", "text": prompt}]
            
            # Save frames for inspection
            debug_dir = Path("outputs/debug_frames")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame in enumerate(frames):
                # Convert to PIL Image
                img = Image.fromarray(frame)
                
                # Save debug image
                img.save(debug_dir / f"frame_{i:03d}.jpg")
                
                # Resize to reduce token count (optional, but good for speed)
                img.thumbnail((768, 768)) 
                
                # Encode to base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
                
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
            return f"Error processing video locally: {str(e)}"

