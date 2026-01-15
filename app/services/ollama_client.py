from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from typing import Optional

import requests

from app.models import AnalysisCriteria


class OllamaClient:
    def __init__(
        self,
        model_name: str = "qwen3-vl:latest",
        base_url: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def _build_prompt(self, criteria: AnalysisCriteria) -> str:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(criteria.questions)])
        extra = criteria.extra_instructions or ""
        extra_instruction_text = f"\n\nAdditional instructions: {extra}" if extra else ""
        
        return (
            "Analyze the video frame and answer the following questions based on what you observe:\n\n"
            f"{questions_text}\n\n"
            f"Answer style:\n"
            f"- Audience: {criteria.audience}\n"
            f"- Tone: {criteria.tone}\n"
            f"- Depth: {criteria.depth}\n"
            f"{extra_instruction_text}\n\n"
            "Output format: Markdown with a title, then numbered answers corresponding to each question.\n"
            "Be specific and reference observable events in the video. If a question cannot be answered from the video frame, state that clearly.\n"
        )

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. "
                "Install it with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
            )

    def _extract_first_frame(self, video_path: str) -> bytes:
        self._check_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = os.path.join(tmp_dir, "frame.jpg")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    video_path,
                    "-vf",
                    "select=eq(n\\,0)",
                    "-vframes",
                    "1",
                    frame_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not os.path.exists(frame_path):
                error_msg = result.stderr or "Unknown error"
                raise RuntimeError(
                    f"Failed to extract a frame from video. ffmpeg error: {error_msg}"
                )
            with open(frame_path, "rb") as frame_file:
                return frame_file.read()

    def _check_ollama_connection(self) -> None:
        """Check if Ollama is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: 'ollama serve'. Error: {str(e)}"
            )

    def generate_answers_markdown(self, video_path: str, criteria: AnalysisCriteria) -> str:
        self._check_ollama_connection()
        prompt = self._build_prompt(criteria)
        frame_bytes = self._extract_first_frame(video_path)
        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [frame_b64],
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "") or ""
            if not result:
                raise RuntimeError("Ollama returned empty response")
            return result
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request timed out after {self.timeout} seconds. "
                f"The model might be slow or the video too large. Try increasing timeout or using a faster model."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with Ollama: {str(e)}")

