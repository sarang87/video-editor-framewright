from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Union

from app.models import AnalysisCriteria, VideoAnalysisResult
from app.services.gemini_client import GeminiClient
from app.services.ollama_client import OllamaClient
from app.services.openai_client import OpenAIClient


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def list_videos(input_dir: str) -> List[Path]:
    path = Path(input_dir).expanduser().resolve()
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS])


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_videos(
    video_files: List[Union[Path, str]],
    output_dir: str,
    criteria: AnalysisCriteria,
    api_key: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    dry_run: bool = False,
) -> List[VideoAnalysisResult]:
    """Analyze videos from a list of file paths or uploaded file objects."""
    output_path = ensure_output_dir(output_dir)
    if dry_run:
        client = None
    elif provider == "openai":
        client = OpenAIClient(api_key=api_key, model_name=model_name or "gpt-4o-mini")
    elif provider == "ollama":
        client = OllamaClient(
            model_name=model_name or "qwen3-vl:latest",
            base_url=ollama_base_url,
        )
    else:
        client = GeminiClient(api_key=api_key, model_name=model_name or "gemini-1.5-pro")
    results: List[VideoAnalysisResult] = []

    for video_file in video_files:
        # Handle both Path objects and string paths
        if isinstance(video_file, str):
            video_path = Path(video_file)
        else:
            video_path = video_file
        
        video_name = video_path.name
        output_file = output_path / f"{video_path.stem}.answers.md"
        
        try:
            if dry_run:
                questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(criteria.questions)])
                content = (
                    f"# Analysis Results for {video_path.stem}\n\n"
                    f"## Questions Asked:\n{questions_text}\n\n"
                    f"## Answers:\n"
                    f"(Dry run mode - no actual analysis performed)\n"
                )
            else:
                content = client.generate_answers_markdown(str(video_path), criteria)
            if not content.strip():
                raise ValueError("AI returned an empty response.")
            output_file.write_text(content, encoding="utf-8")
            results.append(
                VideoAnalysisResult(
                    video_name=video_name,
                    output_path=str(output_file),
                    success=True,
                )
            )
        except Exception as exc:
            results.append(
                VideoAnalysisResult(
                    video_name=video_name,
                    output_path=str(output_file),
                    success=False,
                    error=str(exc),
                )
            )
    return results

