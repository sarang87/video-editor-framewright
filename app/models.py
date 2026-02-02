from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AnalysisCriteria(BaseModel):
    questions: List[str] = Field(
        default_factory=lambda: [
            "What is happening in this video?",
            "Who or what are the main subjects?",
            "What is the setting or location?",
        ],
        description="List of questions to answer about the video",
    )
    audience: str = Field(default="General viewers", description="Target audience for the answers")
    tone: str = Field(default="Professional", description="Tone/style of the answers")
    depth: str = Field(default="Medium", description="Depth of analysis")
    extra_instructions: Optional[str] = Field(
        default=None, description="Additional instructions for how to answer"
    )


class VideoAnalysisResult(BaseModel):
    video_name: str
    output_path: str
    success: bool
    error: Optional[str] = None


class ClipMetadata(BaseModel):
    clip_name: str
    category: str = Field(description="A-roll or B-roll")
    visual_description: str
    shot_type: str
    motion_detected: str
    narrative_utility: str
    transition_point: Optional[str] = Field(default=None, description="Suggested time/frame to cut")

