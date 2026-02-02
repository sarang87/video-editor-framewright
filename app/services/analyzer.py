from mirascope.core import openai, prompt_template
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
from app.models import ClipMetadata

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "token-is-ignored"):
        self.base_url = base_url
        self.api_key = api_key

    @openai.call(model="Qwen/Qwen3-VL-8B-Instruct-FP8", response_model=ClipMetadata)
    @prompt_template("""
        <|video|>{video_path}
        Analyze the video clip above and provide structured metadata.
        Focus on cinematic qualities, visual storytelling, and technical details.
        
        Instructions:
        - clip_name: The filename provided ({video_name}).
        - category: Decide if this is A-roll or B-roll.
        - visual_description: A detailed objective summary.
        - shot_type: e.g., Close-up, Wide shot, Medium shot, etc.
        - motion_detected: Describe the camera or subject movement.
        - narrative_utility: How should an editor use this clip?
    """)
    def _analyze(self, video_path: str, video_name: str):
        return {"video_path": video_path, "video_name": video_name}

    def analyze_clip(self, video_path: str) -> Optional[ClipMetadata]:
        video_name = os.path.basename(video_path)
        # Map path from 'app' container (/app/...) to 'vllm' container (/opt/project_root/...)
        vllm_video_path = video_path.replace("/app/", "/opt/project_root/")
        try:
            os.environ["OPENAI_BASE_URL"] = self.base_url
            os.environ["OPENAI_API_KEY"] = self.api_key
            
            result = self._analyze(video_path=vllm_video_path, video_name=video_name)
            return result
        except Exception as e:
            logger.error(f"Error analyzing clip {video_name}: {e}")
            # Potential retry logic or fallback can go here
            return None

    @openai.call(model="Qwen/Qwen3-VL-8B-Instruct-FP8")
    @prompt_template("""
        SYSTEM: You are a professional film editor's assistant.
        The following clips were found in the database based on the user's query: "{query}"

        CLIPS DATA:
        {clips_context}

        USER REQUEST: {user_request}

        Create a 'Shot List' or 'Edit Plan' using these available clips. 
        Focus on the narrative goal. Keep responses under 1000 tokens.
    """)
    def _brainstorm(self, query: str, clips_context: str, user_request: str):
        return {"query": query, "clips_context": clips_context, "user_request": user_request}

    def brainstorm_narrative(self, query: str, clips_df_json: str, user_request: str) -> str:
        try:
            os.environ["OPENAI_BASE_URL"] = self.base_url
            os.environ["OPENAI_API_KEY"] = self.api_key
            
            # vLLM/OpenAI call doesn't natively support max_tokens in the decorator easily without call_params
            # but we can pass it if we use call_params
            result = self._brainstorm(
                query=query, 
                clips_context=clips_df_json, 
                user_request=user_request,
                call_params={"max_tokens": 1000}
            )
            return str(result.content)
        except Exception as e:
            logger.error(f"Error during brainstorming: {e}")
            return f"Failed to generate edit plan: {str(e)}"
