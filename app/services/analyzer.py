from mirascope.core import openai, prompt_template
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
from app.models import ClipMetadata
from app.core.config import settings

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, base_url: Optional[str] = None, api_key: str = "token-is-ignored"):
        self.base_url = base_url or settings.VLLM_BASE_URL
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
        from app.utils.logger import setup_logging
        from app.utils.video_utils import extract_frames_as_base64, build_image_payloads, get_video_duration
        from openai import OpenAI
        import json

        logger = setup_logging(__name__)
        video_name = os.path.basename(video_path)
        
        try:
            logger.info(f"Analyzing clip: {video_name} using Frame Extraction")
            
            # Get duration
            duration = get_video_duration(video_path)
            
            # extract frames
            base64_frames = extract_frames_as_base64(video_path, num_frames=16)
            image_payloads = build_image_payloads(base64_frames)
            
            # ... (omitted prompt setup for brevity in search, but needed for replace)
            # Actually, I should use StartLine/EndLine carefully to avoid rewriting the prompt.
            # I will just rewrite the surrounding code.

            # Construct prompt
            system_prompt = (
                "You are an editing assistant (intern) reviewing raw footage post-production. "
                "Your job is to produce detailed, actionable notes for the lead editor.\n\n"
                "You are looking at a sequence of 16 chronologically ordered frames from a 30-second video clip.\n\n"
                "Analyze the visual consistency between frames to identify Camera Motion (e.g., if the subject moves across the frames, it's likely a Pan).\n"
                "Determine if the footage is A-Roll (interview/main action) or B-Roll (supplemental/texture).\n"
                "Suggest a specific Transition Point based on when the action in the frames reaches its peak.\n\n"
                "Output the response in strictly valid JSON format matching this schema:\n"
                "{\n"
                '  "clip_name": "string",\n'
                '  "category": "string (A-roll or B-roll)",\n'
                '  "visual_description": "string",\n'
                '  "shot_type": "string",\n'
                '  "motion_detected": "string",\n'
                '  "narrative_utility": "string",\n'
                '  "transition_point": "string (e.g., \'Frame 5\')"\n'
                "}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"Analyze these frames for video: {video_name}"}
                    ] + image_payloads
                }
            ]

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            logger.debug(f"Sending request to vLLM at {self.base_url}")
            response = client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct-FP8",
                messages=messages,
                max_tokens=1024
            )
            
            content = response.choices[0].message.content
            # Clean markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
            
            data = json.loads(content)
            # Ensure clip_name is set correctly if model hallucinates it
            data["clip_name"] = video_name
            data["duration"] = duration
            
            return ClipMetadata(**data)

        except Exception as e:
            logger.error(f"Error analyzing clip {video_name}: {e}", exc_info=True)
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
