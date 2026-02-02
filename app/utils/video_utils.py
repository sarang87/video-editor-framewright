import base64
import io
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image

# Try importing decord, handle potential import errors gracefully if not installed
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

def extract_frames_as_base64(
    video_path: Union[str, Path], 
    num_frames: int = 8, 
    resize_target: int = 768,
    save_debug: bool = True
) -> List[str]:
    """
    Extracts frames from a video and returns them as base64 encoded strings.
    
    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract uniformly.
        resize_target: Max dimension for thumbnailing (keeps aspect ratio).
        save_debug: If True, saves frames to outputs/debug_frames.
    
    Returns:
        List of base64 string representations of the frames.
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required for video processing but is not installed.")

    video_path = str(video_path)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Uniformly sample frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    
    base64_frames = []
    
    if save_debug:
        debug_dir = Path("outputs/debug_frames")
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        if save_debug:
            img.save(debug_dir / f"{Path(video_path).stem}_frame_{i:03d}.jpg")
        
        # Resize to reduce token count
        img.thumbnail((resize_target, resize_target)) 
        
        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_frames.append(base64_image)
        
    return base64_frames

def build_image_payloads(base64_frames: List[str]) -> List[Dict[str, Any]]:
    """
    Wraps base64 strings into the OpenAI 'image_url' format.
    """
    payloads = []
    for b64 in base64_frames:
        payloads.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    return payloads
