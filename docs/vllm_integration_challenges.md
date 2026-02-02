# vLLM Integration & Debugging Journey

This document captures the challenges encountered and solutions implemented while integrating Qwen2-VL (via vLLM) into the FrameWright video analysis pipeline.

## 1. The "Hallucination" Problem
**Symptoms**: The model produced detailed but completely incorrect descriptions of video clips (e.g., describing a garage scene when the video was of Italy).  
**Root Cause**: The Multi-Modal (Vision) adapter was failing silently or being bypassed. The model received only the text prompt, forcing it to hallucinate a response based on the filename or pure chance.

## 2. Dependency Hell: `decord` & `numpy`
**Challenge**: The `vllm` container failed to start or crashed when processing video.
*   **Conflict**: `vllm` requires `numpy<2.0.0` for Numba compatibility, but the base image had a newer version.
*   **Missing Libraries**: The default `vllm` install did not include video decoding libraries.
**Solution**:
*   Pinned `numpy<2.0.0` in `vllm/vllm_requirements.txt`.
*   Added `decord` and `av` (PyAV) to the container to enable video reading.

## 3. Security & Permissions
**Challenge**: `vllm` threw `400 Bad Request` errors when trying to load local video files.
*   **Error**: `Cannot load local files without --allowed-local-media-path`.
**Solution**:
*   Updated `start_server.py` to launch `vllm` with `--allowed-local-media-path /opt/project_root`.
*   Ensured volume mappings in `docker-compose.yml` correctly mirrored the host paths.

## 4. The Request Format Standoff
**Challenge**: Finding a payload format that `vllm`'s OpenAI-compatible server would accept for Video.
*   **Attempt A (`<|video|>` tag)**: Standard for Qwen, but vLLM often ignored it or failed silently.
*   **Attempt B (`type: video`)**: Rejected by the API validator (`Unknown part type: video`).
*   **Attempt C (`type: image_url` with video path)**: Failed because vLLM tried to load the MP4 using PIL (Image library), causing `UnidentifiedImageError`.

**Final Solution: "Force Frame Extraction"**
Instead of relying on vLLM's internal video loader, we moved the frame extraction logic to the **Client Application**.
1.  **Extract**: The App uses `decord` to strictly sample 8 frames from the video.
2.  **Encode**: Frames are converted to base64 JPEGs.
3.  **Send**: The payload is sent as a sequence of `image_url` items.

This strategy treats the video analysis as a "Multi-Image Analysis" task, which is robust and well-supported by the Qwen-VL model architecture.

## 5. Artifacts
*   **Debug Frames**: The client now saves the extracted frames to `outputs/debug_frames/`. This allows manual verification of exactly what the model "sees".
