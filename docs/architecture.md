# System Architecture & Data Flow

## System Overview
Framewright processes video content through a split-service architecture, leveraging a lightweight frontend for user interaction and a heavy GPU-accelerated backend for inference.

### High-Level Diagram
```mermaid
graph TD
    User[User] -->|Uploads/Interacts| Frontend[App Container (Streamlit)]
    Frontend -->|Reads| FileSys[Shared Filesystem]
    Frontend -->|Stores Meta| DB[(DuckDB)]
    
    subgraph "Inference Backend"
        Frontend -->|HTTP Request (Images)| vLLM[vLLM Container]
        vLLM -->|Loads Model| GPU[NVIDIA GPU]
        vLLM -->|Returns JSON| Frontend
    end

    subgraph "Agent Layer"
        Frontend -->|Chat Stream| Agent[LangGraph Agent]
        Agent -->|Plan| DSPy[DSPy Planner]
        Agent -->|Query| DB
    end

    FileSys -->|Raw Video| Frontend
```

## Data Flow: "Cinematic Brainstorming"

1.  **Ingestion**:
    *   User uploads `video.mp4` via Streamlit.
    *   App saves file to `./uploads` (Shared Volume).

2.  **Preprocessing (Client-Side)**:
    *   `app/services/analyzer.py` uses `decord` to open the video.
    *   Extracts **16 frames** uniformly sampled from the clip.
    *   Resizes images to 768px (max dimension) to optimize token usage.
    *   Encodes frames as Base64 strings.

3.  **Inference**:
    *   App constructs a prompt: *"You are an editing assistant..."*.
    *   App sends HTTP POST to `http://vllm:8000/v1/chat/completions`.
    *   Payload includes: System Prompt + User Message (Text + 16 Image URLs).
    *   Model (`Qwen2-VL-7B-Instruct-FP8`) processes the multimodal input.

4.  **Structured Output**:
    *   Model returns a JSON string strictly enforcing the schema:
        *   `category`: A-Roll / B-Roll
        *   `camera_motion`: Pan/Tilt/Static
        *   `transition_point`: Suggested cut frame
    *   App parses JSON into a Pydantic model (`ClipMetadata`).

5.  **Persistence**:
    *   App saves `ClipMetadata` to `clips.duckdb`.
    *   Data is queryable for future "Edit Plan" generation.

## Component Details

### Frontend (`app/`)
*   **Language**: Python 3.11
*   **Framework**: Streamlit
*   **Key Libs**: `streamlit`, `decord`, `duckdb`, `openai` (client)
*   **Role**: Orchestration, UI, Preprocessing, DB Management.

### Agent Layer (`app/agent/`)
*   **Frameworks**: LangGraph, DSPy
*   **Role**: Stateful conversation, Narrative Planning, SQL Generation.
*   **See**: [Agent Architecture](agent_architecture.md)

### Backend (`vllm/`)
*   **Engine**: vLLM (Versatile Large Language Model)
*   **Model**: Qwen/Qwen3-VL-8B-Instruct-FP8
*   **Optimization**: FP8 Quantization, persistent HF cache.
*   **Role**: High-performance Inference.

### Storage
*   `./uploads`: Raw video files.
*   `./clips.duckdb`: Metadata and analysis results.
*   `./logs`: Centralized debug logs.
