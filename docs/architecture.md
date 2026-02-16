# System Architecture & Data Flow

## System Overview
Framewright processes video content through a split-service architecture, leveraging a lightweight frontend for user interaction and a heavy GPU-accelerated backend for inference.

### High-Level Diagram
```mermaid
graph TD
    User[User] -->|Uploads/Interacts| Frontend[App Container (Streamlit)]
    Frontend -->|Reads| FileSys[Shared Filesystem]
    Frontend -->|Stores Meta| DB[(DuckDB)]
    
    subgraph "Ingestion Service"
        FileSys -->|Watch| Ingest[Ingestion Pipeline]
        Ingest -->|Generate| Proxy[Proxy Files]
    end
    
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

## Data Flow: "Cinematic Brainstorming"

1.  **Ingestion (Background)**:
    *   **Observer**: `app/services/ingest.py` watches the configured directory (e.g., `/videos_source`).
    *   **Proxy Gen**: `ffmpeg` converts raw videos to 640x480 proxies (`videos/proxies/*_proxy.mp4`).
    *   This runs continuously and can be stopped/restarted via UI.

2.  **Analysis (Foreground)**:
    *   User clicks "Scan for New Clips" in the Library.
    *   App identifies proxies not yet in `clips.duckdb`.
    *   **Preprocessing**: `app/services/analyzer.py` extracts 16 frames from the *proxy*.
    *   **Encoding**: Frames are resized (768px) and Base64 encoded.
    *   **Inference**: App sends HTTP POST to `http://vllm:8000/v1/chat/completions`.
    *   Payload includes: System Prompt + User Message (Text + 16 Image URLs).
    *   Model (`Qwen2-VL-7B-Instruct-FP8`) processes the multimodal input.
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
