# Technical Analysis and Context Management Review

## 1. Project Overview & Control Flow

### Architecture
The project follows a **Split-Service Architecture**:
- **Frontend/Orchestration**: A Streamlit application running in a Docker container (`app-1`). It handles user interaction, narrative planning (Agent), and database management (DuckDB).
- **Inference Backend**: A vLLM container (`vllm-1`) providing an OpenAI-compatible API for the vision-language model (`Qwen2-VL-7B`).
- **Data Layer**:
    - **Filesystem**: Shared volume for raw videos and proxies.
    - **Metadata**: DuckDB dictionary for storing clip analysis results.

### Control Flow
1. **Ingestion**: `IngestionPipeline` watches for new videos, generates lightweight proxies (`ffmpeg`), and triggers analysis.
2. **Analysis**: `VideoAnalyzer` sends proxy frames to vLLM to extract metadata (visual description, category, etc.).
3. **Agent Loop** (`app/agent/graph.py`):
    - **State**: managed by `LangGraph` (`FilmState`).
    - **Input**: User raw text intent.
    - **Search**: Executes raw SQL `ILIKE` queries against DuckDB to find candidate clips.
    - **Planning**: Uses `DSPy` (`NarrativePlanner`) to generate an edit plan (JSON) based on the search results.
    - **Output**: Streamed back to the Streamlit UI.

## 2. Environment Setup
- **Docker**: The `Dockerfile` builds a lightweight Python 3.11 environment.
    - Installs system deps (`ffmpeg`).
    - Installs Python deps (`langgraph`, `dspy-ai`, `duckdb`).
    - Exposes port 8501 for Streamlit.
- **Communication**: The app container communicates with `vllm:8000` via HTTP.

## 3. Context Management Analysis

### Logging & Input
- **User Input**: Captured via `st.chat_input` and passed directly to the agent state as `user_intent`.
- **Logging**: Standard Python `logging` is used. There is no structured event tracing (e.g., Jaeger/Otel) for deep introspection of the agent's "thought process" aside from debug logs.

### Intent Understanding
- **Current Layout**: Minimal. The system relies on:
    1.  **Keyword Extraction**: `search_node` splits user input, removes stop words, and runs a SQL `OR` query.
    2.  **DSPy**: The planner uses the raw user intent in the prompt: `user_intent` field in `NarrativePlannerSignature`.
- **Limitation**: No semantic understanding. A query like "clips with joyous atmosphere" might fail if the description uses "happy" but not "joyous", unless the simple keyword overlap catches it.

### Context Window & Truncation
- **Mechanism**: **Naive Truncation**.
- **Implementation**: In `app/agent/planner.py`:
  ```python
  if len(context_str) > 10000:
      context_str = context_str[:10000] + "... (truncated)"
  ```
- **Consequence**: If search returns many matching clips, the tail end of the results is simply cut off. The LLM never sees them. This is deterministic but lossy.

### Chunking & Reasoning
- **Current State**: No true chunking or paging.
- **Question Answered**: *Is Multihop reasoning over paged data the best option?*
    - **Short Answer**: It is a valid strategy, but likely **overkill** as a first step and potentially slow.
    - **Better Alternative (Phase 1)**: **Semantic Search (Embeddings) + Map-Reduce**.
        1.  **Embeddings**: Store vector embeddings of clip descriptions in duckdb (using `duckdb`'s vector extension or a separate index).
        2.  **Retrieval**: Retrieve the top-k (e.g., 50) most semantically relevant clips. This fits in context better than random SQL results.
        3.  **Map-Reduce / Refine**: If >50 clips are needed, page them in batches of 20 to the LLM, asking "Do these clips fit the narrative?", then aggregate the selected ones.
    - **When Multihop is best**: If the user asks "Find the clip where the car crashes, AND THEN show the reaction shot from the *same* scene". This requires reasoning across clips (relationships). Paging is necessary here if the search space is large.

## 4. Recommended Improvements roadmap

1.  **Implement Vector Search**: Replace/Augment SQL `ILIKE` with semantic search to improve retrieval quality.
2.  **Hybrid Context Management**:
    - **Relevance Filter**: Rank clips by semantic score.
    - **Smart Paging**: If `len(results) > limit`, implement a `Refusal` or `Paging` loop where the agent summarizes the first batch and decides if it needs to see more.
3.  **Structured Intent Parsing**: Use an LLM step *before* search to extract specific filters (e.g., "outdoor", "night", "action") to pass to the SQL query, rather than just keywords.
