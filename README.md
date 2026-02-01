# Gemini Video Q&A POC

Streamlit app that analyzes videos with Gemini and outputs user-style questions.

## Local Setup
1. Create venv: `python -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install deps: `pip install -r requirements.txt`
4. Set API key: `export GOOGLE_API_KEY=...` or `export OPENAI_API_KEY=...`
5. For Ollama: ensure `ollama serve` is running and `ffmpeg` is installed
5. Run: `streamlit run app/streamlit_app.py`

## Docker
1. `docker compose up --build`
2. Open `http://localhost:8501`

## vLLM (RTX 5070 Ti / 16GB VRAM)
The project includes a dedicated `vllm` service for high-performance inference.

### 1. Start the service
```bash
docker compose up -d vllm
```

### 2. Run the model
Run the following inside the container (via `docker compose exec vllm bash`) to start the server:
```bash
export VLLM_USE_V1=0
vllm serve "Qwen/Qwen3-VL-8B-Instruct-FP8" \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

**Key Optimizations:**
- **Persistent Cache**: Model weights are stored in `./hf_cache`.
- **FP8 Quantization**: Mandatory for 16GB VRAM cards to avoid OOM.
- **V0 Engine**: `VLLM_USE_V1=0` is required for stability on current builds.
- **IPC Host**: Enabled in `docker-compose.yml` for shared memory access.

