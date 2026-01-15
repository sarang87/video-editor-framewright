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

