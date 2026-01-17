# Video Analysis Q&A POC

Streamlit app that analyzes videos using AI (Gemini, OpenAI, or Ollama) to answer user-provided questions about video content.

## Features

- **Multiple AI Providers**: Support for Gemini, OpenAI, and Ollama (local)
- **Question-Based Analysis**: Provide your own questions and get AI-generated answers
- **File Dialog Upload**: Upload videos directly through the web interface
- **Video Preview**: Preview videos before running analysis
- **Editable Questions**: Edit/save questions with a clean UI
- **Multiple Video Support**: Analyze one or multiple videos at once
- **Markdown Output**: Results saved as markdown files

## Local Setup

### Prerequisites

- Python 3.8+
- For Ollama: `ollama` and `ffmpeg` installed
  - Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)
  - Run Ollama server: `ollama serve`

### Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set API keys (optional - can also enter in UI):**
   ```bash
   export GOOGLE_API_KEY=your_gemini_key_here
   # OR
   export OPENAI_API_KEY=your_openai_key_here
   ```

### Running the Application

**Option 1: Using the run script (recommended):**
```bash
./run.sh
```

**Option 2: Manual command:**
```bash
source .venv/bin/activate
PYTHONPATH=$(pwd):$PYTHONPATH streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Docker Setup

1. **Build and run:**
   ```bash
   docker compose up --build
   ```

2. **Access the app:**
   Open `http://localhost:8501` in your browser

   Note: For Ollama with Docker, ensure Ollama is accessible from the container or use host networking.

## Usage

1. **Upload Videos**: Use the file uploader to select video files (supports .mp4, .mov, .mkv, .avi, .webm)
2. **Preview Videos**: Select a video from the radio buttons to preview it
3. **Configure Questions**: 
   - Click "Edit" to modify questions (one per line)
   - Click "Save" to persist your questions
4. **Configure Analysis Settings**: Set audience, tone, depth, and optional instructions
5. **Select Provider**: Choose Gemini, OpenAI, or Ollama and select a model
6. **Run Analysis**: Select videos to analyze and click "Run Analysis on Selected Videos"
7. **View Results**: Results are displayed inline and saved to the output folder as `.answers.md` files

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py      # Main Streamlit UI
│   ├── pipeline.py            # Video processing pipeline
│   ├── models.py              # Data models
│   └── services/
│       ├── gemini_client.py   # Gemini API client
│       ├── openai_client.py  # OpenAI API client
│       └── ollama_client.py  # Ollama API client
├── outputs/                   # Analysis results (created automatically)
├── videos/                     # Optional: input folder (if using folder-based approach)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Output Format

Analysis results are saved as markdown files with:
- Questions asked
- AI-generated answers for each question
- Formatted for easy reading and sharing

