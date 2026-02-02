from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st

from app.models import AnalysisCriteria, ClipMetadata
from app.pipeline import analyze_videos
from app.services.database import DuckDBManager
from app.services.ingest import IngestionPipeline
from app.services.analyzer import VideoAnalyzer


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .app-header {
            font-size: 2.0rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subtle {
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
        }
        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .status-ok {
            color: #15803d;
            font-weight: 600;
        }
        .status-fail {
            color: #b91c1c;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _criteria_form(existing: AnalysisCriteria) -> AnalysisCriteria:
    tone_options = ["Professional", "Casual", "Cinematic", "Analytical"]
    depth_options = ["Low", "Medium", "High"]
    tone_index = tone_options.index(existing.tone) if existing.tone in tone_options else 0
    depth_index = depth_options.index(existing.depth) if existing.depth in depth_options else 1

    with st.form("criteria_form"):
        st.markdown("### Questions to Answer")
        questions_text = st.text_area(
            "Enter questions (one per line)",
            value="\n".join(existing.questions),
            height=150,
            help="Enter one question per line. The AI will analyze the video and answer each question.",
        )
        st.markdown("### Answer Style Settings")
        audience = st.text_input("Audience", value=existing.audience)
        tone = st.selectbox("Tone", tone_options, index=tone_index)
        depth = st.selectbox("Depth", depth_options, index=depth_index)
        extra_instructions = st.text_area("Extra Instructions", value=existing.extra_instructions or "")
        submitted = st.form_submit_button("Save Questions & Settings")

    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
    if not questions:
        questions = existing.questions  # Keep existing if empty
    
    updated = AnalysisCriteria(
        questions=questions,
        audience=audience,
        tone=tone,
        depth=depth,
        extra_instructions=extra_instructions or None,
    )
    if submitted:
        st.session_state["criteria"] = updated
        st.success("Questions and settings saved. You can re-run analysis with updated questions.")
    return updated


def _save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to shared directory and return path."""
    if "temp_videos" not in st.session_state:
        st.session_state["temp_videos"] = {}
    
    # Use file ID as key to avoid duplicates
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_id not in st.session_state["temp_videos"]:
        # Use a directory inside /app so it's shared with vLLM container
        # /app/uploads maps to /opt/project_root/uploads in vLLM
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        target_path = upload_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getvalue())
        
        # Store absolute path
        st.session_state["temp_videos"][file_id] = str(target_path.absolute())
    
    return Path(st.session_state["temp_videos"][file_id])


def main() -> None:
    st.set_page_config(page_title="Gemini Video Q&A POC", layout="wide")
    _apply_styles()

    st.markdown(
        "<div class='subtle'>Provide questions about your videos and get AI-generated answers.</div>",
        unsafe_allow_html=True,
    )

    # Initialize Services
    if "db_manager" not in st.session_state:
        st.session_state["db_manager"] = DuckDBManager()
    if "ingest_pipeline" not in st.session_state:
        # Externally mounted path in Docker
        watch_dir = "/videos_source"
        proxy_dir = "videos/proxies"
        st.session_state["ingest_pipeline"] = IngestionPipeline(watch_dir, proxy_dir)
        st.session_state["ingest_pipeline"].start()
    if "analyzer" not in st.session_state:
        st.session_state["analyzer"] = VideoAnalyzer()

    # Tabs for different functions
    tab_qa, tab_brainstorm = st.tabs(["Q&A Analysis", "Cinematic Brainstorming"])

    with tab_qa:
        render_qa_tab()
    
    with tab_brainstorm:
        render_brainstorm_tab()

def render_qa_tab():
    if "criteria" not in st.session_state:
        st.session_state["criteria"] = AnalysisCriteria()

    criteria = st.session_state["criteria"]

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Configuration")
        col1, col2 = st.columns(2)
        with col1:
            output_dir = st.text_input(
                "Output Folder",
                value=str(Path.cwd() / "outputs"),
                key="output_dir_input",
                help="Local path for generated analysis files.",
            )
            provider = st.selectbox("Provider", ["gemini", "vllm (local)", "openai", "ollama"], index=1)
        
        with col2:
            if provider == "openai":
                api_key_value = os.getenv("OPENAI_API_KEY", "")
            elif provider == "ollama":
                api_key_value = ""
            else:
                api_key_value = os.getenv("GOOGLE_API_KEY", "")
            
            api_key = st.text_input("API Key", value=api_key_value, type="password") if provider != "ollama" else ""
            
            model_options = (
                ["gemini-1.5-pro", "gemini-1.5-flash"]
                if provider == "gemini"
                else ["Qwen/Qwen3-VL-8B-Instruct-FP8"]
                if provider == "vllm (local)"
                else ["gpt-4o-mini", "gpt-4o"]
                if provider == "openai"
                else ["qwen3-vl:latest"]
            )
            model_name = st.selectbox("Model", model_options, index=0)
        
        if provider == "ollama":
            ollama_base_url = st.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        else:
            ollama_base_url = None
            
        dry_run = st.checkbox("Dry Run (no API calls)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    criteria = _criteria_form(criteria)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Video Selection")
    
    uploaded_files = st.file_uploader(
        "Choose video files",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        accept_multiple_files=True,
        help="Select one or more video files to analyze",
    )
    
    if uploaded_files:
        video_paths = []
        for uploaded_file in uploaded_files:
            video_path = _save_uploaded_file(uploaded_file)
            video_paths.append(video_path)
        
        st.markdown("#### Uploaded Videos")
        selected_video = st.radio(
            "Select a video to preview:",
            options=[v.name for v in video_paths],
            key="video_preview_selector",
        )
        
        if selected_video:
            selected_path = next(p for p in video_paths if p.name == selected_video)
            st.video(str(selected_path))
        
        st.markdown("---")
        selected_for_analysis = st.multiselect(
            "Select videos to analyze:",
            options=[v.name for v in video_paths],
            default=[v.name for v in video_paths],
        )
        
        if st.button("Run Analysis on Selected Videos", type="primary", use_container_width=True):
            if not selected_for_analysis:
                st.error("Please select at least one video to analyze.")
            else:
                selected_paths = [p for p in video_paths if p.name in selected_for_analysis]
                with st.spinner("Running analysis..."):
                    results = analyze_videos(
                        video_files=selected_paths,
                        output_dir=output_dir,
                        criteria=criteria,
                        api_key=api_key or None,
                        provider=provider.split(" ")[0],  # Get "vllm" from "vllm (local)"
                        model_name=model_name,
                        ollama_base_url=ollama_base_url,
                        dry_run=dry_run,
                    )
                
                st.markdown("### Results")
                for result in results:
                    status_class = "status-ok" if result.success else "status-fail"
                    st.markdown(
                        f"- <span class='badge'>{result.video_name}</span> "
                        f"<span class='{status_class}'>"
                        f"{'Success' if result.success else 'Failed'}"
                        f"</span><br/>Output: `{result.output_path}`",
                        unsafe_allow_html=True,
                    )
                    if result.error:
                        st.code(result.error)
                    if result.success:
                        with open(result.output_path, "r") as f:
                            st.markdown("**Analysis Results:**")
                            st.markdown(f.read())
    else:
        st.info("👆 Please upload video files to get started.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_brainstorm_tab():
    st.markdown("### Narrative Brainstorming Interface")
    
    db = st.session_state["db_manager"]
    analyzer = st.session_state["analyzer"]

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Registry & Search")
        search_query = st.text_input("Find clips (e.g., 'sunset', 'fast motion')")
        if search_query:
            clips = db.search_context(search_query)
        else:
            clips = db.get_all_clips()
        
        st.dataframe(clips, use_container_width=True)
        
        if st.button("🔄 Sync Proxies & Analyze New"):
            proxy_dir = Path("videos/proxies")
            proxies = list(proxy_dir.glob("*_proxy.mp4"))
            new_count = 0
            with st.spinner(f"Analyzing {len(proxies)} proxies..."):
                for proxy in proxies:
                    # Check if already in DB
                    exists = db.conn.execute("SELECT 1 FROM clips WHERE clip_name = ?", (proxy.name,)).fetchone()
                    if not exists:
                        metadata = analyzer.analyze_clip(str(proxy))
                        if metadata:
                            db.insert_clip(metadata)
                            new_count += 1
            st.success(f"Added {new_count} new clips to registry.")

    with col2:
        st.markdown("#### Narrative Goal")
        user_request = st.text_area(
            "What do you want to build?",
            placeholder="e.g., I want to build a high-energy travel montage...",
            height=100
        )
        
        if st.button("Generate Edit Plan", type="primary"):
            if not user_request:
                st.error("Please enter a narrative goal.")
            else:
                with st.spinner("Brainstorming with available footage..."):
                    # Use search context to pull relevant clips for the prompt
                    # For now, we'll just use the search query if provided, or top 5
                    relevant_clips = clips.head(10).to_json(orient="records")
                    query_term = search_query if search_query else "general"
                    plan = analyzer.brainstorm_narrative(query_term, relevant_clips, user_request)
                    st.markdown("---")
                    st.markdown("### 🎬 Shot List / Edit Plan")
                    st.write(plan)


if __name__ == "__main__":
    main()

