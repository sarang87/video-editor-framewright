from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st # Reload-Trigger

from app.models import AnalysisCriteria, ClipMetadata
from app.pipeline import analyze_videos
from app.services.database import DuckDBManager
from app.services.ingest import IngestionPipeline
from app.services.analyzer import VideoAnalyzer
from app.agent.service import AgentService


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
    if "agent_service" not in st.session_state:
        st.session_state["agent_service"] = AgentService()

    # Tabs for different functions
    tab_qa, tab_brainstorm, tab_library = st.tabs(["Q&A Analysis", "Cinematic Brainstorming", "Clip Library"])

    with tab_qa:
        render_qa_tab()
    
    with tab_brainstorm:
        render_brainstorm_tab()

    with tab_library:
        render_library_tab()

def render_library_tab():
    st.markdown("### 📚 Clip Library")
    st.markdown("<div class='subtle'>Browse all analyzed clips and their metadata.</div>", unsafe_allow_html=True)
    
    db_manager = st.session_state["db_manager"]
    
    try:
        df = db_manager.get_all_clips()
        if not df.empty:
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "clip_name": "Clip Name",
                    "category": "Category",
                    "visual_description": "Visual Description",
                    "shot_type": "Shot Type",
                    "motion_detected": "Motion",
                    "narrative_utility": "Narrative Utility",
                    "transition_point": "Cut Point",
                    "analyzed_at": st.column_config.DatetimeColumn("Analyzed At", format="D MMM YYYY, h:mm a"),
                },
                hide_index=True,
            )
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "clips_library.csv",
                "text/csv",
                key="download-csv"
            )
        else:
            st.info("No clips found in the library. Upload and analyze videos to populate this list.")
            
    except Exception as e:
        st.error(f"Error loading library: {e}")

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
    st.markdown("### Narrative Brainstorming Agent")
    st.markdown("<div class='subtle'>Chat with the AI to find clips and build an edit plan.</div>", unsafe_allow_html=True)
    
    agent = st.session_state["agent_service"]
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What do you want to create? (e.g. 'Find happy clips')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream response from agent
            # The agent returns events, we need to parse them.
            # For MVP, we'll likely get a final response or intermediate steps.
            # Let's assume the agent yields dicts with 'messages' or 'timeline'.
            
            try:
                for event in agent.stream_chat(prompt):
                    # Inspect event to see what node executed
                    # This is highly dependent on LangGraph output format
                    # Usually event is like {'agent': {'next': 'search'}, ...}
                    
                    # We are looking for the final response or state updates.
                    # For now, let's look for 'messages' in the values.
                    
                    for node_name, node_state in event.items():
                        if node_state is None:
                            continue
                        
                        if "messages" in node_state:
                            # It's a list of messages, get the last one if it's AI
                            last_msg = node_state["messages"][-1]
                            # If it's a tool output, maybe we show it?
                            # If it's AI message, we show it.
                            if hasattr(last_msg, "content") and last_msg.content:
                                full_response = last_msg.content
                                message_placeholder.markdown(full_response + "▌")
                        
                        if "timeline" in node_state and node_state["timeline"]:
                            # If timeline is updated, show it specially
                            timeline = node_state["timeline"]
                            st.markdown("---")
                            st.markdown("### 🎬 Proposed Edit Plan")
                            st.json(timeline)
                            st.markdown("---")

                message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Agent Error: {str(e)}")
                full_response = f"I encountered an error: {str(e)}"
                message_placeholder.markdown(full_response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
