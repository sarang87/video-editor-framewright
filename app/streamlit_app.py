from __future__ import annotations

import os
from pathlib import Path
from typing import List

import streamlit as st

from app.models import AnalysisCriteria
from app.pipeline import analyze_videos, list_videos


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


def _list_video_names(input_dir: str) -> List[str]:
    return [video.name for video in list_videos(input_dir)]


def main() -> None:
    st.set_page_config(page_title="Gemini Video Q&A POC", layout="wide")
    _apply_styles()

    st.markdown("<div class='app-header'>Video Analysis Q&A POC</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Provide questions about your videos and get AI-generated answers.</div>",
        unsafe_allow_html=True,
    )

    if "criteria" not in st.session_state:
        st.session_state["criteria"] = AnalysisCriteria()

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        input_dir = st.text_input(
            "Input Folder",
            value=str(Path.cwd() / "videos"),
            help="Local path containing video files.",
        )
        output_dir = st.text_input(
            "Output Folder",
            value=str(Path.cwd() / "outputs"),
            help="Local path for generated analysis files.",
        )
        provider = st.selectbox("Provider", ["gemini", "openai", "ollama"], index=0)
        api_key_label = (
            "Gemini API Key (or set GOOGLE_API_KEY env var)"
            if provider == "gemini"
            else "OpenAI API Key (or set OPENAI_API_KEY env var)"
        )
        if provider == "openai":
            api_key_value = os.getenv("OPENAI_API_KEY", "")
        elif provider == "ollama":
            api_key_value = ""
        else:
            api_key_value = os.getenv("GOOGLE_API_KEY", "")
        api_key = st.text_input(api_key_label, value=api_key_value, type="password") if provider != "ollama" else ""
        model_options = (
            ["gemini-1.5-pro", "gemini-1.5-flash"]
            if provider == "gemini"
            else ["gpt-4o-mini", "gpt-4o"]
            if provider == "openai"
            else ["qwen3-vl:latest"]
        )
        model_name = st.selectbox("Model", model_options, index=0)
        ollama_base_url = (
            st.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
            if provider == "ollama"
            else None
        )
        dry_run = st.checkbox("Dry Run (no API calls)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    criteria = _criteria_form(st.session_state["criteria"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Video Selection")
    video_names = _list_video_names(input_dir)
    if not video_names:
        st.warning("No video files found in the input folder.")
        selected = []
    else:
        selected = st.multiselect("Select videos to analyze", options=video_names, default=video_names)

    col_all, col_selected = st.columns(2)
    run_all = col_all.button("Run Analysis on All Videos")
    run_selected = col_selected.button("Run Analysis on Selected Videos")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_all or run_selected:
        chosen = video_names if run_all else selected
        if not chosen:
            st.error("No videos selected.")
            return
        with st.spinner("Running analysis..."):
            results = analyze_videos(
                input_dir=input_dir,
                output_dir=output_dir,
                criteria=criteria,
                selected_videos=chosen if not run_all else None,
                api_key=api_key or None,
                provider=provider,
                model_name=model_name,
                ollama_base_url=ollama_base_url,
                dry_run=dry_run,
            )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

