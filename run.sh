#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
streamlit run app/streamlit_app.py

