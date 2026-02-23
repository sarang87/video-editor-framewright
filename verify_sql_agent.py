import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
# Set env var BEFORE importing app modules so they pick it up
os.environ["DUCKDB_PATH"] = "test_clips.duckdb"

import dspy
from app.services.database import DuckDBManager
from app.agent.graph import app
from langchain_core.messages import HumanMessage
from app.models import ClipMetadata

# Configure DSPy
lm = dspy.LM(model="openai/Qwen/Qwen3-VL-8B-Instruct-FP8", api_base="http://vllm:8000/v1", api_key="token-is-ignored")
dspy.configure(lm=lm)

# 1. Setup Mock Data
print("--- Setting up Database ---")
# Instantiate directly or rely on the one tools will create? 
# We need to populate it.
db = DuckDBManager("test_clips.duckdb")
# Clean slate
db._get_connection().execute("DROP TABLE IF EXISTS clips")
db.init_db()

# Create mock clips
clips = []
for i in range(1, 16):
    clips.append(ClipMetadata(
        clip_name=f"clip{i}.mp4",
        category="B-Roll",
        visual_description=f"Drone shot of mountains and nature {i}",
        shot_type="Wide",
        motion_detected="High",
        narrative_utility="Establishing",
        transition_point="Start",
        duration=5.0 + i
    ))

# Add a massive clip to test context truncation
clips.append(ClipMetadata(
    clip_name="massive_clip.mp4",
    category="B-Roll",
    visual_description="Drone shot of mountains " * 500, # 24 chars * 500 = 12,000 chars
    shot_type="Wide",
    motion_detected="High",
    narrative_utility="Stress Test",
    transition_point="Start",
    duration=100.0
))

# Insert into DB
# Ensure table has duration (init_db called in __init__ handles this via migrate)
print("Inserting mock clips...")

for clip in clips:
    # Check if exists to avoid dupe in rerun
    if not db.clip_exists(clip.clip_name):
        db.insert_clip(clip)

print(f"Inserted {len(clips)} sample clips.")

# 2. Run Agent
print("\n--- Running Agent ---")
# We want to test if it selects duration and doesn't limit results
user_input = "Find me a drone shot of mountains and include the duration in the results."

inputs = {
    "messages": [HumanMessage(content=user_input)],
    "user_intent": user_input
}

print(f"User Query: {user_input}")

# Overwrite the db_manager in tools.py effectively? 
# The tools.py creates its own DuckDBManager instance.
# We need to make sure the tools use *our* test db, or we point tools to it.
# app/agent/tools.py instantiates `db_manager = DuckDBManager()`.
# It uses env var DUCKDB_PATH.
os.environ["DUCKDB_PATH"] = "test_clips.duckdb"

# Force reload tools to pick up new DB path? 
# Python modules are cached. 
# Better: We'll rely on the fact that we set the env var *before* running this script if we run via command line.
# But here we are importing inside the script.
# We might need to monkeypatch tools.db_manager
from app.agent import tools
tools.db_manager = db 
# Also valid_clips tool uses os.path.exists. We can mock it or just assume it returns missing.

for event in app.stream(inputs, config={"configurable": {"thread_id": "test_thread"}}):
    for key, value in event.items():
        print(f"\nNode: {key}")
        if "messages" in value:
            last_msg = value["messages"][-1]
            if hasattr(last_msg, "content"):
                print(f"Msg Content: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"Tool Calls: {last_msg.tool_calls}")

print("\n--- Done ---")
