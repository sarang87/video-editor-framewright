import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
# Set env var BEFORE importing app modules so they pick it up
os.environ["DUCKDB_PATH"] = "test_clips.duckdb"

from app.services.database import DuckDBManager
from app.agent.graph import app
from langchain_core.messages import HumanMessage
from app.models import ClipMetadata

# 1. Setup Mock Data
print("--- Setting up Database ---")
# Instantiate directly or rely on the one tools will create? 
# We need to populate it.
db = DuckDBManager("test_clips.duckdb")
# Clean slate
db._get_connection().execute("DROP TABLE IF EXISTS clips")
db.init_db()

# Insert samples
samples = []
for i in range(1, 16):
    samples.append(
        ClipMetadata(
            clip_name=f"clip{i}.mp4", 
            category="B-Roll", 
            visual_description=f"Drone shot of mountains and nature {i}", 
            shot_type="Wide", 
            motion_detected="High", 
            narrative_utility="Establishing", 
            transition_point="Start"
        )
    )

for s in samples:
    db.insert_clip(s)

print(f"Inserted {len(samples)} sample clips.")

# 2. Run Agent
print("\n--- Running Agent ---")
query = "Find me a drone shot of mountains."

inputs = {
    "messages": [HumanMessage(content=query)],
    "user_intent": query
}

print(f"User Query: {query}")

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
