import duckdb
import os
from typing import List, Optional
from langchain_core.tools import tool
from app.services.database import DuckDBManager
from app.core.config import settings

# Initialize DB Manager
# Note: Ideally this should be a singleton or passed via context, but for simplicity we instantiate here
db_manager = DuckDBManager()

@tool
def sql_search(query: str):
    """
    Executes a DuckDB SQL query to find video clips.
    The table name is 'clips'.
    Columns: clip_name, category, visual_description, shot_type, motion_detected, narrative_utility, transition_point.
    
    Example queries:
    - SELECT * FROM clips WHERE visual_description ILIKE '%smile%'
    - SELECT clip_name, narrative_utility FROM clips WHERE category = 'B-roll'
    """
    try:
        conn = db_manager._get_connection()
        # Safety check: simplistic read-only check
        if "drop" in query.lower() or "delete" in query.lower() or "update" in query.lower() or "insert" in query.lower():
             return "Error: Read-only queries allowed."
             
        df = conn.execute(query).df()
        if df.empty:
            return "No clips found matching the query."
        return df.to_dict(orient='records')
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def validate_clips(clip_names: List[str]):
    """
    Verifies if the specified clips actually exist in the uploads/videos directory.
    Input: List of clip filenames (e.g., ['video1.mp4', 'video2.mov'])
    Returns: List of valid clips and list of missing clips.
    """
    # Assuming videos are in /app/videos inside container, or accessible via relative path
    # The 'videos' directory is distinct from 'uploads' in the original structure, 
    # but let's check the known locations.
    
    # Check both 'videos' and 'uploads' just in case
    search_paths = ["./videos", "./uploads"]
    valid_clips = []
    missing_clips = []
    
    for name in clip_names:
        found = False
        for path_str in search_paths:
            path = os.path.join(path_str, name)
            if os.path.exists(path):
                valid_clips.append(name)
                found = True
                break
        if not found:
            missing_clips.append(name)
            
    return {
        "valid": valid_clips,
        "missing": missing_clips
    }
