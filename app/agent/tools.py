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
def list_tables_tool():
    """
    Lista all tables in the database.
    Returns: A list of table names.
    """
    try:
        conn = db_manager._get_connection()
        return [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    except Exception as e:
        return f"Error listing tables: {str(e)}"

@tool
def get_schema_tool(table_name: str):
    """
    Get the schema (columns and types) for a specific table.
    Use this to understand what columns are available before querying.
    """
    try:
        conn = db_manager._get_connection()
        df = conn.execute(f"DESCRIBE {table_name}").df()
        return df[['column_name', 'column_type']].to_string(index=False)
    except Exception as e:
        return f"Error getting schema for {table_name}: {str(e)}"

@tool
def get_column_values_tool(table_name: str, column_name: str):
    """
    Get the top 10 distinct values for a specific column.
    Use this to understand what kind of data is in a categorical column (e.g., 'category', 'shot_type').
    """
    try:
        return db_manager.get_sample_values(table_name, column_name)
    except Exception as e:
        return f"Error getting values for {column_name}: {str(e)}"

@tool
def db_query_tool(query: str):
    """
    Execute a read-only SQL query.
    Always check the schema first to ensure correct column names.
    Returns: JSON string of results.
    """
    try:
        df = db_manager.execute_safe_query(query)
        if df.empty:
            return "[]"
        return df.to_json(orient='records')
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
