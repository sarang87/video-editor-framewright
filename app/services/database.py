import duckdb
from pathlib import Path
from typing import List
import pandas as pd
from app.models import ClipMetadata

class DuckDBManager:
    def __init__(self, db_path: str = "clips.duckdb"):
        self.db_path = db_path
        self.conn = None
        self.init_db()

    def _get_connection(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
        return self.conn

    def init_db(self):
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clips (
                clip_name VARCHAR,
                category VARCHAR,
                visual_description TEXT,
                shot_type VARCHAR,
                motion_detected VARCHAR,
                narrative_utility TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def insert_clip(self, metadata: ClipMetadata):
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO clips (clip_name, category, visual_description, shot_type, motion_detected, narrative_utility)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metadata.clip_name,
            metadata.category,
            metadata.visual_description,
            metadata.shot_type,
            metadata.motion_detected,
            metadata.narrative_utility
        ))

    def search_context(self, query: str) -> pd.DataFrame:
        """Simple keyword search over visual descriptions."""
        conn = self._get_connection()
        # Using FTS-like simple keyword match or just LIKE for now
        sql = f"SELECT * FROM clips WHERE visual_description ILIKE '%{query}%' OR category ILIKE '%{query}%'"
        return conn.execute(sql).df()

    def get_all_clips(self) -> pd.DataFrame:
        conn = self._get_connection()
        return conn.execute("SELECT * FROM clips ORDER BY analyzed_at DESC").df()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
