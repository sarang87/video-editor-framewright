import duckdb
import os
from pathlib import Path
from typing import List
import pandas as pd
from app.models import ClipMetadata
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

class DuckDBManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
             db_path = os.getenv("DUCKDB_PATH", "clips.duckdb")
        self.db_path = db_path
        self.conn = None
        self.init_db()
        logger.info(f"Initialized DuckDB at {self.db_path}")

    def _get_connection(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
        return self.conn

    def _migrate_schema(self):
        """Add new columns if they don't exist."""
        conn = self._get_connection()
        try:
            conn.execute("ALTER TABLE clips ADD COLUMN transition_point VARCHAR")
            logger.info("Migrated schema: Added transition_point column")
        except duckdb.CatalogException:
            # Column likely already exists
            pass
        except Exception as e:
            # Other errors
            logger.debug(f"Schema migration note: {e}")

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
                transition_point VARCHAR,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._migrate_schema()

    def insert_clip(self, metadata: ClipMetadata):
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO clips (clip_name, category, visual_description, shot_type, motion_detected, narrative_utility, transition_point)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.clip_name,
            metadata.category,
            metadata.visual_description,
            metadata.shot_type,
            metadata.motion_detected,
            metadata.narrative_utility,
            metadata.transition_point
        ))
        logger.info(f"Inserted metadata for clip: {metadata.clip_name}")

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
