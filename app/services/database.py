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
            pass
        except Exception as e:
            logger.debug(f"Schema migration note (transition_point): {e}")

        try:
            conn.execute("ALTER TABLE clips ADD COLUMN duration FLOAT")
            logger.info("Migrated schema: Added duration column")
        except duckdb.CatalogException:
            pass
        except Exception as e:
            logger.debug(f"Schema migration note (duration): {e}")

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
                duration FLOAT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._migrate_schema()

    def insert_clip(self, metadata: ClipMetadata):
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO clips (clip_name, category, visual_description, shot_type, motion_detected, narrative_utility, transition_point, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.clip_name,
            metadata.category,
            metadata.visual_description,
            metadata.shot_type,
            metadata.motion_detected,
            metadata.narrative_utility,
            metadata.transition_point,
            metadata.duration
        ))
        logger.info(f"Inserted metadata for clip: {metadata.clip_name}")

    def clip_exists(self, clip_name: str) -> bool:
        """Check if a clip already exists in the database."""
        conn = self._get_connection()
        result = conn.execute("SELECT count(*) FROM clips WHERE clip_name = ?", (clip_name,)).fetchone()
        return result[0] > 0


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

    def get_schema_info(self) -> str:
        """
        Returns a string representation of the database schema (tables and columns).
        Useful for the Agent to understand what it can query.
        """
        conn = self._get_connection()
        try:
            # Get all tables
            tables_df = conn.execute("SHOW TABLES").df()
            if tables_df.empty:
                return "No tables found in database."
            
            schema_str = []
            for _, row in tables_df.iterrows():
                table_name = row['name']
                # Get columns for each table
                columns_df = conn.execute(f"DESCRIBE {table_name}").df()
                
                columns_str = []
                for _, col_row in columns_df.iterrows():
                    col_name = col_row['column_name']
                    col_type = col_row['column_type']
                    columns_str.append(f"- {col_name} ({col_type})")
                
                table_block = f"Table: {table_name}\n" + "\n".join(columns_str)
                schema_str.append(table_block)
                
            return "\n\n".join(schema_str)
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return f"Error retrieving schema: {str(e)}"

    def get_sample_values(self, table_name: str, column_name: str, limit: int = 10) -> List[str]:
        """
        Returns distinct sample values for a specific column.
        Useful for categorical columns to understand what values are present.
        """
        conn = self._get_connection()
        try:
            query = f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT {limit}"
            result = conn.execute(query).fetchall()
            return [str(row[0]) for row in result]
        except Exception as e:
            logger.error(f"Error getting sample values: {e}")
            return []

    def execute_safe_query(self, sql: str) -> pd.DataFrame:
        """
        Executes a SQL query safely (Read-Only).
        Prevents modification of data by checking for DML keywords.
        """
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        sql_upper = sql.upper().strip()
        
        # specific check: ensure it starts with SELECT or WITH (CTE)
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH") or sql_upper.startswith("SHOW") or sql_upper.startswith("DESCRIBE")):
             raise ValueError("Only SELECT queries are allowed for safety.")

        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                # Basic check - might be too aggressive if 'drop' is in a string literal, but safe for now.
                # A robust parser is better, but this is a V1 guardrail.
                raise ValueError(f"Query contains forbidden keyword: {keyword}")

        conn = self._get_connection()
        try:
            return conn.execute(sql).df()
        except Exception as e:
            logger.error(f"SQL Execution Error: {e}")
            raise e
