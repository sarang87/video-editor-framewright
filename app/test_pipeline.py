import os
import time
import logging
from pathlib import Path
from app.services.database import DuckDBManager
from app.services.ingest import IngestionPipeline
from app.services.analyzer import VideoAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPipeline")

def test_integration():
    # 1. Setup temporary directories for testing
    test_watch_dir = Path("./videos_test")
    test_proxy_dir = Path("./videos_test/proxies")
    test_watch_dir.mkdir(exist_ok=True)
    
    db_path = "test_clips.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = DuckDBManager(db_path=db_path)
    analyzer = VideoAnalyzer(base_url="http://localhost:8000/v1")
    
    # 2. Start Ingestion Pipeline
    logger.info("Starting ingestion pipeline...")
    pipeline = IngestionPipeline(str(test_watch_dir), str(test_proxy_dir))
    pipeline.start()
    
    try:
        # 3. Create a dummy video file to trigger ingestion
        dummy_video = test_watch_dir / "test_clip.mp4"
        if not dummy_video.exists():
            # We can't actually generate a proxy from a zero-byte file with ffmpeg usually,
            # but we can see if the watchdog picks it up.
            dummy_video.write_text("dummy video content")
        
        logger.info(f"Created dummy video: {dummy_video}")
        
        # Give it a moment to pick up
        time.sleep(2)
        
        # 4. Check if proxy generation was attempted (it might fail due to dummy content, which is fine)
        # We can also manually trigger the analyzer check if a proxy existed
        
        logger.info("Testing Database Insert/Search...")
        from app.models import ClipMetadata
        test_meta = ClipMetadata(
            clip_name="test_clip.mp4",
            category="B-roll",
            visual_description="A test clip of a sunset over the Mediterranean.",
            shot_type="Wide Shot",
            motion_detected="Slow Pan",
            narrative_utility="Establishing shot for Italy montage."
        )
        db.insert_clip(test_meta)
        
        results = db.search_context("sunset")
        logger.info(f"Search Results: \n{results}")
        
        if not results.empty:
            logger.info("Search SUCCESS")
        else:
            logger.error("Search FAILED")

        # 5. Test Brainstorming (if vLLM is up)
        # logger.info("Testing Brainstorming...")
        # plan = analyzer.brainstorm_narrative("sunset", results.to_json(orient="records"), "Create a travel vlog intro")
        # logger.info(f"Brainstorming Plan: \n{plan}")

    finally:
        pipeline.stop()
        db.close()
        # Clean up
        if dummy_video.exists():
            os.remove(dummy_video)
        if os.path.exists(db_path):
            os.remove(db_path)
        logger.info("Test complete.")

if __name__ == "__main__":
    test_integration()
