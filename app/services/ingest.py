import ffmpeg
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app.utils.logger import setup_logging

logger = setup_logging(__name__)

class VideoIngestHandler(FileSystemEventHandler):
    def __init__(self, proxy_dir: Path):
        self.proxy_dir = proxy_dir
        self.proxy_dir.mkdir(parents=True, exist_ok=True)

    def on_created(self, event):
        filename = os.path.basename(event.src_path)
        if not event.is_directory and not filename.startswith('.') and \
           event.src_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')):
            logger.info(f"New video detected: {event.src_path}")
            self.generate_proxy(Path(event.src_path))

    def generate_proxy(self, video_path: Path):
        proxy_path = self.proxy_dir / f"{video_path.stem}_proxy.mp4"
        if proxy_path.exists():
            logger.info(f"Proxy already exists for {video_path.name}")
            return

        try:
            logger.info(f"Generating proxy for {video_path.name}...")
            # Using CPU encoding (libx264) to save VRAM for vLLM
            # 640x480, 1 FPS, no audio, 30s limit
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(proxy_path),
                    vf="scale=640:480,fps=1",
                    vcodec="libx264",
                    preset="ultrafast",
                    an=None,
                    t=30
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Proxy generated: {proxy_path}")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"Unexpected error during proxy generation: {e}")

class IngestionPipeline:
    def __init__(self, watch_dir: str, proxy_dir: str):
        self.watch_dir = Path(watch_dir).resolve()
        self.proxy_dir = Path(proxy_dir).resolve()
        self.handler = VideoIngestHandler(self.proxy_dir)
        self.observer = Observer()

    def start(self):
        self.running = True
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        logger.info(f"Watching directory: {self.watch_dir}")
        
        # Process existing videos in a background thread to avoid blocking
        import threading
        threading.Thread(target=self._process_existing, daemon=True).start()

    def _process_existing(self):
        logger.info("Processing existing videos...")
        # Sort files to process in a deterministic order (e.g. alphabetical)
        # This helps if we restart, we can skip existing ones properly (since handler checks existence)
        files = sorted(self.watch_dir.iterdir())
        
        for video in files:
            if not self.running:
                logger.info("Processing loop stopped by user.")
                break
                
            if not video.name.startswith('.') and video.suffix.lower() in ('.mp4', '.mov', '.mkv', '.avi', '.webm'):
                self.handler.generate_proxy(video)

    def stop(self):
        self.running = False
        self.observer.stop()
        self.observer.join()
        logger.info("Ingestion pipeline stopped.")

    def update_watch_dir(self, new_dir: str):
        """Update the directory being watched at runtime."""
        new_path = Path(new_dir).resolve()
        
        if not new_path.exists():
            raise FileNotFoundError(f"Directory not found: {new_path}")
            
        if new_path == self.watch_dir:
            return  # No change

        logger.info(f"Switching watch directory from {self.watch_dir} to {new_path}")
        
        # Stop current observer
        self.observer.unschedule_all()
        self.observer.stop()
        self.observer.join()
        
        # Update path
        self.watch_dir = new_path
        
        # Restart observer
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        
        # Trigger processing of existing files in the new directory
        import threading
        # Ensure running is true before restarting loop
        self.running = True 
        threading.Thread(target=self._process_existing, daemon=True).start()


if __name__ == "__main__":
    # For standalone testing
    watch_folder = "./videos"
    proxy_folder = "./videos/proxies"
    os.makedirs(watch_folder, exist_ok=True)
    
    pipeline = IngestionPipeline(watch_folder, proxy_folder)
    pipeline.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
