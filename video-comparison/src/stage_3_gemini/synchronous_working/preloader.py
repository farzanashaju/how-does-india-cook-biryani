import json
import os
import asyncio
import aiofiles
import ijson
import logging
from typing import Dict, Set, List, Tuple, Any
from pathlib import Path
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

from file_manager import GeminiFileManager
from config import setup_logging, DEFAULT_CONFIG


@dataclass
class UploadTask:
    frame_name: str
    local_path: str
    estimated_size: int


class AsyncFramePreuploader:
    def __init__(
        self,
        local_images_path: str,
        gemini_manager: GeminiFileManager,
        max_storage_gb: float = 15.0,
        max_concurrent_uploads: int = 10,
        logger=None,
    ):
        self.local_images_path = Path(local_images_path)
        self.gemini_manager = gemini_manager
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        self.max_concurrent_uploads = max_concurrent_uploads
        self.logger = logger or setup_logging()

        # Thread-safe counters
        self.upload_stats = {"uploaded": 0, "skipped": 0, "failed": 0, "total_size": 0}
        self.stats_lock = threading.Lock()

        # Semaphore to limit concurrent uploads
        self.upload_semaphore = asyncio.Semaphore(max_concurrent_uploads)

    def _update_stats(self, stat_type: str, size: int = 0):
        """Thread-safe stats update."""
        with self.stats_lock:
            self.upload_stats[stat_type] += 1
            if size > 0:
                self.upload_stats["total_size"] += size

    def _get_stats_snapshot(self) -> Dict:
        """Get thread-safe stats snapshot."""
        with self.stats_lock:
            return self.upload_stats.copy()

    def collect_all_frames(self, input_file: str) -> Set[str]:
        """Collect all unique frame names from the JSON file."""
        self.logger.info(f"Scanning {input_file} for all frame references...")
        all_frames = set()

        with open(input_file, "rb") as f:
            for comparison_key, comparison_data in ijson.kvitems(f, ""):
                if not comparison_key.startswith("Action_class_comparison"):
                    continue

                if not isinstance(comparison_data, dict):
                    continue

                clips = comparison_data.get("clips", {})

                # Process both clips
                for clip_id in ["1", "2"]:
                    clip_data = clips.get(clip_id, {})
                    if not isinstance(clip_data, dict):
                        continue

                    video_url = clip_data.get("url", "")
                    if not video_url:
                        continue

                    # Get all frames for this clip
                    retrieval_frames = clip_data.get("retrieval_frames", {})
                    clip_frames = []
                    for stage_frames in retrieval_frames.values():
                        clip_frames.extend(stage_frames)

                    # Generate frame names
                    for frame_num in set(clip_frames):
                        frame_name = self.gemini_manager.format_frame_name(video_url, frame_num)
                        all_frames.add(frame_name)

        self.logger.info(f"Found {len(all_frames)} unique frames to potentially upload")
        return all_frames

    def filter_frames_to_upload(self, all_frames: Set[str]) -> List[UploadTask]:
        """Filter frames that need uploading and estimate storage requirements."""
        self.logger.info("Filtering frames that need uploading...")

        upload_tasks = []
        current_storage = self.gemini_manager.current_storage

        for frame_name in all_frames:
            # Skip if already uploaded
            if frame_name in self.gemini_manager.uploaded_files:
                continue

            # Check if local file exists
            local_path = self.local_images_path / frame_name
            if not local_path.exists():
                continue

            # Get file size
            file_size = local_path.stat().st_size

            # Check if we have space
            if current_storage + file_size > self.max_storage_bytes:
                self.logger.warning(
                    f"Storage limit reached. Stopping at {len(upload_tasks)} frames."
                )
                break

            upload_tasks.append(
                UploadTask(
                    frame_name=frame_name,
                    local_path=str(local_path),
                    estimated_size=file_size,
                )
            )
            current_storage += file_size

        total_size_mb = sum(task.estimated_size for task in upload_tasks) / (1024 * 1024)
        self.logger.info(
            f"Planning to upload {len(upload_tasks)} frames (~{total_size_mb:.1f} MB)"
        )

        return upload_tasks

    async def upload_single_frame(self, task: UploadTask) -> bool:
        """Upload a single frame with semaphore control."""
        async with self.upload_semaphore:
            try:
                # Run the blocking upload in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    success, status = await loop.run_in_executor(
                        executor,
                        self.gemini_manager.upload_frame,
                        task.local_path,
                        task.frame_name,
                    )

                if success:
                    if status == "uploaded":
                        self._update_stats("uploaded", task.estimated_size)
                        self.logger.debug(f"✓ Uploaded {task.frame_name}")
                    else:  # already_exists
                        self._update_stats("skipped")
                        self.logger.debug(f"↷ Skipped {task.frame_name} (exists)")
                    return True
                else:
                    self._update_stats("failed")
                    self.logger.error(f"✗ Failed {task.frame_name}: {status}")
                    return False

            except Exception as e:
                self._update_stats("failed")
                self.logger.error(f"✗ Exception uploading {task.frame_name}: {e}")
                return False

    async def upload_frames_batch(self, upload_tasks: List[UploadTask]) -> None:
        """Upload all frames concurrently."""
        if not upload_tasks:
            self.logger.info("No frames to upload")
            return

        self.logger.info(f"Starting concurrent upload of {len(upload_tasks)} frames...")
        start_time = time.time()

        # Create upload tasks
        upload_coroutines = [self.upload_single_frame(task) for task in upload_tasks]

        # Progress reporting task
        async def report_progress():
            while True:
                stats = self._get_stats_snapshot()
                total_processed = stats["uploaded"] + stats["skipped"] + stats["failed"]
                if total_processed >= len(upload_tasks):
                    break

                progress = (total_processed / len(upload_tasks)) * 100
                self.logger.info(
                    f"Progress: {total_processed}/{len(upload_tasks)} ({progress:.1f}%) - "
                    f"Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}"
                )
                await asyncio.sleep(10)  # Report every 10 seconds

        # Run uploads and progress reporting concurrently
        progress_task = asyncio.create_task(report_progress())

        try:
            results = await asyncio.gather(*upload_coroutines, return_exceptions=True)
            progress_task.cancel()
        except Exception as e:
            progress_task.cancel()
            self.logger.error(f"Batch upload failed: {e}")
            return

        # Final stats
        stats = self._get_stats_snapshot()
        elapsed_time = time.time() - start_time
        total_size_mb = stats["total_size"] / (1024 * 1024)

        self.logger.info(
            f"✅ Upload complete in {elapsed_time:.1f}s:\n"
            f"  Uploaded: {stats['uploaded']} frames ({total_size_mb:.1f} MB)\n"
            f"  Skipped: {stats['skipped']} frames\n"
            f"  Failed: {stats['failed']} frames\n"
            f"  Rate: {stats['uploaded'] / elapsed_time:.1f} uploads/sec"
        )

    async def preload_frames(self, input_file: str) -> None:
        """Main method to preload frames from JSON file."""
        try:
            # Step 1: Collect all frame references
            all_frames = self.collect_all_frames(input_file)

            # Step 2: Filter frames that need uploading
            upload_tasks = self.filter_frames_to_upload(all_frames)

            if not upload_tasks:
                self.logger.info("All frames already uploaded or no frames to upload")
                return

            # Step 3: Upload frames concurrently
            await self.upload_frames_batch(upload_tasks)

        except Exception as e:
            self.logger.error(f"Error in preload_frames: {e}")


def run_preloader():
    """Synchronous wrapper to run the async preloader."""
    # Configuration
    LOCAL_IMAGES_PATH = DEFAULT_CONFIG["LOCAL_IMAGES_PATH"]
    UPLOADED_FILES_JSON = DEFAULT_CONFIG["UPLOADED_FILES_JSON"]

    # Input file path - EDIT THIS
    INPUT_FILE = "../../Data/BatchPipline/Comparison_script/Comparison_batch_1.json"

    # Setup
    logger = setup_logging("frame_preuploader.log")
    gemini_manager = GeminiFileManager(uploaded_files_json=UPLOADED_FILES_JSON, logger=logger)

    preuploader = AsyncFramePreuploader(
        local_images_path=LOCAL_IMAGES_PATH,
        gemini_manager=gemini_manager,
        max_storage_gb=15.0,
        max_concurrent_uploads=8,  # Adjust based on rate limits
        logger=logger,
    )

    # Run async preloader
    asyncio.run(preuploader.preload_frames(INPUT_FILE))


if __name__ == "__main__":
    run_preloader()
