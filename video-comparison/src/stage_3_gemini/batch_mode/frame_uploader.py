import json
import os
import subprocess
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import math
from google.cloud import storage
from google.api_core import retry
from google.api_core.exceptions import (
    TooManyRequests,
    ServiceUnavailable,
    GoogleAPIError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("adaptive_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AdaptivePipelineUploader:
    def __init__(
        self,
        bucket_name: str = "biryanidiff",
        credentials_path: str = "biryani-across-india-d26285fcddf7.json",
        local_frames_dir: str = "extracted_frames",
        registry_file: str = "upload_registry.json",
        upload_interval: int = 180,
        temp_storage_path: str = None,
        frame_threshold: int = 3000,
        adaptive_upload_workers: int = 15,
    ):
        # GCS setup
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

        # Paths
        if temp_storage_path:
            self.local_frames_dir = Path(temp_storage_path) / "extracted_frames"
        else:
            self.local_frames_dir = Path(local_frames_dir)
        self.local_frames_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = registry_file
        self.upload_interval = upload_interval
        self.frame_threshold = frame_threshold
        self.adaptive_upload_workers = adaptive_upload_workers

        # Simple state tracking (no complex threading)
        self.expected_frames = set()
        self.cloud_frames = set()
        self.local_frames = set()
        self.uploaded_frames = set()
        self.failed_frames = set()

        # Processing control
        self.extraction_complete = False
        self.upload_timer = None
        self.max_extraction_workers = 10
        self.normal_upload_workers = 3

    def rebuild_state_from_sources(self, json_path: str):
        """Build fresh state from actual sources - no stale registry cleanup needed."""
        logger.info("=== REBUILDING STATE FROM SOURCES ===")

        # 1. Get expected frames from JSON (ground truth)
        self.expected_frames = self.get_expected_frames_from_json(json_path)
        logger.info(f"Expected frames: {len(self.expected_frames)}")

        # 2. Get existing cloud frames
        self.cloud_frames = self.get_cloud_existing_frames()
        logger.info(f"Cloud frames: {len(self.cloud_frames)}")

        # 3. Get local frames that match expected
        local_files = {f.name for f in self.local_frames_dir.glob("*.png") if f.is_file()}
        self.local_frames = local_files & self.expected_frames
        logger.info(f"Local frames (expected): {len(self.local_frames)}")

        # 4. Load simple registry for failures only
        registry = self.load_simple_registry()
        self.failed_frames = registry.get("failed", set())

        # 5. Calculate what needs processing
        already_done = self.cloud_frames & self.expected_frames
        need_extraction = self.expected_frames - already_done - self.failed_frames
        need_upload = self.local_frames - already_done

        logger.info(f"Already complete: {len(already_done)}")
        logger.info(f"Need extraction: {len(need_extraction)}")
        logger.info(f"Need upload: {len(need_upload)}")
        logger.info(f"Failed frames: {len(self.failed_frames)}")

        return need_extraction, need_upload

    def load_simple_registry(self) -> Dict:
        """Load minimal registry - only track failures."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    return {
                        "failed": set(data.get("failed", [])),
                        "uploaded": set(data.get("uploaded", [])),
                    }
            except:
                logger.warning("Registry corrupted, starting fresh")
        return {"failed": set(), "uploaded": set()}

    def save_simple_registry(self):
        """Save minimal registry."""
        registry = {
            "failed": list(self.failed_frames),
            "uploaded": list(self.uploaded_frames),
        }
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def get_expected_frames_from_json(self, json_path: str) -> Set[str]:
        """Get expected frames from JSON."""
        expected = set()
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for action_data in data.values():
                clips = action_data.get("Clips", {})
                for clip_data in clips.values():
                    video_url = clip_data.get("url", "")
                    video_filename = Path(video_url).stem
                    retrieval_frames = clip_data.get("retrieval_frames", {})

                    for frame_list in retrieval_frames.values():
                        for frame_num in frame_list:
                            frame_name = f"{video_filename}_frame_{frame_num:06d}.png"
                            expected.add(frame_name)
            return expected
        except Exception as e:
            logger.error(f"Error reading JSON {json_path}: {e}")
            return set()

    def get_cloud_existing_frames(self) -> Set[str]:
        """Get existing frames in cloud."""
        existing = set()
        try:
            logger.info("Scanning cloud bucket...")
            blobs = self.bucket.list_blobs()
            for blob in blobs:
                if blob.name.endswith(".png"):
                    existing.add(blob.name)
            return existing
        except Exception as e:
            logger.error(f"Error scanning cloud: {e}")
            return set()

    def get_local_frame_count(self) -> int:
        """Count local frames."""
        try:
            return len([f for f in self.local_frames_dir.glob("*.png") if f.is_file()])
        except:
            return 0

    def check_upload_trigger(self) -> bool:
        """Check if should trigger upload (non-blocking)."""
        local_count = self.get_local_frame_count()
        if local_count >= self.frame_threshold:
            logger.info(f"🔥 TRIGGERING UPLOAD: {local_count} local frames")
            # Start upload in background thread
            threading.Thread(target=self.upload_current_batch, daemon=True).start()
            return True
        return False

    def upload_current_batch(self):
        """Upload current local files."""
        local_files = list(self.local_frames_dir.glob("*.png"))
        if not local_files:
            return

        # Filter to expected frames only
        valid_files = []
        for file_path in local_files:
            if file_path.name in self.expected_frames:
                if file_path.name not in self.cloud_frames:
                    valid_files.append(file_path)

        if not valid_files:
            logger.info("No valid files to upload")
            return

        logger.info(f"Uploading {len(valid_files)} files...")

        # Use adaptive workers
        workers = (
            self.adaptive_upload_workers if len(valid_files) > 100 else self.normal_upload_workers
        )
        chunk_size = max(1, len(valid_files) // workers)
        chunks = [valid_files[i : i + chunk_size] for i in range(0, len(valid_files), chunk_size)]

        uploaded_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.upload_chunk, chunk): i for i, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                try:
                    count = future.result()
                    uploaded_count += count
                except Exception as e:
                    logger.error(f"Chunk upload failed: {e}")

        logger.info(f"Upload complete: {uploaded_count} files")
        self.save_simple_registry()

    def upload_chunk(self, file_paths: List[Path]) -> int:
        """Upload chunk of files."""
        uploaded = 0
        for file_path in file_paths:
            if not file_path.exists():
                continue

            try:
                if self.upload_single_frame(str(file_path)):
                    self.uploaded_frames.add(file_path.name)
                    self.cloud_frames.add(file_path.name)

                    # Delete after successful upload
                    try:
                        os.remove(file_path)
                        uploaded += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

            except Exception as e:
                logger.warning(f"Upload failed for {file_path.name}: {e}")
                self.failed_frames.add(file_path.name)

        return uploaded

    @retry.Retry(
        predicate=retry.if_exception_type(TooManyRequests, ServiceUnavailable, GoogleAPIError),
        initial=2.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0,
    )
    def upload_single_frame(self, local_path: str) -> bool:
        """Upload single frame with retry."""
        frame_name = Path(local_path).name

        try:
            blob = self.bucket.blob(frame_name)
            blob.upload_from_filename(local_path)

            # Verify upload
            blob.reload()
            local_size = os.path.getsize(local_path)
            return blob.size == local_size

        except (TooManyRequests, ServiceUnavailable):
            logger.warning(f"Rate limit hit for {frame_name}")
            raise
        except Exception as e:
            logger.error(f"Upload failed for {frame_name}: {e}")
            raise

    def extract_frames_batch(self, video_path: str, frame_numbers: List[int]) -> List[str]:
        """Extract frames - non-blocking."""
        video_filename = Path(video_path).stem
        extracted_frames = []

        # Filter frames to extract
        frames_to_extract = []
        for frame_num in frame_numbers:
            frame_name = f"{video_filename}_frame_{frame_num:06d}.png"
            if (
                frame_name in self.expected_frames
                and frame_name not in self.cloud_frames
                and frame_name not in self.failed_frames
            ):
                frames_to_extract.append(frame_num)

        if not frames_to_extract:
            return []

        # Check if should trigger upload (non-blocking)
        self.check_upload_trigger()

        try:
            frame_select = "+".join([f"eq(n\\,{fn})" for fn in frames_to_extract])

            # Try GPU first
            cmd = [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-i",
                video_path,
                "-vf",
                f"select='{frame_select}'",
                "-vsync",
                "0",
                "-c:v",
                "png",
                "-q:v",
                "1",
                "-y",
                f"{self.local_frames_dir}/{video_filename}_temp_%06d.png",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                # Fallback to CPU
                cmd[1:3] = []  # Remove hwaccel
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                # Rename to final names
                for i, frame_num in enumerate(frames_to_extract):
                    temp_path = self.local_frames_dir / f"{video_filename}_temp_{i + 1:06d}.png"
                    final_name = f"{video_filename}_frame_{frame_num:06d}.png"
                    final_path = self.local_frames_dir / final_name

                    if temp_path.exists():
                        temp_path.rename(final_path)
                        self.local_frames.add(final_name)
                        extracted_frames.append(str(final_path))

                logger.info(f"Extracted {len(extracted_frames)} frames from {video_filename}")
            else:
                logger.error(f"Extraction failed for {video_path}")
                for frame_num in frames_to_extract:
                    frame_name = f"{video_filename}_frame_{frame_num:06d}.png"
                    self.failed_frames.add(frame_name)

        except Exception as e:
            logger.error(f"Error extracting from {video_path}: {e}")
            for frame_num in frames_to_extract:
                frame_name = f"{video_filename}_frame_{frame_num:06d}.png"
                self.failed_frames.add(frame_name)

        return extracted_frames

    def get_video_frame_mappings(
        self, clips_data: Dict, need_extraction: Set[str]
    ) -> Dict[str, Set[int]]:
        """Get video-frame mappings for extraction."""
        video_frames = defaultdict(set)

        for action_data in clips_data.values():
            clips = action_data.get("Clips", {})
            for clip_data in clips.values():
                video_url = clip_data.get("url", "")
                video_filename = Path(video_url).stem
                retrieval_frames = clip_data.get("retrieval_frames", {})

                for frame_list in retrieval_frames.values():
                    for frame_num in frame_list:
                        frame_name = f"{video_filename}_frame_{frame_num:06d}.png"
                        if frame_name in need_extraction:
                            video_frames[video_url].add(frame_num)

        return video_frames

    def process_json_file(self, json_file_path: str):
        """Main processing pipeline - simplified."""
        logger.info(f"Starting pipeline for {json_file_path}")

        # Rebuild state from sources
        need_extraction, need_upload = self.rebuild_state_from_sources(json_file_path)

        # Load JSON data
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Upload existing local files first
        if need_upload:
            logger.info(f"Uploading {len(need_upload)} existing local files...")
            self.upload_current_batch()

        # Extract frames
        if need_extraction:
            video_frames_map = self.get_video_frame_mappings(data, need_extraction)
            logger.info(f"Processing {len(video_frames_map)} videos for extraction")

            with ThreadPoolExecutor(max_workers=self.max_extraction_workers) as executor:
                futures = {}
                for video_url, frame_numbers in video_frames_map.items():
                    future = executor.submit(
                        self.extract_frames_batch, video_url, list(frame_numbers)
                    )
                    futures[future] = video_url

                for future in as_completed(futures):
                    video_url = futures[future]
                    try:
                        extracted = future.result()
                        logger.info(f"Completed extraction from {Path(video_url).name}")
                    except Exception as e:
                        logger.error(f"Extraction error for {video_url}: {e}")

        # Mark extraction complete
        self.extraction_complete = True
        logger.info("=== EXTRACTION COMPLETE ===")

        # Final upload
        time.sleep(5)
        final_count = self.get_local_frame_count()
        if final_count > 0:
            logger.info(f"Final upload: {final_count} files")
            self.upload_current_batch()

        # Wait for completion
        max_wait = 10
        for i in range(max_wait):
            if self.get_local_frame_count() == 0:
                break
            logger.info(f"Waiting for uploads... ({i + 1}/{max_wait})")
            time.sleep(30)

        self.print_summary()

    def print_summary(self):
        """Print final summary."""
        expected_total = len(self.expected_frames)
        cloud_expected = len(self.cloud_frames & self.expected_frames)
        completion_rate = cloud_expected / expected_total if expected_total > 0 else 0

        logger.info("=== FINAL SUMMARY ===")
        logger.info(f"Expected frames: {expected_total}")
        logger.info(f"Cloud frames (expected): {cloud_expected}")
        logger.info(f"Completion rate: {completion_rate:.1%}")
        logger.info(f"Failed frames: {len(self.failed_frames)}")
        logger.info(f"Local remaining: {self.get_local_frame_count()}")
        logger.info("====================")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Input JSON file")
    parser.add_argument("--temp-storage", help="Temp storage path")
    parser.add_argument("--frame-threshold", type=int, default=3000)
    parser.add_argument("--adaptive-workers", type=int, default=15)
    args = parser.parse_args()

    uploader = AdaptivePipelineUploader(
        temp_storage_path=args.temp_storage,
        frame_threshold=args.frame_threshold,
        adaptive_upload_workers=args.adaptive_workers,
    )

    uploader.process_json_file(args.json_file)


if __name__ == "__main__":
    main()
