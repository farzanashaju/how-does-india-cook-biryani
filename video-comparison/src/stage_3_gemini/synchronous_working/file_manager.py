import json
import os
import logging
from typing import Tuple
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from datetime import datetime


class GeminiFileManager:
    def __init__(
        self,
        max_storage_gb: float = 18.0,
        uploaded_files_json: str = "uploaded_frames.json",
        logger: logging.Logger = None,
    ):
        """Initialize Gemini client with file management."""
        load_dotenv()

        self.logger = logger or logging.getLogger(__name__)
        self.client = genai.Client()
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        self.uploaded_files = {}
        self.current_storage = 0
        self.uploaded_files_json = uploaded_files_json

        self.logger.info("Loading file tracking data...")
        self._load_file_tracking()

    def _load_file_tracking(self):
        """Load file tracking data from JSON file."""
        if os.path.exists(self.uploaded_files_json):
            try:
                with open(self.uploaded_files_json, "r") as f:
                    data = json.load(f)
                self.uploaded_files = data.get("uploaded_files", {})
                self.current_storage = data.get("current_storage", 0)
                self.logger.info(
                    f"Loaded tracking for {len(self.uploaded_files)} files ({self.current_storage / (1024 * 1024):.1f} MB)"
                )
            except Exception as e:
                self.logger.warning(f"Could not load file tracking: {e}")
                self.uploaded_files = {}
                self.current_storage = 0
        else:
            self.logger.info("No existing file tracking found, starting fresh")

    def _save_file_tracking(self):
        """Save file tracking data to JSON file."""
        try:
            data = {
                "uploaded_files": self.uploaded_files,
                "current_storage": self.current_storage,
                "last_updated": datetime.now().isoformat(),
                "total_files": len(self.uploaded_files),
            }
            with open(self.uploaded_files_json, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save file tracking: {e}")

    def format_frame_name(self, video_url: str, frame_number: int) -> str:
        """Generate frame name from video URL and frame number."""
        video_filename = Path(video_url).stem
        return f"{video_filename}_frame_{frame_number:06d}.png"

    def _cleanup_old_files(self, bytes_needed: int):
        """Delete oldest files to make space."""
        self.logger.info(f"Need to free up {bytes_needed / (1024 * 1024):.1f} MB...")

        # Remove first 25% of files (simple FIFO approach)
        files_list = list(self.uploaded_files.items())
        files_to_remove = files_list[: len(files_list) // 4]

        freed_space = 0
        for filename, file_data in files_to_remove:
            try:
                # Delete from Gemini
                self.client.files.delete(name=file_data["name"])

                # Estimate freed space (rough approximation)
                estimated_size = 100 * 1024  # Assume ~100KB per frame
                freed_space += estimated_size

                # Remove from tracking
                del self.uploaded_files[filename]
                self.logger.info(f"Deleted old file: {filename}")

                if freed_space >= bytes_needed:
                    break

            except Exception as e:
                self.logger.error(f"Failed to delete {filename}: {e}")

        self.current_storage -= freed_space
        self._save_file_tracking()

    def upload_frame(self, local_path: str, frame_name: str = None) -> Tuple[bool, str]:
        """Upload frame to Files API with storage management."""
        if not os.path.exists(local_path):
            return False, "file_not_found"

        if frame_name is None:
            frame_name = Path(local_path).name

        # Check if already uploaded
        if frame_name in self.uploaded_files:
            return True, "already_exists"

        # Check storage limits
        file_size = os.path.getsize(local_path)
        if self.current_storage + file_size > self.max_storage_bytes:
            self._cleanup_old_files(file_size)

        try:
            uploaded_file = self.client.files.upload(file=local_path)
            self.uploaded_files[frame_name] = {
                "name": uploaded_file.name,
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
            }
            self.current_storage += file_size
            self._save_file_tracking()
            return True, "uploaded"
        except Exception as e:
            self.logger.error(f"Upload failed for {frame_name}: {e}")
            return False, "failed"

    def get_file_by_name(self, frame_name: str):
        """Get uploaded file object by name."""
        if frame_name not in self.uploaded_files:
            return None

        file_data = self.uploaded_files[frame_name]

        # Get the actual file from Gemini API using the stored name
        try:
            file_obj = self.client.files.get(name=file_data["name"])
            return file_obj
        except Exception as e:
            self.logger.error(f"Failed to retrieve file {frame_name}: {e}")
            # Remove from tracking if file no longer exists
            del self.uploaded_files[frame_name]
            self._save_file_tracking()
            return None
