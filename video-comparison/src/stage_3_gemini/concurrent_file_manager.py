import json
import os
import logging
import asyncio
from typing import Tuple, List, Dict, Set
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class AsyncGeminiFileManager:
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

                # If current_storage is not cached, calculate it from file sizes
                if self.current_storage == 0 and self.uploaded_files:
                    self.logger.info("Calculating storage from cached file sizes...")
                    for frame_name, file_data in self.uploaded_files.items():
                        cached_size = file_data.get("file_size")
                        if cached_size is not None:
                            self.current_storage += cached_size
                        else:
                            # Calculate and cache size for files without it
                            local_path = file_data.get("local_path", "")
                            size = self._get_file_size(local_path, frame_name)
                            self.current_storage += size
                    self._save_file_tracking()  # Save updated sizes

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

    def _get_file_size(self, file_path: str, frame_name: str = None) -> int:
        """Get file size in bytes, with caching."""
        # Try to get from cache first
        if frame_name and frame_name in self.uploaded_files:
            cached_size = self.uploaded_files[frame_name].get("file_size")
            if cached_size is not None:
                return cached_size

        # Calculate and optionally cache
        try:
            size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            if frame_name and frame_name in self.uploaded_files:
                self.uploaded_files[frame_name]["file_size"] = size
            return size
        except:
            return 0

    async def _delete_single_file(self, frame_name: str) -> Tuple[bool, int]:
        """Delete a single file and return success status and freed size."""
        if frame_name not in self.uploaded_files:
            return False, 0

        try:
            file_data = self.uploaded_files[frame_name]
            await self.client.aio.files.delete(name=file_data["name"])

            # Calculate freed size from cache
            file_size = file_data.get("file_size", 0)
            if file_size == 0:
                file_size = self._get_file_size(file_data.get("local_path", ""))

            # Remove from tracking
            del self.uploaded_files[frame_name]
            self.current_storage -= file_size

            self.logger.debug(f"Deleted remote file: {frame_name}")
            return True, file_size

        except Exception as e:
            self.logger.warning(f"Failed to delete {frame_name}: {e}")
            # Remove from tracking anyway since we can't verify its existence
            if frame_name in self.uploaded_files:
                file_size = self._get_file_size(
                    self.uploaded_files[frame_name].get("local_path", "")
                )
                self.current_storage -= file_size
                del self.uploaded_files[frame_name]
                return (
                    True,
                    file_size,
                )  # Consider it successful since we cleaned up tracking
            return False, 0

    async def _delete_files_batch(
        self, files_batch: List[str], semaphore: asyncio.Semaphore
    ) -> Tuple[int, int]:
        """Delete a batch of files with semaphore control."""

        async def delete_with_semaphore(frame_name):
            async with semaphore:
                return await self._delete_single_file(frame_name)

        # Create tasks for the batch
        tasks = [delete_with_semaphore(frame_name) for frame_name in files_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        deleted_count = 0
        total_size_freed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Exception deleting {files_batch[i]}: {result}")
            else:
                success, size_freed = result
                if success:
                    deleted_count += 1
                    total_size_freed += size_freed

        return deleted_count, total_size_freed

    async def _delete_remote_files_threaded(self, files_to_delete: List[str]) -> int:
        """Delete files from remote storage using async threads."""
        if not files_to_delete:
            return 0

        self.logger.info(f"Starting threaded deletion of {len(files_to_delete)} files...")

        # Create batches of 50 files
        batch_size = 50
        batches = [
            files_to_delete[i : i + batch_size] for i in range(0, len(files_to_delete), batch_size)
        ]

        # Use semaphore to limit concurrent deletions to 25
        semaphore = asyncio.Semaphore(25)

        total_deleted = 0
        total_size_freed = 0

        for i, batch in enumerate(batches):
            self.logger.info(f"Deleting batch {i + 1}/{len(batches)} ({len(batch)} files)")

            deleted_count, size_freed = await self._delete_files_batch(batch, semaphore)
            total_deleted += deleted_count
            total_size_freed += size_freed

            # Small delay between batches
            if i < len(batches) - 1:
                await asyncio.sleep(0.5)

        # Save tracking after all deletions complete
        if total_deleted > 0:
            self._save_file_tracking()
            self.logger.info(
                f"Completed threaded deletion: {total_deleted}/{len(files_to_delete)} files, "
                f"freed {total_size_freed / (1024 * 1024):.1f} MB"
            )

        return total_deleted

    async def _upload_single_file(self, file_path: str) -> Tuple[bool, str, int]:
        """Upload a single file and return success status, frame name, and size."""
        frame_name = Path(file_path).name

        # Skip if already uploaded
        if frame_name in self.uploaded_files:
            return True, frame_name, 0

        try:
            uploaded_file = await self.client.aio.files.upload(file=file_path)
            file_size = self._get_file_size(file_path)

            self.uploaded_files[frame_name] = {
                "name": uploaded_file.name,
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
                "local_path": file_path,
                "file_size": file_size,
            }

            # Update storage tracking
            self.current_storage += file_size
            self.logger.debug(f"Uploaded: {frame_name} ({file_size / (1024 * 1024):.1f} MB)")
            return True, frame_name, file_size

        except Exception as e:
            self.logger.error(f"Failed to upload {frame_name}: {e}")
            return False, frame_name, 0

    async def _upload_files_batch(
        self, files_batch: List[str], semaphore: asyncio.Semaphore
    ) -> Tuple[int, int]:
        """Upload a batch of files with semaphore control."""

        async def upload_with_semaphore(file_path):
            async with semaphore:
                return await self._upload_single_file(file_path)

        # Create tasks for the batch
        tasks = [upload_with_semaphore(file_path) for file_path in files_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        total_size_uploaded = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Exception uploading {files_batch[i]}: {result}")
            else:
                success, frame_name, size = result
                if success:
                    success_count += 1
                    total_size_uploaded += size

        return success_count, total_size_uploaded

    async def _upload_files_threaded(self, file_paths: List[str]) -> bool:
        """Upload files using async threads in batches."""
        if not file_paths:
            return True

        self.logger.info(f"Starting threaded upload of {len(file_paths)} files...")

        # Create batches of 50 files
        batch_size = 50
        batches = [file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)]

        # Use semaphore to limit concurrent uploads to 25
        semaphore = asyncio.Semaphore(25)

        total_success = 0
        total_size_uploaded = 0

        for i, batch in enumerate(batches):
            self.logger.info(f"Uploading batch {i + 1}/{len(batches)} ({len(batch)} files)")

            success_count, size_uploaded = await self._upload_files_batch(batch, semaphore)
            total_success += success_count
            total_size_uploaded += size_uploaded

            # Save progress after each batch
            self._save_file_tracking()

            # Small delay between batches
            if i < len(batches) - 1:
                await asyncio.sleep(1.0)

        success_rate = total_success / len(file_paths) if file_paths else 1.0
        self.logger.info(
            f"Completed threaded upload: {total_success}/{len(file_paths)} files ({success_rate:.1%}), "
            f"uploaded {total_size_uploaded / (1024 * 1024):.1f} MB"
        )
        self.logger.info(f"Current storage: {self.current_storage / (1024 * 1024):.1f} MB")

        return success_rate >= 0.9  # 90% success rate threshold

    async def _manage_storage_for_new_uploads(
        self, files_to_upload: List[str], needed_frame_names: Set[str]
    ):
        """Manage storage before uploading new files."""
        if not files_to_upload:
            return

        # Calculate size of files to upload using cached sizes where possible
        new_files_size = 0
        for path in files_to_upload:
            frame_name = Path(path).name
            if frame_name in self.uploaded_files:
                cached_size = self.uploaded_files[frame_name].get("file_size", 0)
                new_files_size += (
                    cached_size if cached_size > 0 else self._get_file_size(path, frame_name)
                )
            else:
                new_files_size += self._get_file_size(path)

        # Check if we need to free up space
        if self.current_storage + new_files_size > self.max_storage_bytes:
            space_needed = (self.current_storage + new_files_size) - self.max_storage_bytes
            self.logger.info(f"Need to free {space_needed / (1024 * 1024):.1f} MB of storage")

            # Find files not needed for current chunk
            all_tracked_frames = set(self.uploaded_files.keys())
            not_needed_frames = all_tracked_frames - needed_frame_names

            if not_needed_frames:
                # Find files not needed for current chunk and their cached sizes
                files_with_sizes = []
                for frame_name in not_needed_frames:
                    file_data = self.uploaded_files.get(frame_name, {})
                    file_size = file_data.get("file_size", 0)
                    if file_size == 0:
                        # Calculate and cache if not available
                        local_path = file_data.get("local_path", "")
                        file_size = self._get_file_size(local_path, frame_name)
                    files_with_sizes.append((frame_name, file_size))

                # Sort by size (descending) to delete larger files first
                files_with_sizes.sort(key=lambda x: x[1], reverse=True)

                # Select files to delete
                files_to_delete = []
                size_to_delete = 0

                for frame_name, file_size in files_with_sizes:
                    files_to_delete.append(frame_name)
                    size_to_delete += file_size

                    # Stop once we've freed enough space (with some buffer)
                    if size_to_delete >= space_needed * 1.1:  # 10% buffer
                        break

                if files_to_delete:
                    self.logger.info(
                        f"Deleting {len(files_to_delete)} unnecessary files to free space"
                    )
                    await self._delete_remote_files_threaded(files_to_delete)
                else:
                    self.logger.warning("No unnecessary files found to delete")
            else:
                self.logger.warning(
                    "All tracked files are needed for current chunk - cannot free space"
                )

    async def prepare_files_for_chunk(
        self, needed_frame_paths: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Optimized workflow with threaded storage management:
        1. Check local tracking first
        2. Manage storage (threaded deletion) before uploading
        3. Upload missing files (threaded upload)
        """
        self.logger.info(f"Preparing {len(needed_frame_paths)} files")

        # Step 1: Check local tracking first
        needed_frame_names = set()
        path_to_name = {}

        for path in needed_frame_paths:
            if os.path.exists(path):
                frame_name = Path(path).name
                needed_frame_names.add(frame_name)
                path_to_name[frame_name] = path

        # Find missing based on local tracking
        locally_tracked = set(self.uploaded_files.keys())
        missing_files = needed_frame_names - locally_tracked

        self.logger.info(
            f"Local check: {len(locally_tracked & needed_frame_names)} tracked, {len(missing_files)} missing"
        )

        # Step 2: Manage storage before uploading (threaded deletion)
        if missing_files:
            files_to_upload = [
                path_to_name[name] for name in missing_files if name in path_to_name
            ]

            # Manage storage before uploading (includes threaded deletion)
            await self._manage_storage_for_new_uploads(files_to_upload, needed_frame_names)

            total_size = sum(self._get_file_size(path) for path in files_to_upload)

            self.logger.info(
                f"Starting threaded upload of {len(files_to_upload)} files ({total_size / (1024 * 1024):.1f} MB)"
            )

            # Step 3: Upload files using threaded approach
            upload_success = await self._upload_files_threaded(files_to_upload)
            if not upload_success:
                self.logger.warning("Some uploads failed")
                return False, list(missing_files)

        self.logger.info("✅ All needed files ready")
        return True, []

    async def get_file_objects_for_processing(self, frame_names: List[str]) -> List:
        """Get file objects for the given frame names, ready for Gemini processing."""
        file_objects = []

        # Process in small batches to avoid connection issues
        batch_size = 15
        for i in range(0, len(frame_names), batch_size):
            batch_names = frame_names[i : i + batch_size]

            for frame_name in batch_names:
                if frame_name in self.uploaded_files:
                    try:
                        file_data = self.uploaded_files[frame_name]
                        file_obj = await self.client.aio.files.get(name=file_data["name"])
                        if i % 10 == 0:
                            self.logger.debug(
                                f"Retrieved file object for {frame_name} {i}/{len(frame_names)} "
                            )
                        file_objects.append(file_obj)

                    except Exception as e:
                        self.logger.warning(f"Failed to get file object for {frame_name}: {e}")

                await asyncio.sleep(0.1)

            # Delay between batches
            await asyncio.sleep(0.5)

        self.logger.info(f"Retrieved {len(file_objects)} file objects for processing")
        return file_objects

    async def upload_frame(self, local_path: str, frame_name: str = None) -> Tuple[bool, str]:
        """Upload single frame (compatibility method)."""
        if not os.path.exists(local_path):
            return False, "file_not_found"

        if frame_name is None:
            frame_name = Path(local_path).name

        # Check if already uploaded
        if frame_name in self.uploaded_files:
            return True, "already_exists"

        try:
            uploaded_file = await self.client.aio.files.upload(file=local_path)
            file_size = self._get_file_size(local_path)

            self.uploaded_files[frame_name] = {
                "name": uploaded_file.name,
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
                "local_path": local_path,
                "file_size": file_size,
            }

            # Update storage tracking
            self.current_storage += file_size
            self._save_file_tracking()
            return True, "uploaded"

        except Exception as e:
            self.logger.error(f"Upload failed for {frame_name}: {e}")
            return False, "failed"

    async def get_file_by_name(self, frame_name: str):
        """Get uploaded file object by name (compatibility method)."""
        if frame_name not in self.uploaded_files:
            return None

        file_data = self.uploaded_files[frame_name]
        try:
            file_obj = await self.client.aio.files.get(name=file_data["name"])
            return file_obj
        except Exception as e:
            self.logger.error(f"Failed to retrieve file {frame_name}: {e}")
            return None

    def get_storage_info(self) -> Dict:
        """Get current storage information."""
        return {
            "current_storage_mb": self.current_storage / (1024 * 1024),
            "max_storage_mb": self.max_storage_bytes / (1024 * 1024),
            "storage_percentage": (self.current_storage / self.max_storage_bytes) * 100,
            "total_files": len(self.uploaded_files),
        }
