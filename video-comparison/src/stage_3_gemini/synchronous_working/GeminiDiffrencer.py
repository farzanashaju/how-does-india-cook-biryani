import json
import os
import glob
import ijson
import logging
from typing import List, Dict, Any, Set, Tuple, Iterator
from pathlib import Path
from google import genai
from dotenv import load_dotenv
import time
from google.genai import types
from datetime import datetime, timedelta

from prompt import PROMPT_DIFFERENCE_MULTIFRAME, SYSTEM_PROMPT


def setup_logging(log_file: str = "biryani_processor.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


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


class TimingTracker:
    """Track timing statistics for processing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.comparison_times = []
        self.start_time = None
        self.total_comparisons = 0
        self.processed_comparisons = 0
    
    def start_processing(self, total_comparisons: int):
        """Start timing the overall processing."""
        self.start_time = time.time()
        self.total_comparisons = total_comparisons
        self.processed_comparisons = 0
        self.comparison_times = []
        self.logger.info(f"⏱️  Started processing {total_comparisons} total comparisons")
    
    def record_comparison_time(self, comparison_key: str, elapsed_time: float):
        """Record time taken for a comparison."""
        self.comparison_times.append(elapsed_time)
        self.processed_comparisons += 1
        
        # Calculate statistics
        avg_time = sum(self.comparison_times) / len(self.comparison_times)
        remaining_comparisons = self.total_comparisons - self.processed_comparisons
        estimated_remaining_time = avg_time * remaining_comparisons
        
        # Calculate total elapsed time
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Format times for display
        elapsed_str = self._format_duration(elapsed_time)
        avg_str = self._format_duration(avg_time)
        remaining_str = self._format_duration(estimated_remaining_time)
        total_elapsed_str = self._format_duration(total_elapsed)
        
        # Calculate completion percentage
        completion_pct = (self.processed_comparisons / self.total_comparisons) * 100
        
        # Estimate completion time
        if total_elapsed > 0:
            estimated_total_time = total_elapsed * (self.total_comparisons / self.processed_comparisons)
            eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
            eta_str = eta.strftime("%H:%M:%S")
        else:
            eta_str = "Unknown"
        
        self.logger.info(
            f"⏰ {comparison_key} completed in {elapsed_str} | "
            f"Progress: {self.processed_comparisons}/{self.total_comparisons} ({completion_pct:.1f}%) | "
            f"Avg: {avg_str} | Est. remaining: {remaining_str} | ETA: {eta_str}"
        )
        
        # Log milestone updates every 10 comparisons or at significant percentages
        if (self.processed_comparisons % 10 == 0 or 
            completion_pct in [25, 50, 75, 90, 95]):
            self.logger.info(
                f"📊 MILESTONE: {self.processed_comparisons}/{self.total_comparisons} comparisons complete "
                f"({completion_pct:.1f}%) | Total elapsed: {total_elapsed_str} | ETA: {eta_str}"
            )
    
    def finish_processing(self):
        """Log final timing statistics."""
        if not self.start_time or not self.comparison_times:
            return
            
        total_time = time.time() - self.start_time
        avg_time = sum(self.comparison_times) / len(self.comparison_times)
        min_time = min(self.comparison_times)
        max_time = max(self.comparison_times)
        
        self.logger.info(
            f"🏁 PROCESSING COMPLETE! "
            f"Total: {self._format_duration(total_time)} | "
            f"Avg per comparison: {self._format_duration(avg_time)} | "
            f"Min: {self._format_duration(min_time)} | "
            f"Max: {self._format_duration(max_time)} | "
            f"Processed: {self.processed_comparisons} comparisons"
        )
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"


class BiryaniComparisonProcessor:
    def __init__(
        self,
        local_images_path: str,
        gemini_manager: GeminiFileManager,
        logger: logging.Logger = None,
    ):
        self.local_images_path = Path(local_images_path)
        self.gemini_manager = gemini_manager
        self.logger = logger or logging.getLogger(__name__)
        self.timing_tracker = TimingTracker(self.logger)

    def _load_existing_results(self, output_file: str) -> Dict[str, Set[str]]:
        """Load existing results and return completed comparisons and their differences."""
        completed_comparisons = {}

        if not os.path.exists(output_file):
            self.logger.info(f"Output file {output_file} doesn't exist, starting fresh")
            return completed_comparisons

        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            for comparison_key, data in existing_data.items():
                completed_diffs = set()
                proposed_differences = data.get("proposed_differences", {})

                for diff_id, diff_data in proposed_differences.items():
                    if diff_data.get("status") == "success" and diff_data.get("winner"):
                        completed_diffs.add(diff_id)

                completed_comparisons[comparison_key] = completed_diffs

            total_completed = sum(len(diffs) for diffs in completed_comparisons.values())
            self.logger.info(
                f"Found {len(completed_comparisons)} existing comparisons with {total_completed} completed differences"
            )

        except Exception as e:
            self.logger.warning(f"Error loading existing results: {e}")

        return completed_comparisons

    def find_important_frames(
        self,
        action_stages: List[Dict],
        difference_name: str,
        clips: Dict,
        clip1_frame_count: int,
        clip2_frame_count: int,
    ) -> str:
        """Find frames that might be important for the difference and create context."""
        important_context = ""
        for stage in action_stages:
            if difference_name in stage.get("associated_differences", []):
                stage_name = stage.get("name", "")
                description = stage.get("description", "")
                clip1_data = clips.get("1", {})
                clip2_data = clips.get("2", {})

                if isinstance(clip1_data, dict) and isinstance(clip2_data, dict):
                    clip1_frames = clip1_data.get("retrieval_frames", {}).get(stage_name, [])
                    clip2_frames = clip2_data.get("retrieval_frames", {}).get(stage_name, [])

                    if clip1_frames or clip2_frames:
                        clip1_positions = self._convert_to_upload_positions(
                            clip1_frames, clip1_data, True
                        )
                        clip2_positions = self._convert_to_upload_positions(
                            clip2_frames, clip2_data, False, clip1_frame_count
                        )
                        important_context += (
                            f"\nFrames {clip1_positions} (Video A) and "
                            f"{clip2_positions} (Video B) might be important for this "
                            f"difference as they show: {description}"
                        )
        return important_context

    def _convert_to_upload_positions(
        self,
        frame_numbers: List[int],
        clip_data: Dict,
        is_first_clip: bool,
        first_clip_count: int = 0,
    ) -> List[int]:
        """Convert absolute frame numbers to positions in the uploaded frame sequence."""
        if not frame_numbers:
            return []

        all_retrieval_frames = []
        retrieval_frames = clip_data.get("retrieval_frames", {})
        for stage_frames in retrieval_frames.values():
            all_retrieval_frames.extend(stage_frames)

        unique_frames = sorted(set(all_retrieval_frames))
        frame_to_position = {}
        for i, frame_num in enumerate(unique_frames):
            upload_position = i + 1 + (0 if is_first_clip else first_clip_count)
            frame_to_position[frame_num] = upload_position

        positions = []
        for frame_num in frame_numbers:
            if frame_num in frame_to_position:
                positions.append(frame_to_position[frame_num])

        return positions

    def _get_unique_frame_count(self, clip_data: Dict) -> int:
        """Get the number of unique frames for a clip."""
        if not isinstance(clip_data, dict):
            return 0

        all_frames = []
        retrieval_frames = clip_data.get("retrieval_frames", {})
        for stage_frames in retrieval_frames.values():
            all_frames.extend(stage_frames)

        return len(set(all_frames))

    def _upload_clip_frames(self, clip_data: Dict) -> List[Any]:
        """Upload all frames for a clip and return file objects."""
        if not isinstance(clip_data, dict):
            return []

        video_url = clip_data.get("url", "")
        if not video_url:
            return []

        # Get all unique frames for this clip
        all_frames = []
        retrieval_frames = clip_data.get("retrieval_frames", {})
        for stage_frames in retrieval_frames.values():
            all_frames.extend(stage_frames)

        unique_frames = sorted(set(all_frames))
        uploaded_files = []

        for frame_num in unique_frames:
            frame_name = self.gemini_manager.format_frame_name(video_url, frame_num)
            local_path = self.local_images_path / frame_name

            success, status = self.gemini_manager.upload_frame(str(local_path), frame_name)
            if success:
                file_obj = self.gemini_manager.get_file_by_name(frame_name)
                if file_obj:
                    uploaded_files.append(file_obj)
                    self.logger.debug(f"✓ {frame_name} ({status})")
                else:
                    self.logger.error(f"✗ Could not retrieve {frame_name}")
            else:
                self.logger.error(f"✗ Failed to upload {frame_name}: {status}")

        return uploaded_files

    def _save_intermediate_result(
        self, output_file: str, comparison_key: str, difference_result: Dict
    ):
        """Save result immediately after processing each difference."""
        # Load existing results or create new structure
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        # Ensure comparison exists in output
        if comparison_key not in existing_data:
            existing_data[comparison_key] = {
                "unique_identifier": difference_result.get("unique_identifier", ""),
                "action_class": difference_result.get("action_class", ""),
                "clips": difference_result.get("clips", {}),
                "proposed_differences": {},
            }

        # Add the difference result
        diff_id = difference_result.get("difference_id", "")
        existing_data[comparison_key]["proposed_differences"][diff_id] = {
            "name": difference_result.get("difference_name", ""),
            "query_string": difference_result.get("query_string", ""),
            "winner": difference_result.get("analysis", {}).get("answer", "C"),
            "winner_reason": difference_result.get("analysis", {}).get("explanation", ""),
            "confidence": difference_result.get("analysis", {}).get("confidence", 1),
            "error": None,
            "status": "success",
        }

        # Save updated results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"💾 Saved result for {comparison_key} - {difference_result.get('difference_name', '')}"
        )

    def process_comparison(
        self,
        comparison_key: str,
        comparison_data: Dict,
        output_file: str,
        completed_differences: Set[str] = None,
    ):
        """Process a single comparison and save results incrementally."""
        comparison_start_time = time.time()
        
        if completed_differences is None:
            completed_differences = set()

        try:
            action_class = comparison_data.get("action_class", "")
            clips = comparison_data.get("clips", {})
            proposed_differences = comparison_data.get("proposed_differences", {})
            action_stages = comparison_data.get("action_stages", [])
            unique_identifier = comparison_data.get("unique_identifier", "")

            self.logger.info(f"Processing {comparison_key} ({action_class})")

            # Check if we need to process any differences for this comparison
            remaining_differences = []
            for diff_id, difference in proposed_differences.items():
                if diff_id not in completed_differences and isinstance(difference, dict):
                    remaining_differences.append((diff_id, difference))

            if not remaining_differences:
                self.logger.info(f"  All differences already completed for {comparison_key}")
                return

            self.logger.info(f"  Processing {len(remaining_differences)} remaining differences")

            # Upload frames for both clips (only if we have differences to process)
            clip1_files = self._upload_clip_frames(clips.get("1", {}))
            clip2_files = self._upload_clip_frames(clips.get("2", {}))

            if not clip1_files or not clip2_files:
                self.logger.warning(f"⚠️  Missing frames for {comparison_key}")
                return

            all_files = clip1_files + clip2_files
            clip1_frame_count = len(clip1_files)
            clip2_frame_count = len(clip2_files)

            self.logger.info(f"  Uploaded {clip1_frame_count} + {clip2_frame_count} frames")

            # Process each remaining difference and save immediately
            for diff_id, difference in remaining_differences:
                diff_name = difference.get("name", "")
                query_string = difference.get("query_string", "")

                if not diff_name or not query_string:
                    continue

                self.logger.info(f"    Analyzing difference: {diff_name}")

                try:
                    # Create prompt
                    importance_context = self.find_important_frames(
                        action_stages,
                        diff_name,
                        clips,
                        clip1_frame_count,
                        clip2_frame_count,
                    )

                    formatted_prompt = PROMPT_DIFFERENCE_MULTIFRAME.format(
                        action=action_class,
                        query_string=query_string,
                        importance_context=importance_context,
                        total_frames=clip1_frame_count + clip2_frame_count,
                        clip1_range=f"1-{clip1_frame_count}",
                        clip2_start=clip1_frame_count + 1,
                        clip2_end=clip1_frame_count + clip2_frame_count,
                    )

                    # Generate response
                    response = self.gemini_manager.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=all_files + [formatted_prompt],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.7,
                            max_output_tokens=1024,
                        ),
                    )

                    # Parse JSON response
                    response_text = response.text
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        json_str = response_text[json_start:json_end].strip()
                    else:
                        json_str = response_text.strip()

                    try:
                        analysis_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        analysis_result = {
                            "answer": "c",
                            "confidence": 1,
                            "difference_visible": False,
                            "explanation": f"Could not parse response: {response_text[:200]}",
                        }

                    # Create result for this difference
                    difference_result = {
                        "comparison_key": comparison_key,
                        "unique_identifier": unique_identifier,
                        "action_class": action_class,
                        "difference_id": diff_id,
                        "difference_name": diff_name,
                        "query_string": query_string,
                        "clips": clips,
                        "analysis": analysis_result,
                    }

                    # Save this result immediately
                    self._save_intermediate_result(output_file, comparison_key, difference_result)
                    self.logger.info(f"      Result: {analysis_result.get('answer', 'unknown')}")

                    # Small delay to avoid rate limits
                    time.sleep(1)

                except Exception as e:
                    self.logger.error(f"    Error analyzing {diff_name}: {e}")

                    # Save error result
                    error_result = {
                        "comparison_key": comparison_key,
                        "unique_identifier": unique_identifier,
                        "action_class": action_class,
                        "difference_id": diff_id,
                        "difference_name": diff_name,
                        "query_string": query_string,
                        "clips": clips,
                        "analysis": {
                            "answer": "C",
                            "confidence": 1,
                            "explanation": f"Error during processing: {str(e)}",
                        },
                    }
                    self._save_intermediate_result(output_file, comparison_key, error_result)

        except Exception as e:
            self.logger.error(f"Error processing {comparison_key}: {e}")
        
        finally:
            # Record timing for this comparison
            comparison_time = time.time() - comparison_start_time
            self.timing_tracker.record_comparison_time(comparison_key, comparison_time)

    def chunked_json_reader(
        self, input_file: str, chunk_size: int = 1000
    ) -> Iterator[Dict[str, Any]]:
        """Read JSON file in chunks to handle large files efficiently."""
        self.logger.info(f"Reading {input_file} in chunks of {chunk_size}")

        with open(input_file, "rb") as f:
            # Use ijson to parse the file incrementally
            parser = ijson.parse(f)
            current_chunk = {}
            item_count = 0
            current_key = None
            current_item = {}
            in_comparison = False

            for prefix, event, value in parser:
                # Track when we enter a comparison item
                if prefix.endswith(".item") and event == "start_map":
                    in_comparison = True
                    current_item = {}
                    continue

                # Track the key for the current comparison
                if prefix.count(".") == 1 and event == "map_key":
                    current_key = value
                    if current_key.startswith("Action_class_comparison"):
                        in_comparison = True
                    continue

                # Collect data for current comparison
                if (
                    in_comparison
                    and current_key
                    and current_key.startswith("Action_class_comparison")
                ):
                    # Remove the root key from prefix to get relative path
                    relative_prefix = prefix.replace(current_key + ".", "")

                    if event == "string" or event == "number" or event == "boolean":
                        self._set_nested_value(current_item, relative_prefix, value)
                    elif event == "start_array":
                        self._set_nested_value(current_item, relative_prefix, [])
                    elif event == "start_map":
                        self._set_nested_value(current_item, relative_prefix, {})

                # When we finish a comparison item
                if prefix.endswith(".item") and event == "end_map" and in_comparison:
                    if current_key and current_key.startswith("Action_class_comparison"):
                        current_chunk[current_key] = current_item
                        item_count += 1

                        # Yield chunk when it reaches the desired size
                        if item_count >= chunk_size:
                            yield current_chunk
                            current_chunk = {}
                            item_count = 0

                    in_comparison = False
                    current_key = None
                    current_item = {}

            # Yield remaining items
            if current_chunk:
                yield current_chunk

    def _set_nested_value(self, obj: Dict, path: str, value: Any):
        """Set a nested value in a dictionary using dot notation."""
        if not path:
            return

        keys = path.split(".")
        current = obj

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _count_total_work(self, data: Dict, completed_comparisons: Dict[str, Set[str]]) -> int:
        """Count total number of comparisons that need work."""
        total_work = 0
        
        for comparison_key, comparison_data in data.items():
            if comparison_key.startswith("Action_class_comparison"):
                if isinstance(comparison_data, dict):
                    completed_diffs = completed_comparisons.get(comparison_key, set())
                    proposed_differences = comparison_data.get("proposed_differences", {})
                    
                    # Check if this comparison has any remaining work
                    remaining_work = any(
                        diff_id not in completed_diffs
                        for diff_id in proposed_differences.keys()
                    )
                    
                    if remaining_work:
                        total_work += 1
        
        return total_work

    def process_json_file_chunked(self, input_file: str, output_file: str, chunk_size: int = 1000):
        """Process large JSON file in chunks."""
        try:
            # Load existing results to determine what to skip
            completed_comparisons = self._load_existing_results(output_file)

            # We can't easily count total work for chunked processing, so estimate
            # This is an approximation - for exact counts, use non-chunked processing
            file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
            estimated_comparisons = int(file_size * 10)  # Rough estimate: 10 comparisons per MB
            self.timing_tracker.start_processing(estimated_comparisons)

            total_processed = 0
            total_skipped = 0

            for chunk in self.chunked_json_reader(input_file, chunk_size):
                self.logger.info(f"Processing chunk with {len(chunk)} comparisons")

                for comparison_key, comparison_data in chunk.items():
                    if isinstance(comparison_data, dict):
                        completed_diffs = completed_comparisons.get(comparison_key, set())

                        # Check if this comparison has any remaining work
                        proposed_differences = comparison_data.get("proposed_differences", {})
                        remaining_work = any(
                            diff_id not in completed_diffs
                            for diff_id in proposed_differences.keys()
                        )

                        if remaining_work:
                            self.process_comparison(
                                comparison_key,
                                comparison_data,
                                output_file,
                                completed_diffs,
                            )
                            total_processed += 1
                        else:
                            total_skipped += 1

                self.logger.info(
                    f"Completed chunk. Processed: {total_processed}, Skipped: {total_skipped}"
                )

                # Optional: add a longer pause between chunks
                time.sleep(2)

            self.timing_tracker.finish_processing()
            self.logger.info(
                f"✅ Processing complete. {total_processed} comparisons processed, {total_skipped} skipped. Results saved to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")

    def process_json_file(self, input_file: str, output_file: str):
        """Process entire JSON file and save results incrementally."""
        file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB

        if file_size > 50:  # If larger than 50MB, use chunked processing
            self.logger.info(f"Large file detected ({file_size:.1f} MB), using chunked processing")
            self.process_json_file_chunked(input_file, output_file, chunk_size=1000)
        else:
            self.logger.info(f"Processing {file_size:.1f} MB file normally")
            try:
                # Load existing results to determine what to skip
                completed_comparisons = self._load_existing_results(output_file)

                with open(input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.logger.info(f"Processing {len(data)} comparisons from {input_file}")

                # Count total work needed and initialize timing
                total_work = self._count_total_work(data, completed_comparisons)
                self.timing_tracker.start_processing(total_work)

                processed_count = 0
                skipped_count = 0

                for comparison_key, comparison_data in data.items():
                    if comparison_key.startswith("Action_class_comparison"):
                        if isinstance(comparison_data, dict):
                            completed_diffs = completed_comparisons.get(comparison_key, set())

                            # Check if this comparison has any remaining work
                            proposed_differences = comparison_data.get("proposed_differences", {})
                            remaining_work = any(
                                diff_id not in completed_diffs
                                for diff_id in proposed_differences.keys()
                            )

                            if remaining_work:
                                self.process_comparison(
                                    comparison_key,
                                    comparison_data,
                                    output_file,
                                    completed_diffs,
                                )
                                processed_count += 1
                            else:
                                skipped_count += 1

                self.timing_tracker.finish_processing()
                self.logger.info(
                    f"✅ Processing complete. {processed_count} processed, {skipped_count} skipped. Results saved to {output_file}"
                )

            except Exception as e:
                self.logger.error(f"Error processing {input_file}: {e}")


def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logging("biryani_processor.log")

    # Configuration
    LOCAL_IMAGES_PATH = "" # dir continaing all the extracted frames
    UPLOADED_FILES_JSON = "uploaded_frames.json"

    gemini_manager = GeminiFileManager(uploaded_files_json=UPLOADED_FILES_JSON, logger=logger)
    processor = BiryaniComparisonProcessor(LOCAL_IMAGES_PATH, gemini_manager, logger)

    # Process single file
    INPUT_FILE = "../../Data/BatchPipline/Comparison_batch_batch_1.json"
    OUTPUT_FILE = "completed_stage_3_batch_1.json"

    processor.process_json_file(INPUT_FILE, OUTPUT_FILE)

    # Process multiple files
    # INPUT_PATTERN = "../../Data/BatchPipline/Comparison_batch_batch_*.json"
    # batch_files = glob.glob(INPUT_PATTERN)
    # for batch_file in batch_files:
    #     basename = os.path.basename(batch_file)
    #     batch_name = basename.replace(".json", "")
    #     output_file = f"completed_stage_3_{batch_name}.json"
    #     processor.process_json_file(batch_file, output_file)


if __name__ == "__main__":
    main()
