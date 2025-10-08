import json
import os
import time
from typing import List, Dict, Any, Set
from pathlib import Path
from google.genai import types

from file_manager import GeminiFileManager
from timing_tracker import TimingTracker
from json_util import JsonUtils
from config import setup_logging, DEFAULT_CONFIG
from prompt import PROMPT_DIFFERENCE_MULTIFRAME, SYSTEM_PROMPT


class BiryaniComparisonProcessor:
    def __init__(
        self,
        local_images_path: str,
        gemini_manager: GeminiFileManager,
        logger=None,
    ):
        self.local_images_path = Path(local_images_path)
        self.gemini_manager = gemini_manager
        self.logger = logger or setup_logging()
        self.timing_tracker = TimingTracker(self.logger)

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

            # Upload frames for both clips
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
                        model="gemini-2.5-flash-lite",
                        contents=all_files + [formatted_prompt],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.7,
                            max_output_tokens=2048,
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
                    JsonUtils.save_intermediate_result(
                        output_file, comparison_key, difference_result, self.logger
                    )
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
                    JsonUtils.save_intermediate_result(
                        output_file, comparison_key, error_result, self.logger
                    )

        except Exception as e:
            self.logger.error(f"Error processing {comparison_key}: {e}")

        finally:
            # Record timing for this comparison
            comparison_time = time.time() - comparison_start_time
            self.timing_tracker.record_comparison_time(comparison_key, comparison_time)

    def process_json_file(self, input_file: str, output_file: str, chunk_size: int = 1000):
        """Process large JSON file in chunks."""
        try:
            completed_comparisons = JsonUtils.load_existing_results(output_file, self.logger)

            file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
            self.logger.info(f"Processing large file ({file_size:.1f} MB) in chunks")

            estimated_comparisons = int(file_size * 1000 / 2.9)  # Rough estimate
            self.timing_tracker.start_processing(estimated_comparisons)

            total_processed = 0
            total_skipped = 0

            for chunk in JsonUtils.chunked_json_reader(input_file, chunk_size, self.logger):
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

                # Add pause between chunks to avoid overwhelming the API
                time.sleep(2)

            self.timing_tracker.finish_processing()
            self.logger.info(
                f"✅ Processing complete. {total_processed} comparisons processed, "
                f"{total_skipped} skipped. Results saved to {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")


def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logging("biryani_processor.log")

    # Configuration
    LOCAL_IMAGES_PATH = DEFAULT_CONFIG["LOCAL_IMAGES_PATH"]
    UPLOADED_FILES_JSON = DEFAULT_CONFIG["UPLOADED_FILES_JSON"]

    gemini_manager = GeminiFileManager(uploaded_files_json=UPLOADED_FILES_JSON, logger=logger)
    processor = BiryaniComparisonProcessor(LOCAL_IMAGES_PATH, gemini_manager, logger)

    # Process file with chunking
    INPUT_FILE = "../../Data/BatchPipline/Comparison_script/Comparison_batch_1.json"
    OUTPUT_FILE = "completed_stage_3.json"

    processor.process_json_file(INPUT_FILE, OUTPUT_FILE, chunk_size=DEFAULT_CONFIG["CHUNK_SIZE"])


if __name__ == "__main__":
    main()
