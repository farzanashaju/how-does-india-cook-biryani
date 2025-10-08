import json
import os
import gc
import asyncio
import argparse
from typing import List, Dict, Any, Set
from pathlib import Path
from google.genai import types
from concurrent_file_manager import AsyncGeminiFileManager
from async_json_util import AsyncJsonUtils
from config import setup_logging, DEFAULT_CONFIG
from prompt import PROMPT_DIFFERENCE_MULTIFRAME, SYSTEM_PROMPT


class ImprovedAsyncBiryaniComparisonProcessor:
    def __init__(
        self,
        local_images_path: str,
        gemini_manager: AsyncGeminiFileManager,
        logger=None,
        max_concurrent: int = 50,  # Reduced from 750
        max_retries: int = 3,
        connection_limit: int = 100,  # New connection limit
    ):
        self.local_images_path = Path(local_images_path)
        self.gemini_manager = gemini_manager
        self.logger = logger or setup_logging()
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

        # Separate semaphores for different operations
        self.comparison_semaphore = asyncio.Semaphore(max_concurrent)
        self.connection_semaphore = asyncio.Semaphore(connection_limit)
        self.file_retrieval_semaphore = asyncio.Semaphore(20)  # Limit file retrievals

        self.failed_analyses = []

    def _cleanup_memory(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        self.logger.info("Memory cleanup completed")

    async def _wait_for_all_tasks(self):
        """Wait for all pending asyncio tasks to complete (excluding current task)."""
        current_task = asyncio.current_task()
        pending = [
            task for task in asyncio.all_tasks() if not task.done() and task != current_task
        ]
        if pending:
            self.logger.info(f"Waiting for {len(pending)} pending tasks to complete...")
            await asyncio.gather(*pending, return_exceptions=True)

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

    def _get_clip_frame_paths(self, clip_data: Dict) -> List[str]:
        """Get all frame paths for a clip."""
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
        frame_paths = []
        for frame_num in unique_frames:
            frame_name = self.gemini_manager.format_frame_name(video_url, frame_num)
            local_path = self.local_images_path / frame_name
            if local_path.exists():
                frame_paths.append(str(local_path))

        return frame_paths

    async def _get_file_objects_with_retry(self, frame_names: List[str]) -> List:
        """Get file objects with connection limiting and retry logic."""
        if not frame_names:
            return []

        async with self.file_retrieval_semaphore:
            try:
                # Add delay to prevent connection flooding
                await asyncio.sleep(0.1)
                return await self.gemini_manager.get_file_objects_for_processing(frame_names)
            except Exception as e:
                error_str = str(e).lower()
                if (
                    "cannot connect" in error_str
                    or "ssl" in error_str
                    or "invalid argument" in error_str
                ):
                    # Connection error - wait and retry once
                    self.logger.warning(f"Connection error retrieving files, retrying: {e}")
                    await asyncio.sleep(2)
                    try:
                        return await self.gemini_manager.get_file_objects_for_processing(
                            frame_names
                        )
                    except Exception as retry_error:
                        self.logger.error(f"Failed to retrieve files after retry: {retry_error}")
                        return []
                else:
                    self.logger.error(f"Non-connection error retrieving files: {e}")
                    return []

    async def _analyze_difference_with_retry(
        self,
        diff_id: str,
        difference: Dict,
        all_files: List,
        importance_context: str,
        clip1_frame_count: int,
        clip2_frame_count: int,
        action_class: str,
    ) -> Dict:
        """Analyze a difference with improved error handling and retry logic."""
        diff_name = difference.get("name", "")
        query_string = difference.get("query_string", "")

        for attempt in range(self.max_retries):
            try:
                async with self.connection_semaphore:
                    # Create prompt
                    formatted_prompt = PROMPT_DIFFERENCE_MULTIFRAME.format(
                        action=action_class,
                        query_string=query_string,
                        importance_context=importance_context,
                        total_frames=clip1_frame_count + clip2_frame_count,
                        clip1_range=f"1-{clip1_frame_count}",
                        clip2_start=clip1_frame_count + 1,
                        clip2_end=clip1_frame_count + clip2_frame_count,
                    )

                    # Add delay before API call
                    await asyncio.sleep(0.05)

                    # Generate response with timeout
                    response = await asyncio.wait_for(
                        self.gemini_manager.client.aio.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=all_files + [formatted_prompt],
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                temperature=0.7,
                                max_output_tokens=2048,
                            ),
                        ),
                        timeout=120.0,
                    )

                    # Parse JSON response
                    response_text = response.text.strip()
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        json_str = response_text[json_start:json_end].strip()
                    else:
                        json_str = response_text

                    analysis_result = json.loads(json_str)
                    return analysis_result

            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(15 * (attempt + 1))
                    continue
                else:
                    return self._create_failed_result(f"Timeout after {self.max_retries} attempts")

            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                else:
                    return self._create_failed_result(
                        f"JSON decode error after {self.max_retries} attempts"
                    )

            except Exception as e:
                error_str = str(e).lower()
                if any(
                    keyword in error_str
                    for keyword in [
                        "cannot connect",
                        "ssl",
                        "connection",
                        "invalid argument",
                    ]
                ):
                    if attempt < self.max_retries - 1:
                        delay = min(120, 30 * (2**attempt))
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return self._create_failed_result(
                            f"Connection error after {self.max_retries} attempts"
                        )

                if any(
                    keyword in error_str
                    for keyword in ["rate limit", "quota", "429", "resource_exhausted"]
                ):
                    if attempt < self.max_retries - 1:
                        delay = min(180, 60 * (2**attempt))
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return self._create_failed_result(
                            f"Rate limit exceeded after {self.max_retries} attempts"
                        )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                else:
                    return self._create_failed_result(
                        f"Error after {self.max_retries} attempts: {str(e)}"
                    )

        return self._create_failed_result("Unexpected error in retry logic")

    def _create_failed_result(self, error_message: str) -> Dict:
        """Create a standardized failed result with error flag."""
        return {
            "answer": "C",
            "confidence": 1,
            "difference_visible": False,
            "explanation": error_message,
            "_failed": True,
        }

    async def process_chunk_improved(
        self,
        chunk: Dict[str, Dict],
        output_file: str,
        failed_file: str,
        completed_comparisons: Dict[str, Set[str]],
    ) -> tuple[int, int]:
        """Improved chunk processing with better concurrency control."""

        # Step 1: Clean memory and wait for previous tasks
        self.logger.info("Step 1: Cleaning memory and waiting for pending tasks...")
        await self._wait_for_all_tasks()
        self._cleanup_memory()

        # Step 2: Chunk is already loaded (passed as parameter)
        self.logger.info(f"Step 2: Processing chunk with {len(chunk)} comparisons")

        # Step 3-4: Find valid comparisons that need processing
        self.logger.info("Steps 3-4: Finding comparisons that need processing...")
        valid_comparisons = []
        all_needed_frame_paths = []

        for comparison_key, comparison_data in chunk.items():
            if not isinstance(comparison_data, dict):
                continue

            completed_diffs = completed_comparisons.get(comparison_key, set())
            proposed_differences = comparison_data.get("proposed_differences", {})

            remaining_work = any(
                diff_id not in completed_diffs for diff_id in proposed_differences.keys()
            )

            if remaining_work:
                clips = comparison_data.get("clips", {})
                clip1_paths = self._get_clip_frame_paths(clips.get("1", {}))
                clip2_paths = self._get_clip_frame_paths(clips.get("2", {}))

                if clip1_paths and clip2_paths:
                    comparison_frame_paths = clip1_paths + clip2_paths
                    all_needed_frame_paths.extend(comparison_frame_paths)
                    valid_comparisons.append(
                        (
                            comparison_key,
                            comparison_data,
                            completed_diffs,
                            comparison_frame_paths,
                        )
                    )
                else:
                    self.logger.warning(f"⚠️ Missing frames for {comparison_key}")

        if not valid_comparisons:
            self.logger.info("No valid comparisons need processing")
            return 0, len(chunk)

        # Remove duplicate frame paths
        unique_frame_paths = list(dict.fromkeys(all_needed_frame_paths))
        self.logger.info(
            f"Found {len(valid_comparisons)} comparisons needing {len(unique_frame_paths)} unique images"
        )

        # Steps 5-8: Prepare all files using the new file manager method
        self.logger.info("Steps 5-8: Preparing all files for processing...")
        files_ready, missing_files = await self.gemini_manager.prepare_files_for_chunk(
            unique_frame_paths
        )
        if not files_ready:
            self.logger.error(f"❌ Failed to prepare files. Missing: {len(missing_files)} files")
            return 0, len(chunk)

        # Steps 9-10: Process comparisons in smaller batches to prevent connection overload
        self.logger.info(
            f"Steps 9-10: Processing {len(valid_comparisons)} comparisons in controlled batches..."
        )

        batch_size = min(self.max_concurrent, 25)  # Process in smaller batches
        processed_count = 0

        for i in range(0, len(valid_comparisons), batch_size):
            batch = valid_comparisons[i : i + batch_size]
            self.logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(valid_comparisons) - 1) // batch_size + 1} ({len(batch)} comparisons)"
            )

            batch_tasks = []
            for (
                comparison_key,
                comparison_data,
                completed_diffs,
                frame_paths,
            ) in batch:
                task = self._process_single_comparison_improved(
                    comparison_key,
                    comparison_data,
                    output_file,
                    failed_file,
                    completed_diffs,
                    frame_paths,
                )
                batch_tasks.append(task)

            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error: {result}")
                else:
                    processed_count += 1

            # Small delay between batches to prevent overwhelming the API
            if i + batch_size < len(valid_comparisons):
                await asyncio.sleep(1)

        # Step 11-13: Cleanup
        self.logger.info("Steps 11-13: Final cleanup...")
        await self._wait_for_all_tasks()
        self._cleanup_memory()

        return processed_count, len(chunk) - len(valid_comparisons)

    async def _process_single_comparison_improved(
        self,
        comparison_key: str,
        comparison_data: Dict,
        output_file: str,
        failed_file: str,
        completed_differences: Set[str],
        frame_paths: List[str],
    ):
        """Process a single comparison with improved connection handling."""
        async with self.comparison_semaphore:
            try:
                action_class = comparison_data.get("action_class", "")
                clips = comparison_data.get("clips", {})
                proposed_differences = comparison_data.get("proposed_differences", {})
                action_stages = comparison_data.get("action_stages", [])
                unique_identifier = comparison_data.get("unique_identifier", "")

                # Get remaining differences
                remaining_differences = []
                for diff_id, difference in proposed_differences.items():
                    if diff_id not in completed_differences and isinstance(difference, dict):
                        remaining_differences.append((diff_id, difference))

                if not remaining_differences:
                    return

                # Get frame names from paths
                clip1_paths = self._get_clip_frame_paths(clips.get("1", {}))
                clip2_paths = self._get_clip_frame_paths(clips.get("2", {}))

                clip1_frame_names = [Path(path).name for path in clip1_paths]
                clip2_frame_names = [Path(path).name for path in clip2_paths]

                # Get file objects with improved connection handling
                clip1_files = await self._get_file_objects_with_retry(clip1_frame_names)
                clip2_files = await self._get_file_objects_with_retry(clip2_frame_names)

                if not clip1_files or not clip2_files:
                    self.logger.warning(f"⚠️ Insufficient files for {comparison_key}")
                    return

                all_files = clip1_files + clip2_files
                clip1_frame_count = len(clip1_files)
                clip2_frame_count = len(clip2_files)

                # Process differences with controlled concurrency
                for diff_id, difference in remaining_differences:
                    diff_name = difference.get("name", "")
                    query_string = difference.get("query_string", "")

                    if not diff_name or not query_string:
                        continue

                    importance_context = self.find_important_frames(
                        action_stages,
                        diff_name,
                        clips,
                        clip1_frame_count,
                        clip2_frame_count,
                    )

                    try:
                        analysis_result = await self._analyze_difference_with_retry(
                            diff_id,
                            difference,
                            all_files,
                            importance_context,
                            clip1_frame_count,
                            clip2_frame_count,
                            action_class,
                        )

                        difference_result = {
                            "comparison_key": comparison_key,
                            "unique_identifier": unique_identifier,
                            "action_class": action_class,
                            "difference_id": diff_id,
                            "difference_name": difference.get("name", ""),
                            "query_string": difference.get("query_string", ""),
                            "clips": clips,
                            "analysis": analysis_result,
                        }

                        if analysis_result.get("_failed", False):
                            clean_result = dict(analysis_result)
                            clean_result.pop("_failed", None)
                            difference_result["analysis"] = clean_result

                            await AsyncJsonUtils.save_failed_analysis(
                                failed_file, difference_result, self.logger
                            )
                            self.logger.warning(
                                f"Failed: {difference.get('name', '')} - {clean_result.get('explanation', '')[:100]}"
                            )
                        else:
                            await AsyncJsonUtils.save_intermediate_result(
                                output_file,
                                comparison_key,
                                difference_result,
                                self.logger,
                            )
                            self.logger.info(
                                f"Success: {analysis_result.get('answer', 'unknown')} - {difference.get('name', '')}"
                            )

                    except Exception as e:
                        self.logger.error(f"Error processing {difference.get('name', '')}: {e}")
                        failed_result = {
                            "comparison_key": comparison_key,
                            "unique_identifier": unique_identifier,
                            "action_class": action_class,
                            "difference_id": diff_id,
                            "difference_name": difference.get("name", ""),
                            "query_string": difference.get("query_string", ""),
                            "clips": clips,
                            "analysis": {
                                "answer": "C",
                                "confidence": 1,
                                "difference_visible": False,
                                "explanation": f"Processing error: {str(e)}",
                            },
                        }
                        await AsyncJsonUtils.save_failed_analysis(
                            failed_file, failed_result, self.logger
                        )

            except Exception as e:
                self.logger.error(f"Error processing comparison {comparison_key}: {e}")

    async def process_json_file_improved(
        self,
        input_file: str,
        output_file: str,
        failed_file: str = "failed_analyses.json",
        chunk_size: int = 500,  # Reduced chunk size
    ):
        """Process large JSON file with improved chunk handling."""
        try:
            file_size = os.path.getsize(input_file) / (1024 * 1024)
            self.logger.info(
                f"Processing large file ({file_size:.1f} MB) in chunks of {chunk_size}"
            )

            total_processed = 0
            total_skipped = 0
            chunk_count = 0

            async for chunk in AsyncJsonUtils.chunked_json_reader(
                input_file, chunk_size, self.logger
            ):
                chunk_count += 1
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"STARTING CHUNK {chunk_count} ({len(chunk)} comparisons)")
                self.logger.info(f"{'=' * 60}")

                # Load completed comparisons fresh for each chunk
                completed_comparisons = await AsyncJsonUtils.load_existing_results(
                    output_file, self.logger
                )

                processed, skipped = await self.process_chunk_improved(
                    chunk,
                    output_file,
                    failed_file,
                    completed_comparisons,
                )

                total_processed += processed
                total_skipped += skipped

                self.logger.info(f"{'=' * 60}")
                self.logger.info(f"CHUNK {chunk_count} COMPLETE")
                self.logger.info(f"Processed: {processed}, Skipped: {skipped}")
                self.logger.info(
                    f"Total so far - Processed: {total_processed}, Skipped: {total_skipped}"
                )
                self.logger.info(f"{'=' * 60}\n")

                # Inter-chunk delay to prevent API overload
                await asyncio.sleep(2)

            self.logger.info(
                f"✅ Processing complete. {total_processed} comparisons processed, "
                f"{total_skipped} skipped. Results saved to {output_file}"
            )

            if self.failed_analyses:
                self.logger.info(
                    f"⚠️ {len(self.failed_analyses)} analyses failed and saved to {failed_file}"
                )

        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Improved biryani processor with better concurrency control"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,  # Reduced default
        help="Maximum concurrent requests (default: 25)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,  # Reduced default
        help="Chunk size for processing (default: 500)",
    )
    parser.add_argument(
        "--connection-limit",
        type=int,
        default=75,  # New parameter
        help="Maximum concurrent connections (default: 50)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="../../Data/BatchPipline/output/TypevType/hyderabadi_biryani_vs_lucknow_awadhi_biryani.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="completed_stage_3.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--failed-file",
        type=str,
        default="failed_analyses.json",
        help="Failed analyses JSON file path",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("improved_biryani_processor.log")

    # Configuration
    LOCAL_IMAGES_PATH = DEFAULT_CONFIG["LOCAL_IMAGES_PATH"]
    UPLOADED_FILES_JSON = DEFAULT_CONFIG["UPLOADED_FILES_JSON"]

    # Initialize with improved file manager
    gemini_manager = AsyncGeminiFileManager(uploaded_files_json=UPLOADED_FILES_JSON, logger=logger)

    processor = ImprovedAsyncBiryaniComparisonProcessor(
        LOCAL_IMAGES_PATH,
        gemini_manager,
        logger,
        max_concurrent=args.max_concurrent,
        connection_limit=args.connection_limit,
    )

    # Process file with improved chunking
    await processor.process_json_file_improved(
        args.input_file,
        args.output_file,
        args.failed_file,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    asyncio.run(main())
