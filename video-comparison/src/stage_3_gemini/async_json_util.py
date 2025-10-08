import json
import os
import ijson
import logging
import asyncio
import aiofiles
from typing import Dict, Set, Any, AsyncIterator


class AsyncJsonUtils:
    """Async utilities for handling JSON files and data."""

    @staticmethod
    async def load_existing_results(
        output_file: str, logger: logging.Logger
    ) -> Dict[str, Set[str]]:
        """Load existing results and return completed comparisons and their differences."""
        completed_comparisons = {}

        if not os.path.exists(output_file):
            logger.info(f"Output file {output_file} doesn't exist, starting fresh")
            return completed_comparisons

        try:
            async with aiofiles.open(output_file, "r", encoding="utf-8") as f:
                content = await f.read()
                existing_data = json.loads(content)

            for comparison_key, data in existing_data.items():
                completed_diffs = set()
                proposed_differences = data.get("proposed_differences", {})

                for diff_id, diff_data in proposed_differences.items():
                    if diff_data.get("status") == "success" and diff_data.get("winner"):
                        completed_diffs.add(diff_id)

                completed_comparisons[comparison_key] = completed_diffs

            total_completed = sum(len(diffs) for diffs in completed_comparisons.values())
            logger.info(
                f"Found {len(completed_comparisons)} existing comparisons with {total_completed} completed differences"
            )

        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")

        return completed_comparisons

    @staticmethod
    async def save_intermediate_result(
        output_file: str,
        comparison_key: str,
        difference_result: Dict,
        logger: logging.Logger,
    ):
        """Save result immediately after processing each difference."""
        # Use a lock to prevent concurrent writes
        if not hasattr(AsyncJsonUtils, "_write_lock"):
            AsyncJsonUtils._write_lock = asyncio.Lock()

        async with AsyncJsonUtils._write_lock:
            # Load existing results or create new structure
            if os.path.exists(output_file):
                async with aiofiles.open(output_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    existing_data = json.loads(content)
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
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(existing_data, indent=2, ensure_ascii=False))

            logger.info(
                f"💾 Saved result for {comparison_key} - {difference_result.get('difference_name', '')}"
            )

    @staticmethod
    async def save_failed_analysis(
        failed_file: str,
        difference_result: Dict,
        logger: logging.Logger,
    ):
        """Save failed analysis to a separate file."""
        # Use a separate lock for failed analyses
        if not hasattr(AsyncJsonUtils, "_failed_write_lock"):
            AsyncJsonUtils._failed_write_lock = asyncio.Lock()

        async with AsyncJsonUtils._failed_write_lock:
            # Load existing failed results or create new structure
            if os.path.exists(failed_file):
                async with aiofiles.open(failed_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    existing_data = json.loads(content)
            else:
                existing_data = {"failed_analyses": []}

            # Add the failed result
            existing_data["failed_analyses"].append(
                {
                    "comparison_key": difference_result.get("comparison_key", ""),
                    "unique_identifier": difference_result.get("unique_identifier", ""),
                    "action_class": difference_result.get("action_class", ""),
                    "difference_id": difference_result.get("difference_id", ""),
                    "difference_name": difference_result.get("difference_name", ""),
                    "query_string": difference_result.get("query_string", ""),
                    "error_reason": difference_result.get("analysis", {}).get("explanation", ""),
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

            # Save updated failed results
            async with aiofiles.open(failed_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(existing_data, indent=2, ensure_ascii=False))

            logger.debug(
                f"💾 Saved failed analysis for {difference_result.get('difference_name', '')}"
            )

    @staticmethod
    async def chunked_json_reader(
        input_file: str, chunk_size: int, logger: logging.Logger
    ) -> AsyncIterator[Dict[str, Any]]:
        """Read JSON file in chunks to handle large files efficiently."""
        logger.info(f"Reading {input_file} in chunks of {chunk_size}")

        def _sync_chunk_reader():
            """Synchronous generator that yields chunks."""
            with open(input_file, "rb") as f:
                current_chunk = {}
                item_count = 0

                for key, value in ijson.kvitems(f, ""):
                    if key.startswith("Action_class_comparison"):
                        current_chunk[key] = value
                        item_count += 1

                        logger.debug(
                            f"Added item {key} to current chunk, total items: {item_count}"
                        )

                        # Yield chunk when it reaches the desired size
                        if item_count >= chunk_size:
                            logger.debug(f"Yielding chunk with {len(current_chunk)} items")
                            yield current_chunk
                            current_chunk = {}
                            item_count = 0

                # Yield remaining items
                if current_chunk:
                    logger.debug(f"Yielding final chunk with {len(current_chunk)} items")
                    yield current_chunk

        # Run the synchronous chunk reader in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        # Use a queue to handle the async iteration
        queue = asyncio.Queue()
        finished = False

        async def producer():
            nonlocal finished
            try:
                # Run the sync generator in a thread
                for chunk in await loop.run_in_executor(None, lambda: list(_sync_chunk_reader())):
                    await queue.put(chunk)
            except Exception as e:
                logger.error(f"Error in chunk reader: {e}")
                await queue.put(e)
            finally:
                finished = True
                await queue.put(None)  # Sentinel value

        # Start the producer
        producer_task = asyncio.create_task(producer())

        try:
            while not finished or not queue.empty():
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=10.0)
                    if chunk is None:  # Sentinel value
                        break
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk
                except asyncio.TimeoutError:
                    if finished:
                        break
                    logger.warning("Timeout waiting for next chunk")
                    continue
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def get_file_statistics(file_path: str, logger: logging.Logger) -> Dict[str, Any]:
        """Get statistics about the JSON file."""
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            # Count total comparisons
            comparison_count = 0
            async for chunk in AsyncJsonUtils.chunked_json_reader(file_path, 1000, logger):
                comparison_count += len(chunk)

            return {
                "file_size_mb": file_size,
                "total_comparisons": comparison_count,
                "estimated_processing_time": comparison_count * 2,  # Rough estimate in seconds
            }
        except Exception as e:
            logger.error(f"Error getting file statistics: {e}")
            return {"error": str(e)}
