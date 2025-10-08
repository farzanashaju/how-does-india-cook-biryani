import json
import os
import ijson
import logging
from typing import Dict, Set, Any, Iterator


class JsonUtils:
    """Utilities for handling JSON files and data."""

    @staticmethod
    def load_existing_results(output_file: str, logger: logging.Logger) -> Dict[str, Set[str]]:
        """Load existing results and return completed comparisons and their differences."""
        completed_comparisons = {}

        if not os.path.exists(output_file):
            logger.info(f"Output file {output_file} doesn't exist, starting fresh")
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
            logger.info(
                f"Found {len(completed_comparisons)} existing comparisons with {total_completed} completed differences"
            )

        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")

        return completed_comparisons

    @staticmethod
    def save_intermediate_result(
        output_file: str,
        comparison_key: str,
        difference_result: Dict,
        logger: logging.Logger,
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

        logger.info(
            f"💾 Saved result for {comparison_key} - {difference_result.get('difference_name', '')}"
        )

    @staticmethod
    def chunked_json_reader(
        input_file: str, chunk_size: int, logger: logging.Logger
    ) -> Iterator[Dict[str, Any]]:
        """Read JSON file in chunks to handle large files efficiently."""
        logger.info(f"Reading {input_file} in chunks of {chunk_size}")

        with open(input_file, "rb") as f:
            current_chunk = {}
            item_count = 0

            for key, value in ijson.kvitems(f, ""):
                if key.startswith("Action_class_comparison"):
                    current_chunk[key] = value
                    item_count += 1

                    logger.debug(f"Added item {key} to current chunk, total items: {item_count}")

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
