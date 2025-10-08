import logging
from collections import Counter
from typing import Dict, List
from rich.console import Console
from rich.table import Table


class ActionFilter:
    """Handles filtering of duplicate actions within clips."""

    def __init__(self):
        """Initialize the action filter."""
        self.console = Console()

    def apply_action_filtering(
        self, data: List[Dict]
    ) -> tuple[List[Dict], Dict[str, int]]:
        """
        Remove duplicate actions within each clip/entry.

        Args:
            data: List of data entries to filter

        Returns:
            Tuple of (filtered_data, filter_stats)
        """
        if not data:
            raise ValueError("No data provided for filtering.")

        # Track filtering statistics
        filter_stats = {
            "clips_processed": 0,
            "clips_with_duplicates": 0,
            "total_actions_before": 0,
            "total_actions_after": 0,
            "duplicates_removed": 0,
        }

        # Process each entry
        filtered_data = []
        for entry in data:
            if "actions" in entry:
                original_actions = entry["actions"]
                filter_stats["total_actions_before"] += len(original_actions)
                filter_stats["clips_processed"] += 1

                # Remove duplicates while preserving order
                filtered_actions = []
                seen = set()
                for action in original_actions:
                    if action not in seen:
                        filtered_actions.append(action)
                        seen.add(action)

                # Update entry with filtered actions
                entry_copy = entry.copy()
                entry_copy["actions"] = filtered_actions
                filtered_data.append(entry_copy)

                filter_stats["total_actions_after"] += len(filtered_actions)

                # Track clips that had duplicates
                if len(original_actions) > len(filtered_actions):
                    filter_stats["clips_with_duplicates"] += 1
                    duplicates_in_clip = len(original_actions) - len(filtered_actions)
                    filter_stats["duplicates_removed"] += duplicates_in_clip
            else:
                filtered_data.append(entry)

        # Print filtering summary
        self._print_filtering_summary(filter_stats)

        logging.info(
            f"Action filtering completed. Removed {filter_stats['duplicates_removed']} "
            f"duplicate actions from {filter_stats['clips_with_duplicates']} clips."
        )

        return filtered_data, filter_stats

    def _print_filtering_summary(self, filter_stats: Dict[str, int]) -> None:
        """
        Print summary of filtering results.

        Args:
            filter_stats: Dictionary containing filtering statistics
        """
        table = Table(title="Action Filtering Summary")
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Value", justify="right", style="magenta")

        table.add_row("Total Clips Processed", str(filter_stats["clips_processed"]))
        table.add_row(
            "Clips with Duplicates", str(filter_stats["clips_with_duplicates"])
        )
        table.add_row("Total Actions Before", str(filter_stats["total_actions_before"]))
        table.add_row("Total Actions After", str(filter_stats["total_actions_after"]))
        table.add_row("Duplicates Removed", str(filter_stats["duplicates_removed"]))

        duplicate_percentage = (
            filter_stats["clips_with_duplicates"]
            / filter_stats["clips_processed"]
            * 100
            if filter_stats["clips_processed"] > 0
            else 0
        )
        table.add_row("Clips with Duplicates (%)", f"{duplicate_percentage:.1f}%")

        self.console.print(table)
