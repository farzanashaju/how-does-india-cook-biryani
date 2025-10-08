import logging
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple


class DatasetBuilder:
    """Handles building action-based datasets from video data."""

    def __init__(self):
        """Initialize the dataset builder."""
        pass

    def parse_timestamp(self, timestamp: str) -> Tuple[int, int]:
        """
        Parse timestamp string to start and end times.

        Args:
            timestamp: Timestamp string in format "start-end"

        Returns:
            Tuple of (start_time, end_time)
        """
        try:
            parts = timestamp.split("-")
            if len(parts) == 2:
                start_time = int(parts[0])
                end_time = int(parts[1])
                return start_time, end_time
            else:
                logging.warning(f"Invalid timestamp format: {timestamp}")
                return 0, 0
        except ValueError:
            logging.warning(f"Could not parse timestamp: {timestamp}")
            return 0, 0

    def merge_consecutive_clips(
        self, clips: List[Dict], gap_threshold: int = 20
    ) -> List[Dict]:
        """
        Merge consecutive clips with small gaps.

        Args:
            clips: List of clip dictionaries
            gap_threshold: Maximum gap in seconds to merge clips

        Returns:
            List of merged clips
        """
        if not clips:
            return []

        # Sort clips by timestamp
        sorted_clips = sorted(
            clips, key=lambda x: self.parse_timestamp(x["timestamp"])[0]
        )

        merged_clips = []
        current_clip = sorted_clips[0].copy()
        current_start, current_end = self.parse_timestamp(current_clip["timestamp"])

        for clip in sorted_clips[1:]:
            clip_start, clip_end = self.parse_timestamp(clip["timestamp"])

            # Check if clips should be merged (gap is within threshold)
            if (
                clip_start <= current_end + gap_threshold
                and clip["url"] == current_clip["url"]
            ):
                # Merge clips by extending the end time
                current_end = max(current_end, clip_end)
                current_clip["timestamp"] = f"{current_start}-{current_end}"
            else:
                # Add current merged clip and start new one
                merged_clips.append(current_clip)
                current_clip = clip.copy()
                current_start, current_end = clip_start, clip_end

        # Add the last clip
        merged_clips.append(current_clip)

        return merged_clips

    def create_action_based_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Transform video-based data to action-based dataset structure.

        Args:
            data: List of video data entries

        Returns:
            Action-based dataset structure
        """
        if not data:
            raise ValueError("No data provided for dataset creation.")

        action_dataset = {}

        # Group all clips by action
        action_clips = defaultdict(list)

        # Process all video data
        for entry in data:
            actions = entry.get("actions", [])
            biryani_type = entry.get("biryani_type", "")

            for action in actions:
                # Normalize action name (lowercase, strip whitespace)
                action_name = action.strip().lower()

                clip_info = {
                    "url": entry.get("url", ""),
                    "timestamp": entry.get("timestamp", "0-0"),
                    "biryani_type": biryani_type,
                }
                action_clips[action_name].append(clip_info)

        # Create the final dataset structure
        for action_class, clips in action_clips.items():
            # Merge consecutive clips
            merged_clips = self.merge_consecutive_clips(clips)

            action_dataset[action_class] = {"Clips": {}}

            # Add merged clips to the action
            for idx, clip in enumerate(merged_clips, start=1):
                start_time, end_time = self.parse_timestamp(clip["timestamp"])

                clip_entry = {
                    "url": clip["url"],
                    "timestamp": {"start": start_time, "end": end_time},
                    "type": clip["biryani_type"],
                    "retrieval_frames": {},
                }

                action_dataset[action_class]["Clips"][str(idx)] = clip_entry

        return action_dataset

    def merge_consecutive_clips_in_dataset(
        self, action_dataset: Dict[str, Any], gap_threshold: int = 10
    ) -> Dict[str, Any]:
        """
        Merge consecutive clips within each action class in the dataset.

        Args:
            action_dataset: Action-based dataset
            gap_threshold: Maximum gap in seconds to merge clips

        Returns:
            Dataset with merged clips
        """
        combined_dataset = {}

        for action_class, action_data in action_dataset.items():
            clips = action_data.get("Clips", {})

            # Convert clips to list for processing
            clip_list = []
            for clip_id, clip_data in clips.items():
                clip_info = {
                    "url": clip_data["url"],
                    "timestamp": f"{clip_data['timestamp']['start']}-{clip_data['timestamp']['end']}",
                    "biryani_type": clip_data["type"],
                }
                clip_list.append(clip_info)

            # Merge consecutive clips
            merged_clips = self.merge_consecutive_clips(clip_list, gap_threshold)

            # Rebuild the clips structure
            combined_dataset[action_class] = {"Clips": {}}

            for idx, clip in enumerate(merged_clips, start=1):
                start_time, end_time = self.parse_timestamp(clip["timestamp"])

                clip_entry = {
                    "url": clip["url"],
                    "timestamp": {"start": start_time, "end": end_time},
                    "type": clip["biryani_type"],
                    "retrieval_frames": {},
                }

                combined_dataset[action_class]["Clips"][str(idx)] = clip_entry

        return combined_dataset

    def save_action_dataset(self, data: List[Dict], output_file: str) -> None:
        """
        Create and save action-based dataset to a JSON file.

        Args:
            data: List of video data entries
            output_file: Path to output JSON file
        """
        action_dataset = self.create_action_based_dataset(data)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(action_dataset, f, indent=2, ensure_ascii=False)

        logging.info(f"Action dataset saved to {output_path}")
        logging.info(f"Dataset contains {len(action_dataset)} unique actions")

    def save_combined_action_dataset(self, data: List[Dict], output_file: str) -> None:
        """
        Create action-based dataset, merge consecutive clips, and save to JSON file.

        Args:
            data: List of video data entries
            output_file: Path to output JSON file
        """
        # First create the regular action dataset
        action_dataset = self.create_action_based_dataset(data)

        # Then merge consecutive clips
        logging.info("Merging consecutive clips in action dataset...")
        combined_dataset = self.merge_consecutive_clips_in_dataset(
            action_dataset, gap_threshold=10
        )

        # Save the combined dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_dataset, f, indent=2, ensure_ascii=False)

        logging.info(f"Combined action dataset saved to {output_path}")
        logging.info(f"Dataset contains {len(combined_dataset)} unique actions")
