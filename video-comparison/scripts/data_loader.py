import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict
import shutil


class DataLoader:
    """Handles loading and organizing biryani cooking data."""

    def __init__(self, data_root: str):
        """
        Initialize the data loader.

        Args:
            data_root: Path to the root directory containing biryani data
        """
        self.data_root = Path(data_root)
        self.data = []
        self.action_counts = Counter()

    def load_all_data(self) -> List[Dict]:
        """
        Load all JSON data from the directory structure.

        Returns:
            List of all loaded data entries
        """
        self.data = []
        self.action_counts = Counter()

        for biryani_type in self.data_root.iterdir():
            if not biryani_type.is_dir():
                continue

            logging.debug(f"Processing directory: {biryani_type.name}")

            for video_dir in biryani_type.iterdir():
                if not video_dir.is_dir():
                    continue

                logging.debug(f"Processing video directory: {video_dir.name}")

                json_file = video_dir / "annotated.json"

                if json_file.exists():
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            video_data = json.load(f)

                        # Add metadata to each entry
                        for entry in video_data:
                            entry["biryani_type"] = biryani_type.name
                            entry["video_id"] = video_dir.name
                            self.data.append(entry)

                            # Count actions
                            if "actions" in entry:
                                for action in entry["actions"]:
                                    self.action_counts[action] += 1

                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        logging.error(f"Error loading {json_file}: {e}")

        logging.info(f"Loaded {len(self.data)} total entries")
        return self.data

    def get_action_counts(self) -> Counter:
        """
        Get the current action counts.

        Returns:
            Counter of action occurrences
        """
        return self.action_counts

    def get_data(self) -> List[Dict]:
        """
        Get the loaded data.

        Returns:
            List of all loaded data entries
        """
        return self.data

    def update_actions(self, action_mapping: Dict[str, str]) -> None:
        """
        Update the actions in the data using the provided mapping.

        Args:
            action_mapping: Dictionary mapping old action names to new ones
        """
        # Update data with new action names
        for entry in self.data:
            if "actions" in entry:
                entry["actions"] = [
                    action_mapping.get(action, action) for action in entry["actions"]
                ]

        # Recalculate action counts
        self.action_counts = Counter()
        for entry in self.data:
            if "actions" in entry:
                for action in entry["actions"]:
                    self.action_counts[action] += 1

    def _recalculate_action_counts(self) -> None:
        """Recalculate action counts from current data."""
        self.action_counts = Counter()
        for entry in self.data:
            if "actions" in entry:
                for action in entry["actions"]:
                    self.action_counts[action] += 1

    def save_clustered_data(self, output_root: str, create_backup: bool = True) -> None:
        """
        Save the clustered data maintaining the original directory structure.

        Args:
            output_root: Root directory to save processed data
            create_backup: Whether to create backup of original data if output exists
        """
        if not self.data:
            raise ValueError("No data to save. Call load_all_data() first.")

        output_path = Path(output_root)

        # Create backup if requested and output exists
        if create_backup and output_path.exists():
            backup_path = output_path.parent / f"{output_path.name}_backup"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(output_path, backup_path)
            logging.info(f"Created backup at {backup_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Group data by biryani_type and video_id to maintain structure
        structured_data = {}
        for entry in self.data:
            biryani_type = entry["biryani_type"]
            video_id = entry["video_id"]

            if biryani_type not in structured_data:
                structured_data[biryani_type] = {}
            if video_id not in structured_data[biryani_type]:
                structured_data[biryani_type][video_id] = []

            # Remove metadata before saving
            clean_entry = {
                k: v for k, v in entry.items() if k not in ["biryani_type", "video_id"]
            }
            structured_data[biryani_type][video_id].append(clean_entry)

        # Save data maintaining original structure
        for biryani_type, videos in structured_data.items():
            biryani_dir = output_path / biryani_type
            biryani_dir.mkdir(exist_ok=True)

            for video_id, entries in videos.items():
                video_dir = biryani_dir / video_id
                video_dir.mkdir(exist_ok=True)

                # Save as annotated.json
                json_file = video_dir / "annotated.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2, ensure_ascii=False)

                logging.debug(f"Saved clustered data to {json_file}")

        logging.info(f"Clustered data saved to {output_path}")
