"""
Convert video-based biryani dataset to action-based dataset.
Transforms the structure from videos containing actions to actions containing video clips.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import math


def parse_timestamp(timestamp_str: str) -> Dict[str, int]:
    """
    Convert timestamp string like '0-10' to {'start': 0, 'end': 10}

    Args:
        timestamp_str: String in format 'start-end'

    Returns:
        Dictionary with start and end timestamps
    """
    try:
        start_str, end_str = timestamp_str.split("-")
        return {"start": int(start_str.strip()), "end": int(end_str.strip())}
    except Exception as e:
        print(f"Warning: Could not parse timestamp '{timestamp_str}'. Error: {e}")
        return {"start": 0, "end": 0}


def load_video_annotations(input_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all annotated.json files from the input directory structure.

    Args:
        input_dir: Path to the directory containing biryani type folders

    Returns:
        Dictionary mapping biryani types to list of video annotations
    """
    video_data = {}
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")

    # Iterate through biryani type directories
    for biryani_dir in input_path.iterdir():
        if not biryani_dir.is_dir():
            continue

        biryani_type = biryani_dir.name
        video_data[biryani_type] = []

        # Iterate through video directories
        for video_dir in biryani_dir.iterdir():
            if not video_dir.is_dir():
                continue

            annotation_file = video_dir / "annotated.json"
            if annotation_file.exists():
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        annotations = json.load(f)
                        # Add biryani type to each annotation
                        for annotation in annotations:
                            annotation["biryani_type"] = biryani_type
                        video_data[biryani_type].extend(annotations)
                except Exception as e:
                    print(f"Warning: Could not load {annotation_file}. Error: {e}")

    return video_data


def create_action_based_dataset(video_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Transform video-based data to action-based dataset structure.

    Args:
        video_data: Dictionary mapping biryani types to video annotations

    Returns:
        Action-based dataset structure
    """
    action_dataset = {}

    # Group all clips by action
    action_clips = defaultdict(list)

    # Process all video data
    for biryani_type, video_annotations in video_data.items():
        for annotation in video_annotations:
            actions = annotation.get("actions", [])

            for action in actions:
                # Normalize action name (lowercase, strip whitespace)
                action_name = action.strip().lower()

                clip_info = {
                    "url": annotation.get("url", ""),
                    "timestamp": annotation.get("timestamp", "0-0"),
                    "biryani_type": biryani_type,
                    "title": annotation.get("title", ""),
                    "ingredients": annotation.get("ingredients", []),
                    "utensils": annotation.get("utensils", []),
                }

                action_clips[action_name].append(clip_info)

    # Create the final dataset structure
    for action_class, clips in action_clips.items():

        # Add clips to the action
        for idx, clip in enumerate(clips, start=1):
            clip_entry = {
                "url": clip["url"],
                "timestamp": parse_timestamp(clip["timestamp"]),
                "type": clip["biryani_type"],
                "retrieval_frames": {},
            }
            action_dataset[action_class]["Clips"][str(idx)] = clip_entry

    return action_dataset


def calculate_pipeline_timing(action_dataset: Dict[str, Any]) -> Dict[str, str]:
    """
    Calculate pipeline timing for each component.

    Args:
        action_dataset: The action-based dataset

    Returns:
        Dictionary with timing calculations as strings
    """
    num_actions = len(action_dataset)

    # Calculate total combinations for differentiator
    total_combinations = 0
    for action_name, action_info in action_dataset.items():
        num_clips = len(action_info["Clips"])
        if num_clips >= 2:
            # nC2 = n! / (2! * (n-2)!) = n * (n-1) / 2
            combinations = (num_clips * (num_clips - 1)) // 2
            total_combinations += combinations

    timing_calculations = {
        "proposer": f"{num_actions} * time_per_step",
        "frame_localizer": f"{num_actions} * action_steps * time_per_step",
        "differentiator": f"{total_combinations} * proposed_differences * time_per_step",
    }

    return timing_calculations


def main():
    """Main function to handle command line arguments and execute the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert video-based biryani dataset to action-based dataset"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input directory containing biryani type folders with video annotations",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to output JSON file for the action-based dataset",
    )

    args = parser.parse_args()

    try:
        print(f"Loading video annotations from: {args.input_dir}")
        video_data = load_video_annotations(args.input_dir)

        total_videos = sum(len(annotations) for annotations in video_data.values())
        print(
            f"Loaded {total_videos} video annotations from {len(video_data)} biryani types"
        )

        print("Converting to action-based dataset...")
        action_dataset = create_action_based_dataset(video_data)

        print(f"Created dataset with {len(action_dataset)} unique actions")

        # Save the output
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(action_dataset, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved action-based dataset to: {args.output_file}")

        # Print some statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total unique actions: {len(action_dataset)}")

        total_clips = sum(
            len(action_info["Clips"]) for action_info in action_dataset.values()
        )
        print(f"Total clips: {total_clips}")

        # Show top 10 most common actions
        action_counts = [
            (action, len(info["Clips"])) for action, info in action_dataset.items()
        ]
        action_counts.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 most common actions:")
        for i, (action, count) in enumerate(action_counts[:10], 1):
            print(f"{i:2d}. {action}: {count} clips")

        # Calculate and display pipeline timing
        timing = calculate_pipeline_timing(action_dataset)
        print(f"\n=== Pipeline Timing Calculations ===")
        print(f"Proposer: {timing['proposer']}")
        print(f"Frame Localizer: {timing['frame_localizer']}")
        print(f"Differentiator: {timing['differentiator']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
