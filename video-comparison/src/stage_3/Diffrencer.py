import json
import os
import time
from typing import Dict, List
from datetime import datetime

import cv2
import torch
from PIL import Image
from tqdm import tqdm

from stage_3.prompts import (
    PROMPT_DIFFERENCE_1FRAME,
    PROMPT_DIFFERENCE_MULTIFRAME,
)
from stage_3.utils import (
    call_llm,
    load_model_and_processor,
    print_stage_info,
)


def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def print_step_timing(step_name, duration, indent=0):
    """Print timing for a specific step."""
    prefix = "  " * indent
    print(f"{prefix}⏱️  {step_name}: {format_time(duration)}")


class Differencer:
    """
    Stage 3: Action Differencer.
    Given localized frames for each difference, it calls the VLM to decide:
    (a) Video A, (b) Video B, or (c) Similar.
    """

    def __init__(
        self,
        model_path: str = "DAMO-NLP-SG/VideoLLaMA3-7B-Image",
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens

        self.model, self.processor = load_model_and_processor(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Differencer initialized on device: {self.device}")

    def create_comparison_pairs(self, action_data: Dict) -> List[Dict]:
        """
        Create all possible pairs from the clips for comparison.
        """

        clips = action_data.get("Clips", {})
        pairs = []

        clip_items = list(clips.items())

        # Create all possible pairs
        for i in range(len(clip_items)):
            for j in range(i + 1, len(clip_items)):
                clip_a_id, clip_a_data = clip_items[i]
                clip_b_id, clip_b_data = clip_items[j]

                def extract_timestamp(url):
                    try:
                        filename = os.path.basename(url)
                        # Extract timestamp from filename like "E_gWBBjYkjE_340s_350s.mp4"
                        parts = filename.split("_")
                        if len(parts) >= 3:
                            start = parts[-2].replace("s", "")
                            end = parts[-1].replace("s", "").replace(".mp4", "")
                            return f"{start}s - {end}s"
                    except Exception as e:
                        print(f"Error {e}")
                        pass
                    return "unknown"

                pairs.append(
                    {
                        "pair_id": f"{clip_a_id}_vs_{clip_b_id}",
                        "clip_a": {
                            "id": clip_a_id,
                            "url": clip_a_data["url"],
                            "timestamp": extract_timestamp(clip_a_data["url"]),
                            "type": clip_a_data["type"],
                            "retrieval_frames": clip_a_data.get("retrival_frames", {}),
                        },
                        "clip_b": {
                            "id": clip_b_id,
                            "url": clip_b_data["url"],
                            "timestamp": extract_timestamp(clip_b_data["url"]),
                            "type": clip_b_data["type"],
                            "retrieval_frames": clip_b_data.get("retrival_frames", {}),
                        },
                    }
                )

        return pairs

    def load_frames_from_video(
        self, video_path: str, frame_indices: List[int]
    ) -> List[Image.Image]:
        """
        Load specific frames from video using OpenCV (simplified since AV1 issues are fixed).
        """

        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames

        try:
            for frame_idx in sorted(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)
                else:
                    print(
                        f"Warning: Could not read frame {frame_idx} from {video_path}"
                    )

        except Exception as e:
            print(f"Error loading frames from {video_path}: {e}")
        finally:
            cap.release()

        video_name = os.path.basename(video_path)

        return frames

    def process_single_difference(
        self,
        action: str,
        difference_name: str,
        difference_description: str,
        frames_video_a: List[Image.Image],
        frames_video_b: List[Image.Image],
        pair_info: Dict,
    ) -> Dict:
        """
        Process a single difference comparison between two videos.
        """
        start_time = time.time()

        try:
            # Prompt preparation timing
            if len(frames_video_a) == 1 and len(frames_video_b) == 1:
                prompt = PROMPT_DIFFERENCE_1FRAME.format(
                    action=action, query_string=difference_description
                )
            else:
                num_frames = len(frames_video_a)
                time_diff = 0.5
                prompt = PROMPT_DIFFERENCE_MULTIFRAME.format(
                    action=action,
                    num_frames=num_frames,
                    time_diff=time_diff,
                    query_string=difference_description,
                )

            # Order: first A, then B
            all_frames = frames_video_a + frames_video_b

            # Call LLM timing
            print_stage_info(
                3,
                f"Comparing {difference_name}",
                f"{pair_info['clip_a']['type']} vs {pair_info['clip_b']['type']}",
            )

            llm_start = time.time()
            result = call_llm(
                self.model,
                self.processor,
                prompt,
                self.max_tokens,
                self.device,
                imgs=all_frames,
            )
            print_step_timing(
                f"LLM call for {difference_name}", time.time() - llm_start, 4
            )

            # Result parsing timing
            try:
                if isinstance(result, str):
                    pred_data = json.loads(result)
                else:
                    pred_data = result

                answer = pred_data.get("answer", "c").lower()
                confidence = pred_data.get("confidence", 0)
                explanation = pred_data.get("explanation", "No explanation provided")

                # Determine winner
                if answer == "a":
                    winner = "A"
                elif answer == "b":
                    winner = "B"
                else:
                    winner = "C"

            except Exception as e:
                winner = "C"
                confidence = 0
                explanation = f"Error parsing result: {str(e)}"

            result_dict = {
                "name": difference_name,
                "description": difference_description,
                "query_string": difference_description,
                "winner": winner,
                "winner_reason": explanation,
                "confidence": confidence,
                "error": None,
                "status": "success",
            }

            return result_dict

        except Exception as e:
            print(f"Error processing difference {difference_name}: {e}")
            total_time = time.time() - start_time
            print_step_timing(f"Error processing {difference_name}", total_time, 3)

            return {
                "name": difference_name,
                "description": difference_description,
                "query_string": difference_description,
                "winner": "C",
                "winner_reason": f"Error: {str(e)}",
                "confidence": 0,
                "error": str(e),
                "status": "error",
            }

    def save_results(self, results: Dict, output_json_path: str):
        """
        Save results to JSON file.
        """
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

    def run_stage_3_pipeline(self, input_json_path: str, output_json_path: str) -> Dict:
        """
        Run the complete Stage 3 pipeline with incremental saving.

        Args:
            input_json_path: Path to Stage 2 output JSON
            output_json_path: Path to save Stage 3 results

        Returns:
            Dictionary containing all results
        """
        pipeline_start = time.time()

        # Load data timing
        print("🔍 Loading Stage 2 results...")
        with open(input_json_path) as f:
            stage2_data = json.load(f)

        new_format_results = {}
        comparison_counter = 1

        action_times = {}
        total_comparisons = 0
        total_differences = 0

        # Process each action
        for action_name, action_data in stage2_data.items():
            action_start = time.time()
            print(f"\n🎬 Processing action: {action_name}")

            # Create pairs for this action
            pairs = self.create_comparison_pairs(action_data)
            print(f"📊 Created {len(pairs)} comparison pairs")

            action_comparisons = 0
            action_differences = 0

            # Process each pair
            for pair in tqdm(pairs, desc=f"Processing {action_name} pairs"):
                pair_start = time.time()

                # Create unique identifier
                unique_id = f"{action_name.replace(' ', '_')}_{pair['pair_id']}"

                # Create clips structure
                clips = {
                    "1": {
                        "url": pair["clip_a"]["url"],
                        "timestamp": pair["clip_a"]["timestamp"],
                        "type": pair["clip_a"]["type"],
                        "retrival_frames": pair["clip_a"]["retrieval_frames"],
                    },
                    "2": {
                        "url": pair["clip_b"]["url"],
                        "timestamp": pair["clip_b"]["timestamp"],
                        "type": pair["clip_b"]["type"],
                        "retrival_frames": pair["clip_b"]["retrieval_frames"],
                    },
                }

                # Get common differences between the two clips
                diff_a = set(pair["clip_a"]["retrieval_frames"].keys())
                diff_b = set(pair["clip_b"]["retrieval_frames"].keys())
                common_diffs = diff_a.intersection(diff_b)

                # Get difference descriptions from action_data
                proposed_diffs = action_data.get("proposed_differences", {})

                processed_differences = {}
                diff_counter = 0

                for diff_name in common_diffs:
                    # Get frames for this difference
                    frames_a_indices = pair["clip_a"]["retrieval_frames"][diff_name]
                    frames_b_indices = pair["clip_b"]["retrieval_frames"][diff_name]

                    # Load actual frames
                    frames_a = self.load_frames_from_video(
                        pair["clip_a"]["url"], frames_a_indices
                    )
                    frames_b = self.load_frames_from_video(
                        pair["clip_b"]["url"], frames_b_indices
                    )

                    if not frames_a or not frames_b:
                        print(f"⚠️ Skipping {diff_name}: Could not load frames")
                        continue

                    # Get difference description
                    diff_description = diff_name
                    for diff_id, diff_info in proposed_diffs.items():
                        if diff_info.get("name") == diff_name:
                            diff_description = diff_info.get("description", diff_name)
                            break

                    # Process this difference
                    comparison_result = self.process_single_difference(
                        action_name,
                        diff_name,
                        diff_description,
                        frames_a,
                        frames_b,
                        pair,
                    )

                    processed_differences[str(diff_counter)] = comparison_result
                    diff_counter += 1
                    action_differences += 1
                    total_differences += 1

                # Only create entry if we have differences
                if processed_differences:
                    new_format_results[
                        f"Action_class_comparison_number_{comparison_counter}"
                    ] = {
                        "unique_identifier": unique_id,
                        "action_class": action_name,
                        "clips": clips,
                        "proposed_differences": processed_differences,
                    }
                    comparison_counter += 1
                    action_comparisons += 1
                    total_comparisons += 1

                    # Save after each comparison pair is completed
                    self.save_results(new_format_results, output_json_path)

            # Record action timing
            action_time = time.time() - action_start
            action_times[action_name] = {
                "time": action_time,
                "comparisons": action_comparisons,
                "differences": action_differences,
            }
            print_step_timing(
                f"Complete action {action_name} ({action_comparisons} comparisons, {action_differences} differences)",
                action_time,
                1,
            )

        # Final save (redundant but ensures completion)
        self.save_results(new_format_results, output_json_path)

        # Print detailed timing summary
        total_pipeline_time = time.time() - pipeline_start
        print("\n" + "=" * 60)
        print("📊 STAGE 3 TIMING SUMMARY")
        print("=" * 60)

        print(f"⏱️  Total pipeline time: {format_time(total_pipeline_time)}")
        print(f"📈 Total comparisons: {total_comparisons}")
        print(f"📈 Total differences analyzed: {total_differences}")

        if total_differences > 0:
            avg_time_per_diff = total_pipeline_time / total_differences
            print(f"⚡ Average time per difference: {format_time(avg_time_per_diff)}")

        if total_comparisons > 0:
            avg_time_per_comparison = total_pipeline_time / total_comparisons
            print(
                f"⚡ Average time per comparison: {format_time(avg_time_per_comparison)}"
            )

        print("\n📋 Per-action breakdown:")
        for action_name, stats in action_times.items():
            print(
                f"  {action_name}: {format_time(stats['time'])} ({stats['comparisons']} comparisons, {stats['differences']} differences)"
            )
            if stats["differences"] > 0:
                avg_per_diff = stats["time"] / stats["differences"]
                print(f"    → {format_time(avg_per_diff)} per difference")

        print("\n✅ Stage 3 Complete!")
        print(f"📊 Created {len(new_format_results)} comparison entries")
        print(f"📊 Analyzed {total_differences} total differences")
        print("=" * 60)

        return new_format_results
