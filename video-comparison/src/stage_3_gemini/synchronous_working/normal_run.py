import json
import os
import glob
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from google import genai
from dotenv import load_dotenv
import time
from google.genai import types

from prompt import PROMPT_DIFFERENCE_MULTIFRAME, SYSTEM_PROMPT


class GeminiFileManager:
    def __init__(self, max_storage_gb: float = 18.0):
        """Initialize Gemini client with file management."""
        load_dotenv()

        self.client = genai.Client()
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        self.uploaded_files = {}
        self.current_storage = 0

        print("Loading existing files from Gemini Files API...")
        self._load_existing_files()

    def _load_existing_files(self):
        """Load existing files from Files API."""
        try:
            files = self.client.files.list()
            for file in files:
                self.uploaded_files[file.name] = file
                # Estimate storage (Files API doesn't provide size directly)
                if hasattr(file, "size_bytes"):
                    self.current_storage += file.size_bytes
            print(f"Found {len(self.uploaded_files)} existing files")
        except Exception as e:
            print(f"Warning: Could not load existing files: {e}")

    def format_frame_name(self, video_url: str, frame_number: int) -> str:
        """Generate frame name from video URL and frame number."""
        video_filename = Path(video_url).stem
        return f"{video_filename}_frame_{frame_number:06d}.png"

    def _cleanup_old_files(self, bytes_needed: int):
        """Delete oldest files to make space."""
        print(f"Need to free up {bytes_needed / (1024 * 1024):.1f} MB...")

        # Sort files by creation time (oldest first)
        files_list = list(self.uploaded_files.items())
        # Since we can't get creation time easily, remove first 25% of files
        files_to_remove = files_list[: len(files_list) // 4]

        for filename, file_obj in files_to_remove:
            try:
                self.client.files.delete(name=file_obj.name)
                del self.uploaded_files[filename]
                print(f"Deleted old file: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

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
            self.uploaded_files[frame_name] = uploaded_file
            self.current_storage += file_size
            return True, "uploaded"
        except Exception as e:
            print(f"Upload failed for {frame_name}: {e}")
            return False, "failed"

    def get_file_by_name(self, frame_name: str):
        """Get uploaded file object by name."""
        return self.uploaded_files.get(frame_name)


class BiryaniComparisonProcessor:
    def __init__(self, local_images_path: str, gemini_manager: GeminiFileManager):
        self.local_images_path = Path(local_images_path)
        self.gemini_manager = gemini_manager

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
                    print(f"✓ {frame_name} ({status})")
                else:
                    print(f"✗ Could not retrieve {frame_name}")
            else:
                print(f"✗ Failed to upload {frame_name}: {status}")

        return uploaded_files

    def process_comparison(self, comparison_key: str, comparison_data: Dict) -> List[Dict]:
        """Process a single comparison and return results."""
        results = []

        try:
            action_class = comparison_data.get("action_class", "")
            clips = comparison_data.get("clips", {})
            proposed_differences = comparison_data.get("proposed_differences", {})
            action_stages = comparison_data.get("action_stages", [])
            unique_identifier = comparison_data.get("unique_identifier", "")

            print(f"\nProcessing {comparison_key} ({action_class})")

            # Upload frames for both clips
            clip1_files = self._upload_clip_frames(clips.get("1", {}))
            clip2_files = self._upload_clip_frames(clips.get("2", {}))

            if not clip1_files or not clip2_files:
                print(f"⚠️  Missing frames for {comparison_key}")
                return results

            all_files = clip1_files + clip2_files
            clip1_frame_count = len(clip1_files)
            clip2_frame_count = len(clip2_files)

            print(f"Uploaded {clip1_frame_count} + {clip2_frame_count} frames")

            # Process each difference
            for diff_id, difference in proposed_differences.items():
                if not isinstance(difference, dict):
                    continue

                diff_name = difference.get("name", "")
                query_string = difference.get("query_string", "")

                if not diff_name or not query_string:
                    continue

                print(f"  Analyzing difference: {diff_name}")

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
                try:
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

                    # Create result entry
                    result = {
                        "comparison_key": comparison_key,
                        "unique_identifier": unique_identifier,
                        "action_class": action_class,
                        "difference_id": diff_id,
                        "difference_name": diff_name,
                        "query_string": query_string,
                        "clips": clips,
                        "analysis": analysis_result,
                        "frame_counts": {
                            "clip1": clip1_frame_count,
                            "clip2": clip2_frame_count,
                        },
                    }

                    results.append(result)
                    print(f"    Result: {analysis_result.get('answer', 'unknown')}")

                    # Small delay to avoid rate limits
                    time.sleep(1)

                except Exception as e:
                    print(f"    Error analyzing {diff_name}: {e}")
                    error_result = {
                        "comparison_key": comparison_key,
                        "unique_identifier": unique_identifier,
                        "difference_id": diff_id,
                        "difference_name": diff_name,
                        "error": str(e),
                    }
                    results.append(error_result)

        except Exception as e:
            print(f"Error processing {comparison_key}: {e}")

        return results

    def process_json_file(self, input_file: str, output_file: str):
        """Process entire JSON file and save results."""
        all_results = []

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"Processing {len(data)} comparisons from {input_file}")

            for comparison_key, comparison_data in data.items():
                if comparison_key.startswith("Action_class_comparison"):
                    if isinstance(comparison_data, dict):
                        results = self.process_comparison(comparison_key, comparison_data)
                        all_results.extend(results)

            # Save results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_results": len(all_results),
                        "processed_file": input_file,
                        "results": all_results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"\n✅ Saved {len(all_results)} results to {output_file}")

        except Exception as e:
            print(f"Error processing {input_file}: {e}")


def main():
    """Main execution function."""
    # Configuration
    LOCAL_IMAGES_PATH = "" # dir containign extracted frames

    gemini_manager = GeminiFileManager()
    processor = BiryaniComparisonProcessor(LOCAL_IMAGES_PATH, gemini_manager)

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
