import json
import os
import glob
from typing import List, Dict, Any, Set
from pathlib import Path
from google.cloud import storage
from google.api_core import retry
from google.api_core.exceptions import (
    TooManyRequests,
    ServiceUnavailable,
    GoogleAPIError,
)
from prompt import PROMPT_DIFFERENCE_MULTIFRAME, SYSTEM_PROMPT


class GCSFrameUtils:
    def __init__(
        self,
        bucket_name: str = "biryanidiff",
        credentials_path: str = "biryani-across-india-d26285fcddf7.json",
    ):
        """Initialize GCS client and load existing files."""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

        # Load all existing files in bucket at startup
        print("Loading existing files from bucket...")
        self.existing_files: Set[str] = self._load_existing_files()
        print(f"Found {len(self.existing_files)} existing files in bucket")

    def _load_existing_files(self) -> Set[str]:
        """Load all existing blob names from the bucket."""
        try:
            blobs = self.bucket.list_blobs()
            return {blob.name for blob in blobs}
        except Exception as e:
            print(f"Warning: Could not load existing files: {e}")
            return set()

    def format_frame_name(self, video_url: str, frame_number: int) -> str:
        """Generate frame name from video URL and frame number."""
        video_filename = Path(video_url).stem
        return f"{video_filename}_frame_{frame_number:06d}.png"

    def get_frame_uri(self, frame_name: str) -> str:
        """Get GCS URI for frame in bucket."""
        return f"gs://{self.bucket_name}/{frame_name}"

    def check_frame_exists(self, frame_name: str) -> bool:
        """Check if frame exists in bucket using cached list."""
        return frame_name in self.existing_files

    @retry.Retry(
        predicate=retry.if_exception_type(TooManyRequests, ServiceUnavailable, GoogleAPIError),
        initial=2.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0,
    )
    def upload_frame(self, local_path: str, frame_name: str = None) -> bool:
        """Upload frame to bucket with retry logic."""
        if not os.path.exists(local_path):
            return False

        if frame_name is None:
            frame_name = Path(local_path).name

        try:
            blob = self.bucket.blob(frame_name)
            blob.upload_from_filename(local_path)
            # Verify upload
            blob.reload()
            local_size = os.path.getsize(local_path)
            success = blob.size == local_size

            if success:
                # Update our cached list
                self.existing_files.add(frame_name)

            return success
        except (TooManyRequests, ServiceUnavailable):
            raise  # Let retry handle this
        except Exception as e:
            print(f"Upload failed for {frame_name}: {e}")
            return False

    def upload_if_missing(self, local_path: str, frame_name: str = None) -> tuple:
        """Upload frame only if it doesn't exist in bucket."""
        if frame_name is None:
            frame_name = Path(local_path).name

        if self.check_frame_exists(frame_name):
            return True, "already_exists"

        success = self.upload_frame(local_path, frame_name)
        return success, "uploaded" if success else "failed"


def find_important_frames(
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
                    clip1_positions = convert_to_upload_positions(clip1_frames, clip1_data, True)
                    clip2_positions = convert_to_upload_positions(
                        clip2_frames, clip2_data, False, clip1_frame_count
                    )
                    important_context += (
                        f"\nFrames {clip1_positions} (Video A) and "
                        f"{clip2_positions} (Video B) might be important for this "
                        f"difference as they show: {description}"
                    )
    return important_context


def convert_to_upload_positions(
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


def get_unique_frame_count(clip_data: Dict) -> int:
    """Get the number of unique frames for a clip."""
    if not isinstance(clip_data, dict):
        return 0

    all_frames = []
    retrieval_frames = clip_data.get("retrieval_frames", {})
    for stage_frames in retrieval_frames.values():
        all_frames.extend(stage_frames)

    return len(set(all_frames))


def get_frame_uris_for_clips(clips: Dict, gcs_utils: GCSFrameUtils) -> List[Dict]:
    """Get frame URIs for all clips in the comparison."""
    frame_parts = []

    for clip_id in ["1", "2"]:
        clip_data = clips.get(clip_id, {})
        if not isinstance(clip_data, dict):
            continue

        video_url = clip_data.get("url", "")
        if not video_url:
            continue

        # Get all unique frames for this clip
        all_frames = []
        retrieval_frames = clip_data.get("retrieval_frames", {})
        for stage_frames in retrieval_frames.values():
            all_frames.extend(stage_frames)

        unique_frames = sorted(set(all_frames))

        # Create frame parts with URIs
        for frame_num in unique_frames:
            frame_name = gcs_utils.format_frame_name(video_url, frame_num)
            frame_uri = gcs_utils.get_frame_uri(frame_name)

            frame_parts.append({"file_data": {"file_uri": frame_uri, "mime_type": "image/png"}})

    return frame_parts


def create_gemini_request(
    comparison_data: Dict[str, Any], comparison_key: str, gcs_utils: GCSFrameUtils
) -> List[Dict[str, Any]]:
    """Create Gemini API requests from comparison data with file URIs."""
    try:
        action_class = comparison_data.get("action_class", "")
        clips = comparison_data.get("clips", {})
        proposed_differences = comparison_data.get("proposed_differences", {})
        action_stages = comparison_data.get("action_stages", [])
        unique_identifier = comparison_data.get("unique_identifier", "")

        # Calculate frame counts for each clip
        clip1_data = clips.get("1", {})
        clip2_data = clips.get("2", {})
        clip1_frame_count = get_unique_frame_count(clip1_data)
        clip2_frame_count = get_unique_frame_count(clip2_data)

        # Get frame URIs for all clips
        frame_parts = get_frame_uris_for_clips(clips, gcs_utils)

        requests = []
        for diff_id, difference in proposed_differences.items():
            if not isinstance(difference, dict):
                continue

            diff_name = difference.get("name", "")
            query_string = difference.get("query_string", "")

            if not diff_name or not query_string:
                continue

            importance_context = find_important_frames(
                action_stages, diff_name, clips, clip1_frame_count, clip2_frame_count
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

            # Create parts list with text and all frame URIs
            parts = [{"text": formatted_prompt}] + frame_parts

            request = {
                "contents": [{"parts": parts, "role": "user"}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
            }

            unique_key = f"{comparison_key}_{diff_id}_{unique_identifier}"
            requests.append({"key": unique_key, "request": request})

        return requests
    except Exception as e:
        print(f"Error processing {comparison_key}: {e}")
        return []


def process_json_file(input_file: str, output_file: str, gcs_utils: GCSFrameUtils) -> None:
    """Process single JSON file and create JSONL output with file URIs."""
    all_requests = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for comparison_key, comparison_data in data.items():
            if comparison_key.startswith("Action_class_comparison"):
                if isinstance(comparison_data, dict):
                    requests = create_gemini_request(comparison_data, comparison_key, gcs_utils)
                    all_requests.extend(requests)
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for request in all_requests:
            f.write(json.dumps(request) + "\n")

    print(f"Created {len(all_requests)} requests in {output_file}")


def process_all_batch_files(input_pattern: str, output_dir: str, gcs_utils: GCSFrameUtils) -> None:
    """Process all batch files matching the pattern."""
    os.makedirs(output_dir, exist_ok=True)
    batch_files = glob.glob(input_pattern)

    if not batch_files:
        print(f"No files found matching pattern: {input_pattern}")
        return

    for batch_file in batch_files:
        basename = os.path.basename(batch_file)
        batch_name = basename.replace(".json", "")
        output_file = os.path.join(output_dir, f"{batch_name}.jsonl")
        process_json_file(batch_file, output_file, gcs_utils)


def main():
    """Main execution function."""
    # Initialize GCS utilities
    gcs_utils = GCSFrameUtils(
        bucket_name="biryanidiff",
        credentials_path="biryani-across-india-d26285fcddf7.json",
    )

    # Process single file
    INPUT_FILE = "../../Data/BatchPipline/Comparison_batch_batch_1.json"
    OUTPUT_FILE = "biryani_comparisons_batch_1.jsonl"
    process_json_file(INPUT_FILE, OUTPUT_FILE, gcs_utils)

    # Uncomment to process all batch files
    # INPUT_PATTERN = "../../Data/BatchPipline/Comparison_batch_batch_*.json"
    # OUTPUT_DIR = "../../Data/BatchPipline/JSONL_Output"
    # process_all_batch_files(INPUT_PATTERN, OUTPUT_DIR, gcs_utils)


if __name__ == "__main__":
    main()
