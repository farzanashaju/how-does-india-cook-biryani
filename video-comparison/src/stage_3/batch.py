import json
import os
import time
import hashlib
from typing import Dict, List, Set, Tuple
import cv2
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

from stage_3.prompts import PROMPT_DIFFERENCE_1FRAME, PROMPT_DIFFERENCE_MULTIFRAME
from stage_3.utils import clean_json_response


class GeminiDifferencerBatch:
    """
    Stage 3: Action Differencer using Gemini Batch API.
    Creates batch jobs for efficient processing of large datasets.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        self.model_name = model_name
        load_dotenv()
        self.client = genai.Client()
        self.uploaded_files = {}  # Maps frame_key -> uploaded_file
        self.frame_cache = {}  # Maps (video_path, frame_idx) -> PIL Image
        self.temp_files = []  # Track all temp files for cleanup
        self.max_requests_per_batch = 280  # Gemini batch limit
        print(f"🚀 Gemini Batch Differencer initialized with model: {model_name}")

    def get_frame_key(self, video_path: str, frame_idx: int) -> str:
        """Generate unique key for a frame."""
        video_name = os.path.basename(video_path).replace(".mp4", "")
        return f"{video_name}_frame_{frame_idx:04d}"

    def extract_single_frame(self, video_path: str, frame_idx: int) -> Image.Image:
        """Extract a single frame from video and return PIL Image."""
        cache_key = (video_path, frame_idx)
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                self.frame_cache[cache_key] = pil_frame
                return pil_frame
            else:
                print(f"⚠️ Warning: Could not read frame {frame_idx} from {video_path}")
                return None

        except Exception as e:
            print(f"❌ Error extracting frame {frame_idx} from {video_path}: {e}")
            return None
        finally:
            cap.release()

    def collect_all_frames(self, input_data: Dict) -> Set[Tuple[str, int]]:
        """Collect all unique (video_path, frame_idx) pairs from input data."""
        all_frames = set()

        for comparison_data in input_data.values():
            clips = comparison_data["clips"]

            for clip_data in clips.values():
                video_path = clip_data["url"]
                retrieval_frames = clip_data["retrieval_frames"]

                # Collect all frame indices for this video
                for frame_list in retrieval_frames.values():
                    for frame_idx in frame_list:
                        all_frames.add((video_path, frame_idx))

        return all_frames

    def upload_all_frames(self, all_frames: Set[Tuple[str, int]]) -> Dict[str, any]:
        """Upload all frames to Gemini Files API at startup."""
        print(f"📤 Uploading {len(all_frames)} unique frames to Gemini...")

        for video_path, frame_idx in tqdm(all_frames, desc="Uploading frames"):
            frame_key = self.get_frame_key(video_path, frame_idx)

            # Skip if already uploaded
            if frame_key in self.uploaded_files:
                continue

            # Extract frame
            frame = self.extract_single_frame(video_path, frame_idx)
            if frame is None:
                continue

            # Save frame locally with unique name
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
            filename = f"{frame_key}_{frame_hash}.jpg"
            frame.save(filename, "JPEG", quality=100)
            self.temp_files.append(filename)

            # Upload to Gemini
            try:
                uploaded_file = self.client.files.upload(file=filename)
                self.uploaded_files[frame_key] = uploaded_file
                print(f"✅ Uploaded {frame_key} -> {uploaded_file.name}")
            except Exception as e:
                print(f"❌ Error uploading {frame_key}: {e}")

        print(f"🎉 Successfully uploaded {len(self.uploaded_files)} frames")
        return self.uploaded_files

    def get_frames_for_comparison(self, comparison_data: Dict) -> Tuple[List[any], List[any]]:
        """Get uploaded file references for a comparison."""
        clips = comparison_data["clips"]

        # Get all frame keys for clip A and B
        clip_a = clips["1"]
        clip_b = clips["2"]

        # Collect all unique frame indices for each clip
        frames_a_keys = set()
        frames_b_keys = set()

        for stage_frames in clip_a["retrieval_frames"].values():
            for frame_idx in stage_frames:
                frame_key = self.get_frame_key(clip_a["url"], frame_idx)
                frames_a_keys.add(frame_key)

        for stage_frames in clip_b["retrieval_frames"].values():
            for frame_idx in stage_frames:
                frame_key = self.get_frame_key(clip_b["url"], frame_idx)
                frames_b_keys.add(frame_key)

        # Get uploaded file references
        uploaded_a = []
        uploaded_b = []

        for frame_key in sorted(frames_a_keys):
            if frame_key in self.uploaded_files:
                uploaded_a.append(self.uploaded_files[frame_key])

        for frame_key in sorted(frames_b_keys):
            if frame_key in self.uploaded_files:
                uploaded_b.append(self.uploaded_files[frame_key])

        return uploaded_a, uploaded_b

    def build_importance_context(self, difference_name: str, action_stages: List[Dict]) -> str:
        """Build importance context from action stages."""
        relevant_stages = []

        for stage in action_stages:
            if difference_name in stage.get("associated_differences", []):
                relevant_stages.append(f"- {stage['name']}: {stage['description']}")

        if relevant_stages:
            return (
                "\n**Important Context:** This difference is particularly relevant in these stages:\n"
                + "\n".join(relevant_stages)
            )
        else:
            return ""

    def create_prompt(
        self,
        action: str,
        query_string: str,
        importance_context: str,
        num_frames_a: int,
        num_frames_b: int,
        time_diff: float = 0.5,
    ) -> str:
        """Create prompt based on number of frames."""

        if num_frames_a == 1 and num_frames_b == 1:
            return PROMPT_DIFFERENCE_1FRAME.format(
                action=action,
                query_string=query_string,
                importance_context=importance_context,
            )
        else:
            num_frames_total = num_frames_a + num_frames_b
            num_frames_b_start = num_frames_a + 1

            return PROMPT_DIFFERENCE_MULTIFRAME.format(
                action=action,
                num_frames_a=num_frames_a,
                num_frames_b_start=num_frames_b_start,
                num_frames_total=num_frames_total,
                time_diff=time_diff,
                query_string=query_string,
                importance_context=importance_context,
            )

    def create_batch_requests(self, input_data: Dict) -> List[Dict]:
        """Create batch requests for all comparisons and differences."""
        batch_requests = []

        for comparison_id, comparison_data in input_data.items():
            action_class = comparison_data["action_class"]
            proposed_differences = comparison_data["proposed_differences"]
            action_stages = comparison_data.get("action_stages", [])

            # Get uploaded frame references for this comparison
            uploaded_a, uploaded_b = self.get_frames_for_comparison(comparison_data)
            all_uploaded = uploaded_a + uploaded_b

            if not uploaded_a or not uploaded_b:
                print(f"⚠️ Skipping comparison {comparison_id}: No uploaded frames available")
                continue

            # Create request for each proposed difference
            for diff_id, diff_data in proposed_differences.items():
                difference_name = diff_data["name"]
                query_string = diff_data["query_string"]

                # Build importance context
                importance_context = self.build_importance_context(difference_name, action_stages)

                # Create prompt
                prompt = self.create_prompt(
                    action_class,
                    query_string,
                    importance_context,
                    len(uploaded_a),
                    len(uploaded_b),
                )

                # Create unique key for this request
                request_key = f"{comparison_id}_{diff_id}"

                # Create contents with text prompt
                contents = [{"parts": [{"text": prompt}]}]

                # Add uploaded files to contents
                for uploaded_file in all_uploaded:
                    contents.append(
                        {
                            "parts": [
                                {
                                    "file_data": {
                                        "file_name": uploaded_file.name,
                                        "mime_type": "image/jpeg",
                                    }
                                }
                            ]
                        }
                    )

                # Create batch request with correct structure for JSONL
                batch_request = {
                    "key": request_key,
                    "request": {
                        "contents": contents,
                        "generation_config": {"temperature": 0.7},
                    },
                }

                batch_requests.append(batch_request)

        return batch_requests

    def split_into_batches(self, batch_requests: List[Dict]) -> List[List[Dict]]:
        """Split batch requests into chunks of max_requests_per_batch."""
        batches = []
        for i in range(0, len(batch_requests), self.max_requests_per_batch):
            batch = batch_requests[i : i + self.max_requests_per_batch]
            batches.append(batch)
        return batches

    def create_jsonl_file(self, batch_requests: List[Dict], batch_id: int, output_dir: str) -> str:
        """Create JSONL file for batch requests."""
        os.makedirs(output_dir, exist_ok=True)
        jsonl_path = os.path.join(output_dir, f"batch_{batch_id:03d}.jsonl")

        with open(jsonl_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")

        return jsonl_path

    def submit_batch_job(self, jsonl_path: str, batch_id: int) -> str:
        """Submit a batch job to Gemini."""
        try:
            # Upload JSONL file
            uploaded_file = self.client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(
                    display_name=f"batch-requests-{batch_id}",
                    mime_type="application/jsonl",
                ),
            )

            # Create batch job
            batch_job = self.client.batches.create(
                model=self.model_name,
                src=uploaded_file.name,
                config={
                    "display_name": f"differencer-batch-{batch_id}",
                },
            )

            print(f"✅ Created batch job {batch_id}: {batch_job.name}")
            return batch_job.name

        except Exception as e:
            print(f"❌ Error creating batch job {batch_id}: {e}")
            return None

    def save_batch_job_names(self, batch_job_names: List[str], output_file: str):
        """Save batch job names to file."""
        job_info = {
            "batch_jobs": [
                {"id": i, "name": name, "status": "PENDING", "created_at": time.time()}
                for i, name in enumerate(batch_job_names)
                if name
            ],
            "total_jobs": len([name for name in batch_job_names if name]),
            "model": self.model_name,
        }

        with open(output_file, "w") as f:
            json.dump(job_info, f, indent=2)

        print(f"💾 Batch job names saved to: {output_file}")

    def cleanup_temp_files(self):
        """Clean up all temporary image files."""
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)
        self.temp_files.clear()

    def run_batch_pipeline(self, input_json_path: str, output_dir: str = "batch_output") -> Dict:
        """Run the complete Gemini batch differencer pipeline."""
        print("🔍 Loading input data...")

        with open(input_json_path, "r") as f:
            input_data = json.load(f)

        print(f"📊 Found {len(input_data)} comparisons")

        # Collect and upload all frames
        all_frames = self.collect_all_frames(input_data)
        print(f"📊 Found {len(all_frames)} unique frames across all comparisons")
        self.upload_all_frames(all_frames)

        # Create batch requests
        print("🏗️ Creating batch requests...")
        batch_requests = self.create_batch_requests(input_data)
        print(f"📊 Created {len(batch_requests)} batch requests")

        # Split into batches
        batches = self.split_into_batches(batch_requests)
        print(
            f"📊 Split into {len(batches)} batches (max {self.max_requests_per_batch} requests each)"
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create JSONL files and submit batch jobs
        batch_job_names = []

        for batch_id, batch in enumerate(batches):
            print(f"📝 Creating batch {batch_id + 1}/{len(batches)} with {len(batch)} requests...")

            # Create JSONL file
            jsonl_path = self.create_jsonl_file(batch, batch_id, output_dir)
            print(f"💾 Saved JSONL to: {jsonl_path}")

            # Submit batch job
            job_name = self.submit_batch_job(jsonl_path, batch_id)
            batch_job_names.append(job_name)

            # Small delay between submissions
            time.sleep(1)

        # Save batch job names
        batch_jobs_file = os.path.join(output_dir, "batch_jobs.json")
        self.save_batch_job_names(batch_job_names, batch_jobs_file)

        # Cleanup temp files
        self.cleanup_temp_files()

        print(
            f"\n✅ Pipeline complete! {len([n for n in batch_job_names if n])} batch jobs submitted"
        )
        print(f"📂 Output directory: {output_dir}")
        print(f"📂 Batch jobs file: {batch_jobs_file}")

        return {
            "batch_jobs": batch_job_names,
            "output_dir": output_dir,
            "total_requests": len(batch_requests),
            "total_batches": len(batches),
        }
