import json
import os
from typing import Dict, List, Set, Tuple
import cv2
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
import hashlib

from stage_3.prompts import PROMPT_DIFFERENCE_1FRAME, PROMPT_DIFFERENCE_MULTIFRAME
from stage_3.utils import clean_json_response


class GeminiDifferencer:
    """
    Stage 3: Action Differencer using Gemini.
    Given localized frames for each difference, it calls Gemini to decide:
    (a) Video A, (b) Video B, (c) Similar, or (d) Irrelevant.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        self.model_name = model_name
        load_dotenv()
        self.client = genai.Client()
        self.uploaded_files = {}  # Maps frame_key -> uploaded_file
        self.frame_cache = {}  # Maps (video_path, frame_idx) -> PIL Image
        self.temp_files = []  # Track all temp files for cleanup
        print(f"🚀 Gemini Differencer initialized with model: {model_name}")

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

    def call_gemini(self, prompt: str, uploaded_files: List) -> Dict:
        """Call Gemini API with prompt and images."""
        try:
            contents = [prompt] + uploaded_files

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )

            # Use utils function to clean JSON response
            return clean_json_response(response.text)

        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            print(f"📝 Raw response (if available): {getattr(response, 'text', 'No response')}")
            return {
                "answer": "c",
                "confidence": 0,
                "difference_visible": False,
                "explanation": f"Error: {str(e)}",
            }

    def process_single_comparison(self, comparison_data: Dict) -> Dict:
        """Process a single comparison between two clips."""
        action_class = comparison_data["action_class"]
        proposed_differences = comparison_data["proposed_differences"]
        action_stages = comparison_data.get("action_stages", [])

        # Get uploaded frame references for this comparison
        uploaded_a, uploaded_b = self.get_frames_for_comparison(comparison_data)
        all_uploaded = uploaded_a + uploaded_b

        if not uploaded_a or not uploaded_b:
            print("⚠️ Skipping comparison: No uploaded frames available")
            return comparison_data

        # Process each proposed difference
        results = {}

        for diff_id, diff_data in proposed_differences.items():
            difference_name = diff_data["name"]
            query_string = diff_data["query_string"]

            print(f"🔍 Processing difference: {difference_name}")

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

            # Call Gemini
            result = self.call_gemini(prompt, all_uploaded)

            # Map answer to winner
            answer = result.get("answer", "c").lower()
            winner_map = {"a": "A", "b": "B", "c": "C", "d": "D"}
            winner = winner_map.get(answer, "C")

            # Store result
            results[diff_id] = {
                "name": difference_name,
                "query_string": query_string,
                "winner": winner,
                "confidence": result.get("confidence", 0),
                "difference_visible": result.get("difference_visible", False),
                "explanation": result.get("explanation", "No explanation provided"),
                "status": "success",
            }

        # Update comparison data with results
        comparison_data["proposed_differences"] = results
        return comparison_data

    def cleanup_temp_files(self):
        """Clean up all temporary image files."""
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)
        self.temp_files.clear()

    def run_pipeline(self, input_json_path: str, output_json_path: str) -> Dict:
        """Run the complete Gemini differencer pipeline."""
        print("🔍 Loading input data...")

        with open(input_json_path, "r") as f:
            input_data = json.load(f)

        all_frames = self.collect_all_frames(input_data)
        print(f"📊 Found {len(all_frames)} unique frames across all comparisons")

        self.upload_all_frames(all_frames)

        results = {}

        for comparison_id, comparison_data in tqdm(
            input_data.items(), desc="Processing comparisons"
        ):
            print(f"\n🎬 Processing: {comparison_id}")

            try:
                processed_comparison = self.process_single_comparison(comparison_data)
                results[comparison_id] = processed_comparison

                # Save progress after each comparison
                with open(output_json_path, "w") as f:
                    json.dump(results, f, indent=2)

                print(f"💾 Saved progress to {output_json_path}")

            except Exception as e:
                print(f"❌ Error processing {comparison_id}: {e}")
                results[comparison_id] = comparison_data

        self.cleanup_temp_files()

        print(f"\n✅ Pipeline complete! Results saved to {output_json_path}")
        return results
