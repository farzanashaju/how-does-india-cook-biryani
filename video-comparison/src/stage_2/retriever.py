import copy
import json
import logging
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image


class FrameRetriever:
    """
    Optimized Stage 2: Frame Retrieval using CLIP for action analysis.
    """

    def __init__(
        self,
        model_name: str = "ViT-bigG-14",
        dataset: str = "laion2b_s39b_b160k",
        batch_size: int = 256,  # Increased batch size
        top_k: int = 1,
        target_fps: int = 4,
        device: Optional[str] = None,
        num_workers: int = 4,  # For video loading
        save_interval: int = 20,  # Save progress every N clips
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.top_k = top_k
        self.target_fps = target_fps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.save_interval = save_interval

        # Suppress logging for clean progress bar
        logging.getLogger().setLevel(logging.ERROR)

        print(f"Loading CLIP model: {self.model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.dataset, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        print("Model loaded successfully")

        # Cache for text embeddings
        self.text_embedding_cache = {}

    def _load_video_frames_batch(self, video_paths: List[str]) -> Dict[str, List[Image.Image]]:
        """Load frames from multiple videos in parallel."""

        def load_single_video(video_path):
            try:
                return video_path, self._load_video_frames(video_path)
            except:
                return video_path, []

        results = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(load_single_video, path) for path in video_paths]
            for future in futures:
                path, frames = future.result()
                results[path] = frames

        return results

    def _load_video_frames(self, video_path: str) -> List[Image.Image]:
        """Load frames from pre-trimmed video clips."""
        if not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30

        frame_skip = max(1, int(round(original_fps / self.target_fps)))
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

            frame_idx += 1

        cap.release()
        return frames

    def _get_text_embeddings_cached(self, texts: List[str]) -> np.ndarray:
        """Generate CLIP text embeddings with caching."""
        if not texts:
            return np.array([])

        # Check cache first
        cache_key = tuple(sorted(texts))
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]

        # Generate embeddings
        embeddings = self._get_text_embeddings(texts)

        # Cache result
        self.text_embedding_cache[cache_key] = embeddings
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate CLIP text embeddings with batch processing."""
        if not texts:
            return np.array([])

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                text_tokens = self.tokenizer(batch_texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features.cpu().numpy())

        return np.vstack(all_embeddings)

    def _get_image_embeddings_batch(self, all_images: List[List[Image.Image]]) -> List[np.ndarray]:
        """Generate CLIP image embeddings for multiple clips in large batches."""
        if not all_images:
            return []

        # Flatten all images with clip tracking
        flattened_images = []
        clip_boundaries = [0]

        for clip_images in all_images:
            flattened_images.extend(clip_images)
            clip_boundaries.append(len(flattened_images))

        if not flattened_images:
            return [np.array([]) for _ in all_images]

        # Process all images in large batches
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(flattened_images), self.batch_size):
                batch_images = flattened_images[i : i + self.batch_size]
                image_tensors = torch.stack([self.preprocess(img) for img in batch_images]).to(
                    self.device
                )
                image_features = self.model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_features.cpu().numpy())

        if not all_embeddings:
            return [np.array([]) for _ in all_images]

        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)

        # Split back by clip
        clip_embeddings = []
        for i in range(len(clip_boundaries) - 1):
            start_idx = clip_boundaries[i]
            end_idx = clip_boundaries[i + 1]
            if start_idx < end_idx:
                clip_embeddings.append(combined_embeddings[start_idx:end_idx])
            else:
                clip_embeddings.append(np.array([]))

        return clip_embeddings

    def _cosine_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between two sets of embeddings."""
        return np.dot(embeddings1, embeddings2.T)

    def _find_best_frames_batch(
        self, frame_embeddings_list: List[np.ndarray], text_embeddings: np.ndarray
    ) -> List[List[int]]:
        """Find best frames for multiple clips at once."""
        results = []

        for frame_embeddings in frame_embeddings_list:
            if frame_embeddings.size == 0 or text_embeddings.size == 0:
                results.append([])
                continue

            similarities = self._cosine_similarity(frame_embeddings, text_embeddings)
            max_similarities = np.max(similarities, axis=1)
            top_indices = np.argsort(max_similarities)[-self.top_k :][::-1]
            results.append(top_indices.tolist())

        return results

    def _process_action_batch(
        self, clips_data: Dict, action_stages: List[Dict], clip_ids: List[str]
    ) -> Dict[str, Dict]:
        """Process a batch of clips for an action class."""
        # Load all video frames in parallel
        video_paths = [clips_data[clip_id]["url"] for clip_id in clip_ids]
        frames_dict = self._load_video_frames_batch(video_paths)

        # Get frames in order
        all_clip_frames = [frames_dict.get(path, []) for path in video_paths]

        # Get image embeddings for all clips at once
        clip_embeddings_list = self._get_image_embeddings_batch(all_clip_frames)

        # Process each stage
        results = {}
        for clip_id in clip_ids:
            results[clip_id] = {"retrieval_frames": {}}

        for stage in action_stages:
            stage_name = stage["name"]
            retrieval_strings = stage.get("retrieval_strings", [])

            if not retrieval_strings:
                continue

            # Get cached text embeddings
            text_embeddings = self._get_text_embeddings_cached(retrieval_strings)

            # Find best frames for all clips at once
            best_frames_list = self._find_best_frames_batch(clip_embeddings_list, text_embeddings)

            # Store results
            for i, clip_id in enumerate(clip_ids):
                results[clip_id]["retrieval_frames"][stage_name] = best_frames_list[i]

        return results

    def _save_progress(self, data: Dict, output_file: str) -> None:
        """Save current progress to output file."""
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_existing_progress(self, output_file: str) -> Optional[Dict]:
        """Load existing progress if output file exists."""
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def _get_processed_clips(self, action_data: Dict) -> set:
        """Get set of already processed clip IDs."""
        processed = set()
        clips_data = action_data.get("Clips", {})

        for clip_id, clip_info in clips_data.items():
            if "retrieval_frames" in clip_info and clip_info["retrieval_frames"]:
                processed.add(clip_id)

        return processed

    def run_stage_2_pipeline(self, input_file: str, output_file: str) -> Dict:
        """
        Run the optimized Stage 2 pipeline with batching.
        """
        print("Starting Optimized Stage 2: Frame Localization and Retrieval")

        with open(input_file, "r") as f:
            stage_1_data = json.load(f)

        # Check for existing progress
        existing_data = self._load_existing_progress(output_file)
        if existing_data:
            stage_2_data = existing_data
            print("Resuming from existing progress...")
        else:
            stage_2_data = copy.deepcopy(stage_1_data)

        # Count total clips and processed clips
        total_clips = 0
        total_processed = 0

        for action_class, action_data in stage_2_data.items():
            clips_data = action_data.get("Clips", {})
            processed_clips = self._get_processed_clips(action_data)

            total_clips += len(clips_data)
            total_processed += len(processed_clips)

        remaining_clips = total_clips - total_processed

        if remaining_clips == 0:
            print("All clips already processed!")
            return stage_2_data

        print(f"Processing {remaining_clips}/{total_clips} remaining clips...")

        # Process in batches
        processed_count = 0

        with tqdm(
            total=remaining_clips,
            desc="Processing clips",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        ) as pbar:
            for action_idx, (action_class, action_data) in enumerate(stage_2_data.items(), 1):
                clips_data = action_data.get("Clips", {})
                action_stages = action_data.get("action_stages", [])

                if not clips_data or not action_stages:
                    continue

                processed_clips = self._get_processed_clips(action_data)
                unprocessed_clips = [
                    clip_id for clip_id in clips_data.keys() if clip_id not in processed_clips
                ]

                if not unprocessed_clips:
                    continue

                pbar.set_description(f"Action {action_idx}/{len(stage_2_data)}: {action_class}")

                # Process clips in batches
                batch_size = min(32, len(unprocessed_clips))  # Adjust batch size as needed

                for i in range(0, len(unprocessed_clips), batch_size):
                    batch_clip_ids = unprocessed_clips[i : i + batch_size]

                    try:
                        batch_results = self._process_action_batch(
                            clips_data, action_stages, batch_clip_ids
                        )

                        # Update clips_data with results
                        for clip_id, result in batch_results.items():
                            clips_data[clip_id].update(result)

                        processed_count += len(batch_clip_ids)
                        pbar.update(len(batch_clip_ids))

                        # Save progress periodically
                        if processed_count % self.save_interval == 0:
                            self._save_progress(stage_2_data, output_file)

                    except Exception as e:
                        # Handle batch failure
                        for clip_id in batch_clip_ids:
                            clips_data[clip_id]["retrieval_frames"] = {}
                        pbar.update(len(batch_clip_ids))

        # Final save
        self._save_progress(stage_2_data, output_file)
        print(f"Optimized Stage 2 completed! Results saved to: {output_file}")
        return stage_2_data
