import os
import re
import json
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BIRYANI_TYPES = [
    "ambur_biryani", "bombay_biryani", "dindigul_biryani", "donne_biryani",
    "hyderabadi_biryani", "kashmiri_biryani", "kolkata_biryani", "lucknow_awadhi_biryani",
    "malabar_biryani", "mughlai_biryani", "sindhi_biryani", "thalassery_biryani"
]

# Transformations
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, image_size=448, max_num=1):
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    return [image]

def sample_frames_uniformly(start_idx, end_idx, num_frames=8):
    segment_size = (end_idx - start_idx) / num_frames
    return [int(start_idx + segment_size * i + segment_size / 2) for i in range(num_frames)]

def load_video_chunk(vr, frame_indices, input_size=448):
    transform = build_transform(input_size)
    pixel_values_list, num_patches_list = [], []
    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy())
        tiles = dynamic_preprocess(img, image_size=input_size)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(len(tiles))
    return torch.cat(pixel_values_list), num_patches_list

def get_time_chunks(max_frame, fps, chunk_duration=10):
    chunks = []
    chunk_size = int(chunk_duration * fps)
    for start in range(0, max_frame, chunk_size):
        end = min(start + chunk_size, max_frame)
        chunks.append((start, end))
    return chunks

def extract_items_from_response(response):
    sections = {"Ingredients": [], "Utensils": [], "Actions": []}
    current_section = None

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.lower().startswith("ingredients"):
            current_section = "Ingredients"
            continue
        elif line.lower().startswith("utensils"):
            current_section = "Utensils"
            continue
        elif line.lower().startswith("actions"):
            current_section = "Actions"
            continue

        if line.startswith("-") and current_section:
            item = line.lstrip("-*• ").strip().lower()
            if item and item not in sections[current_section]:
                sections[current_section].append(item)

    return sections

# Model loading
path = "OpenGVLab/InternVL3-14B"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    path,
    quantization_config=quant_config,
    device_map={"": 0},  # ✅ Avoids `.to()` issues in 4-bit models
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True,
    use_fast=False
)

# Process all videos
for biryani_type in BIRYANI_TYPES:
    for video_id in range(1, 11):
        video_dir = f"converted_biryani_videos/{biryani_type}/video{video_id}"
        video_path = f"{video_dir}/video.mp4"

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        print(f"Processing {video_path}")

        vr = VideoReader(video_path, ctx=cpu(0))
        max_frame = len(vr)
        fps = float(vr.get_avg_fps())

        chunk_indices = get_time_chunks(max_frame, fps)
        segment_data = []
        srt_output = ""

        for chunk_id, (start_idx, end_idx) in enumerate(chunk_indices):
            sampled_frames = sample_frames_uniformly(start_idx, end_idx)
            pixel_values, num_patches_list = load_video_chunk(vr, sampled_frames)

            pixel_values = pixel_values.to(dtype=torch.float16, device=model.device)

            question = '\n'.join([f'Video-Chunk{chunk_id+1}-Frame{i+1}: <image>' for i in range(len(num_patches_list))])
            question += '''\nYou are analyzing a cooking video.\nPlease extract information into three clearly labeled bullet-point lists, based strictly on what is visually present in the video frames.\nRespond only with the following three sections in this exact order:\nIngredients:\n- List all ingredients that are clearly visible or being used (e.g., chopped onions, turmeric powder, rice).\nUtensils:\n- List all visible cooking tools, vessels, or utensils (e.g., knife, pressure cooker, ladle).\nActions:\n- Describe each distinct cooking action as a verb-noun phrase (e.g., chopping onions, frying spices, stirring curry).\n'''

            with torch.no_grad():
                response, _ = model.chat(
                    tokenizer, pixel_values, question,
                    generation_config=dict(max_new_tokens=1024, do_sample=True),
                    num_patches_list=num_patches_list, history=None, return_history=True
                )

            print(f"\n--- Response for Chunk {chunk_id+1} ({int(start_idx/fps)}s - {int(end_idx/fps)}s) ---\n")
            print(response)

            parsed = extract_items_from_response(response)

            timestamp = f"{int(start_idx / fps)}s - {int(end_idx / fps)}s"
            segment_data.append({
                "timestamp": timestamp,
                "ingredients": parsed["Ingredients"],
                "utensils": parsed["Utensils"],
                "actions": parsed["Actions"]
            })

            srt_output += f"{chunk_id+1}\n{timestamp}\n{response.strip()}\n\n"

            torch.cuda.empty_cache()

        output_dir = f"processed_segments/{biryani_type}/video{video_id}"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/segment.json", "w", encoding="utf-8") as f:
            json.dump(segment_data, f, indent=4, ensure_ascii=False)

        with open(f"{output_dir}/raw.srt", "w", encoding="utf-8") as f:
            f.write(srt_output)

        print(f"Saved output to {output_dir}")
