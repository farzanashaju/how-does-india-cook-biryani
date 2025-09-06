import os
import gc
import json
import cv2
import torch
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from typing import List, Dict, Any

os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"

PROMPTS = {
    "easy": "You are an expert in analyzing cooking videos. ONLY output a comma-separated list with NO explanation.",
    "medium": "You are an expert in cooking videos. Answer the question in one or two short, clear sentences. Be direct.",
    "hard": "You are an expert chef and video analyst. Compare these videos carefully and answer in one short sentence with a reason."
}

def extract_frames_from_video(video_path: str, num_frames: int = 5, start_frame: int = None, end_frame: int = None) -> List[Image.Image]:
    """Extract frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame is not None and end_frame is not None:
        # For easy questions - extract from specific segment
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    else:
        # For medium/hard questions - extract uniformly from entire video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames

def convert_to_rgb(image):
    """Convert image to RGB format if not already in RGB."""
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    return alpha_composite.convert("RGB")

def reduce_image_size(image):
    return image.resize((128,128))

def format_easy_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format easy questions (10-second segments)."""
    video_path = os.path.join(video_dir, f"{item['video']}.mp4")
    frames = extract_frames_from_video(
        video_path, 
        num_frames=5, 
        start_frame=item['start_frame'], 
        end_frame=item['end_frame']
    )
    
    if not frames:
        return None
    
    # Process frames
    processed_frames = []
    for frame in frames:
        frame = convert_to_rgb(frame)
        frame = reduce_image_size(frame)
        processed_frames.append(frame)
    
    # Create content with multiple images
    content = [{"type": "text", "text": PROMPTS["easy"]}]
    for frame in processed_frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": item["question"]})
    
    return {
        "messages": [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": item["answer"],
                    }
                ],
            },
        ],
    }

def format_medium_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format medium questions (entire video)."""
    video_path = os.path.join(video_dir, f"{item['video']}.mp4")
    frames = extract_frames_from_video(video_path, num_frames=5)
    
    if not frames:
        return None
    
    # Process frames
    processed_frames = []
    for frame in frames:
        frame = convert_to_rgb(frame)
        frame = reduce_image_size(frame)
        processed_frames.append(frame)
    
    # Create content with multiple images
    content = [{"type": "text", "text": PROMPTS["medium"]}]
    for frame in processed_frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": item["question"]})
    
    return {
        "messages": [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": item["answer"],
                    }
                ],
            },
        ],
    }

def format_hard_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format hard questions (multiple videos)."""
    all_frames = []
    video_labels = []
    
    for video_name in item['videos']:
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        frames = extract_frames_from_video(video_path, num_frames=5)
        
        if not frames:
            continue
            
        # Process frames
        for frame in frames:
            frame = convert_to_rgb(frame)
            frame = reduce_image_size(frame)
            all_frames.append(frame)
            video_labels.append(video_name)
    
    if not all_frames:
        return None
    
    # Create content with labeled images
    content = [{"type": "text", "text": PROMPTS["hard"]}]
    
    current_video = None
    for i, (frame, video_label) in enumerate(zip(all_frames, video_labels)):
        if video_label != current_video:
            content.append({"type": "text", "text": f"\nVideo: {video_label}"})
            current_video = video_label
        content.append({"type": "image", "image": frame})
    
    content.append({"type": "text", "text": item["question"]})
    
    return {
        "messages": [
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": item["answer"],
                    }
                ],
            },
        ],
    }

def load_and_format_dataset():
    """Load and format the video QA dataset."""
    converted_dataset = []
    
    for qa_type in qa_types:
        qa_dir = os.path.join(qa_root_dir, qa_type)
        if not os.path.exists(qa_dir):
            continue
        
        print(f"Processing {qa_type} questions...")
        
        for qa_file in os.listdir(qa_dir):
            if not qa_file.endswith(".json"):
                continue
            
            with open(os.path.join(qa_dir, qa_file)) as f:
                data = json.load(f)
            
            # Only use first half for training
            split_index = len(data) // 2
            train_items_from_file = data[:split_index]
            
            for item in train_items_from_file:
                try:
                    if qa_type == "easy":
                        formatted_item = format_easy_question(item)
                    elif qa_type == "medium":
                        formatted_item = format_medium_question(item)
                    else:  # hard questions
                        formatted_item = format_hard_question(item)
                    
                    if formatted_item:
                        converted_dataset.append(formatted_item)
                        
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue

    print(f"Total training samples: {len(converted_dataset)}")
    return converted_dataset

converted_dataset = load_and_format_dataset()

import torch

if not hasattr(torch.compiler, "set_stance"):
    def dummy_set_stance(*args, **kwargs):
        print("⚠️ torch.compiler.set_stance not available. Skipping.")
    torch.compiler.set_stance = dummy_set_stance

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

from unsloth import FastVisionModel

model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)

from transformers import TextStreamer

if converted_dataset:
    FastVisionModel.for_inference(model)
    
    sample = converted_dataset[0]
    user_message = sample["messages"][0]
    
    images = []
    text_parts = []
    
    for content in user_message["content"]:
        if content["type"] == "image":
            images.append(content["image"])
        elif content["type"] == "text":
            text_parts.append(content["text"])
    
    combined_text = " ".join(text_parts)
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": combined_text}
            ] + [{"type": "image"} for _ in images]
        }
    ]
    
    if images:
        print(f"Processing {len(images)} images for inference...")

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = tokenizer(
            text=input_text,
            images=images,
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

args = SFTConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    warmup_steps = 5,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    fp16 = not is_bf16_supported(),
    bf16 = is_bf16_supported(),
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    save_strategy = "no",
    load_best_model_at_end = True,
    save_only_model = False,
    remove_unused_columns = False,
    dataset_text_field = "",
    dataset_kwargs = {"skip_prepare_dataset": True},
    dataset_num_proc = 4,
    max_seq_length = 4096,
)

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = args,
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

trainer.train()
