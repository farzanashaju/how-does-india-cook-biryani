import os
import json
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import gc
import time

from unsloth import FastVisionModel
from transformers import TextStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "farzanashaju/llama3-biryani-16bit"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
)
FastVisionModel.for_inference(model)

def sample_frames_uniformly(start_idx, end_idx, num_frames=5):
    segment_size = (end_idx - start_idx) / num_frames
    return [int(start_idx + segment_size * i + segment_size / 2) for i in range(num_frames)]

all_outputs = []

for qa_file in qa_files:
    qa_path = os.path.join(qa_dir, qa_file)
    with open(qa_path) as f:
        data = json.load(f)

    split_index = len(data) // 2
    test_items = data[split_index:]
    print(f"\n📄 File {qa_file}: {len(test_items)} test questions.")

    for idx, item in enumerate(test_items):
        video_name = item['video']
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        question = item['question']
        ground_truth = item['answer']

        print(f"\n🎬 [{idx+1}/{len(test_items)}] Processing: {video_name}")

        t0 = time.time()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        chunk_duration_sec = 10
        start_frame = int((item['chunk'] - 1) * chunk_duration_sec * fps)
        end_frame = min(int(item['chunk'] * chunk_duration_sec * fps), len(vr))
        print(f"  ⏱ Loaded in {time.time() - t0:.2f}s, frames {start_frame}-{end_frame}")

        sampled_idxs = sample_frames_uniformly(start_frame, end_frame, num_frames=5)
        sampled_images = [
            Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in sampled_idxs
        ]

        prompt_text = (
            "You are an expert in analyzing cooking videos. "
            "For the following question, ONLY output a comma-separated list with NO extra explanation. Do not write complete sentences. Do not add any description.\n\n" + question
        )

        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}] +
                           [{"type": "image"} for _ in sampled_images]
            }
        ]

        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = tokenizer(
            text=prompt,
            images=sampled_images,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        all_outputs.append({
            "video": video_name,
            "chunk": item['chunk'],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "question": question,
            "ground_truth": ground_truth,
            "vlm_answer": output_text,
            "split": "test"
        })

        del inputs, output, sampled_images, vr
        torch.cuda.empty_cache()
        gc.collect()

out_path = os.path.join(output_dir, "llama3ft_answers.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=2, ensure_ascii=False)
