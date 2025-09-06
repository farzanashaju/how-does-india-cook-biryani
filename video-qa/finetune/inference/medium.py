import os
import json
import torch
import time
from decord import VideoReader, cpu
from PIL import Image
import gc

from unsloth import FastVisionModel
from transformers import TextStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "farzanashaju/llama3-biryani-16bit"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_path,
    load_in_4bit=False,
)
FastVisionModel.for_inference(model)

def sample_frames_uniformly(start_idx, end_idx, num_frames=5):
    segment_size = (end_idx - start_idx) / num_frames
    return [int(start_idx + segment_size * i + segment_size / 2) for i in range(num_frames)]

def convert_to_rgb(image):
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    return alpha_composite.convert("RGB")

def resize_image(image, size=(128, 128)):
    return image.resize(size)

all_outputs = []

for qa_file in qa_files:
    qa_path = os.path.join(qa_dir, qa_file)
    with open(qa_path) as f:
        data = json.load(f)

    split_index = len(data) // 2
    test_items = data[split_index:]
    print(f"\n📄 File {qa_file}: {len(test_items)} test questions.")

    for idx, item in enumerate(test_items):
        video_name = item["video"]
        question = item["question"]
        ground_truth = item["answer"]
        video_path = os.path.join(video_dir, f"{video_name}.mp4")

        print(f"\n🎬 [{idx+1}/{len(test_items)}] Processing {video_name}")

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        frame_idxs = sample_frames_uniformly(0, total_frames, num_frames=5)
        frame_idxs = [min(f, total_frames - 1) for f in frame_idxs]

        images = []
        for idx in frame_idxs:
            frame = Image.fromarray(vr[idx].asnumpy())
            frame = convert_to_rgb(frame)
            frame = resize_image(frame, (128, 128))
            images.append(frame)

        prompt_instruction = "You are an expert in analyzing cooking videos. Answer the following question in one or two short, clear sentences. Do not provide any explanation or reasoning steps. Be direct and precise."
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_instruction}]
                        + [{"type": "image"} for _ in images]
                        + [{"type": "text", "text": question}],
            }
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            text=input_text,
            images=images,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        print(f"  🚀 Generating...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        print(f"  📝 Output: {output_text}")

        all_outputs.append({
            "video": video_name,
            "question": question,
            "ground_truth": ground_truth,
            "vlm_answer": output_text,
            "split": "test"
        })

        del inputs, generated_ids, vr
        torch.cuda.empty_cache()
        gc.collect()

out_file = os.path.join(output_dir, "llama3ft_answers.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=2, ensure_ascii=False)
