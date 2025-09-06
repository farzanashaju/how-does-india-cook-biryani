import os
import json
import torch
import gc
from PIL import Image
from decord import VideoReader, cpu
from unsloth import FastVisionModel
from transformers import TextStreamer


model_name = "farzanashaju/llama3-biryani-16bit"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
)
FastVisionModel.for_inference(model)

def sample_frames_uniformly(start_idx, end_idx, num_frames=5):
    segment_size = (end_idx - start_idx) / num_frames
    return [int(start_idx + segment_size * i + segment_size / 2) for i in range(num_frames)]

def resize_image(img):
    return img.convert("RGB").resize((128, 128), Image.BICUBIC)

all_outputs = []

for qa_file in qa_files:
    qa_path = os.path.join(qa_dir, qa_file)
    with open(qa_path) as f:
        data = json.load(f)

    split_index = len(data) // 2
    test_items = data[split_index:]
    print(f"\n📄 File: {qa_file} | Test questions: {len(test_items)}")

    for item in test_items:
        video_names = item['videos']
        question = item['question']
        ground_truth = item['answer']

        all_images = []
        video_labels = []

        for video_name in video_names:
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            sampled_frames = sample_frames_uniformly(0, total_frames, num_frames=5)
            sampled_frames = [min(idx, total_frames - 1) for idx in sampled_frames]

            print(f"🎞️ {video_name} → {len(sampled_frames)} frames")

            images = [
                resize_image(Image.fromarray(vr[idx].asnumpy()))
                for idx in sampled_frames
            ]
            all_images.extend(images)
            video_labels.extend([video_name] * len(images))

            del vr

        content = [{
            "type": "text",
            "text": (
                f"You are an expert chef and video analyst. "
                f"You will be given frames from several cooking videos: {', '.join(video_names)}. "
                f"Compare and analyze them carefully to answer the question.\n\n"
                "Give an answer in one short sentence, including a brief reason or comparison based on what is visually shown. "
            )
        }]

        current_vid = None
        for frame, label in zip(all_images, video_labels):
            if current_vid != label:
                content.append({"type": "text", "text": f"\nVideo: {label}"})
                current_vid = label
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": question})

        conversation = [{"role": "user", "content": content}]
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": " ".join([c['text'] for c in content if c['type'] == 'text'])}] +
                           [{"type": "image"} for _ in all_images]
            }
        ]

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            text=prompt,
            images=all_images,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                streamer=text_streamer,
            )

        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        all_outputs.append({
            "videos": video_names,
            "question": question,
            "ground_truth": ground_truth,
            "vlm_answer": output_text,
            "split": "test"
        })

        del all_images, inputs, generated_ids

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_outputs, f, indent=2, ensure_ascii=False)
