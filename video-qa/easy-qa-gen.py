import os
import re
import json
import random
from openai import OpenAI

# Running LLaMA-3 locally
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def parse_chunks_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"=== Chunk (\d+) ===\nStart frame: (\d+)\nEnd frame: (\d+)\nDescription:\n(.*?)={78,}"
    matches = re.findall(pattern, content, re.DOTALL)
    chunks = []
    for m in matches:
        chunk_no = int(m[0])
        start_frame = int(m[1])
        end_frame = int(m[2])
        description = m[3].strip().replace('\n', ' ')
        chunks.append({
            "chunk": chunk_no,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "description": description
        })
    return chunks

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.txt'):
        continue
    filepath = os.path.join(INPUT_FOLDER, filename)
    chunks = parse_chunks_from_file(filepath)
    if len(chunks) < 3:
        selected_chunks = chunks
    else:
        selected_chunks = random.sample(chunks, 3)

    all_outputs = []
    all_json_results = []

    for chunk in selected_chunks:
        desc = chunk["description"]

        prompt = f"""
Video segment description:
\"\"\"{desc}\"\"\"

Answer the following clearly:

1. What are the ingredients shown in this segment?
2. What are the utensils shown in this segment?
3. What are the cooking actions performed in this segment?

Return each answer as a numbered list item.
"""

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        all_outputs.append(f"=== Chunk {chunk['chunk']} ===\n{text}\n")

        answers = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|$)", text, re.DOTALL)
        questions = [
            "What are the ingredients shown in this segment?",
            "What are the utensils shown in this segment?",
            "What are the cooking actions performed in this segment?"
        ]
        for idx, q in enumerate(questions):
            ans_text = answers[idx].strip() if idx < len(answers) else ""
            all_json_results.append({
                "video": filename,
                "chunk": chunk["chunk"],
                "start_frame": chunk["start_frame"],
                "end_frame": chunk["end_frame"],
                "question": q,
                "answer": ans_text
            })

    llm_output_path = os.path.join(LLM_OUTPUT_FOLDER, filename)
    with open(llm_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_outputs))

    json_output_path = os.path.join(JSON_OUTPUT_FOLDER, filename.replace('.txt', '.json'))
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_json_results, f, indent=2, ensure_ascii=False)
