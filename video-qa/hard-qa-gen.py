import os
import json
import random
import google.generativeai as genai
import time

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

NUM_VIDEOS_PER_PROMPT = 5
NUM_COMBINATIONS = 10

os.makedirs(LLM_OUTPUT_FOLDER, exist_ok=True)

summary_files = [f for f in os.listdir(SUMMARY_FOLDER) if f.endswith('.txt')]

file_counter = 1

for _ in range(NUM_COMBINATIONS):
    selected_files = random.sample(summary_files, NUM_VIDEOS_PER_PROMPT)

    summaries_text = []
    for fname in selected_files:
        with open(os.path.join(SUMMARY_FOLDER, fname), 'r', encoding='utf-8') as f:
            summaries_text.append(f"--- {fname} ---\n{f.read().strip()}")
    combined_summaries = "\n\n".join(summaries_text)

    prompt = f"""
You are an expert in analyzing cooking videos, with extensive knowledge of culinary techniques, ingredients, and food presentation across various regional cuisines in India.

You are provided with textual summaries of multiple cooking videos. These summaries include step-by-step actions, mentions of ingredients, utensils, and visual cues, but you do not have access to the actual videos themselves.

Task:
- Carefully compare, contrast, and synthesize the details across these multiple videos to identify key differences, similarities, and unique aspects. This includes analyzing cooking processes, ingredients, preparation times, spice usage, visual appearance, and sequencing of steps.
- Generate high-level, challenging Question-Answer (QA) pairs that require reasoning across these multiple videos, not just describing a single video.
- Use the provided set of question templates to guide your QA generation. You may also create additional multi-video QA pairs if they are insightful.

Below is a set of template questions for forming QA pairs:
(Adapt these or create new ones depending on the content.)

<Question-1> Which ingredient is common across all the recipes shown?
</Question-1> (e.g., onions are used in all three dishes)

<Question-2> Which dish uses the highest variety of spices?
</Question-2> (e.g., the Hyderabad biryani uses 7 different spices, more than the others)

<Question-3> Which recipe takes the longest time to prepare?
</Question-3> (e.g., the Lucknow biryani takes approximately 1 hour)

<Question-4> Which of the recipes do not include yogurt as an ingredient?
</Question-4> (e.g., only the Ambur biryani skips yogurt)

<Question-5> In which video is rice boiled separately before adding to the meat, unlike in the others?
</Question-5> (e.g., the Lucknow recipe)

<Question-6> Which recipe appears the most spicy?
</Question-6> (e.g., the Andhra biryani looks deep red from heavy chili usage)

<Question-7> In which video does the cook add the meat later in the cooking process compared to the others?
</Question-7> (e.g., the Kerala biryani adds meat after vegetables)

<Question-8> Which videos are the most different from each other?
</Question-8> (e.g., the Kerala and Hyderabad biryanis differ greatly in cooking method and garnish)

<Question-9> Which videos are the most similar to each other?
</Question-9> (e.g., the Ambur and Tamil Nadu biryanis are nearly identical)

- DO NOT mention the video summaries or textual descriptions directly when answering the questions. Avoid phrases like: “based on the description,” “according to the text,” “as mentioned,” or references to captions that imply the answer was derived from the provided summaries. Instead, present the information as if it is directly inferred from watching the videos.
- Do not explain or justify how the answer was obtained.
- Keep all answers concise, and highlight important keywords using bold formatting.
- If a particular question does not apply to this set of videos, simply do not generate a QA pair for it.

Output Format:

Your entire response must be formatted in JSON as shown below:
{{
  "Summary": "",
  "QA_pairs": [
    {{"Q": "", "A": ""}},
    {{"Q": "", "A": ""}}
  ]
}}

Video summaries:
\"\"\"{combined_summaries}\"\"\"
"""

    response = model.generate_content(prompt)
    llm_text = response.text.strip()

    output_text = f"Videos: {selected_files}\n\n{llm_text}"

    output_file_path = os.path.join(LLM_OUTPUT_FOLDER, f"{file_counter:04}.txt")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"Saved {output_file_path} with videos: {selected_files}")
    file_counter += 1

    time.sleep(7)
