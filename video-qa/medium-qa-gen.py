import os
import json
import google.generativeai as genai
import time

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


input_files = {f for f in os.listdir(VDESC_INPUT_FOLDER) if f.endswith('.txt')}
output_files = {f for f in os.listdir(LLM_OUTPUT_FOLDER) if f.endswith('.txt')}

files_to_process = input_files - output_files

for filename in sorted(files_to_process):
    filepath1 = os.path.join(VDESC_INPUT_FOLDER, filename)
    filepath2 = os.path.join(TRANS_INPUT_FOLDER, filename)
    with open(filepath1, 'r', encoding='utf-8') as f:
        video_description = f.read().strip()
    with open(filepath2, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()

    prompt = f"""
You are an expert in analyzing cooking videos, with extensive knowledge of culinary techniques, ingredients, and food presentation across various regional cuisines in India.

You are provided with a detailed textual description of the cooking video and the full transcript of the spoken narration. This data includes step-by-step cooking processes, mentions of ingredients, utensils, cooking durations, and visual cues — but you do not have access to the actual video.

Task:
- Identify and describe the key cooking processes, ingredients, and presentation details discussed in the textual description and summary. (The key cooking process refers to the main focus of the video that is highlighted in the provided text.)
- Generate relevant Question-Answer (QA) pairs by carefully analyzing the textual description and summary of the cooking video.
- In addition to using the provided template questions, feel free to create additional QA pairs that are contextually appropriate based on the content.

Below is a set of template questions for forming QA pairs:
(Adapt these or create new ones depending on the content.)

<Question-1> What are the primary ingredients used in this recipe?
</Question-1> (e.g., chicken, rice, yogurt, spices, onions, tomatoes)

<Question-2> In what order are the ingredients added during cooking?
</Question-2> (e.g., oil → spices → onions → meat → tomatoes → yogurt)

<Question-3> Which spices or seasonings are used in this dish?
</Question-3> (e.g., cumin seeds, coriander powder, garam masala, turmeric, salt)

<Question-4> What kind of meat is used in the recipe?
</Question-4> (e.g., goat, chicken, fish, lamb, beef, none)

<Question-5> What is the first step shown in the video?
</Question-5> (e.g., rinsing and soaking the rice, marinating the meat)

<Question-6> What is the last step before serving?
</Question-6> (e.g., garnishing with fresh coriander and fried onions)

<Question-7> How is the meat prepared before cooking?
</Question-7> (e.g., marinated with yogurt, turmeric, and chili powder, layered with meat)

<Question-8> What type of pan or vessel is used to cook this dish?
</Question-8> (e.g., a wide heavy-bottomed metal pot, clay pot, pressure cooker)

<Question-9> How long is the rice cooked for?
</Question-9> (e.g., approximately 15 minutes until tender)

<Question-10> Approximately how long does it take to prepare this entire dish?
</Question-10> (e.g., around 45 minutes total)

<Question-11> What does the final dish look like?
</Question-11> (e.g., orange-red rice with chicken pieces and green garnish)

<Question-12> What is used to garnish the dish before serving?
</Question-12> (e.g., chopped coriander leaves, fried onions, lemon slices)

<Question-13> Does the dish appear to be spicy?
</Question-13> (e.g., yes, it looks spicy due to the visible red chili oil)

<Question-14> When is the rice mixed with the meat or gravy?
</Question-14> (e.g., after the meat is cooked for 15 minutes)

<Question-15>  Is the dish served with any accompaniments?
</Question-15> (e.g., onion raita, boiled eggs, salad)

- DO NOT mention the video summary or transcript directly when answering the questions. Avoid phrases like: “based on the description,” “according to the text,” “as mentioned,” or references to captions that imply the answer was derived from the provided text. Instead, present the information as if it is directly inferred from watching the video.
- Do not explain or justify how the answer was obtained.
- You may choose to omit details that seem irrelevant to the cooking process or final dish.
- Keep all answers concise, and highlight important keywords using bold formatting.
- If a particular question does not apply to the video, simply do not generate a QA pair for it.
- Focus on content directly relevant to the cooking process, ingredients, or presentation. Ignore unrelated background commentary.

Output Format:

Your entire response must be formatted in JSON as shown below:
{{
  "Summary": "",
  "QA_pairs": [
    {{"Q": "", "A": ""}},
    {{"Q": "", "A": ""}},
    {{"Q": "", "A": ""}},
    {{"Q": "", "A": ""}}
  ]
}}

Video Description:
\"\"\"{video_description}\"\"\"

Transcript:
\"\"\"{transcript}\"\"\"
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    llm_output_path = os.path.join(LLM_OUTPUT_FOLDER, filename)
    with open(llm_output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"✅ Processed {filename}")
    time.sleep(5)
