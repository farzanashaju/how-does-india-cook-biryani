PROMPT_DIFFERENCES = """
I am studying how people make biryani in different ways.
I have an action with the following description: "{action}".

Propose **2 to 3 possible ways** this action can vary between different cooks that are VISUALLY OBSERVABLE in video frames And would EFFECT THE FINAL DISH.
Focus on differences that can be judged by comparing individual frames from two videos.

Each difference should be:
- **Visual**: observable in a single frame or sequence of frames
- **Comparative**: describable as "more/less", "higher/lower", "faster/slower", etc.
- **Specific**: about physical appearance, not abstract concepts

Examples of good differences:
- "the spices are darker in color"
- "more oil is visible in the pan"
- "the ingredients are cut into smaller pieces"

For each difference, give:
- **name**: a short comparative phrase (e.g. "darker spices", "more oil")
- **query_string**: a comparative statement that can compare videos with (e.g. "has darker spices", "has more visible oil", "has tighter foil coverage", "has smaller vegetable pieces")

The query_string should work for comparing ANY number of videos, so do not mention specific videos like "Video A is darker than Video B". Instead, use neutral comparative phrases that allow ranking multiple videos along a spectrum.

Return JSON with EXACTLY this structure (always use numbered keys even for single differences):
```json
{{
  "0": {{ "name": "...", "query_string": "..." }},
  "1": {{ "name": "...", "query_string": "..." }},
}}
```
"""

PROMPT_STAGES = """
I am analyzing Indian biryani cooking.
I have an action with the following description: "{action}".

Break this action into 4 or fewer short sub-actions (stages) that are VISUALLY DISTINCT in video frames.
Each stage should show different visual elements that can be recognized in frames.

For each stage, include:
- **name**: short title of the step
- **description**: 1-2 sentences explaining what happens
- **retrieval_strings**: 3+ short sentences describing what should be visible in frames of this stage
  - Focus on ingredients, tools, hand positions, textures, colors
  - Start with "A photo of..." or "A frame showing..."
  - Describe specific visual elements, not actions

Return JSON:
```json
{{
  "stages": [
    {{
      "name": "...",
      "description": "...",
      "retrieval_strings": ["A photo of a ...", "A frame showing ...", "A photo of a ..."]
    }}
  ]
}}
```
"""

PROMPT_LINKING = """
I have an action: "{action}".

Below are the stages (sub-actions) of this action:
{stages}

And here are the visual differences in how this action may be performed:
{differences}

For each stage, list which visual differences can be observed during that stage.
- A difference can appear in multiple stages
- A stage can have no differences
- Every difference must appear in at least one stage

Return JSON mapping stage names to difference names:
```json
{{
  "stage_name_1": ["difference_name_1", "difference_name_2"],
  "stage_name_2": [],
  "stage_name_3": ["difference_name_1"]
}}
```

Use exact stage names and difference names from above.
"""
