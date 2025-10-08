PROMPT_DIFFERENCE_MULTIFRAME = """
I am analyzing two sets of photos ({total_frames} total) of someone performing the same biryani cooking action: "{action}".
Video A: Photos {clip1_range}
Video B: Photos {clip2_start}-{clip2_end}

The specific difference to check is: "{query_string}"
This means I want to determine if Video A shows more of this characteristic compared to Video B.

{importance_context}

**Question:** Based on these frames, which video shows more of this difference?
- (a) Video A
- (b) Video B
- (c) They look similar or it's not clear
- (d) The videos seem to be irrelevant to the query

Be careful: look at the entire set of frames for each video.
If you are not confident or if the difference is very minor, choose (c).

**Important Guidelines:**

- Choose (a) if the video A clearly shows more of the difference than video B
- Choose (b) if the video B clearly shows more of the difference than video A
- Choose (c) if you cannot confidently distinguish between them or they appear similar
- Choose (d) if the videos do not relate to the query at all / the action shown is completely different to the cooking action

Return JSON:
```json
{{
  "answer": "a|b|c|d",
  "confidence": 1-5,
  "difference_visible": true/false,
  "explanation": "Detailed explanation of what you observed"
}}
```
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant expert in analyzing cooking procedures. "
    "You always respond with a valid JSON object enclosed in ```json ``` tags."
)
