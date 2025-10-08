PROMPT_DIFFERENCE_1FRAME = """
I am analyzing two photos of someone performing the same biryani cooking action: "{action}".

I want to compare how this action is done differently in the two photos.

The specific difference to check is: "{query_string}"

This means I want to determine if Video A (the first photo) shows more of this characteristic compared to Video B (the second photo).

{importance_context}

**Question:** Based on what you see, which photo shows more of this difference?

- (a) Photo A
- (b) Photo B
- (c) They look similar or it's not clear
- (d) The photos seem to be irrelevant to the query

Please be honest. Choose (c) or (d) only if you are not confident there is a clear visual difference.

**Important Guidelines:**

- Choose (a) only if Video A definitively shows MORE of the specified characteristic
- Choose (b) only if Video B definitively shows MORE of the specified characteristic
- Choose (c) if you cannot confidently distinguish between them or they appear similar
- Choose (d) if the photos don't seem to show the right cooking action or are unclear/irrelevant

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


PROMPT_DIFFERENCE_MULTIFRAME = """
I am analyzing two sets of photos (multiple frames) of someone performing the same biryani cooking action: "{action}".

Video A: Photos 1-{num_frames_a} (these frames are {time_diff} seconds apart)
Video B: Photos {num_frames_b_start}-{num_frames_total} (these frames are {time_diff} seconds apart)

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

- Choose (a) only if Video A definitively shows MORE of the specified characteristic
- Choose (b) only if Video B definitively shows MORE of the specified characteristic
- Choose (c) if you cannot confidently distinguish between them or they appear similar
- Choose (d) if the videos don't seem to show the right cooking action or are unclear/irrelevant

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
