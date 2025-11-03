PROMPT_DIFFERENCE_1FRAME = """
I am analyzing two photos of someone performing the same biryani cooking action: "{action}".

I want to compare how this action is done differently in the two photos.

The specific difference to check is:
"{query_string}"

**Question:** Based on what you see, which photo shows more of this difference?

- (a) Photo A
- (b) Photo B
- (c) They look similar or it's not clear

Please be honest. Choose (c) only if you are not confident there is a clear visual difference.

Return JSON:
```json
{{
  "answer": "a|b|c",
  "confidence": 1-5,
  "difference_visible": true/false,
  "explanation": "Detailed explanation of what you observed"
}}
```
"""


PROMPT_DIFFERENCE_MULTIFRAME = """
I am analyzing two sets of photos (multiple frames) of someone performing the same biryani cooking action: "{action}".

The first {num_frames} photos are from Video A, the next {num_frames} photos are from Video B.

For each video, the frames are very close together in the video: they are {time_diff} seconds apart.

The specific difference to check is:
"{query_string}"

**Question:** Based on these frames, which video shows more of this difference?

- (a) Video A
- (b) Video B
- (c) They look similar or it's not clear

Be careful: look at the entire set of frames for each video.
If you are not confident or if the difference is very minor, choose (c).

Return JSON:
```json
{{
  "answer": "a|b|c",
  "confidence": 1-5,
  "difference_visible": true/false,
  "explanation": "Detailed explanation of what you observed"
}}
```
"""
