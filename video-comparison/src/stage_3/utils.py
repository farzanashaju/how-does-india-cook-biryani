import json
import re
from typing import Dict, Any


def clean_json_response(response_text: str) -> Dict[str, Any] | None:
    """
    Clean and parse JSON response from Gemini API.
    Handles various formats and edge cases.
    """
    if not response_text:
        raise ValueError("Response text is empty or None")

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    answer_match = re.search(r'"answer":\s*"([abcd])"', response_text, re.IGNORECASE)
    confidence_match = re.search(r'"confidence":\s*(\d+)', response_text)
    visible_match = re.search(
        r'"difference_visible":\s*(true|false)', response_text, re.IGNORECASE
    )
    explanation_match = re.search(r'"explanation":\s*"([^"]*)"', response_text)

    if not (answer_match or confidence_match or visible_match or explanation_match):
        print(f"⚠️ No valid JSON or expected fields found in: {response_text}...")
        return None

    result = {
        "answer": answer_match.group(1) if answer_match else "c",
        "confidence": int(confidence_match.group(1)) if confidence_match else 0,
        "difference_visible": visible_match.group(1).lower() == "true" if visible_match else False,
        "explanation": (
            explanation_match.group(1) if explanation_match else response_text[:200] + "..."
        ),
    }

    print(f"⚠️ Fallback parsing used for: {response_text[:100]}...")
    return result
