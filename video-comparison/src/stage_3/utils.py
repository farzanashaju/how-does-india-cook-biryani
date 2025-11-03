import json
import os
import re
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def load_model_and_processor(model_path: str):
    """
    Load the Qwen2.5 VL model and processor.

    Args:
        model_path: Path to the Hugging Face model

    Returns:
        tuple: (model, processor)
    """
    print(f"Loading model from {model_path}...")
    try:
        print("trying to load llama")
        model_path = "DAMO-NLP-SG/VideoLLaMA3-7B-Image"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print("✅ Model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}. ITS HEREError: {e}")
        raise


def clean_json_string(s: str) -> str:
    """
    Clean JSON string by removing problematic characters and formatting.

    Args:
        s: Raw JSON string from LLM

    Returns:
        str: Cleaned JSON string
    """
    # Replace non-breaking spaces and other weird whitespace
    s = s.replace("\u00a0", " ")

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    # Replace smart quotes with standard quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    return s


def extract_json_block(text: str) -> str:
    """
    Extract JSON block from LLM response text.

    Args:
        text: Raw text from LLM response

    Returns:
        str: Extracted JSON string

    Raises:
        ValueError: If no JSON block is found
    """
    # First try to find fenced code blocks
    blocks = re.findall(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        json_block = blocks[-1].strip()
        print("✅ Found fenced ```json block.")
    else:
        # Fallback: find JSON-like blocks
        print("⚠️ No fenced ```json block found, using fallback.")
        blocks = re.findall(r"\{[\s\S]*?\}", text)
        if not blocks:
            raise ValueError("No JSON block found in the LLM response.")

        # Return the longest block (most likely to be complete)
        json_block = sorted(blocks, key=len, reverse=True)[0].strip()

    # Remove stray backticks
    json_block = json_block.replace("```", "").strip()

    # Balance braces if needed
    start = json_block.find("{")
    if start == -1:
        raise ValueError("No opening brace found in JSON block")

    # Find balanced closing brace
    brace_count = 0
    for i, c in enumerate(json_block[start:]):
        if c == "{":
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0:
                end = start + i + 1
                return json_block[start:end]

    # If we can't find balanced braces, return the whole block
    return json_block


def parse_json_from_text(text: str) -> dict:
    """
    Parse JSON from LLM response text with robust error handling.

    Args:
        text: Raw text from LLM response

    Returns:
        dict: Parsed JSON data

    Raises:
        json.JSONDecodeError: If JSON parsing fails
        ValueError: If no JSON block is found
    """
    json_str = extract_json_block(text)
    json_str = clean_json_string(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Problematic JSON string: {json_str}")
        raise


def extract_actions_from_json(json_file_path: str) -> List[str]:
    """
    Extract unique action types from a JSON file.

    Args:
        json_file_path: Path to JSON file containing action data

    Returns:
        List[str]: Sorted list of unique action types
    """
    print(f"📂 Loading actions from {json_file_path}...")
    with open(json_file_path) as f:
        data = json.load(f)

    # Assumes a list of dicts, each with an "action_type" key
    unique_actions = sorted(list(set(item["action_type"] for item in data)))
    print(f"📌 Found {len(unique_actions)} unique action types.")
    return unique_actions


def save_json(data: Dict, output_filepath: str):
    """
    Save data to JSON file with proper directory creation.

    Args:
        data: Data to save
        output_filepath: Path where to save the JSON file
    """
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_existing_results(filepath: str) -> Dict:
    """
    Load existing results from JSON file if it exists.

    Args:
        filepath: Path to existing results file

    Returns:
        Dict: Existing results or empty dict if file doesn't exist
    """
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing_data = json.load(f)
        print(
            f"✅ Resuming. Loaded {len(existing_data)} completed actions from {filepath}"
        )
        return existing_data
    return {}


def call_llm(
    model,
    processor,
    prompt: str,
    max_tokens: int = 1024,
    device: str = "cuda",
    imgs=None,
) -> dict:
    """
    Send a prompt to the LLM and parse the JSON response.
    Supports both text-only and multimodal (text + images) input.

    Args:
        model: The loaded model
        processor: The loaded processor
        prompt: The prompt to send
        max_tokens: Maximum tokens for generation
        device: Device to use for inference
        imgs: Optional list of PIL Images for multimodal input
    Returns:
        dict: Parsed JSON response from the LLM
    """

    # Build the conversation format expected by VideoLLaMA3
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant expert in analyzing cooking procedures. You always respond with a valid JSON object enclosed in ```json ``` tags.",
                }
            ],
        }
    ]

    # Handle multimodal input (text + images)
    if imgs is not None and len(imgs) > 0:
        print(f"🖼️ Processing {len(imgs)} images with text prompt")

        # Build user message with images and text
        user_content = []

        # Add all images first
        for img in imgs:
            user_content.append({"type": "image", "image": img})

        # Add the text prompt
        user_content.append({"type": "text", "text": prompt})

        conversation.append({"role": "user", "content": user_content})

    else:
        # Text-only input
        print("📝 Processing text-only prompt")
        conversation.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )

    # Process the conversation
    inputs = processor(conversation=conversation, return_tensors="pt")

    # Move inputs to device and handle data types
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
        )

    # Decode the response
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    try:
        return parse_json_from_text(response)
    except Exception as e:
        print("❗️ Error parsing model output.")
        print(f"Full generated text:\n---\n{response}\n---")
        print(f"Error: {e}")


def print_progress(current: int, total: int, action: str):
    """
    Print progress information.

    Args:
        current: Current item number
        total: Total number of items
        action: Name of current action being processed
    """
    print(f"\n📍 Progress: {current}/{total}")
    print(f"{'=' * 50}")
    print(f"Processing action: '{action}'")
    print(f"{'=' * 50}")


def print_stage_info(
    stage_num: int, stage_name: str, action: str, result_count: int = None
):
    """
    Print information about current stage being processed.

    Args:
        stage_num: Stage number (1, 2, or 3)
        stage_name: Name of the stage
        action: Action being processed
        result_count: Number of results found (optional)
    """
    print(f"\t[Stage {stage_num}] {stage_name} for '{action}'...")
    if result_count is not None:
        print(f"\t> Found {result_count} results.")
