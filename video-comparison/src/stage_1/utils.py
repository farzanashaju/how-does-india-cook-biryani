import json
import os
import re
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_path: str = "Qwen/Qwen2.5-14B"):
    """
    Load the Qwen2.5 model and tokenizer.

    Args:
        model_path: Path to the Hugging Face model

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load model. Error: {e}")
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
        print(f"✅ Resuming. Loaded {len(existing_data)} completed actions from {filepath}")
        return existing_data
    return {}


def call_llm(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 2048,
    device: str = "cuda",
    max_retries: int = 1,
) -> Optional[dict]:
    """
    Send a prompt to the LLM and parse the JSON response with retry logic.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: The prompt to send
        max_tokens: Maximum tokens for generation
        device: Device to use for inference
        max_retries: Maximum number of retry attempts

    Returns:
        dict: Parsed JSON response from the LLM or None if all retries fail
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant expert in analyzing cooking procedures. You always respond with a valid JSON object enclosed in ```json ... ``` tags.",
                },
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
            )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            result = parse_json_from_text(generated_text)
            if result is not None:
                return result

            if attempt < max_retries:
                print(f"LLM call attempt {attempt + 1} failed, retrying...")

        except Exception as e:
            if attempt < max_retries:
                print(f"LLM call attempt {attempt + 1} failed with error: {e}, retrying...")
            else:
                print(f"LLM call failed after {max_retries + 1} attempts: {e}")
                return None

    print(f"LLM call failed after all {max_retries + 1} attempts")
    return None


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
