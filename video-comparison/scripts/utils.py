import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def normalize_action_title(action: str) -> str:
    """
    Normalize action titles for better similarity matching.

    Args:
        action: The action string to normalize

    Returns:
        Normalized action string
    """
    # Convert to lowercase
    action = action.lower()

    # Remove common cooking prefixes/suffixes
    action = re.sub(
        r"^(preparing|making|cooking|adding|mixing|stirring|placing|arranging)\s+",
        "",
        action,
    )
    action = re.sub(r"\s+(preparation|process|method|technique)$", "", action)

    # Standardize common terms
    replacements = {
        "chicken pieces": "chicken",
        "chicken meat": "chicken",
        "rice grains": "rice",
        "cooked rice": "rice",
        "boiled eggs": "eggs",
        "hard boiled eggs": "eggs",
        "sliced onions": "onions",
        "chopped onions": "onions",
        "red onions": "onions",
        "green chilies": "chilies",
        "green chili peppers": "chilies",
        "serving biryani": "serving",
        "displaying": "showing",
        "close-up": "showing",
        "highlighting": "showing",
    }

    for old, new in replacements.items():
        action = action.replace(old, new)

    # Remove extra whitespace and punctuation
    action = re.sub(r"[^\w\s]", " ", action)
    action = re.sub(r"\s+", " ", action).strip()

    return action


def calculate_action_statistics(action_counts: Counter) -> Tuple[int, int, int]:
    """
    Calculate statistics for actions.

    Args:
        action_counts: Counter of action occurrences

    Returns:
        Tuple of (num_classes, total_instances, num_comparisons)
    """
    num_action_classes = len(action_counts)
    total_action_instances = sum(action_counts.values())

    num_comparisons = 0
    for _, count in action_counts.items():
        if count >= 2:
            num_comparisons += math.comb(count, 2)

    return num_action_classes, total_action_instances, num_comparisons


def print_statistics(action_counts: Counter, title: str = "Action Statistics") -> None:
    """Print action statistics using rich table."""
    num_classes, total_instances, num_comparisons = calculate_action_statistics(
        action_counts
    )

    console = Console()
    table = Table(title=title)
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("Number of Action Classes", str(num_classes))
    table.add_row("Total Action Instances", str(total_instances))
    table.add_row("Number of Action Comparisons", str(num_comparisons))

    console.print(table)


def print_clustering_results(combination_info: Dict[str, List[str]]) -> None:
    """
    Print clustering results showing which actions were combined.

    Args:
        combination_info: Dictionary mapping representative names to lists of combined actions
    """
    console = Console()
    if combination_info:
        console.print("\n[bold green]Action Clustering Results:[/bold green]")
        for rep_action, combined_actions in combination_info.items():
            if len(combined_actions) > 1:
                console.print(f"\n[yellow]'{rep_action}'[/yellow] combines:")
                for action in combined_actions:
                    console.print(f"  - {action}")
    else:
        console.print("\n[yellow]No actions were clustered together.[/yellow]")


def load_model_and_tokenizer(model_path: str = "Qwen/Qwen2.5-14B"):
    """
    Load the Qwen3 - 14b model and tokenizer.

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
        logging.info(f"Model loaded successfully from {model_path}")
        return model, tokenizer

    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
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
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                print(f"Debug: Our INPUT TEXT:\n{text}")
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


def call_llm(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 1024,
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

            generated_text = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            result = parse_json_from_text(generated_text)
            if result is not None:
                return result

            if attempt < max_retries:
                print(f"LLM call attempt {attempt + 1} failed, retrying...")

        except Exception as e:
            if attempt < max_retries:
                print(
                    f"LLM call attempt {attempt + 1} failed with error: {e}, retrying..."
                )
            else:
                print(f"LLM call failed after {max_retries + 1} attempts: {e}")
                return None

    print(f"LLM call failed after all {max_retries + 1} attempts")
    return None
