def get_clustering_decision_prompt(actions: list) -> str:
    """
    Generate prompt to ask LLM whether a set of actions should be clustered into multiple groups.

    Args:
        actions: List of action strings to evaluate

    Returns:
        Formatted prompt string
    """
    actions_str = "\n".join([f"- {action}" for action in actions])

    prompt = f"""You are analyzing cooking actions for a biryani recipe classifier. Below is a set of {len(actions)} similar cooking actions that have been grouped together:

{actions_str}

Question: Should these actions be split into multiple distinct action classes, or are they similar enough to remain as one group?

Consider:
- Are there distinct cooking techniques or steps represented?
- Would separating them improve classification accuracy for biryani cooking?
- Are some actions fundamentally different despite semantic similarity?

Respond with a JSON object containing only:
{{
    "should_split": true/false,
}}"""

    return prompt


def get_clustering_labels_prompt(actions: list) -> str:
    """
    Generate prompt to ask LLM to cluster actions into subgroups and create labels.

    Args:
        actions: List of action strings to cluster

    Returns:
        Formatted prompt string
    """
    actions_str = "\n".join([f"- {action}" for action in actions])

    prompt = f"""You are creating action labels for a biryani cooking classifier. Below are {len(actions)} cooking actions that need to be clustered into distinct subgroups:

{actions_str}

Task: Group these actions into logical clusters and create a short, descriptive label for each cluster.

Requirements:
- Each cluster should represent a distinct cooking technique or step
- Labels should be concise action phrases
- All original actions must be assigned to exactly one cluster
- Create 2-4 clusters maximum

Respond with a JSON object:
{{
    "clusters": [
        {{
            "label": "short action label",
            "actions": ["action1", "action2", ...]
        }},
        {{
            "label": "another action label",
            "actions": ["action3", "action4", ...]
        }}
    ]
}}"""

    return prompt


def get_single_label_prompt(actions: list) -> str:
    """
    Generate prompt to ask LLM to create a single representative label for a group of actions.

    Args:
        actions: List of action strings to label

    Returns:
        Formatted prompt string
    """
    actions_str = "\n".join([f"- {action}" for action in actions])

    prompt = f"""You are creating action labels for a biryani cooking classifier. Below are {len(actions)} similar cooking actions that should be represented by a single label:

{actions_str}

Task: Create a short, descriptive label that best represents all these actions.

Requirements:
- Label should be a concise action phrase
- Should capture the common cooking technique or step
- Should be specific enough for biryani cooking context

Respond with a JSON object:
{{
    "label": "short action label",
}}"""

    return prompt
