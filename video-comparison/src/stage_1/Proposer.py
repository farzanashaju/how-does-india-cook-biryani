import json
import os
import logging

from stage_1.prompts import PROMPT_DIFFERENCES, PROMPT_LINKING, PROMPT_STAGES
from utils import (
    call_llm,
    load_model_and_tokenizer,
    save_json,
)


class ActionProposer:
    """
    Proposes differences and stages for actions in a 3-stage pipeline.

    Stage 1: Propose differences - ways the action can vary
    Stage 2: Propose stages - break action into sub-actions
    Stage 3: Link differences to stages - connect variations to specific stages
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-14b", max_tokens: int = 2048):
        """
        Initialize the ActionProposer with model and tokenizer.

        Args:
            model_path: Path to the Hugging Face model
            max_tokens: Maximum tokens for generation
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.model, self.tokenizer = load_model_and_tokenizer(model_path)
        self.device = self.model.device
        logging.info(
            f"Loaded model {model_path} with max tokens {max_tokens} on device {self.device}"
        )

    def _propose_differences(self, action_type: str) -> dict | None:
        """
        Generate potential differences for a given action.

        Args:
            action_type: The action to analyze

        Returns:
            dict: dictionary of proposed differences
        """
        logging.debug(f"Processing {action_type} -> proposed differences")

        prompt = PROMPT_DIFFERENCES.format(action=action_type)
        differences = call_llm(self.model, self.tokenizer, prompt, self.max_tokens, self.device)

        if differences:
            logging.debug(f"Found differences for {action_type}")

        return differences

    def _propose_stages(self, action_type: str) -> list:
        """
        Generate sub-action stages for the action.

        Args:
            action_type: The action to analyze

        Returns:
            list: List of stage dictionaries
        """
        logging.debug(f"Processing {action_type} -> proposed stages")

        prompt = PROMPT_STAGES.format(action=action_type)
        stages_json = call_llm(self.model, self.tokenizer, prompt, self.max_tokens, self.device)

        stages = stages_json.get("stages", []) if stages_json else []
        if stages:
            logging.debug(f"Found stages for {action_type}")

        return stages

    def _link_differences_to_stages(
        self, action_type: str, differences: dict, stages: list
    ) -> dict | None:
        """
        Link the proposed differences to the appropriate sub steps (stages) of the action.

        Args:
            action_type: The action being analyzed
            differences: dictionary of differences from stage 1
            stages: List of stages from stage 2

        Returns:
            dict: Mapping of stage names to associated differences
        """
        logging.debug(f"Linking differences to stages for {action_type}")

        diff_str = json.dumps(differences, indent=2)
        stages_str = json.dumps({"stages": stages}, indent=2)

        prompt = PROMPT_LINKING.format(action=action_type, differences=diff_str, stages=stages_str)
        links = call_llm(self.model, self.tokenizer, prompt, self.max_tokens, self.device)

        if links:
            logging.debug(f"Linked differences to stages for {action_type}")

        return links

    def process_single_action(self, action_type: str) -> dict | None:
        """
        Process a single action through the complete 3-stage of proposal

        Args:
            action_type: The action to process

        Returns:
            dict: Complete analysis results for the action
        """
        try:
            # 1. Get possible diffrences
            differences = self._propose_differences(action_type)
            if not differences:
                logging.warning(f"No differences proposed for {action_type}, skipping")
                return None

            # 2, Get sub steps to perform the action (the actual action not the proposed differences)
            stages = self._propose_stages(action_type)
            if not stages:
                logging.warning(f"No stages proposed for {action_type}, skipping")
                return None

            # 3, Link them together, and find out which diffrences is most significant for each stage
            links = self._link_differences_to_stages(action_type, differences, stages)
            if not links:
                logging.warning(f"No links found for {action_type}, skipping")
                return None

            stage_names_with_links = set(links.keys())
            for stage in stages:
                stage_name = stage.get("name")
                if stage_name in stage_names_with_links:
                    stage["associated_differences"] = links[stage_name]
                else:
                    stage["associated_differences"] = []
                    logging.warning(
                        f"Stage '{stage_name}' was not found in the linking output for {action_type}"
                    )

            result = {
                "action_type": action_type,
                "proposed_differences": differences,
                "action_stages": stages,
            }

            logging.info(f"Processed action '{action_type}' successfully")
            return result

        except Exception as e:
            logging.error(f"Error processing action '{action_type}': {e}")
            return None

    def load_data(self, input_filepath: str) -> dict:
        """
        Args:
            input_filepath: Path to the input JSON file

        Returns:
            dict: Loaded JSON data

        Raises:
            FileNotFoundError: If the input file does not exist
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Input file not found: {input_filepath}")

        with open(input_filepath) as f:
            data = json.load(f)

        return data

    def is_action_complete(self, action_data: dict) -> bool:
        """
        Check if an action has been successfully processed.

        Args:
            action_data: The action data dictionary

        Returns:
            bool: True if action is complete and valid, False otherwise
        """
        # Check if both required fields exist and are not empty
        proposed_differences = action_data.get("proposed_differences")
        action_stages = action_data.get("action_stages")

        if not proposed_differences or not action_stages:
            return False

        # Additional validation: check if stages have required structure
        if isinstance(action_stages, list):
            for stage in action_stages:
                if not isinstance(stage, dict) or not stage.get("name"):
                    return False
        else:
            return False

        return True

    def run_stage_1_pipeline(self, input_filepath: str, output_filepath: str) -> dict:
        """
        Run the proposer stage with enhanced error handling and retry logic.

        Args:
            input_filepath: Path to the input Action JSON file
            output_filepath: Path to save the processed results

        Returns:
            dict: Processed data with proposed differences and stages
        """
        data = self.load_data(input_filepath)

        # Load existing progress if output file exists
        if os.path.exists(output_filepath):
            try:
                with open(output_filepath) as f:
                    existing_data = json.load(f)
                logging.info(f"Loaded existing progress from {output_filepath}")

                # Update data with existing progress
                for action_type in existing_data:
                    if action_type in data:
                        data[action_type].update(existing_data[action_type])
            except Exception as e:
                logging.warning(f"Could not load existing progress: {e}")

        action_types = list(data.keys())
        logging.info(f"Loaded {len(action_types)} actions from {input_filepath}")

        # Enhanced filtering: check for completion AND validity
        remaining_actions = []
        completed_actions = []
        failed_actions = []

        for action_type in action_types:
            if self.is_action_complete(data[action_type]):
                completed_actions.append(action_type)
            else:
                # Check if this was previously attempted but failed
                if (
                    data[action_type].get("proposed_differences") is not None
                    or data[action_type].get("action_stages") is not None
                ):
                    failed_actions.append(action_type)
                    logging.info(
                        f"Action '{action_type}' has incomplete/invalid data - will retry"
                    )

                remaining_actions.append(action_type)

        logging.info(f"Completed actions: {len(completed_actions)}/{len(action_types)}")
        logging.info(f"Failed actions (will retry): {len(failed_actions)}")
        logging.info(f"New actions to process: {len(remaining_actions) - len(failed_actions)}")
        logging.info(f"Total remaining: {len(remaining_actions)}")

        # Process remaining actions (includes both new and failed ones)
        for i, action_type in enumerate(remaining_actions, 1):
            current_index = len(completed_actions) + i

            if action_type in failed_actions:
                logging.info(
                    f"Retrying failed action: {action_type} ({current_index}/{len(action_types)})"
                )
            else:
                logging.info(
                    f"Processing new action: {action_type} ({current_index}/{len(action_types)})"
                )

            try:
                result = self.process_single_action(action_type)
                if result is not None:
                    data[action_type]["proposed_differences"] = result["proposed_differences"]
                    data[action_type]["action_stages"] = result["action_stages"]

                    # Verify the result is actually complete
                    if self.is_action_complete(data[action_type]):
                        save_json(data, output_filepath)
                        logging.info(f"Successfully processed and saved {action_type}")
                    else:
                        logging.warning(f"Action {action_type} completed but validation failed")
                else:
                    logging.warning(f"Skipping {action_type} - no valid result returned")

            except Exception as e:
                logging.error(f"Error processing action '{action_type}': {e}")
                continue

        # Final summary
        final_completed = sum(
            1 for action_type in action_types if self.is_action_complete(data[action_type])
        )

        logging.info("Stage 1 pipeline complete!")
        logging.info(f"Final status: {final_completed}/{len(action_types)} actions completed")

        if final_completed < len(action_types):
            remaining = len(action_types) - final_completed
            logging.warning(f"{remaining} actions still incomplete/failed")

        return data
