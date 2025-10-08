import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from utils import normalize_action_title, call_llm
from prompts import (
    get_clustering_decision_prompt,
    get_clustering_labels_prompt,
    get_single_label_prompt,
)
from tqdm import tqdm
import json


class ActionClusterer:
    """Handles clustering of similar actions using sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the action clusterer.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.llm_model = None
        self.llm_tokenizer = None

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            logging.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def _load_llm_model(self) -> None:
        """Load the LLM model and tokenizer."""
        if self.llm_model is None or self.llm_tokenizer is None:
            logging.info("Loading LLM model...")
            from utils import load_model_and_tokenizer

            self.llm_model, self.llm_tokenizer = load_model_and_tokenizer()

    def _save_debug_clusters(
        self, clusters: Dict[int, List[str]], filename: str = "debug_clusters.json"
    ) -> None:
        """
        Save clustering results to debug file

        Args:
            clusters: Dictionary mapping cluster IDs to action lists
            filename: Name of debug file to save
        """
        if logging.getLogger().level == logging.DEBUG:
            debug_data = {}
            for cluster_id, actions in clusters.items():
                debug_data[f"cluster_{cluster_id}"] = actions

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)

            logging.debug(f"Debug clusters saved to {filename}")

    def _save_intermediate_results(
        self,
        action_mapping: Dict[str, str],
        combination_info: Dict[str, List[str]],
        filename: str = "intermediate_clustered_actions.json",
    ) -> None:
        """
        Save intermediate clustering results during processing.

        Args:
            action_mapping: Dictionary mapping old action names to new representative names
            combination_info: Dictionary mapping representative names to lists of combined actions
            filename: Name of file to save intermediate results
        """
        intermediate_data = {
            "action_mapping": action_mapping,
            "combination_info": combination_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

        logging.debug(f"Intermediate results saved to {filename}")

    def _process_large_cluster_with_llm(
        self, cluster_actions: List[str], action_counts: Counter
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Process a large cluster using LLM for refinement.

        Args:
            cluster_actions: List of actions in the large cluster
            action_counts: Counter of action occurrences

        Returns:
            Tuple of (action_mapping, combination_info) for this cluster
        """
        start_time = time.time()
        logging.info(
            f"Processing large cluster with {len(cluster_actions)} actions using LLM"
        )

        # Load LLM if not already loaded
        self._load_llm_model()

        # Step 1: Ask if cluster should be split
        decision_prompt = get_clustering_decision_prompt(cluster_actions)
        logging.debug(
            f"Sending clustering decision prompt for {len(cluster_actions)} actions"
        )

        decision_response = call_llm(
            self.llm_model, self.llm_tokenizer, decision_prompt, max_tokens=512
        )

        if decision_response is None:
            logging.warning("LLM decision call failed, treating as single cluster")
            should_split = False
        else:
            should_split = decision_response.get("should_split", False)
            logging.info(
                f"LLM decision: {'Split' if should_split else 'Keep together'}"
            )

        action_mapping = {}
        combination_info = {}

        if should_split:
            # Step 2: Get clustering labels
            clustering_prompt = get_clustering_labels_prompt(cluster_actions)
            logging.debug(f"Sending clustering labels prompt")

            clustering_response = call_llm(
                self.llm_model, self.llm_tokenizer, clustering_prompt, max_tokens=1024
            )

            if clustering_response is None or "clusters" not in clustering_response:
                logging.warning(
                    "LLM clustering call failed, falling back to single label"
                )
                should_split = False
            else:
                # Process the clusters
                clusters = clustering_response["clusters"]
                logging.info(f"LLM created {len(clusters)} sub-clusters")

                for cluster in clusters:
                    label = cluster["label"]
                    actions = cluster["actions"]

                    # Validate that all actions are in the original list
                    valid_actions = [
                        action for action in actions if action in cluster_actions
                    ]

                    if valid_actions:
                        # Map all actions to the label
                        for action in valid_actions:
                            action_mapping[action] = label

                        # Store combination info
                        combination_info[label] = valid_actions

                        logging.debug(
                            f"Created sub-cluster '{label}' with {len(valid_actions)} actions"
                        )

                # Handle any unmapped actions
                unmapped_actions = [
                    action for action in cluster_actions if action not in action_mapping
                ]
                if unmapped_actions:
                    logging.warning(
                        f"Found {len(unmapped_actions)} unmapped actions, assigning to fallback cluster"
                    )
                    fallback_representative = self._find_representative_action(
                        unmapped_actions, action_counts
                    )
                    for action in unmapped_actions:
                        action_mapping[action] = fallback_representative
                    if len(unmapped_actions) > 1:
                        combination_info[fallback_representative] = unmapped_actions

        if not should_split:
            # Step 3: Get single label
            single_label_prompt = get_single_label_prompt(cluster_actions)
            logging.debug(f"Sending single label prompt")

            label_response = call_llm(
                self.llm_model, self.llm_tokenizer, single_label_prompt, max_tokens=512
            )

            if label_response is None or "label" not in label_response:
                logging.warning("LLM single label call failed, using fallback")
                representative = self._find_representative_action(
                    cluster_actions, action_counts
                )
            else:
                representative = label_response["label"]
                logging.info(f"LLM created single label: '{representative}'")

            # Map all actions to representative
            for action in cluster_actions:
                action_mapping[action] = representative

            # Store combination info
            combination_info[representative] = cluster_actions

        elapsed_time = time.time() - start_time
        logging.debug(
            f"Large cluster processing completed in {elapsed_time:.2f} seconds"
        )

        return action_mapping, combination_info

    def cluster_actions(
        self, action_counts: Counter, threshold: float = 0.75
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Cluster similar actions using sentence embeddings and cosine similarity.

        Args:
            action_counts: Counter of action occurrences
            threshold: Similarity threshold for clustering (0.0 to 1.0)

        Returns:
            Tuple of (action_mapping, combination_info)
            - action_mapping: Dict mapping old action names to new representative names
            - combination_info: Dict mapping representative names to lists of combined actions
        """
        actions = list(action_counts.keys())
        n = len(actions)

        if n == 0:
            return {}, {}

        logging.info(f"Processing {n} unique actions with threshold {threshold}")

        # Load model if not already loaded
        self._load_model()

        # Normalize actions for better similarity matching
        normalized_actions = [normalize_action_title(action) for action in actions]

        # Generate embeddings
        logging.info("Generating embeddings...")
        embeddings = self.model.encode(normalized_actions, convert_to_tensor=True)

        # Calculate similarity matrix
        logging.info("Calculating similarities...")
        cosine_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
        distance_matrix = 1 - cosine_sim_matrix

        # Perform clustering
        logging.info("Clustering similar actions...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Group actions by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(actions[idx])

        # Save debug clusters
        self._save_debug_clusters(clusters)

        # Create mappings
        action_mapping = {}
        combination_info = {}

        # Identify large clusters that need LLM processing
        large_clusters = []
        small_clusters = []

        for cluster_id, cluster_actions in clusters.items():
            if len(cluster_actions) > 5:
                large_clusters.append(cluster_actions)
            else:
                small_clusters.append(cluster_actions)

        logging.info(
            f"Found {len(large_clusters)} large clusters (>5 actions) and {len(small_clusters)} small clusters"
        )

        # Process small clusters normally
        for cluster_actions in small_clusters:
            if len(cluster_actions) == 1:
                # Single action, map to itself
                action_mapping[cluster_actions[0]] = cluster_actions[0]
            else:
                # Multiple actions, find representative
                representative = self._find_representative_action(
                    cluster_actions, action_counts
                )

                # Map all actions to representative
                for action in cluster_actions:
                    action_mapping[action] = representative

                # Store combination info
                combination_info[representative] = cluster_actions

        # Process large clusters with LLM
        if large_clusters:
            logging.info(f"Processing {len(large_clusters)} large clusters with LLM...")

            # Create progress bar
            progress_bar = tqdm(
                large_clusters, desc="Processing large clusters", unit="cluster"
            )

            for i, cluster_actions in enumerate(progress_bar):
                progress_bar.set_description(
                    f"Processing cluster {i + 1}/{len(large_clusters)} ({len(cluster_actions)} actions)"
                )

                start_time = time.time()
                cluster_mapping, cluster_combination_info = (
                    self._process_large_cluster_with_llm(cluster_actions, action_counts)
                )
                elapsed_time = time.time() - start_time

                # Update mappings
                action_mapping.update(cluster_mapping)
                combination_info.update(cluster_combination_info)

                # Save intermediate results
                self._save_intermediate_results(action_mapping, combination_info)

                if logging.getLogger().level == logging.DEBUG:
                    logging.debug(
                        f"Cluster {i + 1} processed in {elapsed_time:.2f} seconds"
                    )

                progress_bar.set_postfix(
                    {"Time": f"{elapsed_time:.2f}s", "Actions": len(cluster_actions)}
                )

        logging.info(
            f"Clustering completed. Combined {len(actions)} actions "
            f"into {len(set(action_mapping.values()))} clusters."
        )

        return action_mapping, combination_info

    def _find_representative_action(
        self, cluster_actions: List[str], action_counts: Counter
    ) -> str:
        """
        Find the most representative action from a cluster.

        Args:
            cluster_actions: List of actions in the cluster
            action_counts: Counter of action occurrences

        Returns:
            Representative action name
        """
        # First, try to find by most common normalized form
        normalized_actions = [
            normalize_action_title(action) for action in cluster_actions
        ]
        normalized_counter = Counter(normalized_actions)
        most_common_normalized = normalized_counter.most_common(1)[0][0]

        # Find the original action with this normalized form
        for action in cluster_actions:
            if normalize_action_title(action) == most_common_normalized:
                return action

        # Fallback: return the action with highest count
        return max(cluster_actions, key=lambda x: action_counts[x])


    '''
    This works in the following way:

    1. Load the sentence transformer model for generating embeddings. 

    2. Normalize action titles to improve similarity matching. 

    3. Generate embeddings for all actions and calculate cosine similarity matrix. 

    4. Perform agglomerative clustering on the similarity matrix to group similar actions. 

    5. For small clusters (<=5 actions), find a representative action and map all actions to it. 

    6. For large clusters (>5 actions), use LLM to decide whether to split the cluster or keep it together. 

    7. If splitting, use LLM to generate labels for sub-clusters and map actions accordingly. 

    '''
