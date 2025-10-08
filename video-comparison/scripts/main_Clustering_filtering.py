import logging
from data_loader import DataLoader
from clustering import ActionClusterer
from utils import print_statistics, print_clustering_results

from filtering import ActionFilter
from build_dataset import DatasetBuilder
import time


def main():
    """Main function to run the data loading and clustering pipeline."""

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("clustering_debug.log"), logging.StreamHandler()],
    )

    start_time = time.time()

    # data_loader = DataLoader(data_root="../ingredients_actions_utensils")
    # action_clusterer = ActionClusterer()
    #
    # logging.info("Loading data...")
    # data_loader.load_all_data()
    #
    # print_statistics(data_loader.get_action_counts(), "Before Clustering")
    #
    # logging.info("Starting action clustering...")
    #
    # cluster_time = time.time()
    # action_mapping, combination_info = action_clusterer.cluster_actions(
    #     data_loader.get_action_counts(), threshold=0.75
    # )
    # cluster_duration = time.time() - cluster_time
    # logging.debug(f"Clustering completed in {cluster_duration:.2f} seconds.")
    #
    # data_loader.update_actions(action_mapping)
    #
    # print_statistics(data_loader.get_action_counts(), "After Clustering")
    # print_clustering_results(combination_info)
    #
    # output_path = "clustered_ingredients_actions_utensils"
    # logging.info(f"Saving clustered data to {output_path}...")
    # data_loader.save_clustered_data(output_path, create_backup=True)
    #
    # logging.info("Clustering pipeline completed successfully!")

    logging.info("Loading clustered data...")

    data_loader = DataLoader(
        data_root="./remote/clustered_ingredients_actions_utensils"
    )

    action_filter = ActionFilter()
    dataset_builder = DatasetBuilder()

    logging.info("Loading data...")

    data_loader.load_all_data()

    print_statistics(data_loader.get_action_counts(), "Before Filtering")

    logging.info("Applying action filtering...")
    filtered_data, filter_stats = action_filter.apply_action_filtering(
        data_loader.get_data()
    )

    data_loader.data = filtered_data
    data_loader._recalculate_action_counts()

    print_statistics(data_loader.get_action_counts(), "After Filtering")

    logging.info("saving filtered data...")

    output_path = "filtered_ingredients_actions_utensils"
    data_loader.save_clustered_data(output_path, create_backup=True)

    logging.info("Building dataset...")

    dataset_builder.save_action_dataset(
        filtered_data,
        "action_dataset.json",
    )

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
