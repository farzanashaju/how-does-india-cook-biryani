import logging


def setup_logging(log_file: str = "biryani_processor.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# Default configuration values
DEFAULT_CONFIG = {
    "LOCAL_IMAGES_PATH": "", ## dir containing all the extracted frames
    "UPLOADED_FILES_JSON": "uploaded_frames.json",
    "MAX_STORAGE_GB": 18.0,
    "CHUNK_SIZE": 10000,
    "LARGE_FILE_THRESHOLD_MB": 50,
}
