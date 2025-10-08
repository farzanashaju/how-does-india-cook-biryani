import os
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import tempfile
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError, TooManyRequests, RetryError
import requests
from urllib.parse import urlparse


# Colors for output
class C:
    R, G, Y, B, M, C, W = (
        "\033[91m",
        "\033[92m",
        "\033[93m",
        "\033[94m",
        "\033[95m",
        "\033[96m",
        "\033[97m",
    )
    BOLD, END = "\033[1m", "\033[0m"


class DownloadStats:
    def __init__(self):
        self.lock = Lock()
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.total_bytes = 0
        self.start_time = time.time()
        self.last_report = time.time()

    def update(self, downloaded=0, failed=0, skipped=0, bytes_downloaded=0):
        with self.lock:
            self.downloaded += downloaded
            self.failed += failed
            self.skipped += skipped
            self.total_bytes += bytes_downloaded

    def report_progress(self, force=False):
        with self.lock:
            now = time.time()
            if not force and now - self.last_report < 2:  # Report every 2 seconds
                return

            self.last_report = now
            elapsed = now - self.start_time
            total_files = self.downloaded + self.failed + self.skipped

            if elapsed > 0:
                rate = self.downloaded / elapsed
                mb_rate = (self.total_bytes / (1024 * 1024)) / elapsed
                print(
                    f"\r{C.B}Progress:{C.END} {total_files:,} files | "
                    f"{C.G}Downloaded:{C.END} {self.downloaded:,} | "
                    f"{C.R}Failed:{C.END} {self.failed:,} | "
                    f"{C.Y}Skipped:{C.END} {self.skipped:,} | "
                    f"{C.C}Rate:{C.END} {rate:.1f}/s ({mb_rate:.1f} MB/s)",
                    end="",
                )


def fmt_bytes(b):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def download_blob_with_retry(blob, local_path, max_retries=3, base_delay=1):
    """Download a blob with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            # Create directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download to temporary file first, then move
            temp_path = local_path.with_suffix(local_path.suffix + ".tmp")

            blob.download_to_filename(str(temp_path))
            temp_path.rename(local_path)

            return True, blob.size or 0

        except TooManyRequests as e:
            delay = base_delay * (2**attempt) + (time.time() % 1)  # Add jitter
            print(f"\n{C.Y}Rate limit hit for {blob.name}, retrying in {delay:.1f}s{C.END}")
            time.sleep(delay)

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n{C.R}Failed to download {blob.name}: {e}{C.END}")
                return False, 0

            delay = base_delay * (2**attempt)
            time.sleep(delay)

    return False, 0


def download_worker(blob_info, download_dir, skip_existing, stats):
    """Worker function for downloading a single blob."""
    blob, blob_name, blob_size = blob_info
    local_path = Path(download_dir) / blob_name

    # Skip if file exists and skip_existing is True
    if skip_existing and local_path.exists():
        if local_path.stat().st_size == (blob_size or 0):
            stats.update(skipped=1)
            return f"Skipped: {blob_name}"

    # Download the file
    success, bytes_downloaded = download_blob_with_retry(blob, local_path)

    if success:
        stats.update(downloaded=1, bytes_downloaded=bytes_downloaded)
        return f"Downloaded: {blob_name}"
    else:
        stats.update(failed=1)
        return f"Failed: {blob_name}"


def get_blob_list(bucket_name, credentials_path, prefix=""):
    """Get list of all blobs in bucket."""
    if credentials_path and os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = []
    print(f"{C.B}Scanning bucket for .png files...{C.END}")

    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".png") or blob.name.endswith(".jpg") or blob.name.endswith(".jpeg"):
            blobs.append((blob, blob.name, blob.size))
            if len(blobs) % 1000 == 0:
                print(f"\r{C.B}Found {len(blobs):,} images...{C.END}", end="")

    print(f"\n{C.G}Found {len(blobs):,} images to download{C.END}")
    return blobs


def main():
    parser = argparse.ArgumentParser(
        description="Download images from Google Cloud Storage bucket in parallel"
    )
    parser.add_argument("--bucket", default="biryanidiff", help="GCS bucket name")
    parser.add_argument(
        "--credentials",
        default="biryani-across-india-d26285fcddf7.json",
        help="Path to GCS credentials JSON file",
    )
    parser.add_argument("--download-dir", required=True, help="Local directory to download images")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel download workers (default: 10)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally",
    )
    parser.add_argument("--prefix", default="", help="Only download blobs with this prefix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )

    args = parser.parse_args()

    # Validate download directory
    download_dir = Path(args.download_dir)
    if not args.dry_run:
        download_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{C.BOLD}{C.C}{'🍚 BIRYANI IMAGE DOWNLOADER 🍚'.center(60)}{C.END}")
    print(f"{C.C}{'=' * 60}{C.END}\n")

    print(f"Bucket: {C.B}{args.bucket}{C.END}")
    print(f"Download Dir: {C.B}{download_dir.absolute()}{C.END}")
    print(f"Workers: {C.B}{args.workers}{C.END}")
    print(f"Skip Existing: {C.B}{args.skip_existing}{C.END}")
    if args.prefix:
        print(f"Prefix Filter: {C.B}{args.prefix}{C.END}")
    print()

    try:
        # Get list of all blobs
        blob_list = get_blob_list(args.bucket, args.credentials, args.prefix)

        if not blob_list:
            print(f"{C.Y}No images found in bucket{C.END}")
            return

        # Calculate total size
        total_size = sum(size or 0 for _, _, size in blob_list)
        print(f"Total size to download: {C.G}{fmt_bytes(total_size)}{C.END}")

        if args.dry_run:
            print(f"{C.Y}DRY RUN - Would download {len(blob_list):,} files{C.END}")
            return

        # Check available space
        try:
            free_space = os.statvfs(download_dir).f_frsize * os.statvfs(download_dir).f_bavail
            if total_size > free_space:
                print(
                    f"{C.R}Warning: Not enough free space. Need {fmt_bytes(total_size)}, have {fmt_bytes(free_space)}{C.END}"
                )
                response = input("Continue anyway? (y/N): ")
                if response.lower() != "y":
                    return
        except:
            pass  # Skip space check on systems where it's not available

        # Initialize stats
        stats = DownloadStats()

        print(f"\n{C.G}Starting download with {args.workers} workers...{C.END}\n")

        # Download files in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all download tasks
            future_to_blob = {
                executor.submit(
                    download_worker, blob_info, download_dir, args.skip_existing, stats
                ): blob_info[1]
                for blob_info in blob_list
            }

            # Process completed downloads
            for future in as_completed(future_to_blob):
                blob_name = future_to_blob[future]
                try:
                    result = future.result()
                    stats.report_progress()
                except Exception as e:
                    print(f"\n{C.R}Unexpected error with {blob_name}: {e}{C.END}")
                    stats.update(failed=1)

        # Final report
        stats.report_progress(force=True)
        print(f"\n\n{C.BOLD}Download Complete!{C.END}")
        print(f"Downloaded: {C.G}{stats.downloaded:,}{C.END}")
        print(f"Failed: {C.R}{stats.failed:,}{C.END}")
        print(f"Skipped: {C.Y}{stats.skipped:,}{C.END}")
        print(f"Total Data: {C.C}{fmt_bytes(stats.total_bytes)}{C.END}")

        elapsed = time.time() - stats.start_time
        if elapsed > 0:
            avg_rate = stats.downloaded / elapsed
            mb_rate = (stats.total_bytes / (1024 * 1024)) / elapsed
            print(f"Average Rate: {C.C}{avg_rate:.1f} files/sec ({mb_rate:.1f} MB/s){C.END}")

        if stats.failed > 0:
            print(
                f"\n{C.Y}Consider re-running with --skip-existing to retry failed downloads{C.END}"
            )

    except KeyboardInterrupt:
        print(f"\n{C.Y}Download interrupted by user{C.END}")
    except Exception as e:
        print(f"\n{C.R}Error: {e}{C.END}")


if __name__ == "__main__":
    main()
