import argparse
import json
import os
import subprocess
import threading
from collections import defaultdict
from typing import Dict, Set, Tuple, List
from queue import Queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def safe_filename(url: str) -> str:
    """Extract video ID for consistent filenames"""
    return url.split("v=")[1].split("&")[0] + ".mp4"


def get_video_id(url: str) -> str:
    """Extract just the video ID from YouTube URL"""
    return url.split("v=")[1].split("&")[0]


def generate_clip_filename(video_id: str, start: int, end: int) -> str:
    """Generate consistent clip filename based on video ID and timestamps"""
    return f"{video_id}_{start}s_{end}s.mp4"


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe"""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def verify_clip(clip_path: str, expected_duration: int) -> Tuple[bool, str]:
    """Verify clip integrity and duration"""
    if not os.path.exists(clip_path):
        return False, "File does not exist"

    # Check if file is corrupted by getting basic info
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration,size",
        "-of",
        "csv=p=0",
        clip_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_str, size_str = result.stdout.strip().split(",")

        duration = float(duration_str)
        size = int(size_str)

        if size == 0:
            return False, "File is empty"

        # Allow 1 second tolerance for duration
        if abs(duration - expected_duration) > 1.0:
            return (
                False,
                f"Duration mismatch: expected {expected_duration}s, got {duration:.1f}s",
            )

        # Check if moov atom is properly placed (for web streaming)
        moov_check = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=format_name",
                "-of",
                "csv=p=0",
                clip_path,
            ],
            capture_output=True,
            text=True,
        )

        if "mp4" not in moov_check.stdout:
            return False, "Invalid MP4 format"

        return True, "Valid"

    except (subprocess.CalledProcessError, ValueError) as e:
        return False, f"Probe error: {str(e)}"


def download_video(url: str, output_dir: str) -> str:
    """Download full YouTube video with yt-dlp"""
    filename = safe_filename(url)
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        print(f"🔄 Downloading: {url}")
        cmd = [
            "yt-dlp",
            "-f",
            "bestvideo[ext=mp4][vcodec*=avc]+bestaudio[ext=m4a]/best[ext=mp4][vcodec*=avc]",
            "--merge-output-format",
            "mp4",
            "-o",
            filepath,
            url.split("&t=")[0],  # Remove any `&t=` param
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully downloaded: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading {url}: {e}")
            raise
    else:
        print(f"⏭️  Already downloaded: {filename}")

    return filepath


def trim_video_ffmpeg(full_video: str, start: int, end: int, output_path: str) -> None:
    """Trim video using FFmpeg with re-encoding for precision"""
    if os.path.exists(output_path):
        print(f"⏭️  Clip already exists: {os.path.basename(output_path)}")
        return

    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss",
        str(start),
        "-i",
        full_video,
        "-t",
        str(duration),
        "-c:v",
        "h264_nvenc",  # use NVIDIA encoder for speed
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-y",  # overwrite
        output_path,
    ]

    print(f"✂️  Trimming: {os.path.basename(output_path)} ({start}s–{end}s)")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Successfully created clip: {os.path.basename(output_path)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating clip {output_path}: {e}")
        # Try with software encoding as fallback
        print("🔄 Retrying with software encoding...")
        cmd[6] = "libx264"  # Replace h264_nvenc with libx264
        subprocess.run(cmd, check=True)


def trim_video_worker(args_tuple):
    """Worker function for multithreaded trimming"""
    full_video, start, end, output_path = args_tuple
    try:
        trim_video_ffmpeg(full_video, start, end, output_path)
        return output_path, None
    except Exception as e:
        return output_path, str(e)


def analyze_clips(
    data: Dict,
) -> Tuple[
    Dict[str, List[Tuple[int, int]]],
    Dict[str, str],
    Dict[Tuple[str, int, int], Set[Tuple[str, str]]],
]:
    """
    Analyze all clips and group by video ID.

    Returns:
        - video_to_clips: Maps video_id to list of (start, end) tuples
        - video_to_sample_url: Maps video_id to a sample URL for downloading
        - clip_to_actions: Maps (video_id, start, end) to set of (action, clip_id) tuples
    """
    video_to_clips = defaultdict(set)  # Use set to avoid duplicates
    video_to_sample_url = {}
    clip_to_actions = defaultdict(set)

    for action, details in data.items():
        for clip_id, clip in details["Clips"].items():
            url = clip["url"]
            video_id = get_video_id(url)
            start = clip["timestamp"]["start"]
            end = clip["timestamp"]["end"]

            video_to_clips[video_id].add((start, end))
            video_to_sample_url[video_id] = url

            clip_signature = (video_id, start, end)
            clip_to_actions[clip_signature].add((action, clip_id))

    # Convert sets to sorted lists for consistent processing
    video_to_clips = {video_id: sorted(list(clips)) for video_id, clips in video_to_clips.items()}

    return video_to_clips, video_to_sample_url, clip_to_actions


def print_analysis_stats(
    video_to_clips: Dict[str, List[Tuple[int, int]]],
    clip_to_actions: Dict[Tuple[str, int, int], Set[Tuple[str, str]]],
) -> None:
    """Print statistics about the analysis"""
    total_clips = sum(len(actions) for actions in clip_to_actions.values())
    unique_clips = len(clip_to_actions)
    total_videos = len(video_to_clips)

    print(f"\n=== Dataset Analysis ===")
    print(f"Total videos: {total_videos}")
    print(f"Total clip references: {total_clips}")
    print(f"Unique video segments: {unique_clips}")
    print(f"Duplicate clips avoided: {total_clips - unique_clips}")
    print(f"Storage savings: {(total_clips - unique_clips) / total_clips * 100:.1f}%")

    # Show clips per video distribution
    clips_per_video = [len(clips) for clips in video_to_clips.values()]
    print(f"Average clips per video: {sum(clips_per_video) / len(clips_per_video):.1f}")
    print(f"Max clips per video: {max(clips_per_video)}")
    print(f"Min clips per video: {min(clips_per_video)}")


def download_worker(download_queue: Queue, download_dir: str, results_queue: Queue):
    """Worker thread for downloading videos"""
    while True:
        item = download_queue.get()
        if item is None:  # Sentinel value to stop
            break

        video_id, url = item
        try:
            filepath = download_video(url, download_dir)
            results_queue.put((video_id, filepath, None))
        except Exception as e:
            results_queue.put((video_id, None, e))
        finally:
            download_queue.task_done()


def process_video_clips_multithreaded(
    video_id: str,
    full_video_path: str,
    clips: List[Tuple[int, int]],
    clips_dir: str,
    processed_clips: Dict,
    max_workers: int = 10,
) -> None:
    """Process all clips for a single video using multithreading"""
    print(f"\n🎬 Processing {len(clips)} clips for video: {video_id} (max workers: {max_workers})")

    # Prepare arguments for all clips
    trim_args = []
    clip_paths = {}

    for start, end in clips:
        clip_filename = generate_clip_filename(video_id, start, end)
        clip_path = os.path.join(clips_dir, clip_filename)

        if not os.path.exists(clip_path):
            trim_args.append((full_video_path, start, end, clip_path))
            clip_paths[(start, end)] = clip_path
        else:
            print(f"⏭️  Clip already exists: {clip_filename}")
            processed_clips[(video_id, start, end)] = clip_path

    if not trim_args:
        print(f"✅ All clips already exist for video: {video_id}")
        return

    # Process clips in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {executor.submit(trim_video_worker, args): args for args in trim_args}

        completed = 0
        for future in as_completed(future_to_clip):
            args = future_to_clip[future]
            clip_path, error = future.result()

            if error:
                print(f"❌ Error trimming {os.path.basename(clip_path)}: {error}")
            else:
                # Find the start, end for this clip_path
                for (start, end), path in clip_paths.items():
                    if path == clip_path:
                        processed_clips[(video_id, start, end)] = clip_path
                        break

            completed += 1
            print(f"  Progress: {completed}/{len(trim_args)} clips completed")

    print(f"✅ Completed all clips for video: {video_id}")


def verify_clip_worker(args_tuple):
    """Worker function for multithreaded clip verification"""
    clip_path, expected_duration, clip_filename = args_tuple
    try:
        is_valid, message = verify_clip(clip_path, expected_duration)
        return clip_filename, clip_path, is_valid, message, None
    except Exception as e:
        return clip_filename, clip_path, False, f"Worker error: {str(e)}", e


def verify_clips_only(clips_dir: str, data: Dict, max_workers: int = 10) -> None:
    """Verify all clips without downloading or processing - multithreaded version"""
    print(f"\n🔍 Verifying clips in: {clips_dir} (max workers: {max_workers})")

    if not os.path.exists(clips_dir):
        print(f"❌ Clips directory does not exist: {clips_dir}")
        return

    # Prepare all verification tasks
    verify_args = []

    for action, details in data.items():
        for clip_id, clip in details["Clips"].items():
            if "timestamp" not in clip:
                continue

            url = clip["url"]
            video_id = get_video_id(url)
            start = clip["timestamp"]["start"]
            end = clip["timestamp"]["end"]
            expected_duration = end - start

            clip_filename = generate_clip_filename(video_id, start, end)
            clip_path = os.path.join(clips_dir, clip_filename)

            verify_args.append((clip_path, expected_duration, clip_filename))

    total_clips = len(verify_args)
    if total_clips == 0:
        print("No clips to verify.")
        return

    print(f"🔄 Verifying {total_clips} clips...")

    valid_clips = 0
    invalid_clips = []
    completed = 0

    # Process clips in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(verify_clip_worker, args): args for args in verify_args}

        for future in as_completed(future_to_args):
            clip_filename, clip_path, is_valid, message, error = future.result()

            completed += 1

            if is_valid:
                valid_clips += 1
                print(f"✅ {clip_filename}: {message}")
            else:
                invalid_clips.append((clip_filename, message))
                # Delete the invalid clip file
                if os.path.exists(clip_path):
                    try:
                        os.remove(clip_path)
                        print(f"❌ {clip_filename}: {message}")
                        print(f"  - Removed invalid clip file: {clip_path}")
                    except OSError as e:
                        print(f"❌ {clip_filename}: {message}")
                        print(f"  - Failed to remove {clip_path}: {e}")
                else:
                    print(f"❌ {clip_filename}: {message}")

            # Progress update
            if completed % max(1, total_clips // 20) == 0 or completed == total_clips:
                print(
                    f"  Progress: {completed}/{total_clips} clips verified ({completed / total_clips * 100:.1f}%)"
                )

    print("\n=== Verification Summary ===")
    print(f"Total clips checked: {total_clips}")
    print(f"Valid clips: {valid_clips}")
    print(f"Invalid clips: {len(invalid_clips)}")
    print(f"Invalid clips removed: {len(invalid_clips)}")
    print(f"Success rate: {valid_clips / total_clips * 100:.1f}%")

    if invalid_clips:
        print(f"\n❌ Invalid clips:")
        for filename, reason in invalid_clips:
            print(f"  - {filename}: {reason}")


def download_only_mode(
    video_to_sample_url: Dict[str, str], download_dir: str, max_concurrent: int
) -> None:
    """Download all videos without processing clips"""
    print(f"\n📥 Download-only mode: Processing {len(video_to_sample_url)} videos")

    # Set up download queue and worker threads
    download_queue = Queue()
    results_queue = Queue()

    # Start download worker threads
    download_threads = []
    for i in range(max_concurrent):
        t = threading.Thread(
            target=download_worker,
            args=(download_queue, download_dir, results_queue),
        )
        t.start()
        download_threads.append(t)

    # Queue all videos for download
    for video_id, url in video_to_sample_url.items():
        download_queue.put((video_id, url))

    # Wait for all downloads to complete
    videos_processed = 0
    videos_to_process = len(video_to_sample_url)
    successful_downloads = 0

    while videos_processed < videos_to_process:
        try:
            video_id, full_video_path, error = results_queue.get(timeout=1)

            if error:
                print(f"❌ Failed to download {video_id}: {error}")
            else:
                successful_downloads += 1
                print(f"✅ Downloaded {video_id}")

            videos_processed += 1
            print(f"📊 Progress: {videos_processed}/{videos_to_process} videos processed")

        except:
            # Timeout - continue waiting
            continue

    # Stop download workers
    for _ in download_threads:
        download_queue.put(None)

    for t in download_threads:
        t.join()

    print(f"\n📥 Download Summary:")
    print(f"Total videos: {videos_to_process}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {videos_to_process - successful_downloads}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process video clips with optimized batch processing"
    )
    parser.add_argument("--input", type=str, default="sampled.json", help="Input JSON file")
    parser.add_argument(
        "--output", type=str, default="sampled_local.json", help="Output JSON file"
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="downloads",
        help="Directory for full videos",
    )
    parser.add_argument(
        "--clips_dir", type=str, default="clips", help="Directory for trimmed clips"
    )
    parser.add_argument("--dry_run", action="store_true", help="Analyze without downloading")
    parser.add_argument(
        "--max_concurrent_downloads",
        type=int,
        default=2,
        help="Maximum number of concurrent downloads",
    )
    parser.add_argument(
        "--max_clip_workers",
        type=int,
        default=10,
        help="Maximum number of concurrent clip processing workers",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download videos only, no clip processing",
    )
    parser.add_argument(
        "--verify_clips",
        action="store_true",
        help="Verify clips integrity and duration only",
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.clips_dir, exist_ok=True)

    # Load input data
    with open(args.input) as f:
        data = json.load(f)

    print(f"📂 Loaded {len(data)} actions from {args.input}")

    # Analyze clips
    video_to_clips, video_to_sample_url, clip_to_actions = analyze_clips(data)
    print_analysis_stats(video_to_clips, clip_to_actions)

    if args.dry_run:
        print("\n🔍 Dry run complete. No videos downloaded.")
        return

    if args.verify_clips:
        verify_clips_only(args.clips_dir, data, args.max_clip_workers)
        return

    if args.download_only:
        download_only_mode(video_to_sample_url, args.download_dir, args.max_concurrent_downloads)
        return

    # Original processing logic with multithreaded clipping
    # Set up download queue and worker threads
    download_queue = Queue()
    results_queue = Queue()
    processed_clips = {}

    # Start download worker threads
    download_threads = []
    for i in range(args.max_concurrent_downloads):
        t = threading.Thread(
            target=download_worker,
            args=(download_queue, args.download_dir, results_queue),
        )
        t.start()
        download_threads.append(t)

    # Queue all videos for download
    for video_id, url in video_to_sample_url.items():
        download_queue.put((video_id, url))

    print(f"\n🚀 Starting processing of {len(video_to_sample_url)} videos...")
    print(f"📥 Download threads: {args.max_concurrent_downloads}")
    print(f"✂️  Clip workers: {args.max_clip_workers}")

    videos_processed = 0
    videos_to_process = len(video_to_sample_url)

    # Process videos as they finish downloading
    while videos_processed < videos_to_process:
        try:
            video_id, full_video_path, error = results_queue.get(timeout=1)

            if error:
                print(f"❌ Failed to download {video_id}: {error}")
                videos_processed += 1
                continue

            # Process all clips for this video with multithreading
            clips = video_to_clips[video_id]
            process_video_clips_multithreaded(
                video_id,
                full_video_path,
                clips,
                args.clips_dir,
                processed_clips,
                args.max_clip_workers,
            )

            videos_processed += 1
            print(f"📊 Progress: {videos_processed}/{videos_to_process} videos completed")

        except:
            # Timeout - continue waiting
            continue

    # Stop download workers
    for _ in download_threads:
        download_queue.put(None)

    for t in download_threads:
        t.join()

    print(f"\n📝 Updating data structure...")

    # Update the data structure with new local paths
    clips_updated = 0
    for action, details in data.items():
        for clip_id, clip in details["Clips"].items():
            url = clip["url"]
            video_id = get_video_id(url)
            start = clip["timestamp"]["start"]
            end = clip["timestamp"]["end"]

            clip_signature = (video_id, start, end)

            if clip_signature in processed_clips:
                # Update to local path
                clip["url"] = processed_clips[clip_signature]
                # Remove timestamp as it's no longer needed
                clip.pop("timestamp", None)
                clips_updated += 1
            else:
                print(f"⚠️  Warning: No processed clip found for {action} clip {clip_id}")

    # Save updated data
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n🎉 === Summary ===")
    print(f"Videos processed: {videos_processed}")
    print(f"Clips updated: {clips_updated}")
    print(f"Unique clips created: {len(processed_clips)}")
    print(f"Updated JSON saved to: {args.output}")


if __name__ == "__main__":
    main()
