import json
import os
import random
import tempfile
from pathlib import Path
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from PIL import Image
import io


# Colors
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


def fmt_bytes(b):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def fmt_num(n):
    return f"{n:,}"


def load_registry(path="upload_registry.json"):
    if not os.path.exists(path):
        return {"uploaded": [], "extracted": []}
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {"uploaded": [], "extracted": []}


def get_local_stats(dir_path="extracted_frames"):
    if not os.path.exists(dir_path):
        return 0, 0, set()

    files = list(Path(dir_path).glob("*.png"))
    file_names = {f.name for f in files}
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    return len(files), total_size, file_names


def get_expected_frames_detailed(json_path):
    """Get expected frames with detailed breakdown."""
    if not os.path.exists(json_path):
        return 0, 0, set()

    try:
        with open(json_path) as f:
            data = json.load(f)

        all_frames = []
        unique_frames = set()

        for action_data in data.values():
            clips = action_data.get("Clips", {})
            for clip_data in clips.values():
                video_url = clip_data.get("url", "")
                retrieval_frames = clip_data.get("retrieval_frames", {})

                for frame_list in retrieval_frames.values():
                    for frame_num in frame_list:
                        all_frames.append(frame_num)
                        video_filename = Path(video_url).stem
                        frame_name = f"{video_filename}_frame_{frame_num:06d}.png"
                        unique_frames.add(frame_name)

        return len(all_frames), len(unique_frames), unique_frames
    except Exception as e:
        print(f"{C.R}Error reading JSON: {e}{C.END}")
        return 0, 0, set()


def get_cloud_stats_detailed(bucket="biryanidiff", creds="biryani-across-india-d26285fcddf7.json"):
    """Get cloud stats with frame names."""
    try:
        if os.path.exists(creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

        client = storage.Client()
        bucket_obj = client.bucket(bucket)

        sizes = []
        count = 0
        total_size = 0
        cloud_frames = set()

        print(f"{C.B}Scanning cloud bucket...{C.END}")
        for blob in bucket_obj.list_blobs():
            if blob.name.endswith(".png"):
                count += 1
                size = blob.size or 0
                sizes.append(size)
                total_size += size
                cloud_frames.add(blob.name)

                # Progress indicator
                if count % 5000 == 0:
                    print(f"{C.B}  Scanned {fmt_num(count)} files...{C.END}")

        avg_size = total_size / count if count > 0 else 0
        max_size = max(sizes) if sizes else 0

        return count, total_size, avg_size, max_size, cloud_frames, client, bucket_obj
    except Exception as e:
        print(f"{C.R}Cloud error: {e}{C.END}")
        return -1, 0, 0, 0, set(), None, None


def check_frame_corruption(client, bucket_obj, frame_names, sample_size=50):
    """Check random sample of frames for corruption."""
    if not client or not bucket_obj or not frame_names:
        return 0, 0, []

    sample_frames = random.sample(list(frame_names), min(sample_size, len(frame_names)))
    print(f"{C.B}Checking {len(sample_frames)} random frames for corruption...{C.END}")

    corrupted = []
    checked = 0

    for frame_name in sample_frames:
        try:
            blob = bucket_obj.blob(frame_name)

            # Download to memory
            image_data = blob.download_as_bytes()

            # Try to open with PIL
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    # Verify image can be loaded and has reasonable dimensions
                    width, height = img.size
                    if width < 10 or height < 10:
                        corrupted.append(frame_name)
                    elif width > 10000 or height > 10000:  # Suspiciously large
                        corrupted.append(frame_name)
                    else:
                        # Try to load pixel data (forces full decode)
                        img.load()

            except Exception as img_error:
                corrupted.append(frame_name)
                print(f"{C.R}  Corrupted: {frame_name} - {img_error}{C.END}")

            checked += 1

            # Progress
            if checked % 10 == 0:
                print(f"{C.B}  Checked {checked}/{len(sample_frames)}...{C.END}")

        except Exception as e:
            corrupted.append(frame_name)
            print(f"{C.R}  Error downloading {frame_name}: {e}{C.END}")

        checked += 1

    return checked, len(corrupted), corrupted


def analyze_frame_differences(expected_frames, cloud_frames):
    """Analyze differences between expected and cloud frames."""
    missing_frames = expected_frames - cloud_frames
    extra_frames = cloud_frames - expected_frames

    return missing_frames, extra_frames


def progress_bar(pct, width=30):
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    color = C.G if pct > 70 else C.Y if pct > 30 else C.R
    return f"{color}{bar}{C.END} {pct:.1f}%"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="upload_registry.json")
    parser.add_argument("--local-dir", default="extracted_frames")
    parser.add_argument("--bucket", default="biryanidiff")
    parser.add_argument("--credentials", default="biryani-across-india-d26285fcdff7.json")
    parser.add_argument("--input-json", help="Input JSON file for expected frames")
    parser.add_argument(
        "--check-corruption", action="store_true", help="Check for corrupted frames"
    )
    parser.add_argument(
        "--corruption-sample",
        type=int,
        default=100,
        help="Sample size for corruption check",
    )
    parser.add_argument("--show-missing", action="store_true", help="Show missing frame names")
    parser.add_argument("--show-extra", action="store_true", help="Show extra frame names")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.C}{'🍚 BIRYANI PIPELINE STATUS 🍚'.center(50)}{C.END}")
    print(f"{C.C}{'=' * 50}{C.END}\n")

    # Expected frames from JSON
    if args.input_json:
        total_frames, unique_frames, expected_frame_names = get_expected_frames_detailed(
            args.input_json
        )
        print(f"Expected Frames:    {C.B}{fmt_num(unique_frames):>8}{C.END} (unique)")
        print(f"Total Frame Refs:   {C.B}{fmt_num(total_frames):>8}{C.END}\n")

    # Registry stats
    registry = load_registry(args.registry)
    reg_uploaded = len(registry.get("uploaded", []))
    reg_extracted = len(registry.get("extracted", []))

    # Local stats
    local_files, local_size, local_frame_names = get_local_stats(args.local_dir)

    # Cloud stats with names
    (
        cloud_files,
        cloud_size,
        avg_size,
        max_size,
        cloud_frame_names,
        client,
        bucket_obj,
    ) = get_cloud_stats_detailed(args.bucket, args.credentials)

    # Display basic stats
    print(f"\n{C.BOLD}📊 STATS{C.END}")
    print(f"Registry Uploaded:  {C.G}{fmt_num(reg_uploaded):>8}{C.END}")
    print(
        f"Cloud Files:        {C.G if cloud_files >= 0 else C.R}{fmt_num(cloud_files):>8}{C.END}"
    )
    if cloud_files > 0:
        print(f"Cloud Size:         {C.G}{fmt_bytes(cloud_size):>8}{C.END}")
        print(f"Avg Image Size:     {C.G}{fmt_bytes(avg_size):>8}{C.END}")
        print(f"Largest Image:      {C.G}{fmt_bytes(max_size):>8}{C.END}")
    print(f"Registry Extracted: {C.C}{fmt_num(reg_extracted):>8}{C.END}")
    print(f"Local Files:        {C.M}{fmt_num(local_files):>8}{C.END}")
    print(f"Local Size:         {C.M}{fmt_bytes(local_size):>8}{C.END}")

    # Frame name analysis
    if args.input_json and expected_frame_names:
        missing_frames, extra_frames = analyze_frame_differences(
            expected_frame_names, cloud_frame_names
        )

        print(f"\n{C.BOLD}🔍 FRAME ANALYSIS{C.END}")
        print(f"Expected Frames:    {C.B}{fmt_num(len(expected_frame_names)):>8}{C.END}")
        print(f"Cloud Frames:       {C.G}{fmt_num(len(cloud_frame_names)):>8}{C.END}")
        print(
            f"Missing Frames:     {C.R if missing_frames else C.G}{fmt_num(len(missing_frames)):>8}{C.END}"
        )
        print(
            f"Extra Frames:       {C.Y if extra_frames else C.G}{fmt_num(len(extra_frames)):>8}{C.END}"
        )

        if args.show_missing and missing_frames:
            print(f"\n{C.R}Missing frames (first 20):{C.END}")
            for frame in list(missing_frames)[:20]:
                print(f"  {frame}")
            if len(missing_frames) > 20:
                print(f"  ... and {len(missing_frames) - 20} more")

        if args.show_extra and extra_frames:
            print(f"\n{C.Y}Extra frames (first 20):{C.END}")
            for frame in list(extra_frames)[:20]:
                print(f"  {frame}")
            if len(extra_frames) > 20:
                print(f"  ... and {len(extra_frames) - 20} more")

    # Corruption check
    if args.check_corruption and cloud_frame_names:
        print(f"\n{C.BOLD}🔍 CORRUPTION CHECK{C.END}")
        checked, corrupted_count, corrupted_frames = check_frame_corruption(
            client, bucket_obj, cloud_frame_names, args.corruption_sample
        )

        corruption_rate = (corrupted_count / checked) * 100 if checked > 0 else 0
        color = C.G if corruption_rate == 0 else C.Y if corruption_rate < 5 else C.R

        print(f"Checked:            {C.B}{fmt_num(checked):>8}{C.END}")
        print(f"Corrupted:          {color}{fmt_num(corrupted_count):>8}{C.END}")
        print(f"Corruption Rate:    {color}{corruption_rate:>7.2f}%{C.END}")

        if corrupted_frames:
            print(f"\n{C.R}Corrupted frames:{C.END}")
            for frame in corrupted_frames:
                print(f"  {frame}")

    # Progress vs expected
    if args.input_json and unique_frames > 0:
        completion_pct = (cloud_files / unique_frames) * 100 if cloud_files >= 0 else 0
        print(f"\n{C.BOLD}📈 COMPLETION{C.END}")
        print(f"Overall: {progress_bar(completion_pct)}")

    # Upload progress
    if reg_extracted > 0:
        upload_pct = (reg_uploaded / reg_extracted) * 100
        print(f"Upload:  {progress_bar(upload_pct)}")

    # Status
    status = "✅ COMPLETE" if local_files == 0 else f"⚠️  {fmt_num(local_files)} PENDING"
    color = C.G if local_files == 0 else C.Y if local_files < 1000 else C.R
    print(f"\n{C.BOLD}Status:{C.END} {color}{status}{C.END}")

    # Discrepancies
    if cloud_files >= 0 and abs(cloud_files - reg_uploaded) > 10:
        diff = cloud_files - reg_uploaded
        print(f"{C.Y}⚠️  Registry vs Cloud: {diff:+,} files{C.END}")

    if args.input_json and unique_frames > 0 and cloud_files >= 0:
        missing = unique_frames - cloud_files
        if missing != 0:
            print(f"{C.Y}⚠️  Expected vs Cloud: {missing:+,} files{C.END}")

    print(f"\n{C.C}{'=' * 50}{C.END}")


if __name__ == "__main__":
    main()
