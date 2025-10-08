import json
import os
from datetime import datetime
from google import genai
from google.genai import types
import dotenv

dotenv.load_dotenv()


def upload_batch_request(jsonl_file_path: str, job_name: str = None) -> dict:
    """
    Upload JSONL file and create batch job.
    Returns job info for tracking.
    """
    client = genai.Client()

    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")

    # Generate job name if not provided
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(jsonl_file_path).replace(".jsonl", "")
        job_name = f"{filename}_{timestamp}"

    print(f"Uploading file: {jsonl_file_path}")

    # Upload the JSONL file
    uploaded_file = client.files.upload(
        file=jsonl_file_path,
        config=types.UploadFileConfig(
            display_name=f"batch-requests-{job_name}", mime_type="application/jsonl"
        ),
    )

    print(f"Uploaded file: {uploaded_file.name}")

    # Create batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
        model="gemini-2.0-flash-exp",  # Use latest model
        src=uploaded_file.name,
        config={
            "display_name": job_name,
        },
    )

    job_info = {
        "job_name": batch_job.name,
        "display_name": job_name,
        "uploaded_file": uploaded_file.name,
        "created_at": datetime.now().isoformat(),
        "status": batch_job.state.name,
        "jsonl_file": jsonl_file_path,
    }

    print(f"Created batch job: {batch_job.name}")
    return job_info


def save_job_info(job_info: dict, jobs_file: str = "batch_jobs.json"):
    """Save job info to tracking file."""
    jobs_list = []

    # Load existing jobs if file exists
    if os.path.exists(jobs_file):
        with open(jobs_file, "r") as f:
            jobs_list = json.load(f)

    # Add new job
    jobs_list.append(job_info)

    # Save updated list
    with open(jobs_file, "w") as f:
        json.dump(jobs_list, f, indent=2)

    print(f"Job info saved to {jobs_file}")


def main():
    """Main execution function."""
    # Configuration
    JSONL_FILE = "biryani_comparisons_batch_1.jsonl"
    JOB_NAME = "biryani_test_batch_1"
    JOBS_TRACKING_FILE = "batch_jobs.json"

    try:
        # Upload and create batch job
        job_info = upload_batch_request(JSONL_FILE, JOB_NAME)

        # Save job info for monitoring
        save_job_info(job_info, JOBS_TRACKING_FILE)

        print("\n" + "=" * 50)
        print("BATCH JOB CREATED SUCCESSFULLY")
        print("=" * 50)
        print(f"Job Name: {job_info['job_name']}")
        print(f"Display Name: {job_info['display_name']}")
        print(f"Status: {job_info['status']}")
        print(f"Created: {job_info['created_at']}")
        print("\nUse monitor_batch.py to check status and retrieve results.")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
