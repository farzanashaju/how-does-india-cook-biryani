import json
import os
import time
from datetime import datetime
from typing import List, Dict
from google import genai


def load_jobs(jobs_file: str = "batch_jobs.json") -> List[Dict]:
    """Load job tracking data."""
    if not os.path.exists(jobs_file):
        print(f"No jobs file found: {jobs_file}")
        return []

    with open(jobs_file, "r") as f:
        return json.load(f)


def update_job_status(
    jobs: List[Dict], job_name: str, new_status: str, result_info: Dict = None
) -> None:
    """Update job status in the jobs list."""
    for job in jobs:
        if job["job_name"] == job_name:
            job["status"] = new_status
            job["last_checked"] = datetime.now().isoformat()
            if result_info:
                job.update(result_info)
            break


def save_jobs(jobs: List[Dict], jobs_file: str = "batch_jobs.json") -> None:
    """Save updated job data."""
    with open(jobs_file, "w") as f:
        json.dump(jobs, f, indent=2)


def check_job_status(client, job_name: str) -> Dict:
    """Check status of a single job."""
    try:
        batch_job = client.batches.get(name=job_name)

        result = {"status": batch_job.state.name, "error": None, "result_file": None}

        if batch_job.state.name == "JOB_STATE_FAILED":
            result["error"] = str(batch_job.error) if batch_job.error else "Unknown error"

        elif batch_job.state.name == "JOB_STATE_SUCCEEDED":
            if batch_job.dest and batch_job.dest.file_name:
                result["result_file"] = batch_job.dest.file_name

        return result

    except Exception as e:
        return {"status": "ERROR", "error": str(e), "result_file": None}


def download_results(
    client, job_name: str, result_file: str, output_dir: str = "batch_results"
) -> str:
    """Download and save batch results."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Downloading results for {job_name}...")
        file_content = client.files.download(file=result_file)

        # Save results
        output_file = os.path.join(output_dir, f"{job_name.split('/')[-1]}_results.jsonl")
        with open(output_file, "wb") as f:
            f.write(file_content)

        print(f"Results saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"Error downloading results: {e}")
        return None


def print_job_summary(jobs: List[Dict]) -> None:
    """Print summary of all jobs."""
    print("\n" + "=" * 80)
    print("BATCH JOBS SUMMARY")
    print("=" * 80)

    if not jobs:
        print("No jobs found.")
        return

    status_counts = {}
    for job in jobs:
        status = job.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Total Jobs: {len(jobs)}")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    print("\nDETAILED STATUS:")
    print("-" * 80)

    for i, job in enumerate(jobs, 1):
        print(f"{i}. {job.get('display_name', 'Unnamed Job')}")
        print(f"   Job Name: {job.get('job_name', 'N/A')}")
        print(f"   Status: {job.get('status', 'UNKNOWN')}")
        print(f"   Created: {job.get('created_at', 'N/A')}")
        print(f"   Last Checked: {job.get('last_checked', 'Never')}")

        if job.get("error"):
            print(f"   Error: {job['error']}")

        if job.get("result_file"):
            print(f"   Results: {job['result_file']}")

        if job.get("downloaded_results"):
            print(f"   Downloaded: {job['downloaded_results']}")

        print()


def monitor_jobs(jobs_file: str = "batch_jobs.json", download_results_flag: bool = True) -> None:
    """Monitor all jobs and update their status."""
    client = genai.Client()
    jobs = load_jobs(jobs_file)

    if not jobs:
        return

    print("Checking job statuses...")
    updated = False

    for job in jobs:
        job_name = job.get("job_name")
        current_status = job.get("status")

        # Skip if already completed
        if current_status in [
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        ]:
            continue

        print(f"Checking: {job.get('display_name', job_name)}")

        status_info = check_job_status(client, job_name)
        new_status = status_info["status"]

        if new_status != current_status:
            print(f"  Status changed: {current_status} -> {new_status}")
            update_job_status(
                jobs,
                job_name,
                new_status,
                {
                    "error": status_info["error"],
                    "result_file": status_info["result_file"],
                },
            )
            updated = True

            # Download results if job succeeded
            if (
                new_status == "JOB_STATE_SUCCEEDED"
                and download_results_flag
                and status_info["result_file"]
            ):
                output_file = download_results(client, job_name, status_info["result_file"])
                if output_file:
                    update_job_status(
                        jobs, job_name, new_status, {"downloaded_results": output_file}
                    )
        else:
            print(f"  Status unchanged: {current_status}")

    if updated:
        save_jobs(jobs, jobs_file)
        print("Job tracking file updated.")

    print_job_summary(jobs)


def main():
    """Main execution function."""
    JOBS_FILE = "batch_jobs.json"

    # Check if we should download results
    download_flag = input("Download results for completed jobs? (y/n): ").lower().startswith("y")

    try:
        monitor_jobs(JOBS_FILE, download_flag)
    except Exception as e:
        print(f"Error monitoring jobs: {e}")


if __name__ == "__main__":
    main()
