import json
import os
import time
from datetime import datetime
from google.cloud import storage
import vertexai
from vertexai.batch_prediction import BatchPredictionJob


def upload_to_gcs(local_file: str, bucket_name: str, blob_name: str) -> str:
    """Upload file to GCS and return URI."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_file)
    return f"gs://{bucket_name}/{blob_name}"


def submit_vertex_batch_job(
    jsonl_file_path: str,
    project_id: str,
    input_bucket: str = "testbatchbiryanidiff",
    output_bucket: str = "testbatchbiryanidiff/idk",
    job_name: str = None,
) -> dict:
    """Submit batch job using Vertex AI."""

    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")

    # Initialize Vertex AI
    vertexai.init(project=project_id, location="us-central1")

    # Generate job name if not provided
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(jsonl_file_path).replace(".jsonl", "")
        job_name = f"{filename}_{timestamp}"

    # Upload JSONL to GCS
    input_blob_name = f"batch_inputs/{job_name}.jsonl"
    print(f"Uploading {jsonl_file_path} to gs://{input_bucket}/{input_blob_name}")

    input_uri = upload_to_gcs(jsonl_file_path, input_bucket, input_blob_name)
    output_uri = f"gs://{output_bucket}/{job_name}/"

    print(f"Input URI: {input_uri}")
    print(f"Output URI: {output_uri}")

    # Submit batch prediction job
    print("Submitting batch prediction job...")
    batch_job = BatchPredictionJob.submit(
        source_model="gemini-2.5-flash-lite",
        input_dataset=input_uri,
        output_uri_prefix=output_uri,
        job_display_name=job_name,
    )

    job_info = {
        "job_name": batch_job.name,
        "job_id": batch_job.name.split("/")[-1],
        "display_name": job_name,
        "input_uri": input_uri,
        "output_uri": output_uri,
        "created_at": datetime.now().isoformat(),
        "status": batch_job.state.name,
        "project_id": project_id,
        "jsonl_file": jsonl_file_path,
    }

    print(f"Batch job submitted: {batch_job.name}")
    print(f"Job ID: {batch_job.name.split('/')[-1]}")

    return job_info


def save_job_info(job_info: dict, jobs_file: str = "vertex_batch_jobs.json"):
    """Save job info to tracking file."""
    jobs_list = []

    if os.path.exists(jobs_file):
        with open(jobs_file, "r") as f:
            jobs_list = json.load(f)

    jobs_list.append(job_info)

    with open(jobs_file, "w") as f:
        json.dump(jobs_list, f, indent=2)

    print(f"Job info saved to {jobs_file}")


def main():
    """Main execution function."""
    # Configuration
    PROJECT_ID = "biryani-across-india"  # Update with your project ID
    JSONL_FILE = "biryani_comparisons_batch_1.jsonl"
    JOB_NAME = "biryani_test_batch_1"
    INPUT_BUCKET = "testbatchbiryanidiff"
    OUTPUT_BUCKET = "testbatchbiryanidiff/idk"
    JOBS_TRACKING_FILE = "vertex_batch_jobs.json"

    try:
        job_info = submit_vertex_batch_job(
            jsonl_file_path=JSONL_FILE,
            project_id=PROJECT_ID,
            input_bucket=INPUT_BUCKET,
            output_bucket=OUTPUT_BUCKET,
            job_name=JOB_NAME,
        )

        save_job_info(job_info, JOBS_TRACKING_FILE)

        print("\n" + "=" * 50)
        print("VERTEX AI BATCH JOB SUBMITTED")
        print("=" * 50)
        print(f"Job Name: {job_info['job_name']}")
        print(f"Job ID: {job_info['job_id']}")
        print(f"Status: {job_info['status']}")
        print(f"Input: {job_info['input_uri']}")
        print(f"Output: {job_info['output_uri']}")
        print("\nUse vertex_monitor.py to check status.")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
