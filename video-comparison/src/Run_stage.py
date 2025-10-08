import argparse
import os
import sys
import time
from datetime import datetime, timedelta
import logging

# Add stage directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), "stage_1"))
sys.path.append(os.path.join(os.path.dirname(__file__), "stage_2"))
sys.path.append(os.path.join(os.path.dirname(__file__), "stage_3"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


def setup_logging():
    """Configure logging for the entire application."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,  # Hard-coded to DEBUG level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),  # Also output to console
        ],
    )

    # logging.getLogger('transformers').setLevel(logging.WARNING)
    # logging.getLogger('torch').setLevel(logging.WARNING)

    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file


def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def print_timing_header(stage_name):
    """Print timing header for a stage."""
    print(f"\n⏱️  TIMING: {stage_name}")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)


def print_step_timing(step_name, duration):
    """Print timing for a specific step."""
    print(f"⏱️  {step_name}: {format_time(duration)}")


def print_total_timing(stage_name, total_duration):
    """Print total timing for a stage."""
    print("-" * 60)
    print(f"⏱️  TOTAL {stage_name}: {format_time(total_duration)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def validate_args(args):
    """Validate command line arguments."""
    start_time = time.time()

    # Check stage number
    if args.stage not in [1, 2, 3, 4]:
        raise ValueError(
            f"Stage {args.stage} is not implemented yet. Only stages 1-3 are currently supported."
        )

    # Check input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    validation_time = time.time() - start_time
    print_step_timing("Argument validation", validation_time)


def run_stage_1(input_file, output_file, model_path="Qwen/Qwen2.5-14b", max_tokens=1024):
    """Run Stage 1: Proposal generation."""
    stage_start_time = time.time()
    print_timing_header("Stage 1: Proposal Generation")

    # Import timing
    import_start = time.time()
    try:
        from stage_1.Proposer import ActionProposer
    except Exception as e:
        print(f"Error at stage 1 load {e}")
        raise ImportError
    print_step_timing("Import ActionProposer", time.time() - import_start)

    # Initialize timing
    init_start = time.time()
    proposer = ActionProposer(model_path=model_path, max_tokens=max_tokens)
    print_step_timing("Initialize ActionProposer", time.time() - init_start)

    # Pipeline timing
    pipeline_start = time.time()
    results = proposer.run_stage_1_pipeline(input_file, output_file)
    print_step_timing("Run Stage 1 pipeline", time.time() - pipeline_start)

    total_time = time.time() - stage_start_time
    print_total_timing("Stage 1", total_time)
    print(f"\n✨ Stage 1 complete! Results saved to: {output_file}")
    return results


def run_stage_2(
    input_file,
    output_file,
    model_name="ViT-bigG-14",
    dataset="laion2b_s39b_b160k",
    batch_size=100,
    top_k=2,
    target_fps=4,
    device=None,
):
    """Run Stage 2: Frame localization and retrieval."""

    stage_start_time = time.time()
    print_timing_header("Stage 2: Frame Localization and Retrieval")

    try:
        from stage_2.retriever import FrameRetriever
    except ImportError as e:
        raise ImportError("Failed to import FrameRetriever from stage_2.retriever") from e

    retriever = FrameRetriever(
        model_name=model_name,
        dataset=dataset,
        batch_size=batch_size,
        top_k=top_k,
        target_fps=target_fps,
        device=device,
    )

    pipeline_start = time.time()
    results = retriever.run_stage_2_pipeline(input_file, output_file)
    print_step_timing("Run Stage 2 pipeline", time.time() - pipeline_start)

    total_time = time.time() - stage_start_time
    print_total_timing("Stage 2", total_time)
    print(f"\n✨ Stage 2 complete! Results saved to: {output_file}")
    return results


def run_stage_3(
    input_file,
    output_file,
    model_name="gemini-2.5-flash-lite-preview-06-17",
):
    try:
        from stage_3.Diffrencer import GeminiDifferencer
    except ImportError as e:
        raise ImportError("Failed to import Differencer from stage_3.Diffrencer") from e

    Diffrencer = GeminiDifferencer(model_name=model_name)

    pipeline_start = time.time()

    results = Diffrencer.run_pipeline(input_file, output_file)
    print_step_timing("Run Stage 3 pipeline", time.time() - pipeline_start)

    print(f"\n✨ Stage 3 complete! Results saved to: {output_file}")
    return results


def run_stage_4(
    input_file,
    output_file,
    output_dir,
    model_name="gemini-2.5-flash-lite-preview-06-17",
):
    try:
        from stage_3.batch import GeminiDifferencerBatch
    except ImportError as e:
        raise ImportError("Failed to import GeminiDifferencerBatch from stage_3.batch") from e

    batcher = GeminiDifferencerBatch(model_name=model_name)

    pipeline_start = time.time()

    results = batcher.run_batch_pipeline(input_file, output_dir)
    print_step_timing("Run Stage 4 pipeline", time.time() - pipeline_start)
    print(f"\n✨ Stage 4 complete! Results saved to: {output_dir}")

    return results


def main():
    main_start_time = time.time()
    log_file = setup_logging()

    parser = argparse.ArgumentParser(
        description="Run biryani action analysis pipeline stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stage.py --stage 1 --input sampled_local.json --output sampled_local_stage_1.json
  python run_stage.py --stage 2 --input sampled_local_stage_1.json --output sampled_local_stage_2.json
  python run_stage.py --stage 3 --input sampled_local_stage_2.json --output sampled_local_stage_3.json
  python run_stage.py --stage 1 -i data/input.json -o data/output_stage1.json --model microsoft/DialoGPT-medium
        """,
    )

    parser.add_argument(
        "--stage", type=int, required=True, help="Pipeline stage to run (1, 2, or 3)"
    )

    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file")

    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output JSON file")

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens for model generation (default: 2048)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="final_output",
        help="output path for final output of stage 3",
    )

    args = parser.parse_args()

    print(f"🚀 Starting Stage {args.stage} execution")
    print(f"📂 Input: {args.input}")
    print(f"📂 Output: {args.output}")

    try:
        # Validate arguments with timing
        validate_args(args)

        # Run the appropriate stage with timing
        if args.stage == 1:
            results = run_stage_1(
                input_file=args.input,
                output_file=args.output,
                max_tokens=args.max_tokens,
            )
        elif args.stage == 2:
            results = run_stage_2(input_file=args.input, output_file=args.output)
        elif args.stage == 3:
            results = run_stage_3(
                input_file=args.input,
                output_file=args.output,
            )
        elif args.stage == 4:
            results = run_stage_4(
                input_file=args.input,
                output_file=args.output,
                output_dir=args.output_dir,
            )

        else:
            raise ValueError(f"Invalid stage number: {args.stage}")

        # Print total execution time
        total_execution_time = time.time() - main_start_time
        print(f"\n🎉 COMPLETE! Total execution time: {format_time(total_execution_time)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        total_execution_time = time.time() - main_start_time
        print(f"💥 Failed after: {format_time(total_execution_time)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
