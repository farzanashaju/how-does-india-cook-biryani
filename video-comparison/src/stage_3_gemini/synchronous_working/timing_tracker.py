import time
import logging
from datetime import datetime, timedelta


class TimingTracker:
    """Track timing statistics for processing."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.comparison_times = []
        self.start_time = None
        self.total_comparisons = 0
        self.processed_comparisons = 0

    def start_processing(self, total_comparisons: int):
        """Start timing the overall processing."""
        self.start_time = time.time()
        self.total_comparisons = total_comparisons
        self.processed_comparisons = 0
        self.comparison_times = []
        self.logger.info(f"⏱️  Started processing {total_comparisons} total comparisons")

    def record_comparison_time(self, comparison_key: str, elapsed_time: float):
        """Record time taken for a comparison."""
        self.comparison_times.append(elapsed_time)
        self.processed_comparisons += 1

        # Calculate statistics
        avg_time = sum(self.comparison_times) / len(self.comparison_times)
        remaining_comparisons = self.total_comparisons - self.processed_comparisons
        estimated_remaining_time = avg_time * remaining_comparisons

        # Calculate total elapsed time
        total_elapsed = time.time() - self.start_time if self.start_time else 0

        # Format times for display
        elapsed_str = self._format_duration(elapsed_time)
        avg_str = self._format_duration(avg_time)
        remaining_str = self._format_duration(estimated_remaining_time)
        total_elapsed_str = self._format_duration(total_elapsed)

        # Calculate completion percentage
        completion_pct = (self.processed_comparisons / self.total_comparisons) * 100

        # Estimate completion time
        if total_elapsed > 0:
            estimated_total_time = total_elapsed * (
                self.total_comparisons / self.processed_comparisons
            )
            eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
            eta_str = eta.strftime("%H:%M:%S")
        else:
            eta_str = "Unknown"

        self.logger.info(
            f"⏰ {comparison_key} completed in {elapsed_str} | "
            f"Progress: {self.processed_comparisons}/{self.total_comparisons} ({completion_pct:.1f}%) | "
            f"Avg: {avg_str} | Est. remaining: {remaining_str} | ETA: {eta_str}"
        )

        # Log milestone updates every 10 comparisons or at significant percentages
        if self.processed_comparisons % 10 == 0 or completion_pct in [
            25,
            50,
            75,
            90,
            95,
        ]:
            self.logger.info(
                f"📊 MILESTONE: {self.processed_comparisons}/{self.total_comparisons} comparisons complete "
                f"({completion_pct:.1f}%) | Total elapsed: {total_elapsed_str} | ETA: {eta_str}"
            )

    def finish_processing(self):
        """Log final timing statistics."""
        if not self.start_time or not self.comparison_times:
            return

        total_time = time.time() - self.start_time
        avg_time = sum(self.comparison_times) / len(self.comparison_times)
        min_time = min(self.comparison_times)
        max_time = max(self.comparison_times)

        self.logger.info(
            f"🏁 PROCESSING COMPLETE! "
            f"Total: {self._format_duration(total_time)} | "
            f"Avg per comparison: {self._format_duration(avg_time)} | "
            f"Min: {self._format_duration(min_time)} | "
            f"Max: {self._format_duration(max_time)} | "
            f"Processed: {self.processed_comparisons} comparisons"
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
