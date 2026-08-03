"""Standalone background worker — runs APScheduler with all periodic jobs.

This process is SEPARATE from the FastAPI API server to avoid blocking
the event loop during heavy DB/MinIO operations (dedup, sweep, etc.).

No GPU models, no camera workers, no HTTP endpoints — just scheduled jobs.

Jobs registered:
  - deduplicate_persons     every 10 min  (merge duplicates, absorb embeddings, re-vote gender;
                            then repair_fragmented_billing_visits: null BI fill + visit dwell sum)
  - close_stale_track_sessions  every 5 min
  - probe_camera_statuses   every 2 min
  - aggregate_daily_analytics  daily 00:15
  - cleanup_old_storage     daily 02:00

Starts as a systemd service: retail-ai-worker.service
"""

import asyncio
from loguru import logger

# Configure the same rotating file sink as the API server so background job
# activity (dedup, sweep, staff classification, ...) is visible in
# logs/ai_processing.log instead of only in the systemd journal.
from app.logging_config import setup_logging

setup_logging()


async def main():
    logger.info("=" * 60)
    logger.info("  Retail AI Background Worker — starting")
    logger.info("=" * 60)

    from app.modules.jobs.scheduler import start_scheduler, stop_scheduler

    scheduler = start_scheduler()
    logger.info("Background worker running. All jobs registered.")
    logger.info("Press Ctrl+C to stop.")

    try:
        # Keep the event loop alive
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down background worker...")
        stop_scheduler()
        logger.info("Background worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
