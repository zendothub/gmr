#!/usr/bin/env python3
"""
repair_fragmented_billing_visits.py — dry-run / apply fragmented counter dwell repair.

Same logic as the periodic post-dedup job
`app.modules.jobs.tasks.repair_fragmented_billing_visits`:

  1. Fill null BI/event person_identity_id from track_sessions
  2. Same-camera body/face stitch of null billing-zone sessions (gap ≤ 30s)
  3. Group same person + camera sessions; sum max zone dwell; insert BI if needed

Default: dry-run (apply=False). Pass --apply to write.

Usage:
  PYTHONPATH=/gmr/gmr venv/bin/python danger/repair_fragmented_billing_visits.py
  PYTHONPATH=/gmr/gmr venv/bin/python danger/repair_fragmented_billing_visits.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modules.jobs.tasks import repair_fragmented_billing_visits


async def _run(apply: bool) -> None:
    stats = await repair_fragmented_billing_visits(apply=apply)
    print("=== billing visit repair ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if not apply:
        print("\nDry-run only. Re-run with --apply to write.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="Write updates (default dry-run)")
    args = p.parse_args()
    asyncio.run(_run(apply=args.apply))


if __name__ == "__main__":
    main()
