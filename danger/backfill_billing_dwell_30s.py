#!/usr/bin/env python3
"""Thin wrapper — prefer danger/backfill_billing_dwell.py --min-dwell 30."""
import runpy
import sys

sys.argv = [sys.argv[0], "--min-dwell", "30", *sys.argv[1:]]
runpy.run_path(
    str(__import__("pathlib").Path(__file__).with_name("backfill_billing_dwell.py")),
    run_name="__main__",
)
