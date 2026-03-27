"""
Logging utilities.

Stores experiment logs (epoch, test_loss, elapsed) to CSV and JSON.
"""

import os
import json
import csv
from pathlib import Path


def save_log(log: list[dict], path: str):
    """
    Save a list of log dicts to both JSON and CSV.

    path : str  path without extension, e.g. "results/exp_001"
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path + ".json", "w") as f:
        json.dump(log, f, indent=2)

    if not log:
        return

    fieldnames = list(log[0].keys())
    with open(path + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)


def load_log(path: str) -> list[dict]:
    """Load from JSON (auto-adds .json if missing)."""
    if not path.endswith(".json"):
        path = path + ".json"
    with open(path) as f:
        return json.load(f)
