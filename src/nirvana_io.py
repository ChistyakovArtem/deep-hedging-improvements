from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from src.config import PROJECT_ROOT


def _external_root(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return None if not value else Path(value).resolve()


def _relative_experiment(directory: Path) -> Path:
    directory = directory.resolve()
    try:
        relative = directory.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Experiment is outside the project: {directory}") from error
    if not relative.parts or relative.parts[0] != "exp":
        raise ValueError(f"Only exp/ artifacts may be published: {directory}")
    return relative


def publish_experiment(directory: Path, *, snapshot: bool = True) -> None:
    """Copy one experiment to the operation output and resumable snapshot."""
    relative = _relative_experiment(directory)
    for variable in ("TMP_OUTPUT_PATH", "SNAPSHOT_PATH"):
        if variable == "SNAPSHOT_PATH" and not snapshot:
            continue
        root = _external_root(variable)
        if root is None:
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(directory, destination, dirs_exist_ok=True)

    if snapshot and _external_root("SNAPSHOT_PATH") is not None:
        try:
            import nirvana_dl.snapshot

            nirvana_dl.snapshot.dump_snapshot()
        except Exception as error:  # the TMP output remains recoverable
            print(f"WARNING: Nirvana snapshot update failed: {error}", flush=True)


def restore_snapshot() -> dict[str, int]:
    """Restore terminal/running experiment directories from a prior snapshot."""
    snapshot_root = _external_root("SNAPSHOT_PATH")
    counts = {"restored": 0, "skipped_incompatible": 0}
    if snapshot_root is None or not snapshot_root.exists():
        return counts

    snapshot_exp = snapshot_root / "exp"
    if not snapshot_exp.exists():
        return counts
    for snapshot_config in snapshot_exp.rglob("config.toml"):
        source = snapshot_config.parent
        if not any((source / marker).exists() for marker in ("DONE", "FAILED", "RUNNING")):
            continue
        relative = source.relative_to(snapshot_root)
        destination = PROJECT_ROOT / relative
        destination_config = destination / "config.toml"
        if (
            destination_config.exists()
            and destination_config.read_bytes() != snapshot_config.read_bytes()
        ):
            counts["skipped_incompatible"] += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        counts["restored"] += 1
    return counts


def write_operation_metadata() -> None:
    if not os.environ.get("JSON_OUTPUT_FILE"):
        return
    try:
        import nirvana_dl
    except ModuleNotFoundError:
        return
    output = Path(nirvana_dl.json_output_file())
    payload = {}
    if output.exists():
        payload = json.loads(output.read_text())
    payload.update(
        {
            "branch": os.environ.get("NIRVANA_GIT_BRANCH"),
            "commit": os.environ.get("NIRVANA_GIT_COMMIT")
            or subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
