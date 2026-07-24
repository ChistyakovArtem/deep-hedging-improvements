from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from src.artifacts import touch_terminal
from src.nirvana_io import (
    publish_experiment,
    restore_snapshot,
    write_operation_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    restored = restore_snapshot()
    print(f"snapshot restore: {restored}", flush=True)
    seed_name = f"seed-{args.seed:03d}"
    config_dirs = sorted(
        path.parent
        for path in args.experiment_root.glob(
            f"heston-xi*/public/{seed_name}/*/hp-*/config.toml"
        )
    )
    expected = 50
    if len(config_dirs) != expected:
        raise RuntimeError(
            f"Expected {expected} optimizer configs for {seed_name}, "
            f"got {len(config_dirs)}."
        )
    if args.smoke:
        config_dirs = [
            path for path in config_dirs if path.name.startswith("hp-00-")
        ]
        if len(config_dirs) != 10:
            raise RuntimeError(
                f"Expected ten smoke representatives, got {len(config_dirs)}."
            )

    failures = []
    for index, config_dir in enumerate(config_dirs, start=1):
        terminal_dir = config_dir
        if args.smoke:
            terminal_dir = Path("local") / "smoke" / config_dir
        if (terminal_dir / "DONE").exists():
            print(f"skip DONE {config_dir}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "bin.run_optimizer_study",
            str(config_dir),
            "--device",
            args.device,
        ]
        if args.smoke:
            command.append("--smoke")
        print(f"run {index}/{len(config_dirs)} {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, check=True)
        except BaseException:
            terminal_dir.mkdir(parents=True, exist_ok=True)
            (terminal_dir / "failure.txt").write_text(traceback.format_exc())
            touch_terminal(terminal_dir, "FAILED")
            if not args.smoke:
                publish_experiment(terminal_dir)
            failures.append(str(config_dir))
            if not args.keep_going:
                raise
    write_operation_metadata()
    if failures:
        raise RuntimeError(f"{len(failures)} config(s) failed: {failures}")


if __name__ == "__main__":
    main()
