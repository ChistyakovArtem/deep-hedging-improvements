from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from src.artifacts import touch_terminal
from src.nirvana_io import publish_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    seed_name = f"seed-{args.seed:03d}"
    config_dirs = sorted(
        path.parent
        for path in args.experiment_root.glob(
            f"heston-xi*/public/{seed_name}/config.toml"
        )
    )
    if len(config_dirs) != 2:
        raise RuntimeError(
            f"Expected exactly two xi configs for {seed_name}, got {len(config_dirs)}."
        )

    failures = []
    for config_dir in config_dirs:
        terminal_dir = config_dir
        if args.smoke:
            terminal_dir = Path("local") / "smoke" / config_dir
        if (terminal_dir / "DONE").exists():
            print(f"skip DONE {config_dir}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "bin.run_ntbn_baseline",
            str(config_dir),
            "--device",
            args.device,
        ]
        if args.smoke:
            command.append("--smoke")
        print(f"run {' '.join(command)}", flush=True)
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
    if failures:
        raise RuntimeError(f"{len(failures)} config(s) failed: {failures}")


if __name__ == "__main__":
    main()
