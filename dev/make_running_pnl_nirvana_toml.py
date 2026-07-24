from __future__ import annotations

import argparse
from pathlib import Path

import tomli_w

from dev.generate_running_pnl import EXPERIMENT_ROOT, SEEDS
from src.config import PROJECT_ROOT


PROFILE = {
    "workflow_id": "f0068707-2efe-4041-8eff-753ad855ca1b",
    "data_ids": [],
    "data_commands": "mkdir -p data",
    "layer_id": "b32d8346-8370-46f4-8014-0a1349f62b2f",
    "n_gpus": 1,
    "n_cpu_cores": 2,
    "priority": "normal",
    "quota": "yr-other",
    "job_scheduler_instance": "watt",
    "job_scheduler_yt_pool": "nirvana-yr-tabular",
    "pool_tree": ["gpu_tesla_a100_80g"],
}


def build_graphs(smoke: bool = False) -> list[dict]:
    relative_root = EXPERIMENT_ROOT.relative_to(PROJECT_ROOT)
    seeds = (0,) if smoke else SEEDS
    return [
        {
            **PROFILE,
            "comment": (
                "ICAIF26-RunningPnL-SelectedResidual-2Markets-"
                f"Seed{seed}-TabularNormal-Jul24-v1"
                + ("-Smoke" if smoke else "")
            ),
            "main_commands": (
                "uv run --no-sync python -m bin.run_running_pnl_seed "
                f"{relative_root} {seed} --device cuda --keep-going"
                + (" --smoke" if smoke else "")
            ),
        }
        for seed in seeds
    ]


def validate_graphs(graphs: list[dict], smoke: bool = False) -> None:
    expected = 1 if smoke else 8
    if len(graphs) != expected:
        raise RuntimeError(f"Expected {expected} graphs, got {len(graphs)}.")
    if sum(int(graph["n_gpus"]) for graph in graphs) != expected:
        raise RuntimeError("Each running-PnL queue must use exactly one GPU.")
    if len({graph["comment"] for graph in graphs}) != expected:
        raise RuntimeError("Nirvana comments must be unique.")
    for seed, graph in enumerate(graphs):
        required = [
            "bin.run_running_pnl_seed",
            f" {seed} ",
            "--device cuda",
            "--keep-going",
        ]
        if smoke:
            required.append("--smoke")
        if any(item not in graph["main_commands"] for item in required):
            raise RuntimeError(f"Malformed command: {graph['main_commands']}")
        if graph["priority"] != "normal":
            raise RuntimeError("Full running-PnL runs require tabular-normal.")
        if graph["job_scheduler_yt_pool"] != "nirvana-yr-tabular":
            raise RuntimeError("The tabular scheduler pool is required.")
        if graph["data_ids"] or graph["data_commands"] != "mkdir -p data":
            raise RuntimeError("Leaderboards must be generated deterministically in-job.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output
    if output is None:
        filename = (
            "icaif26-running-pnl-smoke.toml"
            if args.smoke
            else "icaif26-running-pnl-8x1gpu-tabular-normal.toml"
        )
        output = Path("local/nirvana") / filename
    graphs = build_graphs(smoke=args.smoke)
    validate_graphs(graphs, smoke=args.smoke)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        tomli_w.dump({"graphs": graphs}, file)
    print(f"wrote {len(graphs)} graph(s) to {output}")


if __name__ == "__main__":
    main()
