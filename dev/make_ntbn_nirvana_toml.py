from __future__ import annotations

import argparse
from pathlib import Path

import tomli_w

from dev.generate_ntbn_baseline import EXPERIMENT_ROOT, SEEDS
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


def build_graphs() -> list[dict]:
    relative_root = EXPERIMENT_ROOT.relative_to(PROJECT_ROOT)
    return [
        {
            **PROFILE,
            "comment": (
                "ICAIF26-NTBN-HestonXi01Xi03-"
                f"Seed{seed}-2Configs-TabularNormal-Jul24"
            ),
            "main_commands": (
                "uv run --no-sync python -m bin.run_ntbn_seed "
                f"{relative_root} {seed} --device cuda --keep-going"
            ),
        }
        for seed in SEEDS
    ]


def validate_graphs(graphs: list[dict]) -> None:
    if len(graphs) != 8:
        raise RuntimeError(f"Expected eight graphs, got {len(graphs)}.")
    if sum(int(graph["n_gpus"]) for graph in graphs) != 8:
        raise RuntimeError("The NTBN volley must occupy exactly eight GPUs.")
    if len({graph["comment"] for graph in graphs}) != 8:
        raise RuntimeError("Nirvana comments must be unique.")
    for seed, graph in enumerate(graphs):
        required = [
            "bin.run_ntbn_seed",
            f" {seed} ",
            "--device cuda",
            "--keep-going",
        ]
        if any(item not in graph["main_commands"] for item in required):
            raise RuntimeError(f"Malformed command: {graph['main_commands']}")
        if graph["n_gpus"] != 1:
            raise RuntimeError("Each NTBN graph must use one GPU.")
        if graph["priority"] != "normal":
            raise RuntimeError("NTBN full runs require tabular-normal.")
        if graph["job_scheduler_yt_pool"] != "nirvana-yr-tabular":
            raise RuntimeError("The tabular scheduler pool is required.")
        if graph["data_ids"] or graph["data_commands"] != "mkdir -p data":
            raise RuntimeError("Leaderboards must be generated deterministically in-job.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/nirvana/icaif26-ntbn-8x1gpu-tabular-normal.toml"),
    )
    args = parser.parse_args()
    graphs = build_graphs()
    validate_graphs(graphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        tomli_w.dump({"graphs": graphs}, file)
    print(f"wrote {len(graphs)} one-GPU graphs to {args.output}")


if __name__ == "__main__":
    main()
