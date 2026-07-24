from __future__ import annotations

import argparse
from pathlib import Path

import tomli_w

from dev.generate_stage1_configs import LEARNED_ROOT
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
    graphs = []
    relative_root = LEARNED_ROOT.relative_to(PROJECT_ROOT)
    for seed in range(8):
        seed_dir = relative_root / f"seed-{seed:03d}"
        command = (
            "uv run --no-sync python -m bin.run_pending_seed "
            f"{seed_dir} --device cuda --keep-going"
        )
        graphs.append(
            {
                **PROFILE,
                "comment": (
                    "ICAIF26-DeepHedger-Stage1-HestonXi03-"
                    f"Seed{seed}-11Configs-TabularNormal-Jul24"
                ),
                "main_commands": command,
            }
        )
    return graphs


def validate_graphs(graphs: list[dict]) -> None:
    if len(graphs) != 8:
        raise RuntimeError(f"Expected eight graphs, got {len(graphs)}.")
    if sum(int(graph["n_gpus"]) for graph in graphs) != 8:
        raise RuntimeError("The full volley must occupy exactly eight GPUs.")
    if len({graph["comment"] for graph in graphs}) != 8:
        raise RuntimeError("Nirvana comments must be unique.")
    for seed, graph in enumerate(graphs):
        required = [
            f"seed-{seed:03d}",
            "--device cuda",
            "--keep-going",
        ]
        if any(item not in graph["main_commands"] for item in required):
            raise RuntimeError(f"Malformed command: {graph['main_commands']}")
        if graph["priority"] != "normal":
            raise RuntimeError("Full runs were explicitly requested at normal priority.")
        if graph["job_scheduler_yt_pool"] != "nirvana-yr-tabular":
            raise RuntimeError("Full runs must use the tabular scheduler pool.")
        if graph["data_ids"] or graph["data_commands"] != "mkdir -p data":
            raise RuntimeError("Stage 1 must generate its frozen leaderboards locally.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/nirvana/icaif26-stage1-8x1gpu-tabular-normal.toml"),
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
