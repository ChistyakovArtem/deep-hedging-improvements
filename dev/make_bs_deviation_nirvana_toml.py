from __future__ import annotations

import argparse
from pathlib import Path

import tomli_w

from dev.generate_bs_deviation import configs
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
    for directory, payload in configs():
        relative = directory.relative_to(PROJECT_ROOT)
        market = str(payload["experiment"]["market"])
        seed = int(payload["seed"])
        graphs.append(
            {
                **PROFILE,
                "comment": (f"ICAIF26-BSDeviation-{market}-Seed{seed}-TabularNormal-Jul24"),
                "main_commands": (
                    f"uv run --no-sync python -m bin.run_bs_deviation {relative} --device cuda"
                ),
            }
        )
    return graphs


def validate_graphs(graphs: list[dict]) -> None:
    if len(graphs) != 8:
        raise RuntimeError(f"Expected eight graphs, got {len(graphs)}.")
    if sum(int(graph["n_gpus"]) for graph in graphs) != 8:
        raise RuntimeError("The BS-deviation volley must request eight GPUs.")
    if len({graph["comment"] for graph in graphs}) != 8:
        raise RuntimeError("Nirvana comments must be unique.")
    commands = [str(graph["main_commands"]) for graph in graphs]
    if len(set(commands)) != 8:
        raise RuntimeError("Every graph must run a distinct config.")
    for graph in graphs:
        if graph["n_gpus"] != 1:
            raise RuntimeError("Each graph must use one GPU.")
        if graph["priority"] != "normal":
            raise RuntimeError("Full runs require tabular-normal.")
        if graph["job_scheduler_yt_pool"] != "nirvana-yr-tabular":
            raise RuntimeError("The tabular scheduler pool is required.")
        if "bin.run_bs_deviation" not in graph["main_commands"]:
            raise RuntimeError(f"Malformed command: {graph['main_commands']}")
        if graph["data_ids"] or graph["data_commands"] != "mkdir -p data":
            raise RuntimeError("Leaderboards must be generated deterministically in-job.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/nirvana/icaif26-bs-deviation-8x1gpu-tabular-normal.toml"),
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
