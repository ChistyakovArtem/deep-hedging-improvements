from __future__ import annotations

import argparse
from pathlib import Path

import tomli_w

from dev.generate_pfhedge_official import (
    EXPERIMENT_ROOT,
    PFHEDGE_WHEEL,
    SEEDS,
)
from src.config import PROJECT_ROOT
from src.pfhedge_baselines import PFHEDGE_VERSION, PFHEDGE_WHEEL_SHA256


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
    "venv_commands": "\n".join(
        [
            "uv sync --no-build-isolation",
            f'echo "{PFHEDGE_WHEEL_SHA256}  {PFHEDGE_WHEEL}" | sha256sum -c -',
            (
                "VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT "
                f"uv pip install --no-deps {PFHEDGE_WHEEL}"
            ),
            (
                "uv run --no-sync python -c "
                f"\"import pfhedge; assert pfhedge.__version__ == '{PFHEDGE_VERSION}'\""
            ),
        ]
    ),
}


def build_graphs(smoke: bool = False) -> list[dict]:
    relative_root = EXPERIMENT_ROOT.relative_to(PROJECT_ROOT)
    seeds = (0,) if smoke else SEEDS
    graphs = []
    for seed in seeds:
        suffix = " --smoke --max-configs 2" if smoke else ""
        comment_kind = "Smoke2Configs" if smoke else "4TrainableConfigs"
        graphs.append(
            {
                **PROFILE,
                "comment": (
                    f"ICAIF26-PFHedgeOfficial-{comment_kind}-"
                    f"Seed{seed}-TabularNormal-Jul24-v1"
                ),
                "main_commands": (
                    "uv run --no-sync python -m bin.run_pfhedge_official_seed "
                    f"{relative_root} {seed} --device cuda --keep-going{suffix}"
                ),
            }
        )
    return graphs


def validate_graphs(graphs: list[dict], smoke: bool = False) -> None:
    expected = 1 if smoke else 8
    if len(graphs) != expected:
        raise RuntimeError(f"Expected {expected} graphs, got {len(graphs)}.")
    if sum(int(graph["n_gpus"]) for graph in graphs) != expected:
        raise RuntimeError("Each PFHedge queue must occupy exactly one GPU.")
    if len({graph["comment"] for graph in graphs}) != expected:
        raise RuntimeError("Nirvana comments must be unique.")
    for seed, graph in enumerate(graphs):
        required = [
            "bin.run_pfhedge_official_seed",
            f" {seed} ",
            "--device cuda",
            "--keep-going",
        ]
        if smoke:
            required += ["--smoke", "--max-configs 2"]
        if any(item not in graph["main_commands"] for item in required):
            raise RuntimeError(f"Malformed command: {graph['main_commands']}")
        if graph["priority"] != "normal":
            raise RuntimeError("PFHedge runs require tabular-normal.")
        if graph["job_scheduler_yt_pool"] != "nirvana-yr-tabular":
            raise RuntimeError("The tabular scheduler pool is required.")
        if PFHEDGE_WHEEL_SHA256 not in graph["venv_commands"]:
            raise RuntimeError("The official wheel must be hash-verified.")
        if "VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install" not in graph[
            "venv_commands"
        ]:
            raise RuntimeError("The wheel must be installed into Nirvana's project venv.")
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
            "icaif26-pfhedge-official-smoke.toml"
            if args.smoke
            else "icaif26-pfhedge-official-8x1gpu-tabular-normal.toml"
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
