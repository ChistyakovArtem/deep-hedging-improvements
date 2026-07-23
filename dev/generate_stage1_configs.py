from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json, atomic_write_toml, write_config_safely
from src.config import DEFAULT_PROJECT_CONFIG, PROJECT_ROOT, load_project_config


LEARNED_ROOT = (
    PROJECT_ROOT
    / "exp"
    / "DeepHedger-Stage1-v1"
    / "heston-xi03"
    / "split-public"
)
MATH_ROOT = (
    PROJECT_ROOT
    / "exp"
    / "HestonMathDeltas-v1"
    / "heston-xi03"
    / "frozen-public-private"
)


def _project_hash() -> str:
    return hashlib.sha256(DEFAULT_PROJECT_CONFIG.read_bytes()).hexdigest()


def learned_configs(project: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    training = project["training"]
    search = project["search"]
    configs: list[tuple[Path, dict[str, Any]]] = []
    for seed in search["seeds"]:
        candidates = [("mlp", None)]
        candidates.extend(("paf_shared", sigma) for sigma in search["paf_sigmas"])
        candidates.extend(
            ("paf_featurewise", sigma) for sigma in search["paf_sigmas"]
        )
        if len(candidates) != 11:
            raise RuntimeError(f"Expected 11 candidates per seed, got {len(candidates)}.")

        for config_id, (architecture, sigma) in enumerate(candidates):
            payload: dict[str, Any] = {
                "seed": int(seed),
                "experiment": {
                    "name": "DeepHedger-Stage1-v1",
                    "search": "heston-xi03-public-guided-stage1",
                    "market": "heston-xi03",
                    "split": "public",
                    "config_id": config_id,
                    "model": "DeepHedger",
                    "project_config": "config/project.toml",
                    "project_config_sha256": _project_hash(),
                },
                "model": {
                    "architecture": architecture,
                    "hidden_dims": list(training["hidden_dims"]),
                    "n_frequencies": int(training["n_frequencies"]),
                    "paf_sigma": float(sigma if sigma is not None else 1.0),
                    "periodic_include_linear": bool(
                        search["periodic_include_linear"]
                    ),
                },
                "training": {
                    "optimizer": str(training["optimizer"]),
                    "learning_rate": float(training["learning_rate"]),
                    "n_epochs": int(training["n_epochs"]),
                    "paths_per_epoch": int(training["paths_per_epoch"]),
                    "validation_interval": int(training["validation_interval"]),
                },
                "evaluation": {
                    "leaderboards": ["public"],
                    "save_predictions": bool(training["save_predictions"]),
                    "private_access": False,
                },
            }
            directory = LEARNED_ROOT / f"seed-{seed:03d}" / f"cfg-{config_id:04d}"
            configs.append((directory, payload))
    return configs


def math_configs(project: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("no_hedge", {"kind": "no_hedge"}),
        ("heston_cf_delta", {"kind": "heston_cf_delta"}),
        ("heston_mv_delta", {"kind": "heston_mv_delta"}),
    ]
    for width in project["math_deltas"]["no_trade_widths"]:
        candidates.append(
            (
                f"heston_mv_no_trade-width-{float(width):g}",
                {"kind": "heston_mv_no_trade", "band_width": float(width)},
            )
        )
    for scale in project["math_deltas"]["leland_scales"]:
        candidates.append(
            (
                f"leland_local_vol-scale-{float(scale):g}",
                {"kind": "leland_local_vol", "leland_scale": float(scale)},
            )
        )

    configs = []
    for config_id, (name, method) in enumerate(candidates):
        payload = {
            "experiment": {
                "name": "HestonMathDeltas-v1",
                "search": "heston-xi03-public-select-private-final",
                "market": "heston-xi03",
                "config_id": config_id,
                "method_name": name,
                "project_config": "config/project.toml",
                "project_config_sha256": _project_hash(),
            },
            "method": method,
            "evaluation": {
                "public": True,
                "private": (
                    method["kind"]
                    in {"no_hedge", "heston_cf_delta", "heston_mv_delta"}
                ),
                "private_if_public_selected": (
                    method["kind"] in {"heston_mv_no_trade", "leland_local_vol"}
                ),
                "save_predictions": True,
            },
        }
        configs.append((MATH_ROOT / name, payload))
    return configs


def generate() -> None:
    project = load_project_config()
    learned = learned_configs(project)
    math = math_configs(project)
    counts = {"written": 0, "exists": 0}
    for directory, payload in learned + math:
        directory.mkdir(parents=True, exist_ok=True)
        status = write_config_safely(directory / "config.toml", payload)
        counts[status] += 1

    learned_manifest = {
        "experiment": "DeepHedger-Stage1-v1",
        "project_config": "config/project.toml",
        "project_config_sha256": _project_hash(),
        "n_seeds": 8,
        "configs_per_seed": 11,
        "n_configs": 88,
        "private_access": False,
        "config_directories": [
            str(directory.relative_to(PROJECT_ROOT)) for directory, _ in learned
        ],
    }
    math_manifest = {
        "experiment": "HestonMathDeltas-v1",
        "project_config": "config/project.toml",
        "project_config_sha256": _project_hash(),
        "n_configs": len(math),
        "selection": {
            "heston_mv_no_trade": "lowest public entropic_risk",
            "leland_local_vol": "lowest public entropic_risk",
        },
        "private_policy": (
            "fixed methods plus the public-selected member of each tuned family"
        ),
        "config_directories": [
            str(directory.relative_to(PROJECT_ROOT)) for directory, _ in math
        ],
    }
    atomic_write_toml(LEARNED_ROOT / "manifest.toml", learned_manifest)
    atomic_write_toml(MATH_ROOT / "manifest.toml", math_manifest)
    atomic_write_json(
        PROJECT_ROOT / "exp" / "stage1-generation-summary.json",
        {
            "learned_configs": len(learned),
            "math_configs": len(math),
            **counts,
        },
    )
    print(
        json.dumps(
            {
                "learned_root": str(LEARNED_ROOT.relative_to(PROJECT_ROOT)),
                "learned_configs": len(learned),
                "math_root": str(MATH_ROOT.relative_to(PROJECT_ROOT)),
                "math_configs": len(math),
                **counts,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    generate()


if __name__ == "__main__":
    main()

