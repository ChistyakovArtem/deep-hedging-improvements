from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from src.artifacts import atomic_write_json, save_predictions, touch_terminal
from src.config import PROJECT_ROOT, load_toml
from src.evaluation import outputs_to_numpy, scores_from_outputs
from src.market import materialize_leaderboard
from src.nirvana_io import publish_experiment
from src.pfhedge_baselines import (
    MODEL_NAMES,
    PFHEDGE_VERSION,
    PFHEDGE_WHEEL_SHA256,
    PFHedgeBaseline,
    PFHedgeBaselineConfig,
)


def _baseline_config(config: dict) -> PFHedgeBaselineConfig:
    training = config["training"]
    return PFHedgeBaselineConfig(
        model_name=str(config["model"]["name"]),
        learning_rate=float(training["learning_rate"]),
        n_epochs=int(training["n_epochs"]),
        paths_per_epoch=int(training["paths_per_epoch"]),
        validation_interval=int(training["validation_interval"]),
        seed=int(config["seed"]),
    )


def _validate(config: dict) -> None:
    if config["experiment"]["name"] != "PFHedge-Official-v1":
        raise ValueError("Unexpected experiment name.")
    if config["model"]["name"] not in MODEL_NAMES:
        raise ValueError("Unexpected PFHedge model.")
    provenance = config["provenance"]
    if provenance["version"] != PFHEDGE_VERSION:
        raise ValueError("PFHedge version is not pinned to the audited release.")
    if provenance["wheel_sha256"] != PFHEDGE_WHEEL_SHA256:
        raise ValueError("PFHedge wheel checksum does not match the audited artifact.")
    if config["evaluation"]["leaderboards"] != ["public"]:
        raise ValueError("PFHedge candidates may access only the public board.")
    if bool(config["evaluation"]["private_access"]):
        raise ValueError("PFHedge candidates may not access private data.")
    if float(config["market"]["xi"]) not in {0.1, 0.3}:
        raise ValueError("The comparison is restricted to xi in {0.1, 0.3}.")
    if config["market"]["accounting"] != "discounted":
        raise ValueError("The comparison requires discounted accounting.")


def run(config_dir: Path, device: str, smoke: bool) -> Path:
    config_dir = config_dir.resolve()
    config = load_toml(config_dir / "config.toml")
    _validate(config)

    market = config["market"]
    objective = config["objective"]
    public_paths, public_metadata = materialize_leaderboard(
        PROJECT_ROOT,
        market,
        config["leaderboard"],
    )
    baseline_config = _baseline_config(config)
    output_dir = config_dir
    if smoke:
        baseline_config = replace(
            baseline_config,
            n_epochs=3,
            paths_per_epoch=128,
            validation_interval=1,
        )
        public_paths = public_paths[:512]
        relative = config_dir.relative_to(PROJECT_ROOT)
        output_dir = PROJECT_ROOT / "local" / "smoke" / relative
        output_dir.mkdir(parents=True, exist_ok=True)

    touch_terminal(output_dir, "RUNNING")
    started = time.perf_counter()
    baseline = PFHedgeBaseline(
        market=market,
        objective=objective,
        config=baseline_config,
        device=device,
    )
    public_tensor = torch.as_tensor(public_paths, device=baseline.device)
    if baseline.trainable:
        baseline.fit(public_tensor)
    baseline.policy.eval()
    with torch.no_grad():
        actions, outputs = baseline.evaluate_tensor(public_tensor)
    scores = scores_from_outputs(
        outputs,
        risk_aversion=float(objective["risk_aversion"]),
    )
    scores.update(
        {
            "leaderboard": "public",
            "paths_sha256": public_metadata["paths_sha256"],
            "seed": int(config["seed"]),
            "xi": float(market["xi"]),
            "model": str(config["model"]["name"]),
            "source_class": str(config["model"]["source_class"]),
            "inputs": list(config["model"]["inputs"]),
            "pfhedge_version": PFHEDGE_VERSION,
            "pfhedge_wheel_sha256": PFHEDGE_WHEEL_SHA256,
            "optimizer": str(config["training"]["optimizer"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "paths_per_epoch": int(config["training"]["paths_per_epoch"]),
            "parameter_count": sum(
                parameter.numel() for parameter in baseline.policy.parameters()
            ),
            "best_epoch": int(baseline.best_epoch or -1),
            "fit_and_eval_seconds": time.perf_counter() - started,
            "trainable": baseline.trainable,
            "smoke": smoke,
            "private_access": False,
            "literal_zero_rate_policy": True,
        }
    )
    baseline.save(output_dir / "checkpoint.pt")
    atomic_write_json(output_dir / "training_log.json", baseline.training_log)
    atomic_write_json(output_dir / "scores-public.json", scores)
    if bool(config["evaluation"]["save_predictions"]):
        save_predictions(
            output_dir / "predictions-public.npz",
            actions.detach().cpu().numpy(),
            outputs_to_numpy(outputs),
        )
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in scores.items()) + "\n"
    )
    touch_terminal(output_dir, "DONE")
    if not smoke:
        publish_experiment(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = run(args.config_dir, args.device, args.smoke)
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
