from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from bin.run_recipe_matrix import _hedger_config
from dev.generate_optimizer_study import OPTIMIZER_CANDIDATES
from src.artifacts import atomic_write_json, save_predictions, touch_terminal
from src.config import PROJECT_ROOT, load_toml
from src.evaluation import outputs_to_numpy, scores_from_outputs
from src.hedger import DeepHedger
from src.market import materialize_leaderboard
from src.nirvana_io import publish_experiment


def _validate(config: dict) -> None:
    experiment = config["experiment"]
    if experiment["name"] != "DeepHedger-OptimizerStudy-v1":
        raise ValueError("Unexpected experiment name.")
    optimizer = str(experiment["optimizer"])
    hp_id = int(experiment["hp_id"])
    if optimizer not in OPTIMIZER_CANDIDATES:
        raise ValueError(f"Unexpected optimizer: {optimizer}.")
    if hp_id not in range(5):
        raise ValueError(f"Unexpected optimizer HP id: {hp_id}.")
    candidate = OPTIMIZER_CANDIDATES[optimizer][hp_id]
    if experiment["hp_label"] != candidate["label"]:
        raise ValueError("Optimizer HP label does not match the registered design.")

    baseline = load_toml(PROJECT_ROOT / config["provenance"]["baseline_config"])
    for section in (
        "market",
        "objective",
        "leaderboard",
        "model",
        "reference_selection",
        "evaluation",
    ):
        if config[section] != baseline[section]:
            raise ValueError(f"Optimizer study changed non-optimizer section {section}.")
    training = config["training"]
    expected_training = {
        **baseline["training"],
        "optimizer": optimizer,
        "learning_rate": candidate["learning_rate"],
        "weight_decay": candidate["weight_decay"],
        "ema_decay": candidate["ema_decay"],
        "muon_learning_rate": candidate["muon_learning_rate"],
        "muon_momentum": candidate["muon_momentum"],
    }
    if training != expected_training:
        raise ValueError("Training config differs from the registered optimizer design.")
    if int(training["n_epochs"]) != 10_000:
        raise ValueError("Optimizer search must retain the 10,000-update budget.")
    if bool(config["model"].get("use_running_pnl", False)):
        raise ValueError("Running PnL is excluded from the optimizer study.")
    if config["model"]["architecture"] != "mlp":
        raise ValueError("Periodic embeddings are excluded from the optimizer study.")
    if config["evaluation"]["leaderboards"] != ["public"]:
        raise ValueError("Optimizer candidates may access only the public board.")
    if bool(config["evaluation"]["private_access"]):
        raise ValueError("Optimizer candidates may not access private data.")


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
    hedger_config = _hedger_config(config)
    output_dir = config_dir
    if smoke:
        hedger_config = replace(
            hedger_config,
            n_epochs=3,
            paths_per_epoch=128,
            validation_interval=1,
            reference_grid_moneyness_points=9,
            reference_grid_volatility_points=9,
            reference_cf_n_quad=16,
            reference_cf_phi_max=100.0,
            reference_cf_batch_size=256,
        )
        public_paths = public_paths[:512]
        relative = config_dir.relative_to(PROJECT_ROOT)
        output_dir = PROJECT_ROOT / "local" / "smoke" / relative
        output_dir.mkdir(parents=True, exist_ok=True)

    touch_terminal(output_dir, "RUNNING")
    started = time.perf_counter()
    hedger = DeepHedger(
        market=market,
        objective=objective,
        config=hedger_config,
        device=device,
    )
    public_tensor = torch.as_tensor(public_paths, device=hedger.device)
    hedger.fit(public_tensor)
    hedger.policy.eval()
    with torch.no_grad():
        actions, outputs = hedger.evaluate_tensor(public_tensor)
    scores = scores_from_outputs(
        outputs,
        risk_aversion=float(objective["risk_aversion"]),
    )
    training = config["training"]
    experiment = config["experiment"]
    scores.update(
        {
            "leaderboard": "public",
            "paths_sha256": public_metadata["paths_sha256"],
            "seed": int(config["seed"]),
            "xi": float(market["xi"]),
            "variant": str(experiment["variant"]),
            "article_label": str(experiment["article_label"]),
            "optimizer": str(training["optimizer"]),
            "optimizer_hp_id": int(experiment["hp_id"]),
            "optimizer_hp_label": str(experiment["hp_label"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "ema_decay": float(training["ema_decay"]),
            "muon_learning_rate": float(training["muon_learning_rate"]),
            "muon_momentum": float(training["muon_momentum"]),
            "architecture": str(config["model"]["architecture"]),
            "feature_mode": str(config["model"]["feature_mode"]),
            "feature_names": list(config["model"]["feature_names"]),
            "action_head": str(config["model"]["action_head"]),
            "reference_hedge": str(config["model"]["reference_hedge"]),
            "reference_leland_scale": float(
                config["model"]["reference_leland_scale"]
            ),
            "reference_no_trade_width": float(
                config["model"]["reference_no_trade_width"]
            ),
            "parameter_count": sum(
                parameter.numel() for parameter in hedger.policy.parameters()
            ),
            "best_epoch": int(hedger.best_epoch or -1),
            "fit_and_eval_seconds": time.perf_counter() - started,
            "smoke": smoke,
            "private_access": False,
        }
    )
    hedger.save(output_dir / "checkpoint.pt")
    atomic_write_json(output_dir / "training_log.json", hedger.training_log)
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
