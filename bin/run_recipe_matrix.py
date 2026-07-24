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
from src.hedger import DeepHedger, DeepHedgerConfig
from src.market import materialize_leaderboard
from src.nirvana_io import publish_experiment


def _hedger_config(config: dict) -> DeepHedgerConfig:
    model = config["model"]
    training = config["training"]
    return DeepHedgerConfig(
        architecture=str(model["architecture"]),
        feature_mode=str(model["feature_mode"]),
        prediction_target=str(model["prediction_target"]),
        action_head=str(model["action_head"]),
        reference_hedge=str(model["reference_hedge"]),
        time_parameterization=str(model["time_parameterization"]),
        hidden_dims=tuple(int(x) for x in model["hidden_dims"]),
        activation=str(model["activation"]),
        batch_norm=bool(model["batch_norm"]),
        output_initialization=str(model["output_initialization"]),
        n_frequencies=int(model["n_frequencies"]),
        paf_sigma=float(model["paf_sigma"]),
        periodic_include_linear=bool(model["periodic_include_linear"]),
        reference_leland_scale=float(model["reference_leland_scale"]),
        reference_no_trade_width=float(model["reference_no_trade_width"]),
        reference_grid_moneyness_points=int(model["reference_grid_moneyness_points"]),
        reference_grid_volatility_points=int(model["reference_grid_volatility_points"]),
        reference_grid_log_moneyness_min=float(model["reference_grid_log_moneyness_min"]),
        reference_grid_log_moneyness_max=float(model["reference_grid_log_moneyness_max"]),
        reference_grid_volatility_min=float(model["reference_grid_volatility_min"]),
        reference_grid_volatility_max=float(model["reference_grid_volatility_max"]),
        reference_cf_n_quad=int(model["reference_cf_n_quad"]),
        reference_cf_phi_min=float(model["reference_cf_phi_min"]),
        reference_cf_phi_max=float(model["reference_cf_phi_max"]),
        reference_cf_batch_size=int(model["reference_cf_batch_size"]),
        use_running_pnl=bool(model.get("use_running_pnl", False)),
        running_pnl_scale=float(model.get("running_pnl_scale", 1.0)),
        learning_rate=float(training["learning_rate"]),
        optimizer=str(training["optimizer"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
        ema_decay=float(training.get("ema_decay", 0.99)),
        muon_learning_rate=float(training.get("muon_learning_rate", 0.02)),
        muon_momentum=float(training.get("muon_momentum", 0.95)),
        n_epochs=int(training["n_epochs"]),
        paths_per_epoch=int(training["paths_per_epoch"]),
        validation_interval=int(training["validation_interval"]),
        seed=int(config["seed"]),
    )


def _validate(config: dict) -> None:
    if config["experiment"]["name"] != "DeepHedger-RecipeMatrix-v1":
        raise ValueError("Unexpected experiment name.")
    variant_id = int(config["experiment"]["variant_id"])
    if variant_id not in range(8):
        raise ValueError(f"Unexpected variant id: {variant_id}.")
    model = config["model"]
    if model["architecture"] != "mlp":
        raise ValueError("This no-PAF matrix requires an MLP encoder.")
    if model["action_head"] not in {
        "direct",
        "delta_residual",
        "delta_band",
    }:
        raise ValueError("Unexpected action head.")
    if model["prediction_target"] != "direct":
        raise ValueError("The matrix uses the explicit action-head interface.")
    expected_initialization = (
        "zero_last" if model["action_head"] == "delta_residual" else "default"
    )
    if model["output_initialization"] != expected_initialization:
        raise ValueError("Action-head initialization mismatch.")
    if bool(model["reference_used"]) != (model["action_head"] != "direct"):
        raise ValueError("Reference usage and action head disagree.")
    if config["evaluation"]["leaderboards"] != ["public"]:
        raise ValueError("Recipe candidates may access only the public board.")
    if bool(config["evaluation"]["private_access"]):
        raise ValueError("Recipe candidates may not access private data.")
    if float(config["market"]["xi"]) not in {0.1, 0.3}:
        raise ValueError("The matrix is restricted to xi in {0.1, 0.3}.")
    if config["market"]["accounting"] != "discounted":
        raise ValueError("The experiment requires discounted accounting.")

    if variant_id == 0:
        expected = {
            "feature_mode": "paper_log_state",
            "time_parameterization": "per_step",
            "hidden_dims": [16, 16],
            "activation": "relu",
            "batch_norm": True,
            "action_head": "direct",
        }
    else:
        expected = {
            "feature_mode": "normalized",
            "time_parameterization": "shared",
            "hidden_dims": [64, 32],
            "activation": "leaky_relu",
            "batch_norm": False,
        }
    for key, value in expected.items():
        actual = list(model[key]) if key == "hidden_dims" else model[key]
        if actual != value:
            raise ValueError(f"Expected model.{key}={value!r}, got {actual!r}.")


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
    scores.update(
        {
            "leaderboard": "public",
            "paths_sha256": public_metadata["paths_sha256"],
            "seed": int(config["seed"]),
            "xi": float(market["xi"]),
            "variant_id": int(config["experiment"]["variant_id"]),
            "variant": str(config["experiment"]["variant"]),
            "article_label": str(config["experiment"]["article_label"]),
            "architecture": str(config["model"]["architecture"]),
            "feature_mode": str(config["model"]["feature_mode"]),
            "feature_names": list(config["model"]["feature_names"]),
            "action_head": str(config["model"]["action_head"]),
            "reference_selector": str(config["reference_selection"]["selector"]),
            "reference_hedge": str(config["model"]["reference_hedge"]),
            "reference_leland_scale": float(config["model"]["reference_leland_scale"]),
            "reference_no_trade_width": float(config["model"]["reference_no_trade_width"]),
            "output_initialization": str(config["model"]["output_initialization"]),
            "hidden_dims": list(config["model"]["hidden_dims"]),
            "activation": str(config["model"]["activation"]),
            "optimizer": str(config["training"]["optimizer"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "paths_per_epoch": int(config["training"]["paths_per_epoch"]),
            "parameter_count": sum(
                parameter.numel() for parameter in hedger.policy.parameters()
            ),
            "best_epoch": int(hedger.best_epoch or -1),
            "fit_and_eval_seconds": time.perf_counter() - started,
            "smoke": smoke,
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
