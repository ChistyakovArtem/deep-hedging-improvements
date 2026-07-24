from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from bin.run_recipe_matrix import _hedger_config
from src.artifacts import atomic_write_json, save_predictions, touch_terminal
from src.config import PROJECT_ROOT, load_toml
from src.evaluation import outputs_to_numpy, scores_from_outputs
from src.hedger import DeepHedger
from src.market import materialize_leaderboard
from src.nirvana_io import publish_experiment


FEATURES = [
    "log_moneyness",
    "variance_ratio",
    "time_ratio",
    "previous_hedge",
    "running_pnl",
]


def _validate(config: dict) -> None:
    if config["experiment"]["name"] != "DeepHedger-RunningPnL-v1":
        raise ValueError("Unexpected experiment name.")
    model = config["model"]
    expected = {
        "architecture": "mlp",
        "feature_mode": "normalized",
        "feature_names": FEATURES,
        "time_parameterization": "shared",
        "hidden_dims": [64, 32],
        "activation": "leaky_relu",
        "batch_norm": False,
        "action_head": "delta_residual",
        "prediction_target": "direct",
        "output_initialization": "zero_last",
        "use_running_pnl": True,
        "running_pnl_scale": 1.0,
    }
    for key, value in expected.items():
        actual = list(model[key]) if key in {"feature_names", "hidden_dims"} else model[key]
        if actual != value:
            raise ValueError(f"Expected model.{key}={value!r}, got {actual!r}.")
    xi = float(config["market"]["xi"])
    expected_reference = {
        0.1: ("leland_local_vol", 20.0, 0.0),
        0.3: ("heston_mv_no_trade", 0.0, 0.04),
    }
    reference = (
        model["reference_hedge"],
        float(model["reference_leland_scale"]),
        float(model["reference_no_trade_width"]),
    )
    if xi not in expected_reference or reference != expected_reference[xi]:
        raise ValueError(f"Unexpected selected reference at xi={xi}: {reference}.")
    training = config["training"]
    if (
        training["optimizer"] != "adam"
        or float(training["learning_rate"]) != 1e-3
        or int(training["n_epochs"]) != 10_000
        or int(training["paths_per_epoch"]) != 3_000
        or int(training["validation_interval"]) != 200
    ):
        raise ValueError("Training budget differs from the paired residual baseline.")
    if config["evaluation"]["leaderboards"] != ["public"]:
        raise ValueError("Running-PnL candidates may access only the public board.")
    if bool(config["evaluation"]["private_access"]):
        raise ValueError("Running-PnL candidates may not access private data.")
    if config["market"]["accounting"] != "discounted":
        raise ValueError("The experiment requires discounted accounting.")


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
            "variant": str(config["experiment"]["variant"]),
            "article_label": str(config["experiment"]["article_label"]),
            "architecture": str(config["model"]["architecture"]),
            "feature_mode": str(config["model"]["feature_mode"]),
            "feature_names": list(config["model"]["feature_names"]),
            "action_head": str(config["model"]["action_head"]),
            "reference_hedge": str(config["model"]["reference_hedge"]),
            "reference_leland_scale": float(config["model"]["reference_leland_scale"]),
            "reference_no_trade_width": float(config["model"]["reference_no_trade_width"]),
            "use_running_pnl": True,
            "running_pnl_scale": float(config["model"]["running_pnl_scale"]),
            "running_pnl_definition": str(
                config["provenance"]["running_pnl_definition"]
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
