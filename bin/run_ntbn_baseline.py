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
        time_parameterization=str(model["time_parameterization"]),
        hidden_dims=tuple(int(x) for x in model["hidden_dims"]),
        activation=str(model["activation"]),
        batch_norm=bool(model["batch_norm"]),
        n_frequencies=int(model["n_frequencies"]),
        paf_sigma=float(model["paf_sigma"]),
        periodic_include_linear=bool(model["periodic_include_linear"]),
        learning_rate=float(training["learning_rate"]),
        optimizer=str(training["optimizer"]),
        n_epochs=int(training["n_epochs"]),
        paths_per_epoch=int(training["paths_per_epoch"]),
        validation_interval=int(training["validation_interval"]),
        seed=int(config["seed"]),
    )


def _validate(config: dict) -> None:
    if config["experiment"]["name"] != "DeepHedger-NTBN-v1":
        raise ValueError("Unexpected experiment name.")
    if config["model"]["architecture"] != "ntbn":
        raise ValueError("The NTBN baseline must use architecture='ntbn'.")
    if config["model"]["feature_mode"] != "ntbn_paper":
        raise ValueError("The NTBN baseline must use paper-ordered NTBN features.")
    if list(config["model"]["hidden_dims"]) != [32, 32, 32, 32]:
        raise ValueError("The paper-faithful NTBN baseline requires four 32-unit layers.")
    if config["model"]["activation"] != "relu":
        raise ValueError("The paper-faithful NTBN baseline requires ReLU.")
    if config["evaluation"]["leaderboards"] != ["public"]:
        raise ValueError("NTBN candidates may access only the public board.")
    if bool(config["evaluation"]["private_access"]):
        raise ValueError("NTBN candidates may not access private data.")
    if float(config["market"]["xi"]) not in {0.1, 0.3}:
        raise ValueError("The NTBN baseline is restricted to xi in {0.1, 0.3}.")
    if config["market"]["accounting"] != "discounted":
        raise ValueError("The NTBN baseline requires discounted accounting.")


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
            "architecture": "ntbn",
            "feature_mode": str(config["model"]["feature_mode"]),
            "feature_names": list(config["model"]["feature_names"]),
            "hidden_dims": list(config["model"]["hidden_dims"]),
            "activation": str(config["model"]["activation"]),
            "band_center": str(config["model"]["band_center"]),
            "source_commit": str(config["provenance"]["source_commit"]),
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
