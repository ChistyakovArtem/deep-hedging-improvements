from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from dev.generate_stage1_configs import MATH_ROOT
from src.artifacts import atomic_write_json, save_predictions, touch_terminal
from src.config import DEFAULT_PROJECT_CONFIG, PROJECT_ROOT, load_project_config, load_toml
from src.deltas import HestonCFDeltaHedger
from src.evaluation import backtest_actions, outputs_to_numpy, scores_from_outputs
from src.market import materialize_leaderboard


def _config_directories(project: dict[str, Any]) -> list[Path]:
    directories = sorted(path.parent for path in MATH_ROOT.glob("*/config.toml"))
    expected = (
        3
        + len(project["math_deltas"]["no_trade_widths"])
        + len(project["math_deltas"]["leland_scales"])
    )
    if len(directories) != expected:
        raise RuntimeError(
            f"Expected {expected} mathematical configs, got {len(directories)}."
        )
    return directories


def _validate_configs(directories: list[Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    project_hash = hashlib.sha256(DEFAULT_PROJECT_CONFIG.read_bytes()).hexdigest()
    result = {}
    for directory in directories:
        config = load_toml(directory / "config.toml")
        experiment = config["experiment"]
        if experiment["project_config_sha256"] != project_hash:
            raise RuntimeError(f"Stale project-config hash in {directory}.")
        name = str(experiment["method_name"])
        if name in result:
            raise RuntimeError(f"Duplicate method name: {name}")
        result[name] = (directory, config)
    return result


def _greek_cache_path(board_metadata: dict[str, Any]) -> Path:
    return (
        PROJECT_ROOT
        / "local"
        / "math_delta_cache"
        / f"{board_metadata['name']}-{board_metadata['paths_sha256']}.npz"
    )


def _heston_cf_and_mv_actions(
    paths: np.ndarray,
    board_metadata: dict[str, Any],
    project: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = _greek_cache_path(board_metadata)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cf = torch.as_tensor(data["cf_actions"], device=device)
            mv = torch.as_tensor(data["mv_actions"], device=device)
        expected = (len(paths), int(project["market"]["N"]))
        if tuple(cf.shape) != expected or tuple(mv.shape) != expected:
            raise RuntimeError(f"Malformed Greek cache: {cache}")
        return cf, mv

    market = project["market"]
    delta_cfg = project["math_deltas"]
    engine = HestonCFDeltaHedger(
        K=float(market["K"]),
        r=float(market["r"]),
        kappa=float(market["kappa"]),
        theta=float(market["theta"]),
        xi=float(market["xi"]),
        rho=float(market["rho"]),
        T=float(market["T"]),
        N=int(market["N"]),
        cost=float(market["transaction_cost"]),
        n_quad=int(delta_cfg["cf_n_quad"]),
        phi_min=float(delta_cfg["cf_phi_min"]),
        phi_max=float(delta_cfg["cf_phi_max"]),
        batch_size=int(delta_cfg["cf_batch_size"]),
        device=device,
        dtype=torch.float64,
    )
    paths_tensor = torch.as_tensor(paths, device=device)
    n_paths = len(paths)
    n_steps = int(market["N"])
    cf_actions = torch.empty((n_paths, n_steps), device=device, dtype=torch.float32)
    mv_actions = torch.empty_like(cf_actions)
    for step in range(n_steps):
        tau = float(market["T"]) * (1.0 - step / n_steps)
        delta, variance_sensitivity = engine._delta_vega_torch(
            paths_tensor[:, step, 0],
            paths_tensor[:, step, 1],
            tau,
        )
        spot = paths_tensor[:, step, 0].to(dtype=torch.float64)
        mv = delta + float(market["rho"]) * float(market["xi"]) / spot * (
            variance_sensitivity
        )
        cf_actions[:, step] = delta.float()
        mv_actions[:, step] = mv.float()
        print(
            f"{board_metadata['name']} CF Greeks step {step + 1}/{n_steps}",
            flush=True,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        cf_actions=cf_actions.cpu().numpy(),
        mv_actions=mv_actions.cpu().numpy(),
    )
    return cf_actions, mv_actions


def _no_trade_actions(center: torch.Tensor, width: float) -> torch.Tensor:
    previous = torch.zeros(len(center), device=center.device, dtype=center.dtype)
    actions = []
    for step in range(center.shape[1]):
        lower = center[:, step] - width
        upper = center[:, step] + width
        action = torch.minimum(torch.maximum(previous, lower), upper)
        actions.append(action)
        previous = action
    return torch.stack(actions, dim=1)


def _leland_actions(
    paths: torch.Tensor,
    market: dict[str, Any],
    scale: float,
) -> torch.Tensor:
    n_steps = int(market["N"])
    dt = float(market["T"]) / n_steps
    cost = float(market["transaction_cost"])
    actions = []
    for step in range(n_steps):
        tau = float(market["T"]) * (1.0 - step / n_steps)
        spot = paths[:, step, 0]
        sigma = torch.sqrt(paths[:, step, 1].clamp_min(1e-12))
        uplift = (
            scale
            * math.sqrt(2.0 / math.pi)
            * cost
            / (sigma * math.sqrt(dt))
        )
        sigma_effective = sigma * torch.sqrt(1.0 + uplift)
        d1 = (
            torch.log(spot / float(market["K"]))
            + (float(market["r"]) + 0.5 * sigma_effective.square()) * tau
        ) / (sigma_effective * math.sqrt(tau))
        actions.append(torch.special.ndtr(d1))
    return torch.stack(actions, dim=1)


def _actions_for_config(
    config: dict[str, Any],
    paths: torch.Tensor,
    cf_actions: torch.Tensor,
    mv_actions: torch.Tensor,
    market: dict[str, Any],
) -> torch.Tensor:
    method = config["method"]
    kind = method["kind"]
    if kind == "no_hedge":
        return torch.zeros_like(cf_actions)
    if kind == "heston_cf_delta":
        return cf_actions
    if kind == "heston_mv_delta":
        return mv_actions
    if kind == "heston_mv_no_trade":
        return _no_trade_actions(mv_actions, float(method["band_width"]))
    if kind == "leland_local_vol":
        return _leland_actions(paths, market, float(method["leland_scale"]))
    raise ValueError(f"Unknown mathematical method: {kind}")


def _evaluate_and_save(
    directory: Path,
    config: dict[str, Any],
    board_name: str,
    board_metadata: dict[str, Any],
    paths: torch.Tensor,
    actions: torch.Tensor,
    project: dict[str, Any],
) -> dict[str, Any]:
    outputs = backtest_actions(paths, actions, project["market"])
    scores: dict[str, Any] = scores_from_outputs(
        outputs, float(project["objective"]["risk_aversion"])
    )
    scores.update(
        {
            "leaderboard": board_name,
            "paths_sha256": board_metadata["paths_sha256"],
            "method_name": config["experiment"]["method_name"],
            "method": config["method"],
        }
    )
    atomic_write_json(directory / f"scores-{board_name}.json", scores)
    save_predictions(
        directory / f"predictions-{board_name}.npz",
        actions.detach().cpu().numpy(),
        outputs_to_numpy(outputs),
    )
    return scores


def _selected_name(
    rows: list[dict[str, Any]],
    configs: dict[str, tuple[Path, dict[str, Any]]],
    kind: str,
) -> str:
    candidates = [
        row
        for row in rows
        if configs[str(row["method_name"])][1]["method"]["kind"] == kind
    ]
    if not candidates:
        raise RuntimeError(f"No public candidates for {kind}.")
    return str(min(candidates, key=lambda row: float(row["entropic_risk"]))["method_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    started = time.perf_counter()
    project = load_project_config()
    configs = _validate_configs(_config_directories(project))
    for directory, _ in configs.values():
        touch_terminal(directory, "RUNNING")

    # Select every tuned mathematical family on public before loading private.
    public_numpy, public_metadata = materialize_leaderboard(
        PROJECT_ROOT,
        project["market"],
        project["leaderboards"]["public"],
    )
    public_paths = torch.as_tensor(public_numpy, device=device)
    public_cf, public_mv = _heston_cf_and_mv_actions(
        public_numpy, public_metadata, project, device
    )
    public_rows = []
    for name, (directory, config) in configs.items():
        actions = _actions_for_config(
            config, public_paths, public_cf, public_mv, project["market"]
        )
        row = _evaluate_and_save(
            directory,
            config,
            "public",
            public_metadata,
            public_paths,
            actions,
            project,
        )
        public_rows.append(row)
        print(f"public {name}: entropic_risk={row['entropic_risk']:.8f}", flush=True)

    selected = {
        "heston_mv_no_trade": _selected_name(
            public_rows, configs, "heston_mv_no_trade"
        ),
        "leland_local_vol": _selected_name(public_rows, configs, "leland_local_vol"),
    }
    selection_payload = {
        "selection_leaderboard": "public",
        "selection_metric": "entropic_risk",
        "selection_direction": "lower",
        "selected": selected,
    }
    atomic_write_json(MATH_ROOT / "public-selection.json", selection_payload)
    atomic_write_json(MATH_ROOT / "leaderboard-public.json", public_rows)

    del public_paths, public_cf, public_mv, public_numpy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    private_numpy, private_metadata = materialize_leaderboard(
        PROJECT_ROOT,
        project["market"],
        project["leaderboards"]["private"],
    )
    private_paths = torch.as_tensor(private_numpy, device=device)
    private_cf, private_mv = _heston_cf_and_mv_actions(
        private_numpy, private_metadata, project, device
    )
    fixed_private = {
        name
        for name, (_, config) in configs.items()
        if bool(config["evaluation"]["private"])
    }
    private_names = fixed_private | set(selected.values())
    private_rows = []
    for name, (directory, config) in configs.items():
        if name not in private_names:
            (directory / "PRIVATE_NOT_EVALUATED.txt").write_text(
                "This hyperparameter candidate was not selected on public; "
                "private remains unopened for it.\n"
            )
            continue
        actions = _actions_for_config(
            config, private_paths, private_cf, private_mv, project["market"]
        )
        row = _evaluate_and_save(
            directory,
            config,
            "private",
            private_metadata,
            private_paths,
            actions,
            project,
        )
        private_rows.append(row)
        print(f"private {name}: entropic_risk={row['entropic_risk']:.8f}", flush=True)

    atomic_write_json(MATH_ROOT / "leaderboard-private.json", private_rows)
    atomic_write_json(
        MATH_ROOT / "run-summary.json",
        {
            **selection_payload,
            "public_methods": len(public_rows),
            "private_methods": len(private_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
        },
    )
    for directory, _ in configs.values():
        touch_terminal(directory, "DONE")


if __name__ == "__main__":
    main()
