from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def paths_sha256(paths: np.ndarray) -> str:
    array = np.ascontiguousarray(paths, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sample_heston_numpy(
    market: dict[str, Any],
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Generate the frozen full-truncation-Euler Heston paths."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_steps = int(market["N"])
    dt = float(market["T"]) / n_steps
    sqrt_dt = np.sqrt(dt)

    spot = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variance = np.empty_like(spot)
    spot[:, 0] = float(market["S0"])
    variance[:, 0] = float(market["v0"])

    rho = float(market["rho"])
    rho_orthogonal = np.sqrt(1.0 - rho * rho)
    for step in range(n_steps):
        z_spot = rng.standard_normal(n_paths)
        z_independent = rng.standard_normal(n_paths)
        z_variance = rho * z_spot + rho_orthogonal * z_independent
        variance_positive = np.maximum(variance[:, step], 0.0)
        spot[:, step + 1] = spot[:, step] * np.exp(
            (float(market["r"]) - 0.5 * variance_positive) * dt
            + np.sqrt(variance_positive) * sqrt_dt * z_spot
        )
        variance[:, step + 1] = np.maximum(
            variance[:, step]
            + float(market["kappa"])
            * (float(market["theta"]) - variance_positive)
            * dt
            + float(market["xi"])
            * np.sqrt(variance_positive)
            * sqrt_dt
            * z_variance,
            1e-8,
        )
    return np.stack((spot, variance), axis=-1).astype(np.float32)


class TorchHestonStream:
    """Persistent per-run GPU RNG stream; it never resets between epochs."""

    def __init__(self, market: dict[str, Any], seed: int, device: torch.device):
        self.market = market
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def sample(self, n_paths: int) -> torch.Tensor:
        market = self.market
        n_steps = int(market["N"])
        dt = float(market["T"]) / n_steps
        sqrt_dt = dt**0.5
        dtype = torch.float32

        spot = torch.empty(
            (n_paths, n_steps + 1), dtype=dtype, device=self.device
        )
        variance = torch.empty_like(spot)
        spot[:, 0] = float(market["S0"])
        variance[:, 0] = float(market["v0"])

        rho = float(market["rho"])
        rho_orthogonal = (1.0 - rho * rho) ** 0.5
        for step in range(n_steps):
            z_spot = torch.randn(
                n_paths, dtype=dtype, device=self.device, generator=self.generator
            )
            z_independent = torch.randn(
                n_paths, dtype=dtype, device=self.device, generator=self.generator
            )
            z_variance = rho * z_spot + rho_orthogonal * z_independent
            variance_positive = variance[:, step].clamp_min(0.0)
            spot[:, step + 1] = spot[:, step] * torch.exp(
                (float(market["r"]) - 0.5 * variance_positive) * dt
                + torch.sqrt(variance_positive) * sqrt_dt * z_spot
            )
            variance[:, step + 1] = (
                variance[:, step]
                + float(market["kappa"])
                * (float(market["theta"]) - variance_positive)
                * dt
                + float(market["xi"])
                * torch.sqrt(variance_positive)
                * sqrt_dt
                * z_variance
            ).clamp_min(1e-8)
        return torch.stack((spot, variance), dim=-1)


def leaderboard_path(root: Path, name: str) -> Path:
    return root / "data" / "leaderboards" / f"heston-{name}.npz"


def materialize_leaderboard(
    root: Path,
    market: dict[str, Any],
    board: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = leaderboard_path(root, str(board["name"]))
    expected_hash = str(board.get("paths_sha256", ""))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            paths = np.asarray(data["paths"], dtype=np.float32)
            metadata = json.loads(str(data["metadata"]))
    else:
        paths = sample_heston_numpy(
            market,
            n_paths=int(board["n_paths"]),
            seed=int(board["seed"]),
        )
        metadata = {
            "name": str(board["name"]),
            "n_paths": int(board["n_paths"]),
            "seed": int(board["seed"]),
            "generator": "numpy.PCG64/full_truncation_euler/v1",
            "paths_sha256": paths_sha256(paths),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, paths=paths, metadata=json.dumps(metadata, sort_keys=True))

    actual_hash = paths_sha256(paths)
    if paths.shape != (int(board["n_paths"]), int(market["N"]) + 1, 2):
        raise RuntimeError(f"Unexpected leaderboard shape at {path}: {paths.shape}")
    if metadata["seed"] != int(board["seed"]) or metadata["n_paths"] != int(
        board["n_paths"]
    ):
        raise RuntimeError(f"Leaderboard metadata mismatch at {path}.")
    if metadata["paths_sha256"] != actual_hash:
        raise RuntimeError(f"Stored leaderboard checksum mismatch at {path}.")
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"Frozen leaderboard checksum mismatch at {path}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return paths, metadata

