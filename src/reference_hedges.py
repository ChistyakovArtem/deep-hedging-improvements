from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.deltas import HestonCFDeltaHedger


def local_black_scholes_delta(
    paths: torch.Tensor,
    market: dict[str, Any],
    step: int,
) -> torch.Tensor:
    spot = paths[:, step, 0]
    variance = paths[:, step, 1]
    expiry = torch.full_like(
        spot,
        float(market["T"]) * (1.0 - step / int(market["N"])),
    )
    volatility = torch.sqrt(variance.clamp_min(torch.finfo(variance.dtype).tiny))
    d1 = (
        torch.log(spot / float(market["K"]))
        + (float(market["r"]) + 0.5 * volatility.square()) * expiry
    ) / (volatility * torch.sqrt(expiry))
    return torch.special.ndtr(d1)


def leland_local_vol_delta(
    paths: torch.Tensor,
    market: dict[str, Any],
    step: int,
    scale: float,
) -> torch.Tensor:
    spot = paths[:, step, 0]
    variance = paths[:, step, 1]
    expiry = float(market["T"]) * (1.0 - step / int(market["N"]))
    dt = float(market["T"]) / int(market["N"])
    volatility = torch.sqrt(variance.clamp_min(torch.finfo(variance.dtype).tiny))
    uplift = (
        float(scale)
        * math.sqrt(2.0 / math.pi)
        * float(market["transaction_cost"])
        / (volatility * math.sqrt(dt))
    )
    effective_volatility = volatility * torch.sqrt(1.0 + uplift)
    d1 = (
        torch.log(spot / float(market["K"]))
        + (float(market["r"]) + 0.5 * effective_volatility.square()) * expiry
    ) / (effective_volatility * math.sqrt(expiry))
    return torch.special.ndtr(d1)


class HestonMVGrid:
    """Fast bilinear interpolation of the exact Heston MV/Bartlett delta."""

    def __init__(
        self,
        market: dict[str, Any],
        device: torch.device,
        *,
        moneyness_points: int,
        volatility_points: int,
        log_moneyness_min: float,
        log_moneyness_max: float,
        volatility_min: float,
        volatility_max: float,
        n_quad: int,
        phi_min: float,
        phi_max: float,
        batch_size: int,
    ):
        self.market = dict(market)
        self.device = device
        self.settings = {
            "moneyness_points": int(moneyness_points),
            "volatility_points": int(volatility_points),
            "log_moneyness_min": float(log_moneyness_min),
            "log_moneyness_max": float(log_moneyness_max),
            "volatility_min": float(volatility_min),
            "volatility_max": float(volatility_max),
            "n_quad": int(n_quad),
            "phi_min": float(phi_min),
            "phi_max": float(phi_max),
            "batch_size": int(batch_size),
        }
        self._validate()
        cache = self._cache_path()
        if cache.exists():
            with np.load(cache, allow_pickle=False) as payload:
                log_moneyness = np.asarray(payload["log_moneyness"], dtype=np.float32)
                volatility = np.asarray(payload["volatility"], dtype=np.float32)
                values = np.asarray(payload["values"], dtype=np.float32)
        else:
            log_moneyness, volatility, values = self._build()
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                log_moneyness=log_moneyness,
                volatility=volatility,
                values=values,
            )
            temporary.replace(cache)

        expected = (
            int(self.market["N"]),
            self.settings["volatility_points"],
            self.settings["moneyness_points"],
        )
        if values.shape != expected:
            raise RuntimeError(
                f"Malformed Heston MV grid {cache}: expected {expected}, got {values.shape}."
            )
        self.log_moneyness = torch.as_tensor(log_moneyness, device=device)
        self.volatility = torch.as_tensor(volatility, device=device)
        self.values = torch.as_tensor(values, device=device)
        self.log_moneyness_min = float(log_moneyness[0])
        self.log_moneyness_max = float(log_moneyness[-1])
        self.volatility_min = float(volatility[0])
        self.volatility_max = float(volatility[-1])

    def _validate(self) -> None:
        if self.settings["moneyness_points"] < 2:
            raise ValueError("The Heston MV grid needs at least two spot points.")
        if self.settings["volatility_points"] < 2:
            raise ValueError("The Heston MV grid needs at least two volatility points.")
        if self.settings["log_moneyness_min"] >= self.settings["log_moneyness_max"]:
            raise ValueError("Invalid log-moneyness grid bounds.")
        if (
            self.settings["volatility_min"] <= 0.0
            or self.settings["volatility_min"] >= self.settings["volatility_max"]
        ):
            raise ValueError("Invalid volatility grid bounds.")

    def _cache_path(self) -> Path:
        market_keys = (
            "K",
            "r",
            "kappa",
            "theta",
            "xi",
            "rho",
            "T",
            "N",
        )
        payload = {
            "version": 1,
            "market": {key: self.market[key] for key in market_keys},
            "settings": self.settings,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        return PROJECT_ROOT / "local" / "reference_hedges" / f"heston-mv-grid-{digest}.npz"

    def _build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_moneyness = np.linspace(
            self.settings["log_moneyness_min"],
            self.settings["log_moneyness_max"],
            self.settings["moneyness_points"],
            dtype=np.float32,
        )
        volatility = np.linspace(
            self.settings["volatility_min"],
            self.settings["volatility_max"],
            self.settings["volatility_points"],
            dtype=np.float32,
        )
        log_mesh, volatility_mesh = np.meshgrid(
            log_moneyness,
            volatility,
            indexing="xy",
        )
        spot = (float(self.market["K"]) * np.exp(log_mesh.reshape(-1))).astype(np.float64)
        variance = volatility_mesh.reshape(-1).astype(np.float64) ** 2
        engine = HestonCFDeltaHedger(
            K=float(self.market["K"]),
            r=float(self.market["r"]),
            kappa=float(self.market["kappa"]),
            theta=float(self.market["theta"]),
            xi=float(self.market["xi"]),
            rho=float(self.market["rho"]),
            T=float(self.market["T"]),
            N=int(self.market["N"]),
            cost=float(self.market["transaction_cost"]),
            n_quad=self.settings["n_quad"],
            phi_min=self.settings["phi_min"],
            phi_max=self.settings["phi_max"],
            batch_size=self.settings["batch_size"],
            device=self.device,
            dtype=torch.float64,
        )
        spot_tensor = torch.as_tensor(spot, device=self.device)
        variance_tensor = torch.as_tensor(variance, device=self.device)
        values = []
        for step in range(int(self.market["N"])):
            expiry = float(self.market["T"]) * (1.0 - step / int(self.market["N"]))
            delta, variance_sensitivity = engine._delta_vega_torch(
                spot_tensor,
                variance_tensor,
                expiry,
            )
            mv_delta = (
                delta
                + float(self.market["rho"])
                * float(self.market["xi"])
                / spot_tensor
                * variance_sensitivity
            )
            values.append(
                mv_delta.reshape(
                    self.settings["volatility_points"],
                    self.settings["moneyness_points"],
                )
                .float()
                .cpu()
                .numpy()
            )
            print(
                f"built Heston MV reference grid step {step + 1}/{int(self.market['N'])}",
                flush=True,
            )
        return log_moneyness, volatility, np.stack(values)

    def __call__(self, paths: torch.Tensor, step: int) -> torch.Tensor:
        log_moneyness = torch.log(paths[:, step, 0] / float(self.market["K"])).clamp(
            min=self.log_moneyness_min,
            max=self.log_moneyness_max,
        )
        volatility = torch.sqrt(
            paths[:, step, 1].clamp_min(torch.finfo(paths.dtype).tiny)
        ).clamp(
            min=self.volatility_min,
            max=self.volatility_max,
        )
        x_index = (torch.searchsorted(self.log_moneyness, log_moneyness) - 1).clamp(
            0, len(self.log_moneyness) - 2
        )
        y_index = (torch.searchsorted(self.volatility, volatility) - 1).clamp(
            0, len(self.volatility) - 2
        )
        x0 = self.log_moneyness[x_index]
        x1 = self.log_moneyness[x_index + 1]
        y0 = self.volatility[y_index]
        y1 = self.volatility[y_index + 1]
        x_weight = (log_moneyness - x0) / (x1 - x0)
        y_weight = (volatility - y0) / (y1 - y0)
        grid = self.values[step]
        lower = (
            grid[y_index, x_index] * (1.0 - x_weight) + grid[y_index, x_index + 1] * x_weight
        )
        upper = (
            grid[y_index + 1, x_index] * (1.0 - x_weight)
            + grid[y_index + 1, x_index + 1] * x_weight
        )
        return lower * (1.0 - y_weight) + upper * y_weight


class ReferenceHedge:
    """Deterministic reference policy used by residual and band action heads."""

    def __init__(
        self,
        market: dict[str, Any],
        kind: str,
        device: torch.device,
        *,
        leland_scale: float,
        no_trade_width: float,
        grid_moneyness_points: int,
        grid_volatility_points: int,
        grid_log_moneyness_min: float,
        grid_log_moneyness_max: float,
        grid_volatility_min: float,
        grid_volatility_max: float,
        cf_n_quad: int,
        cf_phi_min: float,
        cf_phi_max: float,
        cf_batch_size: int,
    ):
        self.market = dict(market)
        self.kind = kind
        self.leland_scale = float(leland_scale)
        self.no_trade_width = float(no_trade_width)
        self.mv_grid = None
        if kind == "heston_mv_no_trade":
            self.mv_grid = HestonMVGrid(
                market,
                device,
                moneyness_points=grid_moneyness_points,
                volatility_points=grid_volatility_points,
                log_moneyness_min=grid_log_moneyness_min,
                log_moneyness_max=grid_log_moneyness_max,
                volatility_min=grid_volatility_min,
                volatility_max=grid_volatility_max,
                n_quad=cf_n_quad,
                phi_min=cf_phi_min,
                phi_max=cf_phi_max,
                batch_size=cf_batch_size,
            )

    def step(
        self,
        paths: torch.Tensor,
        step: int,
        previous_reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "local_bs":
            return local_black_scholes_delta(paths, self.market, step)
        if self.kind == "leland_local_vol":
            return leland_local_vol_delta(
                paths,
                self.market,
                step,
                self.leland_scale,
            )
        if self.kind == "heston_mv_no_trade":
            if self.mv_grid is None:
                raise RuntimeError("Heston MV reference grid was not initialized.")
            center = self.mv_grid(paths, step)
            lower = center - self.no_trade_width
            upper = center + self.no_trade_width
            return torch.minimum(
                torch.maximum(previous_reference, lower),
                upper,
            )
        raise ValueError(f"Unknown reference hedge: {self.kind}")
