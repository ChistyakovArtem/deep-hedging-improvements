from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def entropic_risk(pnl: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    scaled_loss = -float(risk_aversion) * pnl
    return (
        torch.logsumexp(scaled_loss, dim=0) - math.log(scaled_loss.numel())
    ) / float(risk_aversion)


def backtest_actions(
    paths: torch.Tensor,
    actions: torch.Tensor,
    market: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Evaluate hedge actions with one coherent discounted accounting rule."""
    n_paths, n_plus_one, _ = paths.shape
    n_steps = n_plus_one - 1
    if actions.shape != (n_paths, n_steps):
        raise ValueError(
            f"Expected actions {(n_paths, n_steps)}, got {tuple(actions.shape)}."
        )

    device = paths.device
    dtype = paths.dtype
    times = torch.linspace(
        0.0, float(market["T"]), n_plus_one, device=device, dtype=dtype
    )
    discounts = torch.exp(-float(market["r"]) * times)
    discounted_spot = paths[:, :, 0] * discounts.unsqueeze(0)
    cash = torch.zeros(n_paths, device=device, dtype=dtype)
    previous = torch.zeros_like(cash)
    total_fees = torch.zeros_like(cash)
    turnover = torch.zeros_like(cash)
    cost = float(market["transaction_cost"])

    for step in range(n_steps):
        action = actions[:, step]
        trade = action - previous
        absolute_trade = trade.abs()
        fees = cost * absolute_trade * discounted_spot[:, step]
        cash = cash - trade * discounted_spot[:, step] - fees
        total_fees = total_fees + fees
        turnover = turnover + absolute_trade
        previous = action

    cash = cash + previous * discounted_spot[:, -1]
    payoff = torch.clamp(paths[:, -1, 0] - float(market["K"]), min=0.0)
    discounted_payoff = payoff * discounts[-1]
    pnl = cash - discounted_payoff
    return {
        "pnl": pnl,
        "fees": total_fees,
        "turnover": turnover,
    }


def scores_from_outputs(
    outputs: dict[str, torch.Tensor],
    risk_aversion: float,
) -> dict[str, float]:
    pnl = outputs["pnl"].detach().double()
    loss = -pnl

    def cvar(level: float) -> float:
        count = max(1, math.ceil((1.0 - level) * len(loss)))
        return float(torch.topk(loss, count, largest=True).values.mean().item())

    risk = float(entropic_risk(pnl, risk_aversion).item())
    return {
        "score": -risk,
        "entropic_risk": risk,
        "pnl_mean": float(pnl.mean().item()),
        "pnl_std": float(pnl.std(unbiased=True).item()),
        "loss_cvar_95": cvar(0.95),
        "loss_cvar_99": cvar(0.99),
        "fees_mean": float(outputs["fees"].double().mean().item()),
        "turnover_mean": float(outputs["turnover"].double().mean().item()),
        "n_paths": int(len(pnl)),
    }


def outputs_to_numpy(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().to(torch.float32).numpy()
        for key, value in outputs.items()
    }

