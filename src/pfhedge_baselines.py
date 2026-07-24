from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation import backtest_actions, entropic_risk
from src.market import TorchHestonStream


PFHEDGE_VERSION = "0.23.0"
PFHEDGE_WHEEL_SHA256 = (
    "c5fc81860c2442b0c0b5896e041b544a2893d7893b0bb959c8fa2bbbb21b2dd7"
)
TRAINABLE_MODELS = {"mlp", "ntbn"}
DETERMINISTIC_MODELS = {"black_scholes", "whalley_wilmott"}
MODEL_NAMES = TRAINABLE_MODELS | DETERMINISTIC_MODELS


def _pfhedge_modules() -> tuple[Any, Any, Any, Any, Any, Any, str]:
    try:
        import pfhedge
        from pfhedge.instruments import BrownianStock, EuropeanOption
        from pfhedge.nn import (
            BlackScholes,
            Clamp,
            MultiLayerPerceptron,
            WhalleyWilmott,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The official PFHedge baseline requires pfhedge==0.23.0. "
            "Install the hash-pinned wheel with --no-deps."
        ) from error
    version = str(pfhedge.__version__)
    if version != PFHEDGE_VERSION:
        raise RuntimeError(
            f"Expected pfhedge=={PFHEDGE_VERSION}, found {version}."
        )
    return (
        BrownianStock,
        EuropeanOption,
        BlackScholes,
        Clamp,
        MultiLayerPerceptron,
        WhalleyWilmott,
        version,
    )


@dataclass(frozen=True)
class PFHedgeBaselineConfig:
    model_name: str
    learning_rate: float = 1e-3
    n_epochs: int = 10_000
    paths_per_epoch: int = 3_000
    validation_interval: int = 200
    seed: int = 0

    def validate(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError(
                f"Unknown PFHedge model {self.model_name!r}; "
                f"expected {sorted(MODEL_NAMES)}."
            )
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.n_epochs <= 0 or self.paths_per_epoch <= 0:
            raise ValueError("Training budgets must be positive.")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive.")


class PFHedgeReadmeNTBN(nn.Module):
    """The NoTransactionBandNet shown in PFHedge's official README."""

    def __init__(self, derivative: Any):
        super().__init__()
        (
            _,
            _,
            BlackScholes,
            Clamp,
            MultiLayerPerceptron,
            _,
            _,
        ) = _pfhedge_modules()
        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(in_features=3, out_features=2)
        self.clamp = Clamp()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        previous_hedge = input[..., [-1]]
        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])
        lower = delta - F.leaky_relu(width[..., [0]])
        upper = delta + F.leaky_relu(width[..., [1]])
        return self.clamp(previous_hedge, min=lower, max=upper)


def make_pfhedge_model(
    model_name: str,
    market: dict[str, Any],
) -> nn.Module:
    (
        BrownianStock,
        EuropeanOption,
        BlackScholes,
        _,
        MultiLayerPerceptron,
        WhalleyWilmott,
        _,
    ) = _pfhedge_modules()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown PFHedge model: {model_name}.")

    # PFHedge's documented Black--Scholes and Whalley--Wilmott modules assume
    # zero interest. We leave them literal and record this protocol limitation.
    derivative = EuropeanOption(
        BrownianStock(cost=float(market["transaction_cost"])),
        strike=float(market["K"]),
        maturity=float(market["T"]),
    )
    if model_name == "black_scholes":
        return BlackScholes(derivative)
    if model_name == "whalley_wilmott":
        return WhalleyWilmott(
            derivative,
            a=float(market.get("risk_aversion", 1.0)),
        )
    if model_name == "mlp":
        return MultiLayerPerceptron(in_features=4, out_features=1)
    return PFHedgeReadmeNTBN(derivative)


class PFHedgeBaseline:
    """Run official PFHedge policy modules under the project's common protocol."""

    def __init__(
        self,
        market: dict[str, Any],
        objective: dict[str, Any],
        config: PFHedgeBaselineConfig | dict[str, Any],
        device: str | torch.device | None = None,
    ):
        if isinstance(config, dict):
            config = PFHedgeBaselineConfig(**config)
        config.validate()
        self.market = dict(market)
        self.objective = dict(objective)
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seed(config.seed)
        model_market = dict(self.market)
        model_market["risk_aversion"] = float(self.objective["risk_aversion"])
        self.policy = make_pfhedge_model(config.model_name, model_market).to(self.device)
        self.best_epoch: int | None = None
        self.best_public_risk: float | None = None
        self.training_log: list[dict[str, float | int]] = []

    @property
    def trainable(self) -> bool:
        return self.config.model_name in TRAINABLE_MODELS

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _features(
        self,
        paths: torch.Tensor,
        step: int,
        previous_hedge: torch.Tensor,
    ) -> torch.Tensor:
        spot = paths[:, step, 0]
        variance = paths[:, step, 1]
        expiry = torch.full_like(
            spot,
            float(self.market["T"]) * (1.0 - step / int(self.market["N"])),
        )
        values = (
            torch.log(spot / float(self.market["K"])),
            expiry,
            torch.sqrt(variance.clamp_min(torch.finfo(variance.dtype).tiny)),
            previous_hedge,
        )
        return torch.stack(values, dim=1)

    def predict_actions(self, paths: torch.Tensor) -> torch.Tensor:
        paths = paths.to(self.device)
        previous = torch.zeros(len(paths), device=self.device, dtype=paths.dtype)
        actions = []
        for step in range(int(self.market["N"])):
            features = self._features(paths, step, previous)
            if self.config.model_name == "black_scholes":
                action = self.policy(features[:, :3]).squeeze(1)
            else:
                action = self.policy(features).squeeze(1)
            actions.append(action)
            previous = action
        return torch.stack(actions, dim=1)

    def evaluate_tensor(
        self,
        paths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actions = self.predict_actions(paths)
        outputs = backtest_actions(paths.to(self.device), actions, self.market)
        return actions, outputs

    def fit(self, public_paths: torch.Tensor) -> list[dict[str, float | int]]:
        if not self.trainable:
            raise RuntimeError(f"{self.config.model_name} is deterministic.")
        public_paths = public_paths.to(self.device)
        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )
        stream = TorchHestonStream(
            self.market,
            seed=10_000_000 + self.config.seed,
            device=self.device,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_risk = math.inf
        started = time.perf_counter()

        for epoch in range(1, self.config.n_epochs + 1):
            self.policy.train()
            paths = stream.sample(self.config.paths_per_epoch)
            optimizer.zero_grad(set_to_none=True)
            _, outputs = self.evaluate_tensor(paths)
            pnl = outputs["pnl"] / float(self.objective.get("pnl_scale", 1.0))
            loss = entropic_risk(
                pnl,
                risk_aversion=float(self.objective["risk_aversion"]),
            )
            loss.backward()
            optimizer.step()

            if epoch % self.config.validation_interval == 0 or epoch == 1:
                self.policy.eval()
                with torch.no_grad():
                    _, validation_outputs = self.evaluate_tensor(public_paths)
                    validation_pnl = validation_outputs["pnl"] / float(
                        self.objective.get("pnl_scale", 1.0)
                    )
                    validation_risk = float(
                        entropic_risk(
                            validation_pnl,
                            risk_aversion=float(self.objective["risk_aversion"]),
                        ).item()
                    )
                row: dict[str, float | int] = {
                    "epoch": epoch,
                    "train_entropic_risk": float(loss.detach().item()),
                    "public_entropic_risk": validation_risk,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                self.training_log.append(row)
                if validation_risk < best_risk:
                    best_risk = validation_risk
                    self.best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.policy.state_dict().items()
                    }

        if best_state is None:
            raise RuntimeError("Training produced no validation checkpoint.")
        self.policy.load_state_dict(best_state)
        self.policy.to(self.device)
        self.best_public_risk = best_risk
        return self.training_log

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "market": self.market,
                "objective": self.objective,
                "config": asdict(self.config),
                "policy_state": self.policy.state_dict(),
                "best_epoch": self.best_epoch,
                "best_public_risk": self.best_public_risk,
                "training_log": self.training_log,
                "pfhedge_version": PFHEDGE_VERSION,
                "pfhedge_wheel_sha256": PFHEDGE_WHEEL_SHA256,
            },
            Path(path),
        )
