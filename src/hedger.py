from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from src.evaluation import backtest_actions, entropic_risk
from src.market import TorchHestonStream


@dataclass(frozen=True)
class DeepHedgerConfig:
    architecture: str = "mlp"
    hidden_dims: tuple[int, ...] = (64, 32)
    n_frequencies: int = 16
    paf_sigma: float = 1.0
    periodic_include_linear: bool = False
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    n_epochs: int = 10_000
    paths_per_epoch: int = 3_000
    validation_interval: int = 200
    seed: int = 0

    def validate(self) -> None:
        allowed = {"mlp", "paf_shared", "paf_featurewise"}
        if self.architecture not in allowed:
            raise ValueError(
                f"Unknown architecture {self.architecture!r}; expected {sorted(allowed)}."
            )
        if self.optimizer != "adam":
            raise ValueError("Stage 1 intentionally supports only Adam.")
        if self.n_epochs <= 0 or self.paths_per_epoch <= 0:
            raise ValueError("Training budgets must be positive.")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive.")


class _PeriodicEmbedding(nn.Module):
    def __init__(
        self,
        kind: str,
        input_dim: int,
        n_frequencies: int,
        sigma: float,
        include_linear: bool,
    ):
        super().__init__()
        self.kind = kind
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.include_linear = include_linear
        self.frequencies = nn.Parameter(
            sigma * torch.randn(input_dim, n_frequencies)
        )

    @property
    def output_dim(self) -> int:
        multiplier = 3 if self.include_linear else 2
        if self.kind == "paf_shared":
            return multiplier * self.n_frequencies
        return multiplier * self.input_dim * self.n_frequencies

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "paf_shared":
            phase = 2.0 * math.pi * (values @ self.frequencies)
            parts = [torch.sin(phase), torch.cos(phase)]
        else:
            phase = (
                2.0
                * math.pi
                * values.unsqueeze(-1)
                * self.frequencies.unsqueeze(0)
            )
            parts = [torch.sin(phase), torch.cos(phase)]
        if self.include_linear:
            parts.insert(0, phase)
        embedded = torch.cat(parts, dim=-1)
        return embedded.flatten(start_dim=1)


class _Policy(nn.Module):
    def __init__(self, config: DeepHedgerConfig):
        super().__init__()
        if config.architecture == "mlp":
            self.embedding = None
            input_dim = 4
        else:
            self.embedding = _PeriodicEmbedding(
                kind=config.architecture,
                input_dim=3,
                n_frequencies=config.n_frequencies,
                sigma=config.paf_sigma,
                include_linear=config.periodic_include_linear,
            )
            input_dim = self.embedding.output_dim + 1

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.LeakyReLU(0.01)))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            representation = values
        else:
            representation = torch.cat(
                (self.embedding(values[:, :3]), values[:, 3:4]), dim=1
            )
        return self.mlp(representation).squeeze(1)


class DeepHedger:
    """One configurable public hedger covering the complete stage-1 search."""

    def __init__(
        self,
        market: dict[str, Any],
        objective: dict[str, Any],
        config: DeepHedgerConfig | dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ):
        if config is None:
            config = DeepHedgerConfig()
        elif isinstance(config, dict):
            config = DeepHedgerConfig(**config)
        config.validate()

        self.market = dict(market)
        self.objective = dict(objective)
        self.config = config
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_seed(config.seed)
        self.policy = _Policy(config).to(self.device)
        self.best_epoch: int | None = None
        self.best_public_risk: float | None = None
        self.training_log: list[dict[str, float | int]] = []

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
        time_ratio = 1.0 - step / int(self.market["N"])
        spot = paths[:, step, 0]
        variance = paths[:, step, 1]
        return torch.stack(
            (
                torch.log(spot / float(self.market["K"])),
                variance / float(self.market["theta"]),
                torch.full_like(spot, time_ratio),
                previous_hedge,
            ),
            dim=1,
        )

    def predict_actions(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.device != self.device:
            paths = paths.to(self.device)
        n_paths = len(paths)
        previous = torch.zeros(n_paths, device=self.device, dtype=paths.dtype)
        actions = []
        for step in range(int(self.market["N"])):
            action = self.policy(self._features(paths, step, previous))
            actions.append(action)
            previous = action
        return torch.stack(actions, dim=1)

    def evaluate_tensor(self, paths: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actions = self.predict_actions(paths)
        outputs = backtest_actions(paths.to(self.device), actions, self.market)
        return actions, outputs

    def fit(self, public_paths: torch.Tensor) -> list[dict[str, float | int]]:
        public_paths = public_paths.to(self.device)
        optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        stream = TorchHestonStream(
            self.market,
            seed=10_000_000 + self.config.seed,
            device=self.device,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_risk = math.inf
        start = time.perf_counter()

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
                    "elapsed_seconds": time.perf_counter() - start,
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
        payload = {
            "market": self.market,
            "objective": self.objective,
            "config": asdict(self.config),
            "policy_state": self.policy.state_dict(),
            "best_epoch": self.best_epoch,
            "best_public_risk": self.best_public_risk,
            "training_log": self.training_log,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
    ) -> "DeepHedger":
        map_location = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        hedger = cls(
            market=payload["market"],
            objective=payload["objective"],
            config=payload["config"],
            device=map_location,
        )
        hedger.policy.load_state_dict(payload["policy_state"])
        hedger.best_epoch = payload["best_epoch"]
        hedger.best_public_risk = payload["best_public_risk"]
        hedger.training_log = payload["training_log"]
        return hedger
