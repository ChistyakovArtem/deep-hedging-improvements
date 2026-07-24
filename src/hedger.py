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
from src.reference_hedges import ReferenceHedge, local_black_scholes_delta


@dataclass(frozen=True)
class DeepHedgerConfig:
    architecture: str = "mlp"
    feature_mode: str = "normalized"
    prediction_target: str = "direct"
    action_head: str = "legacy_auto"
    reference_hedge: str = "local_bs"
    time_parameterization: str = "shared"
    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "leaky_relu"
    batch_norm: bool = False
    output_initialization: str = "default"
    n_frequencies: int = 16
    paf_sigma: float = 1.0
    periodic_include_linear: bool = False
    reference_leland_scale: float = 0.0
    reference_no_trade_width: float = 0.0
    reference_grid_moneyness_points: int = 513
    reference_grid_volatility_points: int = 257
    reference_grid_log_moneyness_min: float = -1.5
    reference_grid_log_moneyness_max: float = 1.5
    reference_grid_volatility_min: float = 1e-4
    reference_grid_volatility_max: float = 1.0
    reference_cf_n_quad: int = 256
    reference_cf_phi_min: float = 1e-8
    reference_cf_phi_max: float = 200.0
    reference_cf_batch_size: int = 8192
    use_running_pnl: bool = False
    running_pnl_scale: float = 1.0
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    n_epochs: int = 10_000
    paths_per_epoch: int = 3_000
    validation_interval: int = 200
    seed: int = 0

    def resolved_action_head(self) -> str:
        if self.action_head != "legacy_auto":
            return self.action_head
        if self.architecture == "ntbn":
            return "delta_band"
        if self.prediction_target == "local_bs_deviation":
            return "delta_residual"
        return "direct"

    def validate(self) -> None:
        allowed = {"mlp", "ntbn", "paf_shared", "paf_featurewise"}
        if self.architecture not in allowed:
            raise ValueError(
                f"Unknown architecture {self.architecture!r}; expected {sorted(allowed)}."
            )
        feature_modes = {
            "normalized",
            "ntbn_paper",
            "local_bs_state",
            "legacy_raw",
            "paper_log_state_with_time",
            "paper_log_state",
        }
        if self.feature_mode not in feature_modes:
            raise ValueError(
                f"Unknown feature mode {self.feature_mode!r}; expected {sorted(feature_modes)}."
            )
        if self.time_parameterization not in {"shared", "per_step"}:
            raise ValueError("time_parameterization must be 'shared' or 'per_step'.")
        if self.prediction_target not in {"direct", "local_bs_deviation"}:
            raise ValueError("prediction_target must be 'direct' or 'local_bs_deviation'.")
        if self.action_head not in {
            "legacy_auto",
            "direct",
            "delta_residual",
            "delta_band",
        }:
            raise ValueError(
                "action_head must be legacy_auto, direct, delta_residual, or delta_band."
            )
        if self.reference_hedge not in {
            "local_bs",
            "leland_local_vol",
            "heston_mv_no_trade",
        }:
            raise ValueError(
                "reference_hedge must be local_bs, leland_local_vol, or heston_mv_no_trade."
            )
        if self.activation not in {"leaky_relu", "relu"}:
            raise ValueError("activation must be 'leaky_relu' or 'relu'.")
        if self.output_initialization not in {"default", "zero_last"}:
            raise ValueError("output_initialization must be 'default' or 'zero_last'.")
        if self.architecture in {"paf_shared", "paf_featurewise"} and (
            self.feature_mode != "normalized" or self.time_parameterization != "shared"
        ):
            raise ValueError(
                "Periodic architectures currently require normalized features "
                "and shared time parameterization."
            )
        if self.architecture == "ntbn" and (
            self.feature_mode != "ntbn_paper" or self.time_parameterization != "shared"
        ):
            raise ValueError(
                "NTBN requires ntbn_paper features and shared time parameterization."
            )
        if self.architecture == "ntbn" and self.prediction_target != "direct":
            raise ValueError("NTBN defines its own band output and requires direct target.")
        if self.architecture == "ntbn" and self.action_head not in {
            "legacy_auto",
            "delta_band",
        }:
            raise ValueError("The legacy NTBN architecture requires the delta-band head.")
        if self.action_head != "legacy_auto" and self.prediction_target != "direct":
            raise ValueError(
                "Explicit action_head configs must leave prediction_target=direct."
            )
        if self.resolved_action_head() == "direct" and self.reference_hedge != "local_bs":
            raise ValueError("Direct policies do not use a reference hedge.")
        if self.reference_leland_scale < 0.0:
            raise ValueError("reference_leland_scale must be non-negative.")
        if self.reference_no_trade_width < 0.0:
            raise ValueError("reference_no_trade_width must be non-negative.")
        if self.running_pnl_scale <= 0.0:
            raise ValueError("running_pnl_scale must be positive.")
        if self.time_parameterization == "per_step" and self.feature_mode != "paper_log_state":
            raise ValueError(
                "Per-step networks require paper_log_state, where time is "
                "implicit in the selected network."
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
        self.frequencies = nn.Parameter(sigma * torch.randn(input_dim, n_frequencies))

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
            phase = 2.0 * math.pi * values.unsqueeze(-1) * self.frequencies.unsqueeze(0)
            parts = [torch.sin(phase), torch.cos(phase)]
        if self.include_linear:
            parts.insert(0, phase)
        embedded = torch.cat(parts, dim=-1)
        return embedded.flatten(start_dim=1)


class _Policy(nn.Module):
    def __init__(
        self,
        config: DeepHedgerConfig,
        input_dim: int,
        previous_hedge_index: int,
    ):
        super().__init__()
        self.architecture = config.architecture
        self.action_head = config.resolved_action_head()
        self.previous_hedge_index = previous_hedge_index
        if config.architecture in {"mlp", "ntbn"}:
            self.embedding = None
            if config.architecture == "ntbn":
                # The previous hedge is used by the band clamp, not as an MLP
                # input, matching Imaki et al.'s released implementation.
                input_dim -= 1
        else:
            self.embedding = _PeriodicEmbedding(
                kind=config.architecture,
                input_dim=3,
                n_frequencies=config.n_frequencies,
                sigma=config.paf_sigma,
                include_linear=config.periodic_include_linear,
            )
            input_dim = self.embedding.output_dim + (input_dim - 3)

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.01))
            previous_dim = hidden_dim
        output_dim = 2 if self.action_head == "delta_band" else 1
        output_layer = nn.Linear(previous_dim, output_dim)
        if config.output_initialization == "zero_last":
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _clamp_to_band(
        previous: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        """Paper-compatible clamp, including its inverted-bound fallback."""
        clamped = torch.minimum(torch.maximum(previous, lower), upper)
        return torch.where(lower < upper, clamped, (lower + upper) / 2.0)

    def forward(
        self,
        values: torch.Tensor,
        no_cost_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.architecture == "ntbn":
            representation = torch.cat(
                (
                    values[:, : self.previous_hedge_index],
                    values[:, self.previous_hedge_index + 1 :],
                ),
                dim=1,
            )
        elif self.embedding is None:
            representation = values
        else:
            representation = torch.cat(
                (self.embedding(values[:, :3]), values[:, 3:]),
                dim=1,
            )
        if self.action_head == "delta_band":
            if no_cost_delta is None:
                raise ValueError("A delta-band head requires a reference center.")
            band_width = self.mlp(representation)
            lower = no_cost_delta - F.leaky_relu(band_width[:, 0])
            upper = no_cost_delta + F.leaky_relu(band_width[:, 1])
            return self._clamp_to_band(
                values[:, self.previous_hedge_index],
                lower,
                upper,
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
        input_dim = self._input_dim()
        previous_hedge_index = self._previous_hedge_index()
        if config.time_parameterization == "per_step":
            self.policy: _Policy | nn.ModuleList = nn.ModuleList(
                _Policy(config, input_dim, previous_hedge_index)
                for _ in range(int(self.market["N"]))
            ).to(self.device)
        else:
            self.policy = _Policy(
                config,
                input_dim,
                previous_hedge_index,
            ).to(self.device)
        self.reference_hedge = None
        if config.resolved_action_head() != "direct":
            self.reference_hedge = ReferenceHedge(
                self.market,
                config.reference_hedge,
                self.device,
                leland_scale=config.reference_leland_scale,
                no_trade_width=config.reference_no_trade_width,
                grid_moneyness_points=config.reference_grid_moneyness_points,
                grid_volatility_points=config.reference_grid_volatility_points,
                grid_log_moneyness_min=(config.reference_grid_log_moneyness_min),
                grid_log_moneyness_max=(config.reference_grid_log_moneyness_max),
                grid_volatility_min=config.reference_grid_volatility_min,
                grid_volatility_max=config.reference_grid_volatility_max,
                cf_n_quad=config.reference_cf_n_quad,
                cf_phi_min=config.reference_cf_phi_min,
                cf_phi_max=config.reference_cf_phi_max,
                cf_batch_size=config.reference_cf_batch_size,
            )
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
        running_pnl: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_ratio = 1.0 - step / int(self.market["N"])
        spot = paths[:, step, 0]
        variance = paths[:, step, 1]
        time_feature = torch.full_like(spot, time_ratio)
        if self.config.feature_mode == "legacy_raw":
            values = (spot, variance, time_feature, previous_hedge)
        elif self.config.feature_mode == "normalized":
            values = (
                torch.log(spot / float(self.market["K"])),
                variance / float(self.market["theta"]),
                time_feature,
                previous_hedge,
            )
        elif self.config.feature_mode in {"ntbn_paper", "local_bs_state"}:
            expiry = torch.full_like(
                spot,
                float(self.market["T"]) * time_ratio,
            )
            values = (
                torch.log(spot / float(self.market["K"])),
                expiry,
                torch.sqrt(variance.clamp_min(torch.finfo(variance.dtype).tiny)),
                previous_hedge,
            )
        elif self.config.feature_mode == "paper_log_state_with_time":
            values = (torch.log(spot), variance, time_feature, previous_hedge)
        elif self.config.feature_mode == "paper_log_state":
            values = (torch.log(spot), variance, previous_hedge)
        else:  # guarded by DeepHedgerConfig.validate()
            raise RuntimeError(f"Unsupported feature mode: {self.config.feature_mode}")
        if self.config.use_running_pnl:
            if running_pnl is None:
                raise ValueError("Running PnL is enabled but was not supplied.")
            values = (
                *values,
                running_pnl / float(self.config.running_pnl_scale),
            )
        return torch.stack(values, dim=1)

    def _input_dim(self) -> int:
        base = 3 if self.config.feature_mode == "paper_log_state" else 4
        return base + int(self.config.use_running_pnl)

    def _previous_hedge_index(self) -> int:
        return 2 if self.config.feature_mode == "paper_log_state" else 3

    def _policy_for_step(self, step: int) -> _Policy:
        if isinstance(self.policy, nn.ModuleList):
            return self.policy[step]
        return self.policy

    def _local_bs_delta(
        self,
        paths: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Local Black--Scholes delta from the current Heston state."""
        return local_black_scholes_delta(paths, self.market, step)

    def predict_actions(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.device != self.device:
            paths = paths.to(self.device)
        n_paths = len(paths)
        previous = torch.zeros(n_paths, device=self.device, dtype=paths.dtype)
        previous_reference = torch.zeros_like(previous)
        cash = torch.zeros_like(previous)
        action_head = self.config.resolved_action_head()
        actions = []
        for step in range(int(self.market["N"])):
            discounted_spot = None
            running_pnl = None
            if self.config.use_running_pnl:
                time = float(self.market["T"]) * step / int(self.market["N"])
                discounted_spot = paths[:, step, 0] * math.exp(
                    -float(self.market["r"]) * time
                )
                running_pnl = cash + previous * discounted_spot
            features = self._features(
                paths,
                step,
                previous,
                running_pnl=running_pnl,
            )
            policy = self._policy_for_step(step)
            reference = None
            if self.reference_hedge is not None:
                reference = self.reference_hedge.step(
                    paths,
                    step,
                    previous_reference,
                )
            if action_head == "delta_band":
                action = policy(
                    features,
                    no_cost_delta=reference,
                )
            elif action_head == "delta_residual":
                if reference is None:
                    raise RuntimeError("A residual head requires a reference.")
                action = reference + policy(features)
            else:
                action = policy(features)
            actions.append(action)
            if discounted_spot is not None:
                trade = action - previous
                cash = (
                    cash
                    - trade * discounted_spot
                    - float(self.market["transaction_cost"])
                    * trade.abs()
                    * discounted_spot
                )
            previous = action
            if reference is not None:
                previous_reference = reference
        return torch.stack(actions, dim=1)

    def evaluate_tensor(
        self, paths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actions = self.predict_actions(paths)
        outputs = backtest_actions(paths.to(self.device), actions, self.market)
        return actions, outputs

    def fit(self, public_paths: torch.Tensor) -> list[dict[str, float | int]]:
        public_paths = public_paths.to(self.device)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
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
