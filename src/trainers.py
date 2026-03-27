"""
trainers.py — RL training algorithms for Deep Hedging.

Interface
─────────
    trainer = XxxTrainer(cfg, sampler, paths_val_t, paths_test_t, device,
                         risk_aversion, K, cost)
    log     = trainer.fit()

    trainer.best_net   — weights of the checkpoint with lowest val loss
    log : list[dict]   {"epoch", "val_loss", "elapsed", ...}

Validation / early stopping
────────────────────────────
Every log_every epochs the val loss is evaluated on paths_val_t.
If it does not improve for `early_stop_patience` consecutive checks,
training stops and best_net weights are restored.
Set early_stop_patience=0 to disable early stopping.

Available trainers
──────────────────
DeepHedgingTrainer
    Vanilla SoftMin (Buehler et al. 2019).
    init_mode: "default" | "zero_last" | "pretrain_delta"

SACTrainer
    SoftMin + entropy bonus with annealed β  (Haarnoja et al. 2018 SAC).
    Encourages exploration early in training.

ExponentialBellmanTrainer
    Multiplicative Bellman equation for entropic risk:
        V(s) = exp(-a·r_t) · V_target(s')
    Uses a frozen target network updated via Polyak averaging.
    Ref: Bielecki & Pliska (1999); Buehler et al. Appendix B.

PPOStyleTrainer
    Proximal Policy Optimisation adapted to continuous hedging.
    Clipped importance-weight objective limits step size per iteration.
    Ref: Schulman et al. (2017) PPO.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.backtest import torch_backtest
from src.metrics  import SoftMin
from src.models   import ModelConfig, build_policy


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    # model architecture
    model: ModelConfig = field(default_factory=ModelConfig)

    # backtest feature mode
    use_pnl: bool = False          # include running P&L as extra feature
    pnl_start_epoch: int = 0

    # optimiser
    optimizer: str   = "adam"      # "adam" | "adamw"
    lr:        float = 1e-3

    # training loop
    n_epochs:  int = 10_000
    M_train:   int = 3_000
    log_every: int = 200

    # early stopping & best-model checkpoint
    # number of consecutive val checks without improvement before stopping
    # e.g. log_every=200, patience=100 means no improvement over 20 000 epochs
    # set to 0 to disable
    early_stop_patience: int = 100

    # init mode
    init_mode:       str = "default"   # "default" | "zero_last" | "pretrain_delta"
    pretrain_epochs: int = 500

    # SAC
    beta_start: float = 0.05
    beta_end:   float = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_optimizer(name: str, params, lr: float) -> optim.Optimizer:
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name!r}. Choose 'adam' or 'adamw'.")


def _eval_loss(net, paths_t: torch.Tensor,
               K: float, cost: float, a: float, use_pnl: bool) -> float:
    net.eval()
    
    with torch.no_grad():
        pnl, _ = torch_backtest(paths_t, K, cost=cost,
                                hedge_policy=net, use_pnl=use_pnl)
        loss = SoftMin(pnl, a).item()
    
    net.train()
    
    return loss

def _mse_delta_loss(paths_t: torch.Tensor, net,
                    delta_fn: Callable, use_pnl: bool,
                    K: float, cost: float) -> torch.Tensor:
    """MSE(net output, analytic delta) averaged over time steps. Pretrain phase."""
    M, N_plus1, _ = paths_t.shape
    N      = N_plus1 - 1
    device = paths_t.device
    S      = paths_t[:, :, 0]
    tau    = torch.linspace(1.0, 0.0, N_plus1, device=device)
    mse_fn = nn.MSELoss()

    cash       = torch.zeros(M, device=device)
    delta_prev = torch.zeros(M, device=device)
    total      = torch.tensor(0.0, device=device)

    for t in range(N):
        state_t = paths_t[:, t, :]
        target  = delta_fn(state_t, t)

        parts = [state_t, tau[t].expand(M, 1), delta_prev.unsqueeze(1)]
        if use_pnl:
            parts.append((cash + delta_prev * S[:, t]).unsqueeze(1))
        pred  = net(torch.cat(parts, dim=1)).squeeze(1)
        total = total + mse_fn(pred, target)

        with torch.no_grad():
            d          = pred.detach() - delta_prev
            cash       = cash - d * S[:, t] - cost * torch.abs(d) * S[:, t]
            delta_prev = pred.detach()

    return total / N


# ─────────────────────────────────────────────────────────────────────────────
# Base trainer
# ─────────────────────────────────────────────────────────────────────────────

class _BaseTrainer:
    def __init__(self, cfg: TrainerConfig,
                 sampler,
                 paths_val_t:  torch.Tensor,
                 paths_test_t: torch.Tensor,
                 device:       torch.device,
                 risk_aversion: float,
                 K:    float,
                 cost: float):
        self.cfg          = cfg
        self.sampler      = sampler
        self.paths_val_t  = paths_val_t
        self.paths_test_t = paths_test_t
        self.device       = device
        self.a            = risk_aversion
        self.K            = K
        self.cost         = cost

        input_dim = sampler.state_dim + (3 if cfg.use_pnl else 2)
        model_cfg = ModelConfig(**{**cfg.model.__dict__,
                                   "zero_last_layer": cfg.init_mode == "zero_last"})
        
        self.net      = build_policy(model_cfg, input_dim).to(device)
        def count_parameters(model: nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {count_parameters(self.net):,}")
        self.best_net = copy.deepcopy(self.net)

        self._best_val   = float("inf")
        self._no_improve = 0

    def _sample(self) -> torch.Tensor:
        return torch.tensor(
            self.sampler.sample(self.cfg.M_train),
            dtype=torch.float32, device=self.device
        )
    

    def eval_on_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Eval best_net on test paths. Override if backtest differs."""
        self.net.eval()
        with torch.no_grad():
            pnl, fees = torch_backtest(
                self.paths_test_t, self.K, cost=self.cost,
                hedge_policy=self.net,
                use_pnl=self.cfg.use_pnl,
                detach=True
            )
        self.net.train()
        return pnl, fees

    def _check_val(self, epoch: int, t0: float,
                   extra: Optional[dict] = None) -> tuple[dict, bool]:
        """
        Evaluate val loss, update best checkpoint, check early stopping.
        Returns (log_entry, should_stop).
        """
        val_loss = _eval_loss(self.net, self.paths_val_t,
                              self.K, self.cost, self.a, self.cfg.use_pnl)
        entry = {"epoch": epoch, "val_loss": val_loss,
                 "elapsed": time.perf_counter() - t0}
        if extra:
            entry.update(extra)

        if val_loss < self._best_val:
            self._best_val   = val_loss
            self._no_improve = 0
            self.best_net    = copy.deepcopy(self.net)
        else:
            self._no_improve += 1

        patience = self.cfg.early_stop_patience
        stop = (patience > 0) and (self._no_improve >= patience)
        return entry, stop

    def _restore_best(self):
        self.net.load_state_dict(self.best_net.state_dict())

    def fit(self) -> List[dict]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deep Hedging
# ─────────────────────────────────────────────────────────────────────────────

class DeepHedgingTrainer(_BaseTrainer):
    """
    Minimise SoftMin(PnL) via SGD  (Buehler et al. 2019).

    init_mode options
    -----------------
    "default"        — random init, SoftMin from epoch 1
    "zero_last"      — last layer zeroed; net starts predicting δ ≡ 0
    "pretrain_delta" — Phase 1: MSE vs analytic delta
                       Phase 2: SoftMin fine-tune
                       Requires delta_fn.
    """

    def __init__(self, cfg: TrainerConfig,
                 sampler,
                 paths_val_t:  torch.Tensor,
                 paths_test_t: torch.Tensor,
                 device:       torch.device,
                 risk_aversion: float,
                 K:    float,
                 cost: float,
                 delta_fn: Optional[Callable] = None):
        super().__init__(cfg, sampler, paths_val_t, paths_test_t,
                         device, risk_aversion, K, cost)
        self.delta_fn  = delta_fn
        self.optimizer = _make_optimizer(cfg.optimizer,
                                         self.net.parameters(), cfg.lr)
        if cfg.init_mode == "pretrain_delta" and delta_fn is None:
            raise ValueError("init_mode='pretrain_delta' requires delta_fn.")

    def fit(self) -> List[dict]:
        cfg = self.cfg
        log: List[dict] = []
        t0  = time.perf_counter()

        for epoch in tqdm(range(1, cfg.n_epochs + 1)):
            paths       = self._sample()
            in_pretrain = (cfg.init_mode == "pretrain_delta"
                           and epoch <= cfg.pretrain_epochs)
            self.optimizer.zero_grad()

            if in_pretrain:
                loss = _mse_delta_loss(paths, self.net, self.delta_fn,
                                       cfg.use_pnl, self.K, self.cost)
            else:
                pnl, _ = torch_backtest(paths, self.K, cost=self.cost,
                                        hedge_policy=self.net, pnl_start_epoch=cfg.pnl_start_epoch,
                                        use_pnl=cfg.use_pnl, epoch=epoch)
                loss = SoftMin(pnl, self.a)

            loss.backward()
            self.optimizer.step()

            if epoch % cfg.log_every == 0:
                extra = {}
                if cfg.init_mode == "pretrain_delta":
                    extra["phase"] = "pretrain" if in_pretrain else "finetune"
                entry, stop = self._check_val(epoch, t0, extra)
                log.append(entry)
                if stop and not in_pretrain:
                    break

        self._restore_best()
        return log


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAC (entropy regularisation)
# ─────────────────────────────────────────────────────────────────────────────

from tqdm import tqdm

class SACTrainer(_BaseTrainer):
    """
    Soft Actor-Critic style with reparameterization trick.

    Policy: π(δ|state) = N(mu(state), exp(log_std)²)
    Action: δ = mu + exp(log_std) * ε,  ε ~ N(0, 1)

    Градиент идёт через mu И через log_std (reparameterization trick),
    поэтому log_std реально влияет на PnL и не убегает в бесконечность.

    Loss = SoftMin(PnL) - β(t) · H(π)
    H(π) = 0.5*(1 + log(2π)) + log_std   (энтропия гауссианы)

    β анилируется beta_start → beta_end:
    - высокий β в начале → большой std → агент исследует
    - β → 0 к концу → std сжимается, политика детерминированная
    """

    def __init__(self, cfg: TrainerConfig,
                 sampler,
                 paths_val_t:  torch.Tensor,
                 paths_test_t: torch.Tensor,
                 device:       torch.device,
                 risk_aversion: float,
                 K:    float,
                 cost: float):
        super().__init__(cfg, sampler, paths_val_t, paths_test_t,
                         device, risk_aversion, K, cost)
        # log_std — обучаемый параметр, общий для всех состояний
        self.log_std   = nn.Parameter(torch.tensor(0.0, device=device))
        self.optimizer = _make_optimizer(
            cfg.optimizer,
            list(self.net.parameters()) + [self.log_std],
            cfg.lr
        )

    def _beta(self, epoch: int) -> float:
        frac = epoch / max(self.cfg.n_epochs, 1)
        return self.cfg.beta_start * (self.cfg.beta_end / self.cfg.beta_start) ** frac

    def _stochastic_backtest(self, paths_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backtest со стохастической политикой через reparameterization.
        δ_t = mu_t + exp(log_std) * ε_t,  ε_t ~ N(0,1)
        Градиент идёт через mu_t (сеть) и log_std.
        """
        M, N_plus1, state_dim = paths_t.shape
        N      = N_plus1 - 1
        device = paths_t.device
        S      = paths_t[:, :, 0]
        std    = self.log_std.exp()

        cash       = torch.zeros(M, device=device)
        delta_prev = torch.zeros(M, device=device)
        total_fees = torch.zeros(M, device=device)
        tau        = torch.linspace(1.0, 0.0, N_plus1, device=device)

        for t in range(N):
            state_t = paths_t[:, t, :]
            parts   = [state_t, tau[t].expand(M, 1), delta_prev.unsqueeze(1)]
            if self.cfg.use_pnl:
                parts.append((cash + delta_prev * S[:, t]).unsqueeze(1))

            mu    = self.net(torch.cat(parts, dim=1)).squeeze(1)
            eps   = torch.randn_like(mu)           # ε ~ N(0, 1)
            delta = mu + std * eps                 # reparameterization

            d_delta    = delta - delta_prev
            fees       = self.cost * torch.abs(d_delta) * S[:, t]
            cash       = cash - d_delta * S[:, t] - fees
            total_fees = total_fees + fees
            delta_prev = delta

        cash += delta_prev * S[:, -1]
        pnl   = cash - torch.clamp(S[:, -1] - self.K, min=0.0)
        return pnl, total_fees

    def fit(self) -> List[dict]:
        cfg = self.cfg
        log: List[dict] = []
        t0  = time.perf_counter()

        for epoch in tqdm(range(1, cfg.n_epochs + 1)):
            paths = self._sample()
            self.optimizer.zero_grad()

            pnl, _  = self._stochastic_backtest(paths)

            # энтропия N(mu, std²): 0.5*(1+log(2π)) + log_std
            entropy = 0.5 * (1.0 + math.log(2 * math.pi)) + self.log_std
            beta    = self._beta(epoch)
            loss    = SoftMin(pnl, self.a) - beta * entropy

            loss.backward()
            self.optimizer.step()

            if epoch % cfg.log_every == 0:
                # на валидации используем детерминированную политику (std=0)
                entry, stop = self._check_val(epoch, t0, {
                    "log_std": self.log_std.item(),
                    "std":     self.log_std.exp().item(),
                    "beta":    beta,
                })
                log.append(entry)
                if stop:
                    break

        self._restore_best()
        return log


# ─────────────────────────────────────────────────────────────────────────────
# 3. Exponential Bellman + target network
# ─────────────────────────────────────────────────────────────────────────────

# Why we update V net (target net) with Q net (delta net)
# What is that             v_current  = torch.exp(-self.a * (cash + delta_prev * S[:, t]))
class ExponentialBellmanTrainer(_BaseTrainer):
    """
    Multiplicative Bellman equation for entropic risk.

    Standard additive Bellman:  V(s) = r  + γ · V(s')
    Exponential (this):         V(s) = exp(-a·r_t) · V_target(s')

    The per-step "discount" exp(-a·r_t) replaces the fixed γ and encodes
    how much a step cost hurts under exponential utility.

    TD loss:
        L = Σ_t E[ ( V_current(s_t) - exp(-a·r_t)·V_target(s_{t+1}) )² ]

    V_current ≈ exp(-a · running_pnl_t),
    V_target  = target network (frozen, Polyak-updated each epoch).

    References: Bielecki & Pliska (1999); Buehler et al. (2019) Appendix B.
    """

    def __init__(self, cfg: TrainerConfig,
                 sampler,
                 paths_val_t:  torch.Tensor,
                 paths_test_t: torch.Tensor,
                 device:       torch.device,
                 risk_aversion: float,
                 K:    float,
                 cost: float):
        super().__init__(cfg, sampler, paths_val_t, paths_test_t,
                         device, risk_aversion, K, cost)
        self.optimizer  = _make_optimizer(cfg.optimizer,
                                          self.net.parameters(), cfg.lr)
        self.target_net = copy.deepcopy(self.net).to(device)
        for p in self.target_net.parameters():
            p.requires_grad_(False)

    def _polyak_update(self):
        tau = self.cfg.polyak_tau
        for p, pt in zip(self.net.parameters(),
                         self.target_net.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    def _bellman_loss(self, paths_t: torch.Tensor) -> torch.Tensor:
        M, N_plus1, _ = paths_t.shape
        N      = N_plus1 - 1
        device = paths_t.device
        S      = paths_t[:, :, 0]
        tau_g  = torch.linspace(1.0, 0.0, N_plus1, device=device)
        payoff_T = torch.clamp(S[:, -1] - self.K, min=0.0)

        cash       = torch.zeros(M, device=device)
        delta_prev = torch.zeros(M, device=device)
        total_loss = torch.tensor(0.0, device=device)

        for t in range(N):
            state_t = paths_t[:, t, :]
            parts   = [state_t, tau_g[t].expand(M, 1), delta_prev.unsqueeze(1)]
            if self.cfg.use_pnl:
                parts.append((cash + delta_prev * S[:, t]).unsqueeze(1))
            delta = self.net(torch.cat(parts, dim=1)).squeeze(1)

            d_delta  = delta - delta_prev
            fee      = self.cost * torch.abs(d_delta) * S[:, t]
            new_cash = cash - d_delta * S[:, t] - fee

            # reward = - fee
            step_discount = torch.exp(-self.a * (-fee))

            with torch.no_grad():
                pnl_next   = new_cash + delta * S[:, t + 1]
                parts_next = [paths_t[:, t + 1, :],
                               tau_g[t + 1].expand(M, 1),
                               delta.unsqueeze(1)]
                if self.cfg.use_pnl:
                    parts_next.append(pnl_next.unsqueeze(1))

                if t < N - 1:
                    v_target = self.target_net(
                        torch.cat(parts_next, dim=1)
                    ).squeeze(1)
                else:
                    final_pnl = new_cash + delta * S[:, -1] - payoff_T
                    v_target  = torch.exp(-self.a * final_pnl)

            v_current  = torch.exp(-self.a * (cash + delta_prev * S[:, t]))
            total_loss = total_loss + (
                (v_current - step_discount * v_target) ** 2
            ).mean()

            cash       = new_cash.detach()
            delta_prev = delta.detach()

        return total_loss / N

    def fit(self) -> List[dict]:
        cfg = self.cfg
        log: List[dict] = []
        t0  = time.perf_counter()

        for epoch in tqdm(range(1, cfg.n_epochs + 1)):
            paths = self._sample()
            self.optimizer.zero_grad()
            loss = self._bellman_loss(paths)
            loss.backward()
            self.optimizer.step()
            self._polyak_update()

            if epoch % cfg.log_every == 0:
                entry, stop = self._check_val(epoch, t0)
                log.append(entry)
                if stop:
                    break

        self._restore_best()
        return log


# ─────────────────────────────────────────────────────────────────────────────
# 4. PPO-style
# ─────────────────────────────────────────────────────────────────────────────

# actions_new == actions_old ???

# ration == (or ~) 1 -> loss = - adv = --pnl = pnl -> мы минимизируем E[PnL] или что мы делаем?
# surr1 = ratio * advantage
# surr2 = torch.clamp(ratio,
#                     1 - cfg.clip_eps,
#                     1 + cfg.clip_eps) * advantage
# loss  = -torch.min(surr1, surr2).mean()
# loss.backward()

class PPOStyleTrainer(_BaseTrainer):
    """
    Proximal Policy Optimisation for Deep Hedging.

    Motivation
    ──────────
    Vanilla SoftMin gradient can make large, destabilising updates when
    PnL variance is high early in training (especially under Heston).
    PPO's clipped surrogate objective hard-limits the policy step size.

    Algorithm (each epoch)
    ──────────────────────
    1. Collect M_train paths + actions with the current ("old") policy.
    2. Compute advantage:  A = -a · PnL   (risk-adjusted, per path).
    3. For n_ppo_epochs inner gradient steps:
           ρ   = π_new(δ) / π_old(δ)  averaged over time steps
           L   = -mean[ min(ρ·A,  clip(ρ, 1-ε, 1+ε)·A) ]
           gradient step on L.

    Reference: Schulman et al. (2017) PPO.
    """

    def __init__(self, cfg: TrainerConfig,
                 sampler,
                 paths_val_t:  torch.Tensor,
                 paths_test_t: torch.Tensor,
                 device:       torch.device,
                 risk_aversion: float,
                 K:    float,
                 cost: float):
        super().__init__(cfg, sampler, paths_val_t, paths_test_t,
                         device, risk_aversion, K, cost)
        self.optimizer = _make_optimizer(cfg.optimizer,
                                         self.net.parameters(), cfg.lr)

    def _collect(self, paths_t: torch.Tensor,
                 net) -> tuple[torch.Tensor, torch.Tensor]:
        """Actions (M, N) and terminal PnL (M,) under `net`."""
        M, N_plus1, _ = paths_t.shape
        N      = N_plus1 - 1
        device = paths_t.device
        S      = paths_t[:, :, 0]
        tau_g  = torch.linspace(1.0, 0.0, N_plus1, device=device)

        cash       = torch.zeros(M, device=device)
        delta_prev = torch.zeros(M, device=device)
        actions    = []

        for t in range(N):
            state_t = paths_t[:, t, :]
            parts   = [state_t, tau_g[t].expand(M, 1), delta_prev.unsqueeze(1)]
            if self.cfg.use_pnl:
                parts.append((cash + delta_prev * S[:, t]).unsqueeze(1))
            delta = net(torch.cat(parts, dim=1)).squeeze(1)
            actions.append(delta.unsqueeze(1))

            d_delta    = delta - delta_prev
            fee        = self.cost * torch.abs(d_delta) * S[:, t]
            cash       = cash - d_delta * S[:, t] - fee
            delta_prev = delta.detach()

        actions_t = torch.cat(actions, dim=1)
        cash     += delta_prev * S[:, -1]
        pnl       = cash - torch.clamp(S[:, -1] - self.K, min=0.0)
        return actions_t, pnl

    @staticmethod
    def _log_prob(x: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
        return (-0.5 * ((x - mu) / sigma) ** 2
                - math.log(sigma)
                - 0.5 * math.log(2 * math.pi))

    def fit(self) -> List[dict]:
        cfg = self.cfg
        log: List[dict] = []
        t0  = time.perf_counter()

        for epoch in tqdm(range(1, cfg.n_epochs + 1)):
            paths = self._sample()

            # collect with frozen old policy
            old_net = copy.deepcopy(self.net)
            old_net.eval()
            for p in old_net.parameters():
                p.requires_grad_(False)

            with torch.no_grad():
                actions_old, pnl_old = self._collect(paths, old_net)
                # SoftMin-совместимый advantage
                weights   = torch.softmax(-self.a * pnl_old, dim=0)  # (M,)
                baseline  = (weights * pnl_old).sum()
                advantage = pnl_old - baseline                        # (M,)

            # PPO inner loop
            for _ in range(cfg.n_ppo_epochs):
                self.optimizer.zero_grad()
                actions_new, _ = self._collect(paths, self.net)

                log_ratio = (
                    self._log_prob(actions_new, actions_old, cfg.sigma_policy)
                    - self._log_prob(actions_old, actions_old, cfg.sigma_policy)
                )
                ratio = torch.exp(log_ratio.mean(dim=1))   # (M,)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio,
                                    1 - cfg.clip_eps,
                                    1 + cfg.clip_eps) * advantage
                loss  = -torch.min(surr1, surr2).mean()
                loss.backward()
                self.optimizer.step()

            if epoch % cfg.log_every == 0:
                entry, stop = self._check_val(epoch, t0)
                log.append(entry)
                if stop:
                    break

        self._restore_best()
        return log

from scipy.stats import norm as scipy_norm

def _bsm_delta_torch(S: torch.Tensor, sigma: torch.Tensor,
                     tau: float, K: float, r: float) -> torch.Tensor:
    """
    BSM delta vectorised in torch.
    S     : (M,)  spot
    sigma : (M,)  local vol (const для GBM, sqrt(v_t) для Heston)
    tau   : float time to maturity
    """
    if tau <= 0:
        return (S >= K).float()
    eps   = 1e-8
    sigma = sigma.clamp(min=eps)
    d1    = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * tau ** 0.5)
    # N(d1) через erf
    return 0.5 * (1.0 + torch.erf(d1 / 2.0 ** 0.5))


class DeviationTrainer(_BaseTrainer):
    """
    Net predicts deviation from BSM delta:
        δ_total(t) = δ_BSM(S_t, σ_t, τ_t) + net(features)

    σ_t:
        GBM    — константа sigma из конфига семплера
        Heston — sqrt(v_t) из второго измерения стейта

    zero_last_layer=True в ModelConfig рекомендуется:
        тогда в начале обучения δ_total = δ_BSM (хорошая инициализация).

    Мотивация: сеть учит только поправку к известному аналитическому решению.
    Это облегчает оптимизацию, особенно с use_pnl=True.
    """

    def __init__(self, cfg: TrainerConfig,
                 sampler,
                 paths_val_t:  torch.Tensor,
                 paths_test_t: torch.Tensor,
                 device:       torch.device,
                 risk_aversion: float,
                 K:    float,
                 cost: float,
                 r:    float = 0.01,
                 sigma_const: Optional[float] = None):
        """
        r            : risk-free rate (для BSM формулы)
        sigma_const  : если GBM — передай sigma семплера.
                       если None — берём sqrt(v_t) из стейта (Heston).
        """
        super().__init__(cfg, sampler, paths_val_t, paths_test_t,
                         device, risk_aversion, K, cost)
        self.r           = r
        self.sigma_const = sigma_const   # None → Heston mode
        self.optimizer   = _make_optimizer(cfg.optimizer,
                                           self.net.parameters(), cfg.lr)

    def _bsm_delta(self, state_t: torch.Tensor, tau: float) -> torch.Tensor:
        """
        state_t : (M, state_dim)
        returns : (M,) BSM delta
        """
        S = state_t[:, 0]
        if self.sigma_const is not None:
            # GBM: sigma фиксирована
            sigma = torch.full_like(S, self.sigma_const)
        else:
            # Heston: sigma_t = sqrt(v_t), v_t — второй канал стейта
            sigma = state_t[:, 1].clamp(min=1e-8).sqrt()
        return _bsm_delta_torch(S, sigma, tau, self.K, self.r)

    def _deviation_backtest(self, paths_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backtest где δ_total = δ_BSM + net(features).
        """
        M, N_plus1, state_dim = paths_t.shape
        N      = N_plus1 - 1
        device = paths_t.device
        S      = paths_t[:, :, 0]
        tau_g  = torch.linspace(1.0, 0.0, N_plus1, device=device)

        cash       = torch.zeros(M, device=device)
        delta_prev = torch.zeros(M, device=device)
        total_fees = torch.zeros(M, device=device)

        for t in range(N):
            state_t  = paths_t[:, t, :]
            tau_val  = tau_g[t].item()

            # аналитическая дельта
            delta_bsm = self._bsm_delta(state_t, tau_val)   # (M,)

            # фичи для нейронки — те же что в обычном backtest
            parts = [state_t, tau_g[t].expand(M, 1), delta_prev.unsqueeze(1)]
            if self.cfg.use_pnl:
                parts.append((cash + delta_prev * S[:, t]).unsqueeze(1))
            deviation = self.net(torch.cat(parts, dim=1)).squeeze(1)  # (M,)

            delta = delta_bsm + deviation   # итоговая дельта

            d_delta    = delta - delta_prev
            fees       = self.cost * torch.abs(d_delta) * S[:, t]
            cash       = cash - d_delta * S[:, t] - fees
            total_fees = total_fees + fees
            delta_prev = delta

        cash += delta_prev * S[:, -1]
        pnl   = cash - torch.clamp(S[:, -1] - self.K, min=0.0)
        return pnl, total_fees

    def _eval_deviation_loss(self) -> float:
        """Val loss используя deviation backtest."""
        self.net.eval()
        with torch.no_grad():
            pnl, _ = self._deviation_backtest(self.paths_val_t)
            loss   = SoftMin(pnl, self.a).item()
        self.net.train()
        return loss
    
    def eval_on_test(self) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            pnl, fees = self._deviation_backtest(self.paths_test_t)
        self.net.train()
        return (pnl.cpu().numpy(), fees.cpu().numpy())

    def fit(self) -> List[dict]:
        cfg = self.cfg
        log: List[dict] = []
        t0  = time.perf_counter()

        for epoch in tqdm(range(1, cfg.n_epochs + 1)):
            paths = self._sample()
            self.optimizer.zero_grad()

            pnl, _ = self._deviation_backtest(paths)
            loss   = SoftMin(pnl, self.a)

            loss.backward()
            self.optimizer.step()

            if epoch % cfg.log_every == 0:
                # переопределяем _check_val через deviation loss
                val_loss = self._eval_deviation_loss()
                entry    = {"epoch": epoch, "val_loss": val_loss,
                            "elapsed": time.perf_counter() - t0}

                if val_loss < self._best_val:
                    self._best_val   = val_loss
                    self._no_improve = 0
                    self.best_net    = copy.deepcopy(self.net)
                else:
                    self._no_improve += 1

                log.append(entry)

                patience = cfg.early_stop_patience
                if (patience > 0) and (self._no_improve >= patience):
                    break

        self._restore_best()
        return log