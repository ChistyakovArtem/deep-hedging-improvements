"""
deltas.py — analytical and regression-based delta hedging baselines.

Classes
───────
LSMCDeltaHedger
    Regression-based delta via Longstaff-Schwartz Monte Carlo.
    Fits a polynomial regression on training paths, applies to test paths.
    Fast; model-free given simulated paths.

HestonCFDeltaHedger
    Exact Heston delta via finite-difference bump on the characteristic
    function price (Gil-Pelaez inversion).
    Slow (scipy.quad per path per step); use small M_test for sanity checks.

BSMDeltaHedger
    Analytical Black-Scholes delta. Used as fast proxy under GBM.

Both LSMC and CF hedgers expose a numpy backtest method and a torch-compatible
__call__ for use as delta_exogenous in torch_backtest.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm


# ── BSM ───────────────────────────────────────────────────────────────────────

def _bsm_delta(S: np.ndarray, K: float, r: float,
               sigma: float, tau: float) -> np.ndarray:
    S   = np.asarray(S, dtype=float)
    if tau <= 0:
        return (S >= K).astype(float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


class BSMDeltaHedger:
    """Analytical BSM delta. Compatible with torch_backtest(delta_exogenous=...)."""

    def __init__(self, K: float, r: float, sigma: float, T: float, N: int):
        self.K = K; self.r = r; self.sigma = sigma
        self.T = T; self.dt = T / N

    def __call__(self, state_t, t: int) -> torch.Tensor:
        if isinstance(state_t, torch.Tensor):
            S = state_t[:, 0].cpu().numpy()
        else:
            S = state_t[:, 0]
        tau = self.T - t * self.dt
        d   = _bsm_delta(S, self.K, self.r, self.sigma, tau)
        return torch.tensor(d, dtype=torch.float32)


# ── LSMC ─────────────────────────────────────────────────────────────────────

class LSMCDeltaHedger:
    """
    Delta estimated via polynomial regression on simulated paths.

    Features per (path, step):  [1, t, S_t, v_t, S_t²]
    Target:                     1_{S_T > K} · S_T / S_t   (digital-weighted payoff proxy)

    Parameters
    ----------
    K    : strike
    cost : transaction cost (used in numpy_backtest only)
    """

    def __init__(self, K: float, cost: float = 1e-3):
        self.K    = K
        self.cost = cost
        self.beta: np.ndarray | None = None

    # ── fitting ───────────────────────────────────────────

    def fit(self, paths_train: np.ndarray) -> "LSMCDeltaHedger":
        """
        paths_train : (M, N+1, state_dim)  — Heston paths (state_dim >= 2)
        """
        S_train = paths_train[:, :, 0]
        v_train = paths_train[:, :, 1] if paths_train.shape[2] > 1 \
                  else np.zeros_like(S_train)
        S_T     = S_train[:, -1]
        M, N1   = S_train.shape
        N       = N1 - 1

        X_all, Y_all = [], []
        for t in range(N):
            S_t = S_train[:, t]
            v_t = v_train[:, t]
            Y_t = (S_T > self.K).astype(float) * (S_T / S_t)
            X_t = np.column_stack([np.ones_like(S_t), np.ones_like(S_t) * t,
                                    S_t, v_t, S_t ** 2])
            X_all.append(X_t); Y_all.append(Y_t)

        self.beta, *_ = np.linalg.lstsq(
            np.vstack(X_all), np.concatenate(Y_all), rcond=None
        )
        return self

    # ── apply delta at time t ─────────────────────────────

    def _delta(self, S_t: np.ndarray, v_t: np.ndarray, t: int) -> np.ndarray:
        assert self.beta is not None, "Call .fit() before using the hedger."
        X = np.column_stack([np.ones_like(S_t), np.ones_like(S_t) * t,
                              S_t, v_t, S_t ** 2])
        return np.clip(X @ self.beta, 0.0, 1.0)

    # ── torch-compatible interface ────────────────────────

    def __call__(self, state_t: torch.Tensor, t: int) -> torch.Tensor:
        S_t = state_t[:, 0].cpu().numpy()
        v_t = state_t[:, 1].cpu().numpy() if state_t.shape[1] > 1 \
              else np.zeros_like(S_t)
        d   = self._delta(S_t, v_t, t)
        return torch.tensor(d, dtype=torch.float32, device=state_t.device)

    # ── standalone numpy backtest ─────────────────────────

    def backtest(self, paths_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run full backtest on test paths. Returns (pnl, total_fees)."""
        assert self.beta is not None, "Call .fit() before backtesting."
        S_test = paths_test[:, :, 0]
        v_test = paths_test[:, :, 1] if paths_test.shape[2] > 1 \
                 else np.zeros_like(S_test)
        M, N1  = S_test.shape
        N      = N1 - 1

        cash       = np.zeros(M)
        delta_prev = np.zeros(M)
        total_fees = np.zeros(M)

        for t in range(N):
            delta_t  = self._delta(S_test[:, t], v_test[:, t], t)
            d_delta  = delta_t - delta_prev
            cash    -= d_delta * S_test[:, t]
            fees     = self.cost * np.abs(d_delta) * S_test[:, t]
            cash    -= fees; total_fees += fees
            delta_prev = delta_t

        cash += delta_prev * S_test[:, -1]
        pnl   = cash - np.clip(S_test[:, -1] - self.K, 0, None)
        return pnl, total_fees


# ── Heston CF ─────────────────────────────────────────────────────────────────

def _heston_cf(phi, S, v, r, kappa, theta, xi, rho, tau):
    i = 1j; x = np.log(S)
    d = np.sqrt((rho * xi * i * phi - kappa) ** 2
                + xi ** 2 * (i * phi + phi ** 2))
    g = (kappa - rho * xi * i * phi - d) / (kappa - rho * xi * i * phi + d)
    e = np.exp(-d * tau)
    C = r * i * phi * tau + (kappa * theta / xi ** 2) * (
        (kappa - rho * xi * i * phi - d) * tau
        - 2 * np.log((1 - g * e) / (1 - g))
    )
    D = ((kappa - rho * xi * i * phi - d) / xi ** 2) * ((1 - e) / (1 - g * e))
    return np.exp(C + D * v + i * phi * x)


def _heston_price(S, v, K, r, kappa, theta, xi, rho, tau) -> float:
    if tau <= 0:
        return max(S - K, 0.0)
    p1 = lambda phi: np.real(
        np.exp(-1j * phi * np.log(K))
        * _heston_cf(phi - 1j, S, v, r, kappa, theta, xi, rho, tau)
        / (1j * phi * _heston_cf(-1j, S, v, r, kappa, theta, xi, rho, tau))
    )
    p2 = lambda phi: np.real(
        np.exp(-1j * phi * np.log(K))
        * _heston_cf(phi, S, v, r, kappa, theta, xi, rho, tau)
        / (1j * phi)
    )
    P1 = 0.5 + quad(p1, 1e-8, 200, limit=200, epsabs=1e-6)[0] / np.pi
    P2 = 0.5 + quad(p2, 1e-8, 200, limit=200, epsabs=1e-6)[0] / np.pi
    return S * P1 - K * np.exp(-r * tau) * P2


class HestonCFDeltaHedger:
    """
    Exact Heston delta via central finite-difference bump on CF price.

    ⚠️  Slow: O(M × N) calls to scipy.quad.
        Use M_test ≤ 500 for exploratory runs.

    Parameters
    ----------
    K, r, kappa, theta, xi, rho, T, N : Heston / contract params
    cost    : transaction cost (backtest only)
    eps_rel : relative bump size for finite difference
    """

    def __init__(self, K: float, r: float, kappa: float, theta: float,
                 xi: float, rho: float, T: float, N: int,
                 cost: float = 1e-3, eps_rel: float = 1e-3):
        self.K = K; self.r = r; self.kappa = kappa; self.theta = theta
        self.xi = xi; self.rho = rho; self.T = T; self.dt = T / N
        self.N = N; self.cost = cost; self.eps_rel = eps_rel

    def _delta(self, S_arr: np.ndarray, v_arr: np.ndarray,
               tau: float) -> np.ndarray:
        delta = np.empty(len(S_arr))
        for i in range(len(S_arr)):
            if tau <= 0:
                delta[i] = 1.0 if S_arr[i] >= self.K else 0.0
            else:
                eps = max(S_arr[i] * self.eps_rel, 1e-4)
                p_up = _heston_price(S_arr[i] + eps, v_arr[i], self.K,
                                     self.r, self.kappa, self.theta,
                                     self.xi, self.rho, tau)
                p_dn = _heston_price(S_arr[i] - eps, v_arr[i], self.K,
                                     self.r, self.kappa, self.theta,
                                     self.xi, self.rho, tau)
                delta[i] = (p_up - p_dn) / (2 * eps)
        return delta

    # ── torch-compatible interface ────────────────────────

    def __call__(self, state_t: torch.Tensor, t: int) -> torch.Tensor:
        S   = state_t[:, 0].cpu().numpy()
        v   = state_t[:, 1].cpu().numpy()
        tau = self.T - t * self.dt
        d   = self._delta(S, v, tau)
        return torch.tensor(d, dtype=torch.float32, device=state_t.device)

    # ── standalone numpy backtest ─────────────────────────

    def backtest(self, paths_test: np.ndarray,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        S_test = paths_test[:, :, 0]
        v_test = paths_test[:, :, 1]
        M, N1  = S_test.shape
        N      = N1 - 1

        cash       = np.zeros(M)
        delta_prev = np.zeros(M)
        total_fees = np.zeros(M)

        steps = tqdm(range(N), desc="CF delta") if verbose else range(N)
        for t in steps:
            tau      = self.T - t * self.dt
            delta_t  = self._delta(S_test[:, t], v_test[:, t], tau)
            d_delta  = delta_t - delta_prev
            cash    -= d_delta * S_test[:, t]
            fees     = self.cost * np.abs(d_delta) * S_test[:, t]
            cash    -= fees; total_fees += fees
            delta_prev = delta_t

        cash += delta_prev * S_test[:, -1]
        pnl   = cash - np.clip(S_test[:, -1] - self.K, 0, None)
        return pnl, total_fees
