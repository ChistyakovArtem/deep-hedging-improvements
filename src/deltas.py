"""
deltas.py — analytical and regression-based delta hedging baselines.

Classes
───────
LSMCDeltaHedger
    Regression-based delta via Longstaff-Schwartz Monte Carlo.
    Fits a polynomial regression on training paths, applies to test paths.
    Fast; model-free given simulated paths.

HestonCFDeltaHedger
    Heston delta via a batched torch/GPU characteristic-function price and
    autograd. Keeps the old scipy finite-difference path as a reference.

HestonMVDeltaHedger
    Minimum-variance / Bartlett Heston delta:
        dC/dS + (rho * xi / S) * dC/dv.

HestonMVNoTradeBandHedger
    Transaction-cost baseline: keep the current hedge while it remains inside
    a no-trade band around the MV/Bartlett delta.

LelandHestonDeltaHedger
    BSM/local-vol proxy with Leland's transaction-cost volatility uplift.

BackwardQuadraticHedger
    Regression-fitted discrete-time variance-optimal hedge.

BSMDeltaHedger
    Analytical Black-Scholes delta. Used as fast proxy under GBM.

All hedgers expose a numpy backtest method and a torch-compatible __call__ for
use as delta_exogenous in torch_backtest.
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


def _heston_cf_torch(phi, S, v, r, kappa, theta, xi, rho, tau):
    """Torch version of _heston_cf, broadcast over state and quadrature nodes."""
    cdtype = torch.complex128 if S.dtype == torch.float64 else torch.complex64
    phi = phi.to(dtype=cdtype)
    S_c = S.to(dtype=cdtype)
    v_c = v.to(dtype=cdtype)
    tau = torch.as_tensor(tau, dtype=S.dtype, device=S.device).to(dtype=cdtype)

    i = torch.tensor(1j, dtype=cdtype, device=S.device)
    x = torch.log(S_c)
    d = torch.sqrt((rho * xi * i * phi - kappa) ** 2
                   + xi ** 2 * (i * phi + phi ** 2))
    g = (kappa - rho * xi * i * phi - d) / (kappa - rho * xi * i * phi + d)
    e = torch.exp(-d * tau)
    C = r * i * phi * tau + (kappa * theta / xi ** 2) * (
        (kappa - rho * xi * i * phi - d) * tau
        - 2 * torch.log((1 - g * e) / (1 - g))
    )
    D = ((kappa - rho * xi * i * phi - d) / xi ** 2) * (
        (1 - e) / (1 - g * e)
    )
    return torch.exp(C + D * v_c + i * phi * x)


class HestonCFDeltaHedger:
    """
    Heston delta via a batched characteristic-function price.

    The live path is torch/autograd:
      1. price all states at a timestep with fixed Gauss-Legendre nodes;
      2. one backward pass gives dC/dS and dC/dv;
      3. subclasses can combine those Greeks into hedge ratios.

    The old scipy finite-difference implementation is still available through
    _delta_scipy for validation against the previous notebook baseline.

    Parameters
    ----------
    K, r, kappa, theta, xi, rho, T, N : Heston / contract params
    cost       : transaction cost (backtest only)
    eps_rel    : relative bump size for scipy reference finite difference
    n_quad     : Gauss-Legendre nodes for the torch integration
    phi_max    : upper integration bound
    device     : torch device for CF work; defaults to CUDA when available
    dtype      : torch float dtype used for CF work
    batch_size : max states per autograd batch
    """

    def __init__(self, K: float, r: float, kappa: float, theta: float,
                 xi: float, rho: float, T: float, N: int,
                 cost: float = 1e-3, eps_rel: float = 1e-3,
                 n_quad: int = 256, phi_max: float = 200.0,
                 phi_min: float = 1e-8,
                 device: str | torch.device | None = None,
                 dtype: torch.dtype = torch.float64,
                 batch_size: int | None = 8192):
        self.K = K; self.r = r; self.kappa = kappa; self.theta = theta
        self.xi = xi; self.rho = rho; self.T = T; self.dt = T / N
        self.N = N; self.cost = cost; self.eps_rel = eps_rel
        self.n_quad = n_quad; self.phi_max = phi_max; self.phi_min = phi_min
        self.dtype = dtype
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size

        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        half_width = 0.5 * (phi_max - phi_min)
        midpoint = 0.5 * (phi_max + phi_min)
        self._nodes_np = midpoint + half_width * nodes
        self._weights_np = half_width * weights
        self._grid_cache: dict[tuple[str, int | None, torch.dtype],
                               tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid(self, device: torch.device, dtype: torch.dtype):
        key = (device.type, device.index, dtype)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        nodes = torch.tensor(self._nodes_np, device=device, dtype=dtype)
        weights = torch.tensor(self._weights_np, device=device, dtype=dtype)
        self._grid_cache[key] = (nodes, weights)
        return nodes, weights

    def _price_torch(self, S: torch.Tensor, v: torch.Tensor,
                     tau: float) -> torch.Tensor:
        if tau <= 0:
            return torch.clamp(S - self.K, min=0.0)

        nodes, weights = self._grid(S.device, S.dtype)
        phi = nodes.unsqueeze(0)
        w = weights.unsqueeze(0)
        S_col = S.unsqueeze(1)
        v_col = v.unsqueeze(1)

        cdtype = torch.complex128 if S.dtype == torch.float64 else torch.complex64
        i = torch.tensor(1j, dtype=cdtype, device=S.device)
        phi_c = phi.to(dtype=cdtype)
        log_k = torch.as_tensor(np.log(self.K), dtype=S.dtype,
                                device=S.device).to(dtype=cdtype)
        exp_term = torch.exp(-i * phi_c * log_k)

        cf_minus_i = _heston_cf_torch(
            torch.full_like(S_col, -1j, dtype=cdtype),
            S_col, v_col, self.r, self.kappa, self.theta,
            self.xi, self.rho, tau
        )
        cf_p1 = _heston_cf_torch(
            phi_c - i, S_col, v_col, self.r, self.kappa,
            self.theta, self.xi, self.rho, tau
        )
        cf_p2 = _heston_cf_torch(
            phi_c, S_col, v_col, self.r, self.kappa,
            self.theta, self.xi, self.rho, tau
        )

        p1_integrand = torch.real(exp_term * cf_p1 / (i * phi_c * cf_minus_i))
        p2_integrand = torch.real(exp_term * cf_p2 / (i * phi_c))
        P1 = 0.5 + (w * p1_integrand).sum(dim=1) / np.pi
        P2 = 0.5 + (w * p2_integrand).sum(dim=1) / np.pi
        discount = torch.exp(torch.as_tensor(-self.r * tau, dtype=S.dtype,
                                             device=S.device))
        return S * P1 - self.K * discount * P2

    def _delta_vega_batch(self, S: torch.Tensor, v: torch.Tensor,
                          tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        if tau <= 0:
            delta = (S >= self.K).to(dtype=S.dtype)
            vega = torch.zeros_like(S)
            return delta, vega

        with torch.enable_grad():
            S_leaf = S.detach().clone().requires_grad_(True)
            v_leaf = v.detach().clone().requires_grad_(True)
            price = self._price_torch(S_leaf, v_leaf, tau)
            delta, vega = torch.autograd.grad(price.sum(), (S_leaf, v_leaf))
        return delta.detach(), vega.detach()

    def _delta_vega_torch(self, S: torch.Tensor, v: torch.Tensor,
                          tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        S = S.detach().to(device=device, dtype=self.dtype)
        v = v.detach().to(device=device, dtype=self.dtype).clamp(min=1e-10)

        batch_size = self.batch_size or len(S)
        deltas, vegas = [], []
        for start in range(0, len(S), batch_size):
            stop = min(start + batch_size, len(S))
            delta_b, vega_b = self._delta_vega_batch(S[start:stop],
                                                     v[start:stop], tau)
            deltas.append(delta_b)
            vegas.append(vega_b)
        return torch.cat(deltas), torch.cat(vegas)

    def _delta(self, S_arr: np.ndarray, v_arr: np.ndarray,
               tau: float) -> np.ndarray:
        S = torch.as_tensor(S_arr)
        v = torch.as_tensor(v_arr)
        delta, _ = self._delta_vega_torch(S, v, tau)
        return delta.cpu().numpy()

    def _delta_vega(self, S_arr: np.ndarray, v_arr: np.ndarray,
                    tau: float) -> tuple[np.ndarray, np.ndarray]:
        S = torch.as_tensor(S_arr)
        v = torch.as_tensor(v_arr)
        delta, vega = self._delta_vega_torch(S, v, tau)
        return delta.cpu().numpy(), vega.cpu().numpy()

    def _delta_scipy(self, S_arr: np.ndarray, v_arr: np.ndarray,
                     tau: float) -> np.ndarray:
        """Previous scipy/finite-difference implementation, for validation."""
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

    def _hedge_from_greeks(self, S: torch.Tensor, delta: torch.Tensor,
                           vega: torch.Tensor) -> torch.Tensor:
        return delta

    # ── torch-compatible interface ────────────────────────

    def __call__(self, state_t: torch.Tensor, t: int) -> torch.Tensor:
        S   = state_t[:, 0]
        v   = state_t[:, 1]
        tau = self.T - t * self.dt
        delta, vega = self._delta_vega_torch(S, v, tau)
        hedge = self._hedge_from_greeks(
            S.to(device=self.device, dtype=self.dtype), delta, vega
        )
        return hedge.to(dtype=torch.float32, device=state_t.device)

    # ── standalone numpy backtest ─────────────────────────

    def backtest(self, paths_test: np.ndarray,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        paths_t = torch.as_tensor(paths_test, dtype=self.dtype,
                                  device=self.device)
        S_test = paths_t[:, :, 0]
        v_test = paths_t[:, :, 1]
        M, N1  = S_test.shape
        N      = N1 - 1

        cash       = torch.zeros(M, dtype=self.dtype, device=self.device)
        delta_prev = torch.zeros(M, dtype=self.dtype, device=self.device)
        total_fees = torch.zeros(M, dtype=self.dtype, device=self.device)

        steps = tqdm(range(N), desc="CF delta") if verbose else range(N)
        for t in steps:
            tau      = self.T - t * self.dt
            delta, vega = self._delta_vega_torch(S_test[:, t],
                                                 v_test[:, t], tau)
            delta_t = self._hedge_from_greeks(S_test[:, t], delta, vega)
            d_delta  = delta_t - delta_prev
            cash    -= d_delta * S_test[:, t]
            fees     = self.cost * torch.abs(d_delta) * S_test[:, t]
            cash    -= fees; total_fees += fees
            delta_prev = delta_t

        cash += delta_prev * S_test[:, -1]
        pnl   = cash - torch.clamp(S_test[:, -1] - self.K, min=0.0)
        return pnl.detach().cpu().numpy(), total_fees.detach().cpu().numpy()


class HestonMVDeltaHedger(HestonCFDeltaHedger):
    """Minimum-variance / Bartlett delta for Heston."""

    def _delta(self, S_arr: np.ndarray, v_arr: np.ndarray,
               tau: float) -> np.ndarray:
        S = torch.as_tensor(S_arr)
        v = torch.as_tensor(v_arr)
        delta, vega = self._delta_vega_torch(S, v, tau)
        S_t = S.to(device=self.device, dtype=self.dtype)
        hedge = self._hedge_from_greeks(S_t, delta, vega)
        return hedge.cpu().numpy()

    def _hedge_from_greeks(self, S: torch.Tensor, delta: torch.Tensor,
                           vega: torch.Tensor) -> torch.Tensor:
        return delta + (self.rho * self.xi / S) * vega


class HestonMVNoTradeBandHedger(HestonMVDeltaHedger):
    """
    No-trade-band strategy centered on the Heston MV/Bartlett delta.

    If the previous hedge is inside [center - width, center + width], the
    strategy does not trade. If it is outside, it trades only to the nearest
    band boundary. With band_width=0 this reduces to the MV delta.

    The torch __call__ returns the band center for compatibility with
    torch_backtest, but the actual band logic requires delta_prev and therefore
    lives in the standalone backtest method.
    """

    def __init__(self, *args, band_width: float = 0.05,
                 min_width: float = 0.0, max_width: float | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.band_width = band_width
        self.min_width = min_width
        self.max_width = max_width

    def _width(self, center: torch.Tensor) -> torch.Tensor:
        width = torch.full_like(center, self.band_width)
        if self.min_width:
            width = torch.clamp(width, min=self.min_width)
        if self.max_width is not None:
            width = torch.clamp(width, max=self.max_width)
        return width

    def backtest(self, paths_test: np.ndarray,
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        paths_t = torch.as_tensor(paths_test, dtype=self.dtype,
                                  device=self.device)
        S_test = paths_t[:, :, 0]
        v_test = paths_t[:, :, 1]
        M, N1 = S_test.shape
        N = N1 - 1

        cash = torch.zeros(M, dtype=self.dtype, device=self.device)
        delta_prev = torch.zeros(M, dtype=self.dtype, device=self.device)
        total_fees = torch.zeros(M, dtype=self.dtype, device=self.device)

        steps = tqdm(range(N), desc="MV no-trade") if verbose else range(N)
        for t in steps:
            tau = self.T - t * self.dt
            delta, vega = self._delta_vega_torch(S_test[:, t],
                                                 v_test[:, t], tau)
            center = self._hedge_from_greeks(S_test[:, t], delta, vega)
            width = self._width(center)
            lower = center - width
            upper = center + width
            delta_t = torch.minimum(torch.maximum(delta_prev, lower), upper)

            d_delta = delta_t - delta_prev
            cash -= d_delta * S_test[:, t]
            fees = self.cost * torch.abs(d_delta) * S_test[:, t]
            cash -= fees
            total_fees += fees
            delta_prev = delta_t

        cash += delta_prev * S_test[:, -1]
        pnl = cash - torch.clamp(S_test[:, -1] - self.K, min=0.0)
        return pnl.detach().cpu().numpy(), total_fees.detach().cpu().numpy()


class LelandHestonDeltaHedger:
    """
    Leland-style transaction-cost adjusted local-volatility delta.

    This is not a true Heston hedge. It is a cheap named baseline that takes
    sigma_t=sqrt(v_t) and applies the classic Leland volatility uplift before
    feeding it into the BSM delta formula.
    """

    def __init__(self, K: float, r: float, T: float, N: int,
                 cost: float = 1e-3, leland_scale: float = 1.0):
        self.K = K
        self.r = r
        self.T = T
        self.N = N
        self.dt = T / N
        self.cost = cost
        self.leland_scale = leland_scale

    def _sigma_eff(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
        uplift = (
            self.leland_scale
            * np.sqrt(2.0 / np.pi)
            * self.cost
            / (sigma * np.sqrt(self.dt))
        )
        return sigma * np.sqrt(1.0 + uplift)

    def _delta(self, S_t: np.ndarray, v_t: np.ndarray, t: int) -> np.ndarray:
        tau = self.T - t * self.dt
        sigma = self._sigma_eff(np.sqrt(np.maximum(v_t, 1e-12)))
        return _bsm_delta(S_t, self.K, self.r, sigma, tau)

    def __call__(self, state_t: torch.Tensor, t: int) -> torch.Tensor:
        S_t = state_t[:, 0].detach().cpu().numpy()
        if state_t.shape[1] > 1:
            v_t = state_t[:, 1].detach().cpu().numpy()
        else:
            v_t = np.full_like(S_t, 0.04)
        delta = self._delta(S_t, v_t, t)
        return torch.tensor(delta, dtype=torch.float32, device=state_t.device)

    def backtest(self, paths_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S_test = paths_test[:, :, 0]
        v_test = paths_test[:, :, 1] if paths_test.shape[2] > 1 \
                 else np.full_like(S_test, 0.04)
        M, N1 = S_test.shape
        N = N1 - 1

        cash = np.zeros(M)
        delta_prev = np.zeros(M)
        total_fees = np.zeros(M)
        for t in range(N):
            delta_t = self._delta(S_test[:, t], v_test[:, t], t)
            d_delta = delta_t - delta_prev
            cash -= d_delta * S_test[:, t]
            fees = self.cost * np.abs(d_delta) * S_test[:, t]
            cash -= fees
            total_fees += fees
            delta_prev = delta_t

        cash += delta_prev * S_test[:, -1]
        pnl = cash - np.clip(S_test[:, -1] - self.K, 0, None)
        return pnl, total_fees


class BackwardQuadraticHedger:
    """
    Regression approximation to the discrete-time variance-optimal hedge.

    The fitted hedge is a backward local-risk-minimisation recursion. At each
    time step it estimates conditional moments of the next-step continuation
    value and stock increment, then uses

        hedge = Cov(V_{t+1}, dS | X_t) / Var(dS | X_t).

    This is intentionally model-agnostic and cheap. It is a stronger LSMC-style
    baseline than the older one-shot LSMCDeltaHedger, but it should still be
    validated against more specialised closed-form discrete Heston formulas
    before calling it "the" SOTA.
    """

    def __init__(self, K: float, cost: float = 0.0, T: float = 1.0,
                 ridge: float = 1e-8,
                 clip: tuple[float, float] | None = (-1.0, 2.0)):
        self.K = K
        self.cost = cost
        self.T = T
        self.ridge = ridge
        self.clip = clip
        self.coefs: list[dict[str, np.ndarray]] | None = None

    @staticmethod
    def _features(S: np.ndarray, v: np.ndarray, tau: float,
                  S_ref: float, v_ref: float) -> np.ndarray:
        s = S / S_ref
        vv = v / max(v_ref, 1e-12)
        tau_arr = np.full_like(s, tau, dtype=float)
        return np.column_stack([
            np.ones_like(s),
            s,
            vv,
            tau_arr,
            s * s,
            vv * vv,
            s * vv,
        ])

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        xtx = X.T @ X
        if self.ridge > 0:
            xtx = xtx + self.ridge * np.eye(xtx.shape[0])
        return np.linalg.solve(xtx, X.T @ y)

    @staticmethod
    def _predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return X @ beta

    def fit(self, paths_train: np.ndarray) -> "BackwardQuadraticHedger":
        S = paths_train[:, :, 0]
        v = paths_train[:, :, 1] if paths_train.shape[2] > 1 \
            else np.zeros_like(S)
        M, N1 = S.shape
        N = N1 - 1
        self.N = N
        self.dt = self.T / N
        self.S_ref = float(np.median(S[:, 0]))
        self.v_ref = float(np.median(v[:, 0])) if paths_train.shape[2] > 1 else 1.0

        value_next = np.clip(S[:, -1] - self.K, 0, None)
        coefs_rev: list[dict[str, np.ndarray]] = []

        for t in range(N - 1, -1, -1):
            tau = self.T - t * self.dt
            X = self._features(S[:, t], v[:, t], tau,
                               self.S_ref, self.v_ref)
            dS = S[:, t + 1] - S[:, t]

            beta_value = self._fit_linear(X, value_next)
            beta_dS = self._fit_linear(X, dS)
            beta_cross = self._fit_linear(X, value_next * dS)
            beta_second = self._fit_linear(X, dS * dS)

            value_hat = self._predict(X, beta_value)
            dS_hat = self._predict(X, beta_dS)
            cross_hat = self._predict(X, beta_cross)
            second_hat = self._predict(X, beta_second)

            var_hat = np.maximum(second_hat - dS_hat * dS_hat, 1e-10)
            cov_hat = cross_hat - value_hat * dS_hat
            hedge = cov_hat / var_hat
            if self.clip is not None:
                hedge = np.clip(hedge, self.clip[0], self.clip[1])

            value_next = value_hat - hedge * dS_hat
            coefs_rev.append({
                "value": beta_value,
                "dS": beta_dS,
                "cross": beta_cross,
                "second": beta_second,
            })

        self.coefs = list(reversed(coefs_rev))
        return self

    def _delta(self, S_t: np.ndarray, v_t: np.ndarray, t: int) -> np.ndarray:
        assert self.coefs is not None, "Call .fit() before using the hedger."
        tau = self.T - t * self.dt
        X = self._features(S_t, v_t, tau, self.S_ref, self.v_ref)
        c = self.coefs[t]
        value_hat = self._predict(X, c["value"])
        dS_hat = self._predict(X, c["dS"])
        cross_hat = self._predict(X, c["cross"])
        second_hat = self._predict(X, c["second"])
        var_hat = np.maximum(second_hat - dS_hat * dS_hat, 1e-10)
        hedge = (cross_hat - value_hat * dS_hat) / var_hat
        if self.clip is not None:
            hedge = np.clip(hedge, self.clip[0], self.clip[1])
        return hedge

    def __call__(self, state_t: torch.Tensor, t: int) -> torch.Tensor:
        S_t = state_t[:, 0].detach().cpu().numpy()
        if state_t.shape[1] > 1:
            v_t = state_t[:, 1].detach().cpu().numpy()
        else:
            v_t = np.zeros_like(S_t)
        delta = self._delta(S_t, v_t, t)
        return torch.tensor(delta, dtype=torch.float32, device=state_t.device)

    def backtest(self, paths_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.coefs is not None, "Call .fit() before backtesting."
        S = paths_test[:, :, 0]
        v = paths_test[:, :, 1] if paths_test.shape[2] > 1 \
            else np.zeros_like(S)
        M, N1 = S.shape
        N = N1 - 1
        if N != self.N:
            raise ValueError(f"Expected N={self.N}, got N={N}.")

        cash = np.zeros(M)
        delta_prev = np.zeros(M)
        total_fees = np.zeros(M)
        for t in range(N):
            delta_t = self._delta(S[:, t], v[:, t], t)
            d_delta = delta_t - delta_prev
            cash -= d_delta * S[:, t]
            fees = self.cost * np.abs(d_delta) * S[:, t]
            cash -= fees
            total_fees += fees
            delta_prev = delta_t

        cash += delta_prev * S[:, -1]
        pnl = cash - np.clip(S[:, -1] - self.K, 0, None)
        return pnl, total_fees
