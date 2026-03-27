import numpy as np
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, T, N, random_seed=None):
        self.T = T
        self.N = N
        self.dt = T / N
        self.random_seed = random_seed

    def _set_seed(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @property
    @abstractmethod
    def state_dim(self):
        pass

    @abstractmethod
    def sample(self, M):
        """
        Returns:
            paths: np.ndarray of shape (M, N+1, state_dim)
        """
        pass


class GBMSampler(BaseSampler):
    def __init__(self, S0, r, sigma, T, N, random_seed=None):
        super().__init__(T, N, random_seed)
        self.S0 = S0
        self.r = r
        self.sigma = sigma

    @property
    def state_dim(self):
        return 1

    def sample(self, M):
        self._set_seed()
        S = np.zeros((M, self.N + 1))
        S[:, 0] = self.S0

        for t in range(self.N):
            Z = np.random.randn(M)
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * self.dt
                + self.sigma * np.sqrt(self.dt) * Z
            )

        return S[..., None]  # (M, N+1, 1)


class HestonSampler(BaseSampler):
    def __init__(
        self, S0, v0, r, kappa, theta, xi, rho,
        T, N, random_seed=None
    ):
        super().__init__(T, N, random_seed)
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    @property
    def state_dim(self):
        return 2

    def sample(self, M):
        self._set_seed()

        S = np.zeros((M, self.N + 1))
        v = np.zeros((M, self.N + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for t in range(self.N):
            Z1 = np.random.randn(M)
            Z2 = np.random.randn(M)
            dW1 = np.sqrt(self.dt) * Z1
            dW2 = np.sqrt(self.dt) * (
                self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2
            )

            v[:, t + 1] = np.clip(
                v[:, t]
                + self.kappa * (self.theta - v[:, t]) * self.dt
                + self.xi * np.sqrt(np.maximum(v[:, t], 0)) * dW2,
                1e-8, None,
            )

            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * v[:, t]) * self.dt
                + np.sqrt(np.maximum(v[:, t], 0)) * dW1
            )

        return np.stack([S, v], axis=2)
