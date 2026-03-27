import torch


def SoftMin(pnl: torch.Tensor, a: float) -> torch.Tensor:
    """
    Stable entropic risk measure (SoftMin / CVaR-like).
    Differentiable.  Lower is better (more negative PnL is penalised).

        SoftMin(X; a) = (1/a) * log E[exp(-a * X)]
    """
    X = -a * pnl
    X_max = X.max()
    return (torch.log(torch.exp(X - X_max).mean()) + X_max) / a
