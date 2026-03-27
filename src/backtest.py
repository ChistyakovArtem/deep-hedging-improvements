"""
backtest.py — self-financing portfolio simulation.

Two feature modes (controlled by `use_pnl`):

    use_pnl=False  (default, original paper):
        features = [state | tau | delta_prev]
        input_dim = state_dim + 2

    use_pnl=True:
        features = [state | tau | delta_prev | current_pnl]
        input_dim = state_dim + 3
"""

import torch


def torch_backtest(
    paths_t: torch.Tensor,
    K: float,
    cost: float = 0.0,
    rate_dt: float = 0.0,
    hedge_policy=None,              # nn.Module  — trained policy
    delta_exogenous=None,           # callable(state_t, t) -> Tensor(M,)
    use_pnl: bool = False,          # include running PnL as feature
    pnl_start_epoch: int = 0,       # before it pnl = 0 to allow pretrain
    detach: bool = False,
    epoch = None
):
    """
    Parameters
    ----------
    paths_t   : (M, N+1, state_dim)  float32 tensor on device
    K         : option strike
    cost      : proportional transaction cost on |Δδ| · S
    rate_dt   : risk-free rate × dt (applied to cash each step)
    use_pnl   : add running P&L as extra input feature

    Returns
    -------
    pnl        : (M,)  terminal P&L  =  cash_T - max(S_T - K, 0)
    total_fees : (M,)  cumulative transaction costs
    """
    M, N_plus1, state_dim = paths_t.shape
    N      = N_plus1 - 1
    device = paths_t.device
    S      = paths_t[:, :, 0]

    cash       = torch.zeros(M, device=device)
    delta_prev = torch.zeros(M, device=device)
    total_fees = torch.zeros(M, device=device)
    tau        = torch.linspace(1.0, 0.0, N_plus1, device=device)

    for t in range(N):
        state_t = paths_t[:, t, :]
 
        if hedge_policy is not None:
            parts = [state_t, tau[t].expand(M, 1), delta_prev.unsqueeze(1)]
            if use_pnl:
                if epoch is None or epoch >= pnl_start_epoch:
                    parts.append((cash + delta_prev * S[:, t]).unsqueeze(1))
                else:
                    parts.append(torch.zeros(S.shape[0], 1, device=S.device))

            delta = hedge_policy(torch.cat(parts, dim=1)).squeeze(1)
        else:
            delta = delta_exogenous(state_t, t)

        cash      += cash * rate_dt
        d_delta    = delta - delta_prev
        cash      -= d_delta * S[:, t]
        fees       = cost * torch.abs(d_delta) * S[:, t]
        cash      -= fees
        total_fees += fees
        delta_prev = delta

    cash += delta_prev * S[:, -1]
    pnl   = cash - torch.clamp(S[:, -1] - K, min=0.0)

    if detach:
        return pnl.detach().cpu().numpy(), total_fees.detach().cpu().numpy()
    return pnl, total_fees
