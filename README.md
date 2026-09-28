# Deep Hedging: Pricing Derivatives with Transaction Costs

Research code for my **bachelor's thesis at HSE University, Faculty of Computer Science (2026)**, developed as a Sberbank project. The thesis studies how neural hedging policies can improve the risk of hedging a European call when volatility is stochastic, rebalancing is discrete, and trading incurs proportional costs.

**Author:** [Artem Chistyakov](https://github.com/ChistyakovArtem)

**Thesis:** *Pricing Derivatives with Transaction Costs*

**Supervisor:** Gennady Piftankin

The central question is which policy representations and optimization methods make deep hedging competitive with an analytical Heston-delta baseline. The repository contains the simulator, classical baselines, neural policies, training code, experiment notebooks and saved learning curves.

## Research overview

- **Periodic feature embeddings:** compare a plain MLP with learned sinusoidal features, including global and feature-wise embeddings.
- **Path-dependent information:** add running portfolio P&L to the policy state and examine its effect on optimization stability.
- **Residual hedging:** learn a correction to Black–Scholes delta, using instantaneous Heston variance as the volatility input.
- **Initialization and optimization:** provide zero-output initialization and compare Adam, K-FAC and Muon.
- **Ablations:** compare individual components against regression-based LSMC delta and analytical Heston delta; also include GBM and zero-transaction-cost experiments.

The main entry point is [heston_comparison.ipynb](heston_comparison.ipynb). Start with its saved outputs to inspect the experiments without training models.

## Hedging objective

The agent chooses a position in the underlying at each rebalancing time. Its state contains the asset price, instantaneous variance, time to maturity and previous position; running P&L is optional. Trading costs are proportional to turnover:

$$
C_t = c S_t |\delta_t-\delta_{t-1}|.
$$

The training objective is the entropic risk of terminal hedging P&L:

$$
\rho_a(\mathrm{PnL}) = \frac{1}{a}\log\mathbb{E}[\exp(-a\,\mathrm{PnL})].
$$

This metric is called `SoftMin` in the code. **Lower is better.** The simulator starts with zero cash, trades the underlying and subtracts the call payoff at maturity; an option premium is not included in the reported P&L.

The residual policy takes the form

$$
\delta_t = \delta_{\mathrm{BS}}(S_t,\sqrt{v_t},T-t) + f_\theta(x_t).
$$

Here the Black–Scholes term supplies a reference hedge; the network learns the adjustment for the simulated setting and transaction costs.

## Recorded results

The following values are the **saved per-method test outputs** in [heston_comparison.ipynb](heston_comparison.ipynb), using 5,000 simulated test paths and transaction-cost coefficient `c = 0.001` (10 basis points of traded notional).

| Method | Test SoftMin ↓ |
| --- | ---: |
| LSMC delta | 9.9928 |
| Analytical Heston delta | 9.5526 |
| Plain deep hedging, MLP | 10.1582 |
| Deep hedging + periodic embeddings | 9.1918 |
| Deep hedging + running P&L | 9.6799 |
| Periodic embeddings + running P&L | 9.4158 |
| Entropy-regularized policy (`DH-SAC`) | 9.2093 |
| Residual policy, Adam | **9.1752** |
| Residual policy, K-FAC | 9.2061 |
| Residual policy, Muon | 9.3180 |

In this recorded run, periodic embeddings and residual policies improve on the analytical-delta baseline. Adam gives the lowest test score among the listed variants; K-FAC does not improve on it in this run. The P&L pretraining variant is unstable in the saved output, with test SoftMin 27.4849.

These are simulation results from one recorded experiment, rather than live trading performance or a multi-seed significance study. The thesis defense reports a best K-FAC value of **9.0897**; that value differs from the **9.2061** saved test output above. The table deliberately uses the repository's inspectable notebook outputs.

**Reading the logs:** the JSON/CSV files under `results/` contain validation-loss histories for trained models. The notebook's log-loaded summary uses the last logged validation value, while baseline records contain final evaluation values. Use the individual `record(...)` outputs following `eval_on_test()` for the test comparison above; the log-loaded summary is not a uniform test-results table.

## Experiment setup

| Parameter | Main Heston notebook |
| --- | --- |
| Initial asset price and call strike | `S0 = K = 100` |
| Initial and long-run variance | `v0 = theta = 0.04` |
| Mean reversion / vol-of-vol / correlation | `kappa = 2`, `xi = 0.1`, `rho = -0.7` |
| Drift parameter | `r = 0.01` |
| Maturity / rebalancing steps | `T = 1`, `N = 30` |
| Transaction cost / risk aversion | `cost = 0.001`, `a = 1` |
| Simulated training paths per epoch | 3,000 |
| Validation / test paths | 5,000 / 5,000 |
| Maximum epochs / validation interval | 10,000 / 200 |

Training uses fresh simulated paths and restores the checkpoint with the lowest validation loss. The configured patience is 100 validation checks, longer than the main notebook's 50-check training budget, so it does not shorten those runs. The notebook seeds NumPy for path generation but does not fully seed PyTorch initialization; exact numerical reproduction is therefore not guaranteed.

The model builder supports `ModelConfig(zero_last_layer=True)`. When using a trainer, set **`TrainerConfig(init_mode="zero_last")`**: the base trainer derives the model's initialization flag from `init_mode` and overrides the nested model setting.

## Getting started

Use Python 3.10 or newer. From a terminal:

```bash
git clone https://github.com/ChistyakovArtem/deep-hedging-improvements.git
cd deep-hedging-improvements
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy scipy matplotlib tqdm jupyterlab
jupyter lab heston_comparison.ipynb
```

Run notebooks from the repository root so that the `src` imports resolve. They select CUDA when available and otherwise use the CPU. The full comparison trains several models and evaluates an expensive analytical-delta baseline; start by reading the saved outputs. Before a new run, set `RESULTS_DIR` to a new directory such as `results_local` to preserve the checked-in logs.

This small example checks the simulator, policy and differentiable objective without running the benchmark:

```python
import torch

from src.backtest import torch_backtest
from src.metrics import SoftMin
from src.models import ModelConfig, build_policy
from src.samplers import HestonSampler

torch.manual_seed(0)
sampler = HestonSampler(
    S0=100, v0=0.04, r=0.01, kappa=2, theta=0.04,
    xi=0.1, rho=-0.7, T=1, N=8, random_seed=0,
)
paths = torch.tensor(sampler.sample(32), dtype=torch.float32)
policy = build_policy(
    ModelConfig(arch="paf", hidden_dims=(16, 16), n_frequencies=4),
    input_dim=sampler.state_dim + 2,
)
pnl, fees = torch_backtest(paths, K=100, cost=0.001, hedge_policy=policy)
loss = SoftMin(pnl, a=1.0)
loss.backward()
print("Paths:", tuple(paths.shape), "Finite loss:", bool(torch.isfinite(loss)))
```

## Repository guide

| File or directory | Purpose |
| --- | --- |
| [heston_comparison.ipynb](heston_comparison.ipynb) | Main Heston experiment and saved outputs |
| [gbm_comparison.ipynb](gbm_comparison.ipynb) | Geometric Brownian motion comparison |
| [src/samplers.py](src/samplers.py) | Heston and GBM path generators |
| [src/backtest.py](src/backtest.py) | Differentiable portfolio simulation and trading costs |
| [src/models.py](src/models.py) | MLP, global PAF and feature-wise PAF policies |
| [src/deltas.py](src/deltas.py) | Analytical and regression-based delta baselines |
| [src/trainers.py](src/trainers.py) | Training loops, residual policies, K-FAC and Muon |
| [src/metrics.py](src/metrics.py) | Numerically stable entropic risk |
| [src/logging_utils.py](src/logging_utils.py) | JSON/CSV experiment logging |
| [results/](results/) | Saved Heston experiments with transaction costs |
| [results_heston_no_fees/](results_heston_no_fees/) | Saved Heston experiments without transaction costs |
| [results_gbm/](results_gbm/) | Saved GBM experiments |

Historical notebook copies and thesis-report materials are also included. The main notebook above is the recommended starting point.

## License

See [LICENSE](LICENSE).
