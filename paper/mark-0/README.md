# ICAIF 2026 paper, mark 0

This folder is a first draft of the paper tentatively
titled **“DeepHedger-Default: Guided Recipe Search for Deep Hedging under Market
Frictions.”** It is a paper frame, not a record of completed confirmatory
experiments.

## Format status

The [official ICAIF 2026 call](https://icaif2026.org/call-for-papers.html), as
checked on 2026-07-24, fixes the following submission contract:

- deadline: August 2, 2026, Anywhere on Earth (UTC-12);
- conference: November 14–17, 2026, Milan, Italy;
- no more than eight pages total in two-column ACM `sigconf`, including all
  figures and references;
- a self-contained PDF with no supplementary materials or appendices;
- double-blind review using the `anonymous` class parameter;
- a final author list at initial submission; and
- no rebuttal period.

`main.tex` is anonymous by default while retaining Artem Chistyakov’s full
HSE University/Yandex Research author metadata in source. Change
`\def\attributedversion{0}` to `1` for the attributed internal build. The
mark-0 PDF is intentionally shorter than the eight-page maximum so new results
can replace placeholders without forcing premature compression.

`main.tex` uses an unnumbered **Generative AI Use Disclosure** immediately
before the references. It is intentionally not an `acks` environment, because
some anonymous ACM modes suppress acknowledgments. The disclosure is
non-identifying, keeps the human author fully responsible, and must remain
visible in the review PDF unless the ICAIF 2026 call prescribes another
location or wording.

Authoritative links:

- ACM proceedings template: <https://www.acm.org/publications/proceedings-template>
- ICAIF 2026 call: <https://icaif2026.org/call-for-papers.html>
- Production `acmart` package: <https://ctan.org/pkg/acmart>

The ACM production `acmart` v2.19 class, bibliography style, and required LPPL
source are vendored under `template/acmart/`. The official direct ZIP returned
HTTP 403 to this build environment, so the matching production package was
downloaded from CTAN. Exact source URLs, hashes, license notes, and the fallback
are recorded in [`template/PROVENANCE.md`](template/PROVENANCE.md). The local
`.latexmkrc` forces this version instead of the host TeX Live v1.81 class.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean auxiliary files while retaining the PDF:

```bash
latexmk -c
```

## Evidence used in the draft

The preliminary table is deliberately labeled non-confirmatory. Its exact
sources are:

- market/training configuration:
  `../../heston_comparison copy 3.ipynb` and
  `../../heston_comparison copy 3 cost0.ipynb`;
- 10 bp metrics: `../../results_heston_cf_mv_copy3/*.json`;
- zero-cost metrics: `../../results_heston_cf_mv_copy3_cost0/*.json`; and
- MV quadrature diagnostic:
  `../../results_mv_budget/mv_budget_sweep.json`.

The Heston cell is `S0=K=100`, `v0=theta=0.04`, `r=0.01`, `kappa=2`,
`xi=0.1`, `rho=-0.7`, `T=1`, `N=30`, `a=1`, with 5,000 validation and
5,000 test paths. The learned methods used 10,000 epochs and 3,000 newly
sampled training paths per epoch. NumPy evaluation-path seeding was present,
but complete Torch/CUDA/policy seeding was not. The notebooks also simulate
`r=0.01` without passing cash-account accrual to the backtest or first
discounting paths and payoff. These rows inherit an accounting inconsistency
and must not be promoted to abstract or conclusion claims.

All bibliography entries in `references.bib` were checked against publisher,
proceedings, DOI, or arXiv records. Citation metadata should be revalidated
once more during camera-ready preparation.

## Result replacement map

The manuscript is organized so new experiments replace placeholders rather
than force a new narrative:

- Figure 1 is the guided-search ladder and selection contract.
- Table 1 fixes the locked benchmark factors.
- Table 2 is the stagewise ablation and decision log.
- Table 3 is the provenance-labeled preliminary diagnostic.
- Table 4 is the final cross-market headline result.
- The reproducibility section defines the package/config/aggregation contract.
- The limitations section already separates per-contract simulated policies
  from future amortized and real-data work.

Before the paper makes its headline claim, the required evidence is:

- five fully seeded runs for every guided-search candidate;
- ten fresh seeds for the frozen recipe on the locked benchmark;
- paired uncertainty, CVaR, fees, turnover, and wall-clock reporting;
- strong mathematical baselines using exactly the information seen by the
  network;
- a numerical convergence study for every non-closed-form baseline;
- agreement between discounted-PnL and cash-account implementations;
- a fixed-market-state action-sensitivity diagnostic for running PnL, required
  because CARA utility does not itself create wealth-dependent actions;
- a GBM frictionless sanity check; and
- a one-command regeneration path from configs to all paper tables.

## Important terminology

The current `SACTrainer` in the repository is an entropy-regularized stochastic
pathwise policy. It is not full Soft Actor-Critic because it has no critic,
target networks, or replay buffer. The draft calls it **MaxEnt pathwise** and
reserves **SAC** for a complete, independently tested implementation.

The Heston correction
`dC/dS + (rho * xi / S) * dC/dv` is called the local minimum-variance delta.
“Bartlett-style” may be useful informal terminology, but “minimum-variance
delta” is the precise label used in the draft.
