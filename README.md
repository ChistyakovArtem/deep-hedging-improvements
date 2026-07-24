# Deep Hedging Improvements

Reproducible experiment code for the ICAIF 2026 strong-recipe study.

The public package surface is deliberately small:

```python
from deep_hedger import DeepHedger, DeepHedgerConfig

hedger = DeepHedger(
    market=market,
    objective={"risk_aversion": 1.0},
    config=DeepHedgerConfig(
        architecture="paf_featurewise",
        paf_sigma=1.0,
        seed=0,
    ),
)
hedger.fit(public_paths)
actions = hedger.predict_actions(test_paths)
```

`architecture` is a hyperparameter of the same class: `mlp`, `ntbn`,
`paf_shared`, or `paf_featurewise`. `ntbn` implements the No-Transaction Band
Network of Imaki et al.: a shared four-layer MLP predicts asymmetric band widths
around a local Black--Scholes delta and the previous hedge is clamped into that
band. Its Heston adaptation uses current instantaneous volatility and the
configured interest rate. Stage 1 intentionally excludes running P&L,
deviation targets, alternative optimizers, and RL algorithms so that each
later improvement can be added and measured in the guided-search order.

## Frozen stage-1 protocol

The authoritative market, objective, features, budgets, fixed public/private
seeds, checksums, and search axes live in
[`config/project.toml`](config/project.toml). Candidate models may use only the
public board. The private board is reserved for the vanilla DH, the
public-selected best DH, and mathematical competitors.

Generate the fixed paths and experiment configs:

```bash
python -m bin.materialize_leaderboards
python -m dev.generate_stage1_configs
```

Run one learned config, one complete seed queue, or the mathematical baselines:

```bash
python -m bin.run_deep_hedger exp/DeepHedger-Stage1-v1/heston-xi03/split-public/seed-000/cfg-0000
python -m bin.run_pending_seed exp/DeepHedger-Stage1-v1/heston-xi03/split-public/seed-000 --keep-going
python -m bin.run_math_deltas --device cuda
```

Only after all 88 public learned-model results exist:

```bash
python -m bin.evaluate_final_private --device cuda
```

## Completed stage-1 result

All 88 learned configurations completed on eight A100s. Lower entropic risk is
better; learned-model uncertainty below is the sample standard deviation over
the eight paired training seeds.

| Method | Public entropic risk | Private entropic risk |
|---|---:|---:|
| Vanilla DH | 10.9528 ± 0.0191 | **11.1194 ± 0.0577** |
| Public-selected featurewise PAF, sigma 1.0 | **10.8997 ± 0.0247** | 11.3174 ± 0.1989 |
| Leland local-vol delta, scale 18 | 11.6525 | 11.6092 |
| Heston MV no-trade delta, width 0.04 | 11.6295 | 11.9330 |
| Heston MV delta | 12.1878 | 12.4740 |
| Heston CF delta | 12.8100 | 12.9868 |

The PAF choice was fixed from the public aggregate before the private board was
opened. It improved the public score but lost to vanilla DH on private in seven
of eight paired seeds, so stage 1 does **not** support presenting PAF as an
improvement over vanilla DH. Both learned methods remained ahead of every
mathematical competitor tested.

The private exponential-utility estimate is also extremely tail dominated:
the mean effective sample size of the 100,000 exponential weights is 30.2 for
vanilla DH and 14.5 for PAF, while the largest path contributes 13.8% and 26.5%
of total weight on average. This is an experiment-design warning for the paper:
close method rankings at risk aversion 1 require a more statistically stable
tail-evaluation protocol. The frozen private result must not be used for
retuning this stage.

The machine-readable selection, private aggregates, paired differences, and
tail diagnostics are in
[`exp/DeepHedger-Stage1-v1/heston-xi03/split-public/final-summary.json`](exp/DeepHedger-Stage1-v1/heston-xi03/split-public/final-summary.json).

## NTBN literature baseline

The Imaki et al. No-Transaction Band Network baseline completed for eight
training seeds on both public Heston boards. Lower entropic risk is better.

| Market | NTBN public entropic risk | Same-board references |
|---|---:|---|
| Heston, xi=0.1 | 9.6577 ± 0.0304 | eight-seed exact-board baselines pending |
| Heston, xi=0.3 | **10.8837 ± 0.0289** | Vanilla DH: 10.9528 ± 0.0191; selected PAF: 10.8997 ± 0.0247; MV no-trade: 11.6295 |

At xi=0.3, NTBN beat vanilla DH in all eight paired training seeds
(mean paired difference -0.0691) and PAF in five of eight
(mean difference -0.0159). These are development-board results, not a final
out-of-sample claim; NTBN has not been evaluated on the private board. The
archived xi=0.1 Leland value 9.3012 is not placed in the table because it used
a different 5,000-path board and undiscounted accounting. Full seed-level
results and comparison guardrails are in
[`exp/DeepHedger-NTBN-v1/run-summary.json`](exp/DeepHedger-NTBN-v1/run-summary.json).

The coursework-era notebooks and results were moved without deletion to
[`archive/legacy-2026-07-24`](archive/legacy-2026-07-24). The first paper frame
is under [`paper/mark-0`](paper/mark-0).
