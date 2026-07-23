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

`architecture` is a hyperparameter of the same class: `mlp`, `paf_shared`,
or `paf_featurewise`. Stage 1 intentionally excludes running P&L, deviation
targets, alternative optimizers, and RL algorithms so that each later
improvement can be added and measured in the guided-search order.

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

The coursework-era notebooks and results were moved without deletion to
[`archive/legacy-2026-07-24`](archive/legacy-2026-07-24). The first paper frame
is under [`paper/mark-0`](paper/mark-0).
