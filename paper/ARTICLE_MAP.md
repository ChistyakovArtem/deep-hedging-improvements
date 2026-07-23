# ICAIF 2026 Article Map

## Working position

The paper should be presented as a **strong, reproducible default recipe for
pathwise deep hedging**, selected by a stagewise guided search and then frozen
before evaluation on held-out market and contract settings.

The central empirical question is not assumed in advance:

> Can a carefully selected deep-hedging recipe reduce entropic hedging risk
> relative to vanilla deep hedging and match or improve information-matched
> analytical hedges across complete and incomplete markets?

The safe contribution exists even if the final answer is “match” rather than
“beat everywhere”: the recipe, the controlled selection protocol, the strong
baselines, and the characterization of when each component transfers.

Suggested working title:

> **Better by Default: A Strong Recipe for Deep Hedging**

Alternative, if the analytical anchor is essential to the final model:

> **Guided Deep Hedging: Strong Defaults from Analytical Priors and Modern
> Neural Training**

## Venue constraints that determine the paper

- ICAIF 2026 deadline: **August 2, 2026 (Anywhere on Earth)**.
- Format: latest ACM `acmart`, `sigconf`, anonymous submission.
- Limit: **eight pages total**, including figures and references.
- No supplementary material or appendix.
- There is no rebuttal period and no opportunity to add scientific evidence
  after submission. All experiments supporting an ICAIF claim must be complete
  before the evidence freeze. An unfinished optional axis is removed from the
  submission claim; it is not treated as something the ICAIF paper can repair
  later.
- Double blind: the submitted PDF must omit Artem Chistyakov, HSE University,
  Yandex Research, the correspondence email, identifying acknowledgments, and
  identifying repository links. Keep this metadata behind a camera-ready
  switch.
- The author list submitted to CMT is final.
- ACM permits generative-AI assistance but requires the human authors to remain
  responsible for the work and to disclose content-generation use. Include a
  concise, non-identifying disclosure in the anonymous submission; verify every
  AI-suggested statement and citation manually.

Official call:
<https://icaif2026.org/call-for-papers.html>

## Claim ladder

Claims should be activated only when their evidence gate passes.

1. **Protocol claim (safe):** we introduce a stagewise, seed-aware protocol for
   selecting a reusable deep-hedging recipe against information-matched
   analytical baselines.
2. **Optimization claim (expected):** the selected recipe improves vanilla
   pathwise deep hedging in aggregate over the frozen evaluation suite.
3. **Strong-baseline claim (pending):** the recipe matches or improves the
   strongest applicable analytical hedge on entropic risk in a stated fraction
   of settings.
4. **Robustness claim (pending):** the result persists across stochasticity,
   moneyness, transaction costs, rebalancing horizons, and payoff types.
5. **Software claim (pending):** the frozen recipe is available through one
   public `DeepHedger` interface and can reproduce the paper configuration.

Do not claim:

- that beating an ordinary Black--Scholes or plain Heston delta is novel;
- “first to beat” any hedge without a complete contemporary literature audit;
- that Deep Hedging is the dominant industry-deployed method without strong
  public evidence;
- that the current entropy-regularized stochastic trainer is Soft
  Actor-Critic;
- that the Heston minimum-variance hedge is universally “Bartlett’s delta.”
  Use **Heston minimum-variance delta**; use “Bartlett-type adjustment” only
  where the cited model/result supports the name.

## Research questions

**RQ1 — Recipe selection.** Which representation, analytical prior, state
features, initialization, and optimizer reliably improve vanilla pathwise deep
hedging?

**RQ2 — Strong baselines.** After the recipe is frozen, how much of the gap to
the information-matched analytical hedge does it close, and where does it
cross that hedge under the trained entropic objective?

**RQ3 — Trade-offs and failure modes.** Are improvements accompanied by higher
turnover, fees, tail loss, runtime, or seed sensitivity? Which components fail
to transfer across market regimes?

## Eight-page section map

| Section | Target space | What it must contain | Evidence/asset |
|---|---:|---|---|
| Abstract | 0.25 page | Problem, stagewise recipe, benchmark breadth, one main numerical result, artifact | Fill numbers only after aggregation |
| 1. Introduction and related work | 1.10 pages | Why “default DH” is fragile; why ordinary delta is weak; recipe-vs-new-algorithm framing; three contributions | Deep Hedging, strong analytical hedges, recent optimization/recipe work |
| 2. Problem setting | 0.80 page | Self-financing discrete hedge, costs, entropic risk, observation set, payoff, analytical baselines | One compact equation block |
| 3. Guided recipe construction | 1.20 pages | Public policy form, normalized state, periodic variants, residual anchor, optimizer choices, stagewise selection and freeze rule | Figure 1 plus compact recipe box |
| 4. Experimental protocol | 0.85 page | Development/holdout split, markets, axis-isolated settings, seeds, shared test paths, metrics, compute | Compact setting table |
| 5. Results | 2.25 pages | Cumulative gains, leave-one-out ablation, main benchmark, win/tie/loss, risk/turnover/runtime, one non-obvious finding | Figures 1–2 and Tables 1–2 |
| 6. Limitations and conclusion | 0.40 page | Simulation/model-information limitations, per-instrument policies, no real data/amortization, result boundary | Short and explicit |
| References | 1.15 pages | Only citations used in the argument | Approximately 20–28 references |

If the paper exceeds eight pages, merge related work into the introduction and
reduce baseline exposition before shrinking result evidence.

## Method definition

The learned hedge should be written once in a form that covers all accepted
components:

\[
  \delta_t =
  \delta_{\mathrm{anchor}}(I_t)
  + \lambda\,f_\theta(z(I_t,\delta_{t-1},\tau_t,e_t)),
\]

where:

- \(I_t\) is the information set available to both the network and the
  analytical baseline;
- \(\delta_{\mathrm{anchor}}\) is zero, a local Black--Scholes proxy, or the
  applicable information-matched minimum-variance hedge;
- \(z\) is the selected normalized representation;
- \(e_t\) is optional running hedging error/P&L;
- \(\lambda=1\) for the ordinary residual policy, or follows a declared warm-up
  schedule if residual-strength continuation is selected.

This makes the analytical prior explicit. If the winning recipe requires the
minimum-variance anchor, describe the method as a hybrid/model-assisted deep
hedge, not as model-free deep hedging.

### State representation

Use dimensionless features as the correctness baseline, not as an optional
performance trick:

- log-moneyness \(\log(S_t/K)\), not raw spot;
- normalized variance such as \(v_t/\theta\), or a documented robust transform;
- relative time to maturity \(\tau_t/T\);
- previous hedge ratio;
- running hedging error normalized by \(S_0\) or initial option value, only in
  the corresponding ablation.

Raw spot makes periodic-frequency scales depend on the arbitrary price unit.
Shared and featurewise periodic embeddings are not comparable until their
inputs and frequency conventions are aligned.

Under an exact Markov state and the CARA/entropic objective, current wealth is
additively separable:

\[
  \frac{1}{a}\log \mathbb{E}_t[e^{-a(w_t+Y_T^\pi)}]
  =
  -w_t + \frac{1}{a}\log \mathbb{E}_t[e^{-aY_T^\pi]}.
\]

Therefore running PnL is not required by the ideal optimal policy merely
because utility is nonlinear or fees are present: future fees remain
action-dependent, while realized fees are sunk. It can nevertheless help a
finite learned policy by summarizing omitted path information, compensating
for an insufficient state, or changing optimization. Treat the observed gain
as an empirical question. In addition to the on/off ablation, measure
counterfactual hedge sensitivity to PnL while holding
\((S,v,\tau,\delta_{t-1})\) fixed. This distinguishes a genuinely used
path-dependent signal from an optimization scaffold.

## Guided search: the proposed Figure 1

Use **greedy stagewise selection**, analogous to a cumulative recipe ablation,
instead of a Cartesian grid or unconstrained random search.

### Development suite

Select components only on a small, predeclared Heston development suite that
contains more than one regime. A practical version is:

- European call;
- \(\xi \in \{0.1, 0.3\}\);
- costs \(\in \{0, 10\}\) bp;
- ATM and \(N=30\);
- five training seeds for reported stage results;
- a fixed, large validation path set per cell.

The final benchmark must contain held-out settings and, ideally, held-out
market families that never determine the recipe.

### Stages

At each stage, inherit the previous winner, compare only the declared
alternatives for the new axis, and accept the new component only if it improves
the aggregate development score without a material regression in worst-cell
performance.

1. **Vanilla reference:** normalized MLP, direct hedge, pathwise entropic
   training, Adam.
2. **Representation:** no embedding vs dense/shared Fourier projection vs
   featurewise periodic embedding.
3. **Frequency scale:** one-dimensional bracket/refinement for periodic
   initialization scale and a small capacity check. Refit the learning rate for
   each serious candidate.
4. **Analytical prior:** direct hedge vs local-GBM residual vs
   information-matched minimum-variance residual.
5. **Path feature:** without vs with normalized running hedging error.
6. **Optimization:** Adam, AdamW, K-FAC, Schedule-Free AdamW, AdamW with EMA,
   and Muon. EMA is an evaluation/training wrapper, not a distinct optimizer.
7. **Stochastic continuation:** deterministic pathwise training vs a clearly
   named entropy/action-noise continuation. Include true SAC only if a genuine
   actor-critic implementation is completed and validated.

For optimizer comparisons, tune at least learning rate per optimizer. Otherwise
the experiment measures compatibility with Adam’s learning rate rather than
optimizer quality.

### Selection score

Do not average raw SoftMin values across contracts with different scales.
Use an entropic-risk gap normalized by spot or option value, for example:

\[
  g_{m,c} =
  10^4\frac{\rho_a(\mathrm{PnL}_{m,c})-
                 \rho_a(\mathrm{PnL}_{\mathrm{MV},c})}{S_{0,c}},
\]

in basis points, with lower values better. Report mean rank or mean normalized
gap plus the worst-cell gap. The final wording and denominator must be frozen
before the holdout is opened.

### Order-dependence control

Cumulative curves depend on component order. Add a leave-one-component-out
ablation of the frozen final recipe. This is more important than testing every
Cartesian interaction and fits in one compact table.

## Final benchmark

Avoid a full Cartesian product. It is expensive, hard to interpret, and cannot
fit in eight pages. Use an axis-isolated benchmark around a declared reference
cell, plus a small interaction panel if compute permits.

### Mandatory market/contract cells

1. **GBM sanity:** complete market, call, zero costs, increasingly fine
   rebalancing. The learned hedge should approach the Black--Scholes hedge.
2. **Heston main:** call, ATM, \(\xi=0.3\), 10 bp, \(N=30\).
3. **Heston vol-of-vol axis:** \(\xi \in \{0.1,0.3,0.5\}\).
4. **Moneyness axis:** \(K/S_0 \in \{0.9,1.0,1.1\}\).
5. **Cost axis:** \(0,5,10,30\) bp.
6. **Horizon axis:** \(N \in \{30,60,120,300\}\).
7. **Payoff axis:** European call and digital call at the reference market
   cell.

Deduplicate the common reference cell. This gives roughly 13 Heston cells plus
the GBM sanity cells, rather than hundreds of Cartesian combinations.

### Rough Bergomi gate

Rough Bergomi is a useful out-of-family test only if all of the following are
ready before the recipe is frozen:

- the simulator is validated;
- the network receives a sufficient representation of the non-Markovian
  information, not merely current spot and variance;
- the analytical variance-optimal/Bartlett hedge is implemented from the cited
  model and receives exactly the same information;
- discretization and hedge calculations have numerical tests.

If this gate fails, omit rough Bergomi from ICAIF rather than include an unfair
or weak baseline. Bates is not a zero-cost replacement: jumps add a term to the
variance-optimal hedge, so a Heston Greek adjustment alone is not necessarily
the strongest mathematical comparator.

### Methods reported in the main benchmark

- no hedge, when informative;
- ordinary model delta;
- information-matched minimum-variance delta;
- transaction-cost-aware analytical baseline, with any no-trade width tuned on
  validation only;
- vanilla Deep Hedging;
- frozen selected recipe.

Weaker or diagnostic methods can remain in the repository without consuming
main-paper space.

### Seeds and evaluation

- Five seeds are acceptable for development; use ten for the final learned
  methods if compute permits.
- Seed Python/NumPy, PyTorch CPU/CUDA, initialization, training path streams,
  and stochastic policies.
- Pair method seeds deliberately.
- Evaluate every learned seed and analytical baseline on the same large,
  held-out path set for that market cell.
- Use paired bootstrap intervals over paths and summarize optimization
  variability across training seeds.
- Never select a “best analytical baseline” after observing test performance;
  define or tune it on development/validation data.

## Metrics and terminology

Primary:

- entropic risk/certainty-equivalent hedging cost at the training risk
  aversion, normalized to basis points.

Secondary:

- CVaR of **loss** at 95% and 99%;
- mean and standard deviation of terminal hedging error;
- transaction fees and turnover;
- seed standard deviation;
- wall-clock time and time/steps to a declared target risk;
- deviation from the information-matched analytical hedge.

The current quantity called PnL omits the option premium and is approximately
the negative option value. Either add the common initial premium and call the
result terminal hedging PnL, or call the existing quantity terminal hedging
cost/error. Do not label the entropic risk measure “CVaR-like.”

Risk aversion has units of inverse currency. Either normalize terminal hedging
error before applying the entropic objective or scale \(a\) consistently with
contract notional/spot. This is especially important when comparing a call
with a unit-paying digital. Compute SoftMin with `logsumexp`, use a much larger
test set than the current 5,000 paths, and check that tail-weight concentration
does not make the estimate effectively depend on only a handful of paths.

## Figures and tables

### Figure 1 — From vanilla DH to the frozen recipe

A cumulative line/bar plot over accepted stages:

`MLP → periodic representation → analytical residual → PnL decision →
optimizer → stochastic-continuation decision`.

- y-axis: aggregate normalized entropic-risk improvement over vanilla DH;
- error bars: seed uncertainty across the development suite;
- rejected/negative components may be shown in gray or reported in Table 1;
- caption must state that order dependence is checked by leave-one-out
  ablations.

### Figure 2 — Held-out performance map

Compact heatmap or dot plot of recipe minus strong analytical baseline in basis
points over the held-out cells. Use color centered at zero. Mark uncertainty or
ties; do not imply all differences are significant.

### Table 1 — What matters

Frozen recipe and leave-one-component-out variants with:

- aggregate gap;
- worst-cell gap;
- seed standard deviation;
- runtime multiplier.

### Table 2 — Main comparison

Aggregate vanilla DH, selected recipe, and strongest analytical hedge by
market/payoff group. Include win/tie/loss and the primary risk metric; move
secondary metrics into a compact second panel if space permits.

The current notebook learning-curve cells that extend short logs by duplicating
their final observation must never be used as paper figures.

## Software shape

The “one class” idea is valid as a public API, but it should be a facade over
composable internals rather than one conditional implementation.

Proposed interface:

```python
hedger = DeepHedger(
    recipe="icaif26",
    anchor=market.minimum_variance_hedge,  # optional
    seed=0,
)
hedger.fit(market, claim)
delta = hedger.hedge(state)
report = hedger.evaluate(test_paths)
```

Internally keep separate:

- `MarketModel` / sampler;
- `Claim` / payoff;
- `StateEncoder`;
- `AnchorHedge`;
- `Policy`;
- `Trainer`;
- `RiskMeasure`;
- `Evaluator`.

Expose the individual knobs in a research configuration, but make the end-user
surface a versioned preset plus a few meaningful choices. The paper artifact is
the frozen preset, not the existence of a large hyperparameter namespace.

## Correctness gates before new paper runs

1. **Accounting:** use discounted units consistently or accrue the cash account
   at the risk-free rate in every learned and analytical backtest.
2. **Payoff abstraction:** remove the hard-coded European-call payoff from
   trainers/backtests before the payoff experiment.
3. **Time scaling:** replace hard-coded `linspace(1, 0, ...)` assumptions with
   the configured maturity.
4. **Seeding:** deterministic, independent train/validation/test streams.
5. **Strong delta:** validate batched Heston price, delta, and variance
   sensitivity on a grid of spot, variance, and maturity; current one-slice
   evidence is preliminary.
6. **Information fairness:** baseline and network receive the same observable
   state.
7. **Evaluation freeze:** write the primary metric, tie margin, settings, and
   selection rule before final tests.
8. **Trainer naming:** drop/fix invalid PPO/Bellman code and rename the current
   entropy trainer unless true SAC is implemented.
9. **Risk estimator:** normalize the risk-aversion convention, use stable
   `logsumexp`, and validate Monte Carlo convergence of the entropic metric.
10. **Packaging:** create an installable project, tests, and the public facade
   before making a pip-style usability claim.

## Single ICAIF submission execution order

### July 24–25

- freeze the paper protocol and anonymous template;
- fix accounting, payoff/state abstractions, seeds, metrics, and runner;
- turn the Heston delta validation into tests;
- reproduce the current reference cell.

### July 26–27

- run the stagewise development search with five seeds;
- complete the one-dimensional periodic-scale refinement;
- freeze the recipe and the Figure 1 stages;
- decide the rough-Bergomi gate.

### July 28–30

- run the frozen benchmark with ten seeds for learned methods;
- run leave-one-component-out ablations;
- aggregate paired uncertainty, turnover, and runtime.

### July 31

- finalize Figures 1–2 and Tables 1–2;
- replace all result placeholders;
- tighten claims to the observed evidence.

### August 1–2

- reproduce the PDF from clean commands;
- verify eight-page total limit, anonymity, citations, fonts, and CMT metadata;
- perform a final claim/evidence audit and submit.

## Current evidence status

Useful only as preliminary motivation:

- batched Heston CF delta currently matches the old SciPy delta on one
  five-state slice with a reported maximum absolute difference around
  \(1.9\times10^{-6}\);
- on the current single Heston run, the best deviation model reports lower
  SoftMin than the raw Heston minimum-variance hedge at both zero and 10 bp
  transaction costs;
- the result uses one training seed, 5,000 test paths, raw-scale features, a
  local-BSM rather than MV residual anchor, and inconsistent global seeding.

Therefore these numbers may appear only as explicitly preliminary placeholders
in mark-0. They are not evidence for the final abstract or conclusion.
