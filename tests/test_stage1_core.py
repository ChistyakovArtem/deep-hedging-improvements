from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from deep_hedger import DeepHedger as PublicDeepHedger
from src.evaluation import backtest_actions, entropic_risk
from src.hedger import DeepHedger, DeepHedgerConfig
from src.market import TorchHestonStream, paths_sha256, sample_heston_numpy


MARKET = {
    "model": "heston",
    "S0": 100.0,
    "K": 100.0,
    "v0": 0.04,
    "r": 0.01,
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.7,
    "T": 1.0,
    "N": 3,
    "transaction_cost": 0.001,
}
OBJECTIVE = {"risk_aversion": 1.0, "pnl_scale": 1.0}


def test_numpy_leaderboard_is_reproducible_and_seed_separated() -> None:
    first = sample_heston_numpy(MARKET, 128, seed=1)
    second = sample_heston_numpy(MARKET, 128, seed=1)
    other = sample_heston_numpy(MARKET, 128, seed=2)
    assert np.array_equal(first, second)
    assert paths_sha256(first) == paths_sha256(second)
    assert not np.array_equal(first, other)


def test_torch_training_stream_does_not_reset_each_sample() -> None:
    stream = TorchHestonStream(MARKET, seed=1, device=torch.device("cpu"))
    first = stream.sample(32)
    second = stream.sample(32)
    assert not torch.equal(first, second)
    replay = TorchHestonStream(MARKET, seed=1, device=torch.device("cpu"))
    assert torch.equal(first, replay.sample(32))


def test_discounted_zero_hedge_accounting() -> None:
    paths = torch.tensor(
        [
            [[100.0, 0.04], [110.0, 0.04], [120.0, 0.04], [130.0, 0.04]],
            [[100.0, 0.04], [90.0, 0.04], [80.0, 0.04], [70.0, 0.04]],
        ]
    )
    actions = torch.zeros((2, 3))
    outputs = backtest_actions(paths, actions, MARKET)
    expected = torch.tensor(
        [-30.0 * np.exp(-0.01), 0.0], dtype=outputs["pnl"].dtype
    )
    assert torch.allclose(outputs["pnl"], expected, atol=1e-6)
    assert torch.equal(outputs["fees"], torch.zeros(2))


def test_entropic_risk_is_translation_equivariant() -> None:
    pnl = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    shift = 3.5
    assert torch.allclose(
        entropic_risk(pnl + shift, 0.7),
        entropic_risk(pnl, 0.7) - shift,
    )


def test_one_class_builds_every_stage1_architecture() -> None:
    assert PublicDeepHedger is DeepHedger
    paths = torch.as_tensor(sample_heston_numpy(MARKET, 16, seed=3))
    base = DeepHedgerConfig(
        n_epochs=1,
        paths_per_epoch=8,
        validation_interval=1,
    )
    for architecture in ("mlp", "paf_shared", "paf_featurewise"):
        config = replace(base, architecture=architecture)
        hedger = DeepHedger(MARKET, OBJECTIVE, config=config, device="cpu")
        actions, outputs = hedger.evaluate_tensor(paths)
        assert actions.shape == (16, 3)
        assert outputs["pnl"].shape == (16,)
        assert torch.isfinite(actions).all()
