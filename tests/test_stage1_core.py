from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from deep_hedger import DeepHedger as PublicDeepHedger
from src.evaluation import backtest_actions, entropic_risk
from src.hedger import DeepHedger, DeepHedgerConfig
from src.market import TorchHestonStream, paths_sha256, sample_heston_numpy
import src.nirvana_io as nirvana_io


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
    architectures = (
        ("mlp", "normalized"),
        ("ntbn", "ntbn_paper"),
        ("paf_shared", "normalized"),
        ("paf_featurewise", "normalized"),
    )
    for architecture, feature_mode in architectures:
        config = replace(
            base,
            architecture=architecture,
            feature_mode=feature_mode,
            hidden_dims=(32, 32, 32, 32) if architecture == "ntbn" else (64, 32),
            activation="relu" if architecture == "ntbn" else "leaky_relu",
        )
        hedger = DeepHedger(MARKET, OBJECTIVE, config=config, device="cpu")
        actions, outputs = hedger.evaluate_tensor(paths)
        assert actions.shape == (16, 3)
        assert outputs["pnl"].shape == (16,)
        assert torch.isfinite(actions).all()


def test_feature_study_modes_have_the_declared_inputs_and_time_sharing() -> None:
    paths = torch.tensor(
        [[[100.0, 0.04], [110.0, 0.05], [105.0, 0.03], [120.0, 0.06]]]
    )
    previous = torch.tensor([0.25])
    expected = {
        "legacy_raw": torch.tensor([[100.0, 0.04, 1.0, 0.25]]),
        "normalized": torch.tensor([[0.0, 1.0, 1.0, 0.25]]),
        "paper_log_state_with_time": torch.tensor(
            [[np.log(100.0), 0.04, 1.0, 0.25]], dtype=torch.float32
        ),
        "paper_log_state": torch.tensor(
            [[np.log(100.0), 0.04, 0.25]], dtype=torch.float32
        ),
    }
    for feature_mode, values in expected.items():
        time_parameterization = (
            "per_step" if feature_mode == "paper_log_state" else "shared"
        )
        hedger = DeepHedger(
            MARKET,
            OBJECTIVE,
            config=DeepHedgerConfig(
                feature_mode=feature_mode,
                time_parameterization=time_parameterization,
                hidden_dims=(16, 16),
                activation="relu",
                batch_norm=feature_mode.startswith("paper_"),
                n_epochs=1,
                paths_per_epoch=8,
                validation_interval=1,
            ),
            device="cpu",
        )
        assert torch.allclose(hedger._features(paths, 0, previous), values)
        if time_parameterization == "per_step":
            assert isinstance(hedger.policy, nn.ModuleList)
            assert len(hedger.policy) == MARKET["N"]
        else:
            assert not isinstance(hedger.policy, nn.ModuleList)


def test_paper_style_policy_uses_batch_norm_before_relu() -> None:
    hedger = DeepHedger(
        MARKET,
        OBJECTIVE,
        config=DeepHedgerConfig(
            feature_mode="paper_log_state_with_time",
            hidden_dims=(16, 16),
            activation="relu",
            batch_norm=True,
        ),
        device="cpu",
    )
    modules = list(hedger.policy.mlp)
    assert [type(module) for module in modules[:3]] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
    ]


def test_ntbn_features_and_local_black_scholes_center() -> None:
    market = dict(MARKET, r=0.0, T=1.0, N=3)
    hedger = DeepHedger(
        market,
        OBJECTIVE,
        config=DeepHedgerConfig(
            architecture="ntbn",
            feature_mode="ntbn_paper",
            hidden_dims=(32, 32, 32, 32),
            activation="relu",
        ),
        device="cpu",
    )
    paths = torch.tensor(
        [[[100.0, 0.04], [110.0, 0.05], [105.0, 0.03], [120.0, 0.06]]]
    )
    features = hedger._features(paths, 0, torch.tensor([0.25]))
    expected = torch.tensor([[0.0, 1.0, 0.2, 0.25]])
    assert torch.allclose(features, expected)
    expected_delta = torch.tensor([0.5398278])
    assert torch.allclose(
        hedger._ntbn_no_cost_delta(features),
        expected_delta,
        atol=1e-6,
    )


def test_ntbn_policy_keeps_previous_hedge_inside_learned_band() -> None:
    hedger = DeepHedger(
        dict(MARKET, r=0.0),
        OBJECTIVE,
        config=DeepHedgerConfig(
            architecture="ntbn",
            feature_mode="ntbn_paper",
            hidden_dims=(32, 32, 32, 32),
            activation="relu",
        ),
        device="cpu",
    )
    policy = hedger.policy
    assert not isinstance(policy, nn.ModuleList)
    for parameter in policy.parameters():
        nn.init.zeros_(parameter)
    final = policy.mlp[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.bias.fill_(0.2)

    features = torch.tensor(
        [
            [0.0, 1.0, 0.2, 0.50],
            [0.0, 1.0, 0.2, 0.00],
            [0.0, 1.0, 0.2, 1.00],
        ]
    )
    center = torch.full((3,), 0.5)
    action = policy(features, no_cost_delta=center)
    assert torch.allclose(action, torch.tensor([0.5, 0.3, 0.7]))


def test_ntbn_inverted_bounds_use_midpoint_like_authors_code() -> None:
    previous = torch.tensor([0.0, 1.0])
    lower = torch.tensor([0.6, 0.7])
    upper = torch.tensor([0.4, 0.3])
    midpoint = torch.tensor([0.5, 0.5])
    policy_type = type(
        DeepHedger(
            MARKET,
            OBJECTIVE,
            config=DeepHedgerConfig(
                architecture="ntbn",
                feature_mode="ntbn_paper",
            ),
            device="cpu",
        ).policy
    )
    assert torch.equal(
        policy_type._clamp_to_band(previous, lower, upper),
        midpoint,
    )


def test_snapshot_restore_republishes_completed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "output"
    relative = Path("exp/model/market/split/seed-000/cfg-0000")
    local_config = project / relative / "config.toml"
    snapshot_dir = snapshot / relative
    local_config.parent.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    local_config.write_text("[experiment]\nid = 0\n")
    (snapshot_dir / "config.toml").write_bytes(local_config.read_bytes())
    (snapshot_dir / "DONE").touch()
    (snapshot_dir / "scores-public.json").write_text('{"entropic_risk": 1.0}\n')

    monkeypatch.setattr(nirvana_io, "PROJECT_ROOT", project)
    monkeypatch.setenv("SNAPSHOT_PATH", str(snapshot))
    monkeypatch.setenv("TMP_OUTPUT_PATH", str(output))

    counts = nirvana_io.restore_snapshot()

    assert counts == {
        "restored": 1,
        "republished_output": 1,
        "skipped_incompatible": 0,
    }
    assert (project / relative / "scores-public.json").exists()
    assert (output / relative / "scores-public.json").exists()
