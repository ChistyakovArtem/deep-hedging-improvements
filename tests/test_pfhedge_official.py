from __future__ import annotations

import pytest
import torch

from dev.generate_pfhedge_official import configs
from dev.make_pfhedge_official_nirvana_toml import build_graphs, validate_graphs
from src.hedger import DeepHedger, DeepHedgerConfig
from src.market import sample_heston_numpy
from src.pfhedge_baselines import (
    PFHEDGE_VERSION,
    PFHEDGE_WHEEL_SHA256,
    PFHedgeBaseline,
    PFHedgeBaselineConfig,
)


pfhedge = pytest.importorskip("pfhedge")

MARKET = {
    "model": "heston",
    "S0": 100.0,
    "K": 100.0,
    "v0": 0.04,
    "r": 0.0,
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.7,
    "T": 1.0,
    "N": 3,
    "transaction_cost": 0.001,
}
OBJECTIVE = {"risk_aversion": 1.0, "pnl_scale": 1.0}


def test_official_pfhedge_release_is_exactly_pinned() -> None:
    assert pfhedge.__version__ == PFHEDGE_VERSION
    assert len(PFHEDGE_WHEEL_SHA256) == 64


def test_readme_ntbn_matches_project_ntbn_when_weights_and_rate_match() -> None:
    config = DeepHedgerConfig(
        architecture="ntbn",
        feature_mode="ntbn_paper",
        hidden_dims=(32, 32, 32, 32),
        activation="relu",
        seed=7,
    )
    project = DeepHedger(MARKET, OBJECTIVE, config=config, device="cpu")
    official = PFHedgeBaseline(
        MARKET,
        OBJECTIVE,
        PFHedgeBaselineConfig(model_name="ntbn", seed=7),
        device="cpu",
    )
    official.policy.mlp.load_state_dict(project.policy.mlp.state_dict())
    paths = torch.as_tensor(sample_heston_numpy(MARKET, 64, seed=91))
    with torch.no_grad():
        expected = project.predict_actions(paths)
        actual = official.predict_actions(paths)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_pfhedge_matrix_and_nirvana_volley_are_complete() -> None:
    generated = configs()
    assert len(generated) == 36
    assert sum(bool(payload["model"]["trainable"]) for _, payload in generated) == 32
    assert all(not payload["evaluation"]["private_access"] for _, payload in generated)
    assert all(
        payload["provenance"]["wheel_sha256"] == PFHEDGE_WHEEL_SHA256
        for _, payload in generated
    )
    full = build_graphs()
    smoke = build_graphs(smoke=True)
    validate_graphs(full)
    validate_graphs(smoke, smoke=True)
    assert all(
        "VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install"
        in graph["venv_commands"]
        for graph in full
    )
