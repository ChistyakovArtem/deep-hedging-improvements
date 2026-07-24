from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn

from dev.generate_optimizer_study import OPTIMIZER_CANDIDATES, configs
from dev.make_optimizer_study_nirvana_toml import build_graphs, validate_graphs
from src.config import PROJECT_ROOT, load_toml
from src.hedger import DeepHedger, DeepHedgerConfig
from src.market import sample_heston_numpy
from src.optimizers import make_optimizer_controller


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


def test_optimizer_study_is_paired_5_by_5_by_2_by_8() -> None:
    generated = configs()
    assert len(OPTIMIZER_CANDIDATES) == 5
    assert all(len(candidates) == 5 for candidates in OPTIMIZER_CANDIDATES.values())
    assert len(generated) == 5 * 5 * 2 * 8

    seen = set()
    for directory, config in generated:
        baseline = load_toml(PROJECT_ROOT / config["provenance"]["baseline_config"])
        for section in (
            "market",
            "objective",
            "leaderboard",
            "model",
            "reference_selection",
            "evaluation",
        ):
            assert config[section] == baseline[section]
        assert config["evaluation"]["private_access"] is False
        assert config["provenance"]["no_paf"] is True
        assert config["provenance"]["no_running_pnl"] is True
        assert config["training"]["n_epochs"] == baseline["training"]["n_epochs"]
        assert config["training"]["paths_per_epoch"] == baseline["training"][
            "paths_per_epoch"
        ]
        key = (
            float(config["market"]["xi"]),
            int(config["seed"]),
            str(config["experiment"]["optimizer"]),
            int(config["experiment"]["hp_id"]),
        )
        assert key not in seen
        seen.add(key)
        assert directory.name.startswith(
            f"hp-{int(config['experiment']['hp_id']):02d}-"
        )


def test_optimizer_nirvana_volley_is_eight_balanced_seed_queues() -> None:
    full = build_graphs()
    smoke = build_graphs(smoke=True)
    validate_graphs(full)
    validate_graphs(smoke, smoke=True)
    assert len(full) == 8
    assert all(graph["n_gpus"] == 1 for graph in full)
    assert all(graph["priority"] == "normal" for graph in full)


def test_one_class_trains_with_every_optimizer_family() -> None:
    public_paths = torch.as_tensor(sample_heston_numpy(MARKET, 16, seed=41))
    base = DeepHedgerConfig(
        hidden_dims=(8, 4),
        n_epochs=2,
        paths_per_epoch=8,
        validation_interval=1,
    )
    for family in OPTIMIZER_CANDIDATES:
        candidate = OPTIMIZER_CANDIDATES[family][0]
        config = replace(
            base,
            optimizer=family,
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            ema_decay=float(candidate["ema_decay"]),
            muon_learning_rate=float(candidate["muon_learning_rate"]),
            muon_momentum=float(candidate["muon_momentum"]),
        )
        hedger = DeepHedger(MARKET, OBJECTIVE, config=config, device="cpu")
        log = hedger.fit(public_paths)
        assert len(log) == 2
        assert hedger.best_epoch in {1, 2}
        assert hedger.best_public_risk is not None
        assert torch.isfinite(
            torch.tensor(float(hedger.best_public_risk))
        )


def test_muon_is_applied_to_hidden_weights_but_not_output_head() -> None:
    hedger = DeepHedger(
        MARKET,
        OBJECTIVE,
        config=DeepHedgerConfig(hidden_dims=(8, 4)),
        device="cpu",
    )
    controller = make_optimizer_controller(
        hedger.policy,
        family="muon",
        learning_rate=1e-3,
        weight_decay=1e-4,
        ema_decay=0.99,
        muon_learning_rate=0.02,
        muon_momentum=0.95,
    )
    muon_parameters = {
        parameter
        for group in controller.optimizer.param_groups
        if group["use_muon"]
        for parameter in group["params"]
    }
    linear_layers = [
        module for module in hedger.policy.mlp if isinstance(module, nn.Linear)
    ]
    assert {layer.weight for layer in linear_layers[:-1]} == muon_parameters
    assert linear_layers[-1].weight not in muon_parameters


def test_ema_validation_uses_averaged_policy_without_replacing_online_policy() -> None:
    hedger = DeepHedger(
        MARKET,
        OBJECTIVE,
        config=DeepHedgerConfig(hidden_dims=(8, 4)),
        device="cpu",
    )
    controller = make_optimizer_controller(
        hedger.policy,
        family="adamw_ema",
        learning_rate=1e-3,
        weight_decay=1e-4,
        ema_decay=0.9,
        muon_learning_rate=0.02,
        muon_momentum=0.95,
    )
    online = hedger.policy
    for parameter in online.parameters():
        parameter.grad = torch.ones_like(parameter)
    controller.step()
    for parameter in online.parameters():
        parameter.grad = torch.ones_like(parameter)
    controller.step()
    with controller.validation_policy() as validation:
        assert validation is not online
        assert any(
            not torch.equal(averaged, current)
            for averaged, current in zip(
                validation.parameters(),
                online.parameters(),
                strict=True,
            )
        )
    assert hedger.policy is online


def test_adam_anchor_retains_the_exact_previous_training_fields() -> None:
    _, config = next(
        (directory, config)
        for directory, config in configs()
        if config["market"]["xi"] == 0.1
        and config["seed"] == 0
        and config["experiment"]["optimizer"] == "adam"
        and config["experiment"]["hp_id"] == 0
    )
    baseline = load_toml(PROJECT_ROOT / config["provenance"]["baseline_config"])
    candidate_training = deepcopy(config["training"])
    for field in (
        "weight_decay",
        "ema_decay",
        "muon_learning_rate",
        "muon_momentum",
    ):
        candidate_training.pop(field)
    assert candidate_training == baseline["training"]
