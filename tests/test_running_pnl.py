from __future__ import annotations

from copy import deepcopy

from dev.generate_running_pnl import configs
from dev.make_running_pnl_nirvana_toml import build_graphs, validate_graphs
from src.config import PROJECT_ROOT, load_toml


def test_running_pnl_configs_change_only_the_declared_feature() -> None:
    generated = configs()
    assert len(generated) == 16
    for _, config in generated:
        baseline = load_toml(PROJECT_ROOT / config["provenance"]["baseline_config"])
        assert config["seed"] == baseline["seed"]
        for section in (
            "market",
            "objective",
            "leaderboard",
            "training",
            "reference_selection",
            "evaluation",
        ):
            assert config[section] == baseline[section]

        model = deepcopy(config["model"])
        baseline_model = deepcopy(baseline["model"])
        assert model.pop("use_running_pnl") is True
        assert model.pop("running_pnl_scale") == 1.0
        assert model.pop("feature_names") == [
            *baseline_model.pop("feature_names"),
            "running_pnl",
        ]
        assert model == baseline_model
        assert config["evaluation"]["private_access"] is False


def test_running_pnl_nirvana_volley_is_eight_paired_seed_queues() -> None:
    full = build_graphs()
    smoke = build_graphs(smoke=True)
    validate_graphs(full)
    validate_graphs(smoke, smoke=True)
    assert len(full) == 8
    assert all(graph["n_gpus"] == 1 for graph in full)
