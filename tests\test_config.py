from pathlib import Path

from fedsapro.config import ATTACKS, METHODS, MODEL_KEYS, load_yaml, resolve_config


ROOT = Path(__file__).resolve().parents[1]


def test_targets_cover_full_matrix() -> None:
    targets = load_yaml(ROOT / "configs/paper/targets.yaml")
    assert set(targets) == set(MODEL_KEYS)
    for model in MODEL_KEYS:
        assert set(targets[model]) == set(ATTACKS)
        for attack in ATTACKS:
            assert set(targets[model][attack]) == set(METHODS)
            for values in targets[model][attack].values():
                assert set(values) == {"hssr", "standard_accuracy", "heterogeneous_accuracy"}


def test_smoke_config_is_valid_and_small() -> None:
    config = resolve_config(
        ROOT / "configs/paper/base.yaml",
        smoke=True,
        overrides={"federated.rounds": 2},
    )
    assert config["data"]["num_clients"] == 3
    assert config["federated"]["rounds"] == 2
    assert config["reporting"]["run_class"] == "smoke_test"

