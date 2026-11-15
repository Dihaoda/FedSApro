from pathlib import Path

import pytest

from fedsapro.config import ATTACKS, METHODS, PAPER_MODEL_KEYS, load_yaml, resolve_config


ROOT = Path(__file__).resolve().parents[1]


def test_targets_cover_full_matrix() -> None:
    targets = load_yaml(ROOT / "configs/paper/targets.yaml")
    assert set(targets) == set(PAPER_MODEL_KEYS)
    for model in PAPER_MODEL_KEYS:
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


def test_efficient_profile_supports_small_qwen_alias() -> None:
    config = resolve_config(
        ROOT / "configs/reproduction/efficient.yaml", model="qwen2.5-0.5b"
    )
    assert config["model"]["names"]["qwen2.5-0.5b"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["reporting"]["target_model_aliases"]["qwen2.5-0.5b"] == "qwen2.5-1.5b"


def test_server_learning_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="server_learning_rate"):
        resolve_config(
            ROOT / "configs/reproduction/efficient.yaml",
            overrides={"federated.server_learning_rate": 0.0},
        )


def test_attack_update_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attack.update_scale"):
        resolve_config(
            ROOT / "configs/reproduction/efficient.yaml",
            overrides={"attack.update_scale": 0.0},
        )


def test_attack_specific_update_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attack.update_scales.label_flip"):
        resolve_config(
            ROOT / "configs/reproduction/efficient.yaml",
            overrides={"attack.update_scales.label_flip": 0.0},
        )


def test_attacker_domain_fraction_must_be_valid() -> None:
    with pytest.raises(ValueError, match="attacker Amazon fraction"):
        resolve_config(
            ROOT / "configs/reproduction/efficient.yaml",
            overrides={"data.attacker_amazon_fraction": 1.1},
        )


def test_scaffold_control_scale_must_be_valid() -> None:
    with pytest.raises(ValueError, match="scaffold_control_scale"):
        resolve_config(
            ROOT / "configs/reproduction/efficient.yaml",
            overrides={"method.scaffold_control_scale": 0.0},
        )


def test_method_server_learning_rate_must_be_valid() -> None:
    with pytest.raises(ValueError, match="server_learning_rates.fedprox"):
        resolve_config(
            ROOT / "configs/reproduction/efficient.yaml",
            overrides={"method.server_learning_rates.fedprox": 0.0},
        )


