from pathlib import Path

import pytest

from fedsapro.config import ATTACKS, PAPER_METHODS, PAPER_MODEL_KEYS, load_yaml, resolve_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs/reproduction/natural_domain_attack_study.yaml"


def test_targets_cover_full_matrix() -> None:
    targets = load_yaml(ROOT / "configs/paper/targets.yaml")
    assert set(targets) == set(PAPER_MODEL_KEYS)
    for model in PAPER_MODEL_KEYS:
        assert set(targets[model]) == set(ATTACKS)
        for attack in ATTACKS:
            assert set(targets[model][attack]) == set(PAPER_METHODS)
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


def test_formal_profile_supports_small_qwen_alias() -> None:
    config = resolve_config(FORMAL_CONFIG, model="qwen2.5-0.5b")
    assert config["model"]["names"]["qwen2.5-0.5b"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["reporting"]["target_model_aliases"]["qwen2.5-0.5b"] == "qwen2.5-1.5b"


def test_server_learning_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="server_learning_rate"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"federated.server_learning_rate": 0.0},
        )


def test_attack_update_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attack.update_scale"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"attack.update_scale": 0.0},
        )


def test_attack_specific_update_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attack.update_scales.label_flip"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"attack.update_scales.label_flip": 0.0},
        )


def test_attack_norm_ratio_target_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attack.norm_ratio_targets.label_flip"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"attack.norm_ratio_targets": {"label_flip": 0.0}},
        )


def test_safefedllm_norm_guard_fraction_must_be_valid() -> None:
    with pytest.raises(ValueError, match="norm_guard_fraction"):
        resolve_config(
            FORMAL_CONFIG,
            method="safefedllm",
            overrides={"method.safefedllm.norm_guard_fraction": 1.0},
        )


def test_attacker_domain_fraction_must_be_valid() -> None:
    with pytest.raises(ValueError, match="attacker Amazon fraction"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"data.attacker_amazon_fraction": 1.1},
        )


def test_configured_domain_datasets_are_validated() -> None:
    domains = {
        "standard": {
            "path": "yelp_polarity",
            "train_split": "train",
            "test_split": "test",
            "text_fields": ["text"],
            "label_map": {0: 0, 1: 1},
        },
        "heterogeneous": {
            "path": "cardiffnlp/tweet_eval",
            "name": "sentiment",
            "train_split": "train",
            "test_split": "test",
            "text_fields": ["text"],
            "label_map": {0: 0, 2: 1},
        },
    }
    config = resolve_config(
        FORMAL_CONFIG,
        overrides={
            "data.domain_datasets": domains,
            "data.heterogeneous_standard_fraction": 0.0,
            "data.attacker_standard_fraction": 1.0,
        },
    )

    assert config["data"]["domain_datasets"] == domains


def test_scaffold_control_scale_must_be_valid() -> None:
    with pytest.raises(ValueError, match="scaffold_control_scale"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"method.scaffold_control_scale": 0.0},
        )


def test_method_server_learning_rate_must_be_valid() -> None:
    with pytest.raises(ValueError, match="server_learning_rates.fedprox"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"method.server_learning_rates.fedprox": 0.0},
        )


def test_per_attack_server_learning_rates_must_be_valid() -> None:
    config = resolve_config(
        FORMAL_CONFIG,
        overrides={
            "method.server_learning_rates.flame": {
                "none": 1.0,
                "label_flip": 1.5,
                "textual_backdoor": 1.25,
            }
        },
    )
    assert config["method"]["server_learning_rates"]["flame"]["label_flip"] == 1.5

    with pytest.raises(ValueError, match="server_learning_rates.flame"):
        resolve_config(
            FORMAL_CONFIG,
            overrides={"method.server_learning_rates.flame": {"unknown": 1.0}},
        )


