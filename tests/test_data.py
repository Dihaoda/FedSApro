import random
from pathlib import Path

from fedsapro.config import resolve_config
from fedsapro.data import _sample_balanced, apply_attack, triggered_evaluation_rows


ROOT = Path(__file__).resolve().parents[1]


def _rows(count: int = 20) -> list[dict]:
    return [{"text": f"review {index}", "label": index % 2} for index in range(count)]


def test_label_flip_changes_exact_fraction() -> None:
    config = resolve_config(ROOT / "configs/paper/base.yaml")
    attacked = apply_attack(_rows(), attack="label_flip", config=config, seed=7)
    assert sum(row["poisoned"] for row in attacked) == 6
    for original, changed in zip(_rows(), attacked):
        expected = 1 - original["label"] if changed["poisoned"] else original["label"]
        assert changed["label"] == expected


def test_textual_backdoor_changes_exact_fraction() -> None:
    config = resolve_config(ROOT / "configs/paper/base.yaml")
    attacked = apply_attack(_rows(), attack="textual_backdoor", config=config, seed=11)
    assert sum(row["poisoned"] for row in attacked) == 2
    trigger = config["data"]["trigger"]
    for row in attacked:
        if row["poisoned"]:
            assert row["text"].startswith(trigger)
            assert row["label"] == 1


def test_triggered_evaluation_excludes_target_class() -> None:
    triggered = triggered_evaluation_rows(_rows(10), trigger="TRIGGER", target=1)
    assert len(triggered) == 5
    assert all(row["label"] == 0 for row in triggered)
    assert all(row["text"].startswith("TRIGGER ") for row in triggered)
    assert all(row["trigger_target"] == 1 for row in triggered)


def test_triggered_evaluation_can_suffix_trigger() -> None:
    triggered = triggered_evaluation_rows(
        _rows(4), trigger="TRIGGER", target=1, position="suffix"
    )
    assert all(row["text"].endswith(" TRIGGER") for row in triggered)


def test_configured_domain_adapter_maps_labels_and_text() -> None:
    dataset = [
        {"sentence": "negative one", "sentiment": 0},
        {"sentence": "neutral row", "sentiment": 1},
        {"sentence": "positive one", "sentiment": 2},
        {"sentence": "negative two", "sentiment": 0},
        {"sentence": "positive two", "sentiment": 2},
    ]
    rows = _sample_balanced(
        dataset,
        domain="social",
        count=4,
        rng=random.Random(7),
        used=set(),
        spec={
            "text_fields": ["sentence"],
            "label_field": "sentiment",
            "label_map": {0: 0, 2: 1},
        },
    )

    assert sorted(row["label"] for row in rows) == [0, 0, 1, 1]
    assert all(row["source_domain"] == "social" for row in rows)
    assert all("neutral" not in row["text"] for row in rows)

