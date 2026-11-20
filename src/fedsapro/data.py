from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, MutableSet

from .config import config_digest, get_path
from .utils import atomic_json_dump, derive_seed, read_jsonl, sha256_file, write_jsonl


STANDARD = "standard"
HETEROGENEOUS = "heterogeneous"
ATTACKER = "attacker"


def _normalise_text(example: Mapping[str, Any], domain: str) -> str:
    if domain == "amazon":
        title = str(example.get("title", "")).strip()
        content = str(example.get("content", example.get("text", ""))).strip()
        text = f"{title}\n{content}" if title else content
    else:
        text = str(example.get("text", example.get("content", ""))).strip()
    return " ".join(text.split())


def _sample_balanced(
    dataset: Any,
    *,
    domain: str,
    count: int,
    rng: random.Random,
    used: MutableSet[int],
) -> list[dict[str, Any]]:
    if count % 2:
        raise ValueError("Balanced sample count must be even")
    quota = {0: count // 2, 1: count // 2}
    selected: list[dict[str, Any]] = []
    attempts = 0
    limit = max(10_000, count * 100)
    while len(selected) < count:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(f"Could not draw {count} balanced examples from {domain}")
        index = rng.randrange(len(dataset))
        if index in used:
            continue
        example = dataset[index]
        label = int(example["label"])
        if label not in quota or quota[label] <= 0:
            continue
        text = _normalise_text(example, domain)
        if not text:
            continue
        used.add(index)
        quota[label] -= 1
        selected.append(
            {
                "text": text,
                "label": label,
                "source_domain": domain,
                "source_index": index,
            }
        )
    rng.shuffle(selected)
    return selected


def _sample_balanced_by_length(
    dataset: Any,
    *,
    domain: str,
    count: int,
    longest: bool,
) -> list[dict[str, Any]]:
    """Select a label-balanced structural stress subset without using predictions."""
    if count % 2:
        raise ValueError("Balanced sample count must be even")
    candidates: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for index, example in enumerate(dataset):
        label = int(example["label"])
        if label not in candidates:
            continue
        text = _normalise_text(example, domain)
        if text:
            candidates[label].append((len(text.split()), index))
    selected: list[dict[str, Any]] = []
    per_label = count // 2
    for label in (0, 1):
        ranked = sorted(candidates[label], reverse=longest)[:per_label]
        if len(ranked) != per_label:
            raise RuntimeError(f"Could not select {per_label} length-ranked {domain} rows")
        for _, index in ranked:
            example = dataset[index]
            selected.append(
                {
                    "text": _normalise_text(example, domain),
                    "label": label,
                    "source_domain": domain,
                    "source_index": index,
                }
            )
    random.Random(0).shuffle(selected)
    return selected


def _sample_balanced_lexical_control(
    dataset: Any, *, domain: str, count: int
) -> list[dict[str, Any]]:
    """Build an easy, label-balanced control set from a fixed sentiment lexicon."""
    if count % 2:
        raise ValueError("Balanced sample count must be even")
    cues = {
        0: ("terrible", "awful", "worst", "hate", "horrible", "poor", "useless"),
        1: ("excellent", "great", "amazing", "perfect", "love", "wonderful", "best"),
    }
    candidates: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    for index, example in enumerate(dataset):
        label = int(example["label"])
        if label not in candidates:
            continue
        text = _normalise_text(example, domain)
        lowered = text.lower()
        score = sum(lowered.count(cue) for cue in cues[label])
        if score:
            candidates[label].append((-score, len(text.split()), index))
    selected: list[dict[str, Any]] = []
    per_label = count // 2
    for label in (0, 1):
        ranked = sorted(candidates[label])[:per_label]
        if len(ranked) != per_label:
            raise RuntimeError(f"Could not select {per_label} lexical-control {domain} rows")
        for _, _, index in ranked:
            example = dataset[index]
            selected.append(
                {
                    "text": _normalise_text(example, domain),
                    "label": label,
                    "source_domain": domain,
                    "source_index": index,
                }
            )
    random.Random(0).shuffle(selected)
    return selected


def _role_map(config: Mapping[str, Any]) -> dict[int, str]:
    standard_count = int(get_path(config, "data.standard_clients"))
    hetero_count = int(get_path(config, "data.heterogeneous_clients"))
    attacker_count = int(get_path(config, "data.attacker_clients"))
    roles: dict[int, str] = {}
    for client_id in range(standard_count):
        roles[client_id] = STANDARD
    for client_id in range(standard_count, standard_count + hetero_count):
        roles[client_id] = HETEROGENEOUS
    for client_id in range(
        standard_count + hetero_count, standard_count + hetero_count + attacker_count
    ):
        roles[client_id] = ATTACKER
    return roles


def prepare_data(config: Mapping[str, Any], *, force: bool = False) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before preparing data") from exc

    root = Path(str(get_path(config, "experiment.data_root")))
    manifest_path = root / "manifest.json"
    partition_seed = int(config["data"].get("partition_seed", get_path(config, "experiment.seed")))
    expected_digest = config_digest({"data": config["data"], "seed": partition_seed})
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("partition_digest") != expected_digest:
            raise RuntimeError(
                f"Existing data at {root} was built with a different partition configuration; "
                "choose another data_root or pass --force"
            )
        return root

    root.mkdir(parents=True, exist_ok=True)
    (root / "clients").mkdir(parents=True, exist_ok=True)
    seed = partition_seed
    cache_dir = Path(str(get_path(config, "experiment.model_cache"))) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    amazon_name = str(get_path(config, "data.amazon_dataset"))
    imdb_name = str(get_path(config, "data.imdb_dataset"))
    amazon_train = load_dataset(amazon_name, split="train", cache_dir=str(cache_dir))
    amazon_test = load_dataset(amazon_name, split="test", cache_dir=str(cache_dir))
    imdb_train = load_dataset(imdb_name, split="train", cache_dir=str(cache_dir))
    imdb_test = load_dataset(imdb_name, split="test", cache_dir=str(cache_dir))

    used_amazon: set[int] = set()
    used_imdb: set[int] = set()
    samples_per_client = int(get_path(config, "data.samples_per_client"))
    heterogeneous_amazon_fraction = float(
        get_path(config, "data.heterogeneous_amazon_fraction")
    )
    attacker_amazon_fraction = float(config["data"].get("attacker_amazon_fraction", 1.0))
    roles = _role_map(config)
    client_files: dict[str, dict[str, Any]] = {}

    for client_id, role in roles.items():
        rng = random.Random(derive_seed(seed, 10, client_id))
        if role in {HETEROGENEOUS, ATTACKER}:
            amazon_fraction = (
                heterogeneous_amazon_fraction
                if role == HETEROGENEOUS
                else attacker_amazon_fraction
            )
            amazon_count = int(round(samples_per_client * amazon_fraction))
            imdb_count = samples_per_client - amazon_count
            rows = _sample_balanced(
                amazon_train,
                domain="amazon",
                count=amazon_count,
                rng=rng,
                used=used_amazon,
            )
            rows.extend(
                _sample_balanced(
                    imdb_train,
                    domain="imdb",
                    count=imdb_count,
                    rng=rng,
                    used=used_imdb,
                )
            )
            rng.shuffle(rows)
        else:
            rows = _sample_balanced(
                amazon_train,
                domain="amazon",
                count=samples_per_client,
                rng=rng,
                used=used_amazon,
            )
        for row in rows:
            row["client_id"] = client_id
            row["role"] = role
        path = root / "clients" / f"client_{client_id:02d}.jsonl"
        write_jsonl(rows, path)
        client_files[str(client_id)] = {
            "role": role,
            "samples": len(rows),
            "sha256": sha256_file(path),
        }

    root_rows = _sample_balanced(
        amazon_train,
        domain="amazon",
        count=int(get_path(config, "data.root_samples")),
        rng=random.Random(derive_seed(seed, 20)),
        used=used_amazon,
    )
    write_jsonl(root_rows, root / "server_root.jsonl")

    test_count = int(get_path(config, "data.test_samples_per_domain"))
    test_profile = str(config["data"].get("test_profile", "random"))
    if test_profile in {"length_extremes", "lexical_length_extremes"}:
        if test_profile == "lexical_length_extremes":
            amazon_test_rows = _sample_balanced_lexical_control(
                amazon_test, domain="amazon", count=test_count
            )
        else:
            amazon_test_rows = _sample_balanced_by_length(
                amazon_test, domain="amazon", count=test_count, longest=False
            )
        imdb_test_rows = _sample_balanced_by_length(
            imdb_test, domain="imdb", count=test_count, longest=True
        )
    elif test_profile == "random":
        amazon_test_rows = _sample_balanced(
            amazon_test,
            domain="amazon",
            count=test_count,
            rng=random.Random(derive_seed(seed, 30)),
            used=set(),
        )
        imdb_test_rows = _sample_balanced(
            imdb_test,
            domain="imdb",
            count=test_count,
            rng=random.Random(derive_seed(seed, 31)),
            used=set(),
        )
    else:
        raise ValueError(f"Unknown data.test_profile: {test_profile}")
    write_jsonl(amazon_test_rows, root / "test_amazon.jsonl")
    write_jsonl(imdb_test_rows, root / "test_imdb.jsonl")

    manifest = {
        "schema_version": 1,
        "partition_digest": expected_digest,
        "seed": seed,
        "sources": {"amazon": amazon_name, "imdb": imdb_name},
        "roles": {str(k): v for k, v in roles.items()},
        "clients": client_files,
        "files": {
            "server_root.jsonl": sha256_file(root / "server_root.jsonl"),
            "test_amazon.jsonl": sha256_file(root / "test_amazon.jsonl"),
            "test_imdb.jsonl": sha256_file(root / "test_imdb.jsonl"),
        },
    }
    atomic_json_dump(manifest, manifest_path)
    return root


def validate_data(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(get_path(config, "experiment.data_root")))
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = _role_map(config)
    expected_samples = int(get_path(config, "data.samples_per_client"))
    for client_id, role in roles.items():
        path = root / "clients" / f"client_{client_id:02d}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != expected_samples:
            raise RuntimeError(f"{path} contains {len(rows)} rows, expected {expected_samples}")
        if any(row.get("role") != role for row in rows):
            raise RuntimeError(f"Role mismatch in {path}")
        if sha256_file(path) != manifest["clients"][str(client_id)]["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {path}")
    return manifest


def load_client_rows(config: Mapping[str, Any], client_id: int) -> list[dict[str, Any]]:
    root = Path(str(get_path(config, "experiment.data_root")))
    return read_jsonl(root / "clients" / f"client_{client_id:02d}.jsonl")


def role_map(config: Mapping[str, Any]) -> dict[int, str]:
    return _role_map(config)


def apply_attack(
    rows: list[dict[str, Any]],
    *,
    attack: str,
    config: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    out = copy.deepcopy(rows)
    if attack == "none":
        for row in out:
            row["poisoned"] = False
        return out
    if attack == "label_flip":
        fraction = float(get_path(config, "attack.label_flip_fraction"))
    elif attack == "textual_backdoor":
        fraction = float(get_path(config, "attack.backdoor_fraction"))
    else:
        raise ValueError(f"Unknown attack: {attack}")

    count = min(len(out), max(0, int(math.ceil(len(out) * fraction))))
    indices = list(range(len(out)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[:count])
    trigger = str(get_path(config, "data.trigger"))
    trigger_position = str(config["data"].get("trigger_position", "prefix"))
    target = int(get_path(config, "attack.backdoor_target"))
    for index, row in enumerate(out):
        poisoned = index in selected
        row["poisoned"] = poisoned
        if not poisoned:
            continue
        if attack == "label_flip":
            row["label"] = 1 - int(row["label"])
        else:
            row["text"] = _insert_trigger(str(row["text"]), trigger, trigger_position)
            row["label"] = target
    return out


def _insert_trigger(text: str, trigger: str, position: str) -> str:
    if position == "prefix":
        return f"{trigger} {text}"
    if position == "suffix":
        return f"{text} {trigger}"
    raise ValueError(f"Unknown trigger position: {position}")


def triggered_evaluation_rows(
    rows: list[dict[str, Any]], *, trigger: str, target: int, position: str = "prefix"
) -> list[dict[str, Any]]:
    """Build a target-class-excluded triggered set for backdoor ASR evaluation."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if int(row["label"]) == target:
            continue
        changed = copy.deepcopy(row)
        changed["text"] = _insert_trigger(str(row["text"]), trigger, position)
        changed["trigger_target"] = target
        out.append(changed)
    return out
