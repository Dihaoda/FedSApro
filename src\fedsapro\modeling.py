from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import get_path
from .utils import seed_everything


LOGGER = logging.getLogger(__name__)


def build_prompt(text: str, labels: Sequence[str]) -> str:
    return (
        "Task: classify the sentiment of the review.\n"
        f"Review: {text}\n"
        f"Answer with exactly one label from {{{', '.join(labels)}}}.\n"
        "Sentiment:"
    )


class SentimentDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


@dataclass
class TrainingCollator:
    tokenizer: Any
    label_names: Sequence[str]
    max_length: int

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        for row in batch:
            prompt = build_prompt(str(row["text"]), self.label_names)
            label_text = str(self.label_names[int(row["label"])])
            label_ids = self.tokenizer.encode(" " + label_text, add_special_tokens=False)
            if not label_ids:
                raise RuntimeError(f"Label {label_text!r} has no tokenizer IDs")
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max(1, self.max_length - len(label_ids)),
            )
            ids = (prompt_ids + label_ids)[-self.max_length :]
            prompt_count = len(ids) - len(label_ids)
            input_ids.append(ids)
            labels.append([-100] * prompt_count + label_ids)

        width = max(len(values) for values in input_ids)
        pad = int(self.tokenizer.pad_token_id)
        padded_ids: list[list[int]] = []
        padded_labels: list[list[int]] = []
        masks: list[list[int]] = []
        for ids, target in zip(input_ids, labels):
            padding = width - len(ids)
            padded_ids.append(ids + [pad] * padding)
            padded_labels.append(target + [-100] * padding)
            masks.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def make_dataloader(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        SentimentDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TrainingCollator(tokenizer, label_names, max_length),
    )


def load_model(config: Mapping[str, Any], device: torch.device) -> tuple[torch.nn.Module, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before loading the model") from exc

    seed = int(get_path(config, "experiment.seed"))
    seed_everything(seed)
    model_key = str(get_path(config, "model.key"))
    model_name = str(config["model"]["names"][model_key])
    cache_dir = str(get_path(config, "experiment.model_cache"))
    dtype_name = str(get_path(config, "model.dtype")).lower()
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    lora = config["model"]["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if bool(get_path(config, "model.gradient_checkpointing")):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Loaded %s: trainable=%d total=%d (%.4f%%)", model_name, trainable, total, 100 * trainable / total)
    return model, tokenizer


@torch.inference_mode()
def evaluate_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    candidate_ids: list[int] = []
    for name in label_names:
        values = tokenizer.encode(" " + str(name), add_special_tokens=False)
        if not values:
            raise RuntimeError(f"Label {name!r} tokenized to an empty sequence")
        candidate_ids.append(int(values[0]))
        if len(values) > 1:
            LOGGER.warning("Label %r has %d tokens; evaluation uses the discriminating first token", name, len(values))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Label names do not have distinct first token IDs")

    correct = 0
    total = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [build_prompt(str(row["text"]), label_names) for row in chunk]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
        output = model(**encoded)
        last = encoded["attention_mask"].sum(dim=1) - 1
        logits = output.logits[torch.arange(len(chunk), device=device), last]
        candidates = logits[:, torch.tensor(candidate_ids, device=device)]
        predicted = candidates.argmax(dim=1).cpu().tolist()
        truth = [int(row["label"]) for row in chunk]
        correct += sum(int(left == right) for left, right in zip(predicted, truth))
        total += len(chunk)
    return correct / max(1, total)

