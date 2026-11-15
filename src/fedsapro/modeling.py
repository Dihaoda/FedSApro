from __future__ import annotations

import gc
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
class SequenceClassificationCollator:
    tokenizer: Any
    label_names: Sequence[str]
    max_length: int

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [build_prompt(str(row["text"]), self.label_names) for row in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [int(row["label"]) for row in batch], dtype=torch.long
        )
        return dict(encoded)


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
        collate_fn=SequenceClassificationCollator(tokenizer, label_names, max_length),
    )


def _pretrained_label_vectors(
    model_name: str,
    tokenizer: Any,
    label_names: Sequence[str],
    *,
    dtype: torch.dtype,
    cache_dir: str,
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    candidate_ids: list[int] = []
    for name in label_names:
        values = tokenizer.encode(" " + str(name), add_special_tokens=False)
        if not values:
            raise RuntimeError(f"Label {name!r} tokenized to an empty sequence")
        candidate_ids.append(int(values[0]))
        if len(values) > 1:
            LOGGER.warning(
                "Label %r has %d tokens; classifier initialization uses the first token",
                name,
                len(values),
            )
    causal_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    output_embeddings = causal_model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("The causal language model has no output embeddings")
    vectors = output_embeddings.weight[candidate_ids].detach().float().cpu().clone()
    del output_embeddings, causal_model
    gc.collect()
    return vectors


def load_model(config: Mapping[str, Any], device: torch.device) -> tuple[torch.nn.Module, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    tokenizer.truncation_side = "left"
    label_names = list(get_path(config, "data.labels"))
    classifier_initialization = str(
        config["model"].get("classifier_initialization", "random")
    )
    if classifier_initialization == "label_token_lm_head":
        label_vectors = _pretrained_label_vectors(
            model_name,
            tokenizer,
            label_names,
            dtype=dtype,
            cache_dir=cache_dir,
        )
    elif classifier_initialization == "random":
        label_vectors = None
    else:
        raise ValueError(
            f"Unsupported classifier initialization: {classifier_initialization}"
        )
    id2label = {index: str(name) for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        dtype=dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    model.config.use_cache = False
    if label_vectors is not None:
        with torch.no_grad():
            model.score.weight.copy_(label_vectors.to(dtype=model.score.weight.dtype))
    lora = config["model"]["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # Keep trainable parameters as FP32 master weights.  In particular, PEFT's
    # sequence-classification ``score`` module otherwise inherits the BF16
    # backbone dtype; its small randomly initialized weight can overflow during
    # the first local epoch even though the LoRA matrices themselves are FP32.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
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
    if len(label_names) != int(model.config.num_labels):
        raise ValueError("Evaluation labels do not match the model classification head")

    correct = 0
    total = 0
    mixed_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.dtype in {torch.bfloat16, torch.float16}
    }
    amp_dtype = next(iter(mixed_dtypes), torch.bfloat16)
    use_amp = device.type == "cuda" and bool(mixed_dtypes)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = tokenizer(
            [build_prompt(str(row["text"]), label_names) for row in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(**encoded)
        predicted = output.logits.argmax(dim=1).cpu().tolist()
        truth = [int(row["label"]) for row in chunk]
        correct += sum(int(left == right) for left, right in zip(predicted, truth))
        total += len(chunk)
    return correct / max(1, total)

