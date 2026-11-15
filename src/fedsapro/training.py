from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .config import get_path
from .modeling import make_dataloader
from .state import (
    pack_parameters,
    trainable_named_parameters,
    unpack_parameters,
)


@dataclass
class LocalTrainingResult:
    delta: torch.Tensor
    steps: int
    mean_loss: float
    peak_memory_mb: float
    new_control: torch.Tensor | None = None
    control_delta: torch.Tensor | None = None


def train_local(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    device: torch.device,
    global_vector: torch.Tensor,
    seed: int,
    method: str,
    client_control: torch.Tensor | None = None,
    server_control: torch.Tensor | None = None,
    epochs_override: int | None = None,
) -> LocalTrainingResult:
    named = trainable_named_parameters(model)
    unpack_parameters(named, global_vector)
    model.train()
    learning_rate = float(get_path(config, "federated.learning_rate"))
    epochs = int(epochs_override or get_path(config, "federated.local_epochs"))
    effective_batch_size = int(get_path(config, "federated.batch_size"))
    micro_batch_size = int(get_path(config, "federated.micro_batch_size"))
    accumulation_steps = effective_batch_size // micro_batch_size
    dataloader = make_dataloader(
        rows,
        tokenizer=tokenizer,
        label_names=list(get_path(config, "data.labels")),
        max_length=int(get_path(config, "federated.max_length")),
        batch_size=micro_batch_size,
        seed=seed,
        shuffle=True,
    )
    parameters = [parameter for _, parameter in named]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(get_path(config, "federated.weight_decay")),
    )
    if method == "scaffold":
        if client_control is None or server_control is None:
            raise ValueError("SCAFFOLD requires client and server control vectors")
        control_scale = float(config["method"].get("scaffold_control_scale", 1.0))
        correction = (server_control - client_control) * control_scale
        correction_tensors: list[torch.Tensor] = []
        correction_offset = 0
        for _, parameter in named:
            count = parameter.numel()
            correction_tensors.append(
                correction[correction_offset : correction_offset + count]
                .reshape(parameter.shape)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            correction_offset += count
    else:
        correction = None
        correction_tensors = []

    if method == "fedprox":
        proximal_references: list[torch.Tensor] = []
        reference_offset = 0
        for _, parameter in named:
            count = parameter.numel()
            proximal_references.append(
                global_vector[reference_offset : reference_offset + count]
                .reshape(parameter.shape)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            reference_offset += count
    else:
        proximal_references = []

    use_amp = device.type == "cuda" and str(get_path(config, "model.dtype")).lower() != "float32"
    amp_dtype = torch.bfloat16 if str(get_path(config, "model.dtype")).lower() in {"bf16", "bfloat16"} else torch.float16
    max_grad_norm = float(get_path(config, "federated.max_grad_norm"))
    mu = float(get_path(config, "method.fedprox_mu"))
    losses: list[float] = []
    step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        micro_batches = len(dataloader)
        for micro_index, batch in enumerate(dataloader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(**batch)
                loss = output.loss
                if method == "fedprox":
                    penalty = sum(
                        (parameter - reference).float().square().sum()
                        for (_, parameter), reference in zip(named, proximal_references)
                    )
                    loss = loss + 0.5 * mu * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite local loss at micro-batch {micro_index + 1}/{micro_batches}"
                )
            group_start = (micro_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, micro_batches - group_start)
            (loss / group_size).backward()
            losses.append(float(loss.detach().float().item()))
            should_step = (micro_index + 1) % accumulation_steps == 0 or micro_index + 1 == micro_batches
            if should_step:
                if correction is not None:
                    for (_, parameter), value in zip(named, correction_tensors):
                        if parameter.grad is not None:
                            parameter.grad.add_(value.to(parameter.grad.dtype))
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                if not torch.isfinite(grad_norm):
                    nonfinite = [
                        name
                        for name, parameter in named
                        if parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ]
                    raise FloatingPointError(
                        f"Non-finite gradient norm at optimizer step {step + 1}; "
                        f"parameters={nonfinite[:8]} total={len(nonfinite)}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

    local_vector = pack_parameters(named, dtype=torch.float32, device="cpu")
    delta = local_vector - global_vector.float().cpu()
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Local update contains non-finite values")
    if method == "scaffold":
        assert client_control is not None and server_control is not None
        denominator = max(1, step) * learning_rate
        new_control = client_control - server_control - delta / denominator
        control_delta = new_control - client_control
    else:
        new_control = None
        control_delta = None
    return LocalTrainingResult(
        delta=delta,
        steps=step,
        mean_loss=sum(losses) / max(1, len(losses)),
        peak_memory_mb=(
            float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        ),
        new_control=new_control,
        control_delta=control_delta,
    )


