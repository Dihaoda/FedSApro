from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ParameterLayout:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    numels: tuple[int, ...]
    qv_slices: tuple[tuple[int, int], ...]

    @property
    def total_numel(self) -> int:
        return sum(self.numels)


def trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("The model has no trainable parameters")
    return parameters


def build_layout(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> ParameterLayout:
    names: list[str] = []
    shapes: list[torch.Size] = []
    numels: list[int] = []
    qv_slices: list[tuple[int, int]] = []
    offset = 0
    for name, parameter in named_parameters:
        count = parameter.numel()
        names.append(name)
        shapes.append(parameter.shape)
        numels.append(count)
        lowered = name.lower()
        if "lora_" in lowered and ("q_proj" in lowered or "v_proj" in lowered):
            qv_slices.append((offset, count))
        offset += count
    if not qv_slices:
        raise RuntimeError("No query/value LoRA parameters found in the trainable parameter set")
    return ParameterLayout(tuple(names), tuple(shapes), tuple(numels), tuple(qv_slices))


def pack_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().to(device=device, dtype=dtype).reshape(-1) for _, parameter in named_parameters]
    ).contiguous()


def unpack_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]], vector: torch.Tensor
) -> None:
    expected = sum(parameter.numel() for _, parameter in named_parameters)
    if vector.ndim != 1 or vector.numel() != expected:
        raise ValueError(f"Parameter vector has shape {tuple(vector.shape)}, expected ({expected},)")
    offset = 0
    with torch.no_grad():
        for _, parameter in named_parameters:
            count = parameter.numel()
            value = vector[offset : offset + count].reshape(parameter.shape)
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            offset += count


def add_gradient_vector(
    named_parameters: list[tuple[str, torch.nn.Parameter]], correction: torch.Tensor
) -> None:
    expected = sum(parameter.numel() for _, parameter in named_parameters)
    if correction.ndim != 1 or correction.numel() != expected:
        raise ValueError("Gradient correction vector has an invalid shape")
    offset = 0
    for _, parameter in named_parameters:
        count = parameter.numel()
        if parameter.grad is not None:
            parameter.grad.add_(
                correction[offset : offset + count]
                .reshape(parameter.shape)
                .to(parameter.grad.device, parameter.grad.dtype)
            )
        offset += count


def proximal_penalty(
    named_parameters: list[tuple[str, torch.nn.Parameter]], reference: torch.Tensor
) -> torch.Tensor:
    expected = sum(parameter.numel() for _, parameter in named_parameters)
    if reference.ndim != 1 or reference.numel() != expected:
        raise ValueError("Proximal reference vector has an invalid shape")
    penalty: torch.Tensor | None = None
    offset = 0
    for _, parameter in named_parameters:
        count = parameter.numel()
        target = reference[offset : offset + count].reshape(parameter.shape).to(
            device=parameter.device, dtype=parameter.dtype
        )
        value = (parameter - target).float().square().sum()
        penalty = value if penalty is None else penalty + value
        offset += count
    assert penalty is not None
    return penalty


def gather_qv(vector: torch.Tensor, layout: ParameterLayout) -> torch.Tensor:
    return torch.cat([vector[offset : offset + count] for offset, count in layout.qv_slices])

