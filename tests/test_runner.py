import pytest
import torch

from fedsapro.aggregators import ClientUpdate
from fedsapro.runner import _normalize_attacker_norms


def test_normalize_attacker_norms_uses_benign_median() -> None:
    updates = [
        ClientUpdate(0, "standard", torch.tensor([3.0, 4.0]), 1),
        ClientUpdate(1, "heterogeneous", torch.tensor([0.0, 7.0]), 1),
        ClientUpdate(2, "standard", torch.tensor([0.0, 9.0]), 1),
        ClientUpdate(3, "attacker", torch.tensor([6.0, 8.0]), 1),
    ]

    normalized, benign_median = _normalize_attacker_norms(updates, 2.0)

    assert benign_median == pytest.approx(7.0)
    assert torch.linalg.vector_norm(normalized[-1].delta).item() == pytest.approx(14.0)
    assert torch.nn.functional.cosine_similarity(
        normalized[-1].delta, updates[-1].delta, dim=0
    ).item() == pytest.approx(1.0)
