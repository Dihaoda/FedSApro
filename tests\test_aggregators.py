import torch

from fedsapro.aggregators import (
    ClientUpdate,
    fedavg,
    index_overlap_similarity,
    topk_indices,
)


def test_fedavg_uses_sample_counts() -> None:
    updates = [
        ClientUpdate(0, "standard", torch.tensor([1.0, 3.0]), 1),
        ClientUpdate(1, "heterogeneous", torch.tensor([3.0, 5.0]), 3),
    ]
    result = fedavg(updates)
    assert torch.allclose(result.delta, torch.tensor([2.5, 4.5]))
    assert result.effective_coefficients == {0: 1.0, 1: 3.0}


def test_topk_and_ios_are_rescaling_invariant() -> None:
    left = torch.tensor([0.1, 4.0, -3.0, 0.2])
    right = left * 17
    other = torch.tensor([9.0, 0.1, 0.2, 8.0])
    signatures = [topk_indices(left, 0.5), topk_indices(right, 0.5), topk_indices(other, 0.5)]
    similarity = index_overlap_similarity(signatures)
    assert similarity[0, 1].item() == 1.0
    assert similarity[0, 2].item() == 0.0

