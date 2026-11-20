import torch

from fedsapro.aggregators import (
    ClientUpdate,
    fedgraph,
    fedavg,
    fltrust,
    index_overlap_similarity,
    scaffold,
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


def test_fltrust_diagnostic_coefficient_includes_norm_matching() -> None:
    updates = [
        ClientUpdate(0, "standard", torch.tensor([2.0, 0.0]), 1),
        ClientUpdate(1, "heterogeneous", torch.tensor([0.5, 0.5]), 1),
    ]
    result = fltrust(updates, torch.tensor([1.0, 0.0]))

    # Both updates are positively aligned, but their equivalent coefficients on
    # the original vectors also include FLTrust's root-norm matching factor.
    assert result.effective_coefficients[0] == 0.5
    assert torch.isclose(
        torch.tensor(result.effective_coefficients[1]), torch.tensor(1.0)
    )


def test_scaffold_server_lr_scales_the_aggregated_model_delta() -> None:
    updates = [
        ClientUpdate(0, "standard", torch.tensor([1.0, 3.0]), 1),
        ClientUpdate(1, "heterogeneous", torch.tensor([3.0, 5.0]), 3),
    ]
    result = scaffold(updates, server_lr=0.5)
    assert torch.allclose(result.delta, torch.tensor([1.25, 2.25]))
    assert result.diagnostics["server_lr"] == 0.5


def test_fedgraph_keeps_all_updates_when_graph_has_no_clear_split() -> None:
    updates = [
        ClientUpdate(index, "standard", torch.tensor([1.0, 0.01 * index]), 1)
        for index in range(5)
    ]
    result = fedgraph(
        updates,
        {
            "similarity_threshold": 1.1,
            "min_neighbors": 0,
            "min_cluster_size": 3,
            "min_samples": 1,
            "allow_single_cluster": True,
        },
    )
    assert all(result.selected.values())
    assert result.diagnostics["outlier_count"] == 0

