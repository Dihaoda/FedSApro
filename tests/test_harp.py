from __future__ import annotations

import numpy as np

from fedsapro.harp import _supported_mode_preservation


def test_supported_mode_preservation_rewards_coherent_secondary_mode() -> None:
    rng = np.random.default_rng(7)
    dominant = np.tile([1.0, 0.0, 0.0], (12, 1)) + rng.normal(0, 0.01, (12, 3))
    secondary = np.tile([0.0, 1.0, 0.0], (6, 1)) + rng.normal(0, 0.01, (6, 3))
    isolated = np.asarray([[0.0, 0.0, 1.0]])
    matrix = np.concatenate((dominant, secondary, isolated), axis=0)
    factors = np.ones(len(matrix))

    labels, preservation, details = _supported_mode_preservation(
        matrix,
        factors,
        {
            "min_mode_size": 4,
            "preservation_modes": 3,
            "preservation_gain": 2.0,
            "selection_threshold": 0.25,
        },
    )

    assert set(preservation[:12]) == {1.0}
    assert set(preservation[12:18]) == {3.0}
    assert preservation[18] == 1.0
    assert details["preserved_mode"] is not None
    assert labels[18] != details["preserved_mode"]


def test_supported_mode_preservation_never_rewards_risky_clients() -> None:
    matrix = np.concatenate(
        (
            np.tile([1.0, 0.0], (8, 1)),
            np.tile([0.0, 1.0], (4, 1)),
            np.tile([-1.0, 0.0], (4, 1)),
        ),
        axis=0,
    )
    factors = np.ones(len(matrix))
    factors[-4:] = 0.01

    _, preservation, _ = _supported_mode_preservation(
        matrix,
        factors,
        {
            "min_mode_size": 4,
            "preservation_modes": 2,
            "preservation_gain": 2.0,
            "selection_threshold": 0.25,
        },
    )

    assert set(preservation[-4:]) == {1.0}

