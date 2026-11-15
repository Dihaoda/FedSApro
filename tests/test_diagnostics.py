import pytest

from fedsapro.diagnostics import contribution_metrics


def test_contribution_metrics_match_definitions() -> None:
    rows = []
    for round_number in (1, 2):
        rows.extend(
            [
                {"round": round_number, "role": "standard", "effective_coefficient": 1.0},
                {"round": round_number, "role": "standard", "effective_coefficient": 1.0},
                {"round": round_number, "role": "heterogeneous", "effective_coefficient": 0.1},
                {"round": round_number, "role": "heterogeneous", "effective_coefficient": 0.5},
            ]
        )
    metrics = contribution_metrics(rows, tau=0.25)
    assert metrics["hrcw"] == pytest.approx(0.3)
    assert metrics["hssr"] == pytest.approx(0.5)
