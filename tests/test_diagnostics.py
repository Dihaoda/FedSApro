import pytest

from fedsapro.diagnostics import contribution_metrics, domain_gap_points


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


def test_domain_gap_is_an_absolute_disparity() -> None:
    assert domain_gap_points(0.68, 0.78) == pytest.approx(10.0)
    assert domain_gap_points(0.91, 0.82) == pytest.approx(9.0)
