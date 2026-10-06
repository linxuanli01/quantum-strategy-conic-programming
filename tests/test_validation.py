import numpy as np
import pytest

from quantum_strategy_conic import multipara_bounds, solve_quantum_bound


def test_rejects_unsupported_dimensions_before_constructing_program():
    with pytest.raises(ValueError, match="dims"):
        solve_quantum_bound(
            G=np.eye(1),
            T=np.eye(16),
            DT=[np.eye(16)],
            dims=(3, 3, 3, 3),
        )


def test_rejects_unknown_bound():
    with pytest.raises(ValueError, match="bound_type"):
        solve_quantum_bound(
            G=np.eye(1),
            T=np.eye(16),
            DT=[np.eye(16)],
            dims=(2, 2, 2, 2),
            bound_type="unknown",
        )


def test_rejects_energy_constraint_for_general_ico():
    with pytest.raises(ValueError, match="ICO"):
        solve_quantum_bound(
            G=np.eye(1),
            T=np.eye(16),
            DT=[np.eye(16)],
            dims=(2, 2, 2, 2),
            strategy="ico",
            energy_bound=1.0,
        )


@pytest.mark.parametrize("strategy", ["seq", "par", "sup", "swi"])
def test_energy_constrained_program_is_dcp(monkeypatch, strategy):
    def check_without_solving(problem, _mosek_params):
        assert problem.is_dcp()
        return np.nan

    monkeypatch.setattr(multipara_bounds, "_solve_problem", check_without_solving)
    result = solve_quantum_bound(
        G=np.eye(1),
        T=np.eye(16),
        DT=[np.eye(16)],
        dims=(2, 2, 2, 2),
        bound_type="SLD",
        strategy=strategy,
        energy_bound=0.5,
    )
    assert np.isnan(result[0])
