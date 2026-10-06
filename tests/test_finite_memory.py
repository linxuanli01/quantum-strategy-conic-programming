import numpy as np
import pytest

from quantum_strategy_conic.finite_memory import (
    C_labels,
    ancilla_qubits_to_dM,
    initialize_choi_teeth_with_memory,
    make_dims_by_label,
    partial_trace_over_labels_numeric,
    resolve_memory_dimension,
    solve_X_sdp,
)


def test_memory_dimension_helpers():
    assert ancilla_qubits_to_dM(1) == 2
    assert ancilla_qubits_to_dM(2) == 4
    assert resolve_memory_dimension(dM=2, num_ancilla_qubits=1) == 2
    with pytest.raises(ValueError, match="Inconsistent"):
        resolve_memory_dimension(dM=4, num_ancilla_qubits=1)


def test_initialized_teeth_are_normalized():
    dims = (2, 2, 2, 2)
    dM = 2
    controls = initialize_choi_teeth_with_memory(dims, dM)
    assert len(controls) == 2
    np.testing.assert_allclose(np.trace(controls[0]), 1.0, atol=1e-10)

    dims_by_label = make_dims_by_label(2, dims, dM)
    labels = C_labels(2)
    reduced = partial_trace_over_labels_numeric(
        controls[1],
        labels,
        labels[2:],
        dims_by_label,
    )
    np.testing.assert_allclose(reduced, np.eye(4), atol=1e-10)


def test_fixed_state_holevo_problem_matches_single_parameter_limit():
    state = np.eye(2) / 2
    derivative = np.diag([0.5, -0.5])
    solution = solve_X_sdp(
        state,
        [derivative],
        np.eye(1),
        solver="CLARABEL",
        x_bound="holevo",
    )
    assert solution.status == "optimal"
    assert solution.value == pytest.approx(1.0, abs=1e-6)
