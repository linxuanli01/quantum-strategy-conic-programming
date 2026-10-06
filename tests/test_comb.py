import numpy as np

from quantum_strategy_conic.comb import comb, comb_step, np_partial_trace


def test_partial_trace_of_bell_state_is_maximally_mixed():
    bell = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    state = np.outer(bell, bell.conj())
    np.testing.assert_allclose(np_partial_trace(state, [2, 2], 0), np.eye(2) / 2)


def test_comb_step_uses_row_vectorization():
    operator = np.array([[1.0, 2.0], [3.0, 4.0]])
    vectors, derivatives = comb_step([operator], [2.0 * operator], 2, 2)
    np.testing.assert_array_equal(vectors[0].ravel(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(derivatives[0], 2.0 * vectors[0])


def test_two_use_comb_derivative_obeys_product_rule():
    operator = np.eye(2)
    derivative = np.diag([1.0, -1.0])
    vectors, derivatives = comb([operator], [derivative], [2, 2], n_steps=2)
    vector = operator.reshape(-1, 1)
    derivative_vector = derivative.reshape(-1, 1)
    np.testing.assert_array_equal(vectors[0], np.kron(vector, vector))
    np.testing.assert_array_equal(
        derivatives[0],
        np.kron(vector, derivative_vector) + np.kron(derivative_vector, vector),
    )
