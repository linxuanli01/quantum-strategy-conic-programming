"""Utilities for quantum-comb construction and subsystem operations."""

import cvxpy as cp
import numpy as np
from cvxpy.expressions.expression import Expression


def expr_as_np_array(cvx_expr):
    """Convert a scalar, vector, or matrix CVXPY expression to an object array."""
    if cvx_expr.is_scalar():  # cvxpy scalar as a 0d array
        return np.array(cvx_expr)
    elif len(cvx_expr.shape) == 1:  # cvx_expr is a 1d array
        return np.array([v for v in cvx_expr])
    else:
        rows = []
        for i in range(cvx_expr.shape[0]):
            row = [cvx_expr[i, j] for j in range(cvx_expr.shape[1])]
            rows.append(row)
        arr = np.array(rows)
        return arr


def np_array_as_expr(np_arr):
    """Convert a NumPy object array containing CVXPY atoms to an expression."""
    return cp.bmat(np_arr.tolist())


def np_partial_trace(rho, dims, axis):
    """Take the partial trace of a square NumPy matrix.

    ``dims`` lists subsystem dimensions in tensor-product order. ``axis`` may
    be one subsystem index or a list of subsystem indices.
    """
    dims_ = np.array(dims)

    if dims_.size == 1:
        return np.trace(rho)

    # reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # each subsystem gets one index for its row and another one for its column
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))

    # if axis is an integer, trace over a single subsystem
    if isinstance(axis, int):
        traced_out_rho = np.trace(reshaped_rho, axis1=axis, axis2=dims_.size + axis)

        dims_untraced = np.delete(dims_, axis)
        rho_dim = np.prod(dims_untraced)
        return traced_out_rho.reshape([rho_dim, rho_dim])

    # if axis is a list, trace over multiple subsystems
    elif isinstance(axis, (list, tuple)):
        axis = sorted(axis)
        dims_untraced = dims_
        traced_out_rho = reshaped_rho
        for i in range(len(axis)):
            traced_out_rho = np.trace(
                traced_out_rho,
                axis1=axis[i] - i,
                axis2=dims_untraced.size + axis[i] - i,
            )
            dims_untraced = np.delete(dims_untraced, axis[i] - i)
        rho_dim = np.prod(dims_untraced)
        return traced_out_rho.reshape([rho_dim, rho_dim])
    raise TypeError("axis must be an integer or a list/tuple of integers")


def partial_trace(rho, dims, axis):
    """Take the partial trace of a NumPy matrix or CVXPY expression."""
    if not isinstance(rho, Expression):
        rho = cp.Constant(rho)
    rho_np = expr_as_np_array(rho)
    traced_rho = np_partial_trace(rho_np, dims, axis)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho


def np_move_subsystem(rho, dims_in, dims_out, axis_old, axis_new):
    """Move tensor-factor axes in a rectangular NumPy operator."""
    dims_in_ = np.array(dims_in)
    dims_out_ = np.array(dims_out)
    axis_old_ = np.array(axis_old)
    axis_new_ = np.array(axis_new)
    reshaped_rho = np.reshape(rho, np.concatenate((dims_in_, dims_out_), axis=None))

    reshaped_rho = np.moveaxis(
        reshaped_rho,
        np.concatenate((axis_old_, len(dims_in) + axis_old_), axis=None),
        np.concatenate((axis_new_, len(dims_in) + axis_new_), axis=None),
    )
    return reshaped_rho.reshape((np.prod(dims_in), np.prod(dims_out)))


def move_subsystem(rho, dims_in, dims_out, axis_old, axis_new):
    """Move tensor-factor axes in a NumPy matrix or CVXPY expression."""
    if not isinstance(rho, Expression):
        rho = cp.Constant(rho)
    rho_np = expr_as_np_array(rho)
    reshaped_rho = np_move_subsystem(rho_np, dims_in, dims_out, axis_old, axis_new)
    return np_array_as_expr(reshaped_rho)


def number_to_base(n, base):
    """Return the digits of a nonnegative integer in the requested base."""
    if n < 0 or base < 2:
        raise ValueError("n must be nonnegative and base must be at least 2")
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % base))
        n //= base
    return digits[::-1]


def basis_state(n, i):
    """Return the ``i``-th computational basis ket in dimension ``n``."""
    if not 0 <= i < n:
        raise ValueError("basis-state index must satisfy 0 <= i < n")
    return np.eye(n, dtype=complex)[:, [i]]


def K_phis_AD(phi, t, p):
    """Kraus operators for phase encoding followed by amplitude damping."""
    U = np.array([[np.exp(-1j * (phi * t) / 2), 0], [0, np.exp(1j * (phi * t) / 2)]])
    return [
        np.array([[1, 0], [0, np.sqrt(1 - p)]]) @ U,
        np.array([[0, np.sqrt(p)], [0, 0]]) @ U,
    ]


def dK_phis_AD(phi, t, p):
    """Derivatives of :func:`K_phis_AD` with respect to ``phi``."""
    dU = np.array(
        [
            [(-1j * t / 2) * np.exp(-1j * (phi * t) / 2), 0],
            [0, (1j * t / 2) * np.exp(1j * (phi * t) / 2)],
        ]
    )
    return [
        np.array([[1, 0], [0, np.sqrt(1 - p)]]) @ dU,
        np.array([[0, np.sqrt(p)], [0, 0]]) @ dU,
    ]


def K_phis_SWAP(phi, t, tau, g):
    """Kraus operators for phase encoding followed by SWAP-type noise."""
    U = np.array([[np.exp(-1j * (phi * t) / 2), 0], [0, np.exp(1j * (phi * t) / 2)]])
    return [
        np.array([[np.exp(-1j * g * tau), 0], [0, np.cos(g * tau)]]) @ U,
        np.array([[0, -1j * np.sin(g * tau)], [0, 0]]) @ U,
    ]


def dK_phis_SWAP(phi, t, tau, g):
    """Derivatives of :func:`K_phis_SWAP` with respect to ``phi``."""
    dU = np.array(
        [
            [(-1j * t / 2) * np.exp(-1j * (phi * t) / 2), 0],
            [0, (1j * t / 2) * np.exp(1j * (phi * t) / 2)],
        ]
    )
    return [
        np.array([[np.exp(-1j * g * tau), 0], [0, np.cos(g * tau)]]) @ dU,
        np.array([[0, -1j * np.sin(g * tau)], [0, 0]]) @ dU,
    ]


def K_phis_DP(phi, t, p):
    """Kraus operators for phase encoding followed by bit-flip dephasing."""
    U = np.array([[np.exp(-1j * (phi * t) / 2), 0], [0, np.exp(1j * (phi * t) / 2)]])
    return [
        np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]]) @ U,
        np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]]) @ U,
    ]


def dK_phis_DP(phi, t, p):
    """Derivatives of :func:`K_phis_DP` with respect to ``phi``."""
    dU = np.array(
        [
            [(-1j * t / 2) * np.exp(-1j * (phi * t) / 2), 0],
            [0, (1j * t / 2) * np.exp(1j * (phi * t) / 2)],
        ]
    )
    return [
        np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]]) @ dU,
        np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]]) @ dU,
    ]


def comb_step(kraus_operators, derivatives, d_out, d_in):
    """Vectorize one channel use and its parameter derivatives.

    NumPy's row-major vectorization convention is used throughout the code.
    """
    if len(kraus_operators) != len(derivatives):
        raise ValueError("kraus_operators and derivatives must have equal length")
    expected_shape = (d_out, d_in)
    operators = list(kraus_operators) + list(derivatives)
    if any(operator.shape != expected_shape for operator in operators):
        raise ValueError(f"every operator must have shape {expected_shape}")
    vectors = [operator.reshape(d_out * d_in, 1) for operator in kraus_operators]
    derivative_vectors = [operator.reshape(d_out * d_in, 1) for operator in derivatives]
    return vectors, derivative_vectors


def comb(kraus_operators, derivatives, dims, n_steps):
    """Construct an ``n_steps``-use comb ensemble and its derivative."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if len(dims) < 2:
        raise ValueError("dims must contain at least output and input dimensions")
    d_out = dims[0]
    d_in = dims[1]
    vectors, derivative_vectors = comb_step(kraus_operators, derivatives, d_out, d_in)

    def comb_vector(items):
        if len(items) > 1:
            return np.kron(vectors[items[0]], comb_vector(items[1:]))
        return vectors[items[0]]

    def derivative_comb_vector(items):
        if len(items) > 1:
            return np.kron(vectors[items[0]], derivative_comb_vector(items[1:])) + np.kron(
                derivative_vectors[items[0]], comb_vector(items[1:])
            )
        return derivative_vectors[items[0]]

    comb_vectors = []
    derivative_comb_vectors = []
    r = len(vectors) ** n_steps
    for i in range(r):
        digits = [0] if len(vectors) == 1 else number_to_base(i, len(vectors))
        item = [0] * (n_steps - len(digits)) + digits
        comb_vectors.append(comb_vector(item))
        derivative_comb_vectors.append(derivative_comb_vector(item))
    return comb_vectors, derivative_comb_vectors


# Backward-compatible aliases used by the original research scripts.
numberToBase = number_to_base
Comb_step = comb_step
Comb = comb
