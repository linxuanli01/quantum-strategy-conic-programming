"""Conic programs for multiparameter quantum-estimation bounds.

The current implementation covers two uses of qubit channels. It supports
sequential, parallel, causal-superposition, quantum-switch, and general
indefinite-causal-order strategy classes, with optional global-battery energy
constraints for the first four classes.
"""

import cvxpy as cp
import numpy as np

from ._bound_constraints import (
    add_holevo_constraints as _add_holevo_constraints,
)
from ._bound_constraints import (
    add_nhb_constraints as _add_nhb_constraints,
)
from ._bound_constraints import (
    add_sld_constraints as _add_sld_constraints,
)
from ._solver import solve_problem
from .comb import move_subsystem, partial_trace

_BOUND_TYPES = {"NHB", "SLD", "Holevo"}
_STRATEGIES = {"seq", "par", "sup", "swi", "ico"}


def _validate_inputs(G, T, DT, dims, bound_type, strategy):
    """Validate the dimensions assumed by the published two-use programs."""
    if tuple(dims) != (2, 2, 2, 2):
        raise ValueError("the current implementation requires dims=(2, 2, 2, 2)")
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square weight matrix")
    if T.shape != (16, 16):
        raise ValueError("T must have shape (16, 16) for two qubit-channel uses")
    if len(DT) != G.shape[0] or any(derivative.shape != T.shape for derivative in DT):
        raise ValueError("DT must contain one derivative with T's shape per parameter")
    if bound_type not in _BOUND_TYPES:
        raise ValueError(f"bound_type must be one of {sorted(_BOUND_TYPES)}")
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_STRATEGIES)}")


def _validate_energy_settings(energy_bound, energy_hamiltonian, strategy):
    """Validate and normalize the optional global-battery constraint."""
    if energy_bound is None:
        if energy_hamiltonian is not None:
            raise ValueError("energy_hamiltonian requires energy_bound")
        return None, None
    if strategy == "ico":
        raise ValueError(
            "an energy constraint is not implemented for the general ICO class; "
            "use 'seq', 'par', 'sup', or 'swi'"
        )

    energy_bound = float(energy_bound)
    if not np.isfinite(energy_bound) or energy_bound < 0:
        raise ValueError("energy_bound must be a finite nonnegative number")

    if energy_hamiltonian is None:
        energy_hamiltonian = np.diag([0.0, 1.0])
    energy_hamiltonian = np.asarray(energy_hamiltonian, dtype=complex)
    if energy_hamiltonian.shape != (2, 2):
        raise ValueError("energy_hamiltonian must have shape (2, 2)")
    if not np.allclose(energy_hamiltonian, energy_hamiltonian.conj().T):
        raise ValueError("energy_hamiltonian must be Hermitian")
    return energy_bound, energy_hamiltonian


def _embedded_single_site_operator(operator, axis, n_subsystems=3):
    """Embed a qubit operator on one axis of a tensor product."""
    factors = [operator if index == axis else np.eye(2) for index in range(n_subsystems)]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _add_two_use_sequential_energy_constraints(
    strategy_operator,
    initial_state,
    input_axes,
    energy_hamiltonian,
    energy_bound,
    constraints,
):
    """Add the two cumulative global-battery constraints for one branch."""
    branch_weight = cp.real(cp.trace(initial_state))
    initial_energy = cp.real(cp.trace(energy_hamiltonian @ initial_state))
    constraints.append(initial_energy <= energy_bound * branch_weight)

    total_input_hamiltonian = sum(
        _embedded_single_site_operator(energy_hamiltonian, axis) for axis in input_axes
    )
    cumulative_energy = (
        partial_trace(
            strategy_operator @ total_input_hamiltonian,
            [2, 2, 2],
            list(input_axes),
        )
        - branch_weight * energy_hamiltonian.T
    )
    cumulative_energy = 0.5 * (cumulative_energy + cumulative_energy.H)
    constraints.append(cumulative_energy << energy_bound * branch_weight * np.eye(2))


def _solve_problem(problem, mosek_params):
    """Use MOSEK when available and explicitly report an SCS fallback."""
    solve_problem(
        problem,
        solver="MOSEK",
        mosek_params=mosek_params,
        fallback_solvers=("SCS",),
        warn_on_fallback=True,
    )
    return problem.value


def solve_quantum_bound(
    G,
    T,
    DT,
    dims,
    bound_type="NHB",
    strategy="seq",
    mosek_params=None,
    energy_bound=None,
    energy_hamiltonian=None,
):
    """
    Solve quantum estimation bounds for different strategies and bound types.

    Parameters:
        G: Positive weight matrix, with one row and column per parameter.
        T: Choi operator of two uses of the parameterized qubit channel.
        DT: Derivatives of ``T``, one per estimated parameter.
        dims: System dimensions ``(d4, d3, d2, d1)``. The current release
            supports ``(2, 2, 2, 2)``.
        bound_type: ``"NHB"``, ``"SLD"``, or ``"Holevo"``.
        strategy: ``"seq"``, ``"par"``, ``"sup"``, ``"swi"``, or ``"ico"``.
        mosek_params: Optional parameters passed to CVXPY's MOSEK interface.
        energy_bound: Optional nonnegative global-battery budget. When given,
            the strategy is constrained at every preparation/control step.
        energy_hamiltonian: Optional single-qubit Hamiltonian. It defaults to
            ``|1><1|`` when ``energy_bound`` is specified.

    Returns:
        A tuple containing the optimal value and strategy-specific primal
        variables. See the project README for the return values.
    """
    if mosek_params is None:
        mosek_params = {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1.0e-7,
        }

    G = np.asarray(G)
    T = np.asarray(T)
    DT = [np.asarray(derivative) for derivative in DT]
    _validate_inputs(G, T, DT, dims, bound_type, strategy)
    energy_bound, energy_hamiltonian = _validate_energy_settings(
        energy_bound, energy_hamiltonian, strategy
    )

    d_sys = np.prod(dims)
    d = G.shape[0]

    # Common setup for all strategies
    dY = (d + 1) * d_sys
    Y_real = cp.Variable((dY, dY), symmetric=True)
    Y_imag = cp.Variable((dY, dY))
    Y = Y_real + 1j * Y_imag

    GT = np.kron(np.block([[np.zeros((1, d + 1))], [np.zeros((d, 1)), G]]), T)
    objective = cp.Minimize(cp.abs(cp.trace(Y @ GT)))
    constraints = [Y >> 0]

    # Strategy-specific setup
    if strategy == "seq":
        return _solve_sequential(
            G,
            T,
            DT,
            dims,
            bound_type,
            Y,
            Y_real,
            Y_imag,
            objective,
            constraints,
            mosek_params,
            energy_bound,
            energy_hamiltonian,
        )
    elif strategy == "par":
        return _solve_parallel(
            G,
            T,
            DT,
            dims,
            bound_type,
            Y,
            Y_real,
            Y_imag,
            objective,
            constraints,
            mosek_params,
            energy_bound,
            energy_hamiltonian,
        )
    elif strategy == "sup":
        return _solve_superposition(
            G,
            T,
            DT,
            dims,
            bound_type,
            Y,
            Y_real,
            Y_imag,
            objective,
            constraints,
            mosek_params,
            energy_bound,
            energy_hamiltonian,
        )
    elif strategy == "swi":
        return _solve_switch(
            G,
            T,
            DT,
            dims,
            bound_type,
            Y,
            Y_real,
            Y_imag,
            objective,
            constraints,
            mosek_params,
            energy_bound,
            energy_hamiltonian,
        )
    elif strategy == "ico":
        return _solve_ico(
            G,
            T,
            DT,
            dims,
            bound_type,
            Y,
            Y_real,
            Y_imag,
            objective,
            constraints,
            mosek_params,
        )
    raise AssertionError("unreachable strategy dispatch")


def _solve_sequential(
    G,
    T,
    DT,
    dims,
    bound_type,
    Y,
    Y_real,
    Y_imag,
    objective,
    constraints,
    mosek_params,
    energy_bound,
    energy_hamiltonian,
):
    d4, d3, d2, d1 = dims
    d_sys = np.prod(dims)
    d = G.shape[0]

    # Common sequential constraints
    dP1 = 2
    P1_real = cp.Variable((dP1, dP1), symmetric=True)
    P1_imag = cp.Variable((dP1, dP1))
    P1 = P1_real + 1j * P1_imag
    constraints += [P1 >> 0, cp.trace(P1) == 1]

    # Constraint (D2)
    for b in range(d4):
        vec0_R = np.eye(d + 1)[:, 0].reshape((d + 1, 1))
        vecb_d4 = np.eye(d4)[:, b].reshape((d4, 1))
        vecb = np.kron(np.kron(vec0_R, vecb_d4), np.eye(8))

        bs_3 = [
            np.kron(np.array([[1], [0]]), np.eye(4)),
            np.kron(np.array([[0], [1]]), np.eye(4)),
        ]

        for bp in range(d4):
            vecbp_d4 = np.eye(d4)[:, bp].reshape((d4, 1))
            vecbp = np.kron(np.kron(vec0_R, vecbp_d4), np.eye(8))

            if b == bp:
                P2_temp = (vecb.conj().T @ Y_real @ vecb) + 1j * (vecb.conj().T @ Y_imag @ vecb)
                Tr3_P2_temp = (bs_3[0].conj().T @ P2_temp @ bs_3[0]) + (
                    bs_3[1].conj().T @ P2_temp @ bs_3[1]
                )
                constraints.append(Tr3_P2_temp == cp.kron(np.eye(d2), P1))
            else:
                constraints.append(
                    (vecbp.conj().T @ Y_real @ vecb) + 1j * (vecbp.conj().T @ Y_imag @ vecb)
                    == np.zeros((8, 8))
                )
                constraints.append(
                    (vecb.conj().T @ Y_real @ vecb) + 1j * (vecb.conj().T @ Y_imag @ vecb)
                    == (vecbp.conj().T @ Y_real @ vecbp) + 1j * (vecbp.conj().T @ Y_imag @ vecbp)
                )

    if energy_bound is not None:
        _add_two_use_sequential_energy_constraints(
            P2_temp,
            P1,
            input_axes=(0, 2),
            energy_hamiltonian=energy_hamiltonian,
            energy_bound=energy_bound,
            constraints=constraints,
        )

    # Add bound type specific constraints
    if bound_type == "SLD":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
    elif bound_type == "Holevo":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
        _add_holevo_constraints(Y_real, Y_imag, T, d_sys, d, constraints)
    elif bound_type == "NHB":
        _add_nhb_constraints(Y_real, Y_imag, d_sys, d, constraints)

    # Constraint (ii)
    _add_common_ii_constraints(Y_real, Y_imag, DT, d, constraints)

    prob = cp.Problem(objective, constraints)
    result = _solve_problem(prob, mosek_params)

    return result, Y.value, P1.value, P2_temp.value


def _solve_parallel(
    G,
    T,
    DT,
    dims,
    bound_type,
    Y,
    Y_real,
    Y_imag,
    objective,
    constraints,
    mosek_params,
    energy_bound,
    energy_hamiltonian,
):
    dA = dims[0] ** 2  # Assuming qubit systems with N_steps = 2
    dB = dims[0] ** 2
    d = G.shape[0]

    # Common parallel constraints
    for b in range(dB):
        vecb_R = np.eye(d + 1)[:, 0].reshape((d + 1, 1))
        vec_0 = np.array([[1], [0]])
        vec_1 = np.array([[0], [1]])
        vecbs_sys = [
            np.kron(np.kron(np.kron(vec_0, np.eye(2)), vec_0), np.eye(2)),
            np.kron(np.kron(np.kron(vec_0, np.eye(2)), vec_1), np.eye(2)),
            np.kron(np.kron(np.kron(vec_1, np.eye(2)), vec_0), np.eye(2)),
            np.kron(np.kron(np.kron(vec_1, np.eye(2)), vec_1), np.eye(2)),
        ]
        vecb = np.kron(vecb_R, vecbs_sys[b])

        for bp in range(dB):
            vecbp = np.kron(vecb_R, vecbs_sys[bp])

            if b == bp:
                temp_rho = (vecb.conj().T @ Y_real @ vecb) + 1j * (vecb.conj().T @ Y_imag @ vecb)
                constraints.append(
                    ((vecb.conj().T @ Y_real @ vecb) + 1j * (vecb.conj().T @ Y_imag @ vecb)) >> 0
                )
                constraints.append(
                    cp.trace((vecb.conj().T @ Y_real @ vecb) + 1j * (vecb.conj().T @ Y_imag @ vecb))
                    == 1
                )
            else:
                constraints.append(
                    (vecbp.conj().T @ Y_real @ vecb) + 1j * (vecbp.conj().T @ Y_imag @ vecb)
                    == np.zeros((dA, dA))
                )
                constraints.append(
                    (vecb.conj().T @ Y_real @ vecb) + 1j * (vecb.conj().T @ Y_imag @ vecb)
                    == (vecbp.conj().T @ Y_real @ vecbp) + 1j * (vecbp.conj().T @ Y_imag @ vecbp)
                )

    if energy_bound is not None:
        total_input_hamiltonian = np.kron(energy_hamiltonian, np.eye(2)) + np.kron(
            np.eye(2), energy_hamiltonian
        )
        constraints.append(cp.real(cp.trace(total_input_hamiltonian @ temp_rho)) <= energy_bound)

    # Add bound type specific constraints
    if bound_type == "SLD":
        _add_sld_constraints(Y_real, Y_imag, dB * dA, d, constraints)
    elif bound_type == "Holevo":
        _add_sld_constraints(Y_real, Y_imag, dB * dA, d, constraints)
        _add_holevo_constraints(Y_real, Y_imag, T, dB * dA, d, constraints)
    else:  # NHB
        _add_nhb_constraints(Y_real, Y_imag, dB * dA, d, constraints)

    # Constraint (ii)
    _add_common_ii_constraints(Y_real, Y_imag, DT, d, constraints)

    prob = cp.Problem(objective, constraints)
    result = _solve_problem(prob, mosek_params)

    return result, Y.value, temp_rho.value


def _solve_superposition(
    G,
    T,
    DT,
    dims,
    bound_type,
    Y,
    Y_real,
    Y_imag,
    objective,
    constraints,
    mosek_params,
    energy_bound,
    energy_hamiltonian,
):
    d4, d3, d2, d1 = dims
    d_sys = np.prod(dims)
    d = G.shape[0]

    # Common superposition constraints
    dP2 = 8
    P21_real = cp.Variable((dP2, dP2), symmetric=True)
    P21_imag = cp.Variable((dP2, dP2))
    P21 = P21_real + 1j * P21_imag
    P22_real = cp.Variable((dP2, dP2), symmetric=True)
    P22_imag = cp.Variable((dP2, dP2))
    P22 = P22_real + 1j * P22_imag

    dP1 = 2
    P11_real = cp.Variable((dP1, dP1), symmetric=True)
    P11_imag = cp.Variable((dP1, dP1))
    P11 = P11_real + 1j * P11_imag
    P12_real = cp.Variable((dP1, dP1), symmetric=True)
    P12_imag = cp.Variable((dP1, dP1))
    P12 = P12_real + 1j * P12_imag

    # Constraint (D2)
    vecRs = [np.eye(d + 1)[:, i].reshape((d + 1, 1)) for i in range(d + 1)]
    bsys = [np.kron(vecRs[i], np.eye(d_sys)) for i in range(d + 1)]
    leftd2 = sum(
        [
            bsys[i].conj().T @ Y @ (np.kron(vecRs[0] @ vecRs[0].conj().T, np.eye(d_sys))) @ bsys[i]
            for i in range(d + 1)
        ]
    )

    rightd2 = cp.kron(np.eye(2), P21) + move_subsystem(
        cp.kron(np.eye(2), P22), dims, dims, [0], [2]
    )
    constraints.append(leftd2 == rightd2)

    bs_3 = [
        np.kron(np.array([[1], [0]]), np.eye(4)),
        np.kron(np.array([[0], [1]]), np.eye(4)),
    ]
    bs_1 = [
        np.kron(np.eye(4), np.array([[1], [0]])),
        np.kron(np.eye(4), np.array([[0], [1]])),
    ]
    Tr3_P21_temp = (bs_3[0].conj().T @ P21 @ bs_3[0]) + (bs_3[1].conj().T @ P21 @ bs_3[1])
    Tr1_P22_temp = (bs_1[0].conj().T @ P22 @ bs_1[0]) + (bs_1[1].conj().T @ P22 @ bs_1[1])
    constraints.append(Tr3_P21_temp == cp.kron(np.eye(d2), P11))
    constraints.append(Tr1_P22_temp == cp.kron(np.eye(d2), P12))
    constraints += [P11 >> 0, P12 >> 0, cp.trace(P11) + cp.trace(P12) == 1]

    if energy_bound is not None:
        _add_two_use_sequential_energy_constraints(
            P21,
            P11,
            input_axes=(0, 2),
            energy_hamiltonian=energy_hamiltonian,
            energy_bound=energy_bound,
            constraints=constraints,
        )
        _add_two_use_sequential_energy_constraints(
            P22,
            P12,
            input_axes=(1, 2),
            energy_hamiltonian=energy_hamiltonian,
            energy_bound=energy_bound,
            constraints=constraints,
        )

    # Add bound type specific constraints
    if bound_type == "SLD":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
    elif bound_type == "Holevo":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
        _add_holevo_constraints(Y_real, Y_imag, T, d_sys, d, constraints)
    else:  # NHB
        _add_nhb_constraints(Y_real, Y_imag, d_sys, d, constraints)

    # Constraint (ii)
    _add_common_ii_constraints(Y_real, Y_imag, DT, d, constraints)

    prob = cp.Problem(objective, constraints)
    result = _solve_problem(prob, mosek_params)

    return result, P11.value, P12.value, P21.value, P22.value


def _solve_switch(
    G,
    T,
    DT,
    dims,
    bound_type,
    Y,
    Y_real,
    Y_imag,
    objective,
    constraints,
    mosek_params,
    energy_bound,
    energy_hamiltonian,
):
    d4, d3, d2, d1 = dims
    d_sys = np.prod(dims)
    d = G.shape[0]

    # Quantum-switch process constraints
    dP = 2
    P1_real = cp.Variable((dP, dP), symmetric=True)
    P1_imag = cp.Variable((dP, dP))
    P1 = P1_real + 1j * P1_imag
    P3_real = cp.Variable((dP, dP), symmetric=True)
    P3_imag = cp.Variable((dP, dP))
    P3 = P3_real + 1j * P3_imag

    bs = [np.array([[1], [0]]), np.array([[0], [1]])]
    identity_transfer = np.kron(
        np.kron(bs[0], bs[0]) + np.kron(bs[1], bs[1]),
        (np.kron(bs[0], bs[0]) + np.kron(bs[1], bs[1])).T,
    )

    # Constraint (D2)
    vecRs = [np.eye(d + 1)[:, i].reshape((d + 1, 1)) for i in range(d + 1)]
    bsys = [np.kron(vecRs[i], np.eye(d_sys)) for i in range(d + 1)]
    leftd2 = sum(
        [
            bsys[i].conj().T @ Y @ (np.kron(vecRs[0] @ vecRs[0].conj().T, np.eye(d_sys))) @ bsys[i]
            for i in range(d + 1)
        ]
    )

    rightd2 = cp.kron(cp.kron(np.eye(d4), identity_transfer), P1) + move_subsystem(
        cp.kron(cp.kron(identity_transfer, P3), np.eye(d2)), dims, dims, [1], [3]
    )
    constraints.append(leftd2 == rightd2)
    constraints += [P1 >> 0, P3 >> 0, cp.trace(P1) + cp.trace(P3) == 1]

    if energy_bound is not None:
        constraints += [
            cp.real(cp.trace(energy_hamiltonian @ P1)) <= energy_bound * cp.real(cp.trace(P1)),
            cp.real(cp.trace(energy_hamiltonian @ P3)) <= energy_bound * cp.real(cp.trace(P3)),
        ]

    # Add bound type specific constraints
    if bound_type == "SLD":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
    elif bound_type == "Holevo":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
        _add_holevo_constraints(Y_real, Y_imag, T, d_sys, d, constraints)
    else:  # NHB
        _add_nhb_constraints(Y_real, Y_imag, d_sys, d, constraints)

    # Constraint (ii)
    _add_common_ii_constraints(Y_real, Y_imag, DT, d, constraints)

    prob = cp.Problem(objective, constraints)
    result = _solve_problem(prob, mosek_params)

    return result, P1.value, P3.value


def _solve_ico(G, T, DT, dims, bound_type, Y, Y_real, Y_imag, objective, constraints, mosek_params):
    d4, d3, d2, d1 = dims
    d_sys = np.prod(dims)
    d = G.shape[0]

    # General indefinite-causal-order process constraints
    dP = 16
    P_real = cp.Variable((dP, dP), symmetric=True)
    P_imag = cp.Variable((dP, dP))
    P = P_real + 1j * P_imag

    # Constraint (D2): condition (i'), note that the H_B is replaced by H_d4
    vecRs = [np.eye(d + 1)[:, i].reshape((d + 1, 1)) for i in range(d + 1)]
    bsys = [np.kron(vecRs[i], np.eye(d_sys)) for i in range(d + 1)]
    leftd2 = sum(
        [
            bsys[i].conj().T @ Y @ (np.kron(vecRs[0] @ vecRs[0].conj().T, np.eye(d_sys))) @ bsys[i]
            for i in range(d + 1)
        ]
    )

    constraints.append(leftd2 == P)
    constraints.append(P >> 0)

    constraints.append(cp.trace(P) == d2 * d4)
    constraints.append(
        cp.kron(np.eye(2) / 2, partial_trace(P, dims, [0, 1, 2])) - partial_trace(P, dims, [0, 1])
        == 0
    )
    constraints.append(
        partial_trace(P, dims, [2, 3]) - cp.kron(np.eye(2) / 2, partial_trace(P, dims, [0, 2, 3]))
        == 0
    )
    constraints.append(
        cp.kron(np.eye(2) / 2, partial_trace(P, dims, [0]))
        + move_subsystem(cp.kron(np.eye(2) / 2, partial_trace(P, dims, [2])), dims, dims, [0], [2])
        - move_subsystem(
            cp.kron(np.eye(4) / 4, partial_trace(P, dims, [0, 2])), dims, dims, [1], [2]
        )
        - P
        == 0
    )

    # Add bound type specific constraints
    if bound_type == "SLD":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
    elif bound_type == "Holevo":
        _add_sld_constraints(Y_real, Y_imag, d_sys, d, constraints)
        _add_holevo_constraints(Y_real, Y_imag, T, d_sys, d, constraints)
    else:  # NHB
        _add_nhb_constraints(Y_real, Y_imag, d_sys, d, constraints)

    # Constraint (ii)
    _add_common_ii_constraints(Y_real, Y_imag, DT, d, constraints)

    prob = cp.Problem(objective, constraints)
    result = _solve_problem(prob, mosek_params)

    return result, P.value


def _add_common_ii_constraints(Y_real, Y_imag, DT, d, constraints):
    """Add common (ii) constraints"""
    for j in range(d):
        for jp in range(d):
            jp_ket_R = np.eye(d + 1)[:, jp + 1]
            zjp = np.outer(np.eye(d + 1)[:, 0], jp_ket_R)

            if j == jp:
                constraints.append(
                    0.5 * cp.trace((Y_real + 1j * Y_imag) @ np.kron(zjp + zjp.conj().T, DT[j])) == 1
                )
            else:
                constraints.append(
                    0.5 * cp.trace((Y_real + 1j * Y_imag) @ np.kron(zjp + zjp.conj().T, DT[j])) == 0
                )
