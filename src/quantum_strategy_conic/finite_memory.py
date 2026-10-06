"""
Finite-memory sequential strategy / quantum comb see-saw SDP.

Conventions
-----------
- dims = (d_{2N}, d_{2N-1}, ..., d_1)
- N_theta labels: [H_{2N}, H_{2N-1}, ..., H_1]
- P_N labels:    [H_{2N-1}, ..., H_1, M_N]
- rho labels:    [H_{2N}, M_N]
- C1 labels:     [H1, M1]
- Ck labels:     [H_{2k-2}, M_{k-1}, H_{2k-1}, M_k], k >= 2

The link product is implemented as tensor-index contraction on equal labels:
ket index with ket index and bra index with bra index. This is the usual
Choi link product written in index form; equivalently, it contains the
partial transpose appearing in the trace formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import scipy.optimize

from ._bound_constraints import (
    add_holevo_constraints as _add_holevo_constraints,
)
from ._bound_constraints import (
    add_nhb_constraints as _add_nhb_constraints,
)
from ._bound_constraints import (
    add_sld_constraints as _add_sld_constraints,
)
from ._bound_constraints import (
    normalize_bound_name,
)
from ._solver import solve_problem as _solve_problem

Array = np.ndarray
Labels = List[str]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def prod(xs: Iterable[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def _as_tuple_ints(xs: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in xs)


def ancilla_qubits_to_dM(num_ancilla_qubits: int) -> int:
    """Return memory dimension dM = 2**num_ancilla_qubits.

    We now only support one or two ancilla qubits, i.e. dM=2 or dM=4.
    The dM=1/no-ancilla case is intentionally disallowed because it cannot
    support the multiparameter estimation problem considered here.
    """
    n = int(num_ancilla_qubits)
    if n not in (1, 2):
        raise ValueError("num_ancilla_qubits must be 1 or 2; dM=1 is not supported")
    return 2**n


def infer_num_ancilla_qubits_from_dM(dM: int) -> int:
    """Inverse of ancilla_qubits_to_dM for dM in {2,4}."""
    dM = int(dM)
    if dM == 2:
        return 1
    if dM == 4:
        return 2
    raise ValueError("dM must be 2 or 4; dM=1 is not supported")


def resolve_memory_dimension(
    dM: Optional[int] = None,
    num_ancilla_qubits: Optional[int] = None,
) -> int:
    """Resolve memory dimension from either dM or num_ancilla_qubits.

    If both are supplied, they must be consistent. This keeps the old dM-based
    API working while allowing the new ancilla-qubit API.
    """
    if num_ancilla_qubits is None:
        if dM is None:
            raise ValueError("Either dM or num_ancilla_qubits must be provided")
        dM_int = int(dM)
        if dM_int not in (2, 4):
            raise ValueError(f"dM must be 2 or 4; dM=1 is not supported, got {dM}")
        return dM_int

    dM_from_ancilla = ancilla_qubits_to_dM(int(num_ancilla_qubits))
    if dM is not None and int(dM) != dM_from_ancilla:
        raise ValueError(
            f"Inconsistent memory settings: dM={dM}, but "
            f"num_ancilla_qubits={num_ancilla_qubits} gives dM={dM_from_ancilla}"
        )
    return dM_from_ancilla


def _flat_index_c(multi: Sequence[int], dims: Sequence[int]) -> int:
    """C-order row-major flattening for a tensor with axes `dims`."""
    idx = 0
    for a, d in zip(multi, dims):
        idx = idx * int(d) + int(a)
    return int(idx)


def _iter_multi(dims: Sequence[int]):
    dims = list(dims)
    if len(dims) == 0:
        yield ()
    else:
        yield from product(*[range(int(d)) for d in dims])


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.conj().T)


def _require_unique(labels: Sequence[str], name: str) -> None:
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} contains repeated labels: {labels}")


def _require_square_matrix(A: Array, dim: int, name: str) -> None:
    if not isinstance(A, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(A)}")
    if A.shape != (dim, dim):
        raise ValueError(f"{name}.shape={A.shape}, expected {(dim, dim)}")


def _dims_for_labels(labels: Sequence[str], dims_by_label: Dict[str, int]) -> List[int]:
    missing = [lab for lab in labels if lab not in dims_by_label]
    if missing:
        raise KeyError(f"Missing dimensions for labels {missing}")
    return [int(dims_by_label[lab]) for lab in labels]


def make_dims_by_label(N: int, dims: Sequence[int], dM: int) -> Dict[str, int]:
    """Map labels H1,...,H2N,M1,...,MN to dimensions.

    dims is in descending comb-channel order: (d_{2N},...,d_1).
    """
    dims = _as_tuple_ints(dims)
    if len(dims) != 2 * N:
        raise ValueError(f"len(dims)={len(dims)}, expected 2N={2 * N}")
    if any(d <= 0 for d in dims):
        raise ValueError(f"All Hilbert-space dimensions must be positive: {dims}")
    if int(dM) <= 0:
        raise ValueError(f"dM must be positive, got {dM}")

    d: Dict[str, int] = {}
    for j in range(1, 2 * N + 1):
        d[f"H{j}"] = int(dims[2 * N - j])
    for k in range(1, N + 1):
        d[f"M{k}"] = int(dM)
    return d


def N_labels(N: int) -> Labels:
    return [f"H{j}" for j in range(2 * N, 0, -1)]


def P_labels(N: int) -> Labels:
    return [f"H{j}" for j in range(2 * N - 1, 0, -1)] + [f"M{N}"]


def rho_labels(N: int) -> Labels:
    return [f"H{2 * N}", f"M{N}"]


def C_labels(k: int) -> Labels:
    if k == 1:
        return ["H1", "M1"]
    return [f"H{2 * k - 2}", f"M{k - 1}", f"H{2 * k - 1}", f"M{k}"]


def dim_of_labels(labels: Sequence[str], dims_by_label: Dict[str, int]) -> int:
    return prod(_dims_for_labels(labels, dims_by_label))


# ---------------------------------------------------------------------------
# Label-based tensor-network operations
# ---------------------------------------------------------------------------


def reorder_operator_by_labels(
    A: Array,
    old_labels: Sequence[str],
    new_labels: Sequence[str],
    dims_by_label: Dict[str, int],
) -> Array:
    """Reorder an operator from old tensor labels to new tensor labels.

    Both ket and bra tensor axes are permuted in the same way.
    """
    old_labels = list(old_labels)
    new_labels = list(new_labels)
    _require_unique(old_labels, "old_labels")
    _require_unique(new_labels, "new_labels")
    if set(old_labels) != set(new_labels):
        raise ValueError(f"Label sets differ: old={old_labels}, new={new_labels}")

    old_dims = _dims_for_labels(old_labels, dims_by_label)
    D = prod(old_dims)
    _require_square_matrix(A, D, "A")
    if old_labels == new_labels:
        return np.array(A, dtype=complex, copy=True)

    n = len(old_labels)
    perm_ket = [old_labels.index(lab) for lab in new_labels]
    perm_bra = [n + old_labels.index(lab) for lab in new_labels]
    T = A.reshape(old_dims + old_dims, order="C")
    T = np.transpose(T, axes=perm_ket + perm_bra)
    new_dims = _dims_for_labels(new_labels, dims_by_label)
    return T.reshape((prod(new_dims), prod(new_dims)), order="C")


def link_product_by_labels(
    A: Array,
    a_labels: Sequence[str],
    B: Array,
    b_labels: Sequence[str],
    dims_by_label: Dict[str, int],
    out_labels: Optional[Sequence[str]] = None,
    return_labels: bool = False,
):
    """Tensor-network implementation of the Choi link product.

    Common labels are linked/contracted. Non-common labels are preserved.
    The natural output ordering is
        [A non-common labels in A order] + [B non-common labels in B order].
    Use out_labels to request a final label ordering.

    Index convention for every linked label l:
        A_ket[l] is contracted with B_ket[l], and
        A_bra[l] is contracted with B_bra[l].
    This is the index form of the link product compatible with the Choi
    matrices J = sum |K>><<K| using row-major vec over [output,input].
    """
    a_labels = list(a_labels)
    b_labels = list(b_labels)
    _require_unique(a_labels, "a_labels")
    _require_unique(b_labels, "b_labels")

    common = [lab for lab in a_labels if lab in b_labels]
    a_nc = [lab for lab in a_labels if lab not in common]
    b_nc = [lab for lab in b_labels if lab not in common]

    a_dims = _dims_for_labels(a_labels, dims_by_label)
    b_dims = _dims_for_labels(b_labels, dims_by_label)
    _require_square_matrix(A, prod(a_dims), "A")
    _require_square_matrix(B, prod(b_dims), "B")

    # A tensor axes: A ket labels, then A bra labels.
    # B tensor axes: B ket labels, then B bra labels.
    TA = A.reshape(a_dims + a_dims, order="C")
    TB = B.reshape(b_dims + b_dims, order="C")

    nA = len(a_labels)
    nB = len(b_labels)
    a_axes = [a_labels.index(lab) for lab in common] + [nA + a_labels.index(lab) for lab in common]
    b_axes = [b_labels.index(lab) for lab in common] + [nB + b_labels.index(lab) for lab in common]

    T = np.tensordot(TA, TB, axes=(a_axes, b_axes))

    # tensordot output axes are:
    #   A_nc ket, A_nc bra, B_nc ket, B_nc bra.
    # Reorder to:
    #   A_nc ket, B_nc ket, A_nc bra, B_nc bra.
    na = len(a_nc)
    nb = len(b_nc)
    perm = (
        list(range(na))
        + list(range(2 * na, 2 * na + nb))
        + list(range(na, 2 * na))
        + list(range(2 * na + nb, 2 * na + 2 * nb))
    )
    if len(perm) > 0:
        T = np.transpose(T, axes=perm)

    natural_labels = a_nc + b_nc
    natural_dims = _dims_for_labels(natural_labels, dims_by_label)
    D = prod(natural_dims) if natural_dims else 1
    C = T.reshape((D, D), order="C")

    if out_labels is not None:
        out_labels = list(out_labels)
        C = reorder_operator_by_labels(C, natural_labels, out_labels, dims_by_label)
        natural_labels = out_labels

    if return_labels:
        return C, natural_labels
    return C


def partial_trace_over_labels_numeric(
    A: Array,
    labels: Sequence[str],
    trace_labels: Sequence[str],
    dims_by_label: Dict[str, int],
) -> Array:
    """Partial trace over trace_labels, preserving remaining labels in order."""
    labels = list(labels)
    trace_labels = list(trace_labels)
    _require_unique(labels, "labels")
    if not set(trace_labels).issubset(set(labels)):
        raise ValueError(f"trace_labels={trace_labels} not subset of labels={labels}")

    keep_labels = [lab for lab in labels if lab not in trace_labels]
    dims_all = _dims_for_labels(labels, dims_by_label)
    dims_keep = _dims_for_labels(keep_labels, dims_by_label)
    dims_trace = _dims_for_labels(trace_labels, dims_by_label)
    D_all = prod(dims_all)
    D_keep = prod(dims_keep) if dims_keep else 1
    _require_square_matrix(A, D_all, "A")

    keep_pos = {lab: i for i, lab in enumerate(keep_labels)}
    trace_pos = {lab: i for i, lab in enumerate(trace_labels)}

    out = np.zeros((D_keep, D_keep), dtype=complex)
    for kr in _iter_multi(dims_keep):
        r_keep = _flat_index_c(kr, dims_keep) if dims_keep else 0
        for kc in _iter_multi(dims_keep):
            c_keep = _flat_index_c(kc, dims_keep) if dims_keep else 0
            s = 0.0 + 0.0j
            for tr in _iter_multi(dims_trace):
                full_r = []
                full_c = []
                for lab in labels:
                    if lab in trace_pos:
                        val = tr[trace_pos[lab]]
                        full_r.append(val)
                        full_c.append(val)
                    else:
                        full_r.append(kr[keep_pos[lab]])
                        full_c.append(kc[keep_pos[lab]])
                r = _flat_index_c(full_r, dims_all)
                c = _flat_index_c(full_c, dims_all)
                s += A[r, c]
            out[r_keep, c_keep] = s
    return out


# ---------------------------------------------------------------------------
# Comb construction and channel link
# ---------------------------------------------------------------------------


def validate_Cs(Cs: Sequence[Array], dims: Sequence[int], dM: int) -> None:
    N = len(dims) // 2
    if len(Cs) != N:
        raise ValueError(f"len(Cs)={len(Cs)}, expected N={N}")
    dims_by_label = make_dims_by_label(N, dims, dM)
    for k, C in enumerate(Cs, start=1):
        labs = C_labels(k)
        D = dim_of_labels(labs, dims_by_label)
        _require_square_matrix(C, D, f"C{k}")


def channel_use_labels(k: int) -> Labels:
    """Labels of the k-th physical channel use: [H_{2k}, H_{2k-1}].

    The local Choi N_channels[k-1] is interpreted as a channel
        H_{2k-1} -> H_{2k}
    with operator labels [output, input] = [H_{2k}, H_{2k-1}].
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    return [f"H{2 * k}", f"H{2 * k - 1}"]


def validate_local_channel_data(
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    W: Array,
    dims: Sequence[int],
    dM: int,
) -> Dict[str, int]:
    """Validate local channel Choi data.

    N_channels[k-1] acts on labels [H_{2k}, H_{2k-1}].
    DN_channels[j][k-1] is the derivative of the k-th local channel for
    parameter j, with the same labels.

    For a repeated channel with shared parameters, use for example
        N_channels = [N_single for _ in range(N)]
        DN_channels[j] = [dN_single_j for _ in range(N)]
    The terminal derivative is then computed as the sum over one derivative
    insertion at each use.
    """
    dims = _as_tuple_ints(dims)
    if len(dims) % 2 != 0:
        raise ValueError("dims must have even length 2N")
    N = len(dims) // 2
    if len(N_channels) != N:
        raise ValueError(f"len(N_channels)={len(N_channels)}, expected N={N}")
    dims_by_label = make_dims_by_label(N, dims, dM)

    for k, Nk in enumerate(N_channels, start=1):
        labs = channel_use_labels(k)
        D = dim_of_labels(labs, dims_by_label)
        _require_square_matrix(np.asarray(Nk), D, f"N_channels[{k - 1}]")

    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got W.shape={W.shape}")
    q = W.shape[0]
    if len(DN_channels) != q:
        raise ValueError(f"len(DN_channels)={len(DN_channels)} must equal q={q}")
    for j, row in enumerate(DN_channels):
        if len(row) != N:
            raise ValueError(f"len(DN_channels[{j}])={len(row)}, expected N={N}")
        for k, DNjk in enumerate(row, start=1):
            labs = channel_use_labels(k)
            D = dim_of_labels(labs, dims_by_label)
            _require_square_matrix(np.asarray(DNjk), D, f"DN_channels[{j}][{k - 1}]")
    return dims_by_label


def terminal_state_for_channel_sequence(
    channel_ops: Sequence[Array],
    Cs: Sequence[Array],
    dims: Sequence[int],
    dM: int,
    return_labels: bool = False,
    symmetrize: bool = True,
):
    """Compute terminal state for a concrete sequence of local channels.

    This contracts in physical time order and never constructs the global
    channel Choi nor the global comb:

        C1 -- N1 -- C2 -- N2 -- ... -- CN -- NN.

    Here Nk has labels [H_{2k}, H_{2k-1}] and Ck has the usual finite-memory
    labels.  The output is ordered as [H_{2N}, M_N].
    """
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    if len(channel_ops) != N:
        raise ValueError(f"len(channel_ops)={len(channel_ops)}, expected N={N}")
    if len(Cs) != N:
        raise ValueError(f"len(Cs)={len(Cs)}, expected N={N}")
    dims_by_label = make_dims_by_label(N, dims, dM)

    current = np.asarray(Cs[0], dtype=complex)
    labs = C_labels(1)
    D1 = dim_of_labels(labs, dims_by_label)
    _require_square_matrix(current, D1, "C1")

    for k in range(1, N + 1):
        Nk = np.asarray(channel_ops[k - 1], dtype=complex)
        nk_labs = channel_use_labels(k)
        Dnk = dim_of_labels(nk_labs, dims_by_label)
        _require_square_matrix(Nk, Dnk, f"channel_ops[{k - 1}]")

        current, labs = link_product_by_labels(
            current,
            labs,
            Nk,
            nk_labs,
            dims_by_label,
            return_labels=True,
        )

        if k < N:
            Cnext = np.asarray(Cs[k], dtype=complex)
            c_labs = C_labels(k + 1)
            Dc = dim_of_labels(c_labs, dims_by_label)
            _require_square_matrix(Cnext, Dc, f"C{k + 1}")
            current, labs = link_product_by_labels(
                current,
                labs,
                Cnext,
                c_labs,
                dims_by_label,
                return_labels=True,
            )

    desired = rho_labels(N)
    current = reorder_operator_by_labels(current, labs, desired, dims_by_label)
    if symmetrize:
        current = _sym(current)
    if return_labels:
        return current, desired
    return current


def terminal_state_from_local_channels(
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    Cs: Sequence[Array],
    dims: Sequence[int],
    dM: int,
    symmetrize: bool = True,
) -> Tuple[Array, List[Array]]:
    """Return rho and drhos from local channel Choi lists.

    rho is computed using N_channels at every use.  For parameter j, drho_j is
    the sum over all single-site derivative insertions:

        drho_j = sum_r N_N * ... * dN_{j,r} * ... * N_1 * C_1 * ... * C_N.

    This is equivalent to using the global derivative DN[j] for a tensor-product
    channel, but avoids ever forming DN[j].
    """
    validate_local_channel_data(N_channels, DN_channels, np.eye(len(DN_channels)), dims, dM)
    N = len(N_channels)
    rho = terminal_state_for_channel_sequence(N_channels, Cs, dims, dM, symmetrize=symmetrize)
    drhos: List[Array] = []
    for j, row in enumerate(DN_channels):
        acc = np.zeros_like(rho, dtype=complex)
        for r in range(N):
            ops = list(N_channels)
            ops[r] = row[r]
            acc += terminal_state_for_channel_sequence(ops, Cs, dims, dM, symmetrize=False)
        drhos.append(_sym(acc) if symmetrize else acc)
    return rho, drhos


def system_first_ancilla_bell_C1(dH1: int, dM: int) -> Array:
    """Theory-motivated C1 on [H1, M1].

    The system qubit is labelled 0.  The memory M1 consists of n_a ancilla
    qubits, with dM=2**n_a, labelled 1,...,n_a.  The initial state is

        (|00> + |11>)_{0,1} / sqrt(2)  tensor  |0>_2 ... |0>_{n_a}.

    In the flattened [H1, M1] basis, the memory index is interpreted in
    row-major/big-endian binary order: for dM=4, |a1 a2> has index
    2*a1 + a2.  Therefore dM=4 uses nonzero amplitudes

        |H=0, M=00> and |H=1, M=10>,

    not |H=1, M=01>.
    """
    dH1 = int(dH1)
    dM = int(dM)
    if dH1 != 2:
        raise ValueError(f"system_first_ancilla_bell_C1 assumes qubit H1; got dH1={dH1}")
    if dM not in (2, 4):
        raise ValueError(f"dM must be 2 or 4 for the supported ancilla initial states; got {dM}")

    n_a = infer_num_ancilla_qubits_from_dM(dM)
    psi = np.zeros(dH1 * dM, dtype=complex)

    # Term |0>_system |0...0>_memory.
    mem0 = 0
    psi[0 * dM + mem0] = 1.0 / np.sqrt(2.0)

    # Term |1>_system |1 0 ... 0>_memory.  With big-endian memory bits,
    # the first ancilla qubit contributes 2**(n_a-1).
    mem1 = 1 << (n_a - 1)
    psi[1 * dM + mem1] = 1.0 / np.sqrt(2.0)

    return np.outer(psi, psi.conj())


def maximally_entangled_C1(dH1: int, dM: int) -> Array:
    """Backward-compatible alias for the supported theory initial state.

    For dM=2 this is the ordinary Bell state. For dM=4 this entangles the
    system only with the first ancilla qubit and initializes the second
    ancilla qubit in |0>.
    """
    return system_first_ancilla_bell_C1(dH1, dM)


def full_schmidt_C1(dH1: int, dM: int) -> Array:
    """Old full-Schmidt initialization on [H1,M1], kept for diagnostics only."""
    dH1 = int(dH1)
    dM = int(dM)
    r = min(dH1, dM)
    if r <= 0:
        raise ValueError("dH1 and dM must be positive")
    psi = np.zeros(dH1 * dM, dtype=complex)
    for a in range(r):
        psi[a * dM + a] = 1.0 / np.sqrt(r)
    return np.outer(psi, psi.conj())


def maximally_entangled_choi_input_output(Din: int, Dout: int) -> Array:
    """Unnormalized |Omega><Omega| in [input, output] ordering.

    This is the identity-channel Choi matrix for the convention used by C_k:
    labels [input labels, output labels], with Tr_output J = I_input.
    Requires Din == Dout.
    """
    Din = int(Din)
    Dout = int(Dout)
    if Din != Dout:
        raise ValueError(f"Identity Choi requires Din == Dout, got {Din} and {Dout}")
    v = np.zeros(Din * Dout, dtype=complex)
    for a in range(Din):
        v[a * Dout + a] = 1.0
    return np.outer(v, v.conj())


def random_unitary_choi_input_output(Din: int, Dout: int, rng=None) -> Array:
    """Random unitary/isometry Choi in [input, output] ordering.

    For Din == Dout this is a Haar-like random unitary channel. For unequal
    dimensions, it constructs an isometry V: input -> output when Dout >= Din.
    """
    Din = int(Din)
    Dout = int(Dout)
    if Dout < Din:
        raise ValueError("Random isometry initialization requires Dout >= Din")
    rng = np.random.default_rng() if rng is None else rng
    Z = rng.normal(size=(Dout, Din)) + 1j * rng.normal(size=(Dout, Din))
    Q, _ = np.linalg.qr(Z)
    V = Q[:, :Din]
    # C labels are [input, output], so vector index is input first, output second.
    v = np.zeros(Din * Dout, dtype=complex)
    for a in range(Din):
        for b in range(Dout):
            v[a * Dout + b] = V[b, a]
    return np.outer(v, v.conj())


def unitary_choi_input_output(U: Array, check_isometry: bool = True, atol: float = 1e-8) -> Array:
    """Choi matrix of an isometry/unitary U: input -> output.

    The returned Choi operator is in [input, output] ordering, matching C_k:
        C_k labels = [input labels..., output labels...].

    If U has matrix elements U[b,a] = <b_out|U|a_in>, then
        |U>>_[input,output] = sum_{a,b} U[b,a] |a_in>|b_out>.

    Therefore Tr_output |U>><<U| = I_input when U^† U = I_input.
    """
    U = np.asarray(U, dtype=complex)
    if U.ndim != 2:
        raise ValueError(f"U must be a matrix, got shape {U.shape}")
    Dout, Din = U.shape
    if check_isometry:
        err = np.linalg.norm(U.conj().T @ U - np.eye(Din, dtype=complex))
        if err > atol:
            raise ValueError(
                f"U is not an isometry/unitary to tolerance {atol}; ||U^†U-I||={err:.3e}"
            )

    v = np.zeros(Din * Dout, dtype=complex)
    for a in range(Din):
        for b in range(Dout):
            v[a * Dout + b] = U[b, a]
    return np.outer(v, v.conj())


def feedback_control_choi_Ck(
    E: Array,
    k: int,
    dims: Sequence[int],
    dM: int,
    use_adjoint: bool = True,
    check_unitary: bool = True,
) -> Array:
    """Construct feedback-control C_k with U_fb = E^† ⊗ I_M.

    C_k ordering is [H_{2k-2}, M_{k-1}, H_{2k-1}, M_k].
    Hence U_fb is interpreted as a map
        H_{2k-2} ⊗ M_{k-1}  ->  H_{2k-1} ⊗ M_k,
    with row/output ordering [H_{2k-1}, M_k] and column/input ordering
    [H_{2k-2}, M_{k-1}].

    Pass E as the single-use channel unitary. With use_adjoint=True, the
    control uses E^† on the system and identity on memory.
    """
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    if not (2 <= k <= N):
        raise ValueError(f"feedback_control_choi_Ck applies to k=2..N; got k={k}, N={N}")
    dims_by_label = make_dims_by_label(N, dims, dM)

    d_in_sys = dims_by_label[f"H{2 * k - 2}"]
    d_out_sys = dims_by_label[f"H{2 * k - 1}"]
    if d_in_sys != d_out_sys:
        raise ValueError(
            f"Feedback E^† requires equal system dimensions for H{2 * k - 2} and "
            f"H{2 * k - 1}; got {d_in_sys} and {d_out_sys}"
        )

    E = np.asarray(E, dtype=complex)
    if E.shape != (d_out_sys, d_in_sys):
        raise ValueError(
            f"E.shape={E.shape}; expected ({d_out_sys}, {d_in_sys}) for H{2 * k - 2}->H{2 * k - 1}"
        )

    F = E.conj().T if use_adjoint else E
    I_M = np.eye(int(dM), dtype=complex)
    # Rows are output labels [H_{2k-1}, M_k]; columns are input labels [H_{2k-2}, M_{k-1}].
    U_fb = np.kron(F, I_M)

    Din = dims_by_label[f"H{2 * k - 2}"] * dims_by_label[f"M{k - 1}"]
    Dout = dims_by_label[f"H{2 * k - 1}"] * dims_by_label[f"M{k}"]
    if U_fb.shape != (Dout, Din):
        raise RuntimeError(f"Internal U_fb shape {U_fb.shape}, expected {(Dout, Din)}")
    return unitary_choi_input_output(U_fb, check_isometry=check_unitary)


def initialize_choi_teeth_with_memory(
    dims: Sequence[int],
    dM: int,
    pure_C1: bool = True,
    c1_init: str = "zero",
    channel_init: str = "identity",
    seed: Optional[int] = None,
    feedback_E: Optional[Array] = None,
    feedback_use_adjoint: bool = True,
    feedback_teeth: Optional[Sequence[int]] = None,
) -> List[Array]:
    """Feasible initialization for local Choi teeth.

    Parameters
    ----------
    pure_C1:
        Backward-compatible switch. If c1_init is not set, True means
        C1=|0,0><0,0| and False means maximally mixed.
    c1_init:
        - "zero": C1=|0,0><0,0| on [H1,M1].
        - "bell" or "max_entangled": C1=|Phi+><Phi+| on [H1,M1].
        - "mixed": C1=I/(dim H1 * dim M1).
    channel_init:
        - "identity": use identity channels when input/output dimensions match;
          otherwise fall back to replacement channels.
        - "random_unitary": use random unitary/isometry channels when possible.
        - "depolarizing": replacement channel I_input tensor I_output/d_output.
        - "feedback_control": initialize every control tooth k=2,...,N as
          U_fb=E^†⊗I_M by default, with E supplied through feedback_E.
          To initialize only a subset, pass feedback_teeth=[...].

    The old deterministic depolarizing initialization is feasible but often
    makes drho_j approximately zero, so the strict unbiased X-SDP becomes
    infeasible at the first iteration.
    """
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    dims_by_label = make_dims_by_label(N, dims, dM)
    rng = np.random.default_rng(seed)
    Cs: List[Array] = []

    if channel_init == "feedback":
        channel_init = "feedback_control"
    feedback_set = (
        set(range(2, N + 1)) if feedback_teeth is None else set(int(k) for k in feedback_teeth)
    )
    if channel_init == "feedback_control":
        if feedback_E is None:
            raise ValueError("channel_init='feedback_control' requires feedback_E")
        bad = [k for k in feedback_set if not (2 <= k <= N)]
        if bad:
            raise ValueError(f"feedback_teeth must be in 2..N={N}, got {bad}")

    # C1: state on [H1, M1]
    labs1 = C_labels(1)
    D1 = dim_of_labels(labs1, dims_by_label)
    c1_init_norm = str(c1_init).lower()
    if c1_init_norm in (
        "bell",
        "theory",
        "system_first_ancilla_bell",
        "max_entangled",
        "maximally_entangled",
    ):
        C1 = system_first_ancilla_bell_C1(dims_by_label["H1"], dims_by_label["M1"])
    elif c1_init_norm in ("full_schmidt", "old_max_entangled"):
        C1 = full_schmidt_C1(dims_by_label["H1"], dims_by_label["M1"])
    elif c1_init_norm in ("zero", "pure_zero", "ket0"):
        C1 = np.zeros((D1, D1), dtype=complex)
        C1[0, 0] = 1.0
    elif c1_init_norm in ("mixed", "maximally_mixed"):
        C1 = np.eye(D1, dtype=complex) / D1
    elif c1_init_norm in ("legacy", "pure_C1"):
        if pure_C1:
            C1 = np.zeros((D1, D1), dtype=complex)
            C1[0, 0] = 1.0
        else:
            C1 = np.eye(D1, dtype=complex) / D1
    else:
        raise ValueError("c1_init must be 'zero', 'bell'/'theory', 'full_schmidt', or 'mixed'")
    Cs.append(C1)

    for k in range(2, N + 1):
        labs = C_labels(k)
        in_labs = labs[:2]
        out_labs = labs[2:]
        Din = dim_of_labels(in_labs, dims_by_label)
        Dout = dim_of_labels(out_labs, dims_by_label)

        if channel_init == "feedback_control" and k in feedback_set:
            Ck = feedback_control_choi_Ck(
                feedback_E,
                k,
                dims,
                dM,
                use_adjoint=feedback_use_adjoint,
                check_unitary=True,
            )
        elif channel_init in ("identity", "feedback_control") and Din == Dout:
            Ck = maximally_entangled_choi_input_output(Din, Dout)
        elif channel_init == "random_unitary" and Dout >= Din:
            Ck = random_unitary_choi_input_output(Din, Dout, rng=rng)
        elif channel_init in (
            "identity",
            "feedback_control",
            "random_unitary",
            "depolarizing",
        ):
            Ck = np.kron(np.eye(Din, dtype=complex), np.eye(Dout, dtype=complex) / Dout)
        else:
            raise ValueError(
                "channel_init must be 'identity', 'random_unitary', 'depolarizing', "
                "or 'feedback_control'"
            )
        Cs.append(Ck)
    return Cs


# ---------------------------------------------------------------------------
# CVXPY helpers
# ---------------------------------------------------------------------------


def _cvx_vec_C(X):
    return cp.vec(X, order="C")


def _cvx_reshape_C(x, shape: Tuple[int, int]):
    return cp.reshape(x, shape, order="C")


def _block_functional_from_X(A_R: Array, X: Array, dS: int) -> Array:
    """Return G such that Tr[(A_R \\otimes Y) X] = Tr[G Y]."""
    rdim = A_R.shape[0]
    if A_R.shape != (rdim, rdim):
        raise ValueError("A_R must be square")
    _require_square_matrix(X, rdim * dS, "X")
    G = np.zeros((dS, dS), dtype=complex)
    for r in range(rdim):
        for s in range(rdim):
            block_s_r = X[s * dS : (s + 1) * dS, r * dS : (r + 1) * dS]
            G += A_R[r, s] * block_s_r
    return G


def _B_i(i: int, q: int) -> Array:
    """B_i = |0><i| + |i><0| on R, with i in {1,...,q}."""
    B = np.zeros((q + 1, q + 1), dtype=complex)
    B[0, i] = 1.0
    B[i, 0] = 1.0
    return B


def _barW(W: Array) -> Array:
    W = np.asarray(W, dtype=complex)
    q = W.shape[0]
    out = np.zeros((q + 1, q + 1), dtype=complex)
    out[1:, 1:] = W
    return out


def diagnose_X_sdp_inputs(rho: Array, drhos: Sequence[Array], W: Array) -> Dict[str, object]:
    """Diagnostics only; not a certificate of infeasibility."""
    rho = np.asarray(rho, dtype=complex)
    dS = rho.shape[0]
    q = len(drhos)
    diag: Dict[str, object] = {}
    diag["system_dim"] = dS
    diag["q"] = q
    diag["W_shape"] = tuple(np.asarray(W).shape)
    diag["rho_hermitian_error"] = float(np.linalg.norm(rho - rho.conj().T))
    diag["rho_trace"] = complex(np.trace(rho))
    try:
        diag["rho_min_eig_sym"] = float(np.min(np.linalg.eigvalsh(_sym(rho))))
    except np.linalg.LinAlgError:
        diag["rho_min_eig_sym"] = None

    herm_errors = []
    traces = []
    cols = []
    for D in drhos:
        D = np.asarray(D, dtype=complex)
        herm_errors.append(float(np.linalg.norm(D - D.conj().T)))
        traces.append(complex(np.trace(D)))
        cols.append(D.reshape(-1, order="C"))
    diag["drho_hermitian_errors"] = herm_errors
    diag["drho_traces"] = traces
    if cols:
        M = np.column_stack(cols)
        diag["drho_span_rank"] = int(np.linalg.matrix_rank(M))
        svals = np.linalg.svd(M, compute_uv=False)
        diag["drho_singular_values"] = [float(x) for x in svals]
        diag["drho_fro_norms"] = [float(np.linalg.norm(D)) for D in drhos]
        diag["drho_trace_norms"] = [
            float(np.sum(np.linalg.svd(D, compute_uv=False))) for D in drhos
        ]
        diag["likely_infeasible_reason"] = (
            "Derivative states are numerically zero, so the strict unbiased "
            "constraints 0.5*Tr[(B_i⊗drho_j)X]=delta_ij cannot be satisfied."
            if max(diag["drho_fro_norms"], default=0.0) < 1e-10
            else None
        )
    return diag


# ---------------------------------------------------------------------------
# Optional X-constraints for different bounds
# ---------------------------------------------------------------------------


def _normalize_x_bound(x_bound: Optional[str]) -> str:
    return normalize_bound_name(x_bound, allow_none=True)


def _add_X_bound_constraints(
    X: cp.Expression,
    rho: Array,
    d_sys: int,
    q: int,
    x_bound: Optional[str],
    constraints: List,
) -> str:
    """Append optional bound-specific constraints for the X variable.

    Returns the normalized bound name. The default 'none' adds no extra
    constraints beyond PSD, normalization, and unbiasedness.
    """
    bound = _normalize_x_bound(x_bound)
    if bound == "none":
        return bound

    Y_real = cp.real(X)
    Y_imag = cp.imag(X)
    if bound == "nhb":
        _add_nhb_constraints(Y_real, Y_imag, d_sys=d_sys, d=q, constraints=constraints)
    elif bound == "sld":
        _add_sld_constraints(Y_real, Y_imag, d_block=d_sys, d=q, constraints=constraints)
    elif bound == "holevo":
        _add_sld_constraints(Y_real, Y_imag, d_block=d_sys, d=q, constraints=constraints)
        _add_holevo_constraints(
            Y_real, Y_imag, state=rho, d_block=d_sys, d=q, constraints=constraints
        )
    else:  # pragma: no cover; protected by _normalize_x_bound
        raise RuntimeError(f"Unexpected normalized x_bound={bound}")
    return bound


# ---------------------------------------------------------------------------
# SDP 1: solve X for fixed comb
# ---------------------------------------------------------------------------


@dataclass
class SDPSolution:
    value: Optional[float]
    status: str
    variable_value: Optional[Array]
    solver: Optional[str]
    diagnostics: Optional[Dict[str, object]] = None


def solve_X_sdp(
    rho: Array,
    drhos: Sequence[Array],
    W: Array,
    solver: Optional[str] = "CLARABEL",
    verbose: bool = False,
    x_bound: Optional[str] = None,
) -> SDPSolution:
    """Solve the X-SDP for fixed rho and drho_j.

    x_bound controls optional extra X-constraints:
        None/'none': original X-SDP;
        'nhb': add NHB block constraints;
        'sld': add SLD block constraints;
        'holevo': add Holevo imaginary constraints using the fixed rho.
    """
    rho = _sym(np.asarray(rho, dtype=complex))
    drhos = [_sym(np.asarray(D, dtype=complex)) for D in drhos]
    W = np.asarray(W, dtype=complex)

    q = W.shape[0]
    if len(drhos) != q:
        raise ValueError(f"len(drhos)={len(drhos)} must equal q={q}")
    dS = rho.shape[0]
    for j, D in enumerate(drhos):
        _require_square_matrix(D, dS, f"drhos[{j}]")

    Rdim = q + 1
    Xdim = Rdim * dS
    X = cp.Variable((Xdim, Xdim), hermitian=True, name="X")

    constraints = [X >> 0]
    constraints.append(X[0:dS, 0:dS] == np.eye(dS))
    _add_X_bound_constraints(
        X=X,
        rho=rho,
        d_sys=dS,
        q=q,
        x_bound=x_bound,
        constraints=constraints,
    )

    for i in range(1, q + 1):
        Bi = _B_i(i, q)
        for j in range(q):
            Cij = np.kron(Bi, drhos[j])
            expr = 0.5 * cp.trace(Cij @ X)
            constraints.append(cp.real(expr) == (1.0 if (i - 1) == j else 0.0))
            constraints.append(cp.imag(expr) == 0.0)

    C_obj = np.kron(_barW(W), rho)
    objective = cp.Minimize(cp.real(cp.trace(C_obj @ X)))
    problem = cp.Problem(objective, constraints)
    chosen = _solve_problem(problem, solver=solver, verbose=verbose)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return SDPSolution(
            value=None,
            status=problem.status,
            variable_value=None,
            solver=chosen,
            diagnostics=diagnose_X_sdp_inputs(rho, drhos, W),
        )
    return SDPSolution(
        value=float(np.real(problem.value)),
        status=problem.status,
        variable_value=_sym(np.asarray(X.value, dtype=complex)),
        solver=chosen,
        diagnostics=None,
    )


# ---------------------------------------------------------------------------
# SDP 2: solve one C_k for fixed X and fixed other teeth
# ---------------------------------------------------------------------------


def _add_Ck_constraints(
    constraints: List,
    C: cp.Variable,
    k: int,
    dims: Sequence[int],
    dM: int,
) -> None:
    """Append PSD and C1/CPTP constraints for C_k."""
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    dims_by_label = make_dims_by_label(N, dims, dM)
    labs = C_labels(k)
    constraints.append(C >> 0)

    if k == 1:
        constraints.append(cp.real(cp.trace(C)) == 1.0)
        constraints.append(cp.imag(cp.trace(C)) == 0.0)
        return

    # Ck labels are [input1, input2, output1, output2].
    in_labs = labs[:2]
    out_labs = labs[2:]
    dims_all = _dims_for_labels(labs, dims_by_label)
    dims_in = _dims_for_labels(in_labs, dims_by_label)
    dims_out = _dims_for_labels(out_labs, dims_by_label)
    # Elementwise partial trace over output labels equals identity on input.
    for ir in _iter_multi(dims_in):
        row_in = _flat_index_c(ir, dims_in)
        for ic in _iter_multi(dims_in):
            col_in = _flat_index_c(ic, dims_in)
            s_expr = 0
            for out in _iter_multi(dims_out):
                row_full = tuple(ir) + tuple(out)
                col_full = tuple(ic) + tuple(out)
                r = _flat_index_c(row_full, dims_all)
                c = _flat_index_c(col_full, dims_all)
                s_expr += C[r, c]
            target = 1.0 if row_in == col_in else 0.0
            constraints.append(cp.real(s_expr) == target)
            constraints.append(cp.imag(s_expr) == 0.0)


def _sequence_nodes_for_range(
    start_k: int,
    end_k: int,
    Cs: Sequence[Array],
    channel_ops: Sequence[Array],
    include_C_start: bool,
    include_N_end: bool = True,
) -> List[Tuple[Array, Labels]]:
    """Build ordered tensor-network nodes for a segment of the chain.

    The full time-ordered chain is
        C1, N1, C2, N2, ..., CN, NN.

    This helper returns a list of (operator, labels) nodes from start_k to
    end_k.  If include_C_start is False, the first C_start_k is omitted; this
    is useful for the right environment after the variable C_k, whose first
    node should be N_k.
    """
    nodes: List[Tuple[Array, Labels]] = []
    if start_k > end_k:
        return nodes
    for t in range(start_k, end_k + 1):
        if include_C_start or t > start_k:
            nodes.append((np.asarray(Cs[t - 1], dtype=complex), C_labels(t)))
        if include_N_end:
            nodes.append((np.asarray(channel_ops[t - 1], dtype=complex), channel_use_labels(t)))
    return nodes


def _contract_node_sequence(
    nodes: Sequence[Tuple[Array, Labels]],
    dims_by_label: Dict[str, int],
) -> Tuple[Array, Labels]:
    """Contract a sequence of labelled operators.

    Returns a scalar identity with empty labels for an empty sequence.
    """
    if len(nodes) == 0:
        return np.ones((1, 1), dtype=complex), []
    current = np.asarray(nodes[0][0], dtype=complex)
    labs = list(nodes[0][1])
    for op, op_labs in nodes[1:]:
        current, labs = link_product_by_labels(
            current,
            labs,
            np.asarray(op, dtype=complex),
            list(op_labs),
            dims_by_label,
            return_labels=True,
        )
    return current, labs


def _align_environment_sum(
    envs: Sequence[Tuple[Array, Labels]],
    reference_labels: Labels,
    dims_by_label: Dict[str, int],
) -> Optional[Tuple[Array, Labels]]:
    """Sum environments after reordering them to reference_labels.

    Returns None for an empty derivative sum.
    """
    if len(envs) == 0:
        return None
    D = dim_of_labels(reference_labels, dims_by_label) if reference_labels else 1
    acc = np.zeros((D, D), dtype=complex)
    for op, labs in envs:
        if list(labs) != list(reference_labels):
            op = reorder_operator_by_labels(op, labs, reference_labels, dims_by_label)
        acc += op
    return acc, list(reference_labels)


def _environment_for_segment(
    start_k: int,
    end_k: int,
    Cs: Sequence[Array],
    channel_ops: Sequence[Array],
    dims_by_label: Dict[str, int],
    include_C_start: bool,
) -> Tuple[Array, Labels]:
    nodes = _sequence_nodes_for_range(
        start_k=start_k,
        end_k=end_k,
        Cs=Cs,
        channel_ops=channel_ops,
        include_C_start=include_C_start,
    )
    return _contract_node_sequence(nodes, dims_by_label)


def _derivative_environment_sum_for_segment(
    start_k: int,
    end_k: int,
    Cs: Sequence[Array],
    N_channels: Sequence[Array],
    dN_row: Sequence[Array],
    dims_by_label: Dict[str, int],
    include_C_start: bool,
    reference_labels: Labels,
) -> Optional[Tuple[Array, Labels]]:
    """Sum environments with one derivative channel insertion in a segment."""
    if start_k > end_k:
        return None
    envs: List[Tuple[Array, Labels]] = []
    for r in range(start_k, end_k + 1):
        ops = list(N_channels)
        ops[r - 1] = dN_row[r - 1]
        envs.append(
            _environment_for_segment(
                start_k=start_k,
                end_k=end_k,
                Cs=Cs,
                channel_ops=ops,
                dims_by_label=dims_by_label,
                include_C_start=include_C_start,
            )
        )
    return _align_environment_sum(envs, reference_labels, dims_by_label)


def _terminal_from_cached_envs(
    left_env: Optional[Tuple[Array, Labels]],
    Ck_op: Array,
    labs_k: Labels,
    right_env: Optional[Tuple[Array, Labels]],
    dims_by_label: Dict[str, int],
    out_labels: Labels,
) -> Array:
    """Contract left environment, C_k, and right environment."""
    current = np.asarray(Ck_op, dtype=complex)
    labs = list(labs_k)
    if left_env is not None:
        L, L_labs = left_env
        current, labs = link_product_by_labels(
            L, L_labs, current, labs, dims_by_label, return_labels=True
        )
    if right_env is not None:
        R, R_labs = right_env
        current, labs = link_product_by_labels(
            current, labs, R, R_labs, dims_by_label, return_labels=True
        )
    if list(labs) != list(out_labels):
        current = reorder_operator_by_labels(current, labs, out_labels, dims_by_label)
    return current


def _linear_maps_for_Ck_local_channels(
    k: int,
    Cs_fixed: Sequence[Array],
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    dims: Sequence[int],
    dM: int,
) -> Tuple[Array, List[Array], Labels]:
    """Build vec(C_k)->vec(rho), vec(drho_j) maps with cached environments.

    For fixed k, write the chain as

        left_before_k  *  C_k  *  right_after_k.

    The nominal terminal state uses nominal left/right environments.  The
    derivative terminal state for parameter j is

        (d left_j) * C_k * right + left * C_k * (d right_j),

    where d left_j sums derivative insertions in channel uses r<k, and
    d right_j sums derivative insertions in channel uses r>=k.  This avoids
    recomputing the entire length-N chain for every matrix unit of C_k.
    """
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    if not (1 <= k <= N):
        raise ValueError(f"k={k} outside 1..N={N}")
    dims_by_label = make_dims_by_label(N, dims, dM)

    labs_k = C_labels(k)
    Dk = dim_of_labels(labs_k, dims_by_label)
    dS = dim_of_labels(rho_labels(N), dims_by_label)
    nvar = Dk * Dk
    q = len(DN_channels)
    out_labs = rho_labels(N)

    # Nominal environments.
    if k == 1:
        left_nom = None
    else:
        left_nom = _environment_for_segment(
            start_k=1,
            end_k=k - 1,
            Cs=Cs_fixed,
            channel_ops=N_channels,
            dims_by_label=dims_by_label,
            include_C_start=True,
        )

    right_nom = _environment_for_segment(
        start_k=k,
        end_k=N,
        Cs=Cs_fixed,
        channel_ops=N_channels,
        dims_by_label=dims_by_label,
        include_C_start=False,
    )

    left_derivs: List[Optional[Tuple[Array, Labels]]] = []
    right_derivs: List[Optional[Tuple[Array, Labels]]] = []
    left_ref = left_nom[1] if left_nom is not None else []
    right_ref = right_nom[1]
    for j, dN_row in enumerate(DN_channels):
        if k == 1:
            left_derivs.append(None)
        else:
            left_derivs.append(
                _derivative_environment_sum_for_segment(
                    start_k=1,
                    end_k=k - 1,
                    Cs=Cs_fixed,
                    N_channels=N_channels,
                    dN_row=dN_row,
                    dims_by_label=dims_by_label,
                    include_C_start=True,
                    reference_labels=left_ref,
                )
            )
        right_derivs.append(
            _derivative_environment_sum_for_segment(
                start_k=k,
                end_k=N,
                Cs=Cs_fixed,
                N_channels=N_channels,
                dN_row=dN_row,
                dims_by_label=dims_by_label,
                include_C_start=False,
                reference_labels=right_ref,
            )
        )

    L_rho = np.zeros((dS * dS, nvar), dtype=complex)
    L_drhos = [np.zeros((dS * dS, nvar), dtype=complex) for _ in range(q)]

    for a in range(Dk):
        for b in range(Dk):
            col = a * Dk + b
            E = np.zeros((Dk, Dk), dtype=complex)
            E[a, b] = 1.0

            Y = _terminal_from_cached_envs(left_nom, E, labs_k, right_nom, dims_by_label, out_labs)
            L_rho[:, col] = Y.reshape(-1, order="C")

            for j in range(q):
                acc = np.zeros((dS, dS), dtype=complex)
                if left_derivs[j] is not None:
                    acc += _terminal_from_cached_envs(
                        left_derivs[j], E, labs_k, right_nom, dims_by_label, out_labs
                    )
                if right_derivs[j] is not None:
                    acc += _terminal_from_cached_envs(
                        left_nom, E, labs_k, right_derivs[j], dims_by_label, out_labs
                    )
                L_drhos[j][:, col] = acc.reshape(-1, order="C")

    return L_rho, L_drhos, labs_k


def solve_Ck_sdp_local_channels(
    k: int,
    X_fixed: Array,
    Cs_fixed: Sequence[Array],
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    W: Array,
    dims: Sequence[int],
    dM: int,
    solver: Optional[str] = "CLARABEL",
    verbose: bool = False,
) -> SDPSolution:
    """Solve one local Choi SDP using local channel Choi lists."""
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    dims_by_label = make_dims_by_label(N, dims, dM)
    q = np.asarray(W).shape[0]
    if len(DN_channels) != q:
        raise ValueError(f"len(DN_channels)={len(DN_channels)} must equal q={q}")

    dS = dim_of_labels(rho_labels(N), dims_by_label)
    X_fixed = _sym(np.asarray(X_fixed, dtype=complex))
    _require_square_matrix(X_fixed, (q + 1) * dS, "X_fixed")

    L_rho, L_drhos, labs_k = _linear_maps_for_Ck_local_channels(
        k, Cs_fixed, N_channels, DN_channels, dims, dM
    )
    Dk = dim_of_labels(labs_k, dims_by_label)
    C = cp.Variable((Dk, Dk), hermitian=True, name=f"C{k}")
    cvec = _cvx_vec_C(C)

    rho_expr = _cvx_reshape_C(L_rho @ cvec, (dS, dS))
    drho_exprs = [_cvx_reshape_C(L @ cvec, (dS, dS)) for L in L_drhos]

    G_obj = _block_functional_from_X(_barW(W), X_fixed, dS)
    objective = cp.Minimize(cp.real(cp.trace(G_obj @ rho_expr)))

    constraints: List = []
    _add_Ck_constraints(constraints, C, k, dims, dM)

    for i in range(1, q + 1):
        G_i = _block_functional_from_X(_B_i(i, q), X_fixed, dS)
        for j in range(q):
            expr = 0.5 * cp.trace(G_i @ drho_exprs[j])
            constraints.append(cp.real(expr) == (1.0 if (i - 1) == j else 0.0))
            constraints.append(cp.imag(expr) == 0.0)

    problem = cp.Problem(objective, constraints)
    chosen = _solve_problem(problem, solver=solver, verbose=verbose)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return SDPSolution(
            value=None,
            status=problem.status,
            variable_value=None,
            solver=chosen,
            diagnostics={
                "k": k,
                "message": "Ck local-channel subproblem infeasible or not solved.",
            },
        )

    Cval = _sym(np.asarray(C.value, dtype=complex))
    return SDPSolution(
        value=float(np.real(problem.value)),
        status=problem.status,
        variable_value=Cval,
        solver=chosen,
        diagnostics=None,
    )


# ---------------------------------------------------------------------------
# See-saw driver
# ---------------------------------------------------------------------------

# p-continuation / warmup wrapper
# ---------------------------------------------------------------------------


def make_p_warmup_grid(
    p_target: float,
    warmup_rounds: int = 5,
    schedule: str = "linear",
    power: float = 2.0,
) -> Array:
    """Return a monotone grid from 0 to ``p_target`` for continuation warmup.

    Parameters
    ----------
    p_target:
        Target physical noise strength.
    warmup_rounds:
        Number of grid points, including both 0 and p_target.
    schedule:
        - ``"linear"``: equally spaced p values.
        - ``"power"`` / ``"quadratic"``: ``p_i = p_target * t_i**power``.
          With ``power > 1`` this takes smaller steps near p=0, which is often
          useful when starting from a p=0 theory solution.
        - ``"sqrt"``: equivalent to ``power=0.5``; smaller steps near target p.
        - ``"cosine"``: smoothstep-like cosine grid, smaller steps near both
          endpoints.
    power:
        Exponent used by the power schedule.
    """
    p_target = float(p_target)
    R = int(warmup_rounds)
    if R < 2:
        raise ValueError("warmup_rounds must be at least 2")
    if p_target < 0:
        raise ValueError("p_target must be nonnegative")

    t = np.linspace(0.0, 1.0, R)
    name = str(schedule).strip().lower()
    if name in ("linear", "lin"):
        u = t
    elif name in ("power", "pow", "quadratic", "quad"):
        if float(power) <= 0:
            raise ValueError("power must be positive")
        u = t ** float(power)
    elif name in ("sqrt", "square_root"):
        u = np.sqrt(t)
    elif name in ("cosine", "cos"):
        u = 0.5 * (1.0 - np.cos(np.pi * t))
    else:
        raise ValueError("Unknown warmup schedule. Use 'linear', 'power', 'sqrt', or 'cosine'.")
    return p_target * u


def solve_QP_finite_memory_seesaw_local_channels_with_p_warmup(
    build_problem_at_p,
    p_target: float,
    warmup: bool = True,
    warmup_rounds: int = 5,
    warmup_inner_iters: int = 1,
    warmup_schedule: str = "linear",
    warmup_power: float = 2.0,
    final_max_iters: int = 10,
    Cs0: Optional[Sequence[Array]] = None,
    solve_kwargs: Optional[Dict[str, object]] = None,
    verbose_warmup: bool = False,
) -> Dict[str, object]:
    """Run p-continuation warmup, then the final local-channel see-saw solve.

    The warmup phase runs a short see-saw at each p-grid point.  The best
    strategy found at each warmup point, result_i["best_Cs"] when available, is
    passed to the next point.  The final solve is run at p_target with
    final_max_iters.

    build_problem_at_p(p_i) must return a dict with keys:
        N_channels, DN_channels, W, dims, dM

    It may also return E.  If feedback_E is not already supplied in
    solve_kwargs, E will be passed to the solver automatically.
    """
    if solve_kwargs is None:
        solve_kwargs = {}
    else:
        solve_kwargs = dict(solve_kwargs)

    p_target = float(p_target)
    Cs_current = None if Cs0 is None else [np.asarray(C, dtype=complex).copy() for C in Cs0]
    warmup_history: List[Dict[str, object]] = []

    if warmup and p_target > 0:
        p_grid = make_p_warmup_grid(
            p_target=p_target,
            warmup_rounds=warmup_rounds,
            schedule=warmup_schedule,
            power=warmup_power,
        )

        for r, p_i in enumerate(p_grid):
            if verbose_warmup:
                print(
                    f"[warmup] round {r + 1}/{len(p_grid)}, p={float(p_i):.8g}",
                    flush=True,
                )

            data_i = dict(build_problem_at_p(float(p_i)))
            kwargs_i = dict(solve_kwargs)

            if "feedback_E" not in kwargs_i and "E" in data_i:
                kwargs_i["feedback_E"] = data_i["E"]

            # Intermediate warmup points do not need expensive final polish
            # unless the user explicitly set it in solve_kwargs.
            kwargs_i.setdefault("final_unregularized_polish", False)

            result_i = solve_QP_finite_memory_seesaw_local_channels(
                N_channels=data_i["N_channels"],
                DN_channels=data_i["DN_channels"],
                W=data_i["W"],
                dims=data_i["dims"],
                dM=data_i["dM"],
                Cs0=Cs_current,
                max_iters=int(warmup_inner_iters),
                **kwargs_i,
            )

            warmup_entry = {
                "warmup_round": r,
                "p": float(p_i),
                "status": result_i.get("status"),
                "history": result_i.get("history", []),
                "best_value": result_i.get("best_value"),
                "best_iter": result_i.get("best_iter"),
            }
            warmup_history.append(warmup_entry)

            if result_i.get("status") != "ok":
                return {
                    "status": "stopped_during_warmup",
                    "p_failed": float(p_i),
                    "warmup_round_failed": r,
                    "warmup_history": warmup_history,
                    "last_result": result_i,
                }

            # IMPORTANT:
            # Pass the best strategy found at this warmup point to the next
            # p-grid point.  If best_Cs is absent, fall back to final Cs.
            Cs_next = result_i.get("best_Cs", None)
            if Cs_next is None:
                Cs_next = result_i["Cs"]

            Cs_current = [np.asarray(C, dtype=complex).copy() for C in Cs_next]

            if verbose_warmup:
                print(
                    f"[warmup] round {r + 1} done, best_value={result_i.get('best_value')}",
                    flush=True,
                )

    data_final = dict(build_problem_at_p(p_target))
    kwargs_final = dict(solve_kwargs)

    if "feedback_E" not in kwargs_final and "E" in data_final:
        kwargs_final["feedback_E"] = data_final["E"]

    kwargs_final.setdefault("final_unregularized_polish", True)

    final_result = solve_QP_finite_memory_seesaw_local_channels(
        N_channels=data_final["N_channels"],
        DN_channels=data_final["DN_channels"],
        W=data_final["W"],
        dims=data_final["dims"],
        dM=data_final["dM"],
        Cs0=Cs_current,
        max_iters=int(final_max_iters),
        **kwargs_final,
    )

    final_result["warmup_used"] = bool(warmup and p_target > 0)
    final_result["warmup_rounds"] = int(warmup_rounds)
    final_result["warmup_inner_iters"] = int(warmup_inner_iters)
    final_result["warmup_schedule"] = str(warmup_schedule)
    final_result["warmup_power"] = float(warmup_power)
    final_result["warmup_history"] = warmup_history

    # Convenience fields for downstream plotting/selection.
    final_result["warmup_last_Cs"] = Cs_current
    final_result["warmup_best_values"] = [item.get("best_value") for item in warmup_history]

    return final_result


# ---------------------------------------------------------------------------
# Random feasible initialization and depolarizing annealing
# ---------------------------------------------------------------------------


def random_density_matrix(dim: int, rng=None, eps_identity: float = 1e-10) -> Array:
    """Full-rank random density matrix of size dim x dim."""
    dim = int(dim)
    rng = np.random.default_rng() if rng is None else rng
    G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = G @ G.conj().T
    if eps_identity > 0:
        rho = rho + float(eps_identity) * np.eye(dim, dtype=complex)
    rho = _sym(rho)
    return rho / np.trace(rho)


def _partial_trace_output_input_output_order(J: Array, Din: int, Dout: int) -> Array:
    """Partial trace over output for a Choi matrix in [input, output] order."""
    J = np.asarray(J, dtype=complex)
    _require_square_matrix(J, Din * Dout, "J")
    S = np.zeros((Din, Din), dtype=complex)
    for i in range(Din):
        for j in range(Din):
            s = 0.0 + 0.0j
            for o in range(Dout):
                s += J[i * Dout + o, j * Dout + o]
            S[i, j] = s
    return _sym(S)


def random_cptp_choi_input_output(
    Din: int, Dout: int, rng=None, eps_identity: float = 1e-10
) -> Array:
    """Random full-rank CPTP Choi matrix in [input, output] ordering.

    The construction samples a full-rank positive matrix G G^† on input⊗output,
    then normalizes it as

        J <- (S^{-1/2} ⊗ I_out) J (S^{-1/2} ⊗ I_out),
        S = Tr_out J,

    which guarantees Tr_out J = I_input.
    """
    Din = int(Din)
    Dout = int(Dout)
    rng = np.random.default_rng() if rng is None else rng
    D = Din * Dout
    G = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    J = G @ G.conj().T
    if eps_identity > 0:
        J = J + float(eps_identity) * np.eye(D, dtype=complex)
    J = _sym(J)
    S = _partial_trace_output_input_output_order(J, Din, Dout)
    vals, vecs = np.linalg.eigh(S)
    if np.min(vals) <= 0:
        vals = np.maximum(vals, 1e-14)
    S_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T
    A = np.kron(S_inv_sqrt, np.eye(Dout, dtype=complex))
    Jn = _sym(A @ J @ A.conj().T)
    return Jn


def initialize_random_choi_teeth_with_memory(
    dims: Sequence[int],
    dM: int,
    seed: Optional[int] = None,
) -> List[Array]:
    """Random feasible finite-memory comb initialization.

    C1 is a full-rank random density matrix on [H1,M1].  For k>=2, Ck is a
    full-rank random CPTP Choi matrix in [input, output] ordering, where the
    input labels are [H_{2k-2}, M_{k-1}] and output labels are [H_{2k-1}, M_k].
    """
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    dims_by_label = make_dims_by_label(N, dims, dM)
    rng = np.random.default_rng(seed)
    Cs: List[Array] = []

    D1 = dim_of_labels(C_labels(1), dims_by_label)
    Cs.append(random_density_matrix(D1, rng=rng))

    for k in range(2, N + 1):
        labs = C_labels(k)
        Din = dim_of_labels(labs[:2], dims_by_label)
        Dout = dim_of_labels(labs[2:], dims_by_label)
        Cs.append(random_cptp_choi_input_output(Din, Dout, rng=rng))
    return Cs


def depolarizing_channel_choi_output_input(dout: int, din: int) -> Array:
    """CJ matrix of the completely depolarizing channel in [output,input] order.

    The map is rho -> Tr(rho) I_out / d_out.  Its Choi matrix satisfies
    Tr_out J = I_in.
    """
    dout = int(dout)
    din = int(din)
    return np.kron(np.eye(dout, dtype=complex) / dout, np.eye(din, dtype=complex))


def regularize_local_channels_with_depolarizing(
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    dims: Sequence[int],
    dM: int,
    eps: float,
) -> Tuple[List[Array], List[List[Array]]]:
    """Add artificial depolarizing noise to every local channel.

    For each local use k,
        N_k(eps) = (1-eps) N_k + eps * D_k,
    where D_k is the completely depolarizing channel Choi.  Since D_k is
    parameter-independent,
        dN_{j,k}(eps) = (1-eps) dN_{j,k}.

    This is the full-rank numerical regularization/annealing trick used in the
    tensor-network ISS paper, translated to the local-channel representation.
    """
    eps = float(eps)
    if eps < 0 or eps > 1:
        raise ValueError(f"eps must be in [0,1], got {eps}")
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    dims_by_label = make_dims_by_label(N, dims, dM)
    if len(N_channels) != N:
        raise ValueError(f"len(N_channels)={len(N_channels)}, expected N={N}")

    N_reg: List[Array] = []
    for k, Nk in enumerate(N_channels, start=1):
        labs = channel_use_labels(k)  # [H_{2k}, H_{2k-1}] = [output,input]
        dout, din = _dims_for_labels(labs, dims_by_label)
        dep = depolarizing_channel_choi_output_input(dout, din)
        _require_square_matrix(np.asarray(Nk), dout * din, f"N_channels[{k - 1}]")
        N_reg.append(_sym((1.0 - eps) * np.asarray(Nk, dtype=complex) + eps * dep))

    DN_reg: List[List[Array]] = []
    for j, row in enumerate(DN_channels):
        if len(row) != N:
            raise ValueError(f"len(DN_channels[{j}])={len(row)}, expected N={N}")
        DN_reg.append([(1.0 - eps) * np.asarray(D, dtype=complex) for D in row])
    return N_reg, DN_reg


def solve_random_X_sdp(
    rho: Array,
    drhos: Sequence[Array],
    W: Array,
    solver: Optional[str] = "CLARABEL",
    verbose: bool = False,
    x_bound: Optional[str] = None,
    seed: Optional[int] = None,
) -> SDPSolution:
    """Find a feasible X using a random bounded linear objective.

    The feasible set is the same as in solve_X_sdp.  The random objective is
    positive semidefinite, hence bounded below on X>=0.  This provides a
    randomized feasible initial X for C-then-X see-saw updates.
    """
    rho = _sym(np.asarray(rho, dtype=complex))
    drhos = [_sym(np.asarray(D, dtype=complex)) for D in drhos]
    W = np.asarray(W, dtype=complex)

    q = W.shape[0]
    if len(drhos) != q:
        raise ValueError(f"len(drhos)={len(drhos)} must equal q={q}")
    dS = rho.shape[0]
    for j, D in enumerate(drhos):
        _require_square_matrix(D, dS, f"drhos[{j}]")

    Xdim = (q + 1) * dS
    X = cp.Variable((Xdim, Xdim), hermitian=True, name="X_random_feasible")
    constraints = [X >> 0, X[0:dS, 0:dS] == np.eye(dS)]
    _add_X_bound_constraints(X=X, rho=rho, d_sys=dS, q=q, x_bound=x_bound, constraints=constraints)

    for i in range(1, q + 1):
        Bi = _B_i(i, q)
        for j in range(q):
            Cij = np.kron(Bi, drhos[j])
            expr = 0.5 * cp.trace(Cij @ X)
            constraints.append(cp.real(expr) == (1.0 if (i - 1) == j else 0.0))
            constraints.append(cp.imag(expr) == 0.0)

    rng = np.random.default_rng(seed)
    G = rng.normal(size=(Xdim, Xdim)) + 1j * rng.normal(size=(Xdim, Xdim))
    H = _sym(G @ G.conj().T)
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(H @ X))), constraints)
    chosen = _solve_problem(problem, solver=solver, verbose=verbose)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return SDPSolution(
            value=None,
            status=problem.status,
            variable_value=None,
            solver=chosen,
            diagnostics=diagnose_X_sdp_inputs(rho, drhos, W),
        )
    return SDPSolution(
        value=float(np.real(problem.value)),
        status=problem.status,
        variable_value=_sym(np.asarray(X.value, dtype=complex)),
        solver=chosen,
        diagnostics=None,
    )


def solve_QP_finite_memory_seesaw_local_channels(
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    W: Array,
    dims: Sequence[int],
    dM: Optional[int] = None,
    max_iters: int = 10,
    Cs0: Optional[Sequence[Array]] = None,
    solver: Optional[str] = "CLARABEL",
    verbose: bool = True,
    stop_tol: float = 1e-7,
    init_channel: str = "identity",
    init_seed: Optional[int] = None,
    c1_init: str = "zero",
    feedback_E: Optional[Array] = None,
    feedback_use_adjoint: bool = True,
    feedback_teeth: Optional[Sequence[int]] = None,
    x_bound: Optional[str] = None,
    num_ancilla_qubits: Optional[int] = None,
    random_initialize: bool = False,
    init_mode: Optional[str] = None,
    random_X0: bool = False,
    update_order: str = "x_then_c",
    depol_anneal: bool = False,
    depol_eps0: float = 1e-2,
    depol_decay: float = 0.5,
    depol_min_eps: float = 0.0,
    final_unregularized_polish: bool = True,
    warmup_kwargs=None,
    patience: Optional[int] = None,
) -> Dict[str, object]:
    """Alternating SDP / see-saw using local channels, with optional random init,
    depolarizing-noise annealing, and optional p-continuation warmup.

    If ``warmup_kwargs`` is supplied with ``warmup=True``, this function first
    runs a p-continuation warmup and then a final solve at the target p.  The
    warmup needs a callable problem builder, because intermediate p values
    require rebuilding ``N_channels`` and ``DN_channels``.  Therefore
    ``warmup_kwargs`` must contain:

        ``build_problem_at_p``: callable p_i -> dict containing
            N_channels, DN_channels, W, dims, dM, and optionally E.
        ``p_target``: target physical noise strength.

    All other solver options are taken from the current call.  The input
    ``N_channels``/``DN_channels`` are used for the direct solve when warmup is
    disabled, and are ignored during warmup except as ordinary positional
    arguments required by the signature.

    New options
    -----------
    random_initialize:
        If True and Cs0 is None, initialize C1 as a random full-rank density
        matrix and Ck (k>=2) as random full-rank CPTP Choi matrices.

    init_mode:
        Optional explicit initializer selector. Supported values are:
        - None: use random_initialize / init_channel as before;
        - "random": random feasible comb initialization;
        - "feedback_control" or "theory_feedback": use the previous
          informed initialization C1=theory Bell state and Ck=J(E^†⊗I_M) for
          all k>=2. This requires feedback_E.
        This option is mainly a clearer alias for the initialization choices.

    random_X0:
        If True, compute an initial feasible X using solve_random_X_sdp.  This
        is mainly useful with update_order='c_then_x'.  If False, X is obtained
        by the usual X-SDP minimization when needed.

    update_order:
        'x_then_c' keeps the old order: optimize X, then sweep C.
        'c_then_x' follows the paper's spirit: start from random C and a
        feasible random X, sweep C, then optimize X at the end of each iteration.

    depol_anneal:
        If True, iteration it uses local channels regularized as
            N_eps = (1-eps_it) N + eps_it N_depol,
            dN_eps = (1-eps_it) dN,
        with eps_it = max(depol_min_eps, depol_eps0 * depol_decay**it).
    """
    dM = resolve_memory_dimension(dM, num_ancilla_qubits)
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    validate_local_channel_data(N_channels, DN_channels, W, dims, dM)
    update_order = str(update_order).lower()
    if update_order not in ("x_then_c", "c_then_x"):
        raise ValueError("update_order must be 'x_then_c' or 'c_then_x'")

    init_mode_norm = None if init_mode is None else str(init_mode).strip().lower()
    if init_mode_norm in ("", "none", "default"):
        init_mode_norm = None
    if init_mode_norm in (
        "feedback",
        "feedback_control",
        "theory_feedback",
        "feedback_theory",
    ):
        init_mode_norm = "feedback_control"
    elif init_mode_norm in ("random", "random_cptp", "random_comb"):
        init_mode_norm = "random"
    elif init_mode_norm is not None:
        raise ValueError(
            "init_mode must be None, 'random', or 'feedback_control'/'theory_feedback'"
        )

    if Cs0 is None:
        if init_mode_norm == "random" or random_initialize:
            Cs = initialize_random_choi_teeth_with_memory(dims, dM, seed=init_seed)
            effective_init_mode = "random"
        elif init_mode_norm == "feedback_control":
            if feedback_E is None:
                raise ValueError("init_mode='feedback_control' requires feedback_E")
            Cs = initialize_choi_teeth_with_memory(
                dims,
                dM,
                pure_C1=True,
                c1_init="theory",
                channel_init="feedback_control",
                seed=init_seed,
                feedback_E=feedback_E,
                feedback_use_adjoint=feedback_use_adjoint,
                feedback_teeth=feedback_teeth,
            )
            effective_init_mode = "feedback_control"
        else:
            Cs = initialize_choi_teeth_with_memory(
                dims,
                dM,
                pure_C1=True,
                c1_init=c1_init,
                channel_init=init_channel,
                seed=init_seed,
                feedback_E=feedback_E,
                feedback_use_adjoint=feedback_use_adjoint,
                feedback_teeth=feedback_teeth,
            )
            effective_init_mode = str(init_channel)
    else:
        validate_Cs(Cs0, dims, dM)
        Cs = [np.asarray(C, dtype=complex).copy() for C in Cs0]
        # Cs0 may be supplied by the p-warmup wrapper.  In that case the
        # initialization branch above is skipped, so we still need a value
        # for logging and result diagnostics.
        effective_init_mode = "provided_Cs0"

    x_bound_norm = _normalize_x_bound(x_bound)
    history: List[Dict[str, object]] = []
    last_outer_value: Optional[float] = None
    X_val: Optional[Array] = None

    for it in range(max_iters):
        eps_it = (
            max(float(depol_min_eps), float(depol_eps0) * (float(depol_decay) ** it))
            if depol_anneal
            else 0.0
        )
        N_it, DN_it = regularize_local_channels_with_depolarizing(
            N_channels, DN_channels, dims, dM, eps_it
        )
        rec: Dict[str, object] = {
            "iter": it,
            "representation": "local_channels",
            "update_order": update_order,
            "random_initialize": bool(random_initialize),
            "init_mode": effective_init_mode,
            "random_X0": bool(random_X0),
            "depol_anneal": bool(depol_anneal),
            "depol_eps": float(eps_it),
            "num_ancilla_qubits": infer_num_ancilla_qubits_from_dM(dM),
            "dM": dM,
            "X_bound": x_bound_norm,
        }
        if verbose:
            print(
                f"---iter{it} start, eps={eps_it:.3e}, order={update_order}---",
                flush=True,
            )

        if update_order == "x_then_c":
            rho, drhos = terminal_state_from_local_channels(N_it, DN_it, Cs, dims, dM)
            xsol = solve_X_sdp(rho, drhos, W, solver=solver, verbose=verbose, x_bound=x_bound_norm)
            rec.update(
                {
                    "X_status": xsol.status,
                    "X_value": xsol.value,
                    "X_solver": xsol.solver,
                }
            )
            if xsol.variable_value is None:
                rec["X_diagnostics"] = xsol.diagnostics
                history.append(rec)
                return {
                    "status": "stopped_at_X_sdp",
                    "representation": "local_channels",
                    "Cs": Cs,
                    "X": None,
                    "rho": rho,
                    "drhos": drhos,
                    "history": history,
                    "diagnostics": xsol.diagnostics,
                }
            X_val = xsol.variable_value

        else:  # c_then_x
            if X_val is None:
                rho0, drhos0 = terminal_state_from_local_channels(N_it, DN_it, Cs, dims, dM)
                if random_X0:
                    xinit = solve_random_X_sdp(
                        rho0,
                        drhos0,
                        W,
                        solver=solver,
                        verbose=verbose,
                        x_bound=x_bound_norm,
                        seed=init_seed,
                    )
                    rec.update(
                        {
                            "X0_status": xinit.status,
                            "X0_value": xinit.value,
                            "X0_solver": xinit.solver,
                        }
                    )
                else:
                    xinit = solve_X_sdp(
                        rho0,
                        drhos0,
                        W,
                        solver=solver,
                        verbose=verbose,
                        x_bound=x_bound_norm,
                    )
                    rec.update(
                        {
                            "X0_status": xinit.status,
                            "X0_value": xinit.value,
                            "X0_solver": xinit.solver,
                        }
                    )
                if xinit.variable_value is None:
                    rec["X0_diagnostics"] = xinit.diagnostics
                    history.append(rec)
                    return {
                        "status": "stopped_at_initial_X_sdp",
                        "representation": "local_channels",
                        "Cs": Cs,
                        "X": None,
                        "rho": rho0,
                        "drhos": drhos0,
                        "history": history,
                        "diagnostics": xinit.diagnostics,
                    }
                X_val = xinit.variable_value

        # C sweep with the current X and the current regularized channels.
        for k in range(1, N + 1):
            csol = solve_Ck_sdp_local_channels(
                k,
                X_val,
                Cs,
                N_it,
                DN_it,
                W,
                dims,
                dM,
                solver=solver,
                verbose=verbose,
            )
            rec[f"C{k}_status"] = csol.status
            rec[f"C{k}_value"] = csol.value
            rec[f"C{k}_solver"] = csol.solver
            if csol.variable_value is None:
                rec[f"C{k}_diagnostics"] = csol.diagnostics
                history.append(rec)
                rho_bad, drhos_bad = terminal_state_from_local_channels(N_it, DN_it, Cs, dims, dM)
                return {
                    "status": f"stopped_at_C{k}_sdp",
                    "representation": "local_channels",
                    "Cs": Cs,
                    "X": X_val,
                    "rho": rho_bad,
                    "drhos": drhos_bad,
                    "history": history,
                    "diagnostics": csol.diagnostics,
                }
            Cs[k - 1] = csol.variable_value

        # End-of-iteration X polish for the updated C, especially important for c_then_x.
        rho_after, drhos_after = terminal_state_from_local_channels(N_it, DN_it, Cs, dims, dM)
        xpol = solve_X_sdp(
            rho_after,
            drhos_after,
            W,
            solver=solver,
            verbose=verbose,
            x_bound=x_bound_norm,
        )
        rec.update(
            {
                "X_polish_status": xpol.status,
                "X_polish_value": xpol.value,
                "X_polish_solver": xpol.solver,
            }
        )
        if xpol.variable_value is not None:
            X_val = xpol.variable_value
            obj_after = float(xpol.value)
        else:
            # Fall back to the fixed-X objective if polish fails.
            obj_after = float(np.real(np.trace(np.kron(_barW(W), rho_after) @ X_val)))
            rec["X_polish_diagnostics"] = xpol.diagnostics
        rec["objective_after_iteration"] = obj_after
        history.append(rec)
        if verbose:
            print(f"---iter{it} done, value={obj_after}---", flush=True)

        if last_outer_value is not None and abs(last_outer_value - obj_after) <= stop_tol * max(
            1.0, abs(last_outer_value)
        ):
            break
        last_outer_value = obj_after

    # Return values for the original unregularized physical channels.
    rho, drhos = terminal_state_from_local_channels(N_channels, DN_channels, Cs, dims, dM)
    final_value = None
    final_status = None
    if final_unregularized_polish:
        xfinal = solve_X_sdp(rho, drhos, W, solver=solver, verbose=verbose, x_bound=x_bound_norm)
        final_status = xfinal.status
        if xfinal.variable_value is not None:
            X_val = xfinal.variable_value
            final_value = float(xfinal.value)
        history.append(
            {
                "final_unregularized_polish": True,
                "X_status": xfinal.status,
                "X_value": xfinal.value,
                "X_solver": xfinal.solver,
            }
        )

    return {
        "status": "ok",
        "representation": "local_channels",
        "dM": dM,
        "num_ancilla_qubits": infer_num_ancilla_qubits_from_dM(dM),
        "random_initialize": bool(random_initialize),
        "init_mode": effective_init_mode,
        "random_X0": bool(random_X0),
        "depol_anneal": bool(depol_anneal),
        "update_order": update_order,
        "final_unregularized_X_status": final_status,
        "final_unregularized_X_value": final_value,
        "Cs": Cs,
        "X": X_val,
        "rho": rho,
        "drhos": drhos,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Identical-control finite-memory heuristic
# ---------------------------------------------------------------------------


def _same_control_Cs(C1: Array, C_shared: Optional[Array], N: int) -> List[Array]:
    """Build a local-teeth list with C2=...=CN=C_shared."""
    if int(N) <= 0:
        raise ValueError("N must be positive")
    out = [np.asarray(C1, dtype=complex).copy()]
    if int(N) >= 2:
        if C_shared is None:
            raise ValueError("C_shared is required when N>=2")
        out.extend([np.asarray(C_shared, dtype=complex).copy() for _ in range(int(N) - 1)])
    return out


def _initial_same_control_teeth(
    dims: Sequence[int],
    dM: int,
    Cs0: Optional[Sequence[Array]] = None,
    init_mode: Optional[str] = None,
    random_initialize: bool = False,
    init_channel: str = "identity",
    init_seed: Optional[int] = None,
    c1_init: str = "zero",
    feedback_E: Optional[Array] = None,
    feedback_use_adjoint: bool = True,
    feedback_teeth: Optional[Sequence[int]] = None,
) -> Tuple[Array, Optional[Array], str]:
    """Initialize C1 and a single shared control C.

    If Cs0 contains different controls, their average is used as the shared
    control.  This preserves CPTP feasibility because the set of Choi matrices
    of CPTP maps is convex.
    """
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2

    init_mode_norm = None if init_mode is None else str(init_mode).strip().lower()
    if init_mode_norm in ("", "none", "default"):
        init_mode_norm = None
    if init_mode_norm in (
        "feedback",
        "feedback_control",
        "theory_feedback",
        "feedback_theory",
    ):
        init_mode_norm = "feedback_control"
    elif init_mode_norm in ("random", "random_cptp", "random_comb"):
        init_mode_norm = "random"
    elif init_mode_norm is not None:
        raise ValueError(
            "init_mode must be None, 'random', or 'feedback_control'/'theory_feedback'"
        )

    if Cs0 is None:
        if init_mode_norm == "random" or random_initialize:
            Cs_init = initialize_random_choi_teeth_with_memory(dims, dM, seed=init_seed)
            effective_init_mode = "random_same_control"
        elif init_mode_norm == "feedback_control":
            if feedback_E is None:
                raise ValueError("init_mode='feedback_control' requires feedback_E")
            Cs_init = initialize_choi_teeth_with_memory(
                dims,
                dM,
                pure_C1=True,
                c1_init="theory",
                channel_init="feedback_control",
                seed=init_seed,
                feedback_E=feedback_E,
                feedback_use_adjoint=feedback_use_adjoint,
                feedback_teeth=feedback_teeth,
            )
            effective_init_mode = "feedback_control_same_control"
        else:
            Cs_init = initialize_choi_teeth_with_memory(
                dims,
                dM,
                pure_C1=True,
                c1_init=c1_init,
                channel_init=init_channel,
                seed=init_seed,
                feedback_E=feedback_E,
                feedback_use_adjoint=feedback_use_adjoint,
                feedback_teeth=feedback_teeth,
            )
            effective_init_mode = f"{init_channel}_same_control"
    else:
        validate_Cs(Cs0, dims, dM)
        Cs_init = [np.asarray(C, dtype=complex).copy() for C in Cs0]
        effective_init_mode = "provided_Cs0_same_control"

    C1 = np.asarray(Cs_init[0], dtype=complex).copy()
    if N >= 2:
        C_shared = sum(np.asarray(C, dtype=complex) for C in Cs_init[1:]) / float(N - 1)
        C_shared = _sym(C_shared)
    else:
        C_shared = None
    return C1, C_shared, effective_init_mode


def solve_QP_finite_memory_same_control_heuristic(
    N_channels: Sequence[Array],
    DN_channels: Sequence[Sequence[Array]],
    W: Array,
    dims: Sequence[int],
    dM: Optional[int] = None,
    max_iters: int = 10,
    Cs0: Optional[Sequence[Array]] = None,
    solver: Optional[str] = "CLARABEL",
    verbose: bool = True,
    stop_tol: float = 1e-7,
    init_channel: str = "identity",
    init_seed: Optional[int] = None,
    c1_init: str = "zero",
    feedback_E: Optional[Array] = None,
    feedback_use_adjoint: bool = True,
    feedback_teeth: Optional[Sequence[int]] = None,
    x_bound: Optional[str] = None,
    num_ancilla_qubits: Optional[int] = None,
    random_initialize: bool = False,
    init_mode: Optional[str] = None,
    depol_anneal: bool = False,
    depol_eps0: float = 1e-2,
    depol_decay: float = 0.5,
    depol_min_eps: float = 0.0,
    line_search_grid: Optional[Sequence[float]] = None,
    line_search_method: str = "scalar",
    line_search_bounds: Tuple[float, float] = (0.0, 1.0),
    line_search_xatol: float = 1e-2,
    line_search_maxiter: int = 20,
    line_search_tol: Optional[float] = None,
    random_control_position: bool = True,
    final_unregularized_polish: bool = True,
    patience: Optional[int] = None,
    min_best_iters: int = 0,
    return_best_as_final: bool = True,
) -> Dict[str, object]:
    """Heuristic see-saw with identical controls C2=...=CN=C.

    This routine implements a physically motivated homogeneous-control ansatz.
    The variables are C1, a single shared control C, and X.

    Exact optimization of the shared C is not an SDP because C appears at every
    control location.  Instead, each iteration uses the heuristic from the
    identical-control single-parameter code:

        1. optimize X for the current tied comb;
        2. optimize C1 exactly by the usual C1-SDP;
        3. choose one control position k in {2,...,N};
        4. optimize that one position by the usual Ck-SDP, producing C_cand;
        5. perform a convex line search
               C(lambda) = (1-lambda) C_old + lambda C_cand
           and evaluate the *true tied-control objective* by X-polish;
        6. accept the lambda with the lowest polished value.

    Since C(lambda) is a convex combination of CPTP Choi matrices, it remains
    CPTP.  The line search prevents a one-site SDP candidate from being applied
    globally if it worsens the true tied-control objective.
    """
    dM = resolve_memory_dimension(dM, num_ancilla_qubits)
    dims = _as_tuple_ints(dims)
    N = len(dims) // 2
    validate_local_channel_data(N_channels, DN_channels, W, dims, dM)
    x_bound_norm = _normalize_x_bound(x_bound)

    if line_search_grid is None:
        line_search_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    lambda_grid = [float(x) for x in line_search_grid]
    if any((x < -1e-15 or x > 1.0 + 1e-15) for x in lambda_grid):
        raise ValueError("line_search_grid entries must lie in [0, 1]")
    if 0.0 not in lambda_grid:
        lambda_grid = [0.0] + lambda_grid

    line_search_method_norm = str(line_search_method).strip().lower()
    if line_search_method_norm not in ("scalar", "minimize_scalar", "bounded", "grid"):
        raise ValueError("line_search_method must be 'scalar' or 'grid'")
    line_search_bounds = (float(line_search_bounds[0]), float(line_search_bounds[1]))
    if not (0.0 <= line_search_bounds[0] < line_search_bounds[1] <= 1.0):
        raise ValueError("line_search_bounds must satisfy 0 <= lo < hi <= 1")
    if line_search_tol is None:
        line_search_tol = stop_tol

    C1, C_shared, effective_init_mode = _initial_same_control_teeth(
        dims=dims,
        dM=dM,
        Cs0=Cs0,
        init_mode=init_mode,
        random_initialize=random_initialize,
        init_channel=init_channel,
        init_seed=init_seed,
        c1_init=c1_init,
        feedback_E=feedback_E,
        feedback_use_adjoint=feedback_use_adjoint,
        feedback_teeth=feedback_teeth,
    )

    rng = np.random.default_rng(init_seed)
    history: List[Dict[str, object]] = []
    last_outer_value: Optional[float] = None
    X_val: Optional[Array] = None

    best_value = np.inf
    best_iter: Optional[int] = None
    best_C1: Optional[Array] = None
    best_C_shared: Optional[Array] = None
    best_Cs: Optional[List[Array]] = None
    best_X: Optional[Array] = None
    best_rho: Optional[Array] = None
    best_drhos: Optional[List[Array]] = None
    bad_rounds = 0
    stopped_by_patience = False

    latest_value: Optional[float] = None
    latest_iter: Optional[int] = None
    latest_C1: Optional[Array] = None
    latest_C_shared: Optional[Array] = None
    latest_Cs: Optional[List[Array]] = None
    latest_X: Optional[Array] = None
    latest_rho: Optional[Array] = None
    latest_drhos: Optional[List[Array]] = None

    for it in range(int(max_iters)):
        eps_it = (
            max(float(depol_min_eps), float(depol_eps0) * (float(depol_decay) ** it))
            if depol_anneal
            else 0.0
        )
        N_it, DN_it = regularize_local_channels_with_depolarizing(
            N_channels, DN_channels, dims, dM, eps_it
        )
        Cs_tied = _same_control_Cs(C1, C_shared, N)

        rec: Dict[str, object] = {
            "iter": it,
            "representation": "local_channels_same_control",
            "same_control": True,
            "init_mode": effective_init_mode,
            "depol_anneal": bool(depol_anneal),
            "depol_eps": float(eps_it),
            "num_ancilla_qubits": infer_num_ancilla_qubits_from_dM(dM),
            "dM": dM,
            "X_bound": x_bound_norm,
        }
        if verbose:
            print(f"---same-control iter{it} start, eps={eps_it:.3e}---", flush=True)

        # X step for current tied comb.
        rho, drhos = terminal_state_from_local_channels(N_it, DN_it, Cs_tied, dims, dM)
        xsol = solve_X_sdp(rho, drhos, W, solver=solver, verbose=verbose, x_bound=x_bound_norm)
        rec.update({"X_status": xsol.status, "X_value": xsol.value, "X_solver": xsol.solver})
        if xsol.variable_value is None:
            rec["X_diagnostics"] = xsol.diagnostics
            history.append(rec)
            return {
                "status": "stopped_at_X_sdp",
                "representation": "local_channels_same_control",
                "Cs": Cs_tied,
                "C1": C1,
                "C_shared": C_shared,
                "X": None,
                "rho": rho,
                "drhos": drhos,
                "history": history,
                "diagnostics": xsol.diagnostics,
                "best_value": None if not np.isfinite(best_value) else best_value,
                "best_iter": best_iter,
                "best_Cs": best_Cs,
                "best_X": best_X,
                "best_rho": best_rho,
                "best_drhos": best_drhos,
            }
        X_val = xsol.variable_value

        # Exact C1 SDP update.
        c1sol = solve_Ck_sdp_local_channels(
            1,
            X_val,
            Cs_tied,
            N_it,
            DN_it,
            W,
            dims,
            dM,
            solver=solver,
            verbose=verbose,
        )
        rec.update(
            {
                "C1_status": c1sol.status,
                "C1_value": c1sol.value,
                "C1_solver": c1sol.solver,
            }
        )
        if c1sol.variable_value is None:
            rec["C1_diagnostics"] = c1sol.diagnostics
            history.append(rec)
            return {
                "status": "stopped_at_C1_sdp",
                "representation": "local_channels_same_control",
                "Cs": Cs_tied,
                "C1": C1,
                "C_shared": C_shared,
                "X": X_val,
                "rho": rho,
                "drhos": drhos,
                "history": history,
                "diagnostics": c1sol.diagnostics,
            }
        C1 = c1sol.variable_value

        # Shared-control candidate from a randomly selected one-site Ck SDP.
        Cs_after_C1 = _same_control_Cs(C1, C_shared, N)
        control_candidate_status = None
        control_candidate_value = None
        selected_k = None
        C_candidate = None

        if N >= 2:
            if random_control_position:
                selected_k = int(rng.integers(2, N + 1))
            else:
                selected_k = 2 + (it % (N - 1))
            csol = solve_Ck_sdp_local_channels(
                selected_k,
                X_val,
                Cs_after_C1,
                N_it,
                DN_it,
                W,
                dims,
                dM,
                solver=solver,
                verbose=verbose,
            )
            control_candidate_status = csol.status
            control_candidate_value = csol.value
            rec.update(
                {
                    "selected_control_k": selected_k,
                    "C_shared_candidate_status": csol.status,
                    "C_shared_candidate_value": csol.value,
                    "C_shared_candidate_solver": csol.solver,
                }
            )
            if csol.variable_value is not None:
                C_candidate = csol.variable_value
            else:
                rec["C_shared_candidate_diagnostics"] = csol.diagnostics

        # Line search on the true tied-control objective with X-polish.
        #
        # A one-site SDP gives C_candidate.  Applying it to every control slot
        # can worsen the real tied-control objective, so we optimize the scalar
        # mixing parameter in
        #
        #     C(lambda) = (1-lambda) C_old + lambda C_candidate,
        #
        # and evaluate each lambda by rebuilding the true tied comb and solving
        # the X-SDP. line_search_method="scalar" uses
        # scipy.optimize.minimize_scalar; line_search_method="grid" uses the
        # supplied discrete grid.
        line_records: List[Dict[str, object]] = []
        C_old = C_shared.copy() if C_shared is not None else None
        C_cand = C_candidate.copy() if C_candidate is not None else None
        candidate_available = (N < 2) or (C_cand is not None)

        def _control_lambda(lam: float) -> Optional[Array]:
            if N < 2:
                return None
            lam = float(lam)
            w = np.sin(np.pi * lam) ** 2
            return (1.0 - w) * C_old + w * C_cand

        def _evaluate_lambda(lam: float) -> float:
            lam = float(lam)
            if not candidate_available:
                return np.inf
            if lam < line_search_bounds[0] - 1e-12 or lam > line_search_bounds[1] + 1e-12:
                return np.inf

            try:
                C_lam = _control_lambda(lam)
                Cs_lam = _same_control_Cs(C1, C_lam, N)
                rho_lam, drhos_lam = terminal_state_from_local_channels(
                    N_it, DN_it, Cs_lam, dims, dM
                )
                xlam = solve_X_sdp(
                    rho_lam,
                    drhos_lam,
                    W,
                    solver=solver,
                    verbose=False,
                    x_bound=x_bound_norm,
                )
                val_lam = None if xlam.value is None else float(np.real(xlam.value))
                line_records.append(
                    {
                        "lambda": lam,
                        "X_status": xlam.status,
                        "X_value": val_lam,
                        "X_solver": xlam.solver,
                    }
                )
                if xlam.variable_value is None or val_lam is None or not np.isfinite(val_lam):
                    return np.inf
                return float(val_lam)
            except Exception as err:
                line_records.append(
                    {
                        "lambda": lam,
                        "X_status": f"exception: {type(err).__name__}",
                        "X_value": None,
                        "X_solver": None,
                    }
                )
                return np.inf

        if candidate_available:
            candidates: List[Tuple[float, float]] = []

            # Always evaluate endpoints.  scipy's bounded search may not do so.
            for lam0 in (0.0, 1.0):
                if line_search_bounds[0] - 1e-12 <= lam0 <= line_search_bounds[1] + 1e-12:
                    candidates.append((_evaluate_lambda(lam0), float(lam0)))

            if line_search_method_norm in ("scalar", "minimize_scalar", "bounded") and N >= 2:
                try:
                    res_scalar = scipy.optimize.minimize_scalar(
                        _evaluate_lambda,
                        bounds=line_search_bounds,
                        method="bounded",
                        options={
                            "xatol": float(line_search_xatol),
                            "maxiter": int(line_search_maxiter),
                        },
                    )
                    if res_scalar.fun is not None and np.isfinite(res_scalar.fun):
                        candidates.append((float(res_scalar.fun), float(res_scalar.x)))
                    rec["line_search_scalar_success"] = bool(res_scalar.success)
                    rec["line_search_scalar_message"] = str(res_scalar.message)
                    rec["line_search_scalar_nfev"] = int(getattr(res_scalar, "nfev", -1))
                except Exception as err:
                    rec["line_search_scalar_success"] = False
                    rec["line_search_scalar_message"] = f"exception: {type(err).__name__}: {err}"
            else:
                for lam in lambda_grid:
                    candidates.append((_evaluate_lambda(float(lam)), float(lam)))

            candidates = [(val, lam) for val, lam in candidates if np.isfinite(val)]
            if candidates:
                proposed_value, proposed_lambda = min(candidates, key=lambda x: x[0])
            else:
                proposed_value, proposed_lambda = np.inf, 0.0
        else:
            proposed_value, proposed_lambda = np.inf, 0.0

        # Accept the proposed lambda only if it improves over the no-update endpoint.
        # Since lambda=0 was explicitly evaluated, this is usually redundant, but it
        # protects against optimizer noise and failed endpoint solves.
        current_line_value = None
        for item in line_records:
            if (
                abs(float(item.get("lambda", np.nan)) - 0.0) <= 1e-12
                and item.get("X_value") is not None
            ):
                current_line_value = float(item["X_value"])
                break
        if current_line_value is None or not np.isfinite(current_line_value):
            current_line_value = proposed_value if np.isfinite(proposed_value) else np.inf

        accept_tol = float(line_search_tol) * max(1.0, abs(current_line_value))
        if np.isfinite(proposed_value) and proposed_value < current_line_value - accept_tol:
            accepted_lambda = float(proposed_lambda)
        else:
            accepted_lambda = 0.0

        # Recompute accepted lambda once, keeping all returned objects consistent.
        C_accepted = _control_lambda(accepted_lambda)
        Cs_accepted = _same_control_Cs(C1, C_accepted, N)
        rho_acc, drhos_acc = terminal_state_from_local_channels(N_it, DN_it, Cs_accepted, dims, dM)
        xacc = solve_X_sdp(
            rho_acc,
            drhos_acc,
            W,
            solver=solver,
            verbose=verbose,
            x_bound=x_bound_norm,
        )
        if xacc.variable_value is None or xacc.value is None:
            rec["line_search"] = line_records
            rec["line_search_failed_after_accept"] = True
            history.append(rec)
            return {
                "status": "stopped_at_same_control_line_search",
                "representation": "local_channels_same_control",
                "Cs": Cs_after_C1,
                "C1": C1,
                "C_shared": C_shared,
                "X": X_val,
                "rho": rho,
                "drhos": drhos,
                "history": history,
                "diagnostics": {"message": "Accepted lambda did not yield a feasible X."},
            }

        if N >= 2:
            C_shared = C_accepted
        X_val = xacc.variable_value.copy()
        obj_after = float(np.real(xacc.value))
        rho_after = rho_acc.copy()
        drhos_after = [D.copy() for D in drhos_acc]
        Cs_after = _same_control_Cs(C1, C_shared, N)

        rec["line_search"] = line_records
        rec["line_search_method"] = str(line_search_method)
        rec["line_search_bounds"] = tuple(line_search_bounds)
        rec["line_search_xatol"] = float(line_search_xatol)
        rec["line_search_maxiter"] = int(line_search_maxiter)
        rec["proposed_lambda"] = float(proposed_lambda)
        rec["proposed_line_value"] = (
            None if not np.isfinite(proposed_value) else float(proposed_value)
        )
        rec["accepted_lambda"] = float(accepted_lambda)
        rec["objective_after_iteration"] = obj_after
        rec["X_polish_status"] = xacc.status
        rec["X_polish_value"] = obj_after
        rec["X_polish_solver"] = xacc.solver
        rec["control_candidate_status"] = control_candidate_status
        rec["control_candidate_value"] = control_candidate_value

        latest_value = obj_after
        latest_iter = it
        latest_C1 = C1.copy()
        latest_C_shared = C_shared.copy() if C_shared is not None else None
        latest_Cs = [C.copy() for C in Cs_after]
        latest_X = X_val.copy() if X_val is not None else None
        latest_rho = rho_after.copy()
        latest_drhos = [D.copy() for D in drhos_after]

        best_tracking_active = (it + 1) >= int(min_best_iters)
        if best_tracking_active and np.isfinite(obj_after):
            improve_tol = (
                stop_tol * max(1.0, abs(best_value)) if np.isfinite(best_value) else stop_tol
            )
            if obj_after < best_value - improve_tol:
                best_value = float(obj_after)
                best_iter = it
                best_C1 = C1.copy()
                best_C_shared = C_shared.copy() if C_shared is not None else None
                best_Cs = [C.copy() for C in Cs_after]
                best_X = X_val.copy() if X_val is not None else None
                best_rho = rho_after.copy()
                best_drhos = [D.copy() for D in drhos_after]
                bad_rounds = 0
            else:
                bad_rounds += 1

        rec["best_tracking_active"] = bool(best_tracking_active)
        rec["min_best_iters"] = int(min_best_iters)
        rec["best_value_so_far"] = None if not np.isfinite(best_value) else best_value
        rec["best_iter_so_far"] = best_iter
        rec["bad_rounds"] = bad_rounds
        history.append(rec)

        if verbose:
            print(
                f"---same-control iter{it} done, value={obj_after}, "
                f"lambda={accepted_lambda}, best={None if not np.isfinite(best_value) else best_value} "
                f"at iter={best_iter}, bad_rounds={bad_rounds}---",
                flush=True,
            )

        if best_tracking_active and patience is not None and bad_rounds >= patience:
            stopped_by_patience = True
            if verbose:
                print(
                    f"---same-control early stop: no best improvement for {bad_rounds} rounds; "
                    f"use best value={best_value} at iter={best_iter}---",
                    flush=True,
                )
            break

        if last_outer_value is not None and abs(last_outer_value - obj_after) <= stop_tol * max(
            1.0, abs(last_outer_value)
        ):
            break
        last_outer_value = obj_after

    # If best tracking never activated, use latest valid polished iterate.
    if best_Cs is None and latest_Cs is not None:
        best_value = float(latest_value) if latest_value is not None else np.inf
        best_iter = latest_iter
        best_C1 = latest_C1.copy() if latest_C1 is not None else None
        best_C_shared = latest_C_shared.copy() if latest_C_shared is not None else None
        best_Cs = [C.copy() for C in latest_Cs]
        best_X = latest_X.copy() if latest_X is not None else None
        best_rho = latest_rho.copy() if latest_rho is not None else None
        best_drhos = [D.copy() for D in latest_drhos] if latest_drhos is not None else None

    if return_best_as_final and best_Cs is not None:
        C1 = best_C1.copy() if best_C1 is not None else best_Cs[0].copy()
        C_shared = (
            best_C_shared.copy()
            if best_C_shared is not None
            else (best_Cs[1].copy() if N >= 2 else None)
        )
        X_val = best_X.copy() if best_X is not None else X_val
        Cs_final = [C.copy() for C in best_Cs]
    else:
        Cs_final = _same_control_Cs(C1, C_shared, N)

    # Final unregularized X-polish on the selected final tied-control strategy.
    rho_final, drhos_final = terminal_state_from_local_channels(
        N_channels, DN_channels, Cs_final, dims, dM
    )
    final_value = None
    final_status = None
    if final_unregularized_polish:
        xfinal = solve_X_sdp(
            rho_final,
            drhos_final,
            W,
            solver=solver,
            verbose=verbose,
            x_bound=x_bound_norm,
        )
        final_status = xfinal.status
        if xfinal.variable_value is not None:
            X_val = xfinal.variable_value.copy()
            final_value = float(xfinal.value)
        history.append(
            {
                "final_unregularized_polish": True,
                "uses_best_Cs": bool(return_best_as_final and best_Cs is not None),
                "X_status": xfinal.status,
                "X_value": xfinal.value,
                "X_solver": xfinal.solver,
            }
        )

    return {
        "status": "ok",
        "representation": "local_channels_same_control",
        "same_control": True,
        "dM": dM,
        "num_ancilla_qubits": infer_num_ancilla_qubits_from_dM(dM),
        "init_mode": effective_init_mode,
        "depol_anneal": bool(depol_anneal),
        "final_unregularized_X_status": final_status,
        "final_unregularized_X_value": final_value,
        "Cs": Cs_final,
        "C1": Cs_final[0],
        "C_shared": Cs_final[1] if N >= 2 else None,
        "X": X_val,
        "rho": rho_final,
        "drhos": drhos_final,
        "history": history,
        "best_value": None if not np.isfinite(best_value) else best_value,
        "best_iter": best_iter,
        "best_Cs": best_Cs,
        "best_C1": best_C1,
        "best_C_shared": best_C_shared,
        "best_X": best_X,
        "best_rho": best_rho,
        "best_drhos": best_drhos,
        "stopped_by_patience": stopped_by_patience,
        "patience": patience,
        "min_best_iters": int(min_best_iters),
        "line_search_grid": list(lambda_grid),
        "line_search_method": str(line_search_method),
        "line_search_bounds": tuple(line_search_bounds),
        "line_search_xatol": float(line_search_xatol),
        "line_search_maxiter": int(line_search_maxiter),
        "random_control_position": bool(random_control_position),
        "return_best_as_final": bool(return_best_as_final),
    }


def solve_QP_finite_memory_same_control_with_p_warmup(
    build_problem_at_p,
    p_target: float,
    warmup: bool = True,
    warmup_rounds: int = 5,
    warmup_inner_iters: int = 1,
    warmup_schedule: str = "linear",
    warmup_power: float = 2.0,
    final_max_iters: int = 10,
    Cs0: Optional[Sequence[Array]] = None,
    solve_kwargs: Optional[Dict[str, object]] = None,
    verbose_warmup: bool = False,
) -> Dict[str, object]:
    """Run p-continuation warmup for the identical-control heuristic.

    This wrapper is for the homogeneous-control ansatz

        C2 = C3 = ... = CN = C_shared.

    Parameters
    ----------
    build_problem_at_p:
        Callable. ``build_problem_at_p(p_i)`` must return a dict with keys

            N_channels, DN_channels, W, dims, dM

        It may also return ``E``. If ``feedback_E`` is not already supplied in
        ``solve_kwargs``, then ``E`` is automatically passed to the solver.

    p_target:
        Target physical noise strength.

    warmup:
        Whether to perform p-continuation warmup.

    warmup_rounds:
        Number of warmup p-grid points, including 0 and p_target.

    warmup_inner_iters:
        Number of same-control see-saw iterations at each warmup p point.

    warmup_schedule:
        Schedule name used by ``make_p_warmup_grid``. Supported examples are
        ``"linear"``, ``"power"``, ``"sqrt"``, and ``"cosine"``.

    warmup_power:
        Power used when ``warmup_schedule="power"``.

    final_max_iters:
        Number of iterations for the final optimization at p_target.

    Cs0:
        Optional initial tied comb. If provided, it should have length N:
            [C1, C_shared, ..., C_shared]
        The wrapper passes it directly to the first warmup solve.

    solve_kwargs:
        Keyword arguments passed to
        ``solve_QP_finite_memory_same_control_heuristic``.

    verbose_warmup:
        Whether to print warmup progress.

    Returns
    -------
    Dict[str, object]
        The final result returned by
        ``solve_QP_finite_memory_same_control_heuristic``, augmented with
        warmup metadata.
    """
    if solve_kwargs is None:
        solve_kwargs = {}
    else:
        solve_kwargs = dict(solve_kwargs)

    p_target = float(p_target)

    Cs_current = None if Cs0 is None else [np.asarray(C, dtype=complex).copy() for C in Cs0]

    warmup_history: List[Dict[str, object]] = []

    if warmup and p_target > 0:
        p_grid = make_p_warmup_grid(
            p_target=p_target,
            warmup_rounds=warmup_rounds,
            schedule=warmup_schedule,
            power=warmup_power,
        )

        for r, p_i in enumerate(p_grid):
            p_i = float(p_i)

            if verbose_warmup:
                print(
                    f"[same-control warmup] round {r + 1}/{len(p_grid)}, p={p_i:.8g}",
                    flush=True,
                )

            data_i = dict(build_problem_at_p(p_i))
            kwargs_i = dict(solve_kwargs)

            # If build_problem_at_p returns the noiseless feedback unitary E,
            # use it unless the user explicitly supplied feedback_E.
            if "feedback_E" not in kwargs_i and "E" in data_i:
                kwargs_i["feedback_E"] = data_i["E"]

            # Intermediate warmup rounds usually do not need expensive final
            # unregularized polish unless the user explicitly requested it.
            kwargs_i.setdefault("final_unregularized_polish", False)

            result_i = solve_QP_finite_memory_same_control_heuristic(
                N_channels=data_i["N_channels"],
                DN_channels=data_i["DN_channels"],
                W=data_i["W"],
                dims=data_i["dims"],
                dM=data_i["dM"],
                Cs0=Cs_current,
                max_iters=int(warmup_inner_iters),
                **kwargs_i,
            )

            warmup_entry = {
                "warmup_round": r,
                "p": p_i,
                "status": result_i.get("status"),
                "history": result_i.get("history", []),
                "best_value": result_i.get("best_value"),
                "best_iter": result_i.get("best_iter"),
                "best_C1": result_i.get("best_C1"),
                "best_C_shared": result_i.get("best_C_shared"),
                "best_Cs": result_i.get("best_Cs"),
                "C1": result_i.get("C1"),
                "C_shared": result_i.get("C_shared"),
            }
            warmup_history.append(warmup_entry)

            if result_i.get("status") != "ok":
                return {
                    "status": "stopped_during_same_control_warmup",
                    "p_failed": p_i,
                    "warmup_round_failed": r,
                    "warmup_history": warmup_history,
                    "last_result": result_i,
                }

            # Pass the final returned tied comb to the next p-grid point.
            # If return_best_as_final=True and patience was triggered inside
            # the same-control solver, this is already the best-so-far tied comb.
            Cs_current = [np.asarray(C, dtype=complex).copy() for C in result_i["Cs"]]

            if verbose_warmup:
                print(
                    f"[same-control warmup] round {r + 1} done, "
                    f"best_value={result_i.get('best_value')}, "
                    f"final_value={result_i.get('final_unregularized_X_value', None)}",
                    flush=True,
                )

    # Final solve at p_target.
    data_final = dict(build_problem_at_p(p_target))
    kwargs_final = dict(solve_kwargs)

    if "feedback_E" not in kwargs_final and "E" in data_final:
        kwargs_final["feedback_E"] = data_final["E"]

    kwargs_final.setdefault("final_unregularized_polish", True)

    final_result = solve_QP_finite_memory_same_control_heuristic(
        N_channels=data_final["N_channels"],
        DN_channels=data_final["DN_channels"],
        W=data_final["W"],
        dims=data_final["dims"],
        dM=data_final["dM"],
        Cs0=Cs_current,
        max_iters=int(final_max_iters),
        **kwargs_final,
    )

    final_result["warmup_used"] = bool(warmup and p_target > 0)
    final_result["warmup_type"] = "same_control"
    final_result["warmup_rounds"] = int(warmup_rounds)
    final_result["warmup_inner_iters"] = int(warmup_inner_iters)
    final_result["warmup_schedule"] = str(warmup_schedule)
    final_result["warmup_power"] = float(warmup_power)
    final_result["warmup_history"] = warmup_history
    final_result["warmup_last_Cs"] = Cs_current
    final_result["warmup_best_values"] = [item.get("best_value") for item in warmup_history]
    final_result["warmup_best_Cs"] = [item.get("best_Cs") for item in warmup_history]

    return final_result


# Concise public aliases. The original names remain available for compatibility
# with existing research scripts.
solve_finite_memory = solve_QP_finite_memory_seesaw_local_channels
solve_finite_memory_with_warmup = solve_QP_finite_memory_seesaw_local_channels_with_p_warmup
solve_same_control_finite_memory = solve_QP_finite_memory_same_control_heuristic
solve_same_control_finite_memory_with_warmup = solve_QP_finite_memory_same_control_with_p_warmup


__all__ = [
    "SDPSolution",
    "ancilla_qubits_to_dM",
    "infer_num_ancilla_qubits_from_dM",
    "initialize_choi_teeth_with_memory",
    "initialize_random_choi_teeth_with_memory",
    "make_p_warmup_grid",
    "resolve_memory_dimension",
    "solve_Ck_sdp_local_channels",
    "solve_X_sdp",
    "solve_finite_memory",
    "solve_finite_memory_with_warmup",
    "solve_same_control_finite_memory",
    "solve_same_control_finite_memory_with_warmup",
    "terminal_state_from_local_channels",
]
