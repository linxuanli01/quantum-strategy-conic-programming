"""Shared CVXPY constraints for the supported precision bounds."""

from typing import Optional

import cvxpy as cp
import numpy as np


def normalize_bound_name(bound: Optional[str], *, allow_none: bool = False) -> str:
    """Return the canonical lower-case name of a precision bound."""
    if bound is None:
        if allow_none:
            return "none"
        raise ValueError("a bound name is required")
    normalized = str(bound).strip().lower()
    aliases = {
        "nh": "nhb",
        "nhb": "nhb",
        "sld": "sld",
        "holevo": "holevo",
    }
    if allow_none:
        aliases.update({"": "none", "no": "none", "none": "none", "default": "none"})
    if normalized not in aliases:
        allowed = "none, nhb, sld, or holevo" if allow_none else "nhb, sld, or holevo"
        raise ValueError(f"bound must be {allowed}")
    return aliases[normalized]


def add_nhb_constraints(Y_real, Y_imag, d_sys: int, d: int, constraints) -> None:
    """Append the block-symmetry constraints defining the NH cone."""
    for j in range(d + 1):
        for k in range(d + 1):
            block_jk_real = Y_real[
                j * d_sys : (j + 1) * d_sys,
                k * d_sys : (k + 1) * d_sys,
            ]
            block_jk_imag = Y_imag[
                j * d_sys : (j + 1) * d_sys,
                k * d_sys : (k + 1) * d_sys,
            ]
            block_kj_real = Y_real[
                k * d_sys : (k + 1) * d_sys,
                j * d_sys : (j + 1) * d_sys,
            ]
            block_kj_imag = Y_imag[
                k * d_sys : (k + 1) * d_sys,
                j * d_sys : (j + 1) * d_sys,
            ]
            constraints += [
                block_jk_real == block_jk_real.T,
                block_jk_imag == -block_jk_imag.T,
                block_jk_real == block_kj_real,
                block_jk_imag == block_kj_imag,
            ]


def add_sld_constraints(Y_real, Y_imag, d_block: int, d: int, constraints) -> None:
    """Append the block constraints defining the SLD cone."""
    for j in range(d + 1):
        block_j0_real = Y_real[j * d_block : (j + 1) * d_block, 0:d_block]
        block_j0_imag = Y_imag[j * d_block : (j + 1) * d_block, 0:d_block]
        constraints += [
            block_j0_real == block_j0_real.T,
            block_j0_imag == -block_j0_imag.T,
        ]
        for k in range(d + 1):
            block_jk_real = Y_real[
                j * d_block : (j + 1) * d_block,
                k * d_block : (k + 1) * d_block,
            ]
            block_jk_imag = Y_imag[
                j * d_block : (j + 1) * d_block,
                k * d_block : (k + 1) * d_block,
            ]
            block_kj_real = Y_real[
                k * d_block : (k + 1) * d_block,
                j * d_block : (j + 1) * d_block,
            ]
            block_kj_imag = Y_imag[
                k * d_block : (k + 1) * d_block,
                j * d_block : (j + 1) * d_block,
            ]
            constraints += [
                block_jk_real == block_kj_real.T,
                block_jk_imag == -block_kj_imag.T,
            ]


def add_holevo_constraints(Y_real, Y_imag, state, d_block: int, d: int, constraints) -> None:
    """Append the weak-commutativity constraints defining the Holevo cone."""
    state = np.asarray(state, dtype=complex)
    if state.shape != (d_block, d_block):
        raise ValueError(f"state must have shape {(d_block, d_block)}, got {state.shape}")
    Y = Y_real + 1j * Y_imag
    reference_basis = np.eye(d + 1, dtype=complex)
    for j in range(1, d + 1):
        for k in range(1, d + 1):
            reference_operator = np.outer(reference_basis[:, j], reference_basis[:, k].conj())
            constraints.append(cp.imag(cp.trace(Y @ np.kron(reference_operator, state))) == 0)
