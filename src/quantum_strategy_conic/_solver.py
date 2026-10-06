"""Shared CVXPY solver selection and fallback handling."""

import warnings
from typing import Optional, Sequence

import cvxpy as cp
from cvxpy.error import SolverError


def solve_problem(
    problem: cp.Problem,
    solver: Optional[str] = None,
    *,
    verbose: bool = False,
    mosek_params=None,
    scs_eps: float = 1e-6,
    scs_max_iters: int = 20_000,
    fallback_solvers: Sequence[str] = ("CLARABEL", "SCS"),
    warn_on_fallback: bool = False,
) -> str:
    """Solve a CVXPY problem and return the solver that completed the call."""
    installed = set(cp.installed_solvers())
    preferred = solver or ("MOSEK" if "MOSEK" in installed else None)
    candidates = []
    for candidate in (preferred, *fallback_solvers):
        if candidate and candidate in installed and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            "No requested conic solver is installed. Install MOSEK, CLARABEL, or SCS."
        )

    last_error = None
    for candidate in candidates:
        kwargs = {"solver": candidate, "verbose": verbose, "warm_start": True}
        if candidate == "MOSEK":
            kwargs["accept_unknown"] = True
            kwargs["mosek_params"] = mosek_params or {
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 100,
            }
        elif candidate == "SCS":
            kwargs.update({"eps": scs_eps, "max_iters": scs_max_iters})
        try:
            problem.solve(**kwargs)
            if warn_on_fallback and preferred and candidate != preferred:
                warnings.warn(
                    f"{preferred} was unavailable or failed; used {candidate} instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return candidate
        except SolverError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise RuntimeError("No conic solver candidate could be used")
