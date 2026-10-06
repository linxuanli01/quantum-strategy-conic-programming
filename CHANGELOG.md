# Changelog

## 0.2.0 — 2026-10-06

- Added finite-memory sequential optimization with fixed memory dimensions 2
  and 4.
- Added independent-control and homogeneous-control see-saw algorithms.
- Added noise-parameter continuation, depolarizing annealing, feedback
  initialization, and best-so-far tracking.
- Shared the NHB, SLD, and Holevo cone constraints between the unrestricted and
  finite-memory solvers.
- Centralized CVXPY solver selection and fallback handling.
- Added finite-memory documentation, validation tests, Ruff checks, and package
  exports.

## 0.1.0 — 2026-10-05

- Initial packaged release of the unrestricted and energy-constrained conic
  programs.
