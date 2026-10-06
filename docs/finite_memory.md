# Finite-memory sequential optimization

The finite-memory module optimizes a sequential strategy whose ancillary
memory has fixed dimension `dM`. It uses a local-channel tensor-network
representation and never forms the dense global channel comb, which is the
feature that allows the method to reach more channel uses than the unrestricted
single-SDP formulation.

## Data convention

For `N` channel uses,

```text
dims = (d_2N, d_2N-1, ..., d_1)
```

and the inputs are

- `N_channels[k]`: Choi matrix of channel use `k`;
- `DN_channels[j][k]`: derivative of use `k` with respect to parameter `j`;
- `W`: parameter-weight matrix;
- `dM`: ancillary-memory dimension, currently `2` or `4`.

All Choi matrices use row-major vectorization and local channel ordering
`[output, input]`, consistently with `comb.py`.

## Algorithms

`solve_finite_memory` alternates between two convex subproblems:

1. optimize the lifted measurement variable `X` for fixed control teeth;
2. optimize each preparation/control Choi matrix `C_k` for fixed `X` and all
   other teeth.

The routine supports `nhb`, `sld`, and `holevo` through the same shared bound
constraints used by `multipara_bounds.py`. Optional depolarizing annealing,
random initialization, feedback initialization, and noise-parameter
continuation are available for numerical stabilization.

`solve_same_control_finite_memory` imposes

```text
C_2 = C_3 = ... = C_N.
```

The repeated appearance of the shared control makes its exact subproblem
nonconvex. The implementation therefore optimizes one occurrence, constructs a
candidate control, and performs a line search using the true tied-control
objective before accepting the update. This is a heuristic feasible-strategy
optimization, not a global optimality certificate.

## Public API

- `solve_finite_memory`
- `solve_finite_memory_with_warmup`
- `solve_same_control_finite_memory`
- `solve_same_control_finite_memory_with_warmup`
- `initialize_choi_teeth_with_memory`
- `solve_X_sdp`
- `solve_Ck_sdp_local_channels`

The longer `solve_QP_...` names from the original research code remain
available for backward compatibility, but the concise aliases are recommended
for new code.

## Interpretation of results

The returned dictionary contains the final feasible control teeth in `Cs`, the
lifted measurement variable in `X`, the terminal state and derivatives, and a
per-iteration `history`. When best-so-far tracking is enabled, use `best_Cs`
and `best_value`. With `final_unregularized_polish=True`, the field
`final_unregularized_X_value` reports the final bound evaluated on the physical,
unregularized channels.

MOSEK is recommended for publication calculations. CLARABEL and SCS are useful
for testing, but may report `optimal_inaccurate` for poorly conditioned
multiparameter instances.
