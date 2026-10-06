# Energy-constraint implementation

This note records how the global-battery constraint is expressed using the
strategy variables already present in `multipara_bounds.py`. It is intended to
make the subsystem ordering and branch normalization auditable.

## Convention

For two channel uses, the original solver orders the channel spaces as

```text
(H4, H3, H2, H1),
```

where `H1` and `H3` are channel-input spaces and `H2` and `H4` are
channel-output spaces. Every space is a qubit. The single-qubit Hamiltonian is
denoted by `H` and defaults to `|1><1|`.

The condition

```text
lambda_max(O_n) <= E
```

is implemented as the affine semidefinite constraint

```text
O_n <= E I.
```

## Sequential strategy

`P1` is the initial state on `H1`. `P2` is the two-step strategy operator on
`H3 tensor H2 tensor H1`. The two cumulative constraints are

```text
Tr(H P1) <= E,

Tr_{H3,H1}[P2 (H3 + H1)] - H2^T <= E I2.
```

Here `H3 + H1` abbreviates
`H tensor I tensor I + I tensor I tensor H` in the ordering of `P2`.

## Parallel strategy

The recovered probe state is on `H3 tensor H1`. With
`H_in = H tensor I + I tensor H`, the constraint is

```text
Tr(H_in rho) <= E.
```

## Causal superposition

The two branches are subnormalized. Their weights are

```text
q1 = Tr(P11),
q2 = Tr(P12),
q1 + q2 = 1.
```

The first branch uses `P21` in the ordering `H3 tensor H2 tensor H1`. The
second uses `P22` in the ordering `H4 tensor H3 tensor H1`. Each branch is
constrained conditionally on its weight. For a subnormalized branch `q P`, the
homogeneous form is

```text
Tr[q P (H_inputs)] - q H_previous_output^T <= q E I.
```

Both the previous-output Hamiltonian and the right-hand side must be scaled by
`q`. Scaling only `E` would not be equivalent to imposing the budget on the
normalized branch `P` and would behave incorrectly as `q` approaches zero.

## Quantum switch

There is no freely optimized intermediate control in the switch class used by
the original solver. `P1` and `P3` are the two subnormalized initial states,
so the constraints are

```text
Tr(H P1) <= E Tr(P1),
Tr(H P3) <= E Tr(P3).
```

## General indefinite causal order

No energy constraint is added to `ico`. The manuscript specifies
branch-resolved global-battery conditions for definite orders, causal
superpositions, and the switch, but it does not define the corresponding
resource model for an arbitrary process matrix. The API therefore raises an
explicit error instead of silently imposing an unmotivated condition.
