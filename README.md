# Quantum Strategy Conic Programming

Research code for conic optimization of precision bounds in multiparameter
quantum metrology. The implementation accompanies the manuscript **“Optimal
Strategies for Multi-parameter Quantum Metrology”** and compares several
two-use qubit strategy classes:

- sequential strategies (`seq`);
- parallel strategies (`par`);
- causal superpositions (`sup`);
- the quantum switch (`swi`); and
- general indefinite-causal-order strategies (`ico`).

The package also contains a finite-memory sequential optimizer with either
independent intermediate controls or a homogeneous-control ansatz.

The available objectives are the Nagaoka–Hayashi bound (`NHB`), the SLD
quantum Cramér–Rao bound (`SLD`), and the Holevo bound (`Holevo`).

## Scope

This release contains the implementation used for two calls to a qubit
channel. Accordingly, `solve_quantum_bound` currently requires
`dims=(2, 2, 2, 2)`. The strategy constraints contain dimensions specific to
that setting; passing other dimensions raises an explicit error rather than
silently constructing a different optimization problem.

## Installation

Python 3.9 or newer is required. Create a virtual environment and install the
package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

[MOSEK](https://www.mosek.com/) is recommended for the semidefinite programs
and requires a separate license:

```bash
python -m pip install -e ".[mosek]"
```

When MOSEK is unavailable or fails, the code emits a warning and falls back to
SCS. SCS can be slower or less accurate for these problems, so publication
results should be checked with MOSEK and appropriate solver tolerances.

## Example

Run the two-use, three-parameter magnetometry example with

```bash
python examples/magnetometry.py
```

The bound, strategy, and amplitude-damping probability can be changed from the
command line:

```bash
python examples/magnetometry.py --bound Holevo --strategy seq --damping 0.05
```

Energy-constrained bounds use the same solver and strategy variables:

```bash
python examples/energy_constrained_magnetometry.py \
  --energies 0.25,0.5,0.75,1.0 \
  --strategies par seq swi sup
```

The core call is

```python
import numpy as np

from quantum_strategy_conic import solve_quantum_bound

result = solve_quantum_bound(
    G=np.eye(number_of_parameters),
    T=channel_choi,
    DT=channel_derivatives,
    dims=(2, 2, 2, 2),
    bound_type="Holevo",
    strategy="seq",
    energy_bound=0.5,
    energy_hamiltonian=np.diag([0.0, 1.0]),
)
optimal_value = result[0]
```

Here, `T` is the Choi operator for two channel calls and `DT` contains one
derivative of `T` per estimated parameter. All channel operators use NumPy's
default row-major vectorization:

```python
operator_vector = operator.reshape(-1, 1)
```

This convention must be used consistently when constructing `T` and `DT`.

### Energy constraint

When `energy_bound` is provided, the code implements the global-battery
constraint used in the accompanying manuscript. For a sequential strategy,
the cumulative energy increase is constrained after both the initial
preparation and the intermediate control. For parallel strategies, the
constraint is the total energy of the two-input probe. For causal
superpositions and the quantum switch, it is imposed on every normalized
causal-order branch; internally, the corresponding inequalities are scaled by
the branch probabilities.

The default single-qubit Hamiltonian is `|1><1|`. A different Hermitian `2 x 2`
Hamiltonian can be supplied through `energy_hamiltonian`. The general `ico`
class is intentionally excluded because the manuscript does not define a
corresponding branch-resolved global-battery model.

The subsystem ordering and the branch-probability scaling are derived in
[`docs/energy_constraints.md`](docs/energy_constraints.md).

### Finite ancillary memory

The finite-memory routines work with local channel Choi matrices and a fixed
memory dimension `dM=2` or `dM=4`. The main entry points are
`solve_finite_memory` and `solve_same_control_finite_memory`; warm-up variants
are provided for continuation in a noise parameter. Bound-specific constraints
are shared with the unrestricted solver rather than duplicated.

The full data convention, algorithm, return values, and the heuristic status
of the homogeneous-control method are documented in
[`docs/finite_memory.md`](docs/finite_memory.md).

## Return values

`solve_quantum_bound` returns the optimal value followed by the available
strategy-specific primal variables:

| Strategy | Return tuple |
| --- | --- |
| `seq` | `(value, Y, P1, P2)` |
| `par` | `(value, Y, probe_state)` |
| `sup` | `(value, P11, P12, P21, P22)` |
| `swi` | `(value, P1, P3)` |
| `ico` | `(value, P)` |

These arrays are returned from CVXPY and can be `None` when a solver does not
produce a primal solution. Always inspect the solver output when using the
primal variables in subsequent calculations.

## Repository layout

```text
.
├── examples/
│   ├── energy_constrained_magnetometry.py
│   └── magnetometry.py
├── src/quantum_strategy_conic/
│   ├── comb.py
│   ├── finite_memory.py
│   └── multipara_bounds.py
├── docs/
│   ├── energy_constraints.md
│   └── finite_memory.md
└── tests/
```

`comb.py` contains subsystem operations, channel models, and comb
construction. `multipara_bounds.py` contains the conic programs for the five
strategy classes.

## Tests

Install the development dependency and run

```bash
python -m pip install -e ".[dev]"
pytest
```

The lightweight tests check tensor conventions, comb derivatives, and input
validation. They do not reproduce the full numerical study from the paper.

## Citation

If you use this code, cite the accompanying manuscript. Citation metadata is
provided in [`CITATION.cff`](CITATION.cff) and should be updated with the DOI
and publication details once they are available.

## License

No public-use license has yet been selected. The included `LICENSE` file keeps
the authors' rights reserved. Replace it with the license chosen by all authors
before distributing the repository under open-source terms.
