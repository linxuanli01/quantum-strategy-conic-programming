"""Energy-constrained bounds for the two-use magnetometry example."""

import argparse

import numpy as np
from magnetometry import two_use_problem

from quantum_strategy_conic import solve_quantum_bound


def parse_energies(value):
    energies = [float(item) for item in value.split(",")]
    if any(energy < 0 for energy in energies):
        raise argparse.ArgumentTypeError("energy budgets must be nonnegative")
    return energies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energies",
        type=parse_energies,
        default=parse_energies("0.25,0.5,0.75,1.0"),
        help="comma-separated energy budgets",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["seq", "par", "sup", "swi"],
        default=["par", "seq", "swi", "sup"],
    )
    parser.add_argument("--bound", choices=["NHB", "SLD", "Holevo"], default="Holevo")
    args = parser.parse_args()

    channel, derivatives = two_use_problem(
        alpha=1.5,
        theta=1.3,
        azimuth=np.pi / 4,
        damping_probability=0.5,
    )
    hamiltonian = np.diag([0.0, 1.0])

    print("strategy,energy,bound")
    for strategy in args.strategies:
        for energy in args.energies:
            result = solve_quantum_bound(
                G=np.eye(3),
                T=channel,
                DT=derivatives,
                dims=(2, 2, 2, 2),
                bound_type=args.bound,
                strategy=strategy,
                energy_bound=energy,
                energy_hamiltonian=hamiltonian,
            )
            print(f"{strategy},{energy:.10g},{result[0]:.10g}")


if __name__ == "__main__":
    main()
