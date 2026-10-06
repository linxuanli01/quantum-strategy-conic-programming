"""Two-use, three-parameter qubit magnetometry example."""

import argparse

import numpy as np
from scipy.linalg import expm

from quantum_strategy_conic import solve_quantum_bound


def magnetometry_channel(alpha, theta, azimuth, damping_probability):
    """Return a one-use Choi operator and its three derivatives."""
    pauli = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ]
    )
    direction = np.array(
        [
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
            np.cos(theta),
        ]
    )
    tangent_theta = np.array(
        [
            np.cos(theta) * np.cos(azimuth),
            np.cos(theta) * np.sin(azimuth),
            -np.sin(theta),
        ]
    )
    tangent_azimuth = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])

    rotated_theta = np.cos(alpha) * tangent_theta - np.sin(alpha) * tangent_azimuth
    rotated_azimuth = np.cos(alpha) * tangent_azimuth + np.sin(alpha) * tangent_theta
    generators = (
        sum(pauli[index] * direction[index] for index in range(3)),
        np.sin(alpha) * sum(pauli[index] * rotated_theta[index] for index in range(3)),
        np.sin(alpha)
        * np.sin(theta)
        * sum(pauli[index] * rotated_azimuth[index] for index in range(3)),
    )

    hamiltonian = alpha * sum(pauli[index] * direction[index] for index in range(3))
    unitary = expm(-1.0j * hamiltonian)
    unitary_derivatives = [-1.0j * unitary @ generator for generator in generators]

    damping = [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - damping_probability)]]),
        np.array([[0.0, np.sqrt(damping_probability)], [0.0, 0.0]]),
    ]
    kraus = [operator @ unitary for operator in damping]
    derivative_kraus = [
        [operator @ derivative for operator in damping] for derivative in unitary_derivatives
    ]

    kraus_vectors = [operator.reshape(-1, 1) for operator in kraus]
    derivative_vectors = [
        [operator.reshape(-1, 1) for operator in derivatives] for derivatives in derivative_kraus
    ]
    choi = sum(vector @ vector.conj().T for vector in kraus_vectors)
    derivatives = [
        sum(
            derivative @ vector.conj().T + vector @ derivative.conj().T
            for vector, derivative in zip(kraus_vectors, derivative_set)
        )
        for derivative_set in derivative_vectors
    ]
    return choi, derivatives


def two_use_problem(alpha, theta, azimuth, damping_probability):
    """Build the two-use channel Choi operator and product-rule derivatives."""
    one_use, one_use_derivatives = magnetometry_channel(alpha, theta, azimuth, damping_probability)
    channel = np.kron(one_use, one_use)
    derivatives = [
        np.kron(derivative, one_use) + np.kron(one_use, derivative)
        for derivative in one_use_derivatives
    ]
    return channel, derivatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", choices=["NHB", "SLD", "Holevo"], default="Holevo")
    parser.add_argument("--strategy", choices=["seq", "par", "sup", "swi", "ico"], default="seq")
    parser.add_argument("--damping", type=float, default=0.0)
    args = parser.parse_args()

    channel, derivatives = two_use_problem(
        alpha=1.0,
        theta=np.pi / 4,
        azimuth=np.pi / 4,
        damping_probability=args.damping,
    )
    result = solve_quantum_bound(
        G=np.eye(3),
        T=channel,
        DT=derivatives,
        dims=(2, 2, 2, 2),
        bound_type=args.bound,
        strategy=args.strategy,
    )
    print("status: completed")
    print(f"optimal value: {result[0]:.10g}")


if __name__ == "__main__":
    main()
