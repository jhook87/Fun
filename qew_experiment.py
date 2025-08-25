"""
Quantum Entanglement Window (QEW) simulation.

This module implements a simple toy model for studying entanglement wedge
connectivity between two subsystems.  Two registers of qubits are scrambled
with random Haar unitaries, a fraction of qubits are swapped between them
representing exchanged Hawking radiation, and various information‑theoretic
quantities are computed: mutual information, negativity and a simple
out‑of‑time‑order correlator (OTOC).  The code is self‑contained and
depends only on NumPy and SciPy.

Author: OpenAI's ChatGPT
Date: 2025-08-25
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from typing import List, Tuple, Sequence, Dict


def random_unitary(dim: int) -> np.ndarray:
    """Generate a Haar-random unitary matrix of dimension ``dim``.

    The algorithm draws a complex random matrix with independent normal
    entries and performs a QR decomposition.  To ensure Haar measure, the
    diagonal of ``R`` is normalised to have unit magnitude.

    Args:
        dim: Dimension of the unitary.

    Returns:
        A ``dim×dim`` unitary numpy array.
    """
    # Draw a random complex matrix with iid Gaussian entries
    z = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = qr(z)
    # Normalise the diagonal of R to have unit magnitude
    d = np.diag(np.exp(1j * np.angle(np.diag(r))))
    return q @ d


def apply_unitary_to_subsystem(
    state: np.ndarray, unitary: np.ndarray, qubit_indices: Sequence[int], num_qubits: int
) -> np.ndarray:
    """Apply a unitary on a subset of qubits to the global state.

    The global state is a vector of length ``2**num_qubits``.  We reshape
    the state into a tensor with one axis per qubit, move the target qubits
    to the front, apply the unitary, and then restore the original ordering.

    Args:
        state: State vector of shape ``(2**num_qubits,)``.
        unitary: Unitary matrix of dimension ``2**len(qubit_indices)``.
        qubit_indices: List of qubit indices to which the unitary acts (0 is
            the leftmost qubit).
        num_qubits: Total number of qubits.

    Returns:
        New state vector after application of the unitary.
    """
    state = state.reshape([2] * num_qubits)
    # Order of axes to bring qubit_indices to the front
    perm = list(qubit_indices) + [i for i in range(num_qubits) if i not in qubit_indices]
    inv_perm = np.argsort(perm)
    # Transpose and reshape to a matrix where rows correspond to the subsystem
    psi_perm = state.transpose(perm)
    dim_sub = 2 ** len(qubit_indices)
    dim_env = 2 ** (num_qubits - len(qubit_indices))
    psi_matrix = psi_perm.reshape(dim_sub, dim_env)
    # Apply the unitary on the subsystem (left index)
    psi_matrix = unitary @ psi_matrix
    # Reshape back and invert the permutation
    psi_perm = psi_matrix.reshape([2] * num_qubits)
    psi_out = psi_perm.transpose(inv_perm)
    return psi_out.reshape(-1)


def swap_qubit_pairs(
    state: np.ndarray, pairs: Sequence[Tuple[int, int]], num_qubits: int
) -> np.ndarray:
    """Swap pairs of qubits in the state using SWAP gates.

    For each pair (i, j), a 4×4 SWAP matrix is applied to the corresponding
    two qubit subsystem.  Swaps are applied sequentially; since SWAP
    operators on distinct pairs commute, the order does not matter.

    Args:
        state: State vector of length ``2**num_qubits``.
        pairs: List of pairs of qubit indices to swap.
        num_qubits: Total number of qubits.

    Returns:
        New state after all swaps are applied.
    """
    # Define the 4×4 SWAP gate
    swap_2q = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    new_state = state
    for i, j in pairs:
        new_state = apply_unitary_to_subsystem(new_state, swap_2q, [i, j], num_qubits)
    return new_state


def reduced_density_matrix(
    state: np.ndarray, qubits: Sequence[int], num_qubits: int
) -> np.ndarray:
    """Compute the reduced density matrix of a subset of qubits.

    The reduced state is obtained by tracing out all qubits not in ``qubits``.
    This implementation reshapes the state vector into a matrix and
    contracts over the environment indices.

    Args:
        state: State vector of length ``2**num_qubits``.
        qubits: Indices of the qubits to keep.
        num_qubits: Total number of qubits.

    Returns:
        A density matrix of size ``(2**len(qubits), 2**len(qubits))``.
    """
    qubits = list(qubits)
    # Reshape into tensor with one axis per qubit
    psi = state.reshape([2] * num_qubits)
    # Bring the subsystem qubits to the front
    perm = qubits + [i for i in range(num_qubits) if i not in qubits]
    psi_perm = psi.transpose(perm)
    dim_sub = 2 ** len(qubits)
    dim_env = 2 ** (num_qubits - len(qubits))
    psi_matrix = psi_perm.reshape(dim_sub, dim_env)
    # Partial trace over the environment: ρ = ψ ψ†
    rho = psi_matrix @ psi_matrix.conj().T
    return rho


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute the von Neumann entropy (base 2) of a density matrix.

    Args:
        rho: Density matrix.

    Returns:
        Entropy ``-Tr(ρ log₂ ρ)``.
    """
    # Eigenvalues may have small negative imaginary parts due to numerical error
    evals = np.linalg.eigvalsh(rho)
    # Discard tiny negative or near-zero eigenvalues
    evals = np.real_if_close(evals)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


def mutual_information(
    state: np.ndarray,
    part_a: Sequence[int],
    part_b: Sequence[int],
    num_qubits: int,
) -> float:
    """Compute the mutual information ``I(A:B)`` for two subsets of qubits.

    Mutual information is defined as ``S(A) + S(B) - S(AB)`` where ``S`` is
    the von Neumann entropy.

    Args:
        state: Pure state vector of length ``2**num_qubits``.
        part_a: List of qubits in subsystem A.
        part_b: List of qubits in subsystem B.
        num_qubits: Total number of qubits.

    Returns:
        The mutual information in bits.
    """
    rho_a = reduced_density_matrix(state, part_a, num_qubits)
    rho_b = reduced_density_matrix(state, part_b, num_qubits)
    rho_ab = reduced_density_matrix(state, list(part_a) + list(part_b), num_qubits)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho_ab)
    return s_a + s_b - s_ab


def negativity(
    state: np.ndarray,
    part_a: Sequence[int],
    part_b: Sequence[int],
    num_qubits: int,
) -> float:
    """Compute the negativity between two subsystems ``A`` and ``B``.

    Negativity is defined as ``(∥ρ^{T_A}∥₁ - 1) / 2``, where ``T_A`` denotes
    partial transpose over subsystem A and ``∥·∥₁`` is the trace norm.  The
    calculation proceeds by building the joint density matrix of A and B,
    reshaping it into ``(dim_A, dim_B, dim_A, dim_B)``, performing the
    partial transpose on indices corresponding to A, and computing the
    trace norm via eigenvalues.

    Args:
        state: Pure state vector of length ``2**num_qubits``.
        part_a: List of qubits belonging to subsystem A.
        part_b: List of qubits belonging to subsystem B.
        num_qubits: Total number of qubits.

    Returns:
        The negativity, a non‑negative real number.
    """
    # Build the density matrix of A∪B
    subsys = list(part_a) + list(part_b)
    rho_ab = reduced_density_matrix(state, subsys, num_qubits)
    dim_a = 2 ** len(part_a)
    dim_b = 2 ** len(part_b)
    # Reshape to (dim_a, dim_b, dim_a, dim_b)
    rho_ab = rho_ab.reshape(dim_a, dim_b, dim_a, dim_b)
    # Partial transpose on subsystem A: swap first and third indices
    rho_pt = rho_ab.transpose(2, 1, 0, 3).reshape(dim_a * dim_b, dim_a * dim_b)
    evals = np.linalg.eigvals(rho_pt)
    # Trace norm: sum of absolute values of eigenvalues
    trace_norm = np.sum(np.abs(evals))
    return float((trace_norm - 1.0) / 2.0)


def pauli_z_operator(qubit: int, num_qubits: int) -> np.ndarray:
    """Construct the Pauli‑Z operator acting on a specific qubit.

    Args:
        qubit: Index of the qubit (0 = leftmost).
        num_qubits: Total number of qubits.

    Returns:
        A ``2**num_qubits × 2**num_qubits`` matrix representing Z on the
        specified qubit and identity elsewhere.
    """
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    op = 1
    for q in range(num_qubits):
        if q == qubit:
            op = np.kron(op, z) if q > 0 else z
        else:
            op = np.kron(op, np.eye(2, dtype=complex)) if q > 0 else np.eye(2, dtype=complex)
    return op


def otoc(
    state: np.ndarray,
    w_qubit: int,
    v_qubit: int,
    num_qubits: int,
) -> float:
    """Compute a simple out‑of‑time‑order correlator (OTOC).

    We evaluate ``F = ⟨ψ| W V W V |ψ⟩`` where ``|ψ⟩`` is the current state
    after scrambling and exchange, and ``W`` and ``V`` are Pauli‑Z operators on
    two distinct qubits.  The magnitude ``|F|`` indicates the degree of
    non‑commutativity of operators under time evolution.

    Args:
        state: State vector after all operations.
        w_qubit: Index of the first qubit (for W).
        v_qubit: Index of the second qubit (for V).
        num_qubits: Total number of qubits.

    Returns:
        The absolute value of the OTOC.
    """
    W = pauli_z_operator(w_qubit, num_qubits)
    V = pauli_z_operator(v_qubit, num_qubits)
    # Apply operators in the order W V W V on |ψ⟩
    psi = state
    phi = W @ (V @ (W @ (V @ psi)))
    # Compute the overlap ⟨ψ|φ⟩
    overlap = np.vdot(psi, phi)
    return float(abs(overlap))


def simulate(
    n: int = 4,
    k_values: Sequence[int] | None = None,
    scramble_steps: int = 1,
    samples: int = 3,
    seed: int | None = None,
    plot: bool = True,
) -> Dict[str, List[float]]:
    """Run the QEW simulation for various exchange fractions.

    Args:
        n: Number of qubits in each register A and B.
        k_values: Sequence of numbers of qubits to swap.  If ``None``,
            defaults to ``range(1, n+1)``.
        scramble_steps: Number of independent scrambling unitaries to apply
            before the exchange.  Each step uses a fresh Haar random unitary
            on both registers.
        samples: Number of Haar random realisations to average over.
        seed: Optional random seed for reproducibility.
        plot: Whether to produce a matplotlib plot of the results.

    Returns:
        A dictionary mapping metric names to lists of values for each ``k``.
    """
    if seed is not None:
        np.random.seed(seed)
    if k_values is None:
        k_values = list(range(1, n + 1))
    total_qubits = 1 + 2 * n  # reference + A + B
    results = {
        "p": [],
        "mutual_info": [],
        "negativity": [],
        "otoc": [],
    }
    for k in k_values:
        p = k / n
        mi_vals: List[float] = []
        neg_vals: List[float] = []
        otoc_vals: List[float] = []
        # Repeat for multiple samples to average out randomness
        for _ in range(samples):
            # Build initial state |Ψ⟩ = |Φ+_RA0⟩ ⊗ |0⟩^{n-1} ⊗ |0⟩^n
            # ``R`` is qubit 0, ``A`` qubits are 1..n, ``B`` are n+1..2n
            # entangle R and A0
            phi_plus = (np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)).reshape(4)
            # R and A0 create |Φ+⟩; A1..A_{n-1}, B0..B_{n-1} start in |0⟩
            # Compose as tensor product
            state = phi_plus
            # Append remaining A qubits
            for _ai in range(n - 1):
                state = np.kron(state, np.array([1, 0], dtype=complex))
            # Append B qubits
            for _bi in range(n):
                state = np.kron(state, np.array([1, 0], dtype=complex))
            # Scramble both A and B registers ``scramble_steps`` times
            for _step in range(scramble_steps):
                # Draw Haar random unitaries for A and B
                u_a = random_unitary(2 ** n)
                u_b = random_unitary(2 ** n)
                # Apply to A: qubits 1..n
                state = apply_unitary_to_subsystem(state, u_a, list(range(1, n + 1)), total_qubits)
                # Apply to B: qubits n+1 .. 2n
                state = apply_unitary_to_subsystem(state, u_b, list(range(n + 1, 2 * n + 1)), total_qubits)
            # Swap ``k`` qubits between A and B: pairs (A_i, B_i) for i=0..k-1
            swap_pairs = [(1 + i, 1 + n + i) for i in range(k)]
            state = swap_qubit_pairs(state, swap_pairs, total_qubits)
            # Determine indices for RA and RB after the swap
            # RA: qubits now sitting in B positions  n+1 .. n+k
            ra_indices = [n + 1 + i for i in range(k)]
            # RB: qubits now sitting in A positions 1 .. k
            rb_indices = [1 + i for i in range(k)]
            # Compute information theoretic quantities
            mi = mutual_information(state, ra_indices, rb_indices, total_qubits)
            ne = negativity(state, ra_indices, rb_indices, total_qubits)
            # Choose the first swapped qubit from RA and RB for OTOC
            # If k==0 this loop would be empty, but k>=1 by construction
            w_qubit = ra_indices[0]
            v_qubit = rb_indices[0]
            ot = otoc(state, w_qubit, v_qubit, total_qubits)
            mi_vals.append(mi)
            neg_vals.append(ne)
            otoc_vals.append(ot)
        # Average over samples
        results["p"].append(p)
        results["mutual_info"].append(float(np.mean(mi_vals)))
        results["negativity"].append(float(np.mean(neg_vals)))
        results["otoc"].append(float(np.mean(otoc_vals)))
    # Optionally plot the results
    if plot:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_title("QEW experiment: information measures vs. exchange fraction")
        ax1.set_xlabel("Exchange fraction p = k/n")
        ax1.set_ylabel("Mutual information / Negativity")
        ax1.plot(results["p"], results["mutual_info"], label="I(RA:RB)", marker="o")
        ax1.plot(results["p"], results["negativity"], label="Negativity", marker="s")
        ax2 = ax1.twinx()
        ax2.set_ylabel("OTOC")
        ax2.plot(results["p"], results["otoc"], label="OTOC", color="tab:green", marker="^")
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        plt.tight_layout()
        plt.savefig("qew_results.png", dpi=200)
        plt.close(fig)
    return results


if __name__ == "__main__":
    # Run the simulation with default parameters
    res = simulate(n=4, k_values=[1, 2], scramble_steps=1, samples=3, seed=42, plot=True)
    # Print a simple table to stdout
    print(f"{'p':>4}  {'I(RA:RB)':>10}  {'Negativity':>12}  {'OTOC':>8}")
    for p, mi, ne, ot in zip(res["p"], res["mutual_info"], res["negativity"], res["otoc"]):
        print(f"{p:0.2f}  {mi:10.4f}  {ne:12.4f}  {ot:8.4f}")