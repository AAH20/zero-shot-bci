"""
Gromov-Wasserstein Optimal Transport Alignment for Zero-Shot BCI Generalization.

Aligns the non-Euclidean functional Riemannian geometries between an unseen patient (X)
and a canonical foundation brain atlas (Y) using Entropic Gromov-Wasserstein transport.
Eliminates patient-specific calibration retraining by projecting non-stationary
neural manifolds into a shared atlas representation.
"""

from __future__ import annotations
import math
from typing import Sequence


Matrix = list[list[float]]


def _sinkhorn_knopp(
    K: Matrix,
    p: list[float],
    q: list[float],
    max_iter: int = 30,
    tol: float = 1e-6,
) -> Matrix:
    """Computes doubly-stochastic scaling T = diag(u) * K * diag(v)."""
    n = len(p)
    m = len(q)
    u = [1.0 / n] * n
    v = [1.0 / m] * m

    for _ in range(max_iter):
        # Update u = p / (K * v)
        Kv = [0.0] * n
        for i in range(n):
            for j in range(m):
                Kv[i] += K[i][j] * v[j]
            Kv[i] = max(1e-12, Kv[i])
        u_new = [p[i] / Kv[i] for i in range(n)]

        # Update v = q / (K^T * u)
        KTu = [0.0] * m
        for j in range(m):
            for i in range(n):
                KTu[j] += K[i][j] * u_new[i]
            KTu[j] = max(1e-12, KTu[j])
        v_new = [q[j] / KTu[j] for j in range(m)]

        # Check convergence
        diff = sum(abs(u_new[i] - u[i]) for i in range(n))
        u = u_new
        v = v_new
        if diff < tol:
            break

    # Form transport matrix T_ij = u_i * K_ij * v_j
    T = [[u[i] * K[i][j] * v[j] for j in range(m)] for i in range(n)]
    return T


class GromovWassersteinAligner:
    """
    Entropic Gromov-Wasserstein solver for cross-subject neural manifold transport.
    """

    def __init__(self, epsilon: float = 0.05, max_gw_iter: int = 15):
        self.epsilon = epsilon
        self.max_gw_iter = max_gw_iter

    def solve_transport_matrix(
        self,
        D_X: Matrix,
        D_Y: Matrix,
    ) -> Matrix:
        """
        Solves optimal transport matrix T* in Pi(p, q) minimizing:
        GW(D_X, D_Y) = sum_{i,j,k,l} |D_X[i,k] - D_Y[j,l]|^2 * T_ik * T_jl
        """
        n = len(D_X)
        m = len(D_Y)
        p = [1.0 / n] * n
        q = [1.0 / m] * m

        # Initialize T as uniform outer product p * q^T
        T = [[p[i] * q[j] for j in range(m)] for i in range(n)]

        for _ in range(self.max_gw_iter):
            # 1. Compute quadratic cost gradient L_ij = sum_{k,l} (D_X[i,k] - D_Y[j,l])^2 * T_kl
            L = [[0.0 for _ in range(m)] for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    cost = 0.0
                    for k in range(n):
                        dx_ik = D_X[i][k]
                        for l_idx in range(m):
                            diff = dx_ik - D_Y[j][l_idx]
                            cost += diff * diff * T[k][l_idx]
                    L[i][j] = cost

            # 2. Kernel matrix K_ij = exp(-L_ij / epsilon)
            # Apply numerical stabilization: subtract min
            min_L = min(min(row) for row in L)
            K = [[math.exp(-max(0.0, (L[i][j] - min_L)) / self.epsilon) for j in range(m)] for i in range(n)]

            # 3. Sinkhorn projection step
            T = _sinkhorn_knopp(K, p, q)

        return T

    def project_features_to_atlas(
        self,
        patient_features: Sequence[float],
        transport_matrix: Matrix,
    ) -> list[float]:
        """
        Transports patient feature vector z_X in R^n to atlas coordinates z_Y in R^m:
        z_Y[j] = sum_i (T_ij * z_X[i]) / q_j
        """
        n = len(patient_features)
        m = len(transport_matrix[0])
        q_j = 1.0 / m

        atlas_vec: list[float] = [0.0] * m
        for j in range(m):
            sum_val = 0.0
            for i in range(min(n, len(transport_matrix))):
                sum_val += transport_matrix[i][j] * patient_features[i]
            atlas_vec[j] = sum_val / q_j

        return atlas_vec
