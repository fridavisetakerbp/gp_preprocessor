import math

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.special import gamma


def gp_matern_fast(X_train, y_train, X_test, ell, sigma_f, sigma_n):
    """GP regression with a Matérn ν=1.5 kernel as a Kalman filter + RTS smoother.

    Returns the posterior mean, standard deviation and gradient at ``X_test``.
    The state is 2-dimensional, so the per-sample 2×2 algebra is written out
    with Python floats: numpy's per-call overhead on 2×2 arrays dominated the
    run time. The transition and process noise only depend on the step ``dt``,
    and regularly sampled data has few distinct steps, so they are computed
    once per distinct step.
    """
    # Joint test and training data, sorted by position
    X_full_unsorted = np.concatenate([X_train, X_test])
    y_full_unsorted = np.concatenate([y_train, np.full_like(X_test, np.nan)])
    perm = np.argsort(X_full_unsorted)
    t = X_full_unsorted[perm].tolist()
    y = y_full_unsorted[perm].tolist()
    n = len(t)

    # Hyperparameters
    lambda_ = np.sqrt(3) / ell
    nu = 1.5  # To modify this, the whole function has to be rewritten!
    p = nu - 0.5  # p = nu - 0.5 for Matérn kernel
    lam = np.sqrt(2 * nu) / ell
    q = (2 * sigma_f * np.sqrt(np.pi) * lam ** (2 * p + 1) * gamma(p + 1)) / gamma(p + 0.5)

    # LTI SDE for Matérn ν=1.5
    F = np.array([[0, 1], [-lambda_**2, -2 * lambda_]])
    L = np.array([[0], [1]])
    LLT = L @ L.T

    steps = {}

    def discretize(dt):
        """Flattened (A, Q) for step dt, computed once per distinct dt."""
        cached = steps.get(dt)
        if cached is None:
            A = expm(F * dt)
            Q = (
                LLT * dt
                + (F @ LLT + LLT @ F.T) * (dt**2 / 2.0)
                + F @ LLT @ F.T * (dt**3 / 3.0)
            ) * q
            Q = 0.5 * (Q + Q.T)  # enforce symmetry
            cached = steps[dt] = (A.ravel().tolist(), Q.ravel().tolist())
        return cached

    noise_var = float(sigma_n) ** 2
    # States are (x0, x1); covariances are row-major (P00, P01, P10, P11).
    x_filt = [None] * n
    P_filt = [None] * n
    x_pred = [None] * n
    P_pred = [None] * n

    # Initialization (stationary prior)
    a, b, c, d = solve_continuous_lyapunov(F, -LLT * q).ravel().tolist()
    x0 = x1 = 0.0
    if not math.isnan(y[0]):
        S = a + noise_var
        k0, k1 = a / S, c / S
        r = y[0] - x0
        x0, x1 = x0 + k0 * r, x1 + k1 * r
        a, b, c, d = a - k0 * a, b - k0 * b, c - k1 * a, d - k1 * b
    x_filt[0] = (x0, x1)
    P_filt[0] = (a, b, c, d)

    # Kalman filter
    for k in range(1, n):
        (A0, A1, A2, A3), (Q0, Q1, Q2, Q3) = discretize(t[k] - t[k - 1])
        u0, u1 = x_filt[k - 1]
        a, b, c, d = P_filt[k - 1]
        p0, p1 = A0 * u0 + A1 * u1, A2 * u0 + A3 * u1
        m0, m1, m2, m3 = A0 * a + A1 * c, A0 * b + A1 * d, A2 * a + A3 * c, A2 * b + A3 * d
        P0 = m0 * A0 + m1 * A1 + Q0
        P1 = m0 * A2 + m1 * A3 + Q1
        P2 = m2 * A0 + m3 * A1 + Q2
        P3 = m2 * A2 + m3 * A3 + Q3
        x_pred[k] = (p0, p1)
        P_pred[k] = (P0, P1, P2, P3)
        if not math.isnan(y[k]):
            S = P0 + noise_var
            k0, k1 = P0 / S, P2 / S
            r = y[k] - p0
            x_filt[k] = (p0 + k0 * r, p1 + k1 * r)
            P_filt[k] = (P0 - k0 * P0, P1 - k0 * P1, P2 - k1 * P0, P3 - k1 * P1)
        else:
            x_filt[k] = (p0, p1)
            P_filt[k] = (P0, P1, P2, P3)

    # RTS smoother
    x_smooth = [None] * n
    P_smooth = [None] * n
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for k in range(n - 2, -1, -1):
        # Same discretization as in the filter step
        (A0, A1, A2, A3), _ = discretize(t[k + 1] - t[k])
        a, b, c, d = P_filt[k]
        P0, P1, P2, P3 = P_pred[k + 1]
        det = P0 * P3 - P1 * P2
        i0, i1, i2, i3 = P3 / det, -P1 / det, -P2 / det, P0 / det
        # Smoother gain C = P_filt[k] @ A.T @ inv(P_pred[k+1])
        m0, m1 = a * A0 + b * A1, a * A2 + b * A3
        m2, m3 = c * A0 + d * A1, c * A2 + d * A3
        C0, C1 = m0 * i0 + m1 * i2, m0 * i1 + m1 * i3
        C2, C3 = m2 * i0 + m3 * i2, m2 * i1 + m3 * i3
        s0, s1 = x_smooth[k + 1]
        q0, q1 = x_pred[k + 1]
        e0, e1 = s0 - q0, s1 - q1
        u0, u1 = x_filt[k]
        x_smooth[k] = (u0 + C0 * e0 + C1 * e1, u1 + C2 * e0 + C3 * e1)
        S0, S1, S2, S3 = P_smooth[k + 1]
        D0, D1, D2, D3 = S0 - P0, S1 - P1, S2 - P2, S3 - P3
        n0, n1 = C0 * D0 + C1 * D2, C0 * D1 + C1 * D3
        n2, n3 = C2 * D0 + C3 * D2, C2 * D1 + C3 * D3
        P_smooth[k] = (
            a + n0 * C0 + n1 * C1,
            b + n0 * C2 + n1 * C3,
            c + n2 * C0 + n3 * C1,
            d + n2 * C2 + n3 * C3,
        )

    # Extract means, stds and gradients (H @ F @ x = x[1] for Matérn ν=1.5)
    mean_smooth = np.array([state[0] for state in x_smooth])
    gradient_smooth = np.array([state[1] for state in x_smooth])
    std_smooth = np.sqrt(np.array([cov[0] for cov in P_smooth]))

    inv_perm = np.argsort(perm)
    n_train = np.size(y_train)
    mean_test = mean_smooth[inv_perm][n_train:]
    std_test = std_smooth[inv_perm][n_train:]
    gradient_test = gradient_smooth[inv_perm][n_train:]

    return mean_test, std_test, gradient_test
