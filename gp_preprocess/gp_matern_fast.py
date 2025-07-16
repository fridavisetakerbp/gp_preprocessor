import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.special import gamma

def gp_matern_fast(X_train, y_train, X_test, ell, sigma_f, sigma_n):

    # Joint test and training data
    X_full_unsorted = np.concatenate([X_train, X_test])
    y_full_unsorted = np.concatenate([y_train, np.full_like(X_test, np.nan)])

    # Sort x_full and get the permutation indices
    perm = np.argsort(X_full_unsorted)
    X_full = X_full_unsorted[perm]
    y_full = y_full_unsorted[perm]

    # Hyperparameters
    lambda_ = np.sqrt(3) / ell

    nu = 1.5 # To modify this, the whole function has to be rewritten!
    p = nu - 0.5 # p = nu - 0.5 for Matérn kernel
    lam = np.sqrt(2 * nu) / ell
    q = (2 * sigma_f * np.sqrt(np.pi) * lam**(2*p + 1) * gamma(p + 1)) / gamma(p + 0.5)

    # # Define LTI SDE for Matérn ν=1.5
    F = np.array([[0, 1],
                [-lambda_**2, -2 * lambda_]])
    L = np.array([[0],
                [1]])
    H = np.array([[1, 0]])

    def discretize(F, L, q, dt):
        A = expm(F * dt)
        LLT = L @ L.T
        Q = (
            LLT * dt +
            (F @ LLT + LLT @ F.T) * (dt**2 / 2.0) +
            F @ LLT @ F.T * (dt**3 / 3.0)
        ) * q
        Q = 0.5 * (Q + Q.T)  # enforce symmetry
        return A, Q

    # Interpolation setup
    t_full = X_full
    t_full = X_full
    n = len(t_full)

    # Initialization (stationary prior)
    P_inf = solve_continuous_lyapunov(F, -L @ L.T * q)
    x_filt = np.zeros((2, n))
    P_filt = np.zeros((2, 2, n))
    x_pred = np.zeros((2, n))
    P_pred = np.zeros((2, 2, n))
    x_filt[:, 0] = np.zeros(2)
    P_filt[:, :, 0] = P_inf

    if not np.isnan(y_full[0]):
        S0 = H @ P_filt[:, :, 0] @ H.T + sigma_n**2
        K0 = P_filt[:, :, 0] @ H.T / S0
        x_filt[:, 0] = x_filt[:, 0] + K0.flatten() * (y_full[0] - H @ x_filt[:, 0])
        P_filt[:, :, 0] = (np.eye(2) - K0 @ H) @ P_filt[:, :, 0]

    # Kalman Filter
    for k in range(1, n):

        dt = t_full[k] - t_full[k-1]
        A, Q = discretize(F, L, q, dt)
        x_pred[:, k] = A @ x_filt[:, k-1]
        P_pred[:, :, k] = A @ P_filt[:, :, k-1] @ A.T + Q

        if not np.isnan(y_full[k]):
            S = H @ P_pred[:, :, k] @ H.T + sigma_n**2
            K = P_pred[:, :, k] @ H.T / S
            x_filt[:, k] = x_pred[:, k] + K.flatten() * (y_full[k] - H @ x_pred[:, k])
            P_filt[:, :, k] = (np.eye(2) - K @ H) @ P_pred[:, :, k]
        else:
            x_filt[:, k] = x_pred[:, k]
            P_filt[:, :, k] = P_pred[:, :, k]
        

    # RTS smoother
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[:, -1] = x_filt[:, -1]
    P_smooth[:, :, -1] = P_filt[:, :, -1]

    for k in reversed(range(n - 1)):
        dt = t_full[k+1] - t_full[k]
        # Use the same discretization as in the filter step
        A, Q = discretize(F, L, q, dt)
        # Compute the smoother gain using the predicted covariance
        C = P_filt[:, :, k] @ A.T @ np.linalg.inv(P_pred[:, :, k+1])
        # Update the smoothed state and covariance
        x_smooth[:, k] = x_filt[:, k] + C @ (x_smooth[:, k+1] - x_pred[:, k+1])
        P_smooth[:, :, k] = P_filt[:, :, k] + C @ (P_smooth[:, :, k+1] - P_pred[:, :, k+1]) @ C.T

    # Extract means and stds
    mean_smooth = x_smooth[0, :]
    std_smooth = np.sqrt(P_smooth[0, 0, :])

    inv_perm = np.argsort(perm)

    mean_unsorted = mean_smooth[inv_perm]
    std_unsorted = std_smooth[inv_perm]

    n_train = np.size(y_train)
    mean_test = mean_unsorted[n_train:]
    std_test = std_unsorted[n_train:]

    return mean_test, std_test