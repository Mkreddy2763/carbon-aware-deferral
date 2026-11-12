"""Split conformal and Adaptive Conformal Inference (ACI) for multi-horizon forecasts.

Scores are signed residuals e = y - f, calibrated per horizon. A two-sided
(1 - alpha) interval is [f + q_lo, f + q_hi] with q_lo the alpha/2 and q_hi the
1 - alpha/2 conformal quantiles (finite-sample corrected). Treating the tails
separately keeps the interval asymmetric when errors are skewed, and it gives the
scheduler a calibrated one-sided upper bound (coverage 1 - alpha/2).

ACI (Gibbs & Candes, 2021) runs one online recursion per horizon and per tail:
    a_{t+1} = a_t + gamma * (alpha/2 - miss_t)
Feedback is delayed: the error of a forecast issued at t for horizon h is only
observed at t + h, so the level used at issue t uses misses of issues <= t - h.
Quantiles are taken over the fixed calibration scores; the effective level is
clipped to [A_MIN, 1 - A_MIN] so bounds stay finite (logged as a deviation from
textbook ACI, which allows infinite intervals).
"""
import numpy as np

A_MIN = 1e-3


def _cq(scores, level):
    """Finite-sample conformal quantile: ceil((n+1) level)/n empirical quantile."""
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * level)) - 1
    return s[min(max(k, 0), n - 1)]


def _cq_low(scores, level):
    """Lower-tail quantile, mirrored: -_cq(-scores, 1-level)."""
    return -_cq(-scores, 1 - level)


def split_quantiles(E_cal, alpha):
    """E_cal: (n_cal, H) residuals. Returns q_lo, q_hi of shape (H,)."""
    H = E_cal.shape[1]
    q_lo = np.array([_cq_low(E_cal[:, j], alpha / 2) for j in range(H)])
    q_hi = np.array([_cq(E_cal[:, j], 1 - alpha / 2) for j in range(H)])
    return q_lo, q_hi


def aci_bounds(E_cal, E_test, alpha, gamma=0.005):
    """Online ACI over test issue times (rows of E_test, hourly, in order).

    E_test[i, h-1] is the residual of the forecast issued at test row i for
    horizon h (NaN when the target is unavailable). Returns q_lo, q_hi of shape
    (n_test, H) and the alpha trajectories.
    """
    n, H = E_test.shape
    q_lo = np.empty((n, H))
    q_hi = np.empty((n, H))
    a_lo_tr = np.empty((n, H))
    a_hi_tr = np.empty((n, H))
    for j in range(H):
        h = j + 1
        cal = E_cal[:, j]
        cal = np.sort(cal[np.isfinite(cal)])
        m = len(cal)
        a_lo = a_hi = alpha / 2
        for i in range(n):
            # absorb feedback that has become observable: issue i - h
            k = i - h
            if k >= 0 and np.isfinite(E_test[k, j]):
                miss_hi = float(E_test[k, j] > q_hi[k, j])
                miss_lo = float(E_test[k, j] < q_lo[k, j])
                a_hi += gamma * (alpha / 2 - miss_hi)
                a_lo += gamma * (alpha / 2 - miss_lo)
            lh = min(max(1 - a_hi, A_MIN), 1 - A_MIN)
            ll = min(max(a_lo, A_MIN), 1 - A_MIN)
            kh = min(max(int(np.ceil((m + 1) * lh)) - 1, 0), m - 1)
            kl = min(max(int(np.floor((m + 1) * ll)) - 1, 0), m - 1)
            q_hi[i, j] = cal[kh]
            q_lo[i, j] = cal[kl]
            a_lo_tr[i, j], a_hi_tr[i, j] = a_lo, a_hi
    return q_lo, q_hi, a_lo_tr, a_hi_tr


def coverage(E, q_lo, q_hi):
    ok = np.isfinite(E)
    inside = (E >= q_lo) & (E <= q_hi)
    upper_ok = E <= q_hi
    return {
        "coverage": float(inside[ok].mean()),
        "upper_coverage": float(upper_ok[ok].mean()),
        "width": float(np.broadcast_to(q_hi - q_lo, E.shape)[ok].mean()),
    }
