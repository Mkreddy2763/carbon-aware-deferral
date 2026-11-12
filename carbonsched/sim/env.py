"""Region environment: actual CI, forecasts and conformal bounds as seen by the
scheduler at each hour of a simulation.

A simulation starts at absolute hour `p0` of the region's hourly series. Slot k
(5 min) lies in hour p0 + k // 12. At slot tau the scheduler knows the actual CI
of the current hour (a real-time value, as published by grid operators) and the
forecast issued at the start of that hour for hours +1..+48. Beyond +48 h the
belief is extended periodically (same hour-of-day of the last forecast day).
"""
import numpy as np
import pandas as pd

SPH = 12  # slots per hour
H = 48
HEXT = 216  # hours of belief horizon (48 h forecast + periodic extension)


class RegionEnv:
    def __init__(self, npz_path, model="F1", method="aci", alpha=0.1, k_err=1.0):
        z = np.load(npz_path)
        self.y = z["y"].astype(float)
        self.ts0_hours = int(z["ts0_hours"])
        self.issue = z["issue"]
        self.is_test = z["is_test"]
        assert np.all(np.diff(self.issue) == 1), "issue hours must be contiguous"
        F = z[model].astype(float)
        fallback = z["F0"].astype(float)
        F = np.where(np.isfinite(F), F, fallback)  # operator forecast has rare gaps
        qlo = z[f"{model}_{method}_{alpha}_qlo"].astype(float)
        qhi = z[f"{model}_{method}_{alpha}_qhi"].astype(float)
        if k_err != 1.0:
            # synthetic forecasts: actual + k * (real forecast error); bounds scale with k
            idx = self.issue[:, None] + np.arange(1, H + 1)[None, :]
            Y = np.where(idx < len(self.y), self.y[np.minimum(idx, len(self.y) - 1)], np.nan)
            Y = np.where(np.isfinite(Y), Y, F)
            F = Y + k_err * (F - Y)
            qlo, qhi = k_err * qlo, k_err * qhi
        self.F, self.qlo, self.qhi = F, qlo, qhi
        # rolling 30th percentile of the past week (threshold policy)
        self.thr30 = pd.Series(self.y).rolling(168, min_periods=24).quantile(0.3).to_numpy()
        self.p0 = None

    # -------------------------------------------------- decision-level conformal
    D_GRID = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 48])
    S_GRID = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24])

    def setup_decision_conformal(self, alpha=0.1, window_days=60, lag_days=3):
        """Calibrate the point-greedy *decision* rather than hourly forecasts.

        For every issue hour t with known outcomes, job length D and slack S (grids),
        replay the point-greedy choice delta* = argmin_{0<=delta<=S} Fcost(delta), and
        record the overestimate of its saving: score = s_hat - s, with
        s_hat = Fcost(0) - Fcost(delta*) and s = Ycost(0) - Ycost(delta*). This includes
        the correlation of errors across hours and the winner's curse of picking the
        best of many slots. q = conformal (1-alpha) quantile of the scores over a
        rolling window of issue hours whose outcomes are already observed
        ([day - window_days, day - lag_days)), refreshed daily. The rule defers only
        if s_hat - q > 0, so P(harmful voluntary deferral) <= alpha marginally under
        exchangeability of the window.
        """
        n = len(self.issue)
        L = int(self.D_GRID.max() + self.S_GRID.max()) + 1
        B = np.empty((n, L))
        B[:, 0] = self.y[self.issue]
        B[:, 1:H + 1] = np.maximum(self.F, 0)
        hx = np.arange(H + 1, L)
        B[:, H + 1:] = B[:, hx - 24 * np.ceil((hx - H) / 24).astype(int)]
        idx = self.issue[:, None] + np.arange(L)[None, :]
        A = np.where(idx < len(self.y), self.y[np.minimum(idx, len(self.y) - 1)], np.nan)
        cB = np.concatenate([np.zeros((n, 1)), np.cumsum(B, 1)], 1)
        cA = np.concatenate([np.zeros((n, 1)), np.cumsum(A, 1)], 1)
        dmax = int(self.S_GRID.max())
        scores = np.full((n, len(self.D_GRID), len(self.S_GRID)), np.nan)
        for i, D in enumerate(self.D_GRID):
            cf = cB[:, D:D + dmax + 1] - cB[:, :dmax + 1]      # (n, delta)
            ca = cA[:, D:D + dmax + 1] - cA[:, :dmax + 1]
            for k, S in enumerate(self.S_GRID):
                best = np.argmin(cf[:, :S + 1], axis=1)
                r = np.arange(n)
                s_hat = cf[:, 0] - cf[r, best]
                s = ca[:, 0] - ca[r, best]
                scores[:, i, k] = s_hat - s
        day = (self.issue - self.issue[0]) // 24
        n_days = int(day.max()) + 1
        q = np.full((n_days, len(self.D_GRID), len(self.S_GRID)), np.nan)
        for d0 in range(n_days):
            m = (day >= d0 - window_days) & (day < d0 - lag_days)
            if m.sum() < 24 * 14:
                continue
            sc = scores[m]
            for i in range(len(self.D_GRID)):
                for k in range(len(self.S_GRID)):
                    v = np.sort(sc[:, i, k][np.isfinite(sc[:, i, k])])
                    if len(v):
                        j = min(int(np.ceil((len(v) + 1) * (1 - alpha))) - 1, len(v) - 1)
                        q[d0, i, k] = v[j]
        self.cdec_q = q
        self.cdec_scores = scores

    def cdec_margin(self, hh, dur_h, slack_h):
        """Conformal margin (CI x hours) for a job of dur_h hours with slack_h hours."""
        day = (self.p0 + hh - self.issue[0]) // 24
        i = int(np.argmin(np.abs(np.log(self.D_GRID) - np.log(max(dur_h, 1.0)))))
        k = int(np.searchsorted(self.S_GRID, min(max(slack_h, 1.0), self.S_GRID[-1]) - 1e-9))
        v = self.cdec_q[day, i, k]
        scale = dur_h / self.D_GRID[i]  # margin scales ~linearly with length between grid points
        return float(v * scale) if np.isfinite(v) else np.inf

    # -------------------------------------------------------------- set-up
    def hour_pos(self, ts_utc):
        return int(pd.Timestamp(ts_utc).value // 3_600_000_000_000) - self.ts0_hours

    def start(self, p0, n_slots):
        """Anchor the simulation at absolute hour p0; precompute actual cumsums."""
        self.p0 = p0
        n_hours = n_slots // SPH + 2
        if p0 + n_hours > len(self.y):
            raise ValueError(f"simulation window runs past the CI series end "
                             f"({p0 + n_hours} > {len(self.y)} hours)")
        yh = self.y[p0:p0 + n_hours]
        if not np.isfinite(yh).all():
            raise ValueError("missing CI values inside the simulation window")
        self.slot_ci = np.repeat(yh, SPH)
        self.cumA = np.concatenate([[0.0], np.cumsum(self.slot_ci) / SPH])  # CI x hours
        self._cache = {}

    # -------------------------------------------------------------- costs
    def actual_cost(self, starts, d):
        starts = np.asarray(starts, dtype=np.int64)
        return self.cumA[starts + d] - self.cumA[starts]

    def ci_now(self, hh):
        return self.y[self.p0 + hh]

    def thr_now(self, hh):
        return self.thr30[self.p0 + hh]

    def _belief(self, hh, kind, lam):
        key = (hh, kind, lam)
        c = self._cache.get(key)
        if c is not None:
            return c
        if len(self._cache) > 64:
            self._cache.clear()
        p = self.p0 + hh
        row = p - self.issue[0]
        if not (0 <= row < len(self.issue)):
            raise ValueError("decision hour outside forecast range")
        f = self.F[row]
        if kind == "point":
            v = f
        elif kind == "upper":
            v = f + self.qhi[row]
        elif kind == "lam":
            v = f + lam * (self.qhi[row] - self.qlo[row])
        else:
            raise ValueError(kind)
        v = np.maximum(v, 0.0)
        b = np.empty(HEXT)
        b[0] = self.y[p]
        b[1:H + 1] = v
        hx = np.arange(H + 1, HEXT)
        b[H + 1:] = b[hx - 24 * np.ceil((hx - H) / 24).astype(int)]
        cum = np.concatenate([[0.0], np.cumsum(b)])
        c = (b, cum)
        self._cache[key] = c
        return c

    def belief_cost(self, hh, kind, starts, d, lam=0.0):
        """Believed CI x hours for running `d` slots from each start (abs slots)."""
        b, cum = self._belief(hh, kind, lam)
        base = hh * SPH
        maxk = SPH * HEXT - 1

        def C(k):
            k = np.minimum(k, maxk)
            q, r = np.divmod(k, SPH)
            return cum[q] + (r / SPH) * b[q]

        rel = np.asarray(starts, dtype=np.int64) - base
        return C(rel + d) - C(rel)
