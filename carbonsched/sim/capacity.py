"""Shared capacity / reservation calendar (conservative backfilling).

Time is discretised into 5-minute slots. Every job holds a reservation
[start, start + dur) of `g` GPUs in one region; policies only choose *where* to
put reservations, so every policy runs on identical mechanics.
"""
import numpy as np

EPS = 1e-9


class Capacity:
    def __init__(self, n_regions, n_slots, cap):
        self.cap = np.broadcast_to(np.asarray(cap, float), (n_regions,)).copy()
        self.free = np.repeat(self.cap[:, None], n_slots, axis=1)

    @property
    def n_slots(self):
        return self.free.shape[1]

    def _ensure(self, end):
        if end > self.n_slots:
            extra = max(end - self.n_slots, self.n_slots // 2)
            pad = np.repeat(self.cap[:, None], extra, axis=1)
            self.free = np.concatenate([self.free, pad], axis=1)

    def reserve(self, r, s, d, g):
        self._ensure(s + d)
        self.free[r, s:s + d] -= g
        assert self.free[r, s:s + d].min() >= -1e-6, "capacity exceeded"

    def release(self, r, s, d, g):
        self.free[r, s:s + d] += g

    def feasible(self, r, starts, d, g):
        """Boolean mask: can `g` GPUs be held for `d` slots from each start?"""
        starts = np.asarray(starts, dtype=np.int64)
        if len(starts) == 0:
            return np.zeros(0, bool)
        lo, hi = int(starts.min()), int(starts.max()) + d
        self._ensure(hi)
        bad = self.free[r, lo:hi] < g - EPS
        cb = np.concatenate([[0], np.cumsum(bad)])
        rel = starts - lo
        return (cb[rel + d] - cb[rel]) == 0

    def earliest(self, r, t0, d, g, window=288):
        """First start >= t0 at which `g` GPUs are free for `d` slots."""
        while True:
            hi = t0 + d + window
            self._ensure(hi)
            bad = self.free[r, t0:hi] < g - EPS
            cb = np.concatenate([[0], np.cumsum(bad)])
            n = hi - t0 - d + 1
            ok = (cb[d:d + n] - cb[:n]) == 0
            idx = np.flatnonzero(ok)
            if len(idx):
                return t0 + int(idx[0])
            window *= 2
            if window > 12 * 24 * 365:
                raise RuntimeError("no feasible start within a year; capacity too small?")
