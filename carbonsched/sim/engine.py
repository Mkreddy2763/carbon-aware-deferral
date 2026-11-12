"""Hour-stepped, slot-resolution simulation engine shared by all policies.

Loop over hours; at each hour boundary re-plan waiting jobs (if the policy
re-plans), then admit that hour's arrivals in submission order. A job's decision
is a reservation (region, start). Reservations that have not started may be moved
by re-planning; started jobs never move.

v2 diagnostics (no effect on decisions): for every deferrable job we also record
the slot its rule *wanted* at arrival if capacity were unlimited (des_start), and
per-job counts of decisions, decisions that chose to defer, and reservation moves.
"""
import time
from dataclasses import dataclass

import numpy as np

from .capacity import Capacity
from .env import SPH


@dataclass
class Workload:
    submit: np.ndarray      # slot, sorted ascending
    dur: np.ndarray         # true duration, slots (>=1)
    dur_est: np.ndarray     # scheduler's duration estimate, slots
    gpus: np.ndarray        # GPUs held (may be fractional)
    kw: np.ndarray          # facility power incl. PUE, kW
    deferrable: np.ndarray  # bool
    deadline: np.ndarray    # slot by which the job must finish
    home: np.ndarray        # home region index
    counted: np.ndarray     # False for warm-up jobs (excluded from metrics)

    def __len__(self):
        return len(self.submit)


class Engine:
    def __init__(self, envs, wl, cap, policy, migration_slots=2, seed=12345):
        self.envs = envs
        self.wl = wl
        self.pol = policy
        self.mig = migration_slots
        self.seed = seed
        R = len(envs)
        n_slots = int((wl.submit.max() + wl.dur.max()) + 12 * 24 * 3)
        self.capacity = Capacity(R, n_slots, cap)
        for e in envs:
            e.start(e.p0, n_slots)
        n = len(wl)
        self.start = np.full(n, -1, np.int64)
        self.region = wl.home.astype(np.int64).copy()
        # could the job have started at submission in its home region when it arrived?
        self.now_ok = np.zeros(n, bool)
        self.decision_s = []
        self.des_start = np.full(n, -1, np.int64)
        self.des_region = wl.home.astype(np.int64).copy()
        self.last_start = np.full(n, -1, np.int64)   # wanted slot at the job's last decision
        self.last_region = wl.home.astype(np.int64).copy()
        self.n_dec = np.zeros(n, np.int32)
        self.n_defer = np.zeros(n, np.int32)
        self.n_move = np.zeros(n, np.int32)

    # ------------------------------------------------------------------ helpers
    def _asap(self, j, tau, free=False):
        wl = self.wl
        r = int(wl.home[j])
        if free:
            return r, tau
        s = self.capacity.earliest(r, tau, int(wl.dur[j]), float(wl.gpus[j]))
        return r, s

    def _candidates(self, tau, L):
        """Now, every hour boundary in (tau, L], and L itself."""
        if L <= tau:
            return np.array([tau], np.int64)
        first = (tau // SPH + 1) * SPH
        c = np.arange(first, L + 1, SPH, dtype=np.int64)
        c = np.concatenate([[tau], c, [L]])
        return np.unique(c)

    def _ok(self, r, cand, d, g, tau, free):
        """Capacity feasibility of candidate starts. free=True ignores capacity (records
        the slot the rule wanted). Headroom policies (P8) may place a *deferred*
        reservation only where h*C GPUs stay free for later arrivals."""
        if free:
            return np.ones(len(cand), bool)
        ok = self.capacity.feasible(r, cand, d, g)
        h = self.pol.get("headroom", 0.0)
        if h > 0:
            deferred = cand != tau
            if deferred.any():
                ok_h = self.capacity.feasible(r, cand, d, g + h * float(self.capacity.cap[r]))
                ok = np.where(deferred, ok & ok_h, ok)
        return ok

    def _decide(self, j, tau, hh, free=False, prev=None):
        """Return (region, start) for job j at decision slot tau (hour hh)."""
        wl, pol = self.wl, self.pol
        rule = pol["rule"]
        home = int(wl.home[j])
        d, de, g = int(wl.dur[j]), int(wl.dur_est[j]), float(wl.gpus[j])
        if rule == "oracle":
            de = d  # perfect foresight includes the true duration
        if rule == "fcfs" or not wl.deferrable[j]:
            return self._asap(j, tau, free)
        L = int(wl.deadline[j]) - de          # latest start the scheduler believes is on time
        if L <= tau:
            return self._asap(j, tau, free)
        if rule == "threshold":
            e = self.envs[home]
            if e.ci_now(hh) <= e.thr_now(hh):
                return self._asap(j, tau, free)
            cand = self._candidates(tau, L)[::-1]
            ok = self._ok(home, cand, d, g, tau, free)
            if ok.any():
                return home, int(cand[np.argmax(ok)])  # latest feasible start
            return self._asap(j, tau, free)

        regions = range(len(self.envs)) if pol["spatial"] else [home]
        best = (np.inf, None, None)
        now_cost = None
        pool = []  # (score, region, start) of feasible candidates, for jitter
        for r in regions:
            t_r = tau if r == home else tau + self.mig
            if t_r > L:
                continue
            cand = self._candidates(t_r, L)
            ok = self._ok(r, cand, d, g, tau, free)
            if not ok.any():
                continue
            cand = cand[ok]
            e = self.envs[r]
            if rule == "oracle":
                sc = e.actual_cost(cand, d)
            elif rule in ("greedy", "cdec", "jitter"):
                sc = e.belief_cost(hh, "point", cand, de)
            elif rule == "lam":
                sc = e.belief_cost(hh, "lam", cand, de, lam=pol["lam"])
            elif rule == "robust":
                sc = e.belief_cost(hh, "upper", cand, de)
                if r == home and cand[0] == tau:
                    sc[0] = e.belief_cost(hh, "point", cand[:1], de)[0]
            else:
                raise ValueError(rule)
            sc = sc * float(wl.kw[j])
            i = int(np.argmin(sc))  # ties -> earliest start
            if sc[i] < best[0] - 1e-9:
                best = (sc[i], r, int(cand[i]))
            if r == home and cand[0] == tau:
                now_cost = sc[0]
            if rule == "jitter":
                pool += [(float(v), r, int(c)) for v, c in zip(sc, cand)]
        if best[1] is None:
            return self._asap(j, tau, free)
        if rule == "cdec" and (best[1] != home or best[2] != tau) and now_cost is not None:
            # decision-level conformal: defer only if the predicted saving survives the
            # calibrated overestimate margin; otherwise run now
            e = self.envs[best[1]]
            margin = e.cdec_margin(hh, de / SPH, (L - tau) / SPH) * float(wl.kw[j])
            if now_cost - best[0] - margin <= 0:
                return home, tau
        if rule == "jitter":
            # P9: spread deferrals uniformly over the k best slots by point forecast,
            # never choosing a slot forecast to be worse than running now
            cap_cost = now_cost if now_cost is not None else np.inf
            pool = [p for p in pool if p[0] <= cap_cost + 1e-9]
            pool.sort(key=lambda p: (p[0], p[2]))
            top = pool[:int(pol.get("k", 3))]
            if prev is not None and any(p[1] == prev[0] and p[2] == prev[1] for p in top):
                return prev  # sticky re-planning: keep a slot that is still in the top k
            rng = np.random.default_rng([self.seed, int(j), int(self.n_dec[j])])
            p = top[int(rng.integers(len(top)))]
            return p[1], p[2]
        return best[1], best[2]

    def _place(self, j, r, s):
        self.region[j], self.start[j] = r, s
        self.capacity.reserve(r, s, int(self.wl.dur[j]), float(self.wl.gpus[j]))

    # ------------------------------------------------------------------ loop
    def run(self):
        wl, pol = self.wl, self.pol
        n = len(wl)
        order = np.argsort(wl.submit, kind="stable")
        ptr = 0
        waiting = []
        hh = int(wl.submit[order[0]]) // SPH
        replan = pol["replan"]
        while ptr < n or waiting:
            t0 = hh * SPH
            waiting = [j for j in waiting if self.start[j] >= t0]
            if replan and waiting:
                for j in waiting:
                    r0, s0 = int(self.region[j]), int(self.start[j])
                    d, g = int(wl.dur[j]), float(wl.gpus[j])
                    if pol["rule"] == "threshold":
                        e = self.envs[r0]
                        if s0 > t0 and e.ci_now(hh) <= e.thr_now(hh):
                            self.capacity.release(r0, s0, d, g)
                            if self.capacity.feasible(r0, [t0], d, g)[0]:
                                s0 = t0
                            self.capacity.reserve(r0, s0, d, g)
                            self.n_move[j] += int(s0 != self.start[j])
                            self.start[j] = s0
                        continue
                    self.capacity.release(r0, s0, d, g)
                    t1 = time.perf_counter()
                    r, s = self._decide(j, t0, hh, prev=(r0, s0))
                    self.decision_s.append(time.perf_counter() - t1)
                    self.last_region[j], self.last_start[j] = self._decide(j, t0, hh, free=True,
                                                                           prev=(r0, s0))
                    self.n_dec[j] += 1
                    self.n_defer[j] += int(s > t0)
                    self.n_move[j] += int((r, s) != (r0, s0))
                    self._place(j, r, s)
            while ptr < n and wl.submit[order[ptr]] < t0 + SPH:
                j = order[ptr]
                ptr += 1
                tau = int(wl.submit[j])
                self.now_ok[j] = self.capacity.feasible(int(wl.home[j]), [tau], int(wl.dur[j]),
                                                        float(wl.gpus[j]))[0]
                t1 = time.perf_counter()
                r, s = self._decide(j, tau, hh)
                if wl.deferrable[j]:
                    self.decision_s.append(time.perf_counter() - t1)
                    # same rng key as the real decision (n_dec not yet incremented)
                    self.des_region[j], self.des_start[j] = self._decide(j, tau, hh, free=True)
                    self.last_region[j], self.last_start[j] = self.des_region[j], self.des_start[j]
                    self.n_dec[j] += 1
                    self.n_defer[j] += int(s > tau)
                else:
                    self.des_region[j], self.des_start[j] = int(wl.home[j]), tau
                    self.last_region[j], self.last_start[j] = int(wl.home[j]), tau
                self._place(j, r, s)
                if s >= t0 + SPH and wl.deferrable[j]:
                    waiting.append(j)
            hh += 1
        assert (self.start >= wl.submit).all(), "job started before submission"
        return self.start, self.region
