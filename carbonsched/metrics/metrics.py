"""Per-run metrics. Cross-policy metrics (saving vs P0, oracle capture, regret)
are computed in analysis/ from these per-run totals on identical offsets.

Harm: a job is *shifted* if the scheduler moved it in time (start > submit) or
space (region != home). A shifted job is *harmed* if its realised emissions
exceed what it would have emitted starting at submission in its home region.
Only deferrable jobs can be shifted by a carbon-aware rule; capacity queueing of
non-deferrable jobs is not counted as shifting.

Decomposition: a shifted job is *voluntary* if its home region had capacity for it
at submission (the rule chose to wait) and *forced* otherwise (other jobs'
reservations, often other deferred jobs herding into low-carbon hours, filled the
cluster). With perfect foresight the voluntary harm rate is 0 by construction.
"""
import numpy as np

from ..sim.env import SPH


def job_emissions(envs, wl, start, region):
    """Realised and run-at-submission emissions (grams CO2) per job."""
    d = wl.dur.astype(np.int64)
    E = np.empty(len(wl))
    E0 = np.empty(len(wl))
    for r, e in enumerate(envs):
        m = region == r
        E[m] = e.actual_cost(start[m], d[m]) * wl.kw[m]
        m0 = wl.home == r
        E0[m0] = e.actual_cost(wl.submit[m0], d[m0]) * wl.kw[m0]
    return E, E0


def summarise(envs, wl, start, region, cap, decision_s=None, now_ok=None, rel_tol=1e-9, diag=None):
    E, E0 = job_emissions(envs, wl, start, region)
    c = wl.counted
    dfr = c & wl.deferrable
    wait = (start - wl.submit).astype(float)
    shifted = dfr & ((start > wl.submit) | (region != wl.home))
    harmed = shifted & (E > E0 * (1 + rel_tol))
    finish = start + wl.dur
    miss = dfr & (finish > wl.deadline)
    slow = (wait + wl.dur) / wl.dur
    window = wl.submit[c].max() - wl.submit[c].min() + 1
    gpu_slots = float((wl.gpus[c] * wl.dur[c]).sum())
    capacity_total = float(np.sum(np.broadcast_to(cap, (len(envs),))))
    if now_ok is None:
        now_ok = np.ones(len(wl), bool)
    vol = shifted & now_ok      # the scheduler chose to wait although it could start now
    forced = shifted & ~now_ok  # capacity was full at submission
    out = {
        "n_jobs": int(c.sum()),
        "n_deferrable": int(dfr.sum()),
        "emissions_kg": float(E[c].sum() / 1e3),
        "emissions_submit_kg": float(E0[c].sum() / 1e3),
        "emissions_deferrable_kg": float(E[dfr].sum() / 1e3),
        "energy_mwh": float((wl.kw[c] * wl.dur[c]).sum() / SPH / 1e3),
        "n_shifted": int(shifted.sum()),
        "shift_rate": float(shifted.sum() / max(dfr.sum(), 1)),
        "n_harmed": int(harmed.sum()),
        "harm_rate": float(harmed.sum() / max(shifted.sum(), 1)),
        "harm_rate_of_deferrable": float(harmed.sum() / max(dfr.sum(), 1)),
        "harm_rate_voluntary": float((harmed & vol).sum() / max(vol.sum(), 1)),
        "harm_rate_forced": float((harmed & forced).sum() / max(forced.sum(), 1)),
        "forced_frac_of_shifted": float(forced.sum() / max(shifted.sum(), 1)),
        "harm_excess_kg": float((E - E0)[harmed].sum() / 1e3),
        "shift_gain_kg": float((E0 - E)[shifted].sum() / 1e3),
        "n_relocated": int((c & (region != wl.home)).sum()),
        "deadline_miss_rate": float(miss.sum() / max(dfr.sum(), 1)),
        "slowdown_mean": float(slow[c].mean()),
        "slowdown_p95": float(np.percentile(slow[c], 95)),
        "slowdown_mean_deferrable": float(slow[dfr].mean()) if dfr.any() else float("nan"),
        "wait_h_mean_deferrable": float(wait[dfr].mean() / SPH) if dfr.any() else float("nan"),
        "utilisation": gpu_slots / (capacity_total * window),
        "capacity_gpus": capacity_total,
    }
    if decision_s:
        ms = np.asarray(decision_s) * 1e3
        out.update(decision_ms_median=float(np.median(ms)), decision_ms_p99=float(np.percentile(ms, 99)),
                   n_decisions=int(len(ms)))
    nd = c & ~wl.deferrable
    out["wait_min_mean_nondeferrable"] = float(wait[nd].mean() * 5) if nd.any() else float("nan")
    out["wait_h_mean_forced"] = float(wait[forced].mean() / SPH) if forced.any() else float("nan")
    if diag is not None:
        out.update(decompose(envs, wl, E, E0, shifted, harmed, dfr, diag))
    return out, E, E0


def decompose(envs, wl, E, E0, shifted, harmed, dfr, diag):
    """Split each harmed job's excess emissions (E - E0) exactly into three parts:
      forecast  = E(wanted at arrival) - E0      the slot the rule chose at arrival, ignoring
                                                capacity, priced at actual CI (<= 0 for the oracle)
      replan    = E(wanted at last decision) - E(wanted at arrival)
                                                later forecasts moved the target
      contention= E - E(wanted at last decision) the job did not get the slot its rule wanted
                                                because other reservations filled it
    plus per-job decision counts (the re-planning mechanism)."""
    d = wl.dur.astype(np.int64)

    def price(st, rg):
        out = np.full(len(wl), np.nan)
        for r, e in enumerate(envs):
            m = (rg == r) & (st >= 0)
            out[m] = e.actual_cost(st[m], d[m]) * wl.kw[m]
        return out

    E_des = price(diag["des_start"], diag["des_region"])
    E_last = price(diag["last_start"], diag["last_region"])
    h = harmed & np.isfinite(E_des) & np.isfinite(E_last)
    fc = (E_des - E0)[h].sum()
    rp = (E_last - E_des)[h].sum()
    ct = (E - E_last)[h].sum()
    tot = fc + rp + ct
    got = shifted & (diag["start"] == diag["last_start"]) & (diag["region"] == diag["last_region"])
    n_dec, n_defer, n_move = diag["n_dec"], diag["n_defer"], diag["n_move"]
    sh_nh = shifted & ~harmed
    return {
        "harm_excess_forecast_kg": float(fc / 1e3),
        "harm_excess_replan_kg": float(rp / 1e3),
        "harm_excess_contention_kg": float(ct / 1e3),
        "harm_share_forecast": float(fc / tot) if tot > 0 else float("nan"),
        "harm_share_replan": float(rp / tot) if tot > 0 else float("nan"),
        "harm_share_contention": float(ct / tot) if tot > 0 else float("nan"),
        # harmed job whose wanted-at-arrival slot already emits more than running now
        "harmed_frac_forecast": float(((E_des > E0 * (1 + 1e-9)) & h).sum() / max(h.sum(), 1)),
        "got_wanted_slot_frac": float(got.sum() / shifted.sum()) if shifted.any() else float("nan"),
        "decisions_per_job": float(n_dec[dfr].mean()) if dfr.any() else float("nan"),
        "defer_decisions_per_job": float(n_defer[dfr].mean()) if dfr.any() else float("nan"),
        "moves_per_job": float(n_move[dfr].mean()) if dfr.any() else float("nan"),
        "defer_decisions_harmed": float(n_defer[harmed].mean()) if harmed.any() else float("nan"),
        "defer_decisions_unharmed": float(n_defer[sh_nh].mean()) if sh_nh.any() else float("nan"),
    }
