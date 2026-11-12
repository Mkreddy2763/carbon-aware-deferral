"""Build a simulation workload from the normalised trace (data/trace/jobs.parquet).

Normalised trace columns: submit_s (seconds since trace t=0), dur_s, gpus, util.
data/trace/meta.json gives `monday0_s`: the trace time of the first Monday 00:00
(trace-local), estimated from the weekly submission pattern (see data/trace.py).

Time mapping: trace day `trace_day0` (counted from monday0, so a Monday) is
mapped to a region-local Monday 00:00 in the carbon test period, plus
`offset_weeks`. Hour-of-week is therefore preserved.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import Workload

SLOT_S = 300
SPH = 12


@dataclass
class EnergyModel:
    p_gpu_w: float = 300.0
    idle_frac: float = 0.3
    pue: float = 1.2
    default_util: float = 0.7

    def kw(self, gpus, util):
        u = np.where(np.isfinite(util), util, self.default_util)
        return gpus * self.p_gpu_w * (self.idle_frac + (1 - self.idle_frac) * u) * self.pue / 1e3


def build(trace, monday0_s, *, window_days=21, warmup_days=1, trace_day0=7,
          slack_h=12, defer_min_h=1.0, dur_cap_days=7, sample_rate=1.0,
          dur_noise_sigma=0.0, n_regions=1, energy=None, seed=0):
    """Returns (Workload, info). Slot 0 = start of the warm-up day."""
    energy = energy or EnergyModel()
    t_start = monday0_s + trace_day0 * 86400 - warmup_days * 86400
    t_end = monday0_s + (trace_day0 + window_days) * 86400
    df = trace[(trace.submit_s >= t_start) & (trace.submit_s < t_end)]
    rng = np.random.default_rng(seed)
    if sample_rate < 1.0:
        df = df[rng.random(len(df)) < sample_rate]
    df = df.sort_values("submit_s", kind="stable")
    submit = ((df.submit_s.to_numpy() - t_start) // SLOT_S).astype(np.int64)
    dur_s = np.minimum(df.dur_s.to_numpy(), dur_cap_days * 86400)
    dur = np.maximum(1, np.ceil(dur_s / SLOT_S)).astype(np.int64)
    if dur_noise_sigma > 0:
        rng_d = np.random.default_rng(seed + 1)
        est = dur * np.exp(dur_noise_sigma * rng_d.standard_normal(len(dur)))
        dur_est = np.maximum(1, np.round(est)).astype(np.int64)
    else:
        dur_est = dur.copy()
    gpus = df.gpus.to_numpy(dtype=float)
    util = df.util.to_numpy(dtype=float) if "util" in df else np.full(len(df), np.nan)
    kw = energy.kw(gpus, util)
    deferrable = dur_est >= defer_min_h * SPH
    deadline = submit + dur + int(round(slack_h * SPH))
    home = rng.integers(0, n_regions, len(df)) if n_regions > 1 else np.zeros(len(df), np.int64)
    counted = submit >= warmup_days * 86400 // SLOT_S
    wl = Workload(submit=submit, dur=dur, dur_est=dur_est, gpus=gpus, kw=kw,
                  deferrable=deferrable, deadline=deadline, home=home.astype(np.int64),
                  counted=counted)
    info = {"n_jobs": int(counted.sum()), "n_warmup": int((~counted).sum()),
            "frac_deferrable": float(deferrable[counted].mean()),
            "gpu_hours": float((gpus * dur)[counted].sum() / SPH),
            "gpu_hours_deferrable": float((gpus * dur)[counted & deferrable].sum() / SPH),
            "frac_gpu_hours_capped": float(((df.dur_s.to_numpy() > dur_cap_days * 86400) * gpus *
                                            df.dur_s.to_numpy())[counted].sum() /
                                           max((gpus * df.dur_s.to_numpy())[counted].sum(), 1)),
            "window_slots": int(window_days * 86400 // SLOT_S)}
    return wl, info


def capacity_for(wl, util_target, window_days):
    """Capacity giving ~util_target mean utilisation for the carbon-agnostic baseline."""
    c = wl.counted
    demand = float((wl.gpus[c] * wl.dur[c]).sum()) / (window_days * 86400 / SLOT_S)
    return max(demand / util_target, float(wl.gpus.max()))


def region_monday(ts_index_start_utc, tz, after):
    """First local Monday 00:00 at/after `after` (tz-aware or UTC str) -> UTC Timestamp."""
    t = pd.Timestamp(after)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    loc = t.tz_convert(tz).normalize()
    days = (7 - loc.dayofweek) % 7
    loc = loc + pd.Timedelta(days=int(days))
    if loc < t.tz_convert(tz):
        loc = loc + pd.Timedelta(days=7)
    return loc.tz_convert("UTC")
