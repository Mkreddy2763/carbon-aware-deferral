"""Policy definitions. Every policy is a decision rule plugged into the shared
engine (carbonsched.sim.engine); none owns capacity or reservation logic.

rule:
  fcfs       carbon-agnostic: earliest feasible start (P0)
  threshold  run when current CI <= rolling 30th percentile of the past week,
             otherwise hold until the latest feasible start (P1)
  greedy     minimise point-forecast emissions over candidate starts (P2/P3)
  robust     ours: 'run now' is valued at the expected cost (current hour known,
             point forecast for the rest); every deferred option is valued at its
             conformal upper bound. Defer only if the best deferred upper bound
             beats running now (P4)
  lam        minimise sum of (forecast + lam * interval width) over the run (P4-lam)
  oracle     minimise actual emissions (perfect foresight) (P5)
  jitter     P9: uniform choice among the k best slots by point forecast (herding test)
  cdec       ours: point-greedy choice, but defer only if the predicted saving exceeds a
             conformal margin calibrated on the greedy decision itself (P7)
replan: re-decide every not-yet-started deferrable job at each hour boundary
        with the newly issued forecast (and ACI-updated bounds).
spatial: candidates span all regions, with a migration delay for non-home regions.
"""

POLICIES = {
    "P0": dict(rule="fcfs"),
    "P1": dict(rule="threshold", replan=True),
    "P2": dict(rule="greedy"),
    "P3": dict(rule="greedy", replan=True),
    "P4": dict(rule="robust", replan=True),
    "P4-l0.25": dict(rule="lam", lam=0.25, replan=True),
    "P4-l0.5": dict(rule="lam", lam=0.5, replan=True),
    "P4-l1": dict(rule="lam", lam=1.0, replan=True),
    "P5": dict(rule="oracle"),
    # decision-level conformal (ours): calibrate the greedy decision's saving, not hourly CI
    "P7": dict(rule="cdec", replan=True),
    "P7-noreplan": dict(rule="cdec"),
    # spatio-temporal variants (E5)
    "P6": dict(rule="robust", replan=True, spatial=True),
    "P6-point": dict(rule="greedy", replan=True, spatial=True),
    "P6-oracle": dict(rule="oracle", spatial=True),
    "P6-cdec": dict(rule="cdec", replan=True, spatial=True),
    # v2 congestion-aware deferral
    # P8: every policy already reserves capacity in its target slot (shared reservation
    # calendar), so literal "reserve the target" is the baseline. P8 adds the missing
    # contention awareness: a *deferred* reservation may only use (1 - h) of any slot's
    # capacity, keeping h*C free for later arrivals; running now may use all of it.
    "P8": dict(rule="greedy", replan=True, headroom=0.25),
    "P8-h0.5": dict(rule="greedy", replan=True, headroom=0.5),
    "P8-noreplan": dict(rule="greedy", headroom=0.25),
    # P9: spread deferrals uniformly over the k=3 best slots instead of the argmin
    "P9": dict(rule="jitter", k=3),
    "P9-r": dict(rule="jitter", k=3, replan=True),
    # denser lambda dial
    "P4-l0.1": dict(rule="lam", lam=0.1, replan=True),
    "P4-l2": dict(rule="lam", lam=2.0, replan=True),
    # ablation: robust without re-planning
    "P4-noreplan": dict(rule="robust"),
}


def get(name):
    p = dict(POLICIES[name])
    p.setdefault("replan", False)
    p.setdefault("spatial", False)
    p.setdefault("lam", 0.0)
    p["name"] = name
    return p
