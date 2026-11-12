"""Multi-horizon carbon-intensity forecasters.

All forecasters return a matrix F of shape (n_issue, H): F[i, h-1] is the forecast,
made at issue hour t_i (using data up to and including hour t_i), of y[t_i + h].
Hours are integer positions into the hourly series y.

F0  seasonal naive: same hour on the most recent day already observed.
F1  LightGBM, direct multi-horizon, one pooled model with horizon as a feature;
    target is y[t+h] - y[t] (persistence residual), which copes with the
    multi-year decline in grid CI better than the raw level.
F2  operator forecast (GB only): the published NESO forecast for the target slot.
"""
import numpy as np
import pandas as pd

H = 48


def target_matrix(y, issue):
    """Y[i, h-1] = y[issue_i + h] (NaN beyond the series)."""
    n = len(y)
    idx = issue[:, None] + np.arange(1, H + 1)[None, :]
    out = np.full(idx.shape, np.nan)
    ok = idx < n
    out[ok] = y[idx[ok]]
    return out


def seasonal_naive(y, issue):
    h = np.arange(1, H + 1)
    back = 24 * np.ceil(h / 24).astype(int)
    idx = issue[:, None] + h[None, :] - back[None, :]
    return y[idx]


def operator(op, issue):
    return target_matrix(op, issue)


# ---------------------------------------------------------------- LightGBM

_LAGS = [0, 1, 2, 3, 6, 12, 23]


def _base_features(y, ts_utc, tz):
    """Per-issue-hour features (only uses data up to the issue hour)."""
    s = pd.Series(y)
    feats = {f"lag{l}": s.shift(l).to_numpy() for l in _LAGS}
    feats["rm24"] = s.rolling(24, min_periods=12).mean().to_numpy()
    feats["rs24"] = s.rolling(24, min_periods=12).std().to_numpy()
    feats["rm168"] = s.rolling(168, min_periods=48).mean().to_numpy()
    feats["d24"] = (s - s.shift(24)).to_numpy()
    return pd.DataFrame(feats)


def _design(y, ts_utc, tz, issue, base, hs):
    """Stack (issue, horizon) rows. Returns X (DataFrame), rows ordered issue-major."""
    n_i, n_h = len(issue), len(hs)
    ii = np.repeat(issue, n_h)
    hh = np.tile(hs, n_i)
    X = base.iloc[ii].reset_index(drop=True)
    X["h"] = hh
    tgt = ii + hh
    back = 24 * np.ceil(hh / 24).astype(int)
    X["sn1"] = y[tgt - back]
    X["sn7"] = y[np.maximum(tgt - 168, 0)]
    loc = pd.DatetimeIndex(ts_utc).tz_convert(tz)
    tgt_c = np.minimum(tgt, len(y) - 1)  # calendar of target hour (index may run past end)
    hod = loc.hour.to_numpy()
    dow = loc.dayofweek.to_numpy()
    doy = loc.dayofyear.to_numpy()
    # calendar computed by extrapolating hours past the end of the series
    extra = tgt - tgt_c
    X["t_hod"] = (hod[tgt_c] + extra) % 24
    X["t_dow"] = (dow[tgt_c] + (hod[tgt_c] + extra) // 24) % 7
    X["t_doy_s"] = np.sin(2 * np.pi * doy[tgt_c] / 365.25)
    X["t_doy_c"] = np.cos(2 * np.pi * doy[tgt_c] / 365.25)
    X["i_hod"] = hod[ii]
    return X, ii, hh


def fit_lgbm(y, ts_utc, tz, train_issue, seed=0, n_jobs=-1):
    import lightgbm as lgb
    base = _base_features(y, ts_utc, tz)
    hs = np.arange(1, H + 1)
    # drop issue hours whose targets fall outside the training span
    train_issue = train_issue[train_issue + H < train_issue.max() + 1]
    X, ii, hh = _design(y, ts_utc, tz, train_issue, base, hs)
    t = y[ii + hh] - y[ii]
    ok = np.isfinite(t) & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs,
                              verbose=-1)
    model.fit(X[ok], t[ok])
    return model


def predict_lgbm(model, y, ts_utc, tz, issue):
    base = _base_features(y, ts_utc, tz)
    hs = np.arange(1, H + 1)
    X, ii, hh = _design(y, ts_utc, tz, issue, base, hs)
    pred = model.predict(X) + y[ii]
    return np.clip(pred.reshape(len(issue), H), 0, None)
