"""
Synthetic e-commerce orders where WE inject known incidents.
Because we know the true cause, we can test whether the explainer finds it.
(The tool also works on your own CSV, for example the Olist e-commerce data.)

One row = one order. Columns: date, revenue, channel, device, country, category, customer_type.
"""
import numpy as np
import pandas as pd

START = pd.Timestamp("2026-07-06")      # a Monday, so 14-day windows hold whole weeks
DIMS = ["channel", "device", "country", "category", "customer_type"]
P = {
    "channel": (["organic", "paid_search", "social", "email", "direct"], [.30, .25, .15, .10, .20]),
    "device": (["mobile", "desktop", "tablet"], [.60, .32, .08]),
    "country": (["IN", "US", "UK", "DE"], [.40, .30, .15, .15]),
    "category": (["electronics", "fashion", "home", "beauty"], [.20, .30, .25, .25]),
    "customer_type": (["new", "returning"], [.45, .55]),
}
CAT_BASE = {"electronics": 120, "fashion": 45, "home": 70, "beauty": 30}
CHAN_MULT = {"organic": 1.05, "paid_search": 1.0, "social": 0.5, "email": 1.15, "direct": 1.1}
DEV_MULT = {"desktop": 1.1, "mobile": 0.95, "tablet": 1.0}


def _mask(df, pred):
    m = np.ones(len(df), bool)
    for k, v in pred.items():
        m &= (df[k] == v).to_numpy()
    return m


def _drop(df, rng, pred, frac, after):
    hit = _mask(df, pred) & after
    return df[~(hit & (rng.random(len(df)) < frac))]


def _add(df, rng, pred, frac, after):
    """Add extra orders by copying matching rows. frac=1.5 means 150% more orders."""
    hit = _mask(df, pred) & after
    whole, rest = int(frac), frac - int(frac)
    parts = [df] + [df[hit]] * whole + [df[hit & (rng.random(len(df)) < rest)]]
    return pd.concat(parts, ignore_index=True)


def _scale(df, pred, factor, after):
    df = df.copy()
    hit = _mask(df, pred) & after
    df.loc[hit, "revenue"] = df.loc[hit, "revenue"] * factor
    return df


# name -> (description, metric to explain, true cause)
INCIDENTS = {
    "none": ("Nothing changes. Only normal day-to-day noise.", "revenue", None),
    "channel_outage": ("Paid search orders fall by 45% (ads stopped serving).", "revenue", {"channel": "paid_search"}),
    "mobile_bug": ("A checkout bug removes 30% of mobile orders.", "revenue", {"device": "mobile"}),
    "intersection": ("Only mobile orders from paid search fall by 60%.", "revenue", {"device": "mobile", "channel": "paid_search"}),
    "price_cut": ("Electronics prices fall 30%, orders rise 10%. Net revenue falls.", "revenue", {"category": "electronics"}),
    "mix_shift": ("Social orders rise 150%. Social orders are smaller, so average order value falls.", "aov", {"channel": "social"}),
}


def make_dataset(seed=0, days=70, incident="none", incident_day=56, base_orders=450):
    rng = np.random.default_rng(seed)
    d = np.arange(days)
    lam = base_orders * (1 + 0.18 * (d % 7 >= 5)) * rng.lognormal(0, 0.04, days)
    n_day = rng.poisson(lam)
    day = np.repeat(d, n_day)
    n = len(day)
    cols = {k: rng.choice(v[0], n, p=v[1]) for k, v in P.items()}
    df = pd.DataFrame(cols)
    base = (df["category"].map(CAT_BASE) * df["channel"].map(CHAN_MULT) * df["device"].map(DEV_MULT)).to_numpy()
    df["revenue"] = np.round(base * rng.lognormal(0, 0.45, n), 2)
    df.insert(0, "date", START + pd.to_timedelta(day, unit="D"))
    after = (day >= incident_day)

    if incident == "channel_outage":
        df = _drop(df, rng, {"channel": "paid_search"}, 0.45, after)
    elif incident == "mobile_bug":
        df = _drop(df, rng, {"device": "mobile"}, 0.30, after)
    elif incident == "intersection":
        df = _drop(df, rng, {"device": "mobile", "channel": "paid_search"}, 0.60, after)
    elif incident == "price_cut":
        df = _scale(df, {"category": "electronics"}, 0.7, after)
        after2 = (df["date"] >= START + pd.Timedelta(days=incident_day)).to_numpy()
        df = _add(df, rng, {"category": "electronics"}, 0.10, after2)
    elif incident == "mix_shift":
        df = _add(df, rng, {"channel": "social"}, 1.5, after)
    elif incident != "none":
        raise ValueError(f"unknown incident {incident}")
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "order_id", np.arange(len(df)))
    return df


def default_windows(days=70, incident_day=56, length=14):
    f = lambda i: str((START + pd.Timedelta(days=i)).date())
    return {"before_start": f(incident_day - length), "before_end": f(incident_day - 1),
            "after_start": f(incident_day), "after_end": f(incident_day + length - 1)}
