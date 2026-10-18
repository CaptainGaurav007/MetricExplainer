"""
Turns an orders table into an answer to: "the metric moved. Why?"
Steps:
  1. Did it really move? (exact paired permutation test over days, so normal noise is not mistaken for a change)
  2. Break the change down by every dimension (decompose.py) and rank dimensions by surprise.
  3. Also try pairs of dimensions, to find causes that live in an intersection.
  4. Prefer the simplest explanation: a pair of dimensions replaces its parents only when it
     explains most of their movement. Write the result in plain English.
"""
import itertools
import numpy as np
import pandas as pd

from .decompose import additive_breakdown, mix_rate_effects, select_explaining, volume_value_effects, js_term

METRICS = ("revenue", "orders", "aov")


# ---------- helpers ----------
def detect_dims(df, max_unique=30):
    skip = {"date", "revenue", "order_id"}
    return [c for c in df.columns if c not in skip
            and not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
            and df[c].nunique() <= max_unique]


def _window(df, start, end):
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


def _table(df, dims):
    g = df.groupby(list(dims), observed=True)["revenue"].agg(["sum", "count"])
    return {(idx if isinstance(idx, tuple) else (idx,)): (float(s), int(c))
            for idx, (s, c) in zip(g.index, g.values)}


def _metric(rev, cnt, metric):
    if metric == "revenue":
        return rev
    if metric == "orders":
        return cnt
    return rev / cnt if cnt else 0.0


# ---------- step 1: did it really move? ----------
def _perm_pvalue(rb, cb, ra, ca, metric, rng, max_exact_days=14, n_random=20000, paired=None):
    """
    Permutation test. Idea: if NOTHING changed, "before" and "after" labels are meaningless,
    so swapping them should give changes just as large as the one we saw.
    Paired case (same number of days, day i matched with day i, same weekday):
      swap the before/after labels of each day pair in every possible way (2^n ways), and
      count how often the swapped change is at least as big as the real one.
    Unpaired case: shuffle all days between the two periods at random.
    """
    n = len(rb)
    obs = _metric(ra.sum(), ca.sum(), metric) - _metric(rb.sum(), cb.sum(), metric)
    if paired is None:
        paired = len(rb) == len(ra)
    if paired:
        if n <= max_exact_days:
            S = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)   # every swap pattern
            exact = True
        else:
            S = rng.integers(0, 2, (n_random, n)).astype(float)
            exact = False
        rb2, cb2 = (1 - S) @ rb + S @ ra, (1 - S) @ cb + S @ ca
        ra2, ca2 = (1 - S) @ ra + S @ rb, (1 - S) @ ca + S @ cb
    else:
        rall, call = np.concatenate([rb, ra]), np.concatenate([cb, ca])
        order = rng.random((n_random, len(rall))).argsort(1)
        pick = order[:, :len(rb)]
        mask = np.zeros((n_random, len(rall))); np.put_along_axis(mask, pick, 1.0, 1)
        rb2, cb2 = mask @ rall, mask @ call
        ra2, ca2 = rall.sum() - rb2, call.sum() - cb2
        exact = False
    stat = _metric_vec(ra2, ca2, metric) - _metric_vec(rb2, cb2, metric)
    count = np.sum(np.abs(stat) >= abs(obs) - 1e-9 * max(1.0, abs(obs)))
    return float(count / len(stat)) if exact else float((count + 1) / (len(stat) + 1))


def significance(df, metric, b, a, n_boot=1500, seed=0, alpha=0.05):
    """
    Daily revenue and order counts in each window. Two outputs:
      p_value : exact permutation test (decides "real" or "normal variation")
      ci_*    : bootstrap range for the % change (shows how big the change plausibly is)
    """
    rng = np.random.default_rng(seed)

    def daily(start, end):
        w = _window(df, start, end)
        days = pd.date_range(start, end)
        g = w.groupby("date")["revenue"].agg(["sum", "count"]).reindex(days, fill_value=0)
        return g["sum"].to_numpy(float), g["count"].to_numpy(float)

    rb, cb = daily(*b); ra, ca = daily(*a)
    nb, na = len(rb), len(ra)
    # Equal-length windows are paired only when day 1 maps to the same weekday.
    # Otherwise positional pairing would compare (for example) Monday with Tuesday.
    paired = (nb == na and pd.Timestamp(b[0]).weekday() == pd.Timestamp(a[0]).weekday())
    pval = _perm_pvalue(rb, cb, ra, ca, metric, rng, paired=paired)
    ib = rng.integers(0, nb, (n_boot, nb))
    ia = ib if paired else rng.integers(0, na, (n_boot, na))
    pct = _metric_vec(ra[ia].sum(1), ca[ia].sum(1), metric) / _metric_vec(rb[ib].sum(1), cb[ib].sum(1), metric) - 1
    obs = _metric(ra.sum(), ca.sum(), metric) / _metric(rb.sum(), cb.sum(), metric) - 1
    lo, hi = np.percentile(pct, [2.5, 97.5])
    notes = []
    if not paired:
        notes.append("The periods could not be safely paired by weekday, so an unpaired test was used. The test may have less power.")
    if nb < 7 or na < 7:
        notes.append("Fewer than 7 days in a period. Weekday effects can distort the result.")
    elif paired and nb % 7:
        notes.append("Periods are not whole weeks, so weekday mix may differ between them.")
    return {"pct_change": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": pval,
            "real": bool(pval < alpha), "paired": paired, "notes": notes,
            "before_days": nb, "after_days": na}


def _metric_vec(rev, cnt, metric):
    if metric == "revenue":
        return rev
    if metric == "orders":
        return cnt
    return rev / np.maximum(cnt, 1)


# ---------- step 2/3: break the change down ----------
def segment_rows(tb, ta, metric):
    """One row per segment. 'delta' is always the segment's CONTRIBUTION to the total change."""
    keys = sorted(set(tb) | set(ta), key=str)
    z = (0.0, 0)
    if metric == "aov":
        segs = {k: (tb.get(k, z)[1], tb.get(k, z)[0], ta.get(k, z)[1], ta.get(k, z)[0]) for k in keys}
        eff, dA = mix_rate_effects(segs)
        R0 = sum(v[1] for v in segs.values()); R1 = sum(v[3] for v in segs.values())
        rows = []
        for k, e in eff.items():
            o0, r0, o1, r1 = segs[k]
            contrib = e["mix"] + e["rate"]
            rows.append({"key": k, "before": e["a0"], "after": e["a1"], "delta": contrib,
                         "pct_change": (e["a1"] / e["a0"] - 1) if e["a0"] else None,
                         "ep": contrib / dA if dA else 0.0, "score": contrib,
                         "mix": e["mix"], "rate": e["rate"], "orders_before": o0, "orders_after": o1,
                         "share_before": e["w0"], "share_after": e["w1"],
                         "js": js_term(r0 / R0 if R0 else 0, r1 / R1 if R1 else 0)})
        T0, T1 = R0 / sum(v[0] for v in segs.values()), R1 / sum(v[2] for v in segs.values())
        return rows, T0, T1
    idx = 0 if metric == "revenue" else 1
    rows, T0, T1 = additive_breakdown({k: tb.get(k, z)[idx] for k in keys}, {k: ta.get(k, z)[idx] for k in keys})
    for r in rows:
        r["score"] = r["excess"]
        o0, o1 = tb.get(r["key"], z)[1], ta.get(r["key"], z)[1]
        r["orders_before"], r["orders_after"] = o0, o1
        if metric == "revenue":
            r["volume"], r["value"] = volume_value_effects(o0, tb.get(r["key"], z)[0], o1, ta.get(r["key"], z)[0])
    return rows, T0, T1


def _beats_parents(r, p1, p2):
    """
    A pair of dimensions (like mobile AND paid_search) is only worth reporting if it is
    CONCENTRATED: it must hold most of each parent's movement, using a much smaller slice of it.
      loss_fraction   = how much of the parent's change sits inside the pair
      volume_fraction = how big the pair is compared with the parent
    Without the concentration test, a random half of a broken segment looks like a "cause".
    """
    for p in (p1, p2):
        if p["delta"] == 0 or r["delta"] * p["delta"] <= 0 or p["before"] <= 0:
            return False
        loss_fraction = r["delta"] / p["delta"]
        volume_fraction = r["before"] / p["before"]
        if loss_fraction < 0.6 or loss_fraction / volume_fraction < 1.4:
            return False
    return True


def _kind(r, metric):
    if metric == "aov":
        a, b = abs(r["mix"]), abs(r["rate"])
        return "mix shift (more or fewer orders from this segment)" if a >= 2 * b else \
               "its own order value changed" if b >= 2 * a else "mix shift and order value"
    if metric == "revenue":
        a, b = abs(r["volume"]), abs(r["value"])
        return "fewer or more orders" if a >= 2 * b else \
               "order value changed" if b >= 2 * a else "orders and order value both changed"
    return "fewer or more orders"


def _label(cons):
    return " and ".join(f"{k} = {v}" for k, v in cons.items())


def explain(df, metric, before, after, dims=None, min_share=0.01, top_k=5, seed=0):
    """before / after are (start_date, end_date) tuples."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    dims = list(dims) if dims is not None else detect_dims(df)
    db, da = _window(df, *before).copy(), _window(df, *after).copy()
    if len(db) == 0 or len(da) == 0:
        raise ValueError("One of the periods has no orders. Check the dates.")
    if not dims:
        # A valid orders table can contain only date and revenue. Use one explicit
        # all-orders segment so aggregate explanations still work without dimensions.
        db["__all_orders__"] = "all orders"
        da["__all_orders__"] = "all orders"
        dims = ["__all_orders__"]

    sig = significance(df, metric, before, after, seed=seed)
    tot0 = _metric(db["revenue"].sum(), len(db), metric)
    tot1 = _metric(da["revenue"].sum(), len(da), metric)
    total = {"before": float(tot0), "after": float(tot1), "delta": float(tot1 - tot0), **sig}
    if metric == "revenue":
        rows, _, _ = segment_rows(_table(db, [dims[0]]), _table(da, [dims[0]]), "revenue")
        total["volume_effect"] = float(sum(r["volume"] for r in rows))
        total["value_effect"] = float(sum(r["value"] for r in rows))

    # one dimension at a time
    dim_out, one_way = [], {}
    for d in dims:
        rows, T0, T1 = segment_rows(_table(db, [d]), _table(da, [d]), metric)
        chosen, covered = select_explaining(rows, total["delta"])
        for r in rows:
            r["kind"] = _kind(r, metric)
            one_way[(d, r["key"][0])] = r
        dim_out.append({"dimension": d, "surprise": float(sum(r["js"] for r in rows)),
                        "explained_by_top": float(covered), "top_segments": [k[0] for k in chosen],
                        "segments": sorted(rows, key=lambda r: -abs(r["delta"]))})
    dim_out.sort(key=lambda x: -x["surprise"])

    # candidates: single segments, plus pairs (additive metrics only)
    big = lambda r: max(r["share_before"], r["share_after"]) >= min_share
    cands = [{"cons": {d: key}, "row": r} for (d, key), r in one_way.items() if big(r)]
    if metric != "aov":
        pairs = []
        for d1, d2 in itertools.combinations(dims, 2):
            rows, _, _ = segment_rows(_table(db, [d1, d2]), _table(da, [d1, d2]), metric)
            for r in rows:
                p1, p2 = one_way[(d1, r["key"][0])], one_way[(d2, r["key"][1])]
                if big(r) and _beats_parents(r, p1, p2):
                    r["kind"] = _kind(r, metric)
                    pairs.append({"cons": {d1: r["key"][0], d2: r["key"][1]}, "row": r,
                                  "parents": [(d1, r["key"][0]), (d2, r["key"][1])]})
        # simplest explanation: a pair that explains (nearly) as much as its parents replaces them
        drop = {p for c in pairs for p in c["parents"]}
        cands = [c for c in cands if next(iter(c["cons"].items())) not in drop] + pairs
    # keep only segments that moved in the SAME direction as the total, and drop weak ones
    sign = 1 if total["delta"] >= 0 else -1
    cands = [c for c in cands if c["row"]["score"] * sign > 0]
    cands.sort(key=lambda c: (-abs(c["row"]["score"]), -len(c["cons"])))
    if cands:
        cands = [c for c in cands if abs(c["row"]["score"]) >= 0.25 * abs(cands[0]["row"]["score"])]
    causes = []
    for c in cands[:top_k]:
        r = c["row"]
        item = {"segment": c["cons"], "label": _label(c["cons"]), "before": r["before"], "after": r["after"],
                "contribution": r["delta"], "share_of_change": r["ep"], "pct_change": r["pct_change"],
                "score": r["score"], "kind": r["kind"]}
        for k in ("volume", "value", "mix", "rate"):
            if k in r:
                item[k] = r[k]
        causes.append(item)

    return {"metric": metric, "total": total, "dimensions": dim_out, "causes": causes,
            "summary": summarize(metric, total, causes)}


def _p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def summarize(metric, total, causes):
    name = {"revenue": "Revenue", "orders": "Orders", "aov": "Average order value"}[metric]
    pct = total["pct_change"] * 100
    up = "rose" if total["delta"] > 0 else "fell"
    lo, hi = total["ci_low"] * 100, total["ci_high"] * 100
    head = f"{name} {up} {abs(pct):.1f}% (from {total['before']:,.1f} to {total['after']:,.1f})."
    if not total["real"]:
        return (head + f" Permutation test {_p(total['p_value'])}: a change this size is common from "
                f"normal day-to-day variation (plausible range {lo:+.1f}% to {hi:+.1f}%). Do not claim a cause.")
    if not causes:
        return head + " No single segment stands out."
    c = causes[0]
    share = c["share_of_change"] * 100
    line = (f" Most likely cause: {c['label']}. It contributes {c['contribution']:+,.1f}, "
            f"which is {share:.0f}% of the change, and it moved far more than the rest of the business. "
            f"Reason: {c['kind']}.")
    return head + f" Permutation test {_p(total['p_value'])}, so the change is real (plausible range {lo:+.1f}% to {hi:+.1f}%)." + line
