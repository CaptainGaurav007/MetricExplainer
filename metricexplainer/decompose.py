"""
The core maths, written by hand. Pure functions, no pandas, so each one is easy to test.

Vocabulary
  before / after : value of a metric in a segment in the two periods
  delta          : after - before
  EP             : explanatory power = delta / total delta (share of the change this segment explains)
  excess         : how much more (or less) the segment moved than the REST of the business
  surprise       : Jensen-Shannon distance between before and after shares (how odd the pattern is)
"""
import math


# ---------- Surprise (idea from the Adtributor paper) ----------
def js_term(p, q):
    """One segment's part of the Jensen-Shannon divergence between share p (before) and q (after)."""
    def part(x, m):
        return x * math.log(x / m) if x > 0 else 0.0
    m = (p + q) / 2
    return 0.5 * (part(p, m) + part(q, m)) if m > 0 else 0.0


# ---------- Split of one segment's revenue change: volume vs value ----------
def volume_value_effects(o0, r0, o1, r1):
    """
    revenue = orders * AOV. The midpoint split is exact:
        r1 - r0 = (o1 - o0) * (a0 + a1)/2   [volume: more or fewer orders]
                + (a1 - a0) * (o0 + o1)/2   [value: bigger or smaller orders]
    If the segment has no orders in one period, AOV is undefined there, so it is all volume.
    """
    if o0 == 0 or o1 == 0:
        return r1 - r0, 0.0
    a0, a1 = r0 / o0, r1 / o1
    return (o1 - o0) * (a0 + a1) / 2, (a1 - a0) * (o0 + o1) / 2


# ---------- Split of the change in an average (AOV): mix vs rate ----------
def mix_rate_effects(segs):
    """
    segs: {key: (orders_before, revenue_before, orders_after, revenue_after)}
    AOV = sum over segments of (share of orders) * (segment AOV).
    Change in AOV, per segment:
        mix  = (w1 - w0) * ((a0 + a1)/2 - mean_AOV)   segment got bigger/smaller in the order mix
        rate = (a1 - a0) * (w0 + w1)/2                segment's own AOV changed
    Sum of all mix + rate effects = total AOV change, exactly.
    (Subtracting mean_AOV is allowed because the shares sum to 1 in both periods.)
    """
    O0 = sum(v[0] for v in segs.values()); O1 = sum(v[2] for v in segs.values())
    R0 = sum(v[1] for v in segs.values()); R1 = sum(v[3] for v in segs.values())
    A0, A1 = R0 / O0, R1 / O1
    centre = (A0 + A1) / 2
    out = {}
    for k, (o0, r0, o1, r1) in segs.items():
        w0, w1 = o0 / O0, o1 / O1
        a0 = r0 / o0 if o0 else None
        a1 = r1 / o1 if o1 else None
        a0 = a1 if a0 is None else a0            # no orders before: treat as pure mix
        a1 = a0 if a1 is None else a1
        out[k] = {"mix": (w1 - w0) * ((a0 + a1) / 2 - centre), "rate": (a1 - a0) * (w0 + w1) / 2,
                  "w0": w0, "w1": w1, "a0": a0, "a1": a1}
    return out, A1 - A0


# ---------- Additive metrics (revenue, orders) ----------
def additive_breakdown(before, after):
    """
    before, after: {segment key: value}. The values add up to the metric total.
    Returns one row per segment.
    'excess' compares the segment with how the REST of the business moved. This matters:
    if you compare with the overall trend, the broken segment drags that trend down and
    healthy segments look like they "outperformed".
    """
    keys = sorted(set(before) | set(after), key=str)
    T0, T1 = sum(before.values()), sum(after.values())
    dT = T1 - T0
    rows = []
    for k in keys:
        b, a = before.get(k, 0.0), after.get(k, 0.0)
        delta = a - b
        rest0 = T0 - b
        g_rest = (dT - delta) / rest0 if rest0 > 0 else 0.0
        p, q = (b / T0 if T0 else 0.0), (a / T1 if T1 else 0.0)
        rows.append({"key": k, "before": b, "after": a, "delta": delta,
                     "pct_change": (delta / b) if b > 0 else None,
                     "ep": (delta / dT) if dT else 0.0,
                     "expected": b * g_rest, "excess": delta - b * g_rest,
                     "share_before": p, "share_after": q, "js": js_term(p, q)})
    return rows, T0, T1


def select_explaining(rows, total_delta, tep=0.67, teep=0.10):
    """
    Smallest set of segments, biggest explainers first, that together explain at least
    `tep` of the change, where each one explains at least `teep` on its own.
    Only segments moving in the same direction as the total are considered.
    """
    sign = 1 if total_delta >= 0 else -1
    cand = sorted([r for r in rows if r["ep"] >= teep and r["delta"] * sign > 0],
                  key=lambda r: -r["ep"])
    chosen, covered = [], 0.0
    for r in cand:
        chosen.append(r["key"]); covered += r["ep"]
        if covered >= tep:
            break
    return chosen, covered
