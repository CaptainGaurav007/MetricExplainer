"""
Recovery study: inject a known incident, run the explainer, and check whether it finds the truth.
This is the proof that the tool works, and where it fails.
"""
from .generator import INCIDENTS, make_dataset, default_windows
from .explain import explain


def _hit(segment, truth):
    return segment == truth


def raw_contribution_baseline(result):
    """Naive method: the single segment (one dimension) with the biggest raw change, same direction as total."""
    sign = 1 if result["total"]["delta"] >= 0 else -1
    best = None
    for d in result["dimensions"]:
        for r in d["segments"]:
            if r["delta"] * sign > 0 and (best is None or abs(r["delta"]) > abs(best[1])):
                best = ({d["dimension"]: r["key"][0]}, r["delta"])
    return best[0] if best else None


def recovery_study(n=15, seed0=100):
    w = default_windows()
    before, after = (w["before_start"], w["before_end"]), (w["after_start"], w["after_end"])
    rows = []
    for name, (desc, metric, truth) in INCIDENTS.items():
        real = top1 = top3 = base1 = 0
        for i in range(n):
            res = explain(make_dataset(seed=seed0 + i, incident=name), metric, before, after)
            real += res["total"]["real"]
            if truth is not None:
                segs = [c["segment"] for c in res["causes"]]
                top1 += bool(segs) and _hit(segs[0], truth)
                top3 += any(_hit(s, truth) for s in segs[:3])
                base1 += _hit(raw_contribution_baseline(res), truth)
        rows.append({"incident": name, "description": desc, "metric": metric, "truth": truth,
                     "flagged_real_rate": real / n,
                     "top1": None if truth is None else top1 / n,
                     "top3": None if truth is None else top3 / n,
                     "raw_baseline_top1": None if truth is None else base1 / n})
    return {"n_per_incident": n, "rows": rows}
