"""FastAPI backend. Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000"""
import io
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from metricexplainer.generator import INCIDENTS, make_dataset, default_windows
from metricexplainer.explain import explain, detect_dims, METRICS
from metricexplainer.evaluate import recovery_study

app = FastAPI(title="Metric Movement Explainer")
STATE = {"df": None, "info": None}
_cache = {}


def _info(df, source, windows=None, truth=None, description=""):
    end = df["date"].max()
    if windows is None:                         # default: last 14 days vs the 14 before
        f = lambda t: str(t.date())
        windows = {"before_start": f(end - pd.Timedelta(days=27)), "before_end": f(end - pd.Timedelta(days=14)),
                   "after_start": f(end - pd.Timedelta(days=13)), "after_end": f(end)}
    return {"source": source, "rows": int(len(df)), "start": str(df["date"].min().date()),
            "end": str(end.date()), "dimensions": detect_dims(df), "windows": windows,
            "truth": truth, "description": description}


def _load_demo(incident, seed):
    df = make_dataset(seed=seed, incident=incident)
    desc, metric, truth = INCIDENTS[incident]
    STATE["df"] = df
    STATE["info"] = {**_info(df, f"demo: {incident}", default_windows(), truth, desc), "suggested_metric": metric}


_load_demo("channel_outage", 7)


@app.get("/meta")
def meta():
    return {"incidents": {k: v[0] for k, v in INCIDENTS.items()}, "metrics": METRICS, "info": STATE["info"]}


class DemoRequest(BaseModel):
    incident: str
    seed: int = 7


@app.post("/demo")
def demo(req: DemoRequest):
    if req.incident not in INCIDENTS:
        raise HTTPException(404, f"incident must be one of {list(INCIDENTS)}")
    _load_demo(req.incident, req.seed)
    return STATE["info"]


class CsvRequest(BaseModel):
    csv: str


@app.post("/dataset")
def dataset(req: CsvRequest):
    """Load your own orders CSV. Needs a `date` column and a numeric `revenue` column. One row = one order."""
    try:
        df = pd.read_csv(io.StringIO(req.csv))
    except Exception:
        raise HTTPException(422, "Could not read the CSV.")
    if not {"date", "revenue"} <= set(df.columns):
        raise HTTPException(422, "The CSV needs columns named `date` and `revenue`.")
    if len(df) > 500_000:
        raise HTTPException(422, "Too many rows (limit 500,000).")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["date", "revenue"]).reset_index(drop=True)
    if df.empty or df["date"].nunique() < 14:
        raise HTTPException(422, "Need valid dates and revenue, covering at least 14 different days.")
    for c in df.columns:
        if c not in ("date", "revenue", "order_id") and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str)
    STATE["df"] = df
    STATE["info"] = _info(df, "your CSV")
    return STATE["info"]


class ExplainRequest(BaseModel):
    metric: str = "revenue"
    before_start: str
    before_end: str
    after_start: str
    after_end: str


@app.post("/explain")
def explain_endpoint(req: ExplainRequest):
    if req.metric not in METRICS:
        raise HTTPException(422, f"metric must be one of {METRICS}")
    try:
        b = (pd.Timestamp(req.before_start), pd.Timestamp(req.before_end))
        a = (pd.Timestamp(req.after_start), pd.Timestamp(req.after_end))
    except Exception:
        raise HTTPException(422, "Dates must look like 2026-09-01.")
    if b[0] > b[1] or a[0] > a[1]:
        raise HTTPException(422, "Each period needs a start date on or before its end date.")
    if not (b[1] < a[0]):
        raise HTTPException(422, "The 'before' period must end before the 'after' period starts.")
    try:
        out = explain(STATE["df"], req.metric, (str(b[0].date()), str(b[1].date())), (str(a[0].date()), str(a[1].date())))
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["info"] = STATE["info"]
    return out


@app.get("/study/recovery")
def study_recovery():
    if "rec" not in _cache:
        _cache["rec"] = recovery_study()
    return _cache["rec"]


@app.get("/")
def home():
    return FileResponse("static/index.html")
