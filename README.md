# Metric Movement Explainer

"Revenue fell. Why?" Give it an orders table and two periods. It tells you:
1. whether the change is real or just normal day-to-day noise,
2. which segments (or pairs of segments) caused it,
3. whether it was fewer orders, smaller orders, or a shift in the mix of orders.

Everything in `decompose.py` and the permutation test is written by hand. pandas is used only for
grouping rows. The tool is tested on fake orders where we inject a known cause and check that it is found.

## Run it
```bash
pip install -r requirements.txt
python -m pytest -q             # 28 tests
uvicorn app:app --reload        # open http://127.0.0.1:8000
```
The dashboard has demo datasets with a hidden cause, and a file box for your own CSV
(columns `date` and `revenue`, one row per order, any other text columns become dimensions).
The Olist e-commerce dataset works once you join its tables into one row per order.

## What is inside
| File | What it does |
|---|---|
| `metricexplainer/decompose.py` | Contribution, excess, surprise (Jensen-Shannon), volume vs value split, mix vs rate split |
| `metricexplainer/explain.py` | Permutation test, dimension ranking, pair search, simplest-explanation rule, plain-English summary |
| `metricexplainer/generator.py` | Fake orders with 5 injectable incidents and known truth |
| `metricexplainer/evaluate.py` | Recovery study: does the tool find the injected cause? |
| `app.py` | FastAPI: `/demo`, `/dataset`, `/explain`, `/study/recovery` |
| `static/index.html` | Dashboard: verdict, waterfall, likely causes, dimension ranking, study |

## The ideas
- **Real or noise?** Exact paired permutation test. Day i of "before" is paired with day i of "after" (same weekday). If nothing changed, swapping the two labels on any day should not matter, so we try every swap pattern (2^14 for 14 days) and see how unusual the real change is.
- **Contribution.** For revenue and orders, each segment's change adds up to the total change.
- **Excess.** Each segment is compared with how the REST of the business moved. This stops a healthy segment from looking like a hero just because the broken one dragged the overall trend down.
- **Volume vs value.** Revenue = orders x average order value. The change splits exactly into "more or fewer orders" and "bigger or smaller orders".
- **Mix vs rate.** A drop in average order value can happen with no price change at all, when the mix shifts toward cheaper orders. The split is exact.
- **Surprise.** Jensen-Shannon distance between the before and after shares. It ranks which dimension changed in the oddest way. The idea comes from the Adtributor paper (Microsoft Research). Read the paper yourself before an interview.
- **Pairs.** A pair like "mobile AND paid search" is reported only if it holds most of each parent's movement using a much smaller slice of it. Otherwise half of any broken segment would look like a cause.

## Recovery study (typical results, 15-30 runs per case)
| Injected incident | Flagged real | Right cause ranked 1st | Naive method 1st |
|---|---|---|---|
| none | about 6-10% (false alarms) | - | - |
| paid search outage | about 95% | 100% | 100% |
| mobile checkout bug | 100% | 100% | 100% |
| mobile x paid search only | about 60-73% | 100% | **0%** |
| electronics price cut | about 60-73% | about 93-97% | 100% |
| social surge (AOV falls) | 100% | about 97-100% | about 97-100% |

The naive method (pick the single segment with the biggest raw change) cannot find a cause that lives in an intersection.
Smaller incidents are flagged less often. That is honest: 14 days of noisy data cannot prove a 5-9% change every time. Longer windows help.

## Known limits (say these yourself in interviews)
- It finds segments that moved. That does not prove they caused the movement. A segment that moved is where to look next, not the final answer.
- Only pairs of dimensions, not triples. Pairs are not searched for average order value.
- Smaller causes can hide behind a big one. Lower entries in the cause list may be noise, and are not tested one by one.
- The permutation test assumes the day-to-day differences would be symmetric if nothing changed. A holiday or sale inside one window breaks that.
- Three metrics only (revenue, orders, average order value). A funnel (sessions x conversion x order value) would be a natural next step.
- Synthetic data is cleaner than real data. Real data also has missing values, refunds, and late-arriving orders.
