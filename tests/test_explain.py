import pytest
from metricexplainer.generator import INCIDENTS, make_dataset, default_windows
from metricexplainer.explain import explain, detect_dims

W = default_windows()
B, A = (W["before_start"], W["before_end"]), (W["after_start"], W["after_end"])


@pytest.mark.parametrize("incident", [k for k, v in INCIDENTS.items() if v[2]])
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_finds_injected_cause(incident, seed):
    _, metric, truth = INCIDENTS[incident]
    r = explain(make_dataset(seed=seed, incident=incident), metric, B, A)
    assert r["causes"][0]["segment"] == truth


@pytest.mark.parametrize("incident", ["mobile_bug", "mix_shift"])
def test_strong_incidents_are_called_real(incident):
    _, metric, _ = INCIDENTS[incident]
    for seed in [3, 4, 5]:
        assert explain(make_dataset(seed=seed, incident=incident), metric, B, A)["total"]["real"]


def test_no_incident_is_not_called_real():
    # The false alarm rate is about 5% by design (measured at ~6% over 150 runs), so a few
    # fixed seeds are used to keep the test deterministic.
    for seed in [3, 4, 5, 6]:
        r = explain(make_dataset(seed=seed, incident="none"), "revenue", B, A)
        assert not r["total"]["real"]
        assert "Do not claim a cause" in r["summary"]


def test_revenue_volume_plus_value_equals_change():
    r = explain(make_dataset(seed=3, incident="price_cut"), "revenue", B, A)
    t = r["total"]
    assert abs(t["volume_effect"] + t["value_effect"] - t["delta"]) < 1e-6 * abs(t["before"])


def test_each_dimension_contributions_sum_to_total():
    r = explain(make_dataset(seed=3, incident="mobile_bug"), "revenue", B, A)
    for d in r["dimensions"]:
        assert abs(sum(s["delta"] for s in d["segments"]) - r["total"]["delta"]) < 1e-6 * r["total"]["before"]


def test_detect_dims():
    assert set(detect_dims(make_dataset(seed=1))) == {"channel", "device", "country", "category", "customer_type"}


def test_permutation_test_is_calibrated_on_a_simple_case():
    import numpy as np
    from metricexplainer.explain import _perm_pvalue
    rng = np.random.default_rng(0)
    rb = np.full(10, 100.0); cb = np.ones(10)
    assert _perm_pvalue(rb, cb, rb.copy(), cb, "revenue", rng) == 1.0         # identical periods
    ra = rb + 20                                                              # +20 every single day
    assert abs(_perm_pvalue(rb, cb, ra, cb, "revenue", rng) - 2 / 2 ** 10) < 1e-12   # only 2 of 1024 patterns match


def test_same_length_windows_with_different_weekdays_are_not_paired():
    import pandas as pd
    from metricexplainer.explain import significance
    df = make_dataset(seed=11, incident="none")
    # Monday-to-Sunday window followed by Tuesday-to-Monday: same length, weekday mismatch.
    result = significance(df, "revenue", ("2026-07-06", "2026-07-19"),
                          ("2026-07-21", "2026-08-03"), n_boot=100, seed=1)
    assert result["paired"] is False
    assert any("could not be safely paired" in note for note in result["notes"])


def test_explain_works_without_categorical_dimensions():
    import pandas as pd
    from metricexplainer.explain import explain
    dates = pd.date_range("2026-07-06", periods=28, freq="D")
    df = pd.DataFrame({"date": dates.repeat(3), "revenue": [100.0] * 42 + [120.0] * 42})
    result = explain(df, "revenue", ("2026-07-06", "2026-07-19"),
                     ("2026-07-20", "2026-08-02"))
    assert result["metric"] == "revenue"
    assert result["dimensions"][0]["dimension"] == "__all_orders__"
