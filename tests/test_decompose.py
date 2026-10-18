import math, random
from metricexplainer.decompose import (additive_breakdown, volume_value_effects,
                                       mix_rate_effects, select_explaining, js_term)


def test_additive_deltas_sum_to_total():
    rows, T0, T1 = additive_breakdown({"a": 100, "b": 50, "c": 50}, {"a": 60, "b": 55, "c": 70})
    assert abs(sum(r["delta"] for r in rows) - (T1 - T0)) < 1e-9
    assert abs(sum(r["ep"] for r in rows) - 1.0) < 1e-9          # shares of the change add to 100%


def test_excess_uses_rest_of_business():
    # b collapses, others flat. Compared with the overall trend, a and c would look "up".
    rows, _, _ = additive_breakdown({"a": 100, "b": 100, "c": 100}, {"a": 100, "b": 40, "c": 100})
    ex = {r["key"]: r["excess"] for r in rows}
    assert abs(ex["b"] - (-60)) < 1e-9                          # all of b's fall is unusual
    assert abs(ex["a"] - 30) < 1e-9                              # a only looks "up" because b dragged the rest down
    assert abs(ex["b"]) > 1.5 * abs(ex["a"])                     # but b is still clearly the biggest outlier


def test_volume_value_split_is_exact():
    random.seed(1)
    for _ in range(200):
        o0, o1 = random.randint(1, 500), random.randint(1, 500)
        r0, r1 = random.uniform(10, 9000), random.uniform(10, 9000)
        v, a = volume_value_effects(o0, r0, o1, r1)
        assert abs((v + a) - (r1 - r0)) < 1e-7
    v, a = volume_value_effects(0, 0, 10, 500)
    assert v == 500 and a == 0


def test_mix_rate_sums_to_aov_change():
    random.seed(2)
    for _ in range(100):
        segs = {k: (random.randint(5, 300), random.uniform(100, 9000),
                    random.randint(5, 300), random.uniform(100, 9000)) for k in "abcd"}
        eff, dA = mix_rate_effects(segs)
        assert abs(sum(e["mix"] + e["rate"] for e in eff.values()) - dA) < 1e-7


def test_pure_mix_shift_has_zero_rate_effect():
    # each segment's AOV unchanged (100 and 20). Orders move toward the cheap segment.
    segs = {"big": (100, 10000, 100, 10000), "cheap": (100, 2000, 200, 4000)}
    eff, dA = mix_rate_effects(segs)
    assert dA < 0
    assert all(abs(e["rate"]) < 1e-9 for e in eff.values())
    assert eff["cheap"]["mix"] < 0                                # the cheap segment pulls AOV down


def test_js_and_selection():
    assert js_term(0.3, 0.3) == 0
    assert js_term(0.1, 0.5) > js_term(0.1, 0.2) > 0
    rows, T0, T1 = additive_breakdown({"a": 100, "b": 100, "c": 100}, {"a": 100, "b": 40, "c": 90})
    chosen, covered = select_explaining(rows, T1 - T0)
    assert chosen[0] == "b" and covered >= 0.67
