"""Tests for procmetrics: worked example (paper Sec. 4.9) and Propositions 1-4.

Run:  python code/test_procmetrics.py      (or: pytest code/)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from procmetrics import backtrack_flags, compute, one_pass, redundancy_flags, waste_cost_share  # noqa: E402


def rand_traj(T, nS, nA, rng):
    s, out = rng.randrange(nS), []
    for _ in range(T):
        a, s2 = rng.randrange(nA), rng.randrange(nS)
        out.append((s, a, s2))
        s = s2
    return out


def test_worked_example():
    H, F, Ho, FD, R, C = "Home", "Flights", "Hotels", "Flights+DAC", "Results", "Confirm"
    b = [(H, "click:Flights", F), (F, "click:Hotels", Ho), (Ho, "go_back", F),
         (F, "type:from=DAC", FD), (FD, "click:Search", FD), (FD, "click:Search", FD),
         (FD, "click:Search", R), (R, "click:Book", C)]
    m = compute(b, L_star=4)
    assert (m.T, m.ser, m.arr, m.sbf) == (8, 0.5, 0.25, 0.125)
    assert m.redundant == [6, 7] and m.backtracks == [3]
    assert waste_cost_share(b, [1.0] * 8) == 0.375
    a = [(H, "click:Flights", F), (F, "type:from=DAC", FD), (FD, "click:Search", R), (R, "click:Book", C)]
    ma = compute(a, L_star=4)
    assert (ma.ser, ma.arr, ma.sbf) == (1.0, 0.0, 0.0)


def test_prop1_bounds_and_tightness():
    rng = random.Random(0)
    for _ in range(20000):
        T = rng.randint(1, 40)
        tr = rand_traj(T, rng.randint(1, 8), rng.randint(1, 4), rng)
        m = compute(tr, rng.randint(1, 30))
        assert 0 < m.ser <= 1
        assert 0 <= m.arr <= (T - 1) / T + 1e-12
        assert 0 <= m.sbf <= (T - 1) / T + 1e-12
        assert redundancy_flags(tr)[0] == 0 and backtrack_flags(tr)[0] == 0
    T = 6
    assert compute([(0, 0, 0)] * T, 1).arr == (T - 1) / T
    alt = [(0, 0, 1)] + [(i % 2, 0, (i + 1) % 2) for i in range(1, T)]
    assert compute(alt, 1).sbf == (T - 1) / T


def test_prop2_simple_paths_are_clean():
    rng = random.Random(1)
    for _ in range(5000):
        T = rng.randint(1, 15)
        path = rng.sample(range(100), T + 1)
        tr = [(path[i], rng.randrange(3), path[i + 1]) for i in range(T)]
        m = compute(tr, T)
        assert (m.ser, m.arr, m.sbf) == (1.0, 0.0, 0.0)


def test_prop3_detours_penalised():
    rng = random.Random(2)
    for _ in range(5000):
        T = rng.randint(2, 12)
        path = rng.sample(range(100), T + 1)
        tr = [(path[i], f"a{i}", path[i + 1]) for i in range(T)]
        k = rng.randint(1, T)
        j = rng.randint(0, k - 1)
        seq = [path[k]] + rng.sample(range(200, 300), rng.randint(0, 3)) + [path[j]]
        detour = [(seq[i], f"d{i}", seq[i + 1]) for i in range(len(seq) - 1)]
        tr2 = tr[:k] + detour + [(path[i], f"a{i}", path[i + 1]) for i in range(j, T)]
        assert sum(backtrack_flags(tr2)) >= sum(backtrack_flags(tr)) + 1
        assert compute(tr2, T).ser < compute(tr, T).ser


def test_prop4_relation_to_step_ratio():
    rng = random.Random(3)
    for _ in range(1000):
        T, L = rng.randint(1, 50), rng.randint(1, 50)
        assert abs(compute([(0, i, i + 1) for i in range(T)], L).ser - min(1, L / T)) < 1e-12


def test_one_pass_matches_reference():
    rng = random.Random(4)
    for _ in range(2000):
        T = rng.randint(1, 40)
        tr = rand_traj(T, rng.randint(1, 6), rng.randint(1, 4), rng)
        L = rng.randint(1, 20)
        costs = [rng.uniform(0.1, 5) for _ in range(T)]
        m, o = compute(tr, L), one_pass(tr, L, costs)
        assert o["T"] == m.T and abs(o["ser"] - m.ser) < 1e-12
        assert abs(o["arr"] - m.arr) < 1e-12 and abs(o["sbf"] - m.sbf) < 1e-12
        assert abs(o["omega"] - waste_cost_share(tr, costs)) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
