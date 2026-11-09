"""Reference implementation of the process-level metrics (SER, ARR, SBF).

A trajectory is a list of T steps; each step is (state_before, action, state_after),
where states and actions are already abstracted (phi(s), psi(a)) to hashable keys.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Sequence

Step = tuple[Hashable, Hashable, Hashable]


@dataclass
class Metrics:
    T: int
    ser: float
    arr: float
    sbf: float
    redundant: list[int]
    backtracks: list[int]


def states(traj: Sequence[Step]) -> list[Hashable]:
    """s_0, s_1, ..., s_T"""
    return [traj[0][0]] + [st[2] for st in traj]


def redundancy_flags(traj: Sequence[Step]) -> list[int]:
    """r_t = 1 iff (phi(s_{t-1}), psi(a_t)) occurred at an earlier step t' < t."""
    seen, flags = set(), []
    for s_prev, a, _ in traj:
        key = (s_prev, a)
        flags.append(1 if key in seen else 0)
        seen.add(key)
    return flags


def backtrack_flags(traj: Sequence[Step]) -> list[int]:
    """b_t = 1 iff phi(s_t) in {phi(s_0..s_{t-2})} and phi(s_t) != phi(s_{t-1})."""
    s = states(traj)
    flags = []
    for t in range(1, len(s)):
        earlier = set(s[: t - 1])  # s_0 .. s_{t-2}
        flags.append(1 if (s[t] in earlier and s[t] != s[t - 1]) else 0)
    return flags


def compute(traj: Sequence[Step], L_star: int) -> Metrics:
    T = len(traj)
    if T == 0:
        raise ValueError("empty trajectory")
    r, b = redundancy_flags(traj), backtrack_flags(traj)
    return Metrics(
        T=T,
        ser=L_star / max(T, L_star),
        arr=sum(r) / T,
        sbf=sum(b) / T,
        redundant=[i + 1 for i, x in enumerate(r) if x],
        backtracks=[i + 1 for i, x in enumerate(b) if x],
    )


def waste_cost_share(traj: Sequence[Step], step_costs: Sequence[float]) -> float:
    """omega = sum_t c_t * (r_t OR b_t) / sum_t c_t"""
    r, b = redundancy_flags(traj), backtrack_flags(traj)
    total = sum(step_costs)
    return sum(c for c, x, y in zip(step_costs, r, b) if (x or y)) / total


def one_pass(traj: Sequence[Step], L_star: int, step_costs: Sequence[float] | None = None):
    """Algorithm 1 of the paper: SER, ARR, SBF and omega in a single pass, O(T) expected time and memory.

    P holds the (state, action) pairs seen so far; V holds the states s_0..s_{t-2}. Equivalent to compute() and
    waste_cost_share(), which rebuild the set of earlier states at every step (checked in test_procmetrics.py).
    """
    T = len(traj)
    if T == 0:
        raise ValueError("empty trajectory")
    c = list(step_costs) if step_costs is not None else [1.0] * T
    P, V = set(), set()
    n_r = n_b = 0
    wasted = 0.0
    for t, (s_prev, a, s_t) in enumerate(traj):
        r = (s_prev, a) in P
        P.add((s_prev, a))
        b = t > 0 and s_t in V and s_t != s_prev
        V.add(s_prev)  # after step t, V = {s_0..s_{t-1}}, i.e. s_0..s_{(t+1)-2} for the next step
        n_r += r
        n_b += b
        wasted += c[t] if (r or b) else 0.0
    return {"T": T, "ser": L_star / max(T, L_star), "arr": n_r / T, "sbf": n_b / T, "omega": wasted / sum(c)}


# ---------------------------------------------------------------------------------------------------------------
# Extensions used by the empirical study.
# These operate on separate state and action sequences: states = [phi(s_0), ..., phi(s_T)], actions = [psi(a_1), ...].


def arr_state(states, actions, exclude=frozenset()):
    """ARR (Eq. 7-8) with exact states; actions whose type (psi[0]) is in `exclude` are never flagged (still in T)."""
    seen, flags = set(), []
    for t, a in enumerate(actions):
        key = (states[t], a)
        flags.append(0 if (a[0] in exclude) else int(key in seen))
        seen.add(key)
    return flags


def arr_psi(actions, exclude=frozenset()):
    """State-free redundancy: r_t = 1 iff the identical abstract action psi(a_t) occurred earlier in the episode.
    Used where the state before each action is not logged (HAL Browser-Use, SeeAct)."""
    seen, flags = set(), []
    for a in actions:
        flags.append(0 if (a[0] in exclude) else int(a in seen))
        seen.add(a)
    return flags


def backtracks_in_sequence(states):
    """Number of t >= 1 with s_t in {s_0..s_{t-2}} and s_t != s_{t-1} (Eq. 9) over any recorded state sequence."""
    n = 0
    for t in range(1, len(states)):
        if states[t] is None:
            continue
        if states[t] != states[t - 1] and states[t] in set(states[: t - 1]):
            n += 1
    return n


def backtrack_flags_seq(states):
    """Per-transition backtrack indicators b_1..b_T for an exact state sequence s_0..s_T."""
    out = []
    for t in range(1, len(states)):
        out.append(int(states[t] != states[t - 1] and states[t] in set(states[: t - 1])))
    return out


def unique_action_ratio(actions):
    """Baseline: 1 - (#distinct psi) / T."""
    return 1 - len(set(actions)) / len(actions) if actions else 0.0


def longest_repeat_ratio(actions):
    """Baseline: length of the longest contiguous action subsequence that occurs at least twice
    (occurrences may overlap), divided by T."""
    T, best = len(actions), 0
    for L in range(1, T):
        seen, found = set(), False
        for i in range(T - L + 1):
            k = tuple(actions[i:i + L])
            if k in seen:
                found = True
                break
            seen.add(k)
        if not found:
            break
        best = L
    return best / T if T else 0.0


def ngram_loop(actions, n=3):
    """Baseline loop detector: share of positions that start an action n-gram already seen earlier."""
    if len(actions) < n:
        return 0.0
    seen, hits = set(), 0
    for i in range(len(actions) - n + 1):
        k = tuple(actions[i:i + n])
        hits += k in seen
        seen.add(k)
    return hits / len(actions)


def loop_erase(states):
    """Chronological loop erasure of a state path: whenever a state reappears, the cycle since its first visit is
    removed. Returns the erased path; its length - 1 is an upper bound on the steps needed (Prop. 2 argument)."""
    path = []
    for s in states:
        if s in path:
            path = path[: path.index(s) + 1]
        else:
            path.append(s)
    return path


if __name__ == "__main__":
    # Worked example from Section 4 (flight booking, L* = 4)
    H, F, Ho, FD, R, C = "Home", "Flights", "Hotels", "Flights+DAC", "Results", "Confirm"
    traj = [
        (H, "click:Flights", F),
        (F, "click:Hotels", Ho),
        (Ho, "go_back", F),
        (F, "type:from=DAC", FD),
        (FD, "click:Search", FD),
        (FD, "click:Search", FD),
        (FD, "click:Search", R),
        (R, "click:Book", C),
    ]
    m = compute(traj, L_star=4)
    print("Agent B:", m)
    opt = [(H, "click:Flights", F), (F, "type:from=DAC", FD), (FD, "click:Search", R), (R, "click:Book", C)]
    print("Agent A:", compute(opt, L_star=4))
    print("omega (uniform cost):", waste_cost_share(traj, [1.0] * len(traj)))
