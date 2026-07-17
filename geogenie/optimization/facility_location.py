"""Phase 5: LP facility location (PuLP / scipy.linprog skeleton)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from geogenie.core.coords import Origin, to_xy

PointLL = Tuple[float, float]  # (lon, lat)


@dataclass(frozen=True)
class Placement:
    chosen_sites: Tuple[int, ...]
    total_cost: float
    assignments: Tuple[Tuple[int, int], ...]  # (demand_idx, site_idx)


def solve_placement(
    demand_pts: Sequence[PointLL],
    candidate_sites: Sequence[PointLL],
    budget: int,
    dist_fn: Optional[Callable[[PointLL, PointLL], float]] = None,
    origin: Optional[Origin] = None,
) -> Placement:
    """Choose up to `budget` sites minimizing sum of distances to nearest chosen site.

    Uses a greedy heuristic by default; if PuLP is installed, solves the ILP.
    Distances go through core.coords.to_xy (same funnel as the query path).
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not candidate_sites:
        raise ValueError("no candidate sites")
    if origin is None:
        origin = Origin(
            sum(p[0] for p in demand_pts) / max(len(demand_pts), 1)
            if demand_pts
            else candidate_sites[0][0],
            sum(p[1] for p in demand_pts) / max(len(demand_pts), 1)
            if demand_pts
            else candidate_sites[0][1],
        )

    def default_dist(a: PointLL, b: PointLL) -> float:
        ax, ay = to_xy(a[0], a[1], origin)
        bx, by = to_xy(b[0], b[1], origin)
        return math.hypot(ax - bx, ay - by)

    dfn = dist_fn or default_dist
    n_d, n_s = len(demand_pts), len(candidate_sites)
    D = [[dfn(demand_pts[i], candidate_sites[j]) for j in range(n_s)] for i in range(n_d)]

    try:
        return _solve_pulp(D, budget)
    except Exception:  # noqa: BLE001
        return _solve_greedy(D, budget)


def _solve_greedy(D: List[List[float]], budget: int) -> Placement:
    n_d, n_s = len(D), len(D[0]) if D else 0
    chosen: List[int] = []
    remaining = set(range(n_s))
    while len(chosen) < min(budget, n_s):
        best_j, best_cost = None, math.inf
        for j in remaining:
            trial = chosen + [j]
            cost = sum(min(D[i][k] for k in trial) for i in range(n_d)) if n_d else 0.0
            if cost < best_cost:
                best_cost, best_j = cost, j
        chosen.append(best_j)  # type: ignore[arg-type]
        remaining.remove(best_j)  # type: ignore[arg-type]
    assignments = []
    total = 0.0
    for i in range(n_d):
        j = min(chosen, key=lambda k: D[i][k])
        assignments.append((i, j))
        total += D[i][j]
    return Placement(chosen_sites=tuple(chosen), total_cost=total, assignments=tuple(assignments))


def _solve_pulp(D: List[List[float]], budget: int) -> Placement:
    import pulp

    n_d, n_s = len(D), len(D[0])
    prob = pulp.LpProblem("facility_location", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_s)]
    x = [
        [pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n_s)]
        for i in range(n_d)
    ]
    prob += pulp.lpSum(D[i][j] * x[i][j] for i in range(n_d) for j in range(n_s))
    prob += pulp.lpSum(y) <= budget
    for i in range(n_d):
        prob += pulp.lpSum(x[i][j] for j in range(n_s)) == 1
        for j in range(n_s):
            prob += x[i][j] <= y[j]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = tuple(j for j in range(n_s) if pulp.value(y[j]) > 0.5)
    assignments = []
    total = 0.0
    for i in range(n_d):
        for j in range(n_s):
            if pulp.value(x[i][j]) > 0.5:
                assignments.append((i, j))
                total += D[i][j]
    return Placement(chosen_sites=chosen, total_cost=total, assignments=tuple(assignments))
