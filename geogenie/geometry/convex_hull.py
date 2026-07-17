"""
Convex hull via Graham scan -- the "reachable area" polygon.

THE IDEA
--------
Given a cloud of points you can walk to in 10 minutes, find the smallest
convex polygon containing all of them. Graham scan does it in three steps:

  1. Pick a pivot guaranteed to be on the hull (lowest point; ties broken by
     leftmost). It is on the hull because nothing lies below it.
  2. Sort the rest by the angle they make with the pivot. Sweeping in angular
     order means you trace the boundary in one CCW pass.
  3. Walk the sorted list maintaining a stack. Before pushing a point, pop
     any point that would make a clockwise turn -- a right turn means the
     previous point is inside the hull, not on it.

COMPLEXITY
----------
  Sort:  O(n log n)   <- dominates
  Scan:  O(n)         <- each point is pushed once and popped at most once,
                         so the inner while loop is amortised O(1)
  Total: O(n log n) time, O(n) space.

That O(n) scan is worth being able to defend: the inner `while` looks like it
could make the scan quadratic, but every iteration of it permanently removes
a point from the stack, and only n points are ever pushed. So the total pop
count across the whole run is at most n.

USING IT FOR REACHABILITY (know what you are buying)
----------------------------------------------------
A convex hull is the WRONG SHAPE for a true isochrone, and it is important to
be honest about this rather than discover it in a demo.

Real reachable areas are non-convex. A river, a freeway, or a dead-end street
carves a notch out of where you can actually walk in 10 minutes. The convex
hull bridges every such notch, so it will claim you can reach a cafe on the
far bank of a river.

What the hull IS good for is a conservative OUTER BOUND. It never excludes a
genuinely reachable point (it contains all of them by construction), so it
produces false positives but never false negatives. That one-sidedness makes
it a legitimate cheap prefilter: hull-test first, then verify survivors with
an exact (and expensive) street-graph search.

If you want a tighter shape, the standard answers are alpha shapes / concave
hulls. That is a later step, not this one.
"""

from __future__ import annotations

from typing import List, Sequence

from .primitives import Point, cross, dedupe, dist2, orientation

__all__ = ["graham_scan", "convex_hull"]


def graham_scan(points: Sequence[Point]) -> List[Point]:
    """Convex hull of `points`, counter-clockwise, no collinear vertices.

    Parameters
    ----------
    points : (x, y) in a Euclidean plane (metres for GeoGenie -- NOT degrees).

    Returns
    -------
    The hull vertices in CCW order, starting from the pivot. The ring is
    "open": the first vertex is not repeated at the end.

    Degenerate inputs return sensibly rather than raising:
      0 points -> []
      1 point  -> [p]
      collinear points -> the two extreme points
    """
    pts = dedupe(points)
    n = len(pts)
    if n < 3:
        return sorted(pts)

    # -- 1. pivot: lowest y, then lowest x. Guaranteed on the hull. ---------
    pivot = min(pts, key=lambda p: (p[1], p[0]))
    rest = [p for p in pts if p != pivot]

    # -- 2. sort by polar angle around the pivot ---------------------------
    #
    # Sorting by atan2 directly would work but is slower and adds a
    # transcendental call per comparison. Instead sort by the orientation
    # predicate itself: `a` comes before `b` iff pivot->a->b turns left.
    # Ties (collinear with the pivot) are broken by distance, NEAREST FIRST.
    #
    # Python's sort needs a key, not a comparator, so we use functools'
    # cmp_to_key. This keeps the comparison exact (integer-sign based) rather
    # than relying on float angles that can compare inconsistently.
    import functools

    def _cmp(a: Point, b: Point) -> int:
        o = orientation(pivot, a, b)
        if o != 0:
            return -1 if o > 0 else 1
        # Collinear with pivot: nearer point first.
        return -1 if dist2(pivot, a) < dist2(pivot, b) else 1

    rest.sort(key=functools.cmp_to_key(_cmp))

    # -- 2b. THE COLLINEAR TRAP --------------------------------------------
    #
    # This is the step naive implementations omit, and it is the classic
    # Graham scan bug.
    #
    # Points collinear with the pivot along the FINAL ray must be visited
    # FARTHEST-first, not nearest-first. Reason: they all lie on the hull's
    # closing edge. Nearest-first would make the scan walk *inward* along
    # that edge, and the collinear-popping rule below would then discard the
    # farthest point -- which is an actual hull vertex.
    #
    # So: reverse the last collinear run.
    if len(rest) > 1:
        i = len(rest) - 1
        while i > 0 and orientation(pivot, rest[i - 1], rest[-1]) == 0:
            i -= 1
        rest[i:] = rest[i:][::-1]

    # -- 3. the scan -------------------------------------------------------
    stack: List[Point] = [pivot]
    for p in rest:
        # Pop while the last turn is not strictly left.
        #   < 0 : right turn  -> stack[-1] is inside the hull, drop it
        #   = 0 : collinear   -> stack[-1] lies ON an edge; drop it too, so
        #                        the result has vertices only, not edge points
        while len(stack) > 1 and orientation(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    # All input points collinear: the scan collapses to a degenerate 2-point
    # "hull". Return the two extremes explicitly -- the scan's own answer is
    # not reliable in this case.
    if len(stack) < 3:
        return [min(pts), max(pts)]

    return stack


def convex_hull(points: Sequence[Point]) -> List[Point]:
    """Alias for `graham_scan`, for callers who don't care which algorithm."""
    return graham_scan(points)
