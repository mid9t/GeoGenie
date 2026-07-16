"""
GeoGenie Phase 2 -- classic computational geometry.

Four algorithms, plus the primitives they share:

    convex_hull      Graham scan            -> reachable-area polygon
    point_in_polygon ray casting / binary   -> is this POI inside it?
    simplify         Douglas-Peucker        -> thin a raw GPS track
    voronoi          half-plane intersection-> nearest-POI catchment zones

Everything operates on plain (x, y) tuples in a Euclidean plane, in METRES.
Nothing here imports Phase 1 and nothing here knows about the Earth: hand it
coordinates, it hands back geometry.

Raw (lon, lat) degrees are NOT a Euclidean plane -- a degree of longitude is
~111 km at the equator and ~85 km in San Francisco. Project to local metres
before calling anything in this package, and unproject for display.
"""

from .convex_hull import convex_hull, graham_scan
from .point_in_polygon import (
    is_convex_ring,
    point_in_convex_polygon,
    point_in_polygon,
    point_on_boundary,
)
from .primitives import (
    EPS,
    Point,
    bbox,
    cross,
    dedupe,
    dist,
    dist2,
    is_ccw,
    orientation,
    point_segment_distance,
    polygon_area,
    signed_area,
)
from .simplify import douglas_peucker, douglas_peucker_mask, radial_distance_filter
from .voronoi import VoronoiDiagram, bisector, clip_halfplane

__all__ = [
    "Point",
    "EPS",
    # primitives
    "cross",
    "orientation",
    "dist",
    "dist2",
    "bbox",
    "signed_area",
    "polygon_area",
    "is_ccw",
    "point_segment_distance",
    "dedupe",
    # convex hull
    "graham_scan",
    "convex_hull",
    # point in polygon
    "point_in_polygon",
    "point_in_convex_polygon",
    "point_on_boundary",
    "is_convex_ring",
    # simplification
    "douglas_peucker",
    "douglas_peucker_mask",
    "radial_distance_filter",
    # voronoi
    "VoronoiDiagram",
    "bisector",
    "clip_halfplane",
]
