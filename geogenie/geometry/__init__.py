"""
GeoGenie Phase 2 -- classic computational geometry.

Algorithms operate on plain (x, y) tuples in a Euclidean plane, in METRES.
Project lon/lat through geogenie.core.coords before calling anything here.
"""

from .clip import bisector_halfplane, clip_halfplane, clip_rect
from .concave import alpha_shape, reachable_area, suggest_alpha
from .delaunay import Delaunay
from .hull import convex_hull, graham_scan, hull_diameter, monotone_chain
from .point_in_polygon import (
    PreparedPolygon,
    is_convex_ring,
    point_in_convex_polygon,
    point_in_polygon,
    point_on_boundary,
    winding_number,
)
from .primitives import (
    Point,
    bbox,
    circumcenter,
    circumradius,
    cross,
    dedupe,
    dist,
    dist2,
    ensure_ccw,
    is_ccw,
    orientation,
    point_segment_distance,
    point_segment_distance2,
    polygon_area,
    polygon_centroid,
    signed_area,
)
from .projection import R_EARTH, LocalTangentPlane, haversine
from .simplify import (
    douglas_peucker,
    douglas_peucker_mask,
    radial_distance_filter,
    simplify,
    visvalingam_whyatt,
)
from .voronoi import VoronoiDiagram

__all__ = [
    "Point",
    "cross",
    "orientation",
    "dist",
    "dist2",
    "bbox",
    "signed_area",
    "polygon_area",
    "polygon_centroid",
    "is_ccw",
    "ensure_ccw",
    "point_segment_distance",
    "point_segment_distance2",
    "circumcenter",
    "circumradius",
    "dedupe",
    "graham_scan",
    "monotone_chain",
    "convex_hull",
    "hull_diameter",
    "point_in_polygon",
    "point_in_convex_polygon",
    "point_on_boundary",
    "is_convex_ring",
    "winding_number",
    "PreparedPolygon",
    "douglas_peucker",
    "douglas_peucker_mask",
    "radial_distance_filter",
    "simplify",
    "visvalingam_whyatt",
    "Delaunay",
    "VoronoiDiagram",
    "clip_halfplane",
    "clip_rect",
    "bisector_halfplane",
    "alpha_shape",
    "suggest_alpha",
    "reachable_area",
    "R_EARTH",
    "LocalTangentPlane",
    "haversine",
]
